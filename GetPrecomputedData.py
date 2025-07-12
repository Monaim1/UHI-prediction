import warnings
warnings.filterwarnings('ignore')
import numpy as np
import xarray as xr
import geopandas as gpd
import pandas as pd
import rasterio
from tqdm import tqdm
from shapely.geometry import Point
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import block_reduce
import concurrent.futures
import datetime
import os




# Conversion factors
LAT_TO_M = 111320  # Meters per degree latitude
LON_TO_M = 85300   # Meters per degree longitude



# Helper functions for GLCM-based texture metrics
def compute_glcm_metrics(image, distances=[1], angles=[0], levels=8):
    """
    Why these metrics?
    - Contrast gives an idea of the intensity variation.
    - Correlation measures how correlated a pixel is to its neighbor.
    - Energy reflects textural uniformity.
    - Entropy measures randomness.
    Such texture information can be important in UHI modeling because urban surfaces (concrete, asphalt, vegetation) 
    have different textural signatures that influence heat retention and reflection.
        """
    # Quantize image to [0, levels-1]
    image_min = np.nanmin(image)
    image_max = np.nanmax(image)
    if image_max == image_min:
        quantized = np.zeros_like(image, dtype=np.uint8)
    else:
        quantized = np.floor((image - image_min) / (image_max - image_min) * (levels - 1)).astype(np.uint8)
    glcm = graycomatrix(quantized, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    # Compute entropy manually
    glcm_nonzero = glcm[glcm > 0]
    entropy = -np.sum(glcm_nonzero * np.log(glcm_nonzero))
    return {
        'glcm_contrast': contrast,
        'glcm_correlation': correlation,
        'glcm_energy': energy,
        'glcm_entropy': entropy
    }



# Conversion functions
def meters_to_degrees_lat(m): return m / LAT_TO_M
def meters_to_degrees_lon(m): return m / LON_TO_M
def deg_area_to_m2(area_deg2): return area_deg2 * (LAT_TO_M * LON_TO_M)

class AnnulusProcessor:
    """
    This class is used to compute statistics for an annular region around a point. It includes methods to:
Create an annulus (the difference between two buffers) around a point.
Compute distance weights based on the inverse square of the distance from the point (with normalization).
Calculate weighted statistics (mean, std, median, etc.) for data within the annulus.
They capture information from different spatial scales. For example, surface properties within 30 m might capture immediate surroundings (like pavement type), 
while a larger annulus (e.g., 300, 600 m) might capture broader urban patterns (like building density and water bodies).
"""

    def __init__(self, downsample_factor=10):
        self.downsample_factor = downsample_factor
        self.buffer_cache = {}

    def _get_annulus_mask(self, point, inner, outer):
        # Create a unique key for caching
        key = (float(point.x), float(point.y), inner, outer)
        if key not in self.buffer_cache:
            # Convert meters to degrees
            inner_deg = meters_to_degrees_lat(inner)
            outer_deg = meters_to_degrees_lat(outer)
            # Create buffers
            buffer_outer = point.buffer(outer_deg)
            buffer_inner = point.buffer(inner_deg)
            # Store annulus geometry in cache
            self.buffer_cache[key] = buffer_outer.difference(buffer_inner)
        
        return self.buffer_cache[key]

    def _compute_distance_weights(self, clipped_data, point_coords):
        transform = clipped_data.rio.transform()
        height, width = clipped_data.shape
        
        # Generate grid coordinates for the clipped region
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)
        
        # Convert pixel coordinates to geographic coordinates
        geo_x = transform[2] + (xx + 0.5) * transform[0]
        geo_y = transform[5] + (yy + 0.5) * transform[4]
        
        # Calculate distances in meters
        dx = (geo_x - point_coords[0]) * LON_TO_M
        dy = (geo_y - point_coords[1]) * LAT_TO_M
        distances = np.sqrt(dx**2 + dy**2) + 1e-6  # Avoid division by zero
        
        # Compute inverse distance squared weights
        weights = 1 / (distances ** 2)
        weights /= np.nansum(weights)  # Normalize
        
        return weights

    def _calculate_weighted_stats(self, data, weights):
        """Calculate statistics with shape validation"""
        # Ensure matching dimensions
        assert data.shape == weights.shape, \
            f"Data shape {data.shape} doesn't match weights shape {weights.shape}"
        
        valid_mask = ~np.isnan(data)
        valid_data = data[valid_mask]
        valid_weights = weights[valid_mask]
        
        if valid_data.size == 0:
            return {k: np.nan for k in ['mean', 'std', 'median', 'min', 'max']}
        
        # Normalize valid weights subset
        valid_weights /= valid_weights.sum()
        
        return {
            'mean': np.sum(valid_data * valid_weights),
            'std': np.sqrt(np.sum(valid_weights * (valid_data - np.sum(valid_data * valid_weights))**2)),
            'median': np.median(valid_data),
            'min': np.min(valid_data),
            'max': np.max(valid_data)
        }

# --- Combined Statistics Extraction Function ---
def extract_combined_stats_for_points(dataset, gdf_points, window=5, percentiles=[20, 80],
                                      vars_to_process=None, annuli=None, glcm_vars=None, processor=None):
    """
    For each point in gdf_points, compute:
      1. Fixed-window local statistics from the xarray.Dataset.
      2. For each provided annulus (a tuple (inner, outer) in meters),
         compute distance-weighted statistics and GLCM texture metrics.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Must have "longitude" and "latitude" coordinates.
        For annulus-based clipping, the dataset should have been opened with rioxarray.
    gdf_points : GeoDataFrame
        GeoDataFrame with Point geometries (EPSG:4326) and "Latitude"/"Longitude" columns.
    window : int, optional
        Size of the fixed window (default 5 for 5x5 pixels).
    percentiles : list of int, optional
        Percentiles to compute for the window data.
    vars_to_process : list of str, optional
        Variables to process. If None, all variables in dataset are processed.
    annuli : list of tuple, optional
        List of (inner, outer) distances in meters for annulus-based stats.
    glcm_vars : list of str, optional
        Subset of variables for which to compute GLCM texture metrics.
    processor : AnnulusProcessor instance, optional
        An instance to process annulus buffers and compute weights.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with one row per point and columns for each computed statistic.
    """
    results = []
    # Get coordinate arrays (assumed 1D)
    lon_arr = dataset.coords["longitude"].values
    lat_arr = dataset.coords["latitude"].values

    if vars_to_process is None:
        vars_to_process = list(dataset.data_vars.keys())
    
    half = window // 2

    for idx, row in tqdm(gdf_points.iterrows(), total=len(gdf_points), desc="Processing points"):
        point = row.geometry
        x_pt = point.x
        y_pt = point.y

        # Find the nearest pixel indices
        ix = int(np.abs(lon_arr - x_pt).argmin())
        iy = int(np.abs(lat_arr - y_pt).argmin())

        # Determine window slice indices (ensuring within bounds)
        x_start = max(ix - half, 0)
        x_end   = min(ix + half + 1, len(lon_arr))
        y_start = max(iy - half, 0)
        y_end   = min(iy + half + 1, len(lat_arr))

        # Dictionary to store statistics for this point
        point_stats = {
            "Latitude": y_pt,
            "Longitude": x_pt
        }

        # -- Fixed Window Statistics --
        for var in vars_to_process:
            # Extract the fixed window
            window_data = dataset[var].isel(longitude=slice(x_start, x_end),
                                            latitude=slice(y_start, y_end))
            wdata = window_data.values.astype(float)
            local_mean = np.nanmean(wdata)
            local_std  = np.nanstd(wdata)
            local_std_filled = local_std if local_std > 0 else 1e-10
            local_min  = np.nanmin(wdata)
            local_max  = np.nanmax(wdata)
            local_sum  = np.nansum(wdata)
            # Assume central pixel is at [half, half] (works best if window is odd)
            central_val = wdata[half, half] if wdata.shape[0] > half and wdata.shape[1] > half else np.nan

            local_zscore = (central_val - local_mean) / local_std_filled
            local_range = local_max - local_min
            local_range_filled = local_range if local_range > 0 else 1e-10
            local_norm = (central_val - local_min) / local_range_filled

            # Compute percentiles
            perc_dict = {}
            for p in percentiles:
                perc = np.nanpercentile(wdata, p)
                perc_dict[p] = perc

            # Hotspot flag based on 90th percentile
            hotspot_flag = 1 if central_val > perc_dict.get(90, np.nan) else 0

            # Save computed fixed-window statistics with variable-specific keys
            point_stats[f"{var}_local_mean"] = local_mean
            point_stats[f"{var}_local_std"] = local_std
            point_stats[f"{var}_local_min"] = local_min
            point_stats[f"{var}_local_max"] = local_max
            point_stats[f"{var}_local_sum"] = local_sum
            point_stats[f"{var}_local_zscore"] = local_zscore
            point_stats[f"{var}_local_norm_anomaly"] = local_norm
            for p, perc in perc_dict.items():
                point_stats[f"{var}_local_p{p}"] = perc
            point_stats[f"{var}_local_hotspot_flag"] = hotspot_flag

        # -- Annulus-based Metrics --
        if annuli and processor is not None:
            # Ensure the dataset has rioxarray methods (i.e. dataset.rio.clip is available)
            for inner, outer in annuli:
                suffix = f"_{inner}_{outer}"
                annulus = processor._get_annulus_mask(point, inner, outer)
                try:
                    # Clip the dataset to the annulus area (all_touched=True to include boundary pixels)
                    clipped = dataset.rio.clip([annulus], all_touched=True)
                except Exception as e:
                    # If clipping fails, skip annulus for this point
                    continue

                # Get the affine transform from one of the clipped variables (assumes all share the same transform)
                sample_var = list(clipped.data_vars)[0]
                transform = clipped[sample_var].rio.transform()

                # Compute distance weights using the first available variable in the clipped data
                sample_array = clipped[sample_var].values
                weights = processor._compute_distance_weights(sample_array, (point.x, point.y), transform)

                for var in clipped.data_vars:
                    arr = clipped[var].values
                    valid = arr[~np.isnan(arr)]
                    if valid.size == 0:
                        point_stats.update({
                            f"{var}_mean{suffix}": np.nan,
                            f"{var}_std{suffix}": np.nan,
                            f"{var}_median{suffix}": np.nan,
                            f"{var}_min{suffix}": np.nan,
                            f"{var}_max{suffix}": np.nan,
                            f"{var}_range{suffix}": np.nan
                        })
                        continue

                    # Calculate weighted statistics for the annulus region
                    w_stats = processor._calculate_weighted_stats(arr, weights)
                    point_stats.update({
                        f"{var}_mean{suffix}": w_stats['mean'],
                        f"{var}_std{suffix}": w_stats['std'],
                        f"{var}_median{suffix}": w_stats['median'],
                        f"{var}_min{suffix}": w_stats['min'],
                        f"{var}_max{suffix}": w_stats['max'],
                        f"{var}_range{suffix}": w_stats['max'] - w_stats['min']
                    })

                    # If requested and if enough valid pixels exist, compute GLCM texture metrics.
                    if glcm_vars and var in glcm_vars and valid.size > 100:
                        try:
                            texture = compute_glcm_metrics(arr)
                            point_stats.update({
                                f"{var}_contrast{suffix}": texture['glcm_contrast'],
                                f"{var}_correlation{suffix}": texture['glcm_correlation'],
                                f"{var}_energy{suffix}": texture['glcm_energy'],
                                f"{var}_entropy{suffix}": texture['glcm_entropy'],
                            })
                        except Exception:
                            pass

        results.append(point_stats)

    result_df = pd.DataFrame(results)
    return result_df


def extract_buffer_data_optimized(data, gdf_points, annuli, downsample_factor=10):
    processor = AnnulusProcessor(downsample_factor)
    glcm_vars = ['NDVI', 'NDBI', 'NDWI', 'LST_Proxy', 'Albedo']
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        args = [(i, row, data, annuli, processor, glcm_vars) 
                for i, row in gdf_points.iterrows()]
        results = list(tqdm(executor.map(extract_combined_stats_for_points, args), total=len(args)))
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    print('=====Program started at ' + str(datetime.datetime.now()) + "=====\n")
    print("====  Reading NetCDF files ======\n")

    sentinel2_data = xr.open_dataset("Satellite_Data/Sentinel2_indices_26022025.nc")
    sentinel2_data = sentinel2_data.persist()
    landsat_data = xr.open_dataset("Satellite_Data/Sentinel2_indices_26022025.nc")
    landsat_data = sentinel2_data.persist()

    # ground_df = pd.read_csv("Submission_Mapped_Data/Submission_template_UHI2025-v2.csv")
    train_FEdata = pd.read_csv("Train_Data/Training_data_uhi_index_2025-02-18.csv")
    Submission_FEdata = pd.read_csv('Submission_Mapped_Data/Submission_template_UHI2025-v2.csv')
    
    train_output_path = "Train_Data/Final/Sentinel2_Train_v12032025.csv"
    Submission_output_path = "Submission_Mapped_Data/Final/Sentinel2_Submission_v12032025.csv"

    annuli = [
        (0, 30), (30, 80), (80, 150),(150, 300), (300, 600)
    ]
    
    train_FEdata = gpd.GeoDataFrame(
            train_FEdata,
            geometry=[Point(lon, lat) for lon, lat in zip(train_FEdata['Longitude'], train_FEdata['Latitude'])],
            crs="EPSG:4326"
        )
    Submission_FEdata = gpd.GeoDataFrame(
            Submission_FEdata,
            geometry=[Point(lon, lat) for lon, lat in zip(Submission_FEdata['Longitude'], Submission_FEdata['Latitude'])],
            crs="EPSG:4326"
        ) 

    S2_Submissiondata = extract_buffer_data_optimized(sentinel2_data, Submission_FEdata,annuli, downsample_factor=10)
    S2_Submissiondata.to_csv(Submission_output_path, index=False)



          
    

        
    
