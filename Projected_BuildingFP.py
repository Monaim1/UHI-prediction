import os
import datetime
import logging
import pyproj
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import transform
from shapely.affinity import scale
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from scipy.stats import skew, kurtosis
# Supress Warnings 
import warnings
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


warnings.filterwarnings('ignore')

# Conversion factors specific to NYC
LAT_TO_M = 111320  # Meters per degree latitude (approximate)
LON_TO_M = 85300   # Meters per degree longitude at NYC latitude (approximate)

def meters_to_degrees_lat(m):
    """Convert a distance in meters to degrees in latitude."""
    return m / LAT_TO_M

def meters_to_degrees_lon(m):
    """Convert a distance in meters to degrees in longitude."""
    return m / LON_TO_M

def deg_distance_to_m_lat(dist_deg):
    """Convert a degree distance (latitude) to meters."""
    return dist_deg * LAT_TO_M

def deg_distance_to_m_lon(dist_deg):
    """Convert a degree distance (longitude) to meters."""
    return dist_deg * LON_TO_M

def deg_area_to_m2(area_deg2):
    return area_deg2 * (LAT_TO_M * LON_TO_M)




def calculate_heat_index(T_celsius, Relative_Humidity):
    # Convert Celsius to Fahrenheit
    T = (T_celsius * 9/5) + 32
    R = Relative_Humidity
    
    # Constants
    c1 = -42.379
    c2 = 2.04901523
    c3 = 10.14333127
    c4 = -0.22475541
    c5 = -6.83783e-03
    c6 = -5.481717e-02
    c7 = 1.22874e-03
    c8 = 8.5282e-04
    c9 = -1.99e-06
    
    # Heat index formula
    HI = (c1 + c2*T + c3*R + c4*T*R + c5*T**2 + c6*R**2 +
          c7*T**2*R + c8*T*R**2 + c9*T**2*R**2)
    
    return HI

class UrbanMetricsProcessor:
    def __init__(self, buildings_gdf):
        """
        Initialize processor with building footprints and coordinate transformations.
        In this version, we work in the original EPSG:4326 using approximate conversions.
        
        Parameters:
        -----------
        buildings_gdf : GeoDataFrame
            Building footprints in EPSG:4326.
        """
        self.buildings = buildings_gdf.copy()
        self.buildings.crs = "EPSG:4326"
        
        # Set up CRS objects (if needed later)
        self.wgs84 = pyproj.CRS.from_epsg(4326)
        self.nyc_proj = pyproj.CRS.from_epsg(2263)

        # Transformation functions (available if a full reprojection is later desired)
        self.project = partial(
            pyproj.transform, 
            pyproj.Transformer.from_crs(self.wgs84, self.nyc_proj, always_xy=True).transform
        )
        self.unproject = partial(
            pyproj.transform, 
            pyproj.Transformer.from_crs(self.nyc_proj, self.wgs84, always_xy=True).transform
        )

    def compute_building_orientation(self, geom):
        """
        Compute a simple orientation (in degrees, 0-180) for a building footprint
        using the angle of its minimum rotated rectangle's longest edge.
        """
        try:
            mrr = geom.minimum_rotated_rectangle
            coords = list(mrr.exterior.coords)
            # Compute edge lengths for the first 4 edges
            edges = [np.hypot(coords[i+1][0] - coords[i][0], coords[i+1][1] - coords[i][1])
                     for i in range(4)]
            if not edges:
                return np.nan
            longest_edge_index = np.argmax(edges)
            p1, p2 = coords[longest_edge_index], coords[longest_edge_index+1]
            angle_rad = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
            return np.degrees(angle_rad) % 180
        except Exception as e:
            print(f"Orientation computation error: {e}")
            return np.nan

    def compute_annulus_metrics(self, point, annulus_bounds, 
                            wind_dir_deg, wind_speed, air_temp, 
                            rel_humidity, solar_flux, 
                            solar_elevation_deg=41.37):
        """
        For a given point (in EPSG:4326) and annulus bounds (inner_radius_m, outer_radius_m in meters),
        compute a suite of metrics based on intersecting building footprints (also in EPSG:4326).
        
        This version uses separate conversion factors for latitude and longitude and creates an
        elliptical buffer that approximates a circular buffer in meter space.
        
        Parameters:
          point : shapely.geometry.Point
              Input point in EPSG:4326.
          annulus_bounds : tuple
              (inner_radius_m, outer_radius_m) in meters.
          solar_elevation_deg : float
              Solar elevation angle in degrees.
        
        Returns:
          dict: Dictionary with computed metrics.
        """

        inner_m, outer_m = annulus_bounds

        # Create local meter-based coordinate system
        meter_crs = pyproj.CRS.from_proj4(
            f"+proj=aeqd +lat_0={point.y} +lon_0={point.x} +units=m +datum=WGS84"
        )

        # Convert buildings and point to meter projection
        buildings_meter = self.buildings.to_crs(meter_crs)
        center_point = gpd.GeoSeries([point], crs="EPSG:4326").to_crs(meter_crs)[0]

        # Create true circular buffers
        inner_buffer = center_point.buffer(inner_m)
        outer_buffer = center_point.buffer(outer_m)
        annulus_meter = outer_buffer.difference(inner_buffer)

        # Clip buildings to annulus in meter space
        annulus_gdf = gpd.GeoDataFrame(geometry=[annulus_meter], crs=meter_crs)
        annulus_buildings = gpd.overlay(buildings_meter, annulus_gdf, how="intersection")
        num_buildings = len(annulus_buildings)
        
        # Initialize metrics dictionary
        metrics = {
            'num_buildings': num_buildings,
            'min_distance_m': None,
            'max_distance_m': None,
            'mean_distance_m': None,
            'std_distance_m': None,
            'intersection_area_m2': 0.0,
            'min_height': None,
            'max_height': None,
            'mean_height': None,
            'median_height': None,
            'height_std': None,
            'weighted_area':0,
            'weighted_volume':0,
            'intersection_volume_m3': 0.0,
            'wind_obstruction_index': 0.0,
            'building_coverage': 0.0,
            'solar_obstruction_index': 0.0,
            'building_density': 0.0,
            'avg_building_fp_area_m2': 0.0,
            'median_building_fp_area_m2': 0.0,
            'effective_building_height': 0.0,
            'normalized_wind_obstruction_index': 0.0,
            'svf_proxy': 1.0,
            'height_skewness': None,
            'height_kurtosis': None,
            'height_IQR': None,
            'avg_compactness': None,
            'std_compactness': None,
            'building_volume_density': 0.0,
            'urban_roughness_proxy': 0.0,
            'avg_nearest_neighbor_m': None,
            'clark_evans_index': None,
            'avg_shadow_length_m': None,
            'avg_building_orientation_deg': None,
            'thermal_comfort_index': None,  
            'wind_chill_factor': None       
        }
        
        if num_buildings > 0:
             # Calculate distances in meter projection
            centroids = annulus_buildings.geometry.centroid
            coords = np.array([[c.x, c.y] for c in centroids])
            dists_m = np.linalg.norm(coords - [center_point.x, center_point.y], axis=1)
            dists_m = np.where(dists_m == 0, 1e-3, dists_m)
            
            # ===== IDW Weight Calculations =====
            # General weights (p=2 for inverse square law)
            weights = 1 / (dists_m ** 2)
            weights /= weights.sum()  # Normalize
            
            # Wind-specific weights (p=1 for linear decay)
            wind_weights = 1 / dists_m
            wind_weights /= wind_weights.sum()
            
            # Store basic distance metrics
            metrics.update({
                'min_distance_m': np.min(dists_m),
                'max_distance_m': np.max(dists_m),
                'mean_distance_m': np.mean(dists_m),
                'std_distance_m': np.std(dists_m)
            })
            
            # Calculate areas directly in meter projection
            annulus_area = annulus_meter.area
            intersection_area = annulus_buildings.geometry.area.sum()
            
            # IDW-weighted spatial metrics
            elevations = annulus_buildings["ELEVATION_max"].values  # assumed in meters
            # After:
            building_areas = annulus_buildings.geometry.area
            weighted_area = np.sum(building_areas * weights)
            weighted_volume = np.sum(building_areas * elevations * weights)
            
            metrics.update({
                'intersection_area_m2': intersection_area,
                'intersection_volume_m3' : np.sum(building_areas * elevations ),
                'weighted_area': weighted_area,
                'weighted_volume': weighted_volume,
                'building_coverage': float(intersection_area / annulus_area) if annulus_area > 0 else 0.0,
                'svf_proxy': 1.0 - (intersection_area / annulus_area) if annulus_area > 0 else 1.0,
                'min_height': np.min(elevations),
                'max_height': np.max(elevations),
                'mean_height': np.mean(elevations),
                'median_height': np.median(elevations),
                'height_std': np.std(elevations),
                'height_skewness': skew(elevations),
                'height_kurtosis': kurtosis(elevations),
                'height_IQR': np.subtract(*np.percentile(elevations, [75, 25]))
            })

            # ===== Wind/Solar Analysis =====
            wind_dir_rad = np.radians(270 - wind_dir_deg) % (2 * np.pi) # np.radians(wind_dir_deg)
            # Calculate bearing angles from center to buildings
            dx_m = coords[:, 0]  # East-West distances
            dy_m = coords[:, 1]  # North-South distances
            alpha = np.arctan2(dy_m, dx_m)  # CCW from East
            
            # Wind obstruction calculation
            wind_contrib = np.cos(wind_dir_rad - alpha) * elevations * wind_weights
            metrics['wind_obstruction_index'] = wind_speed * np.sum(wind_contrib[wind_contrib < 0])
            
            metrics['normalized_wind_obstruction_index'] = metrics['wind_obstruction_index'] / metrics["building_coverage"] if metrics["building_coverage"] > 0 else 0
            
            # Solar metrics
            metrics['solar_obstruction_index'] = (weighted_volume / annulus_area) * (solar_flux / 1000)
            shadow_lengths = elevations / np.tan(np.radians(solar_elevation_deg))
            metrics['avg_shadow_length_m'] = np.sum(shadow_lengths * weights)

            # ===== Thermal Metrics =====
            base_heat = calculate_heat_index(air_temp, rel_humidity)
            metrics['thermal_comfort_index'] = base_heat + (weighted_volume * 0.001)
            metrics['wind_chill_factor'] = 13.12 + 0.6215*air_temp - 11.37*(wind_speed**0.16) + 0.3965*air_temp*(wind_speed**0.16)

            
            # ===== Building Characteristics =====
            fp_perimeters_m = annulus_buildings.geometry.length  # Already in meters

            compactness = (4 * np.pi * building_areas) / (fp_perimeters_m ** 2)
            weighted_compactness = np.sum(compactness * weights)

            metrics.update({
                'building_density': num_buildings / annulus_area if annulus_area > 0 else np.nan,
                'avg_building_fp_area_m2': np.sum(building_areas * weights),
                'median_building_fp_area_m2': np.median(building_areas),
                'effective_building_height': (weighted_volume / weighted_area if weighted_area > 1e-6 else np.nan),
                'avg_compactness': weighted_compactness,
                'std_compactness': np.std(compactness),
                'building_volume_density': (weighted_volume / annulus_area if annulus_area > 0 else np.nan),
                'urban_roughness_proxy': (
                    metrics['effective_building_height'] * 
                    (np.clip(metrics['building_coverage'], 0.001, 1) ** (1/3)) 
                    if not np.isnan(metrics['effective_building_height']) 
                    else np.nan
                )
            })
            # P) Building Orientation
            orientations_rad = np.radians(annulus_buildings.geometry.apply(self.compute_building_orientation))
            mean_angle = np.arctan2(
                np.sum(np.sin(orientations_rad) * weights),
                np.sum(np.cos(orientations_rad) * weights)
            )
            metrics['avg_building_orientation_deg'] = np.degrees(mean_angle) % 180



              # ===== Spatial Pattern Analysis =====
            if num_buildings >= 2 :
                # Calculate pairwise distances between buildings
                centroid_coords = np.column_stack([dx_m, dy_m])
                centroid_dists = squareform(pdist(centroid_coords))
                np.fill_diagonal(centroid_dists, np.inf)
                nn_dists = np.min(centroid_dists, axis=1)
                
                metrics['avg_nearest_neighbor_m'] = np.mean(nn_dists)
                
                # Clark-Evans Index Calculation
                if metrics['building_density'] > 0:
                    expected_nn = 0.5 / np.sqrt(metrics['building_density'])
                    metrics['clark_evans_index'] = (
                        metrics['avg_nearest_neighbor_m'] / expected_nn
                    )
                else:
                    metrics['clark_evans_index'] = np.nan
            else:
                metrics['avg_nearest_neighbor_m'] = np.nan
                metrics['clark_evans_index'] = np.nan

        return metrics

# Module-level function for processing a single point
def process_point(args):    
    row, metrics_processor, annuli = args
    sh_point = row.geometry

    weather_params = {
        'wind_dir_deg': row['Wind Direction [degrees]'],
        'wind_speed': row['Avg Wind Speed [m/s]'],
        'air_temp': row['Air Temp at Surface [degC]'],
        'rel_humidity': row['Relative Humidity [percent]'],
        'solar_flux': row['Solar Flux [W/m^2]']
    }
 
    
    metrics_all = {
        "Latitude": sh_point.y,
        "Longitude": sh_point.x,
        **weather_params
    }
    
    for inner, outer in annuli:
        suffix = f"_{inner}_{outer}"
 
        try:
            metrics = metrics_processor.compute_annulus_metrics(
                    sh_point, 
                    (inner, outer),
                    **weather_params  # Pass precomputed weather params
                )
            metrics_all.update({f"{key}{suffix}": value for key, value in metrics.items()})
        except Exception as e:
            print(f"Error processing point {sh_point} for annulus {inner}-{outer}: {e}")
    return metrics_all


# Your main mapping function now uses the global process_point function
def map_buildingFP_optimized(train_FEdata, gpd_BuildingFP, parallel=True):

    # Initialize the metrics processor
    metrics_processor = UrbanMetricsProcessor(gpd_BuildingFP)
    
    # Define annuli in meters
    annuli = [
        (0, 20), (0, 50), (0, 100), (0, 200),
        (0, 400), (0, 700), (0, 1000) ]

    # Create a list of arguments for each point
    args_list = [(row, metrics_processor, annuli) for _, row in train_FEdata.iterrows()]
    
    if parallel and cpu_count() > 1:
        with Pool(processes=max(cpu_count() - 1, 1)) as pool:
            results = list(tqdm(pool.imap(process_point, args_list),
                                total=len(args_list), desc="Mapping building metrics"))
    else:
        results = [process_point(args) for args in tqdm(args_list, desc="Mapping building metrics")]
    
    results_df = pd.DataFrame(results)
    merged_df = train_FEdata.merge(results_df, on=['Latitude', 'Longitude'], how='left')
    return merged_df

def visualize_annulus(buildings_gdf, point, inner_radius_m, outer_radius_m):
    """
    Visualizes buildings within an annulus using accurate metric calculations.
    
    Parameters:
        buildings_gdf (GeoDataFrame): Building footprints in EPSG:4326
        point (Point): Center point in EPSG:4326
        inner_radius_m (float): Inner radius in meters
        outer_radius_m (float): Outer radius in meters
    """
    # Create local meter-based coordinate system centered on point
    meter_crs = pyproj.CRS.from_proj4(
        f"+proj=aeqd +lat_0={point.y} +lon_0={point.x} +units=m +datum=WGS84"
    )
    
    # Convert geometries to meter-based projection
    buildings_meter = buildings_gdf.to_crs(meter_crs)
    center_point = gpd.GeoSeries([point], crs="EPSG:4326").to_crs(meter_crs)[0]
    
    # Create true circular buffers in meter space
    inner_buffer = center_point.buffer(inner_radius_m)
    outer_buffer = center_point.buffer(outer_radius_m)
    annulus_meter = outer_buffer.difference(inner_buffer)
    
    # Find intersecting buildings
    annulus_gdf = gpd.GeoDataFrame(geometry=[annulus_meter], crs=meter_crs)
    intersecting_buildings = gpd.overlay(buildings_meter, annulus_gdf, how="intersection")
    
    # Calculate areas directly in meter-based CRS
    annulus_area = annulus_meter.area
    intersection_area = intersecting_buildings.geometry.area.sum()
    
    print(f"Annulus {inner_radius_m}-{outer_radius_m}m:\n"
          f" - Total buildings: {len(intersecting_buildings)}\n"
          f" - Intersection area: {intersection_area:.2f} m²\n"
          f" - Annulus area: {annulus_area:.2f} m²\n"
          f" - Coverage ratio: {intersection_area/annulus_area:.2%}")

    # Create plot in meter coordinates
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot annulus boundary
    gpd.GeoSeries([annulus_meter]).boundary.plot(
        ax=ax, color='red', linewidth=2, label="Annulus boundary"
    )
    
    # Plot intersecting buildings
    intersecting_buildings.plot(
        ax=ax, color='dimgray', edgecolor='black', 
        alpha=0.7, label="Intersecting buildings"
    )
    
    # Plot center point
    gpd.GeoSeries([center_point]).plot(
        ax=ax, color='blue', markersize=100, 
        marker='x', label="Center point"
    )
    
    # Formatting
    ax.set_title(f"Buildings in {inner_radius_m}-{outer_radius_m}m Annulus\n"
                 f"(Total Coverage: {intersection_area/annulus_area:.2%})")
    ax.set_xlabel("Meters (Local Projection)")
    ax.set_ylabel("Meters (Local Projection)")
    plt.legend()
    plt.show()




def main():
    try:
        print('=====Program started at ' + str(datetime.datetime.now()) + "=====\n")
        logging.info("Starting UHI data processing")

        
        # Input file paths
        train_data_path = 'Train_Data/Final/Training_data_PD_Weather_2025-02-18.csv'
        Submission_data_path = 'Submission_Mapped_Data/Final/Submission_PD_Weather_2025-02-18.csv'

        building_fp_path = "Train_Data/Building_Footprint_Elevation_Aggregated_12032025.geojson"
        train_output_path = "Train_Data/Final/Train_Buildings_v18032025_NegativeMask.csv"
        Submission_output_path = "Submission_Mapped_Data/Final/Submission_Buildings_v18032025_NegativeMask.csv"
                
        train_FEdata = pd.read_csv(train_data_path)
        Submission_FEdata = pd.read_csv(Submission_data_path)
        gpd_BuildingFP = gpd.read_file(building_fp_path)
        
        
        
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
        logging.info("Submission Data: Starting building footprint mapping")
        FP_Submissiondata = map_buildingFP_optimized(Submission_FEdata, gpd_BuildingFP)
        FP_Submissiondata.to_csv(Submission_output_path, index=False)

        
        logging.info("Train Data : Starting building footprint mapping")
        FP_Traindata = map_buildingFP_optimized(train_FEdata, gpd_BuildingFP)
        FP_Traindata.to_csv(train_output_path, index=False)

        logging.info("Processing completed successfully")
        print(f'=====Program completed at {datetime.datetime.now()}=====\n')
    
    except Exception as e:
        logging.error(f"An error occurred during processing: {e}", exc_info=True)

if __name__ == '__main__':
    main()

    
