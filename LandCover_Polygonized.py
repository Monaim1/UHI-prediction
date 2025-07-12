import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon
from tqdm.auto import tqdm
from scipy.spatial import cKDTree
import concurrent.futures
from tqdm.auto import tqdm
from geopandas.sindex import SpatialIndex
from multiprocessing import Pool, cpu_count
# Supress Warnings
import warnings
warnings.filterwarnings('ignore')


# Global variables to be initialized in each worker process
global_landcover_gdf = None
global_landcover_sindex = None

def init_worker(landcover_gdf, landcover_sindex):
    """
    Initialize each worker with the global GeoDataFrame and spatial index.
    """
    global global_landcover_gdf, global_landcover_sindex
    global_landcover_gdf = landcover_gdf
    global_landcover_sindex = landcover_sindex


def compute_point_metrics(annulus_geom, landcover_gdf, landcover_sindex, point):
    """
    Compute landcover metrics within a given annulus geometry.
    This version reduces redundant operations and leverages vectorized calculations.
    """
    metrics = {
        'building': 0,
        'grass_shrub': 0,
        'bare_ground':0,
        'open_water':0,
        'railroad':0,
        'other_impervious': 0,
        'road': 0,
        'tree_canopy': 0,
        'building_coverage': 0,
        'grass_shrub_coverage': 0,
        'bare_ground_coverage':0,
        'open_water_coverage':0,
        'railroad_coverage':0,
        'other_impervious_coverage': 0,
        'road_coverage': 0,
        'tree_canopy_coverage': 0,
    }
    
    possible_matches = list(landcover_sindex.intersection(annulus_geom.bounds))
    
    filtered = landcover_gdf.iloc[possible_matches]
    clipped = gpd.clip(filtered, annulus_geom).copy() 

    if clipped.empty:
        return metrics

    clipped["clipped_area"] = clipped.geometry.area
    total_area = clipped["clipped_area"].sum() + 1e-6
    class_areas = clipped.groupby("LandCover_Class")["clipped_area"].sum().to_dict()
    metrics.update(class_areas)
    class_coverage = {k: (v / total_area)*100 for k,v in class_areas.items() }
    metrics.update({f"{cls}_coverage": val for cls, val in class_coverage.items()})
    # --- Water ---
    original_water = filtered[filtered["LandCover_Class"] == "open_water"]
    # --- Vegetation ---
    original_veg = filtered[filtered["LandCover_Class"].isin(["tree_canopy", "grass_shrub"])]
    
    # --- Impervious clipped ---
    impervious_mask_clipped = clipped["LandCover_Class"].isin(["building","bare_ground", "road", "other_impervious", "railroad"])
    clipped_impervious = clipped[impervious_mask_clipped]

    # --- Water Metrics ---
    if not original_water.empty:
        all_water_edges = original_water["boundary"].unary_union
        metrics["closest_water_distance"] = point.distance(all_water_edges)
        # Compute water boundaries once using unary_union
        # Get precomputed centroids and full areas  for IDW-weighted water area calculation
        water_areas = original_water["full_area"].values  # Full area of water bodies

                # Instead of using KDTree with centroids, compute edge distances directly:
        water_edge_distances = np.array([point.distance(boundary) for boundary in original_water["boundary"]])
        water_weights = 1 / (np.square(water_edge_distances) + 1e-6)
        total_water_weight = np.sum(water_weights)
        
        metrics["IDW_full_water_area"] = np.sum(water_areas * water_weights) / total_water_weight if total_water_weight > 0 else 0
        # # Cooling potential: Use clipped water area within annulus
        # clipped_water_area = clipped_water["clipped_area"].sum()
        metrics["water_cooling_potential"] = (water_areas.sum() / total_area) * np.log1p(metrics["IDW_full_water_area"] + 1e-6)

    else:
        metrics.update({
            "closest_water_distance": np.nan,
            "IDW_full_water_area":np.nan,
            "water_cooling_potential":0
        })


    if not original_veg.empty:
        all_veg_edges = original_veg["boundary"].unary_union
        metrics["closest_vegetation_distance"] = point.distance(all_veg_edges)

        veg_areas = original_veg["full_area"].values

        # Corrected: use vegetation boundaries for edge distances
        veg_edge_distances = np.array([point.distance(boundary) for boundary in original_veg["boundary"]])
        veg_weights = 1 / (np.square(veg_edge_distances) + 1e-6)
        veg_total_weight = np.sum(veg_weights)

        veg_patches = original_veg.geometry.unary_union
        metrics["vegetation_edge_ratio"] = veg_patches.length / veg_patches.area
        metrics["vegetation_fragmentation"] = (original_veg.shape[0] / (veg_patches.area + 1e-6))  # Patches per unit area
        metrics["IDW_vegetation_area"] = np.sum(veg_areas * veg_weights) / veg_total_weight if veg_total_weight > 0 else 0

        # Compute vertical complexity (avoid division-by-zero)
        complexities = [poly.area / poly.length for poly in original_veg.geometry if poly.length > 0]
        metrics["vegetation_vertical_complexity"] = np.mean(complexities) if complexities else 0
    else:
        metrics.update({
            "closest_vegetation_distance": np.nan,
            "vegetation_edge_ratio": np.nan,
            "IDW_vegetation_area": 0,
            "vegetation_vertical_complexity": 0,
            "vegetation_fragmentation": 0
        })


    if not clipped_impervious.empty:
        areas = clipped_impervious["full_area"].values

        impervious_edge_distances = np.array([point.distance(poly.boundary) for poly in clipped_impervious.geometry])
        impervious_weights = 1.0 / (np.square(impervious_edge_distances) + 1e-6)
    
        # Distance to nearest impervious edge (full polygons)
        metrics["closest_impervious_distance"] = np.min(impervious_edge_distances)
        metrics["IDW_full_impervious_area"] = np.sum(areas * impervious_weights) / (np.sum(impervious_weights) + 1e-6)
        metrics["largest_impervious_patch"] = clipped_impervious["full_area"].max()

        impervious_agg = clipped_impervious.unary_union
        albedo_map = {"road": 0.1, "building": 0.2, "other_impervious": 0.15, "railroad":0.15}
        clipped_impervious["albedo"] = clipped_impervious["LandCover_Class"].map(albedo_map)
        metrics["effective_albedo"] = ((clipped_impervious["clipped_area"] * clipped_impervious["albedo"]).sum() / total_area)
        metrics["impervious_contiguity"] = impervious_agg.area / total_area
        metrics["impervious_edge_density"] = impervious_agg.length / total_area
    else:
        metrics.update({
                "IDW_full_impervious_area": 0,
                "closest_impervious_distance": np.nan,
                "largest_impervious_patch": 0,
                "impervious_contiguity": 0,
                "impervious_edge_density": 0,
                "effective_albedo": 0})

    # =================================================================
    # 5. Composite UHI Index (Improved Normalization)
    # =================================================================
    try:
        # Normalize metrics to [0,1] range
        norm = {
            "impervious_contiguity": min(metrics.get("impervious_contiguity", 0), 1),
            "vegetation_fragmentation": 1 - np.exp(-metrics.get("vegetation_fragmentation", 0)),
            "effective_albedo": metrics.get("effective_albedo", 0.15) / 0.3  # Assume max 0.3
        }
        
        metrics["UHI_risk_index"] = (
            0.4 * norm["impervious_contiguity"] +
            0.3 * (1 - norm["effective_albedo"]) -
            0.5 * norm["vegetation_fragmentation"] -
            0.2 * (1 / (metrics.get("closest_water_distance", 100) + 1))
        )
    except Exception as e:
        print(f"UHI index error: {e}")
        metrics["UHI_risk_index"] = np.nan

    return metrics


def calculate_landcover_annuli_metrics_vector(gdf, landcover_gdf, landcover_sindex, annuli):
    """Calculate annular metrics with enhanced UHI predictors."""
    
    results = []
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Processing points"):
        point_geom = row.geometry
        row_metrics = {}
        
        # Calculate full-buffer metrics (0 to max radius)
        max_radius = annuli[-1][1]
        full_buffer = point_geom.buffer(max_radius * 3.28084)
        full_metrics = compute_point_metrics(full_buffer, landcover_gdf,landcover_sindex, point_geom)
        for k, v in full_metrics.items():
            row_metrics[f"full_{k}"] = v
            
        # Process annuli
        for inner_r, outer_r in annuli:
            outer_buf = point_geom.buffer(outer_r * 3.28084)
            inner_buf = point_geom.buffer(inner_r * 3.28084)
            annulus_geom = outer_buf.difference(inner_buf)
            
            annulus_metrics = compute_point_metrics(annulus_geom, landcover_gdf,landcover_sindex, point_geom)
            
            # Add annulus-specific metrics
            suffix = f"_{int(inner_r)}_{int(outer_r)}"
            for k, v in annulus_metrics.items():
                row_metrics[k + suffix] = v
        
        results.append(row_metrics)
    
    # Merge results with original data
    metrics_df = pd.DataFrame(results)
    return gdf.join(metrics_df)

def process_row(args):
    """Process one point (row) and compute its full-buffer and annulus metrics."""
    row, annuli = args
    point_geom = row.geometry
    row_metrics = {}
    
    # Compute full-buffer metrics
    # max_radius = annuli[-1][1]
    # full_buffer = point_geom.buffer(max_radius * 3.28084)
    # full_metrics = compute_point_metrics(full_buffer, global_landcover_gdf, global_landcover_sindex, point_geom)
    # for k, v in full_metrics.items():
    #     row_metrics[f"full_{k}"] = v

    # Process each annulus for this point
    for inner_r, outer_r in annuli:
        outer_buf = point_geom.buffer(outer_r * 3.28084)
        inner_buf = point_geom.buffer(inner_r * 3.28084)
        annulus_geom = outer_buf.difference(inner_buf)
        # Note: Make sure you call the correct compute function
        annulus_metrics = compute_point_metrics(annulus_geom, global_landcover_gdf, global_landcover_sindex, point_geom)
        suffix = f"_{int(inner_r)}_{int(outer_r)}"
        for k, v in annulus_metrics.items():
            row_metrics[k + suffix] = v

    return row_metrics


def calculate_landcover_annuli_metrics_vector_parallel(csv_path, landcover_gdf, landcover_sindex, annuli, crs="EPSG:2263"):
    """Calculate annular metrics in parallel across points."""
    # Load the CSV and convert coordinates to a GeoDataFrame
    data_df = pd.read_csv(csv_path)
    geometry = [Point(lon, lat) for lon, lat in zip(data_df['Longitude'], data_df['Latitude'])]
    gdf = gpd.GeoDataFrame(data_df, geometry=geometry, crs="EPSG:4326").to_crs(crs)
    
    # Create a list of arguments for each point
    args_list = [(row, annuli) for _, row in  gdf.iterrows()]
    
    
    if cpu_count() > 1:
        with Pool(
            processes=max(cpu_count() - 1, 1),
            initializer=init_worker,  # Add this line
            initargs=(landcover_gdf, landcover_sindex),  # Pass data to workers
        ) as pool:
            results = list(tqdm(pool.imap(process_row, args_list, chunksize=10), 
                                total=len(args_list)))
    else:
        results = [process_row(args) for args in tqdm(args_list, desc="Mapping LandCover metrics")]
    
    results_df = pd.DataFrame(results)
    # Merge on a unique identifier (e.g., index) to avoid ambiguous merges
    results_df.index.name = "index"
    gdf = gdf.reset_index().set_index("index")
    merged_df = gdf.join(results_df)
    return merged_df


def main():
    # Read the GeoPackage ONCE at the start
    gpkg_path = "NYCOpenData/landcover_nyc_2021_6in_Clipped_5_Polygonized.gpkg"
    landcover_gdf = gpd.read_file(gpkg_path).to_crs("EPSG:2263")
    landcover_gdf["full_area"] = landcover_gdf.geometry.area  # Original polygon area
    landcover_gdf["centroid"] = landcover_gdf.geometry.centroid  # Original centroid
    landcover_gdf["boundary"] = landcover_gdf.geometry.boundary
    landcover_gdf["LandCover_Class"] = landcover_gdf.LandCover.map({
        1: 'tree_canopy', 2: 'grass_shrub', 3: 'bare_ground',
        4: 'open_water', 5: 'building', 6: 'road', 
        7: 'other_impervious', 8: 'railroad'
    })
    landcover_sindex = landcover_gdf.sindex  


    annuli_in_meters = [
        (0, 10), (0, 20)]

    # Process training data 
    train_result = calculate_landcover_annuli_metrics_vector_parallel(
        csv_path="Train_Data/Training_data_uhi_index_2025-02-18.csv",
        landcover_gdf=landcover_gdf, 
        landcover_sindex=landcover_sindex,
        annuli=annuli_in_meters
    )
    train_result.to_csv("Train_Data/Final/Polygonized_LandCover_Train_IDW_50Circles_18032025.csv", index=False)

    # Process submission data
    submission_result = calculate_landcover_annuli_metrics_vector_parallel(
        csv_path="Submission_Mapped_Data/Submission_template_UHI2025-v2.csv",
        landcover_gdf=landcover_gdf, 
        landcover_sindex=landcover_sindex,
        annuli=annuli_in_meters
    )
    submission_result.to_csv("Submission_Mapped_Data/Final/Polygonized_LandCover_Submission_IDW_50Circles_18032025.csv", index=False)

if __name__ == '__main__':
    main()