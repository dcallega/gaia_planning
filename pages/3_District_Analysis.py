import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import json
from data_utils import (
    ensure_population_csv,
    prepare_population_dataframe,
    POPULATION_CACHE_HASH_FUNCS,
)
from spatial_utils import load_district_boundaries, assign_districts_to_dataframe
from sklearn.neighbors import BallTree

# Page config
st.set_page_config(
    page_title="Planning", page_icon="assets/gaia_icon.png", layout="wide"
)

# Load brand CSS
try:
    with open("brand.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass  # CSS not found, continue with default styling


# Function to parse GPS coordinates
def parse_gps_coordinates(gps_string):
    """Parse GPS string format: 'lat lon elevation accuracy'"""
    try:
        parts = str(gps_string).strip().split()
        if len(parts) >= 2:
            lat = float(parts[0])
            lon = float(parts[1])
            return lat, lon
    except:
        return None, None
    return None, None


# Cache the clinic data
@st.cache_data
def load_clinic_data():
    df = pd.read_csv("data/GAIA MHC Clinic Stops GPS.xlsx - Clinic stops GPS.csv")

    # Parse GPS coordinates
    df[["latitude", "longitude"]] = df["collect_gps_coordinates"].apply(
        lambda x: pd.Series(parse_gps_coordinates(x))
    )

    # Remove rows with invalid coordinates
    df = df.dropna(subset=["latitude", "longitude"])

    return df


def normalize_facility_type(facility_type):
    """
    Normalize facility types by grouping similar types together.
    """
    if pd.isna(facility_type):
        return facility_type
    
    facility_type = str(facility_type).strip()
    
    # Group all hospitals together
    if facility_type in ["Hospital", "District Hospital", "Central Hospital"]:
        return "Hospital"
    
    # Group Health Centre and Health Post together
    if facility_type in ["Health Centre", "Health Post"]:
        return "Health Centre"
    
    # Return original type for others
    return facility_type


# Cache the MHFR facilities data
@st.cache_data
def load_mhfr_facilities():
    df = pd.read_csv("data/MHFR_Facilities.csv")

    # Rename columns to lowercase for consistency
    df = df.rename(
        columns={
            "LATITUDE": "latitude",
            "LONGITUDE": "longitude",
            "COMMON NAME": "common_name",
            "OWNERSHIP": "ownership",
            "TYPE": "type",
            "STATUS": "status",
            "NAME": "name",
            "DISTRICT": "district",
        }
    )

    # Convert coordinates to numeric, handling any errors
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Remove rows with invalid coordinates
    df = df.dropna(subset=["latitude", "longitude"])

    # Filter out rows with empty or zero coordinates
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)]

    # Filter out facilities without a type
    df = df[df["type"].notna() & (df["type"].str.strip() != "")]

    # Filter out "Private" and "Unclassified" facility types
    df = df[~df["type"].isin(["Private", "Unclassified"])]

    # Normalize facility types (group hospitals, group health centres/posts)
    df["type"] = df["type"].apply(normalize_facility_type)

    return df


# Cache population data loading
@st.cache_data(persist="disk", show_spinner=False)
def load_population_data(dataset_name):
    """
    Load FULL population density data filtered to country boundaries.
    """
    # Check for pre-computed cache first
    cache_file = f"data/.cache/mwi_{dataset_name}_2020_filtered.parquet"
    
    try:
        import os
        if os.path.exists(cache_file):
            df = pd.read_parquet(cache_file)
            return prepare_population_dataframe(df, dataset_name)
    except Exception as e:
        print(f"Could not load cache: {e}")
    
    # If no cache, ensure the CSV exists (extract from zip if needed)
    csv_path = ensure_population_csv(dataset_name)
    df = pd.read_csv(csv_path)
    df = df[df[f"mwi_{dataset_name}_2020"] > 0.5]
    
    # Filter to only points inside country boundaries
    from spatial_utils import filter_points_in_country
    df = filter_points_in_country(df, lat_col='latitude', lon_col='longitude')

    df = prepare_population_dataframe(df, dataset_name)
    
    # Save to cache for next time
    try:
        import os
        os.makedirs('data/.cache', exist_ok=True)
        df.to_parquet(cache_file, index=False)
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")
    
    return df


# Cache district assignment separately
@st.cache_data(
    persist="disk",
    show_spinner=False,
    hash_funcs=POPULATION_CACHE_HASH_FUNCS,
)
def assign_districts_to_population(population_df, dataset_name):
    """Assign districts to population data - cached separately for performance"""
    cached_file = f"data/.cache/mwi_{dataset_name}_2020_with_districts.parquet"
    
    try:
        if pd.io.common.file_exists(cached_file):
            cached_df = pd.read_parquet(cached_file)
            if len(cached_df) == len(population_df):
                return prepare_population_dataframe(cached_df, dataset_name)
    except:
        pass
    
    # Assign districts (expensive)
    df = assign_districts_to_dataframe(
        population_df, 
        lat_col='latitude', 
        lon_col='longitude'
    )

    df = prepare_population_dataframe(df, dataset_name)
    
    # Save for next time
    try:
        import os
        os.makedirs('data/.cache', exist_ok=True)
        df.to_parquet(cached_file, index=False)
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")
    
    return df


@st.cache_data
def get_district_list():
    """Get list of all districts"""
    try:
        district_geojson = load_district_boundaries()
        districts = [f['properties']['shapeName'] for f in district_geojson['features']]
        return sorted(districts)
    except:
        return []


@st.cache_data
def get_district_boundary(district_name):
    """Get the boundary for a specific district"""
    try:
        district_geojson = load_district_boundaries()
        for feature in district_geojson['features']:
            if feature['properties']['shapeName'] == district_name:
                return {
                    "type": "FeatureCollection",
                    "features": [feature]
                }
        return None
    except:
        return None


@st.cache_data
def load_subdistrict_boundaries():
    """Load level 3 (Traditional Authority) boundaries"""
    try:
        with open("data/boundaries/malawi_level3.geojson", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


@st.cache_data
def get_subdistrict_boundaries_for_district(district_name):
    """Get sub-district boundaries filtered to a specific district"""
    try:
        level3_geojson = load_subdistrict_boundaries()
        if not level3_geojson:
            return None
        
        # Filter features where NAME_2 matches the district name
        # Note: GADM uses different naming conventions, may need adjustment
        filtered_features = []
        for feature in level3_geojson['features']:
            props = feature['properties']
            # Try to match by NAME_2 (district name in GADM)
            if props.get('NAME_2', '').lower() in district_name.lower() or \
               district_name.lower() in props.get('NAME_2', '').lower() or \
               props.get('NAME_1', '').lower() in district_name.lower() or \
               district_name.lower() in props.get('NAME_1', '').lower():
                filtered_features.append(feature)
        
        if filtered_features:
            return {
                "type": "FeatureCollection",
                "features": filtered_features
            }
        return None
    except:
        return None


from shapely.geometry import shape, Point
from shapely.ops import unary_union


def aggregate_population_by_subdistrict(population_df, subdistrict_geojson, pop_column):
    """
    Aggregate population data by sub-district boundaries.
    
    Args:
        population_df: DataFrame with population points
        subdistrict_geojson: GeoJSON with sub-district boundaries
        pop_column: Name of the population column
        
    Returns:
        GeoJSON with population aggregated by sub-district
    """
    if not subdistrict_geojson or len(population_df) == 0:
        return None
    
    # Create a copy of the geojson with population data
    result_geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    
    # Convert population points to shapely Points once
    from geopandas import GeoDataFrame, points_from_xy
    
    # Create GeoDataFrame for efficient spatial operations
    gdf_pop = GeoDataFrame(
        population_df,
        geometry=points_from_xy(population_df['longitude'], population_df['latitude']),
        crs="EPSG:4326"
    )
    
    for feature in subdistrict_geojson['features']:
        # Get the polygon
        polygon = shape(feature['geometry'])
        
        # Use spatial indexing for faster point-in-polygon test
        mask = gdf_pop.geometry.within(polygon)
        points_in_region = gdf_pop[mask]
        
        # Aggregate population
        pop_in_region = points_in_region[pop_column].sum() if len(points_in_region) > 0 else 0
        count_points = len(points_in_region)
        
        # Add population data to feature properties
        new_feature = feature.copy()
        new_feature['properties'] = feature['properties'].copy()
        new_feature['properties']['population'] = float(pop_in_region)
        new_feature['properties']['point_count'] = count_points
        new_feature['properties']['population_density'] = float(pop_in_region) if count_points > 0 else 0
        
        result_geojson['features'].append(new_feature)
    
    return result_geojson


def get_color_for_population(population, min_pop, max_pop):
    """
    Get RGB color for population value using teal scale.
    
    Args:
        population: Population value
        min_pop: Minimum population in dataset
        max_pop: Maximum population in dataset
        
    Returns:
        [R, G, B, A] color array
    """
    if max_pop == min_pop:
        return [128, 203, 196, 180]  # Medium teal if all same
    
    # Normalize to 0-1
    normalized = (population - min_pop) / (max_pop - min_pop)
    
    # Teal color scale (light to dark)
    if normalized < 0.2:
        return [224, 242, 241, 150]  # Very light teal
    elif normalized < 0.4:
        return [178, 223, 219, 170]  # Light teal
    elif normalized < 0.6:
        return [128, 203, 196, 190]  # Medium-light teal
    elif normalized < 0.8:
        return [38, 166, 154, 210]   # Medium-dark teal
    else:
        return [0, 121, 107, 230]    # Dark teal


def sample_for_visualization(df, sample_size=30000):
    """Sample population data for map visualization only."""
    if len(df) <= sample_size:
        return df
    return df.sample(n=sample_size, random_state=42)


# Constants for mobile clinic planning
EARTH_RADIUS_KM = 6371.0088
MAX_TRAVEL_DISTANCE_KM = 30.0  # Maximum distance from hospital
STOPS_PER_CREW = 5  # Each crew operates 5 stops per week
SERVICE_RADIUS_KM = 5.0  # Service radius for each stop


def get_color_for_unserved(population, min_pop, max_pop):
    """
    Get RGBA color for unserved population using a red scale.
    """
    if max_pop == min_pop:
        return [244, 199, 199, 180]
    
    normalized = (population - min_pop) / (max_pop - min_pop)
    
    if normalized < 0.2:
        return [255, 235, 238, 160]  # very light red
    elif normalized < 0.4:
        return [255, 205, 210, 180]
    elif normalized < 0.6:
        return [239, 154, 154, 200]
    elif normalized < 0.8:
        return [229, 115, 115, 220]
    else:
        return [198, 40, 40, 240]  # dark red


def select_stops_for_hospital(hospital, gap_points, pop_column):
    """
    Select up to STOPS_PER_CREW high-impact stops for a hospital using a greedy max-coverage heuristic.
    
    Args:
        hospital: Pandas Series representing the hospital row.
        gap_points: DataFrame of uncovered population points (must retain original indices).
        pop_column: Name of the population column.
        
    Returns:
        (stops_df, covered_indices):
            stops_df: DataFrame with columns [latitude, longitude, population, distance_from_hospital].
            covered_indices: set of indices from gap_points that are newly covered by the selected stops.
    """
    if gap_points is None or len(gap_points) == 0:
        return pd.DataFrame(), set()
    
    hospital_lat = hospital['latitude']
    hospital_lon = hospital['longitude']
    
    # Vectorized haversine distance from hospital to candidate gap points
    distances = haversine_distance_km(
        hospital_lat,
        hospital_lon,
        gap_points['latitude'].to_numpy(dtype=np.float64),
        gap_points['longitude'].to_numpy(dtype=np.float64),
    )
    
    within_mask = distances <= MAX_TRAVEL_DISTANCE_KM
    if not np.any(within_mask):
        return pd.DataFrame(), set()
    
    nearby = gap_points.loc[within_mask].copy()
    nearby['distance_from_hospital'] = distances[within_mask]
    
    service_radius_rad = SERVICE_RADIUS_KM / EARTH_RADIUS_KM
    remaining = nearby.copy()
    
    selected_stops = []
    covered_indices = set()
    
    while len(selected_stops) < STOPS_PER_CREW and len(remaining) > 0:
        best_idx = remaining[pop_column].idxmax()
        best_point = remaining.loc[best_idx]
        
        if best_point[pop_column] <= 0:
            break
        
        remaining_coords_rad = np.deg2rad(
            remaining[['latitude', 'longitude']].to_numpy(dtype=np.float64)
        )
        best_coord_rad = np.deg2rad(
            [[best_point['latitude'], best_point['longitude']]]
        )
        
        remaining_tree = BallTree(remaining_coords_rad, metric='haversine')
        neighbors_pos = remaining_tree.query_radius(best_coord_rad, r=service_radius_rad)[0]
        
        if neighbors_pos.size == 0:
            remaining = remaining.drop(index=best_idx)
            continue
        
        covered = remaining.iloc[neighbors_pos]
        total_population = covered[pop_column].sum()
        
        if total_population <= 0:
            remaining = remaining.drop(index=covered.index)
            continue
        
        weights = covered[pop_column].to_numpy(dtype=np.float64)
        weighted_lat = np.average(covered['latitude'], weights=weights)
        weighted_lon = np.average(covered['longitude'], weights=weights)
        
        distance_to_hospital = haversine_distance_km(
            hospital_lat,
            hospital_lon,
            weighted_lat,
            weighted_lon,
        )
        
        if distance_to_hospital > MAX_TRAVEL_DISTANCE_KM:
            weighted_lat = float(best_point['latitude'])
            weighted_lon = float(best_point['longitude'])
            distance_to_hospital = float(best_point['distance_from_hospital'])
        
        selected_stops.append(
            {
                "latitude": float(weighted_lat),
                "longitude": float(weighted_lon),
                "population": float(total_population),
                "distance_from_hospital": float(distance_to_hospital),
            }
        )
        
        covered_indices.update(covered.index.tolist())
        remaining = remaining.drop(index=covered.index)
    
    if not selected_stops:
        return pd.DataFrame(), set()
    
    stops_df = pd.DataFrame(selected_stops)
    return stops_df, covered_indices


def compute_unserved_population(
    district_population,
    district_facilities,
    existing_mobile_stops,
    pop_column,
):
    """
    Identify population points not currently served by existing facilities or GAIA mobile stops.
    """
    coverage_sources = []
    
    if len(district_facilities) > 0:
        coverage_sources.append(
            district_facilities[['latitude', 'longitude']].dropna()
        )
    
    if existing_mobile_stops is not None and len(existing_mobile_stops) > 0:
        coverage_sources.append(
            existing_mobile_stops[['latitude', 'longitude']].dropna()
        )
    
    if coverage_sources:
        combined_sources = pd.concat(coverage_sources, ignore_index=True)
    else:
        combined_sources = pd.DataFrame(columns=['latitude', 'longitude'])
    
    service_radius_rad = SERVICE_RADIUS_KM / EARTH_RADIUS_KM
    
    if len(combined_sources) > 0:
        coords_rad = np.deg2rad(
            combined_sources[['latitude', 'longitude']].to_numpy(dtype=np.float64)
        )
        facility_tree = BallTree(coords_rad, metric='haversine')
        
        pop_coords_rad = np.deg2rad(
            district_population[['latitude', 'longitude']].to_numpy(dtype=np.float64)
        )
        
        dist_rad, _ = facility_tree.query(pop_coords_rad, k=1)
        gap_mask = dist_rad.ravel() > service_radius_rad
        gap_points = district_population[gap_mask].copy()
    else:
        gap_points = district_population.copy()
    
    return gap_points


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two points in kilometers."""
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return EARTH_RADIUS_KM * c


def plan_mobile_clinic_stops(
    district_population,
    district_facilities,
    existing_mobile_stops,
    district_name,
    num_crews,
    pop_column="mwi_general_2020",
):
    """
    Plan mobile clinic stops for a district.
    
    Args:
        district_population: DataFrame with population points in the district
        district_facilities: DataFrame with facilities in the district
        existing_mobile_stops: DataFrame with existing GAIA clinic stops serving the district
        district_name: Name of the district
        num_crews: Number of crews to deploy
        pop_column: Name of the population column to optimize
        
    Returns:
        DataFrame with proposed clinic stop locations
    """
    if len(district_population) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    hospitals = district_facilities[
        district_facilities['type'] == 'Hospital'
    ].copy()
    
    if len(hospitals) == 0:
        # If no hospitals, fall back to the largest available facilities as dispatch points.
        if pop_column in district_facilities.columns:
            hospitals = district_facilities.nlargest(1, pop_column).copy()
        else:
            hospitals = district_facilities.head(1).copy()
    
    if len(hospitals) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    gap_points = compute_unserved_population(
        district_population,
        district_facilities,
        existing_mobile_stops,
        pop_column,
    )
    initial_gap_points = gap_points.copy()
    
    service_radius_rad = SERVICE_RADIUS_KM / EARTH_RADIUS_KM
    
    if len(gap_points) == 0:
        return pd.DataFrame(), initial_gap_points
    
    hospital_candidates = hospitals.dropna(subset=['latitude', 'longitude']).copy()
    if len(hospital_candidates) == 0:
        return pd.DataFrame(), initial_gap_points
    
    current_gap_points = gap_points.copy()
    all_stops = []
    
    for crew_idx in range(num_crews):
        best_plan = None
        best_covered_indices = set()
        best_hospital = None
        best_coverage = 0.0
        
        for _, hospital in hospital_candidates.iterrows():
            selected_stops, covered_indices = select_stops_for_hospital(
                hospital,
                current_gap_points,
                pop_column,
            )
            
            if len(selected_stops) == 0:
                continue
            
            total_coverage = selected_stops['population'].sum()
            completeness_bonus = 1 if len(selected_stops) == STOPS_PER_CREW else 0
            
            if (
                total_coverage > best_coverage
                or (
                    np.isclose(total_coverage, best_coverage)
                    and completeness_bonus > 0
                )
            ):
                best_plan = selected_stops
                best_covered_indices = covered_indices
                best_hospital = hospital
                best_coverage = total_coverage
        
        if best_plan is None or len(best_plan) == 0:
            break
        
        hospital_name = best_hospital.get(
            'common_name',
            best_hospital.get('name', f'Hospital {crew_idx + 1}')
        )
        
        best_plan = best_plan.sort_values('population', ascending=False).reset_index(drop=True)
        best_plan['crew_id'] = crew_idx + 1
        best_plan['hospital_name'] = hospital_name
        best_plan['hospital_lat'] = float(best_hospital['latitude'])
        best_plan['hospital_lon'] = float(best_hospital['longitude'])
        best_plan['district'] = district_name
        best_plan['stop_number'] = range(1, len(best_plan) + 1)
        
        all_stops.append(best_plan)
        
        if best_covered_indices:
            current_gap_points = current_gap_points.drop(index=list(best_covered_indices))
            if len(current_gap_points) == 0:
                break
    
    if all_stops:
        return pd.concat(all_stops, ignore_index=True), initial_gap_points
    return pd.DataFrame(), initial_gap_points


def create_star_polygon(center_lon, center_lat, radius_km=1.5, num_points=5):
    """
    Create a star polygon centered at given coordinates.
    
    Args:
        center_lon: Longitude of star center
        center_lat: Latitude of star center
        radius_km: Radius of the star in kilometers
        num_points: Number of points on the star (default 5)
    
    Returns:
        List of [lon, lat] coordinates forming a star polygon
    """
    # Convert radius from km to degrees (approximate)
    radius_deg = radius_km / 111.0  # 1 degree ≈ 111 km
    inner_radius_deg = radius_deg * 0.4  # Inner points are 40% of outer radius
    
    import math
    coordinates = []
    
    # Create star by alternating between outer and inner radius points
    for i in range(num_points * 2):
        angle = (i * math.pi / num_points) - (math.pi / 2)  # Start from top
        
        if i % 2 == 0:  # Outer point
            r = radius_deg
        else:  # Inner point
            r = inner_radius_deg
        
        lon = center_lon + r * math.cos(angle) / math.cos(math.radians(center_lat))
        lat = center_lat + r * math.sin(angle)
        coordinates.append([lon, lat])
    
    # Close the polygon
    coordinates.append(coordinates[0])
    
    return coordinates


def create_rectangle_polygon(center_lon, center_lat, width_km=1.2, height_km=1.2):
    """
    Create a rectangle polygon centered at given coordinates.
    
    Args:
        center_lon: Longitude of rectangle center
        center_lat: Latitude of rectangle center
        width_km: Width of the rectangle in kilometers
        height_km: Height of the rectangle in kilometers
    
    Returns:
        List of [lon, lat] coordinates forming a rectangle polygon
    """
    # Convert from km to degrees (approximate)
    half_height_deg = (height_km / 2) / 111.0  # 1 degree ≈ 111 km
    half_width_deg = (width_km / 2) / 111.0
    
    # Adjust for latitude distortion
    lon_scale = 1.0 / np.cos(np.radians(center_lat))
    
    # Create rectangle (4 corners)
    coordinates = [
        [center_lon - half_width_deg * lon_scale, center_lat + half_height_deg],  # Top-left
        [center_lon + half_width_deg * lon_scale, center_lat + half_height_deg],  # Top-right
        [center_lon + half_width_deg * lon_scale, center_lat - half_height_deg],  # Bottom-right
        [center_lon - half_width_deg * lon_scale, center_lat - half_height_deg],  # Bottom-left
        [center_lon - half_width_deg * lon_scale, center_lat + half_height_deg]   # Close
    ]
    
    return coordinates


def create_cross_polygon(center_lon, center_lat, radius_km=0.5):
    """
    Create a cross (plus sign) polygon centered at given coordinates.
    
    Args:
        center_lon: Longitude of cross center
        center_lat: Latitude of cross center
        radius_km: Radius of the cross in kilometers
    
    Returns:
        List of [lon, lat] coordinates forming a cross polygon
    """
    # Convert radius from km to degrees (approximate)
    radius_deg = radius_km / 111.0  # 1 degree ≈ 111 km
    arm_width = radius_deg * 0.35  # Width of each arm
    
    # Adjust for latitude distortion
    lon_scale = 1.0 / np.cos(np.radians(center_lat))
    
    # Create cross shape (12 points forming a + shape)
    coordinates = [
        # Top arm (going clockwise from top-left)
        [center_lon - arm_width * lon_scale, center_lat + arm_width],
        [center_lon - arm_width * lon_scale, center_lat + radius_deg],
        [center_lon + arm_width * lon_scale, center_lat + radius_deg],
        [center_lon + arm_width * lon_scale, center_lat + arm_width],
        # Right arm
        [center_lon + radius_deg * lon_scale, center_lat + arm_width],
        [center_lon + radius_deg * lon_scale, center_lat - arm_width],
        [center_lon + arm_width * lon_scale, center_lat - arm_width],
        # Bottom arm
        [center_lon + arm_width * lon_scale, center_lat - radius_deg],
        [center_lon - arm_width * lon_scale, center_lat - radius_deg],
        [center_lon - arm_width * lon_scale, center_lat - arm_width],
        # Left arm
        [center_lon - radius_deg * lon_scale, center_lat - arm_width],
        [center_lon - radius_deg * lon_scale, center_lat + arm_width],
        # Close the polygon
        [center_lon - arm_width * lon_scale, center_lat + arm_width]
    ]
    
    return coordinates


def district_analysis_page():
    """District-level analysis page"""
    # Import here to avoid circular imports
    from app import render_navigation
    
    # Hero section
    st.markdown(
        """
        <div class="hero-lite">
          <h1>District-Level Analysis</h1>
          <p>In-depth analysis of population coverage and facility distribution by district</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Navigation bar after hero
    if "navigation_pages" in st.session_state:
        render_navigation(st.session_state.navigation_pages)

    # Get list of districts
    districts = get_district_list()
    
    if not districts:
        st.error("Could not load district boundaries. Please ensure boundaries are downloaded.")
        return

    # Create layout: controls on top, map below
    col_district, col_crew = st.columns(2)

    with col_district:
        st.markdown("### Select District")
        
        # District selector
        selected_district = st.selectbox(
            "District",
            districts,
            index=districts.index("Chikwawa") if "Chikwawa" in districts else 0,
            label_visibility="collapsed"
        )
    
    with col_crew:
        st.markdown("### Number of Mobile Clinic Crews")
        
        # Crew selector
        num_crews = st.selectbox(
            "Crews to deploy",
            options=[1, 2, 3, 4, 5],
            index=0,
            label_visibility="collapsed",
            help="Each crew will have 5 weekly clinic stops within 30km of a hospital"
        )

    # Load data
    try:
        clinic_df = load_clinic_data()
        mhfr_df = load_mhfr_facilities()
        
        # Load population data (General population only)
        dataset_name = "general"
        pop_column = "mwi_general_2020"
        
        with st.spinner("Loading population data..."):
            population_df = load_population_data(dataset_name)
        
        # Assign districts to population data
        with st.spinner("Assigning districts to population data..."):
            population_df = assign_districts_to_population(
                population_df,
                dataset_name
            )
        
        # Filter population data for selected district
        district_population = population_df[
            population_df['assigned_district'] == selected_district
        ].copy()
        
        # Assign districts to facilities if not already done
        if 'assigned_district' not in mhfr_df.columns:
            with st.spinner("Assigning districts to facilities..."):
                mhfr_df = assign_districts_to_dataframe(
                    mhfr_df,
                    lat_col='latitude',
                    lon_col='longitude'
                )
        
        if 'assigned_district' not in clinic_df.columns:
            with st.spinner("Assigning districts to clinics..."):
                clinic_df = assign_districts_to_dataframe(
                    clinic_df,
                    lat_col='latitude',
                    lon_col='longitude'
                )
        
        # Filter facilities for selected district
        district_mhfr = mhfr_df[
            mhfr_df['assigned_district'] == selected_district
        ].copy()
        
        district_clinics = clinic_df[
            clinic_df['assigned_district'] == selected_district
        ].copy()
        
        # Plan mobile clinic stops based on selected number of crews
        with st.spinner(f"Planning deployment for {num_crews} crew(s)..."):
            proposed_stops, coverage_gaps = plan_mobile_clinic_stops(
                district_population,
                district_mhfr,
                district_clinics,
                selected_district,
                num_crews,
                pop_column=pop_column,
            )
        
        # Add CSS for button styles
        st.markdown(
            """
        <style>
        .stButton > button[kind="primary"] {
            background-color: #3A5A34 !important;
            border: 2px solid #3A5A34 !important;
            color: white !important;
            border-radius: 0.5rem !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.2s ease !important;
            box-shadow: none !important;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #2f4c2b !important;
            border: 2px solid #2f4c2b !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        }
        .stButton > button[kind="secondary"] {
            background-color: #f4f6f4 !important;
            border: 2px solid #3A5A34 !important;
            color: #3A5A34 !important;
            border-radius: 0.5rem !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.2s ease !important;
            box-shadow: none !important;
        }
        .stButton > button[kind="secondary"]:hover {
            background-color: #e8ebe7 !important;
            border: 2px solid #2f4c2b !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # Add choropleth legend above facility legend
        # st.markdown("""
        # <div style="background-color: #f8f9fa; padding: 10px; border-radius: 8px; margin-bottom: 15px;">
        #     <div style="font-size: 14px; font-weight: bold; margin-bottom: 8px; color: #2C3E50;">
        #         📊 Population Density (Choropleth Map)
        #     </div>
        #     <div style="display: flex; align-items: center; justify-content: space-between; margin: 0 10px;">
        #         <span style="font-size: 12px; color: #666;">Low</span>
        #         <div style="flex: 1; height: 20px; margin: 0 10px; background: linear-gradient(to right, 
        #             rgb(224, 242, 241), 
        #             rgb(178, 223, 219), 
        #             rgb(128, 203, 196), 
        #             rgb(38, 166, 154), 
        #             rgb(0, 121, 107)); 
        #             border-radius: 4px; border: 1px solid #ccc;">
        #         </div>
        #         <span style="font-size: 12px; color: #666;">High</span>
        #     </div>
        # </div>
        # """, unsafe_allow_html=True)

        legend_items = [
            ("Hospital", "🔴"),
            ("Health Centre", "🟢"),
            ("Clinic", "🟠"),
            ("Dispensary", "🟣"),
            ("GAIA Mobile", "🔵"),
        ]

        # Initialize session state for facility visibility
        if "district_facility_visibility" not in st.session_state:
            st.session_state.district_facility_visibility = {
                item[0]: True for item in legend_items if item[0] != "GAIA Mobile"
            }
            st.session_state.district_facility_visibility["GAIA Mobile"] = True

        # Create clickable legend items
        legend_cols = st.columns(5)
        facility_visibility = {}

        for col, (facility_type, emoji) in zip(legend_cols, legend_items):
            with col:
                if facility_type == "GAIA Mobile":
                    if "district_gaia_mobile_visible" not in st.session_state:
                        st.session_state.district_gaia_mobile_visible = True
                    is_visible = st.session_state.district_gaia_mobile_visible
                else:
                    is_visible = st.session_state.district_facility_visibility.get(
                        facility_type, True
                    )

                button_text = f"{emoji} {facility_type}"
                button_type = "primary" if is_visible else "secondary"

                if st.button(
                    button_text,
                    key=f"district_btn_{facility_type}",
                    width="stretch",
                    type=button_type,
                ):
                    if facility_type == "GAIA Mobile":
                        st.session_state.district_gaia_mobile_visible = not is_visible
                    else:
                        st.session_state.district_facility_visibility[facility_type] = (
                            not is_visible
                        )
                    st.rerun()

                facility_visibility[facility_type] = is_visible

        # Create map layers
        layers = []

        # Population density choropleth layer (replaces heatmap)
        if len(district_population) > 0:
            # Get sub-district boundaries for the selected district
            subdistrict_boundaries = get_subdistrict_boundaries_for_district(selected_district)
            
            if subdistrict_boundaries:
                # Aggregate population by sub-district
                with st.spinner("Creating choropleth map..."):
                    choropleth_data = aggregate_population_by_subdistrict(
                        district_population,
                        subdistrict_boundaries,
                        pop_column
                    )
                
                if choropleth_data and choropleth_data['features']:
                    # Get min/max population for color scaling
                    populations = [f['properties']['population'] for f in choropleth_data['features']]
                    min_pop = min(populations) if populations else 0
                    max_pop = max(populations) if populations else 1
                    
                    # Add colors to each feature
                    for feature in choropleth_data['features']:
                        pop = feature['properties']['population']
                        color = get_color_for_population(pop, min_pop, max_pop)
                        feature['properties']['fill_color'] = color
                    
                    # Create choropleth layer
                    layers.append(
                        pdk.Layer(
                            "GeoJsonLayer",
                            data=choropleth_data,
                            opacity=0.7,
                            stroked=True,
                            filled=True,
                            extruded=False,
                            wireframe=False,
                            get_fill_color="properties.fill_color",
                            get_line_color=[150, 150, 150, 100],  # Light gray borders
                            line_width_min_pixels=1,
                            pickable=True,
                            auto_highlight=True,
                        )
                    )
            else:
                # Fallback to simple visualization if boundaries not available
                st.info("Sub-district boundaries not available. Using simple point visualization.")
                sampled_pop = sample_for_visualization(district_population, 10000)
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=sampled_pop,
                        get_position=["longitude", "latitude"],
                        get_radius=200,
                        get_fill_color=[128, 203, 196, 100],  # Teal color
                        pickable=False,
                    )
                )

        # MHFR Facilities layers
        if len(district_mhfr) > 0:
            facility_colors = {
                "Hospital": [220, 20, 60, 220],
                "Health Centre": [50, 205, 50, 200],
                "Clinic": [255, 165, 0, 200],
                "Dispensary": [147, 112, 219, 200],
            }

            for facility_type in district_mhfr["type"].unique():
                if not facility_visibility.get(facility_type, True):
                    continue

                type_df = district_mhfr[district_mhfr["type"] == facility_type].copy()
                type_df = type_df.dropna(subset=["latitude", "longitude"])

                if len(type_df) > 0:
                    # Hospitals get green rectangle with white cross
                    if facility_type == "Hospital":
                        rectangle_features = []
                        cross_features = []
                        
                        for idx, row in type_df.iterrows():
                            # TWEAK SIZE HERE: adjust rectangle_size (in kilometers)
                            # The cross will automatically scale proportionally
                            rectangle_size = 3.7  # Rectangle width & height (in km)
                            
                            # Create green rectangle background
                            rect_coords = create_rectangle_polygon(
                                row['longitude'], 
                                row['latitude'], 
                                width_km=rectangle_size,
                                height_km=rectangle_size
                            )
                            
                            rect_feature = {
                                "type": "Feature",
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [rect_coords]
                                },
                                "properties": {
                                    "common_name": row.get('common_name', ''),
                                    "name": row.get('name', ''),
                                    "type": row.get('type', ''),
                                    "ownership": row.get('ownership', ''),
                                    "status": row.get('status', ''),
                                    "district": row.get('district', '')
                                }
                            }
                            rectangle_features.append(rect_feature)
                            
                            # Create white cross on top (proportional to rectangle)
                            # Cross is 65% of rectangle size with nice padding
                            cross_size = rectangle_size * 0.325  # 32.5% radius = 65% width
                            cross_coords = create_cross_polygon(
                                row['longitude'], 
                                row['latitude'], 
                                radius_km=cross_size
                            )
                            
                            cross_feature = {
                                "type": "Feature",
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [cross_coords]
                                },
                                "properties": {
                                    "common_name": row.get('common_name', ''),
                                    "name": row.get('name', ''),
                                    "type": row.get('type', ''),
                                    "ownership": row.get('ownership', ''),
                                    "status": row.get('status', ''),
                                    "district": row.get('district', '')
                                }
                            }
                            cross_features.append(cross_feature)
                        
                        # Add green rectangle layer
                        rectangles_geojson = {
                            "type": "FeatureCollection",
                            "features": rectangle_features
                        }
                        
                        layers.append(
                            pdk.Layer(
                                "GeoJsonLayer",
                                data=rectangles_geojson,
                                opacity=0.95,
                                stroked=True,
                                filled=True,
                                extruded=False,
                                get_fill_color=[34, 139, 34, 255],  # Forest green fill
                                get_line_color=[25, 100, 25, 255],   # Darker green border
                                line_width_min_pixels=2,
                                pickable=True,
                                auto_highlight=True,
                            )
                        )
                        
                        # Add white cross layer on top
                        crosses_geojson = {
                            "type": "FeatureCollection",
                            "features": cross_features
                        }
                        
                        layers.append(
                            pdk.Layer(
                                "GeoJsonLayer",
                                data=crosses_geojson,
                                opacity=1.0,
                                stroked=False,
                                filled=True,
                                extruded=False,
                                get_fill_color=[255, 255, 255, 255],  # White fill
                                pickable=True,
                                auto_highlight=True,
                            )
                        )
                    else:
                        # Other facilities use circles
                        radius = 300
                        radius_min = 4
                        radius_max = 30
                        line_width = 1
                        line_color = [0, 0, 0]
                        color = facility_colors.get(facility_type, [128, 128, 128, 180])

                        layers.append(
                            pdk.Layer(
                                "ScatterplotLayer",
                                data=type_df,
                                get_position=["longitude", "latitude"],
                                get_color=color,
                                get_radius=radius,
                                pickable=True,
                                opacity=0.8,
                                stroked=True,
                                filled=True,
                                line_width_min_pixels=line_width,
                                get_line_color=line_color,
                                radius_min_pixels=radius_min,
                                radius_max_pixels=radius_max,
                            )
                        )

        # GAIA clinics layer
        gaia_visible = st.session_state.get("district_gaia_mobile_visible", True)
        if gaia_visible and len(district_clinics) > 0:
            clinic_clean = district_clinics.dropna(subset=["latitude", "longitude"])

            if len(clinic_clean) > 0:
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=clinic_clean,
                        get_position=["longitude", "latitude"],
                        get_color=[0, 150, 255, 200],
                        get_radius=800,
                        pickable=True,
                        opacity=0.8,
                        stroked=True,
                        filled=True,
                        line_width_min_pixels=1,
                        get_line_color=[0, 0, 0],
                        radius_min_pixels=4,
                        radius_max_pixels=30,
                    )
                )

        # District boundary layer
        district_boundary = get_district_boundary(selected_district)
        if district_boundary:
            layers.append(
                pdk.Layer(
                    "GeoJsonLayer",
                    data=district_boundary,
                    opacity=0.1,
                    stroked=True,
                    filled=True,
                    extruded=False,
                    wireframe=True,
                    get_fill_color=[58, 90, 52, 30],
                    get_line_color=[58, 90, 52, 255],
                    line_width_min_pixels=3,
                    pickable=True,
                )
            )

        # Calculate map center from district data
        center_lat = -13.5
        center_lon = 34.0
        zoom_level = 9

        if len(district_population) > 0:
            center_lat = district_population['latitude'].mean()
            center_lon = district_population['longitude'].mean()
        elif len(district_mhfr) > 0:
            center_lat = district_mhfr['latitude'].mean()
            center_lon = district_mhfr['longitude'].mean()

        # Create the map
        view_state = pdk.ViewState(
            latitude=float(center_lat),
            longitude=float(center_lon),
            zoom=zoom_level,
            pitch=0,
        )

        r = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_provider="carto",
            map_style="road",
        )

        st.pydeck_chart(r, width="stretch")

        # Display summary metrics
        st.markdown("---")
        st.markdown(f"### 📊 {selected_district} District Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_pop = district_population[pop_column].sum() if len(district_population) > 0 else 0
            st.metric("Total Population", f"{total_pop:,.0f}")
        
        with col2:
            num_facilities = len(district_mhfr)
            st.metric("Health Facilities", num_facilities)
        
        with col3:
            num_gaia = len(district_clinics)
            st.metric("GAIA Clinic Stops", num_gaia)
        
        with col4:
            if len(district_population) > 0:
                pop_points = len(district_population)
                st.metric("Population Data Points", f"{pop_points:,}")
            else:
                st.metric("Population Data Points", "0")
        
        # Display recommendations
        if len(proposed_stops) > 0:
            st.markdown("---")
            st.markdown("### ⭐ Recommended Mobile Clinic Deployments")
            
            map_col, details_col = st.columns((2, 1))
            
            with map_col:
                st.markdown("#### Unserved Population & Deployment Plan")
                
                deployment_layers = []
                
                # Choropleth of uncovered population (red scale)
                if len(coverage_gaps) > 0:
                    subdistrict_boundaries = get_subdistrict_boundaries_for_district(selected_district)
                    
                    if subdistrict_boundaries:
                        choropleth_data = aggregate_population_by_subdistrict(
                            coverage_gaps,
                            subdistrict_boundaries,
                            pop_column
                        )
                        
                        if choropleth_data and choropleth_data['features']:
                            populations = [f['properties']['population'] for f in choropleth_data['features']]
                            min_pop = min(populations) if populations else 0
                            max_pop = max(populations) if populations else 1
                            
                            for feature in choropleth_data['features']:
                                pop = feature['properties']['population']
                                feature['properties']['fill_color'] = get_color_for_unserved(pop, min_pop, max_pop)
                            
                            deployment_layers.append(
                                pdk.Layer(
                                    "GeoJsonLayer",
                                    data=choropleth_data,
                                    opacity=0.75,
                                    stroked=True,
                                    filled=True,
                                    extruded=False,
                                    wireframe=False,
                                    get_fill_color="properties.fill_color",
                                    get_line_color=[180, 70, 70, 120],
                                    line_width_min_pixels=1,
                                    pickable=True,
                                    auto_highlight=True,
                                )
                            )
                    else:
                        sampled_gaps = sample_for_visualization(coverage_gaps, 15000)
                        deployment_layers.append(
                            pdk.Layer(
                                "ScatterplotLayer",
                                data=sampled_gaps,
                                get_position=["longitude", "latitude"],
                                get_radius=200,
                                get_fill_color=[229, 115, 115, 160],
                                pickable=False,
                            )
                        )
                else:
                    st.info("✅ All population points are within reach of existing services.")
                
                # Hospital hubs (dots)
                hospitals_for_map = pd.DataFrame()
                if len(proposed_stops) > 0:
                    hospitals_for_map = (
                        proposed_stops[['hospital_name', 'hospital_lat', 'hospital_lon']]
                        .dropna()
                        .drop_duplicates()
                        .rename(columns={'hospital_lat': 'latitude', 'hospital_lon': 'longitude'})
                    )
                
                if len(hospitals_for_map) == 0:
                    hospitals_for_map = (
                        district_mhfr[district_mhfr['type'] == 'Hospital'][['common_name', 'latitude', 'longitude']]
                        .rename(columns={'common_name': 'hospital_name'})
                        .dropna()
                    )
                
                if len(hospitals_for_map) > 0:
                    rectangle_features = []
                    cross_features = []
                    for _, row in hospitals_for_map.iterrows():
                        rectangle_size = 3.7
                        rect_coords = create_rectangle_polygon(
                            row['longitude'],
                            row['latitude'],
                            width_km=rectangle_size,
                            height_km=rectangle_size
                        )
                        rectangle_features.append(
                            {
                                "type": "Feature",
                                "geometry": {"type": "Polygon", "coordinates": [rect_coords]},
                                "properties": {
                                    "hospital_name": row.get('hospital_name', row.get('common_name', '')),
                                },
                            }
                        )
                        cross_coords = create_cross_polygon(
                            row['longitude'],
                            row['latitude'],
                            radius_km=rectangle_size * 0.325,
                        )
                        cross_features.append(
                            {
                                "type": "Feature",
                                "geometry": {"type": "Polygon", "coordinates": [cross_coords]},
                                "properties": {
                                    "hospital_name": row.get('hospital_name', row.get('common_name', '')),
                                },
                            }
                        )
                    
                    if rectangle_features:
                        deployment_layers.append(
                            pdk.Layer(
                                "GeoJsonLayer",
                                data={"type": "FeatureCollection", "features": rectangle_features},
                                opacity=0.95,
                                stroked=True,
                                filled=True,
                                extruded=False,
                                get_fill_color=[34, 139, 34, 255],
                                get_line_color=[25, 100, 25, 255],
                                line_width_min_pixels=2,
                                pickable=True,
                                auto_highlight=True,
                            )
                        )
                    if cross_features:
                        deployment_layers.append(
                            pdk.Layer(
                                "GeoJsonLayer",
                                data={"type": "FeatureCollection", "features": cross_features},
                                opacity=1.0,
                                stroked=False,
                                filled=True,
                                extruded=False,
                                get_fill_color=[255, 255, 255, 255],
                                pickable=True,
                                auto_highlight=True,
                            )
                        )
                
                # Proposed stops (stars)
                proposed_clean = proposed_stops.dropna(subset=["latitude", "longitude"]).copy()
                if len(proposed_clean) > 0:
                    star_features = []
                    for _, row in proposed_clean.iterrows():
                        star_coords = create_star_polygon(
                            row['longitude'],
                            row['latitude'],
                            radius_km=1.5,
                            num_points=5
                        )
                        feature = {
                            "type": "Feature",
                            "geometry": {"type": "Polygon", "coordinates": [star_coords]},
                            "properties": {
                                "crew_id": row.get('crew_id', ''),
                                "stop_number": row.get('stop_number', ''),
                                "hospital_name": row.get('hospital_name', ''),
                                "population": row.get('population', 0),
                                "distance_from_hospital": row.get('distance_from_hospital', 0),
                            },
                        }
                        star_features.append(feature)
                    
                    stars_geojson = {
                        "type": "FeatureCollection",
                        "features": star_features
                    }
                    
                    deployment_layers.append(
                        pdk.Layer(
                            "GeoJsonLayer",
                            data=stars_geojson,
                            opacity=0.95,
                            stroked=True,
                            filled=True,
                            extruded=False,
                            get_fill_color=[255, 215, 0, 255],
                            get_line_color=[255, 140, 0, 255],
                            line_width_min_pixels=2,
                            pickable=True,
                            auto_highlight=True,
                        )
                    )
                
                deployment_center_lat = center_lat
                deployment_center_lon = center_lon
                
                if len(coverage_gaps) > 0:
                    deployment_center_lat = coverage_gaps['latitude'].mean()
                    deployment_center_lon = coverage_gaps['longitude'].mean()
                elif len(proposed_clean) > 0:
                    deployment_center_lat = proposed_clean['latitude'].mean()
                    deployment_center_lon = proposed_clean['longitude'].mean()
                
                deployment_view_state = pdk.ViewState(
                    latitude=float(deployment_center_lat) if not pd.isna(deployment_center_lat) else float(center_lat),
                    longitude=float(deployment_center_lon) if not pd.isna(deployment_center_lon) else float(center_lon),
                    zoom=zoom_level,
                    pitch=0,
                )
                
                if deployment_layers:
                    deployment_map = pdk.Deck(
                        layers=deployment_layers,
                        initial_view_state=deployment_view_state,
                        map_provider="carto",
                        map_style="road",
                        tooltip={
                            "html": """
                                <b>{hospital_name}</b><br/>
                                Crew {crew_id}, Stop {stop_number}<br/>
                                Population: {population}<br/>
                                Distance from hub: {distance_from_hospital:.1f} km
                            """,
                            "style": {"backgroundColor": "#2C3E50", "color": "white"},
                        },
                    )
                    st.pydeck_chart(deployment_map, width="stretch")
                else:
                    st.info("No deployment layers to display.")
            
            with details_col:
                st.markdown("#### Crew Schedules")
                total_additional_coverage = proposed_stops['population'].sum()
                st.metric("Total Newly Covered Residents", f"{total_additional_coverage:,.0f}")
                
                crew_ids = sorted(proposed_stops['crew_id'].unique())
                for idx, crew_id in enumerate(crew_ids):
                    crew_stops = proposed_stops[proposed_stops['crew_id'] == crew_id]
                    hospital_name = crew_stops.iloc[0]['hospital_name']
                    crew_total = crew_stops['population'].sum()
                    
                    with st.expander(
                        f"🚑 Crew {crew_id}: {hospital_name} (~{crew_total:,.0f} people)",
                        expanded=(idx == 0),
                    ):
                        formatted_rows = []
                        for _, stop in crew_stops.sort_values('stop_number').iterrows():
                            stop_number = int(stop.get('stop_number', 0))
                            population_served = f"{stop.get('population', 0):,.0f}"
                            distance_km = stop.get('distance_from_hospital', 0.0)
                            formatted_rows.append(
                                f"<div style='padding:6px 10px; border:1px solid #e0e0e0; border-radius:6px; margin-bottom:6px;'>"
                                f"<strong>Stop {stop_number}</strong> — {population_served} people<br/>"
                                f"<span style='color:#6c757d;'>Distance from hub: {distance_km:.1f} km</span>"
                                f"</div>"
                            )
                        
                        if formatted_rows:
                            st.markdown("".join(formatted_rows), unsafe_allow_html=True)
                        else:
                            st.caption("No deployable stops identified.")
                        
                        st.caption(f"Total newly covered population: {crew_total:,.0f}")
        else:
            st.markdown("---")
            st.info("ℹ️ No recommendations available. This may indicate excellent existing coverage or insufficient population data.")

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.exception(e)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div class="footer">
          <strong>Coverage insights:</strong> Data-driven view of population coverage, facility access, and recommended mobile clinic deployments.
        </div>
        """,
        unsafe_allow_html=True,
    )


# # Run the page
district_analysis_page()

