import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import json
from spatial_utils import assign_districts_to_dataframe, load_district_boundaries, load_country_boundary, filter_points_in_country
from sklearn.neighbors import BallTree

# Page config must be first
st.set_page_config(
    page_title="GAIA Planning", page_icon="assets/gaia_icon.png", layout="wide"
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
    
    Groups:
    - All hospitals (Hospital, District Hospital, Central Hospital) -> "Hospital"
    - Health Centre and Health Post -> "Health Centre"
    - Others remain as-is
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


# Cache population data loading with district assignments
@st.cache_data(persist="disk", show_spinner=False)
def load_population_data(dataset_name):
    """Load population density data with sampling for performance"""
    # Check if we have a pre-computed version with districts
    cached_file = f"data/.cache/mwi_{dataset_name}_2020_with_districts.parquet"
    
    try:
        if pd.io.common.file_exists(cached_file):
            print(f"Loading cached population data with districts from {cached_file}")
            return pd.read_parquet(cached_file)
    except:
        pass
    
    # Otherwise load and process
    file_path = f"data/mwi_{dataset_name}_2020.csv"

    # Sample the data for performance (every 10th row for faster loading)
    df = pd.read_csv(file_path)

    # Sample to reduce points (adjust this for performance)
    sample_size = min(50000, len(df))
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    # Filter out zero or very low population values for cleaner visualization
    df = df[df[f"mwi_{dataset_name}_2020"] > 0.5]

    # Filter out population points outside the country boundary
    df = filter_points_in_country(df, lat_col='latitude', lon_col='longitude')
    
    # Assign districts (expensive but only done once)
    print(f"Assigning districts to {len(df)} population points...")
    df = assign_districts_to_dataframe(df, lat_col='latitude', lon_col='longitude')
    
    # Save for next time
    try:
        import os
        os.makedirs('data/.cache', exist_ok=True)
        df.to_parquet(cached_file, index=False)
        print(f"Saved cached population data to {cached_file}")
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")

    return df


@st.cache_data(persist="disk", show_spinner=False, max_entries=20)
def calculate_coverage_metrics(population_df, facilities_df, pop_column, service_radius_km=5.0):
    """
    Calculate coverage metrics for population data
    
    Args:
        population_df: DataFrame with population points
        facilities_df: DataFrame with facility locations
        pop_column: Name of population column
        service_radius_km: Service radius in kilometers
        
    Returns:
        Dictionary with coverage metrics
    """
    if population_df is None or len(population_df) == 0 or facilities_df is None or len(facilities_df) == 0:
        return None
    
    # Build spatial index for facilities
    EARTH_RADIUS_KM = 6371.0088
    coords_rad = np.deg2rad(facilities_df[["latitude", "longitude"]].to_numpy(dtype=np.float64))
    tree = BallTree(coords_rad, metric="haversine")
    
    # Get population points
    pop_coords_rad = np.deg2rad(population_df[["latitude", "longitude"]].to_numpy(dtype=np.float64))
    population = population_df[pop_column].to_numpy(dtype=float)
    
    # Find nearest facility distance for each population point
    dist_rad, _ = tree.query(pop_coords_rad, k=1)
    dist_km = dist_rad.ravel() * EARTH_RADIUS_KM
    
    # Calculate metrics
    total_population = population.sum()
    service_radius_rad = service_radius_km / EARTH_RADIUS_KM
    covered_mask = dist_rad.ravel() <= service_radius_rad
    covered_population = population[covered_mask].sum()
    
    # Calculate distance percentiles
    p50 = np.percentile(dist_km, 50)
    p75 = np.percentile(dist_km, 75)
    p95 = np.percentile(dist_km, 95)
    
    return {
        'total_population': total_population,
        'covered_population': covered_population,
        'coverage_pct': (covered_population / total_population * 100) if total_population > 0 else 0,
        'p50_distance_km': p50,
        'p75_distance_km': p75,
        'p95_distance_km': p95
    }


@st.cache_data(persist="disk", show_spinner=False, max_entries=20)
def calculate_district_coverage_metrics(population_df, facilities_df, pop_column, service_radius_km=5.0):
    """
    Calculate coverage metrics broken down by district
    
    Args:
        population_df: DataFrame with population points (must have 'assigned_district' column)
        facilities_df: DataFrame with facility locations
        pop_column: Name of population column
        service_radius_km: Service radius in kilometers
        
    Returns:
        DataFrame with coverage metrics by district
    """
    if population_df is None or len(population_df) == 0 or facilities_df is None or len(facilities_df) == 0:
        return None
    
    # Check if districts are already assigned
    if 'assigned_district' not in population_df.columns:
        return None  # Districts should be pre-assigned in load_population_data
    
    EARTH_RADIUS_KM = 6371.0088
    
    districts = population_df['assigned_district'].dropna().unique()
    
    # Build spatial index once for all facilities
    coords_rad = np.deg2rad(facilities_df[["latitude", "longitude"]].to_numpy(dtype=np.float64))
    tree = BallTree(coords_rad, metric="haversine")
    
    district_metrics = []
    
    for district in districts:
        # Filter population for this district
        district_pop = population_df[population_df['assigned_district'] == district]
        
        if len(district_pop) == 0:
            continue
        
        # Calculate distances for this district's population
        pop_coords_rad = np.deg2rad(district_pop[["latitude", "longitude"]].to_numpy(dtype=np.float64))
        population = district_pop[pop_column].to_numpy(dtype=float)
        
        dist_rad, _ = tree.query(pop_coords_rad, k=1)
        dist_km = dist_rad.ravel() * EARTH_RADIUS_KM
        
        # Calculate metrics
        total_population = population.sum()
        service_radius_rad = service_radius_km / EARTH_RADIUS_KM
        covered_mask = dist_rad.ravel() <= service_radius_rad
        covered_population = population[covered_mask].sum()
        
        p50 = np.percentile(dist_km, 50)
        p75 = np.percentile(dist_km, 75)
        p95 = np.percentile(dist_km, 95)
        
        district_metrics.append({
            'District': district,
            'Total Population': int(total_population),
            'Covered Population': int(covered_population),
            'Coverage %': round(covered_population / total_population * 100, 1) if total_population > 0 else 0,
            'P50 Distance (km)': round(p50, 1),
            'P75 Distance (km)': round(p75, 1),
            'P95 Distance (km)': round(p95, 1)
        })
    
    if district_metrics:
        return pd.DataFrame(district_metrics).sort_values('District')
    else:
        return None


def render_navigation(pages):
    """Render navigation bar"""
    num_cols = len(pages)
    columns = st.columns(num_cols, vertical_alignment="bottom")
    for col, page in zip(columns, pages):
        col.page_link(page, icon=page.icon, label=page.title)


def map_page():
    """Main map visualization page"""
    # Hero section
    st.markdown(
        """
        <div class="hero-lite">
          <h1>GAIA Mobile Clinic Planning – Malawi</h1>
          <p>Visualizing clinic locations and population coverage</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Navigation bar after hero - pages will be defined later
    if "navigation_pages" in st.session_state:
        render_navigation(st.session_state.navigation_pages)

    # Initialize controls with session state defaults
    if "show_clinics" not in st.session_state:
        st.session_state.show_clinics = True
    if "show_mhfr" not in st.session_state:
        st.session_state.show_mhfr = True
    if "show_population" not in st.session_state:
        st.session_state.show_population = True
    if "show_boundaries" not in st.session_state:
        st.session_state.show_boundaries = True
    if "selected_dataset" not in st.session_state:
        st.session_state.selected_dataset = "General Population"

    # Use session state values for data loading
    show_clinics = st.session_state.show_clinics
    show_mhfr = st.session_state.show_mhfr
    show_population = st.session_state.show_population
    show_boundaries = st.session_state.show_boundaries
    selected_dataset = st.session_state.selected_dataset

    population_datasets = {
        "General Population": "general",
        "Women": "women",
        "Men": "men",
        "Children (Under 5)": "children_under_five",
        "Youth (15-24)": "youth_15_24",
        "Elderly (60+)": "elderly_60_plus",
        "Women of Reproductive Age (15-49)": "women_of_reproductive_age_15_49",
    }

    # Load data
    try:
        clinic_df = load_clinic_data()
        mhfr_df = load_mhfr_facilities()

        st.markdown("---")

        # st.markdown("### :material/map: Map")
        
        # Load population data
        population_df = None
        if show_population:
            with st.spinner(f"Loading {selected_dataset} data..."):
                population_df = load_population_data(
                    population_datasets[selected_dataset]
                )
                pop_column = f"mwi_{population_datasets[selected_dataset]}_2020"
        


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

        legend_items = [
            ("Hospital", "🔴"),
            ("Health Centre", "🟢"),
            ("Clinic", "🟠"),
            ("Dispensary", "🟣"),
            ("GAIA Mobile", "🔵"),
        ]

        # Initialize session state for facility visibility
        if "facility_visibility" not in st.session_state:
            st.session_state.facility_visibility = {
                item[0]: True for item in legend_items if item[0] != "GAIA Mobile"
            }
            st.session_state.facility_visibility["GAIA Mobile"] = True

        # Create clickable legend items
        legend_cols = st.columns(5)
        facility_visibility = {}

        for col, (facility_type, emoji) in zip(legend_cols, legend_items):
            with col:
                if facility_type == "GAIA Mobile":
                    if "gaia_mobile_visible" not in st.session_state:
                        st.session_state.gaia_mobile_visible = show_clinics
                    is_visible = st.session_state.gaia_mobile_visible
                else:
                    is_visible = st.session_state.facility_visibility.get(
                        facility_type, True
                    )

                button_text = f"{emoji} {facility_type}"
                button_type = "primary" if is_visible else "secondary"

                if st.button(
                    button_text,
                    key=f"btn_{facility_type}",
                    use_container_width=True,
                    type=button_type,
                ):
                    if facility_type == "GAIA Mobile":
                        st.session_state.gaia_mobile_visible = not is_visible
                    else:
                        st.session_state.facility_visibility[facility_type] = (
                            not is_visible
                        )
                    st.rerun()

                facility_visibility[facility_type] = is_visible

        # Create map layers
        layers = []

        # Population density layer
        if show_population and population_df is not None:
            pop_column = f"mwi_{population_datasets[selected_dataset]}_2020"

            pop_max = population_df[pop_column].max()
            pop_min = population_df[pop_column].min()

            population_df["color"] = population_df[pop_column].apply(
                lambda x: [
                    int(255 * (x - pop_min) / (pop_max - pop_min)),
                    int(100 * (1 - (x - pop_min) / (pop_max - pop_min))),
                    50,
                    150,
                ]
            )

            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=population_df,
                    get_position=["longitude", "latitude"],
                    get_color="color",
                    get_radius=100,
                    pickable=True,
                    opacity=0.3,
                    stroked=False,
                    filled=True,
                    radius_min_pixels=1,
                    radius_max_pixels=5,
                )
            )

        # MHFR Facilities layers
        if show_mhfr and len(mhfr_df) > 0:
            facility_colors = {
                "Hospital": [220, 20, 60, 220],
                "Health Centre": [50, 205, 50, 200],
                "Clinic": [255, 165, 0, 200],
                "Dispensary": [147, 112, 219, 200],
            }

            # Process each normalized facility type
            for facility_type in mhfr_df["type"].unique():
                if not facility_visibility.get(facility_type, True):
                    continue

                type_df = mhfr_df[mhfr_df["type"] == facility_type].copy()
                type_df = type_df.dropna(subset=["latitude", "longitude"])
                type_df = type_df[
                    (type_df["latitude"].notna()) & (type_df["longitude"].notna())
                ]

                # Use larger radius for hospitals
                radius = 3000 if facility_type == "Hospital" else 2000
                radius_min = 8 if facility_type == "Hospital" else 6
                radius_max = 80 if facility_type == "Hospital" else 60
                line_width = 3 if facility_type == "Hospital" else 2
                line_color = [139, 0, 0] if facility_type == "Hospital" else [0, 0, 0]

                color = facility_colors.get(facility_type, [128, 128, 128, 180])

                if len(type_df) > 0:
                    layers.append(
                        pdk.Layer(
                            "ScatterplotLayer",
                            data=type_df,
                            get_position=["longitude", "latitude"],
                            get_color=color,
                            get_radius=radius,
                            pickable=True,
                            opacity=0.9 if facility_type == "Hospital" else 0.8,
                            stroked=True,
                            filled=True,
                            line_width_min_pixels=line_width,
                            get_line_color=line_color,
                            radius_min_pixels=radius_min,
                            radius_max_pixels=radius_max,
                        )
                    )

        # GAIA clinics layer
        gaia_visible = show_clinics and st.session_state.get(
            "gaia_mobile_visible", True
        )
        if gaia_visible and len(clinic_df) > 0:
            clinic_clean = clinic_df.copy()
            clinic_clean = clinic_clean.dropna(subset=["latitude", "longitude"])
            clinic_clean = clinic_clean[
                (clinic_clean["latitude"].notna()) & (clinic_clean["longitude"].notna())
            ]

            if len(clinic_clean) > 0:
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=clinic_clean,
                        get_position=["longitude", "latitude"],
                        get_color=[0, 150, 255, 200],
                        get_radius=1000,
                        pickable=True,
                        opacity=0.8,
                        stroked=True,
                        filled=True,
                        line_width_min_pixels=2,
                        get_line_color=[0, 0, 0],
                    )
                )

        # District boundaries layer
        if show_boundaries:
            try:
                district_geojson = load_district_boundaries()
                
                layers.append(
                    pdk.Layer(
                        "GeoJsonLayer",
                        data=district_geojson,
                        opacity=0.2,
                        stroked=True,
                        filled=True,
                        extruded=False,
                        wireframe=True,
                        get_fill_color=[58, 90, 52, 50],  # GAIA green with transparency
                        get_line_color=[58, 90, 52, 255],  # GAIA green for borders
                        line_width_min_pixels=2,
                        pickable=True,
                    )
                )
            except FileNotFoundError:
                st.warning("District boundaries not found. Run download_boundaries.py to get them.")

        # Calculate map center
        center_lat = -13.5
        center_lon = 34.0

        try:
            all_lats = []
            all_lons = []

            if show_clinics and len(clinic_df) > 0:
                valid_lats = clinic_df["latitude"].dropna()
                valid_lons = clinic_df["longitude"].dropna()
                all_lats.extend(valid_lats.tolist())
                all_lons.extend(valid_lons.tolist())

            if show_mhfr and len(mhfr_df) > 0:
                valid_lats = mhfr_df["latitude"].dropna()
                valid_lons = mhfr_df["longitude"].dropna()
                all_lats.extend(valid_lats.tolist())
                all_lons.extend(valid_lons.tolist())

            if len(all_lats) > 0 and len(all_lons) > 0:
                calc_lat = float(np.mean(all_lats))
                calc_lon = float(np.mean(all_lons))
                if not np.isnan(calc_lat) and not np.isnan(calc_lon):
                    if -90 <= calc_lat <= 90 and -180 <= calc_lon <= 180:
                        center_lat = calc_lat
                        center_lon = calc_lon
        except Exception as e:
            # If anything goes wrong, use default center
            st.warning(f"Using default map center due to: {str(e)}")
            center_lat = -13.5
            center_lon = 34.0

        # Create the map
        view_state = pdk.ViewState(
            latitude=float(center_lat),
            longitude=float(center_lon),
            zoom=7,
            pitch=0,
        )

        if len(layers) > 0:
            r = pdk.Deck(
                layers=layers,
                initial_view_state=view_state,
                tooltip={
                    "html": """
                        <b>{common_name}</b><br/>
                        <b>Type:</b> {type}<br/>
                        <b>Ownership:</b> {ownership}<br/>
                        <b>Status:</b> {status}<br/>
                        <b>District:</b> {district}<br/>
                        <b>GAIA Clinic:</b> {clinic_name}<br/>
                        <b>Location:</b> {clinic_stop}
                    """,
                    "style": {"backgroundColor": "steelblue", "color": "white"},
                },
                map_provider="carto",
                map_style="road",
            )
        else:
            r = pdk.Deck(
                layers=[],
                initial_view_state=view_state,
                map_provider="carto",
                map_style="road",
            )

        st.pydeck_chart(r)
        
        # Calculate coverage metrics based on visible facilities
        visible_facilities = []
        
        # Collect visible MHFR facilities
        if show_mhfr and len(mhfr_df) > 0:
            for facility_type in mhfr_df["type"].unique():
                if facility_visibility.get(facility_type, True):
                    type_df = mhfr_df[mhfr_df["type"] == facility_type]
                    visible_facilities.append(type_df)
        
        # Add GAIA clinics if visible
        gaia_visible = show_clinics and st.session_state.get("gaia_mobile_visible", True)
        if gaia_visible and len(clinic_df) > 0:
            visible_facilities.append(clinic_df)
        
        # Combine all visible facilities
        if visible_facilities:
            combined_facilities = pd.concat(visible_facilities, ignore_index=True)
        else:
            combined_facilities = pd.DataFrame()
        
        # Calculate coverage metrics
        coverage_metrics = None
        if show_population and population_df is not None and len(combined_facilities) > 0:
            coverage_metrics = calculate_coverage_metrics(
                population_df, 
                combined_facilities, 
                pop_column,
                service_radius_km=5.0
            )
        
        # Display key metrics
        st.markdown("---")
        st.markdown("### 📊 Coverage Metrics")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        if coverage_metrics:
            col1.metric(
                "Total Population", 
                f"{coverage_metrics['total_population']:,.0f}",
                help=f"Total {selected_dataset.lower()} in sampled areas"
            )
            col2.metric(
                "Covered Population", 
                f"{coverage_metrics['covered_population']:,.0f}",
                help="Population within 5km of selected facilities"
            )
            col3.metric(
                "Coverage", 
                f"{coverage_metrics['coverage_pct']:.1f}%",
                help="Percentage of population within 5km of selected facilities"
            )
            col4.metric(
                "P50 Distance", 
                f"{coverage_metrics['p50_distance_km']:.1f} km",
                help="Median distance to nearest selected facility"
            )
            col5.metric(
                "P75 Distance", 
                f"{coverage_metrics['p75_distance_km']:.1f} km",
                help="75th percentile distance to nearest selected facility"
            )
            col6.metric(
                "P95 Distance", 
                f"{coverage_metrics['p95_distance_km']:.1f} km",
                help="95th percentile distance to nearest selected facility"
            )
        else:
            col1.metric("Total Population", "N/A")
            col2.metric("Covered Population", "N/A")
            col3.metric("Coverage", "N/A")
            col4.metric("P50 Distance", "N/A")
            col5.metric("P75 Distance", "N/A")
            col6.metric("P95 Distance", "N/A")
        
        # District-level breakdown
        if show_population and population_df is not None and len(combined_facilities) > 0:
            st.markdown("---")
            st.markdown("### 📍 Coverage by District")
            
            district_metrics = calculate_district_coverage_metrics(
                population_df,
                combined_facilities,
                pop_column,
                service_radius_km=5.0
            )
            
            if district_metrics is not None:
                st.dataframe(
                    district_metrics,
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
                
                # Add download button
                csv = district_metrics.to_csv(index=False)
                st.download_button(
                    label="📥 Download District Metrics as CSV",
                    data=csv,
                    file_name=f"district_coverage_{selected_dataset.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No district data available. Population points need to be assigned to districts.")

        # District assignment section
        if show_boundaries:
            st.markdown("---")
            st.markdown("### 🗺️ District Analysis")
            
            with st.expander("📊 Facility Distribution by District", expanded=False):
                # Show MHFR facilities by district
                if show_mhfr and len(mhfr_df) > 0:
                    st.markdown("#### Health Facilities by District")
                    
                    # Count facilities with and without district info
                    facilities_with_district = mhfr_df[mhfr_df['district'].notna()]
                    facilities_without_district = mhfr_df[mhfr_df['district'].isna()]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("With District Info", len(facilities_with_district))
                    with col2:
                        st.metric("Without District Info", len(facilities_without_district))
                    
                    # Show distribution by district
                    if len(facilities_with_district) > 0:
                        district_counts = facilities_with_district['district'].value_counts().reset_index()
                        district_counts.columns = ['District', 'Count']
                        st.dataframe(district_counts, use_container_width=True, height=300)
                    
                    # Option to assign districts to facilities without them
                    if len(facilities_without_district) > 0:
                        st.markdown("---")
                        st.markdown("#### Assign Districts to Facilities")
                        st.write(f"There are {len(facilities_without_district)} facilities without district information.")
                        
                        if st.button("🔍 Auto-assign Districts Using Spatial Join"):
                            with st.spinner("Assigning districts based on coordinates..."):
                                try:
                                    # Assign districts to facilities without them
                                    assigned_df = assign_districts_to_dataframe(
                                        facilities_without_district,
                                        lat_col='latitude',
                                        lon_col='longitude'
                                    )
                                    
                                    # Count successful assignments
                                    successful = assigned_df['assigned_district'].notna().sum()
                                    
                                    st.success(f"✅ Successfully assigned {successful} out of {len(facilities_without_district)} facilities to districts!")
                                    
                                    # Show sample of assignments
                                    st.markdown("**Sample assignments:**")
                                    sample = assigned_df[assigned_df['assigned_district'].notna()][
                                        ['common_name', 'latitude', 'longitude', 'assigned_district']
                                    ].head(10)
                                    st.dataframe(sample, use_container_width=True)
                                    
                                except Exception as e:
                                    st.error(f"Error assigning districts: {str(e)}")
                
                # Show GAIA clinics by district if they have district info
                if show_clinics and len(clinic_df) > 0:
                    st.markdown("---")
                    st.markdown("#### GAIA Mobile Clinics")
                    
                    if st.button("🔍 Assign Districts to GAIA Clinics"):
                        with st.spinner("Assigning districts to GAIA clinics..."):
                            try:
                                assigned_clinics = assign_districts_to_dataframe(
                                    clinic_df,
                                    lat_col='latitude',
                                    lon_col='longitude'
                                )
                                
                                successful = assigned_clinics['assigned_district'].notna().sum()
                                st.success(f"✅ Successfully assigned {successful} out of {len(clinic_df)} clinic stops to districts!")
                                
                                # Show distribution
                                district_counts = assigned_clinics['assigned_district'].value_counts().reset_index()
                                district_counts.columns = ['District', 'Clinic Stops']
                                st.dataframe(district_counts, use_container_width=True)
                                
                            except Exception as e:
                                st.error(f"Error assigning districts: {str(e)}")

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.exception(e)

    # Footer
    st.markdown(
        """
        <div class="footer">
          <strong>Data Sources:</strong><br/>
          • GAIA Mobile Health Clinic GPS locations<br/>
          • Malawi Health Facility Registry (MHFR) - Facility locations and classifications<br/>
          • High Resolution Population Density Maps for Malawi (Meta Data for Good)
        </div>
        """,
        unsafe_allow_html=True,
    )


# Define pages after map_page is defined
pages = [
    st.Page(map_page, icon=":material/map:", title="Map"),
    st.Page(
        "pages/2_Coverage_Analysis.py",
        icon=":material/analytics:",
        title="Coverage Analysis",
    ),
    st.Page("pages/visit_logs.py", icon=":material/description:", title="Visit Logs"),
]

# Store in session state for navigation function
st.session_state.navigation_pages = pages

# Set up navigation
current_page = st.navigation(pages=pages, position="hidden")

# Banner
st.markdown(
    """
    <div class="gaia-nav">
      <div class="inner">
        <div class="gaia-brand">
          <span class="gaia-bracket">[</span><span class="gaia-title">GAIA global health</span><span class="gaia-bracket">]</span>
        </div>
        <div style="font-size:.92rem;color:#475569;">Malawi</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Run the current page
current_page.run()
