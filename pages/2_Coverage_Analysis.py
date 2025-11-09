import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
from scipy.spatial import cKDTree

st.set_page_config(page_title="Coverage Analysis", page_icon="🎯", layout="wide")

# Load brand CSS
try:
    with open("brand.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass  # CSS not found, continue with default styling

# Navigation and Hero
st.markdown(
    '''
    <div class="gaia-nav">
      <div class="inner">
        <div class="gaia-brand">
          <span class="gaia-bracket">[</span><span class="gaia-title">GAIA global health</span><span class="gaia-bracket">]</span>
        </div>
        <div style="font-size:.92rem;color:#475569;">Malawi</div>
      </div>
    </div>
    <div class="hero-lite">
      <h1>Healthcare Coverage Analysis</h1>
      <p>Analyzing population coverage by functional hospitals and GAIA clinics</p>
    </div>
    ''',
    unsafe_allow_html=True
)

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
    df = pd.read_csv('data/GAIA MHC Clinic Stops GPS.xlsx - Clinic stops GPS.csv')
    
    # Parse GPS coordinates
    df[['latitude', 'longitude']] = df['collect_gps_coordinates'].apply(
        lambda x: pd.Series(parse_gps_coordinates(x))
    )
    
    # Remove rows with invalid coordinates
    df = df.dropna(subset=['latitude', 'longitude'])
    
    return df

# Cache the MHFR facilities data
@st.cache_data
def load_mhfr_facilities():
    df = pd.read_csv('data/MHFR_Facilities.csv')
    
    # Rename columns to lowercase for consistency
    df = df.rename(columns={
        'LATITUDE': 'latitude', 
        'LONGITUDE': 'longitude',
        'COMMON NAME': 'common_name',
        'OWNERSHIP': 'ownership',
        'TYPE': 'type',
        'STATUS': 'status',
        'NAME': 'name',
        'DISTRICT': 'district'
    })
    
    # Convert coordinates to numeric, handling any errors
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    # Remove rows with invalid coordinates
    df = df.dropna(subset=['latitude', 'longitude'])
    
    # Filter out rows with empty or zero coordinates
    df = df[(df['latitude'] != 0) & (df['longitude'] != 0)]
    
    return df

# Cache population data loading
@st.cache_data
def load_population_data(dataset_name):
    """Load population density data"""
    file_path = f'data/mwi_{dataset_name}_2020.csv'
    df = pd.read_csv(file_path)
    
    # Filter out zero or very low population values
    pop_column = f'mwi_{dataset_name}_2020'
    df = df[df[pop_column] > 0.1]
    
    return df

# Function to calculate coverage
@st.cache_data
def calculate_coverage(population_df, facilities_df, radius_km, pop_column):
    """
    Calculate what percentage of population is within radius_km of any facility
    """
    if len(facilities_df) == 0:
        return 0, 0, population_df.copy()
    
    # Create KD-Tree for fast spatial queries
    # Convert km to degrees (rough approximation: 1 degree ≈ 111 km)
    radius_degrees = radius_km / 111.0
    
    # Build tree from facilities
    facility_coords = facilities_df[['latitude', 'longitude']].values
    tree = cKDTree(facility_coords)
    
    # Query for each population point
    pop_coords = population_df[['latitude', 'longitude']].values
    distances, _ = tree.query(pop_coords)
    
    # Mark points within radius
    covered = distances <= radius_degrees
    
    # Calculate coverage
    total_pop = population_df[pop_column].sum()
    covered_pop = population_df.loc[covered, pop_column].sum()
    coverage_pct = (covered_pop / total_pop * 100) if total_pop > 0 else 0
    
    # Add coverage info to dataframe
    result_df = population_df.copy()
    result_df['covered'] = covered
    result_df['distance_km'] = distances * 111.0  # Convert back to km
    
    return covered_pop, coverage_pct, result_df

# Load data
try:
    clinic_df = load_clinic_data()
    mhfr_df = load_mhfr_facilities()
    
    # Sidebar controls
    st.sidebar.markdown("### ⚙️ Coverage Settings")
    st.sidebar.markdown("---")
    
    # Coverage radius
    coverage_radius = st.sidebar.slider(
        "Coverage Radius (km)",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        help="Distance from a facility to be considered 'covered'"
    )
    
    # Population dataset selector
    population_datasets = {
        "General Population": "general",
        "Women": "women",
        "Men": "men",
        "Children (Under 5)": "children_under_five",
        "Youth (15-24)": "youth_15_24",
        "Elderly (60+)": "elderly_60_plus",
        "Women of Reproductive Age (15-49)": "women_of_reproductive_age_15_49"
    }
    
    selected_dataset = st.sidebar.selectbox(
        "Population Dataset",
        options=list(population_datasets.keys()),
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏥 Facility Filters")
    
    # Include GAIA clinics
    include_gaia = st.sidebar.checkbox("Include GAIA Mobile Clinics", value=True)
    
    # MHFR Facility Type filter
    st.sidebar.markdown("**MHFR Facility Types:**")
    facility_types = sorted(mhfr_df['type'].unique().tolist())
    
    # Default to hospitals and health centres
    default_types = [t for t in facility_types if 'Hospital' in t or 'Health Centre' in t]
    if not default_types:
        default_types = facility_types[:3] if len(facility_types) >= 3 else facility_types
    
    selected_types = st.sidebar.multiselect(
        "Select Facility Types",
        options=facility_types,
        default=default_types
    )
    
    # Status filter
    st.sidebar.markdown("**Facility Status:**")
    status_options = sorted(mhfr_df['status'].unique().tolist())
    selected_status = st.sidebar.multiselect(
        "Select Status",
        options=status_options,
        default=["Functional"]
    )
    
    # Ownership filter (main feature for comparing Government vs CHAM)
    st.sidebar.markdown("**Facility Ownership:**")
    ownership_options = sorted(mhfr_df['ownership'].unique().tolist())
    selected_ownership = st.sidebar.multiselect(
        "Select Ownership",
        options=ownership_options,
        default=ownership_options,
        help="Filter by Government, CHAM, Private, etc."
    )
    
    # Apply filters to MHFR facilities
    filtered_mhfr = mhfr_df.copy()
    
    if len(selected_types) > 0:
        filtered_mhfr = filtered_mhfr[filtered_mhfr['type'].isin(selected_types)]
    
    if len(selected_status) > 0:
        filtered_mhfr = filtered_mhfr[filtered_mhfr['status'].isin(selected_status)]
    
    if len(selected_ownership) > 0:
        filtered_mhfr = filtered_mhfr[filtered_mhfr['ownership'].isin(selected_ownership)]
    
    # Combine facilities
    all_facilities = filtered_mhfr[['latitude', 'longitude', 'common_name', 'type', 'ownership']].copy()
    
    if include_gaia:
        gaia_facilities = clinic_df[['latitude', 'longitude', 'clinic_name']].copy()
        gaia_facilities['common_name'] = gaia_facilities['clinic_name']
        gaia_facilities['type'] = 'GAIA Mobile Clinic'
        gaia_facilities['ownership'] = 'GAIA'
        gaia_facilities = gaia_facilities[['latitude', 'longitude', 'common_name', 'type', 'ownership']]
        all_facilities = pd.concat([all_facilities, gaia_facilities], ignore_index=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Total Facilities Selected:** {len(all_facilities)}")
    
    # Load population data
    with st.spinner(f"Loading {selected_dataset} data..."):
        population_df = load_population_data(population_datasets[selected_dataset])
        pop_column = f'mwi_{population_datasets[selected_dataset]}_2020'
    
    # Calculate overall coverage
    with st.spinner("Calculating coverage..."):
        covered_pop, coverage_pct, pop_with_coverage = calculate_coverage(
            population_df, all_facilities, coverage_radius, pop_column
        )
    
    # Display key metrics
    st.markdown("---")
    st.markdown("### 📊 Coverage Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Population",
            f"{population_df[pop_column].sum():,.0f}"
        )
    
    with col2:
        st.metric(
            "Covered Population",
            f"{covered_pop:,.0f}"
        )
    
    with col3:
        st.metric(
            "Coverage Percentage",
            f"{coverage_pct:.1f}%"
        )
    
    with col4:
        st.metric(
            "Facilities",
            f"{len(all_facilities)}"
        )
    
    # Comparison by ownership
    st.markdown("---")
    st.markdown("### 🏛️ Coverage Comparison by Ownership")
    
    ownership_comparison = []
    
    # Get unique ownerships from selected facilities
    unique_ownerships = all_facilities['ownership'].unique()
    
    for ownership in unique_ownerships:
        ownership_facilities = all_facilities[all_facilities['ownership'] == ownership]
        
        if len(ownership_facilities) > 0:
            cov_pop, cov_pct, _ = calculate_coverage(
                population_df, ownership_facilities, coverage_radius, pop_column
            )
            
            ownership_comparison.append({
                'Ownership': ownership,
                'Facilities': len(ownership_facilities),
                'Covered Population': f"{cov_pop:,.0f}",
                'Coverage %': f"{cov_pct:.1f}%"
            })
    
    comparison_df = pd.DataFrame(ownership_comparison)
    comparison_df = comparison_df.sort_values('Facilities', ascending=False)
    
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Visualization: Side-by-side comparison
    st.markdown("---")
    st.markdown("### 📈 Ownership Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Facilities by Ownership**")
        facility_counts = all_facilities['ownership'].value_counts()
        st.bar_chart(facility_counts)
    
    with col2:
        st.markdown("**Facility Types Distribution**")
        type_counts = all_facilities['type'].value_counts()
        st.bar_chart(type_counts)
    
    # Map visualization
    st.markdown("---")
    st.markdown("### 🗺️ Coverage Map")
    
    # Create map layers
    layers = []
    
    # Population layer - color by coverage
    pop_sample = pop_with_coverage.sample(n=min(30000, len(pop_with_coverage)), random_state=42)
    
    # Color based on coverage
    pop_sample['color'] = pop_sample['covered'].apply(
        lambda x: [50, 205, 50, 150] if x else [255, 50, 50, 100]  # Green if covered, red if not
    )
    
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=pop_sample,
            get_position=['longitude', 'latitude'],
            get_color='color',
            get_radius=200,
            pickable=True,
            opacity=0.4,
            stroked=False,
            filled=True,
            radius_min_pixels=1,
            radius_max_pixels=4,
        )
    )
    
    # Facilities layer - color by ownership
    ownership_colors = {
        'Government': [30, 144, 255, 220],  # Dodger blue
        'Christian Health Association of Malawi (CHAM)': [255, 140, 0, 220],  # Dark orange
        'Private': [147, 112, 219, 220],  # Medium purple
        'GAIA': [0, 255, 127, 220],  # Spring green
        'Mission/Faith-based (other than CHAM)': [255, 215, 0, 220],  # Gold
    }
    
    for ownership in all_facilities['ownership'].unique():
        ownership_fac = all_facilities[all_facilities['ownership'] == ownership]
        color = ownership_colors.get(ownership, [128, 128, 128, 220])
        
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=ownership_fac,
                get_position=['longitude', 'latitude'],
                get_color=color,
                get_radius=coverage_radius * 1000,  # Convert km to meters
                pickable=True,
                opacity=0.3,
                stroked=True,
                filled=True,
                line_width_min_pixels=2,
                get_line_color=[0, 0, 0, 180],
                radius_min_pixels=8,
                radius_max_pixels=100,
            )
        )
    
    # Calculate map center
    center_lat = all_facilities['latitude'].mean() if len(all_facilities) > 0 else -13.5
    center_lon = all_facilities['longitude'].mean() if len(all_facilities) > 0 else 34.0
    
    view_state = pdk.ViewState(
        latitude=float(center_lat),
        longitude=float(center_lon),
        zoom=7,
        pitch=0,
    )
    
    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={
            "html": """
                <b>{common_name}</b><br/>
                <b>Type:</b> {type}<br/>
                <b>Ownership:</b> {ownership}<br/>
                <b>Covered:</b> {covered}<br/>
                <b>Distance:</b> {distance_km:.1f} km
            """,
            "style": {"backgroundColor": "steelblue", "color": "white"}
        },
        map_provider="carto",
        map_style="road",
    )
    
    st.pydeck_chart(r)
    
    # Legend
    st.markdown("---")
    st.markdown("### 🎨 Map Legend")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Population Coverage:**")
        st.markdown("🟢 Green = Covered (within radius)")
        st.markdown("🔴 Red = Not covered")
    
    with col2:
        st.markdown("**Facility Ownership:**")
        st.markdown("🔵 Government")
        st.markdown("🟠 CHAM")
        st.markdown("🟣 Private")
        st.markdown("🟢 GAIA")
        st.markdown("🟡 Other Mission/Faith-based")
    
    # Detailed coverage statistics
    st.markdown("---")
    st.markdown("### 📉 Distance Distribution")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        within_5km = (pop_with_coverage['distance_km'] <= 5).sum()
        within_5km_pct = (within_5km / len(pop_with_coverage) * 100)
        st.metric("Within 5 km", f"{within_5km_pct:.1f}%")
    
    with col2:
        within_10km = (pop_with_coverage['distance_km'] <= 10).sum()
        within_10km_pct = (within_10km / len(pop_with_coverage) * 100)
        st.metric("Within 10 km", f"{within_10km_pct:.1f}%")
    
    with col3:
        within_20km = (pop_with_coverage['distance_km'] <= 20).sum()
        within_20km_pct = (within_20km / len(pop_with_coverage) * 100)
        st.metric("Within 20 km", f"{within_20km_pct:.1f}%")
    
    # Facility details table
    st.markdown("---")
    st.markdown("### 🏥 Facility Details")
    
    display_facilities = all_facilities.copy()
    display_facilities = display_facilities.sort_values(['ownership', 'type', 'common_name'])
    
    st.dataframe(
        display_facilities,
        use_container_width=True,
        hide_index=True
    )
    
    # Export functionality
    st.markdown("---")
    st.markdown("### 💾 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export facilities
        csv_facilities = display_facilities.to_csv(index=False)
        st.download_button(
            label="Download Facilities CSV",
            data=csv_facilities,
            file_name=f"facilities_coverage_{coverage_radius}km.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export coverage data
        csv_comparison = comparison_df.to_csv(index=False)
        st.download_button(
            label="Download Coverage Comparison CSV",
            data=csv_comparison,
            file_name=f"coverage_comparison_{coverage_radius}km.csv",
            mime="text/csv"
        )

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.exception(e)

# Footer
st.markdown(
    '''
    <div class="footer">
      <strong>Coverage Analysis:</strong><br/>
      • Coverage is calculated based on straight-line distance (as the crow flies)<br/>
      • Actual travel distance may be significantly different due to terrain and roads<br/>
      • Population data from Meta Data for Good High Resolution Population Density Maps
    </div>
    ''',
    unsafe_allow_html=True
)

