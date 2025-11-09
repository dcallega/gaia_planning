import streamlit as st
import pandas as pd
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
    """
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
    """,
    unsafe_allow_html=True,
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
    df = pd.read_csv("data/GAIA MHC Clinic Stops GPS.xlsx - Clinic stops GPS.csv")

    # Parse GPS coordinates
    df[["latitude", "longitude"]] = df["collect_gps_coordinates"].apply(
        lambda x: pd.Series(parse_gps_coordinates(x))
    )

    # Remove rows with invalid coordinates
    df = df.dropna(subset=["latitude", "longitude"])

    return df


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

    return df


# Cache population data loading
@st.cache_data
def load_population_data(dataset_name):
    """Load population density data"""
    file_path = f"data/mwi_{dataset_name}_2020.csv"
    df = pd.read_csv(file_path)

    # Filter out zero or very low population values
    pop_column = f"mwi_{dataset_name}_2020"
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
    facility_coords = facilities_df[["latitude", "longitude"]].values
    tree = cKDTree(facility_coords)

    # Query for each population point
    pop_coords = population_df[["latitude", "longitude"]].values
    distances, _ = tree.query(pop_coords)

    # Mark points within radius
    covered = distances <= radius_degrees

    # Calculate coverage
    total_pop = population_df[pop_column].sum()
    covered_pop = population_df.loc[covered, pop_column].sum()
    coverage_pct = (covered_pop / total_pop * 100) if total_pop > 0 else 0

    # Add coverage info to dataframe
    result_df = population_df.copy()
    result_df["covered"] = covered
    result_df["distance_km"] = distances * 111.0  # Convert back to km

    return covered_pop, coverage_pct, result_df


# Helper function to calculate Gini coefficient
def calculate_gini(distances, weights):
    """Calculate Gini coefficient for access distribution"""
    if len(distances) == 0:
        return 0.0

    # Sort by distance
    sorted_indices = np.argsort(distances)
    sorted_distances = distances[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Calculate cumulative distribution
    cum_weights = np.cumsum(sorted_weights)
    total_weight = cum_weights[-1]

    if total_weight == 0:
        return 0.0

    # Normalize
    cum_weights_norm = cum_weights / total_weight
    cum_distances_norm = np.cumsum(sorted_distances * sorted_weights) / np.sum(
        sorted_distances * sorted_weights
    )

    # Calculate Gini using trapezoidal rule
    n = len(sorted_distances)
    gini = 0.0
    for i in range(1, n):
        gini += (cum_weights_norm[i] - cum_weights_norm[i - 1]) * (
            cum_distances_norm[i] + cum_distances_norm[i - 1]
        )

    return 1 - 2 * gini


# Function to identify healthcare desert zones
def identify_desert_zones(population_df, facilities_df, threshold_km, pop_column):
    """Identify areas with no access within threshold_km"""
    if len(facilities_df) == 0:
        # All areas are deserts if no facilities
        desert_pop = population_df[pop_column].sum()
        return len(population_df), desert_pop, population_df.copy()

    # Calculate distances
    radius_degrees = threshold_km / 111.0
    facility_coords = facilities_df[["latitude", "longitude"]].values
    tree = cKDTree(facility_coords)
    pop_coords = population_df[["latitude", "longitude"]].values
    distances, _ = tree.query(pop_coords)
    distances_km = distances * 111.0

    # Identify deserts (areas beyond threshold)
    is_desert = distances_km > threshold_km
    desert_df = population_df.copy()
    desert_df["is_desert"] = is_desert
    desert_df["distance_km"] = distances_km

    num_desert_areas = is_desert.sum()
    desert_pop = population_df.loc[is_desert, pop_column].sum()

    return num_desert_areas, desert_pop, desert_df


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
        help="Distance from a facility to be considered 'covered'",
    )

    # Population dataset selector
    population_datasets = {
        "General Population": "general",
        "Women": "women",
        "Men": "men",
        "Children (Under 5)": "children_under_five",
        "Youth (15-24)": "youth_15_24",
        "Elderly (60+)": "elderly_60_plus",
        "Women of Reproductive Age (15-49)": "women_of_reproductive_age_15_49",
    }

    selected_dataset = st.sidebar.selectbox(
        "Population Dataset", options=list(population_datasets.keys()), index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏥 Facility Filters")

    # Include GAIA clinics
    include_gaia = st.sidebar.checkbox("Include GAIA Mobile Clinics", value=True)

    # MHFR Facility Type filter
    st.sidebar.markdown("**MHFR Facility Types:**")
    facility_types = sorted(mhfr_df["type"].unique().tolist())

    # Default to hospitals and health centres
    default_types = [
        t for t in facility_types if "Hospital" in t or "Health Centre" in t
    ]
    if not default_types:
        default_types = (
            facility_types[:3] if len(facility_types) >= 3 else facility_types
        )

    selected_types = st.sidebar.multiselect(
        "Select Facility Types", options=facility_types, default=default_types
    )

    # Status filter
    st.sidebar.markdown("**Facility Status:**")
    status_options = sorted(mhfr_df["status"].unique().tolist())
    selected_status = st.sidebar.multiselect(
        "Select Status", options=status_options, default=["Functional"]
    )

    # Ownership filter (main feature for comparing Government vs CHAM)
    st.sidebar.markdown("**Facility Ownership:**")
    ownership_options = sorted(mhfr_df["ownership"].unique().tolist())
    selected_ownership = st.sidebar.multiselect(
        "Select Ownership",
        options=ownership_options,
        default=ownership_options,
        help="Filter by Government, CHAM, Private, etc.",
    )

    # Apply filters to MHFR facilities
    filtered_mhfr = mhfr_df.copy()

    if len(selected_types) > 0:
        filtered_mhfr = filtered_mhfr[filtered_mhfr["type"].isin(selected_types)]

    if len(selected_status) > 0:
        filtered_mhfr = filtered_mhfr[filtered_mhfr["status"].isin(selected_status)]

    if len(selected_ownership) > 0:
        filtered_mhfr = filtered_mhfr[
            filtered_mhfr["ownership"].isin(selected_ownership)
        ]

    # Combine facilities
    all_facilities = filtered_mhfr[
        ["latitude", "longitude", "common_name", "type", "ownership"]
    ].copy()

    if include_gaia:
        gaia_facilities = clinic_df[["latitude", "longitude", "clinic_name"]].copy()
        gaia_facilities["common_name"] = gaia_facilities["clinic_name"]
        gaia_facilities["type"] = "GAIA Mobile Clinic"
        gaia_facilities["ownership"] = "GAIA"
        gaia_facilities = gaia_facilities[
            ["latitude", "longitude", "common_name", "type", "ownership"]
        ]
        all_facilities = pd.concat([all_facilities, gaia_facilities], ignore_index=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Total Facilities Selected:** {len(all_facilities)}")

    # Load population data
    with st.spinner(f"Loading {selected_dataset} data..."):
        population_df = load_population_data(population_datasets[selected_dataset])
        pop_column = f"mwi_{population_datasets[selected_dataset]}_2020"

    # Calculate overall coverage
    with st.spinner("Calculating coverage..."):
        covered_pop, coverage_pct, pop_with_coverage = calculate_coverage(
            population_df, all_facilities, 5, pop_column  # Use 5km for overall coverage
        )

        # Calculate coverage at selected radius for other metrics
        _, _, pop_with_coverage_selected = calculate_coverage(
            population_df, all_facilities, coverage_radius, pop_column
        )

    # ============================================================================
    # SECTION 1: COVERAGE & ACCESS
    # ============================================================================
    st.markdown("---")
    st.markdown("## 📍 Coverage & Access")

    # Overall Coverage: % population within 5km
    st.markdown("### Overall Coverage")
    st.markdown("**Percentage of population within 5km of care**")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Population", f"{population_df[pop_column].sum():,.0f}")

    with col2:
        st.metric("Covered Population (5km)", f"{covered_pop:,.0f}")

    with col3:
        st.metric("Coverage Percentage (5km)", f"{coverage_pct:.1f}%")

    with col4:
        st.metric("Total Facilities", f"{len(all_facilities)}")

    # Average Distance to Care
    st.markdown("---")
    st.markdown("### Average Distance to Care")

    distances = pop_with_coverage_selected["distance_km"].values
    weights = pop_with_coverage_selected[pop_column].values

    mean_distance = np.average(distances, weights=weights)
    median_distance = np.median(distances)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Distance", f"{mean_distance:.2f} km")
    with col2:
        st.metric("Median Distance", f"{median_distance:.2f} km")

    # Distance distribution histogram
    st.markdown("**Distance Distribution**")
    distance_bins = [0, 5, 10, 15, 20, 30, 50, 100, float("inf")]
    distance_labels = [
        "0-5",
        "5-10",
        "10-15",
        "15-20",
        "20-30",
        "30-50",
        "50-100",
        "100+",
    ]
    pop_with_coverage_selected["distance_bin"] = pd.cut(
        pop_with_coverage_selected["distance_km"],
        bins=distance_bins,
        labels=distance_labels,
    )
    distance_dist = pop_with_coverage_selected.groupby("distance_bin")[pop_column].sum()
    st.bar_chart(distance_dist)

    # Healthcare Desert Zones
    st.markdown("---")
    st.markdown("### Healthcare Desert Zones")
    st.markdown("**Areas with no access within specified threshold**")

    desert_threshold = st.slider(
        "Desert Threshold (km)",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help="Areas beyond this distance are considered healthcare deserts",
    )

    num_deserts, desert_pop, desert_df = identify_desert_zones(
        population_df, all_facilities, desert_threshold, pop_column
    )
    desert_pct = (
        (desert_pop / population_df[pop_column].sum() * 100)
        if population_df[pop_column].sum() > 0
        else 0
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Desert Areas", f"{num_deserts:,}")
    with col2:
        st.metric("Population in Deserts", f"{desert_pop:,.0f}")
    with col3:
        st.metric("Desert Population %", f"{desert_pct:.1f}%")

    # Top 10 largest desert zones by population
    if num_deserts > 0:
        st.markdown("**Largest Desert Zones (by population)**")
        top_deserts = (
            desert_df[desert_df["is_desert"]]
            .nlargest(10, pop_column)[[pop_column, "distance_km"]]
            .copy()
        )
        top_deserts.columns = ["Population", "Distance (km)"]
        st.dataframe(top_deserts, use_container_width=True, hide_index=True)

    # Trend Over Time (placeholder - would need historical data)
    st.markdown("---")
    st.markdown("### Trend Over Time")
    st.info(
        "📊 Trend analysis requires historical data. This feature will show coverage improvements as new sites are added over time."
    )

    # Simulated trend (if we had multiple time points)
    st.markdown("**Coverage Improvement Simulation**")
    num_new_sites = st.slider(
        "Number of New Sites to Add",
        min_value=0,
        max_value=50,
        value=10,
        step=5,
        help="Simulate adding new sites to see projected coverage improvement",
    )

    if num_new_sites > 0 and len(all_facilities) > 0:
        # Simple simulation: add random sites in high-population areas
        high_pop_areas = population_df.nlargest(num_new_sites * 10, pop_column)
        simulated_sites = high_pop_areas.sample(
            n=min(num_new_sites, len(high_pop_areas)), random_state=42
        )
        simulated_facilities = pd.DataFrame(
            {
                "latitude": simulated_sites["latitude"].values,
                "longitude": simulated_sites["longitude"].values,
                "common_name": [
                    f"Simulated Site {i+1}" for i in range(len(simulated_sites))
                ],
                "type": "Simulated",
                "ownership": "Simulated",
            }
        )

        combined_facilities = pd.concat(
            [all_facilities, simulated_facilities], ignore_index=True
        )
        sim_cov_pop, sim_cov_pct, _ = calculate_coverage(
            population_df, combined_facilities, 5, pop_column
        )

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Coverage (5km)", f"{coverage_pct:.1f}%")
        with col2:
            st.metric("Projected Coverage (5km)", f"{sim_cov_pct:.1f}%")

        improvement = sim_cov_pct - coverage_pct
        st.metric("Projected Improvement", f"+{improvement:.1f} percentage points")

    # ============================================================================
    # SECTION 2: EQUITY METRICS
    # ============================================================================
    st.markdown("---")
    st.markdown("## ⚖️ Equity Metrics")

    # Coverage by Demographics
    st.markdown("### Coverage by Demographics")
    st.markdown("**Comparing coverage across different population groups**")

    demographic_groups = {
        "Women of Reproductive Age (15-49)": "women_of_reproductive_age_15_49",
        "Children (Under 5)": "children_under_five",
        "Elderly (60+)": "elderly_60_plus",
        "General Population": "general",
    }

    demographic_coverage = []

    for group_name, dataset_key in demographic_groups.items():
        try:
            demo_df = load_population_data(dataset_key)
            demo_pop_col = f"mwi_{dataset_key}_2020"
            demo_cov_pop, demo_cov_pct, _ = calculate_coverage(
                demo_df, all_facilities, 5, demo_pop_col
            )
            demographic_coverage.append(
                {
                    "Demographic Group": group_name,
                    "Total Population": f"{demo_df[demo_pop_col].sum():,.0f}",
                    "Covered Population": f"{demo_cov_pop:,.0f}",
                    "Coverage %": f"{demo_cov_pct:.1f}%",
                }
            )
        except Exception as e:
            st.warning(f"Could not load data for {group_name}: {str(e)}")

    if demographic_coverage:
        demo_df_display = pd.DataFrame(demographic_coverage)
        st.dataframe(demo_df_display, use_container_width=True, hide_index=True)

        # Coverage comparison chart
        coverage_pcts = [
            float(row["Coverage %"].replace("%", "")) for row in demographic_coverage
        ]
        group_names = [row["Demographic Group"] for row in demographic_coverage]
        coverage_chart_df = pd.DataFrame(
            {"Demographic Group": group_names, "Coverage %": coverage_pcts}
        )
        st.bar_chart(coverage_chart_df.set_index("Demographic Group"))

    # Disparity Indices: Gini coefficient
    st.markdown("---")
    st.markdown("### Disparity Indices")
    st.markdown(
        "**Gini coefficient of access distribution (0 = perfect equality, 1 = perfect inequality)**"
    )

    distances = pop_with_coverage_selected["distance_km"].values
    weights = pop_with_coverage_selected[pop_column].values

    gini_coefficient = calculate_gini(distances, weights)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Gini Coefficient", f"{gini_coefficient:.3f}")
    with col2:
        # Interpretation
        if gini_coefficient < 0.3:
            interpretation = "Low inequality"
            color = "green"
        elif gini_coefficient < 0.5:
            interpretation = "Moderate inequality"
            color = "orange"
        else:
            interpretation = "High inequality"
            color = "red"
        st.markdown(
            f"**Interpretation:** <span style='color:{color}'>{interpretation}</span>",
            unsafe_allow_html=True,
        )

    # Vulnerability Heat Map (table format since we removed maps)
    st.markdown("---")
    st.markdown("### Vulnerability Analysis")
    st.markdown("**Areas with high-risk populations and low access**")

    # Identify vulnerable areas: high population density but far from facilities
    vulnerability_df = pop_with_coverage_selected.copy()
    vulnerability_df["population_density"] = vulnerability_df[pop_column]
    vulnerability_df["vulnerability_score"] = (
        vulnerability_df["population_density"]
        / vulnerability_df["population_density"].max()
        * (vulnerability_df["distance_km"] / vulnerability_df["distance_km"].max())
    )

    # Top vulnerable areas
    top_vulnerable = vulnerability_df.nlargest(20, "vulnerability_score")[
        [pop_column, "distance_km", "vulnerability_score"]
    ].copy()
    top_vulnerable.columns = [
        "Population",
        "Distance (km)",
        "Vulnerability Score",
    ]
    top_vulnerable["Vulnerability Score"] = top_vulnerable["Vulnerability Score"].round(
        4
    )

    st.markdown("**Top 20 Most Vulnerable Areas**")
    st.dataframe(top_vulnerable, use_container_width=True, hide_index=True)

    # Vulnerability distribution
    st.markdown("**Vulnerability Score Distribution**")
    vuln_bins = pd.qcut(
        vulnerability_df["vulnerability_score"],
        q=5,
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
    )
    vuln_dist = vulnerability_df.groupby(vuln_bins)[pop_column].sum()
    st.bar_chart(vuln_dist)

    # ============================================================================
    # SECTION 3: CLINIC PERFORMANCE (GAIA Only)
    # ============================================================================
    st.markdown("---")
    st.markdown("## 🏥 Clinic Performance")
    st.markdown("**Analysis of GAIA clinic sites only**")

    # Filter to GAIA clinics only
    gaia_only = all_facilities[all_facilities["ownership"] == "GAIA"].copy()

    if len(gaia_only) > 0:
        st.markdown(f"**Total GAIA Clinics:** {len(gaia_only)}")

        # Estimated vs. Actual Impact
        st.markdown("### Estimated vs. Actual Impact")
        st.info(
            "📊 Actual impact data (patients served, utilization) would come from clinic records. This section shows estimated impact based on population coverage."
        )

        # Calculate estimated impact for each GAIA clinic
        clinic_impact = []
        for idx, clinic in gaia_only.iterrows():
            # Create a single-facility dataframe
            single_clinic = pd.DataFrame([clinic])
            est_cov_pop, est_cov_pct, _ = calculate_coverage(
                population_df, single_clinic, 5, pop_column
            )
            clinic_impact.append(
                {
                    "Clinic Name": clinic["common_name"],
                    "Estimated Population Reached": f"{est_cov_pop:,.0f}",
                    "Estimated Coverage %": f"{est_cov_pct:.2f}%",
                }
            )

        impact_df = pd.DataFrame(clinic_impact)
        st.dataframe(impact_df, use_container_width=True, hide_index=True)

        # Scatter plot: Estimated impact
        st.markdown("**Estimated Impact Distribution**")
        est_impact_values = [
            float(row["Estimated Population Reached"].replace(",", ""))
            for row in clinic_impact
        ]
        clinic_names_short = [
            name[:30] + "..." if len(name) > 30 else name
            for name in [row["Clinic Name"] for row in clinic_impact]
        ]
        impact_chart_df = pd.DataFrame(
            {
                "Clinic": clinic_names_short,
                "Estimated Population Reached": est_impact_values,
            }
        )
        st.bar_chart(impact_chart_df.set_index("Clinic"))

        # Utilization Rate (placeholder)
        st.markdown("---")
        st.markdown("### Utilization Rate")
        st.info(
            "📊 Utilization rate requires actual patient visit data. This would show: Patients Served vs. Capacity"
        )

        # Service Mix (placeholder)
        st.markdown("---")
        st.markdown("### Service Mix")
        st.info(
            "📊 Service mix breakdown requires clinic visit records showing types of care delivered (e.g., primary care, vaccinations, maternal health, etc.)"
        )

        # Top Performing Sites
        st.markdown("---")
        st.markdown("### Top Performing Sites")
        st.markdown("**Ranked by estimated impact per resource unit**")

        # Create a sortable version for ranking
        impact_df_sorted = impact_df.copy()
        impact_df_sorted["Est Pop Num"] = (
            impact_df_sorted["Estimated Population Reached"]
            .str.replace(",", "")
            .astype(float)
        )
        top_performers = impact_df_sorted.nlargest(10, "Est Pop Num")[
            [
                "Clinic Name",
                "Estimated Population Reached",
                "Estimated Coverage %",
            ]
        ]
        st.dataframe(top_performers, use_container_width=True, hide_index=True)

        # Underperforming Sites
        st.markdown("---")
        st.markdown("### Underperforming Sites")
        st.markdown(
            "**Sites flagged for investigation (low estimated impact relative to location)**"
        )

        # Flag sites with low impact despite being in high-population areas
        underperformers = []
        for idx, clinic in gaia_only.iterrows():
            # Check nearby population density
            nearby_pop = population_df[
                (abs(population_df["latitude"] - clinic["latitude"]) < 0.1)
                & (abs(population_df["longitude"] - clinic["longitude"]) < 0.1)
            ][pop_column].sum()

            single_clinic = pd.DataFrame([clinic])
            est_cov_pop, _, _ = calculate_coverage(
                population_df, single_clinic, 5, pop_column
            )

            # If nearby population is high but coverage is low, flag it
            if nearby_pop > 1000 and est_cov_pop < 500:
                underperformers.append(
                    {
                        "Clinic Name": clinic["common_name"],
                        "Nearby Population": f"{nearby_pop:,.0f}",
                        "Estimated Coverage": f"{est_cov_pop:,.0f}",
                        "Flag Reason": "High nearby population, low coverage",
                    }
                )

        if underperformers:
            underperformers_df = pd.DataFrame(underperformers)
            st.dataframe(underperformers_df, use_container_width=True, hide_index=True)
        else:
            st.success("✅ No underperforming sites flagged at this time.")
    else:
        st.warning(
            "⚠️ No GAIA clinics found in the selected facilities. Enable 'Include GAIA Mobile Clinics' in the sidebar."
        )

    # ============================================================================
    # SECTION 4: RECOMMENDATION IMPACT PROJECTIONS
    # ============================================================================
    st.markdown("---")
    st.markdown("## 🎯 Recommendation Impact Projections")
    st.markdown("**Projected improvements if recommendations are implemented**")

    # Projection controls
    st.markdown("### Projection Parameters")
    col1, col2 = st.columns(2)

    with col1:
        num_recommended_sites = st.number_input(
            "Number of Recommended Sites",
            min_value=0,
            max_value=100,
            value=10,
            step=1,
            help="Number of new sites to add in projections",
        )

    with col2:
        recommended_radius = st.slider(
            "Recommended Site Coverage Radius (km)",
            min_value=5,
            max_value=30,
            value=10,
            step=5,
            help="Coverage radius for recommended sites",
        )

    if num_recommended_sites > 0:
        # Simulate adding recommended sites
        st.markdown("---")
        st.markdown("### If Implemented: Projected Coverage Improvements")

        # Find optimal locations (high population, currently underserved)
        high_vulnerability = vulnerability_df.nlargest(
            num_recommended_sites * 5, "vulnerability_score"
        )
        recommended_locations = high_vulnerability.sample(
            n=min(num_recommended_sites, len(high_vulnerability)), random_state=42
        )

        recommended_facilities = pd.DataFrame(
            {
                "latitude": recommended_locations["latitude"].values,
                "longitude": recommended_locations["longitude"].values,
                "common_name": [
                    f"Recommended Site {i+1}" for i in range(len(recommended_locations))
                ],
                "type": "Recommended",
                "ownership": "Recommended",
            }
        )

        # Calculate projected coverage
        projected_facilities = pd.concat(
            [all_facilities, recommended_facilities], ignore_index=True
        )
        proj_cov_pop, proj_cov_pct, _ = calculate_coverage(
            population_df, projected_facilities, 5, pop_column
        )

        # Current vs Projected
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Coverage (5km)", f"{coverage_pct:.1f}%")
        with col2:
            st.metric("Projected Coverage (5km)", f"{proj_cov_pct:.1f}%")
        with col3:
            improvement = proj_cov_pct - coverage_pct
            st.metric(
                "Improvement", f"+{improvement:.1f} pp", delta=f"+{improvement:.1f}%"
            )

        # Population Reached
        st.markdown("---")
        st.markdown("### Population Reached")
        additional_pop = proj_cov_pop - covered_pop
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Additional People Served", f"{additional_pop:,.0f}")
        with col2:
            st.metric("Total Population Reached", f"{proj_cov_pop:,.0f}")

        # Equity Gains
        st.markdown("---")
        st.markdown("### Equity Gains")
        st.markdown("**Change in disparity metrics**")

        # Calculate projected Gini
        proj_distances = pop_with_coverage_selected["distance_km"].values.copy()
        # Update distances for areas near recommended sites
        for idx, rec_site in recommended_facilities.iterrows():
            rec_coords = np.array([[rec_site["latitude"], rec_site["longitude"]]])
            pop_coords = population_df[["latitude", "longitude"]].values
            rec_tree = cKDTree(rec_coords)
            rec_distances, _ = rec_tree.query(pop_coords)
            rec_distances_km = rec_distances * 111.0

            # Update if recommended site is closer
            proj_distances = np.minimum(proj_distances, rec_distances_km)

        proj_gini = calculate_gini(proj_distances, weights)
        gini_change = gini_coefficient - proj_gini

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Gini Coefficient", f"{gini_coefficient:.3f}")
        with col2:
            st.metric("Projected Gini Coefficient", f"{proj_gini:.3f}")
        with col3:
            st.metric(
                "Equity Improvement",
                f"-{gini_change:.3f}",
                delta=f"Improved" if gini_change > 0 else "No change",
            )

        # Cost-Benefit Analysis
        st.markdown("---")
        st.markdown("### Cost-Benefit Analysis")
        st.info(
            "📊 Cost data would be needed for detailed cost-benefit analysis. Showing estimated patients served per site."
        )

        avg_pop_per_site = (
            additional_pop / num_recommended_sites if num_recommended_sites > 0 else 0
        )
        st.metric("Average Additional Population per Site", f"{avg_pop_per_site:,.0f}")

        # Summary table
        st.markdown("**Projection Summary**")
        projection_summary = pd.DataFrame(
            {
                "Metric": [
                    "Current Coverage (5km)",
                    "Projected Coverage (5km)",
                    "Coverage Improvement",
                    "Additional Population Reached",
                    "Current Gini Coefficient",
                    "Projected Gini Coefficient",
                    "Equity Improvement",
                    "Avg Population per New Site",
                ],
                "Value": [
                    f"{coverage_pct:.1f}%",
                    f"{proj_cov_pct:.1f}%",
                    f"+{improvement:.1f} pp",
                    f"{additional_pop:,.0f}",
                    f"{gini_coefficient:.3f}",
                    f"{proj_gini:.3f}",
                    f"-{gini_change:.3f}",
                    f"{avg_pop_per_site:,.0f}",
                ],
            }
        )
        st.dataframe(projection_summary, use_container_width=True, hide_index=True)

    # Export functionality
    st.markdown("---")
    st.markdown("### 💾 Export Data")

    # Prepare facilities data for export
    display_facilities = all_facilities.copy()
    display_facilities = display_facilities.sort_values(
        ["ownership", "type", "common_name"]
    )

    col1, col2 = st.columns(2)

    with col1:
        # Export facilities
        csv_facilities = display_facilities.to_csv(index=False)
        st.download_button(
            label="Download Facilities CSV",
            data=csv_facilities,
            file_name=f"facilities_coverage_{coverage_radius}km.csv",
            mime="text/csv",
        )

    with col2:
        # Export demographic coverage data if available
        if demographic_coverage:
            demo_export_df = pd.DataFrame(demographic_coverage)
            csv_demographic = demo_export_df.to_csv(index=False)
            st.download_button(
                label="Download Demographic Coverage CSV",
                data=csv_demographic,
                file_name=f"demographic_coverage_{coverage_radius}km.csv",
                mime="text/csv",
            )
        else:
            # Fallback: export coverage summary
            coverage_summary = pd.DataFrame(
                {
                    "Metric": [
                        "Total Population",
                        "Covered Population (5km)",
                        "Coverage % (5km)",
                        "Total Facilities",
                    ],
                    "Value": [
                        f"{population_df[pop_column].sum():,.0f}",
                        f"{covered_pop:,.0f}",
                        f"{coverage_pct:.1f}%",
                        f"{len(all_facilities)}",
                    ],
                }
            )
            csv_summary = coverage_summary.to_csv(index=False)
        st.download_button(
            label="Download Coverage Summary CSV",
            data=csv_summary,
            file_name=f"coverage_summary_{coverage_radius}km.csv",
            mime="text/csv",
        )

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.exception(e)

# Footer
st.markdown(
    """
    <div class="footer">
      <strong>Coverage Analysis:</strong><br/>
      • Coverage is calculated based on straight-line distance (as the crow flies)<br/>
      • Actual travel distance may be significantly different due to terrain and roads<br/>
      • Population data from Meta Data for Good High Resolution Population Density Maps
    </div>
    """,
    unsafe_allow_html=True,
)
