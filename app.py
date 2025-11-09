import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np

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
    """Load population density data with sampling for performance"""
    file_path = f"data/mwi_{dataset_name}_2020.csv"

    # Sample the data for performance (every 10th row for faster loading)
    df = pd.read_csv(file_path)

    # Sample to reduce points (adjust this for performance)
    sample_size = min(50000, len(df))
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    # Filter out zero or very low population values for cleaner visualization
    df = df[df[f"mwi_{dataset_name}_2020"] > 0.5]

    return df


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
    if "selected_dataset" not in st.session_state:
        st.session_state.selected_dataset = "General Population"

    # Use session state values for data loading
    show_clinics = st.session_state.show_clinics
    show_mhfr = st.session_state.show_mhfr
    show_population = st.session_state.show_population
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

        st.markdown("### :material/map: Map")
        # GAIA Clinic filters
        if show_clinics:
            clinic_names = ["All Clinics"] + sorted(
                clinic_df["clinic_name"].unique().tolist()
            )
            selected_clinic = st.selectbox(
                "Filter GAIA Clinics", clinic_names, key="clinic_filter"
            )

            if selected_clinic != "All Clinics":
                clinic_df = clinic_df[clinic_df["clinic_name"] == selected_clinic]

        # MHFR Facilities filters
        if show_mhfr:
            with st.expander("🏥 Health Facility Filters", expanded=False):
                col1, col2, col3 = st.columns(3)

                with col1:
                    status_options = sorted(mhfr_df["status"].unique().tolist())
                    selected_status = st.multiselect(
                        "Status", options=status_options, default=["Functional"]
                    )

                with col2:
                    ownership_options = sorted(mhfr_df["ownership"].unique().tolist())
                    selected_ownership = st.multiselect(
                        "Ownership",
                        options=ownership_options,
                        default=ownership_options,
                    )

                with col3:
                    type_options = sorted(mhfr_df["type"].unique().tolist())
                    selected_types = st.multiselect(
                        "Facility Type", options=type_options, default=type_options
                    )

            # Apply filters
            if len(selected_status) > 0:
                mhfr_df = mhfr_df[mhfr_df["status"].isin(selected_status)]

            if len(selected_ownership) > 0:
                mhfr_df = mhfr_df[mhfr_df["ownership"].isin(selected_ownership)]

            if len(selected_types) > 0:
                mhfr_df = mhfr_df[mhfr_df["type"].isin(selected_types)]

        # Load population data
        population_df = None
        if show_population:
            with st.spinner(f"Loading {selected_dataset} data..."):
                population_df = load_population_data(
                    population_datasets[selected_dataset]
                )
                pop_column = f"mwi_{population_datasets[selected_dataset]}_2020"
        # Map controls
        with st.expander("⚙️ Map Controls", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Data Layers**")
                show_clinics = st.checkbox(
                    "🚐 GAIA Mobile Clinics",
                    value=st.session_state.show_clinics,
                    key="controls_show_clinics",
                )
                show_mhfr = st.checkbox(
                    "🏥 Health Facilities",
                    value=st.session_state.show_mhfr,
                    key="controls_show_mhfr",
                )
                show_population = st.checkbox(
                    "📊 Population Density",
                    value=st.session_state.show_population,
                    key="controls_show_population",
                )

                # Update session state
                if show_clinics != st.session_state.show_clinics:
                    st.session_state.show_clinics = show_clinics
                    st.rerun()
                if show_mhfr != st.session_state.show_mhfr:
                    st.session_state.show_mhfr = show_mhfr
                    st.rerun()
                if show_population != st.session_state.show_population:
                    st.session_state.show_population = show_population
                    st.rerun()

            with col2:
                st.markdown("**Population Dataset**")
                selected_dataset = st.selectbox(
                    "Select Dataset",
                    options=list(population_datasets.keys()),
                    index=list(population_datasets.keys()).index(
                        st.session_state.selected_dataset
                    ),
                    key="controls_selected_dataset",
                )

                # Update session state
                if selected_dataset != st.session_state.selected_dataset:
                    st.session_state.selected_dataset = selected_dataset
                    st.rerun()
        # Interactive Facility Type Legend
        st.markdown("🎨 Facility Type Filter: Click to toggle facility types")

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
            ("Maternity", "🩷"),
            ("GAIA Mobile", "🔵"),
        ]

        # Initialize session state for facility visibility
        if "facility_visibility" not in st.session_state:
            st.session_state.facility_visibility = {
                item[0]: True for item in legend_items if item[0] != "GAIA Mobile"
            }
            st.session_state.facility_visibility["GAIA Mobile"] = True

        # Create clickable legend items
        legend_cols = st.columns(6)
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
                "Maternity": [255, 105, 180, 200],
                "Health Post": [100, 149, 237, 200],
                "Unclassified": [128, 128, 128, 180],
            }

            hospital_types = ["Hospital", "District Hospital", "Central Hospital"]
            if facility_visibility.get("Hospital", True):
                hospitals_df = mhfr_df[mhfr_df["type"].isin(hospital_types)].copy()
                hospitals_df = hospitals_df.dropna(subset=["latitude", "longitude"])
                hospitals_df = hospitals_df[
                    (hospitals_df["latitude"].notna())
                    & (hospitals_df["longitude"].notna())
                ]

                if len(hospitals_df) > 0:
                    layers.append(
                        pdk.Layer(
                            "ScatterplotLayer",
                            data=hospitals_df,
                            get_position=["longitude", "latitude"],
                            get_color=facility_colors.get(
                                "Hospital", [220, 20, 60, 220]
                            ),
                            get_radius=3000,
                            pickable=True,
                            opacity=0.9,
                            stroked=True,
                            filled=True,
                            line_width_min_pixels=3,
                            get_line_color=[139, 0, 0],
                            radius_min_pixels=8,
                            radius_max_pixels=80,
                        )
                    )

            other_facilities = mhfr_df[~mhfr_df["type"].isin(hospital_types)].copy()
            for facility_type in other_facilities["type"].unique():
                if not facility_visibility.get(facility_type, True):
                    continue

                type_df = other_facilities[
                    other_facilities["type"] == facility_type
                ].copy()
                type_df = type_df.dropna(subset=["latitude", "longitude"])
                type_df = type_df[
                    (type_df["latitude"].notna()) & (type_df["longitude"].notna())
                ]

                color = facility_colors.get(facility_type, [128, 128, 128, 180])

                if len(type_df) > 0:
                    layers.append(
                        pdk.Layer(
                            "ScatterplotLayer",
                            data=type_df,
                            get_position=["longitude", "latitude"],
                            get_color=color,
                            get_radius=2000,
                            pickable=True,
                            opacity=0.8,
                            stroked=True,
                            filled=True,
                            line_width_min_pixels=2,
                            get_line_color=[0, 0, 0],
                            radius_min_pixels=6,
                            radius_max_pixels=60,
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
        except Exception:
            pass

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
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        if show_clinics:
            col1.metric("GAIA Clinic Stops", len(clinic_df))
        if show_mhfr:
            col2.metric("Health Facilities", len(mhfr_df))
        if show_population and population_df is not None:
            col3.metric("Population Points", f"{len(population_df):,}")

        # Population statistics
        if show_population and population_df is not None:
            st.markdown("---")
            st.markdown("### 📈 Population Density Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Minimum", f"{population_df[pop_column].min():.2f}")
            with col2:
                st.metric("Average", f"{population_df[pop_column].mean():.2f}")
            with col3:
                st.metric("Maximum", f"{population_df[pop_column].max():.2f}")

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
