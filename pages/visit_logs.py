import streamlit as st
import pandas as pd
import os
from datetime import datetime, date, time
import json
from app import render_navigation

st.set_page_config(page_title="Visit Logs", page_icon="📋", layout="wide")

# Load brand CSS
try:
    with open("brand.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass  # CSS not found, continue with default styling

# Navigation and Hero
st.markdown(
    """
    <div class="hero-lite">
      <h1>Visit Logs Management System</h1>
      <p>Capture ground truth data to validate models and improve recommendations</p>
    </div>
    """,
    unsafe_allow_html=True,
)
# Navigation bar after hero - pages will be defined later
if "navigation_pages" in st.session_state:
    render_navigation(st.session_state.navigation_pages)
# Data storage path
DATA_DIR = "data"
VISIT_LOGS_FILE = os.path.join(DATA_DIR, "visit_logs.csv")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize session state for form data
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False
if "edit_index" not in st.session_state:
    st.session_state.edit_index = None
if "patient_entries" not in st.session_state:
    st.session_state.patient_entries = []


# Load clinic data for dropdown
@st.cache_data
def load_clinic_stops():
    """Load clinic stops for dropdown selection"""
    try:
        df = pd.read_csv("data/GAIA MHC Clinic Stops GPS.xlsx - Clinic stops GPS.csv")
        # Create a combined identifier
        df["clinic_stop_id"] = df.apply(
            lambda row: f"{row['clinic_name']} - {row['clinic_stop']}", axis=1
        )
        return sorted(df["clinic_stop_id"].unique().tolist())
    except Exception as e:
        st.warning(f"Could not load clinic data: {str(e)}")
        return []


# Load existing visit logs
def load_visit_logs():
    """Load visit logs from CSV file"""
    if os.path.exists(VISIT_LOGS_FILE):
        try:
            df = pd.read_csv(VISIT_LOGS_FILE)
            # Convert date columns back to datetime
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.date
            return df
        except Exception as e:
            st.error(f"Error loading visit logs: {str(e)}")
            return pd.DataFrame()
    return pd.DataFrame()


# Save visit logs
def save_visit_logs(df):
    """Save visit logs to CSV file"""
    try:
        df.to_csv(VISIT_LOGS_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving visit logs: {str(e)}")
        return False


# Validate form data (for bulk upload compatibility)
def validate_visit_log(data):
    """Validate visit log data"""
    errors = []

    # Required fields
    if not data.get("clinic_stop_id"):
        errors.append("Clinic/MHC stop ID is required")
    if not data.get("date"):
        errors.append("Date is required")
    if (
        data.get("total_patients_seen") is None
        or data.get("total_patients_seen", 0) < 0
    ):
        errors.append("Total patients seen must be a non-negative number")
    if (
        data.get("wait_time_average") is not None
        and data.get("wait_time_average", 0) < 0
    ):
        errors.append("Wait time must be non-negative")

    # Validate time format
    if data.get("operating_hours_start") and data.get("operating_hours_end"):
        try:
            start = datetime.strptime(data["operating_hours_start"], "%H:%M").time()
            end = datetime.strptime(data["operating_hours_end"], "%H:%M").time()
            if start >= end:
                errors.append("Operating hours: start time must be before end time")
        except (ValueError, TypeError):
            errors.append("Operating hours must be in HH:MM format")

    return errors


# Calculate aggregated statistics from patient entries
def calculate_aggregates(patient_entries):
    """Calculate aggregated statistics from individual patient entries"""
    if not patient_entries:
        return {}

    df = pd.DataFrame(patient_entries)

    aggregates = {
        # Age brackets
        "age_0_5": len(df[df["age_bracket"] == "0-5"]),
        "age_6_14": len(df[df["age_bracket"] == "6-14"]),
        "age_15_24": len(df[df["age_bracket"] == "15-24"]),
        "age_25_49": len(df[df["age_bracket"] == "25-49"]),
        "age_50_59": len(df[df["age_bracket"] == "50-59"]),
        "age_60_plus": len(df[df["age_bracket"] == "60+"]),
        # Gender
        "gender_male": len(df[df["gender"] == "M"]),
        "gender_female": len(df[df["gender"] == "F"]),
        # Distance traveled
        "distance_lt_1km": len(df[df["distance"] == "<1km"]),
        "distance_1_3km": len(df[df["distance"] == "1-3km"]),
        "distance_3_5km": len(df[df["distance"] == "3-5km"]),
        "distance_5_10km": len(df[df["distance"] == "5-10km"]),
        "distance_gt_10km": len(df[df["distance"] == ">10km"]),
        # Patient type
        "first_time_patients": len(df[df["patient_type"] == "first_time"]),
        "returning_patients": len(df[df["patient_type"] == "returning"]),
        # Services (check if any patient received this service)
        "service_curative_care": (
            df["service_curative_care"].any()
            if "service_curative_care" in df.columns
            else False
        ),
        "service_chronic_disease": (
            df["service_chronic_disease"].any()
            if "service_chronic_disease" in df.columns
            else False
        ),
        "service_maternal_child": (
            df["service_maternal_child"].any()
            if "service_maternal_child" in df.columns
            else False
        ),
        "service_preventive_care": (
            df["service_preventive_care"].any()
            if "service_preventive_care" in df.columns
            else False
        ),
        "service_reproductive_health": (
            df["service_reproductive_health"].any()
            if "service_reproductive_health" in df.columns
            else False
        ),
        # Metrics
        "total_patients_seen": len(df),
        "referrals_made": df["referred"].sum() if "referred" in df.columns else 0,
        "patients_turned_away": (
            len(df[df["turned_away"] == True]) if "turned_away" in df.columns else 0
        ),
        "wait_time_average": (
            df["wait_time"].mean()
            if "wait_time" in df.columns and df["wait_time"].notna().any()
            else 0
        ),
    }

    return aggregates


# Create visit log entry
def create_visit_log_entry(clinic_info, patient_entries, qualitative_notes):
    """Create a visit log entry with aggregated statistics"""
    aggregates = calculate_aggregates(patient_entries)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "visit_id": f"{clinic_info.get('clinic_stop_id', '')}_{clinic_info.get('date', '')}_{datetime.now().strftime('%H%M%S')}",
        "clinic_stop_id": clinic_info.get("clinic_stop_id", ""),
        "date": clinic_info.get("date", ""),
        "operating_hours_start": clinic_info.get("operating_hours_start", ""),
        "operating_hours_end": clinic_info.get("operating_hours_end", ""),
        # Aggregated Patient Demographics
        "age_0_5": aggregates.get("age_0_5", 0),
        "age_6_14": aggregates.get("age_6_14", 0),
        "age_15_24": aggregates.get("age_15_24", 0),
        "age_25_49": aggregates.get("age_25_49", 0),
        "age_50_59": aggregates.get("age_50_59", 0),
        "age_60_plus": aggregates.get("age_60_plus", 0),
        "gender_male": aggregates.get("gender_male", 0),
        "gender_female": aggregates.get("gender_female", 0),
        "distance_lt_1km": aggregates.get("distance_lt_1km", 0),
        "distance_1_3km": aggregates.get("distance_1_3km", 0),
        "distance_3_5km": aggregates.get("distance_3_5km", 0),
        "distance_5_10km": aggregates.get("distance_5_10km", 0),
        "distance_gt_10km": aggregates.get("distance_gt_10km", 0),
        "first_time_patients": aggregates.get("first_time_patients", 0),
        "returning_patients": aggregates.get("returning_patients", 0),
        # Service Delivery Metrics
        "total_patients_seen": aggregates.get("total_patients_seen", 0),
        "service_curative_care": aggregates.get("service_curative_care", False),
        "service_chronic_disease": aggregates.get("service_chronic_disease", False),
        "service_maternal_child": aggregates.get("service_maternal_child", False),
        "service_preventive_care": aggregates.get("service_preventive_care", False),
        "service_reproductive_health": aggregates.get(
            "service_reproductive_health", False
        ),
        "referrals_made": aggregates.get("referrals_made", 0),
        "patients_turned_away": aggregates.get("patients_turned_away", 0),
        "wait_time_average": round(aggregates.get("wait_time_average", 0), 1),
        # Qualitative Notes
        "community_feedback": qualitative_notes.get("community_feedback", ""),
        "access_barriers": qualitative_notes.get("access_barriers", ""),
        "unmet_needs": qualitative_notes.get("unmet_needs", ""),
    }
    return entry


# Save individual patient records
def save_patient_records(visit_id, clinic_info, patient_entries):
    """Save individual patient records to a separate file"""
    if not patient_entries:
        return True

    PATIENT_RECORDS_FILE = os.path.join(DATA_DIR, "patient_records.csv")

    records = []
    for patient in patient_entries:
        record = {
            "visit_id": visit_id,
            "clinic_stop_id": clinic_info.get("clinic_stop_id", ""),
            "date": clinic_info.get("date", ""),
            "age_bracket": patient.get("age_bracket", ""),
            "gender": patient.get("gender", ""),
            "distance": patient.get("distance", ""),
            "patient_type": patient.get("patient_type", ""),
            "service_curative_care": patient.get("service_curative_care", False),
            "service_chronic_disease": patient.get("service_chronic_disease", False),
            "service_maternal_child": patient.get("service_maternal_child", False),
            "service_preventive_care": patient.get("service_preventive_care", False),
            "service_reproductive_health": patient.get(
                "service_reproductive_health", False
            ),
            "referred": patient.get("referred", False),
            "turned_away": patient.get("turned_away", False),
            "wait_time": patient.get("wait_time", 0),
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Append to existing file or create new
    if os.path.exists(PATIENT_RECORDS_FILE):
        try:
            existing_df = pd.read_csv(PATIENT_RECORDS_FILE)
            df = pd.concat([existing_df, df], ignore_index=True)
        except Exception:
            # If file exists but is corrupted, start fresh
            pass

    try:
        df.to_csv(PATIENT_RECORDS_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving patient records: {str(e)}")
        return False


# Main interface
st.markdown("---")

# Tabs for different functions
tab1, tab2, tab3, tab4 = st.tabs(
    ["📝 New Entry", "📊 View Logs", "📤 Bulk Upload", "📈 Analytics"]
)

# ============================================================================
# TAB 1: NEW ENTRY FORM
# ============================================================================
with tab1:
    st.markdown("### Enter New Visit Log")
    st.markdown(
        "Fill out the form below to record a clinic visit. All fields marked with * are required."
    )

    clinic_stops = load_clinic_stops()

    # Clinic Information Section
    st.markdown("#### 🏥 Clinic Information")
    col1, col2 = st.columns(2)

    with col1:
        clinic_stop_id = st.selectbox(
            "Clinic/MHC Stop ID *",
            options=[""] + clinic_stops if clinic_stops else [""],
            help="Select the clinic stop where the visit occurred",
        )
        visit_date = st.date_input(
            "Date *", value=date.today(), help="Date of the clinic visit"
        )

    with col2:
        operating_start = st.text_input(
            "Operating Hours Start (HH:MM)",
            value="",
            placeholder="09:00",
            help="Clinic opening time in 24-hour format",
        )
        operating_end = st.text_input(
            "Operating Hours End (HH:MM)",
            value="",
            placeholder="17:00",
            help="Clinic closing time in 24-hour format",
        )

    st.markdown("---")

    # Patient Entry Section
    st.markdown("#### 👥 Add Patients")
    st.markdown("Enter information for each patient seen during this visit.")

    # Patient entry form
    with st.expander("➕ Add New Patient", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            age_bracket = st.selectbox(
                "Age Bracket *",
                options=["", "0-5", "6-14", "15-24", "25-49", "50-59", "60+"],
                help="Select patient's age bracket",
            )
            gender = st.selectbox(
                "Gender *",
                options=["", "M", "F"],
                help="Select patient's gender",
            )
            distance = st.selectbox(
                "Distance Traveled *",
                options=["", "<1km", "1-3km", "3-5km", "5-10km", ">10km"],
                help="Distance patient traveled to reach clinic",
            )

        with col2:
            patient_type = st.selectbox(
                "Patient Type *",
                options=["", "first_time", "returning"],
                format_func=lambda x: (
                    "First-time"
                    if x == "first_time"
                    else "Returning" if x == "returning" else ""
                ),
                help="Is this a first-time or returning patient?",
            )
            wait_time = st.number_input(
                "Wait Time (minutes)",
                min_value=0,
                value=0,
                step=1,
                help="Patient's wait time in minutes",
            )

        st.markdown("**Services Provided (check all that apply):**")
        col1, col2, col3 = st.columns(3)

        with col1:
            p_service_curative = st.checkbox("Curative care", key="p_curative")
            p_service_chronic = st.checkbox(
                "Chronic disease management", key="p_chronic"
            )

        with col2:
            p_service_maternal = st.checkbox("Maternal/child health", key="p_maternal")
            p_service_preventive = st.checkbox(
                "Preventive care/vaccinations", key="p_preventive"
            )

        with col3:
            p_service_reproductive = st.checkbox(
                "Reproductive health", key="p_reproductive"
            )

        col1, col2 = st.columns(2)
        with col1:
            p_referred = st.checkbox(
                "Referred to higher-level facility", key="p_referred"
            )
        with col2:
            p_turned_away = st.checkbox("Turned away (capacity)", key="p_turned_away")

        # Add patient button
        if st.button("➕ Add Patient to List", use_container_width=True):
            if not age_bracket or not gender or not distance or not patient_type:
                st.error(
                    "❌ Please fill in all required fields (Age, Gender, Distance, Patient Type)"
                )
            else:
                patient_entry = {
                    "age_bracket": age_bracket,
                    "gender": gender,
                    "distance": distance,
                    "patient_type": patient_type,
                    "service_curative_care": p_service_curative,
                    "service_chronic_disease": p_service_chronic,
                    "service_maternal_child": p_service_maternal,
                    "service_preventive_care": p_service_preventive,
                    "service_reproductive_health": p_service_reproductive,
                    "referred": p_referred,
                    "turned_away": p_turned_away,
                    "wait_time": wait_time,
                }
                st.session_state.patient_entries.append(patient_entry)
                st.success(
                    f"✅ Patient added! Total patients: {len(st.session_state.patient_entries)}"
                )
                st.rerun()

    # Display added patients
    st.markdown("---")
    st.markdown(f"#### 📋 Patients Added ({len(st.session_state.patient_entries)})")

    if st.session_state.patient_entries:
        # Show patient list with remove option
        for idx, patient in enumerate(st.session_state.patient_entries):
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
            with col1:
                st.write(
                    f"**Patient {idx + 1}:** {patient['age_bracket']}, {patient['gender']}, {patient['distance']}, {patient['patient_type']}"
                )
            with col2:
                services = []
                if patient.get("service_curative_care"):
                    services.append("Curative")
                if patient.get("service_chronic_disease"):
                    services.append("Chronic")
                if patient.get("service_maternal_child"):
                    services.append("Maternal")
                if patient.get("service_preventive_care"):
                    services.append("Preventive")
                if patient.get("service_reproductive_health"):
                    services.append("Reproductive")
                st.write(f"Services: {', '.join(services) if services else 'None'}")
            with col3:
                st.write(f"Wait: {patient.get('wait_time', 0)} min")
            with col4:
                if patient.get("referred"):
                    st.write("🔄 Referred")
                if patient.get("turned_away"):
                    st.write("❌ Turned away")
            with col5:
                if st.button("🗑️ Remove", key=f"remove_{idx}"):
                    st.session_state.patient_entries.pop(idx)
                    st.rerun()

        # Show aggregated statistics
        st.markdown("---")
        st.markdown("#### 📊 Calculated Aggregates")
        aggregates = calculate_aggregates(st.session_state.patient_entries)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Patients", aggregates.get("total_patients_seen", 0))
            st.write(f"**Age Distribution:**")
            st.write(f"- 0-5: {aggregates.get('age_0_5', 0)}")
            st.write(f"- 6-14: {aggregates.get('age_6_14', 0)}")
            st.write(f"- 15-24: {aggregates.get('age_15_24', 0)}")
            st.write(f"- 25-49: {aggregates.get('age_25_49', 0)}")
            st.write(f"- 50-59: {aggregates.get('age_50_59', 0)}")
            st.write(f"- 60+: {aggregates.get('age_60_plus', 0)}")

        with col2:
            st.write(f"**Gender:**")
            st.write(f"- Male: {aggregates.get('gender_male', 0)}")
            st.write(f"- Female: {aggregates.get('gender_female', 0)}")
            st.write(f"**Patient Type:**")
            st.write(f"- First-time: {aggregates.get('first_time_patients', 0)}")
            st.write(f"- Returning: {aggregates.get('returning_patients', 0)}")
            st.write(f"**Distance:**")
            st.write(f"- <1km: {aggregates.get('distance_lt_1km', 0)}")
            st.write(f"- 1-3km: {aggregates.get('distance_1_3km', 0)}")
            st.write(f"- 3-5km: {aggregates.get('distance_3_5km', 0)}")
            st.write(f"- 5-10km: {aggregates.get('distance_5_10km', 0)}")
            st.write(f"- >10km: {aggregates.get('distance_gt_10km', 0)}")

        with col3:
            st.metric("Referrals", aggregates.get("referrals_made", 0))
            st.metric("Turned Away", aggregates.get("patients_turned_away", 0))
            st.metric(
                "Avg Wait Time", f"{aggregates.get('wait_time_average', 0):.1f} min"
            )
            st.write(f"**Services Provided:**")
            if aggregates.get("service_curative_care"):
                st.write("✓ Curative Care")
            if aggregates.get("service_chronic_disease"):
                st.write("✓ Chronic Disease")
            if aggregates.get("service_maternal_child"):
                st.write("✓ Maternal/Child")
            if aggregates.get("service_preventive_care"):
                st.write("✓ Preventive Care")
            if aggregates.get("service_reproductive_health"):
                st.write("✓ Reproductive Health")

        # Clear all button
        if st.button("🗑️ Clear All Patients", use_container_width=True):
            st.session_state.patient_entries = []
            st.rerun()
    else:
        st.info("No patients added yet. Use the form above to add patients.")

    st.markdown("---")

    # Qualitative Notes Section
    st.markdown("#### 📝 Qualitative Notes")

    community_feedback = st.text_area(
        "Community Feedback",
        value="",
        height=100,
        help="Any feedback received from the community",
    )

    access_barriers = st.text_area(
        "Access Barriers Observed",
        value="",
        height=100,
        help="Any barriers to accessing care that were observed",
    )

    unmet_needs = st.text_area(
        "Unmet Needs Identified",
        value="",
        height=100,
        help="Any unmet healthcare needs that were identified",
    )

    # Submit button
    st.markdown("---")
    if st.button("💾 Save Visit Log", use_container_width=True, type="primary"):
        # Validate clinic info
        clinic_info = {
            "clinic_stop_id": clinic_stop_id,
            "date": visit_date,
            "operating_hours_start": operating_start,
            "operating_hours_end": operating_end,
        }

        if not clinic_stop_id:
            st.error("❌ Clinic/MHC Stop ID is required")
        elif not st.session_state.patient_entries:
            st.error("❌ Please add at least one patient")
        else:
            # Validate operating hours if provided
            if operating_start and operating_end:
                try:
                    start = datetime.strptime(operating_start, "%H:%M").time()
                    end = datetime.strptime(operating_end, "%H:%M").time()
                    if start >= end:
                        st.error(
                            "❌ Operating hours: start time must be before end time"
                        )
                    else:
                        save_visit_log(
                            clinic_info,
                            st.session_state.patient_entries,
                            {
                                "community_feedback": community_feedback,
                                "access_barriers": access_barriers,
                                "unmet_needs": unmet_needs,
                            },
                        )
                except ValueError:
                    st.error("❌ Operating hours must be in HH:MM format")
            else:
                save_visit_log(
                    clinic_info,
                    st.session_state.patient_entries,
                    {
                        "community_feedback": community_feedback,
                        "access_barriers": access_barriers,
                        "unmet_needs": unmet_needs,
                    },
                )


def save_visit_log(clinic_info, patient_entries, qualitative_notes):
    """Save the complete visit log"""
    # Create aggregated entry
    entry = create_visit_log_entry(clinic_info, patient_entries, qualitative_notes)
    visit_id = entry["visit_id"]

    # Load existing logs
    df = load_visit_logs()

    # Add new entry
    if df.empty:
        df = pd.DataFrame([entry])
    else:
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)

    # Save aggregated log
    if save_visit_logs(df):
        # Save individual patient records
        save_patient_records(visit_id, clinic_info, patient_entries)

        st.success("✅ Visit log saved successfully!")
        st.balloons()

        # Clear patient entries
        st.session_state.patient_entries = []
        st.session_state.form_submitted = True
        st.rerun()


# ============================================================================
# TAB 2: VIEW LOGS
# ============================================================================
with tab2:
    st.markdown("### View and Manage Visit Logs")

    df = load_visit_logs()

    if df.empty:
        st.info(
            "📋 No visit logs found. Start by creating a new entry in the 'New Entry' tab."
        )
    else:
        st.markdown(f"**Total Visit Logs:** {len(df)}")

        # Filters
        st.markdown("#### 🔍 Filters")
        col1, col2, col3 = st.columns(3)

        with col1:
            if "date" in df.columns:
                date_range = st.date_input(
                    "Date Range",
                    value=(
                        df["date"].min() if not df.empty else date.today(),
                        df["date"].max() if not df.empty else date.today(),
                    ),
                    help="Filter by date range",
                )

        with col2:
            if "clinic_stop_id" in df.columns:
                clinic_filter = st.multiselect(
                    "Filter by Clinic",
                    options=(
                        sorted(df["clinic_stop_id"].unique().tolist())
                        if not df.empty
                        else []
                    ),
                    help="Select clinics to view",
                )

        with col3:
            search_term = st.text_input(
                "Search",
                placeholder="Search in notes...",
                help="Search in feedback, barriers, or unmet needs",
            )

        # Apply filters
        filtered_df = df.copy()

        if (
            "date" in filtered_df.columns
            and isinstance(date_range, tuple)
            and len(date_range) == 2
        ):
            filtered_df = filtered_df[
                (filtered_df["date"] >= date_range[0])
                & (filtered_df["date"] <= date_range[1])
            ]

        if clinic_filter:
            filtered_df = filtered_df[filtered_df["clinic_stop_id"].isin(clinic_filter)]

        if search_term:
            search_cols = ["community_feedback", "access_barriers", "unmet_needs"]
            mask = (
                filtered_df[search_cols]
                .apply(
                    lambda x: x.astype(str).str.contains(
                        search_term, case=False, na=False
                    )
                )
                .any(axis=1)
            )
            filtered_df = filtered_df[mask]

        st.markdown(f"**Filtered Results:** {len(filtered_df)} logs")

        # Display logs
        if not filtered_df.empty:
            # Select columns to display
            display_cols = [
                "date",
                "clinic_stop_id",
                "total_patients_seen",
                "operating_hours_start",
                "operating_hours_end",
                "wait_time_average",
            ]
            available_cols = [col for col in display_cols if col in filtered_df.columns]

            st.dataframe(
                filtered_df[available_cols].sort_values("date", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

            # Detailed view
            st.markdown("#### 📋 Detailed View")
            selected_index = st.selectbox(
                "Select a log to view details",
                options=range(len(filtered_df)),
                format_func=lambda x: f"{filtered_df.iloc[x]['date']} - {filtered_df.iloc[x]['clinic_stop_id']} ({filtered_df.iloc[x]['total_patients_seen']} patients)",
            )

            if selected_index is not None:
                log = filtered_df.iloc[selected_index]

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Clinic Information:**")
                    st.write(f"- **Clinic:** {log.get('clinic_stop_id', 'N/A')}")
                    st.write(f"- **Date:** {log.get('date', 'N/A')}")
                    st.write(
                        f"- **Operating Hours:** {log.get('operating_hours_start', 'N/A')} - {log.get('operating_hours_end', 'N/A')}"
                    )
                    st.write(
                        f"- **Total Patients:** {log.get('total_patients_seen', 0)}"
                    )
                    st.write(
                        f"- **Wait Time:** {log.get('wait_time_average', 'N/A')} minutes"
                    )

                with col2:
                    st.markdown("**Patient Demographics:**")
                    st.write(f"- **Age 0-5:** {log.get('age_0_5', 0)}")
                    st.write(f"- **Age 6-14:** {log.get('age_6_14', 0)}")
                    st.write(f"- **Age 15-24:** {log.get('age_15_24', 0)}")
                    st.write(f"- **Age 25-49:** {log.get('age_25_49', 0)}")
                    st.write(f"- **Age 50-59:** {log.get('age_50_59', 0)}")
                    st.write(f"- **Age 60+:** {log.get('age_60_plus', 0)}")
                    st.write(
                        f"- **Male:** {log.get('gender_male', 0)} | **Female:** {log.get('gender_female', 0)}"
                    )

                st.markdown("**Services Provided:**")
                services = []
                if log.get("service_curative_care"):
                    services.append("Curative Care")
                if log.get("service_chronic_disease"):
                    services.append("Chronic Disease Management")
                if log.get("service_maternal_child"):
                    services.append("Maternal/Child Health")
                if log.get("service_preventive_care"):
                    services.append("Preventive Care")
                if log.get("service_reproductive_health"):
                    services.append("Reproductive Health")
                st.write(", ".join(services) if services else "None specified")

                st.markdown("**Notes:**")
                if log.get("community_feedback"):
                    st.write(f"**Community Feedback:** {log.get('community_feedback')}")
                if log.get("access_barriers"):
                    st.write(f"**Access Barriers:** {log.get('access_barriers')}")
                if log.get("unmet_needs"):
                    st.write(f"**Unmet Needs:** {log.get('unmet_needs')}")

            # Export filtered data
            st.markdown("---")
            st.markdown("#### 💾 Export Data")
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered Logs (CSV)",
                data=csv_data,
                file_name=f"visit_logs_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        else:
            st.warning("No logs match the selected filters.")

# ============================================================================
# TAB 3: BULK UPLOAD
# ============================================================================
with tab3:
    st.markdown("### Bulk Upload Visit Logs")
    st.markdown(
        "Upload a CSV file with visit log data. The file should contain columns matching the visit log structure."
    )

    # Template download
    st.markdown("#### 📥 Download Template")
    template_data = {
        "clinic_stop_id": ["Example Clinic - Example Stop"],
        "date": ["2024-01-15"],
        "operating_hours_start": ["09:00"],
        "operating_hours_end": ["17:00"],
        "age_0_5": [5],
        "age_6_14": [3],
        "age_15_24": [8],
        "age_25_49": [12],
        "age_50_59": [4],
        "age_60_plus": [2],
        "gender_male": [18],
        "gender_female": [16],
        "distance_lt_1km": [10],
        "distance_1_3km": [12],
        "distance_3_5km": [8],
        "distance_5_10km": [4],
        "distance_gt_10km": [0],
        "first_time_patients": [15],
        "returning_patients": [19],
        "total_patients_seen": [34],
        "service_curative_care": [True],
        "service_chronic_disease": [False],
        "service_maternal_child": [True],
        "service_preventive_care": [True],
        "service_reproductive_health": [False],
        "referrals_made": [2],
        "patients_turned_away": [0],
        "wait_time_average": [45],
        "community_feedback": ["Positive feedback from community"],
        "access_barriers": ["None observed"],
        "unmet_needs": ["Need for more chronic disease management"],
    }
    template_df = pd.DataFrame(template_data)
    template_csv = template_df.to_csv(index=False)

    st.download_button(
        label="📥 Download CSV Template",
        data=template_csv,
        file_name="visit_logs_template.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # File upload
    st.markdown("#### 📤 Upload CSV File")
    uploaded_file = st.file_uploader(
        "Choose a CSV file", type=["csv"], help="Upload a CSV file with visit log data"
    )

    if uploaded_file is not None:
        try:
            # Read uploaded file
            upload_df = pd.read_csv(uploaded_file)

            st.success(f"✅ File loaded successfully! Found {len(upload_df)} rows.")

            # Display preview
            st.markdown("#### 👀 Preview (first 5 rows)")
            st.dataframe(upload_df.head(), use_container_width=True)

            # Validate columns
            required_cols = ["clinic_stop_id", "date", "total_patients_seen"]
            missing_cols = [
                col for col in required_cols if col not in upload_df.columns
            ]

            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            else:
                # Add timestamp if not present
                if "timestamp" not in upload_df.columns:
                    upload_df["timestamp"] = datetime.now().isoformat()

                # Validate data
                st.markdown("#### ✅ Validation")
                validation_errors = []

                for idx, row in upload_df.iterrows():
                    row_dict = row.to_dict()
                    errors = validate_visit_log(row_dict)
                    if errors:
                        validation_errors.append(f"Row {idx + 1}: {', '.join(errors)}")

                if validation_errors:
                    st.warning("⚠️ Validation issues found:")
                    for error in validation_errors[:10]:  # Show first 10 errors
                        st.write(f"- {error}")
                    if len(validation_errors) > 10:
                        st.write(f"... and {len(validation_errors) - 10} more errors")
                else:
                    st.success("✅ All rows validated successfully!")

                # Upload button
                if st.button("💾 Import Visit Logs", use_container_width=True):
                    # Load existing logs
                    existing_df = load_visit_logs()

                    # Combine
                    if existing_df.empty:
                        combined_df = upload_df
                    else:
                        combined_df = pd.concat(
                            [existing_df, upload_df], ignore_index=True
                        )

                    # Save
                    if save_visit_logs(combined_df):
                        st.success(
                            f"✅ Successfully imported {len(upload_df)} visit logs!"
                        )
                        st.balloons()
                        st.cache_data.clear()  # Clear cache to reload data

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)

# ============================================================================
# TAB 4: ANALYTICS
# ============================================================================
with tab4:
    st.markdown("### Visit Logs Analytics")

    df = load_visit_logs()

    if df.empty:
        st.info("📊 No data available for analytics. Create some visit logs first.")
    else:
        # Summary statistics
        st.markdown("#### 📊 Summary Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_logs = len(df)
            st.metric("Total Visit Logs", total_logs)

        with col2:
            total_patients = (
                df["total_patients_seen"].sum()
                if "total_patients_seen" in df.columns
                else 0
            )
            st.metric("Total Patients Served", f"{total_patients:,.0f}")

        with col3:
            avg_patients = (
                df["total_patients_seen"].mean()
                if "total_patients_seen" in df.columns
                else 0
            )
            st.metric("Avg Patients per Visit", f"{avg_patients:.1f}")

        with col4:
            unique_clinics = (
                df["clinic_stop_id"].nunique() if "clinic_stop_id" in df.columns else 0
            )
            st.metric("Unique Clinics", unique_clinics)

        # Patient demographics breakdown
        st.markdown("---")
        st.markdown("#### 👥 Patient Demographics")

        age_cols = [
            "age_0_5",
            "age_6_14",
            "age_15_24",
            "age_25_49",
            "age_50_59",
            "age_60_plus",
        ]
        available_age_cols = [col for col in age_cols if col in df.columns]

        if available_age_cols:
            age_totals = df[available_age_cols].sum()
            age_chart_df = pd.DataFrame(
                {
                    "Age Group": [
                        col.replace("age_", "").replace("_", "-")
                        for col in available_age_cols
                    ],
                    "Patients": age_totals.values,
                }
            )
            st.bar_chart(age_chart_df.set_index("Age Group"))

        # Gender distribution
        if "gender_male" in df.columns and "gender_female" in df.columns:
            gender_totals = {
                "Male": df["gender_male"].sum(),
                "Female": df["gender_female"].sum(),
            }
            gender_chart_df = pd.DataFrame(
                {
                    "Gender": list(gender_totals.keys()),
                    "Patients": list(gender_totals.values()),
                }
            )
            st.bar_chart(gender_chart_df.set_index("Gender"))

        # Services provided
        st.markdown("---")
        st.markdown("#### 🏥 Services Provided")

        service_cols = [
            "service_curative_care",
            "service_chronic_disease",
            "service_maternal_child",
            "service_preventive_care",
            "service_reproductive_health",
        ]
        available_service_cols = [col for col in service_cols if col in df.columns]

        if available_service_cols:
            service_counts = df[available_service_cols].sum()
            service_names = {
                "service_curative_care": "Curative Care",
                "service_chronic_disease": "Chronic Disease",
                "service_maternal_child": "Maternal/Child",
                "service_preventive_care": "Preventive Care",
                "service_reproductive_health": "Reproductive Health",
            }
            service_chart_df = pd.DataFrame(
                {
                    "Service": [
                        service_names.get(col, col) for col in available_service_cols
                    ],
                    "Number of Visits": service_counts.values,
                }
            )
            st.bar_chart(service_chart_df.set_index("Service"))

        # Clinic performance
        st.markdown("---")
        st.markdown("#### 🏥 Clinic Performance")

        if "clinic_stop_id" in df.columns and "total_patients_seen" in df.columns:
            clinic_performance = (
                df.groupby("clinic_stop_id")
                .agg({"total_patients_seen": ["sum", "mean", "count"]})
                .round(1)
            )
            clinic_performance.columns = [
                "Total Patients",
                "Avg per Visit",
                "Number of Visits",
            ]
            clinic_performance = clinic_performance.sort_values(
                "Total Patients", ascending=False
            )

            st.dataframe(clinic_performance, use_container_width=True)

        # Time trends
        st.markdown("---")
        st.markdown("#### 📈 Trends Over Time")

        if "date" in df.columns and "total_patients_seen" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df_sorted = df.sort_values("date")
            daily_patients = df_sorted.groupby("date")["total_patients_seen"].sum()

            trend_chart_df = pd.DataFrame(
                {"Date": daily_patients.index, "Total Patients": daily_patients.values}
            )
            st.line_chart(trend_chart_df.set_index("Date"))

# Footer
st.markdown(
    """
    <div class="footer">
      <strong>Visit Logs Management System:</strong><br/>
      • Data is stored locally in CSV format<br/>
      • All patient data is aggregated for privacy<br/>
      • Regular backups are recommended<br/>
      • For offline use, data syncs when connection is restored
    </div>
    """,
    unsafe_allow_html=True,
)
