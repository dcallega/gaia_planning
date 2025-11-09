# Quick Start: Coverage Analysis Library

## Yes, You Can Configure Everything! ✅

### Question: "Can I analyze only FREE sites at 4.5km?"

**Answer: Absolutely!**

```python
from coverage_lib import CoverageAnalyzer

analyzer = CoverageAnalyzer(service_radius_km=4.5)
analyzer.load_facilities("data/MHFR_Facilities.csv", filter_functional=True)
analyzer.facilities_df = analyzer.facilities_df[
    analyzer.facilities_df['ownership'] == 'Government'
]
analyzer.build_spatial_index()
coverage = analyzer.compute_basic_coverage("data/mwi_general_2020.csv")
```

**Result:** 35.19% coverage (825 free facilities, 4.5km radius)

---

## Common Configurations

### 1️⃣ Change Service Radius

```python
# Just change one parameter!
analyzer = CoverageAnalyzer(service_radius_km=4.5)  # or 3.0, 10.0, etc.
```

### 2️⃣ Filter by Ownership (Free vs Paid)

```python
# Government (free)
analyzer.facilities_df = analyzer.facilities_df[
    analyzer.facilities_df['ownership'] == 'Government'
]

# Exclude private (keep all free/subsidized)
analyzer.facilities_df = analyzer.facilities_df[
    analyzer.facilities_df['ownership'] != 'Private'
]
```

### 3️⃣ Filter by Type

```python
# Only hospitals
analyzer.facilities_df = analyzer.facilities_df[
    analyzer.facilities_df['type'].isin(['Hospital', 'District Hospital'])
]

# Only primary care
analyzer.facilities_df = analyzer.facilities_df[
    analyzer.facilities_df['type'].isin(['Clinic', 'Health Centre'])
]
```

### 4️⃣ Analyze Different Populations

```python
# Children under 5
coverage = analyzer.compute_basic_coverage("data/mwi_children_under_five_2020.csv")

# Women of reproductive age
coverage = analyzer.compute_basic_coverage("data/mwi_women_of_reproductive_age_15_49_2020.csv")

# Elderly
coverage = analyzer.compute_basic_coverage("data/mwi_elderly_60_plus_2020.csv")
```

### 5️⃣ Combine Multiple Filters

```python
# Free primary care for children, 4km radius
analyzer = CoverageAnalyzer(service_radius_km=4.0)
analyzer.load_facilities("data/MHFR_Facilities.csv", filter_functional=True)

analyzer.facilities_df = analyzer.facilities_df[
    (analyzer.facilities_df['ownership'] == 'Government') &
    (analyzer.facilities_df['type'].isin(['Clinic', 'Health Centre', 'Dispensary']))
]

analyzer.build_spatial_index()
coverage = analyzer.compute_basic_coverage("data/mwi_children_under_five_2020.csv")
```

---

## Available Data Filters

### Ownership Types (Malawi MHFR)
- `Government` (825 facilities) ← **Typically FREE**
- `Private` (333 facilities) ← **Typically PAID**
- `Christian Health Association of Malawi (CHAM)` (190)
- `Non-Government` (46)
- `Mission/Faith-based (other than CHAM)` (25)
- `Other` (19)
- `Parastatal` (9)

### Facility Types
- `Hospital`, `District Hospital`, `Central Hospital`
- `Health Centre`
- `Clinic`
- `Dispensary`
- `Maternity`
- `Health Post`

### Population Datasets
- `mwi_general_2020.csv` - All population
- `mwi_women_2020.csv` - Women only
- `mwi_men_2020.csv` - Men only
- `mwi_children_under_five_2020.csv` - Children <5
- `mwi_youth_15_24_2020.csv` - Youth 15-24
- `mwi_women_of_reproductive_age_15_49_2020.csv` - Women 15-49
- `mwi_elderly_60_plus_2020.csv` - Elderly 60+

---

## Complete Example Scripts

### Already Created For You:

1. **`coverage_5km.py`** - Basic coverage (all facilities, 5km)
2. **`coverage_overlap_analysis.py`** - Find redundancy and gaps
3. **`planning_workflow_example.py`** - Complete planning workflow
4. **`coverage_custom_config.py`** ✨ - **Free sites at 4.5km example**

### Run Any Example:

```bash
# Activate environment
source .venv/bin/activate

# Run your custom configuration
python coverage_custom_config.py
```

---

## Quick Comparison Results

| Configuration | Facilities | Radius | Coverage |
|--------------|-----------|---------|----------|
| **All Functional** | 1,448 | 5.0 km | **46.15%** |
| **Free Only** | 825 | 4.5 km | **35.19%** |
| **Hospitals Only** | 116 | 10.0 km | ~25% (est.) |

---

## Documentation Files

📖 **Full Guides:**
- `COVERAGE_LIBRARY_README.md` - Complete library documentation
- `CONFIGURATION_GUIDE.md` - All configuration options
- `LIBRARY_USAGE_SUMMARY.md` - Overview and results

🚀 **Quick Reference:**
- This file (`QUICK_START_COVERAGE.md`)

---

## Integration with Your App

```python
import streamlit as st
from coverage_lib import CoverageAnalyzer

# Add to your Streamlit sidebar
st.sidebar.header("Coverage Settings")

# Let users configure the analysis
radius = st.sidebar.slider("Service Radius (km)", 1.0, 10.0, 5.0, 0.5)
ownership = st.sidebar.multiselect(
    "Facility Ownership",
    ['Government', 'Private', 'CHAM', 'Other'],
    default=['Government']
)

# Run analysis with user settings
analyzer = CoverageAnalyzer(service_radius_km=radius)
analyzer.load_facilities("data/MHFR_Facilities.csv", filter_functional=True)
analyzer.facilities_df = analyzer.facilities_df[
    analyzer.facilities_df['ownership'].isin(ownership)
]
analyzer.build_spatial_index()

if st.button("Compute Coverage"):
    coverage = analyzer.compute_basic_coverage("data/mwi_general_2020.csv")
    st.metric("Coverage", f"{coverage['coverage_pct']:.2f}%")
```

---

## Next Steps

1. ✅ **Try the example:** `python coverage_custom_config.py`
2. ✅ **Modify parameters:** Change radius, ownership, etc.
3. ✅ **Integrate into your app:** Use in Streamlit or notebooks
4. ✅ **Plan mobile clinics:** Use MobileClinicPlanner with custom settings

**Everything is configurable! 🎛️**

