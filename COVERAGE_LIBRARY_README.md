# Coverage Analysis Library Documentation

## Overview

This library provides tools for analyzing healthcare facility coverage and planning mobile clinic deployments. It's designed specifically for GAIA's mobile health clinic operations in Malawi.

## Library Components

### 1. `coverage_lib.py` - Core Coverage Analysis

**Main Class: `CoverageAnalyzer`**

Provides methods for:
- Loading facilities and building spatial indices
- Computing basic coverage statistics
- Analyzing coverage overlap and redundancy
- Identifying critical facilities
- Finding coverage gaps

### 2. `mobile_clinic_planner.py` - Mobile Clinic Planning

**Main Class: `MobileClinicPlanner`**

Plans mobile clinic routes accounting for:
- Hospital-based deployment
- Weekly schedules (Monday-Friday, 5 stops)
- Off-road travel constraints (1 hour / 30km from hospital)
- Maximizing coverage of underserved populations

## Installation

```bash
# Activate virtual environment
source .venv/bin/activate

# Install required packages
pip install pandas numpy scikit-learn
```

## Quick Start

### Basic Coverage Analysis

```python
from coverage_lib import CoverageAnalyzer

# Initialize analyzer
analyzer = CoverageAnalyzer(service_radius_km=5.0)

# Load facilities
analyzer.load_facilities("data/MHFR_Facilities.csv", filter_functional=True)
analyzer.build_spatial_index()

# Compute basic coverage
coverage = analyzer.compute_basic_coverage("data/mwi_general_2020.csv")

print(f"Coverage: {coverage['coverage_pct']:.2f}%")
print(f"Covered: {coverage['covered']:,.0f} people")
print(f"Uncovered: {coverage['uncovered']:,.0f} people")
```

### Analyze Overlap and Redundancy

```python
# Compute overlap analysis
overlap = analyzer.compute_overlap_analysis("data/mwi_general_2020.csv")

# Find redundant facilities (>90% overlap)
redundant = analyzer.identify_redundant_facilities(overlap, redundancy_threshold=0.9)
print(f"Found {len(redundant)} redundant facilities")

# Find critical facilities (unique coverage >10,000)
critical = analyzer.identify_critical_facilities(overlap, min_unique_coverage=10000)
print(f"Found {len(critical)} critical facilities")
```

### Find Coverage Gaps

```python
# Identify areas >10km from any facility
gaps = analyzer.find_coverage_gaps("data/mwi_general_2020.csv", max_distance_km=10.0)

total_underserved = gaps['mwi_general_2020'].sum()
print(f"Underserved population: {total_underserved:,.0f}")
```

### Plan Mobile Clinic Routes

```python
from mobile_clinic_planner import MobileClinicPlanner, print_planning_summary

# Initialize planner
planner = MobileClinicPlanner(analyzer)

# Identify deployment hospitals
planner.identify_deployment_hospitals()

# Plan routes (max 10 teams)
routes = planner.plan_mobile_clinic_network(
    gaps,
    max_teams=10,
    min_population_per_route=15000
)

# Estimate coverage impact
impact = planner.estimate_new_coverage(routes, "data/mwi_general_2020.csv")

# Print summary
print_planning_summary(routes, impact)

# Save results
routes.to_csv('proposed_routes.csv', index=False)
```

## Complete Workflow Example

See `planning_workflow_example.py` for a complete end-to-end workflow that:

1. Analyzes current coverage
2. Identifies redundant facilities
3. Identifies critical facilities
4. Finds coverage gaps
5. Plans mobile clinic routes
6. Estimates coverage impact

Run with:
```bash
python planning_workflow_example.py
```

## Configuration Parameters

### CoverageAnalyzer

- **service_radius_km**: Service radius in kilometers (default: 5.0)
- **filter_functional**: Only include functional facilities (default: True)
- **chunksize**: Rows to process at once for memory efficiency (default: 200,000)

### MobileClinicPlanner

- **MAX_TRAVEL_DISTANCE_KM**: Maximum distance from hospital (default: 30km)
  - Based on 1 hour at 30 km/h off-road speed
- **STOPS_PER_WEEK**: Number of clinic stops (default: 5, Monday-Friday)
- **SERVICE_RADIUS_KM**: Service radius of each clinic stop (default: 5.0km)
- **min_population_per_route**: Minimum population to justify a route (default: 15,000)

## Output Files

The workflow generates several CSV files:

### From CoverageAnalyzer:
- `redundant_facilities.csv` - Facilities with high coverage overlap
- `critical_facilities.csv` - Facilities serving unique populations
- `coverage_gaps.csv` - Population points far from any facility

### From MobileClinicPlanner:
- `proposed_mobile_clinic_routes.csv` - Detailed clinic stop locations
- `mobile_clinic_summary.csv` - Summary statistics by hospital

## Understanding the Results

### Redundancy Analysis

**High redundancy (>90%)** means most people served by this facility are also within reach of another facility. These areas are:
- ✅ **Well-served** - good for population health
- ⚠️ **Potential for reallocation** - resources could be redirected

### Critical Facilities

**High unique coverage** means this facility serves many people that no other facility reaches. These facilities are:
- ⚠️ **Irreplaceable** - closing would leave population uncovered
- ✅ **High-impact** - essential for maintaining coverage

### Mobile Clinic Routes

Each route represents:
- 🚐 **One mobile clinic team** (staff + vehicle)
- 📅 **Weekly schedule** (5 stops, Monday-Friday)
- 🏥 **Hospital base** (team deploys from here each morning)
- 🎯 **Target population** (estimated people reached)

## Integration with Main Application

You can use these libraries in your Streamlit app or other applications:

```python
from coverage_lib import CoverageAnalyzer
from mobile_clinic_planner import MobileClinicPlanner

# In your app initialization
@st.cache_resource
def get_analyzer():
    analyzer = CoverageAnalyzer()
    analyzer.load_facilities("data/MHFR_Facilities.csv")
    analyzer.build_spatial_index()
    return analyzer

# Use in your app
analyzer = get_analyzer()
coverage = analyzer.compute_basic_coverage("data/mwi_general_2020.csv")

# Display in Streamlit
st.metric("Coverage", f"{coverage['coverage_pct']:.2f}%")
st.metric("People Covered", f"{coverage['covered']:,.0f}")
```

## Algorithm Details

### Spatial Indexing
- Uses **BallTree** with haversine metric for efficient spatial queries
- Time complexity: O(n log m) where n=population points, m=facilities
- Accounts for Earth's curvature using haversine distance

### Coverage Calculation
- Processes population data in chunks for memory efficiency
- Finds nearest facility for each population point
- Counts population within service radius

### Overlap Analysis
- Finds ALL facilities within service radius (not just nearest)
- Computes unique vs. shared coverage for each facility
- Identifies critical facilities based on unique coverage

### Mobile Clinic Planning
- Uses DBSCAN clustering to identify high-density gap areas
- Selects optimal clinic locations to maximize population reach
- Ensures routes stay within 1-hour travel time from hospital base
- Accounts for weekly schedule (5 stops per team)

## Performance Notes

- Population dataset: ~3.7M rows, processes in ~30-60 seconds
- Facility dataset: ~1,750 facilities (functional only)
- Memory usage: <4GB RAM with default chunksize
- Outputs: All results saved to CSV for further analysis

## Extending the Library

### Add New Population Datasets

```python
# Use different population groups
analyzer.compute_basic_coverage("data/mwi_women_2020.csv")
analyzer.compute_basic_coverage("data/mwi_children_under_five_2020.csv")
```

### Customize Service Radius

```python
# Analyze different service radii
analyzer_3km = CoverageAnalyzer(service_radius_km=3.0)
analyzer_10km = CoverageAnalyzer(service_radius_km=10.0)
```

### Filter Facilities

```python
# Only include specific facility types
hospitals = analyzer.facilities_df[
    analyzer.facilities_df['type'].isin(['Hospital', 'District Hospital'])
]
analyzer.facilities_df = hospitals
analyzer.build_spatial_index()
```

## Support

For questions or issues with the library, refer to:
- Code documentation in `coverage_lib.py`
- Example workflow in `planning_workflow_example.py`
- Original analysis scripts: `coverage_5km.py`, `coverage_overlap_analysis.py`

## License

Developed for GAIA Global Health - Malawi Planning Project

