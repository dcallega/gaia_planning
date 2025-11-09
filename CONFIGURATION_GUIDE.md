# Coverage Analysis Configuration Guide

## Overview

The coverage library is highly configurable. This guide shows all the ways you can customize your analysis.

## Quick Example: Free Sites at 4.5km

```python
from coverage_lib import CoverageAnalyzer

# Initialize with custom service radius
analyzer = CoverageAnalyzer(service_radius_km=4.5)

# Load and filter to free functional sites
analyzer.load_facilities("data/MHFR_Facilities.csv", filter_functional=True)
analyzer.facilities_df = analyzer.facilities_df[
    analyzer.facilities_df['ownership'] == 'Government'
]

# Build index and compute coverage
analyzer.build_spatial_index()
coverage = analyzer.compute_basic_coverage("data/mwi_general_2020.csv")
```

**Result:** 35.19% coverage with 825 free government facilities at 4.5km radius

## Configuration Parameters

### 1. Service Radius

Change the distance people can reasonably travel:

```python
# Conservative (3 km)
analyzer = CoverageAnalyzer(service_radius_km=3.0)

# Standard (5 km) - default
analyzer = CoverageAnalyzer(service_radius_km=5.0)

# Extended (10 km) - for areas with good transport
analyzer = CoverageAnalyzer(service_radius_km=10.0)
```

### 2. Facility Status Filter

Filter by operational status:

```python
# Only functional facilities (default)
analyzer.load_facilities("data/MHFR_Facilities.csv", filter_functional=True)

# All facilities (including non-functional)
analyzer.load_facilities("data/MHFR_Facilities.csv", filter_functional=False)
```

### 3. Ownership Filter (Free vs. Paid)

Filter by who operates the facility:

```python
# Load all functional facilities first
analyzer.load_facilities("data/MHFR_Facilities.csv", filter_functional=True)

# Filter to FREE sites (Government)
analyzer.facilities_df = analyzer.facilities_df[
    analyzer.facilities_df['ownership'] == 'Government'
]

# Filter to CHAM facilities
analyzer.facilities_df = analyzer.facilities_df[
    analyzer.facilities_df['ownership'] == 'Christian Health Association of Malawi (CHAM)'
]

# Filter to multiple ownership types
free_types = ['Government', 'Non-Government']
analyzer.facilities_df = analyzer.facilities_df[
    analyzer.facilities_df['ownership'].isin(free_types)
]

# Filter out private (everything except private)
analyzer.facilities_df = analyzer.facilities_df[
    analyzer.facilities_df['ownership'] != 'Private'
]
```

**Available ownership types in the data:**
- `Government` (825 facilities) - Typically free
- `Private` (333 facilities) - Typically paid
- `Christian Health Association of Malawi (CHAM)` (190 facilities) - Mixed
- `Non-Government` (46 facilities)
- `Mission/Faith-based (other than CHAM)` (25 facilities)
- `Other` (19 facilities)
- `Parastatal` (9 facilities)
- `Aquaid Lifeline` (1 facility)

### 4. Facility Type Filter

Filter by type of healthcare facility:

```python
# Load facilities
analyzer.load_facilities("data/MHFR_Facilities.csv", filter_functional=True)

# Only hospitals
analyzer.facilities_df = analyzer.facilities_df[
    analyzer.facilities_df['type'].isin(['Hospital', 'District Hospital', 'Central Hospital'])
]

# Only primary care (clinics and health centres)
analyzer.facilities_df = analyzer.facilities_df[
    analyzer.facilities_df['type'].isin(['Clinic', 'Health Centre', 'Dispensary'])
]

# Exclude maternity-only facilities
analyzer.facilities_df = analyzer.facilities_df[
    analyzer.facilities_df['type'] != 'Maternity'
]
```

### 5. District Filter

Analyze coverage in specific districts:

```python
# Single district
analyzer.facilities_df = analyzer.facilities_df[
    analyzer.facilities_df['district'] == 'Lilongwe'
]

# Multiple districts
districts = ['Lilongwe', 'Blantyre', 'Mzuzu']
analyzer.facilities_df = analyzer.facilities_df[
    analyzer.facilities_df['district'].isin(districts)
]

# Exclude certain districts
analyzer.facilities_df = analyzer.facilities_df[
    ~analyzer.facilities_df['district'].isin(['Chiradzulu'])
]
```

### 6. Population Dataset

Analyze coverage for different population groups:

```python
# General population (default)
coverage = analyzer.compute_basic_coverage("data/mwi_general_2020.csv")

# Women only
coverage = analyzer.compute_basic_coverage("data/mwi_women_2020.csv")

# Children under 5
coverage = analyzer.compute_basic_coverage("data/mwi_children_under_five_2020.csv")

# Women of reproductive age (15-49)
coverage = analyzer.compute_basic_coverage("data/mwi_women_of_reproductive_age_15_49_2020.csv")

# Youth (15-24)
coverage = analyzer.compute_basic_coverage("data/mwi_youth_15_24_2020.csv")

# Elderly (60+)
coverage = analyzer.compute_basic_coverage("data/mwi_elderly_60_plus_2020.csv")

# Men only
coverage = analyzer.compute_basic_coverage("data/mwi_men_2020.csv")
```

### 7. Performance Tuning

Adjust chunk size for memory/speed trade-off:

```python
# Smaller chunks (less memory, slower)
coverage = analyzer.compute_basic_coverage("data/mwi_general_2020.csv", chunksize=50_000)

# Larger chunks (more memory, faster)
coverage = analyzer.compute_basic_coverage("data/mwi_general_2020.csv", chunksize=500_000)

# Default
coverage = analyzer.compute_basic_coverage("data/mwi_general_2020.csv", chunksize=200_000)
```

## Common Configuration Scenarios

### Scenario 1: Public Healthcare Access

*"What percentage of population can reach a FREE government facility within 5km?"*

```python
analyzer = CoverageAnalyzer(service_radius_km=5.0)
analyzer.load_facilities("data/MHFR_Facilities.csv", filter_functional=True)
analyzer.facilities_df = analyzer.facilities_df[
    analyzer.facilities_df['ownership'] == 'Government'
]
analyzer.build_spatial_index()
coverage = analyzer.compute_basic_coverage("data/mwi_general_2020.csv")
# Result: 35.19% coverage
```

### Scenario 2: Maternal Healthcare Access

*"What percentage of women of reproductive age can reach a maternity facility within 3km?"*

```python
analyzer = CoverageAnalyzer(service_radius_km=3.0)
analyzer.load_facilities("data/MHFR_Facilities.csv", filter_functional=True)
# Include facilities that provide maternal services
analyzer.facilities_df = analyzer.facilities_df[
    analyzer.facilities_df['type'].isin(['Hospital', 'Health Centre', 'Maternity', 'Clinic'])
]
analyzer.build_spatial_index()
coverage = analyzer.compute_basic_coverage("data/mwi_women_of_reproductive_age_15_49_2020.csv")
```

### Scenario 3: Rural Hospital Access

*"What percentage of rural population can reach a hospital within 10km?"*

```python
analyzer = CoverageAnalyzer(service_radius_km=10.0)
analyzer.load_facilities("data/MHFR_Facilities.csv", filter_functional=True)
analyzer.facilities_df = analyzer.facilities_df[
    analyzer.facilities_df['type'].isin(['Hospital', 'District Hospital', 'Central Hospital'])
]
analyzer.build_spatial_index()
coverage = analyzer.compute_basic_coverage("data/mwi_general_2020.csv")
```

### Scenario 4: District-Specific Analysis

*"What is healthcare coverage in Lilongwe district?"*

```python
analyzer = CoverageAnalyzer(service_radius_km=5.0)
analyzer.load_facilities("data/MHFR_Facilities.csv", filter_functional=True)
analyzer.facilities_df = analyzer.facilities_df[
    analyzer.facilities_df['district'] == 'Lilongwe'
]
analyzer.build_spatial_index()
# Note: This will compute coverage against ALL population, not just Lilongwe population
coverage = analyzer.compute_basic_coverage("data/mwi_general_2020.csv")
```

### Scenario 5: Combined Filters

*"Coverage for children under 5 at free primary care facilities within 4km"*

```python
analyzer = CoverageAnalyzer(service_radius_km=4.0)
analyzer.load_facilities("data/MHFR_Facilities.csv", filter_functional=True)

# Filter to free facilities
analyzer.facilities_df = analyzer.facilities_df[
    analyzer.facilities_df['ownership'] == 'Government'
]

# Filter to primary care facilities
analyzer.facilities_df = analyzer.facilities_df[
    analyzer.facilities_df['type'].isin(['Clinic', 'Health Centre', 'Dispensary'])
]

analyzer.build_spatial_index()
coverage = analyzer.compute_basic_coverage("data/mwi_children_under_five_2020.csv")
```

## Mobile Clinic Planning Configurations

### Custom Travel Distance

```python
from mobile_clinic_planner import MobileClinicPlanner

# Modify the constant before creating planner
import mobile_clinic_planner
mobile_clinic_planner.MAX_TRAVEL_DISTANCE_KM = 40.0  # Extend to 40km

planner = MobileClinicPlanner(analyzer)
```

### Custom Population Threshold

```python
# Only plan routes that serve at least 20,000 people
routes = planner.plan_mobile_clinic_network(
    gaps,
    max_teams=10,
    min_population_per_route=20000  # Increased from 15,000
)
```

### Limit to Specific Hospitals

```python
# Manually filter hospitals
planner.identify_deployment_hospitals()
planner.hospitals = planner.hospitals[
    planner.hospitals['district'].isin(['Lilongwe', 'Mzuzu'])
]

# Then plan routes
routes = planner.plan_mobile_clinic_network(gaps)
```

## Validation and Comparison

Compare different configurations:

```python
# Configuration 1: All facilities, 5km
analyzer1 = CoverageAnalyzer(service_radius_km=5.0)
analyzer1.load_facilities("data/MHFR_Facilities.csv", filter_functional=True)
analyzer1.build_spatial_index()
coverage1 = analyzer1.compute_basic_coverage("data/mwi_general_2020.csv")

# Configuration 2: Free only, 4.5km
analyzer2 = CoverageAnalyzer(service_radius_km=4.5)
analyzer2.load_facilities("data/MHFR_Facilities.csv", filter_functional=True)
analyzer2.facilities_df = analyzer2.facilities_df[
    analyzer2.facilities_df['ownership'] == 'Government'
]
analyzer2.build_spatial_index()
coverage2 = analyzer2.compute_basic_coverage("data/mwi_general_2020.csv")

# Compare
print(f"All facilities (5km):      {coverage1['coverage_pct']:.2f}%")
print(f"Free facilities (4.5km):   {coverage2['coverage_pct']:.2f}%")
print(f"Difference:                {coverage1['coverage_pct'] - coverage2['coverage_pct']:.2f}pp")
```

## Tips

1. **Always filter before building spatial index** - More efficient
2. **Use .copy()** to avoid pandas warnings when modifying dataframes
3. **Save filtered facility lists** - Document what facilities were included
4. **Compare configurations** - Understand impact of different assumptions
5. **Validate results** - Check facility counts make sense before running analysis

## Example: Complete Custom Workflow

```python
from coverage_lib import CoverageAnalyzer
from mobile_clinic_planner import MobileClinicPlanner

# Custom configuration: Free sites at 4.5km
analyzer = CoverageAnalyzer(service_radius_km=4.5)
analyzer.load_facilities("data/MHFR_Facilities.csv", filter_functional=True)
analyzer.facilities_df = analyzer.facilities_df[
    analyzer.facilities_df['ownership'] == 'Government'
].copy()
analyzer.build_spatial_index()

# Analyze coverage
coverage = analyzer.compute_basic_coverage("data/mwi_general_2020.csv")
print(f"Free site coverage: {coverage['coverage_pct']:.2f}%")

# Find gaps
gaps = analyzer.find_coverage_gaps("data/mwi_general_2020.csv", max_distance_km=10.0)

# Plan mobile clinics
planner = MobileClinicPlanner(analyzer)
routes = planner.plan_mobile_clinic_network(gaps, max_teams=10)

# Save results with descriptive name
routes.to_csv('proposed_routes_free_sites_4.5km.csv', index=False)
```

---

**The library is fully flexible - configure it to match your exact analysis needs!**

