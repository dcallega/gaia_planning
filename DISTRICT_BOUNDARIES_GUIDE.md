# District Boundaries and Spatial Analysis Guide

This guide explains how to use the district boundary visualization and spatial analysis features in the GAIA Planning Map application.

## Overview

The application now supports:
1. **Visualizing district boundaries** on the map
2. **Associating data points with districts** using spatial joins
3. **Analyzing facility and clinic distribution** by district

## Features

### 1. District Boundary Visualization

#### Enabling District Boundaries
- In the sidebar, check the **"🗺️ District Boundaries"** checkbox
- District boundaries will appear on the map with:
  - Light green fill (GAIA brand color with transparency)
  - Dark green borders
  - Interactive tooltips showing district names

#### What's Displayed
- **28 districts** of Malawi based on official administrative boundaries
- Boundaries are from [geoBoundaries](https://www.geoboundaries.org/) - an open-source database of political administrative boundaries

### 2. District Assignment for Data Points

#### Automatic District Assignment
The app can automatically determine which district any point (latitude/longitude) belongs to using spatial joins.

#### For Health Facilities (MHFR Data)
1. Navigate to the **"🗺️ District Analysis"** section at the bottom
2. Expand **"📊 Facility Distribution by District"**
3. View statistics on:
   - Facilities with existing district information
   - Facilities without district information
4. Click **"🔍 Auto-assign Districts Using Spatial Join"** to automatically assign districts based on coordinates

#### For GAIA Mobile Clinics
1. In the same section, scroll to **"GAIA Mobile Clinics"**
2. Click **"🔍 Assign Districts to GAIA Clinics"**
3. View the distribution of clinic stops by district

### 3. Batch Processing

For large-scale district assignment, use the batch processing script:

```bash
# Activate your virtual environment
source .venv/bin/activate

# Run the district assignment script
python assign_districts.py
```

This will:
- Process all MHFR facilities and assign districts
- Process all GAIA clinic stops and assign districts
- Save results to new CSV files:
  - `data/MHFR_Facilities_with_districts.csv`
  - `data/GAIA_Clinics_with_districts.csv`

## Technical Details

### Files Added/Modified

1. **download_boundaries.py**
   - Downloads GeoJSON boundary files from geoBoundaries
   - Creates: `data/boundaries/malawi_districts.geojson`
   - Creates: `data/boundaries/malawi_regions.geojson`
   - Creates: `data/boundaries/malawi_country.geojson`

2. **spatial_utils.py**
   - Core spatial analysis functions
   - `point_to_district()`: Determines which district a point belongs to
   - `assign_districts_to_dataframe()`: Batch assigns districts to all rows in a dataframe
   - Uses Shapely library for geometric operations

3. **assign_districts.py**
   - Batch processing script for district assignment
   - Processes facilities, clinics, and optionally population data
   - Saves results to new CSV files

4. **app.py** (updated)
   - Added district boundary visualization layer
   - Added district analysis section
   - Interactive district assignment features

### Dependencies

New dependency added to `requirements.txt`:
- **shapely==2.0.6**: For geometric operations and spatial joins

### District Matching Algorithm

The spatial join works by:
1. Loading district polygon boundaries from GeoJSON
2. Creating a Point geometry from latitude/longitude
3. Testing which polygon contains the point
4. Returning the district name from the polygon's properties

```python
from spatial_utils import point_to_district

# Example usage
district = point_to_district(
    lat=-13.9626,  # Lilongwe coordinates
    lon=33.7741,
    district_geojson=district_boundaries
)
# Returns: "Lilongwe"
```

## Data Sources

- **Administrative Boundaries**: [geoBoundaries](https://www.geoboundaries.org/) - Open-source political administrative boundaries
- **District Names**: Match the 28 districts shown in the reference map provided

## Use Cases

### 1. Coverage Analysis
Identify which districts have:
- High/low density of health facilities
- GAIA mobile clinic coverage
- Population density by district

### 2. Planning
- Identify underserved districts
- Plan new clinic routes
- Optimize resource allocation

### 3. Data Quality
- Verify existing district assignments
- Fill in missing district information
- Identify potential data errors (coordinates outside Malawi)

### 4. Reporting
- Generate district-level statistics
- Create district summaries
- Export district-tagged data for further analysis

## District List

The 28 districts of Malawi (as shown in the map):
1. Balaka
2. Blantyre
3. Chikwawa
4. Chiradzulu
5. Chitipa
6. Dedza
7. Dowa
8. Karonga
9. Kasungu
10. Likoma
11. Lilongwe
12. Machinga
13. Mangochi
14. Mchinji
15. Mulanje
16. Mwanza
17. Mzimba
18. Neno
19. Nkhata Bay
20. Nkhotakota
21. Nsanje
22. Ntcheu
23. Ntchisi
24. Phalombe
25. Rumphi
26. Salima
27. Thyolo
28. Zomba

## Troubleshooting

### Boundaries Not Showing
If boundaries don't appear:
1. Run `python download_boundaries.py` to download the boundary files
2. Check that `data/boundaries/` directory exists
3. Verify GeoJSON files are present

### District Assignment Not Working
If spatial joins fail:
1. Ensure shapely is installed: `pip install shapely`
2. Check that coordinates are valid (not NaN or 0)
3. Verify coordinates are within Malawi bounds:
   - Latitude: -17° to -9°
   - Longitude: 32° to 36°

### Performance Issues
For large datasets:
- Use the batch processing script instead of interactive assignment
- Process data in chunks
- Consider sampling population data before assignment

## Future Enhancements

Potential additions:
- Region-level analysis (3 regions: Northern, Central, Southern)
- Distance to nearest district capital
- District-level statistics (area, population totals)
- Export district summary reports
- Filter facilities by district
- District comparison tools

