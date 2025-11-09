# Choropleth Map Implementation Guide

## Overview

The District Analysis page now uses a **choropleth map** instead of a heatmap to display population density. A choropleth map divides the district into discrete sub-regions (Traditional Authorities) and colors each region based on its population density.

## What Changed?

### Before (Heatmap)
- Smooth gradient showing population density
- No clear boundaries between regions
- Based on individual population data points

### After (Choropleth)
- Discrete colored regions with clear boundaries
- Each region represents a Traditional Authority (sub-district)
- Teal color scale: lighter teal = lower population, darker teal = higher population
- Interactive tooltips showing population per region

## Technical Implementation

### 1. Data Sources

**Sub-District Boundaries:**
- Downloaded from GADM (Global Administrative Areas Database)
- Level 3 boundaries for Malawi (Traditional Authorities)
- File: `data/boundaries/malawi_level3.geojson`
- Contains 3,126 administrative units

### 2. Key Functions

**`load_subdistrict_boundaries()`**
- Loads the level 3 GeoJSON boundaries
- Cached for performance

**`get_subdistrict_boundaries_for_district(district_name)`**
- Filters sub-district boundaries to match the selected district
- Uses fuzzy matching to handle naming variations between datasets

**`aggregate_population_by_subdistrict(population_df, subdistrict_geojson, pop_column)`**
- Aggregates population points within each sub-district polygon
- Uses GeoPandas for efficient spatial operations
- Returns GeoJSON with population totals per region

**`get_color_for_population(population, min_pop, max_pop)`**
- Maps population values to teal color scale
- 5 discrete color bins for clarity

### 3. Color Scale

The choropleth uses shades of teal:

| Population Percentile | Color | RGB |
|---------------------|-------|-----|
| 0-20% | Very Light Teal | `[224, 242, 241]` |
| 20-40% | Light Teal | `[178, 223, 219]` |
| 40-60% | Medium-Light Teal | `[128, 203, 196]` |
| 60-80% | Medium-Dark Teal | `[38, 166, 154]` |
| 80-100% | Dark Teal | `[0, 121, 107]` |

### 4. Map Layers

The map now includes:
1. **Choropleth layer** - Colored sub-district regions (population density)
2. **Facility layers** - Hospitals, health centers, clinics, dispensaries
3. **GAIA mobile clinics** - Existing mobile clinic stops
4. **Recommended stops** - Star-shaped markers for new clinic locations
5. **District boundary** - Outline of the selected district

## Features

### Interactive Tooltips

Hover over any sub-district region to see:
- Region name (Traditional Authority)
- Total population in that region
- Number of data points used for calculation
- Color intensity explanation

### Visual Legend

A gradient legend at the top of the map shows:
- Color scale from low to high population density
- Clear labeling of what colors represent

### Performance Optimization

- Uses GeoPandas spatial indexing for fast point-in-polygon tests
- Caches boundary data to avoid repeated loading
- Optimized aggregation algorithm for large datasets

## Usage

1. Select a district from the dropdown
2. The map automatically loads sub-district boundaries for that district
3. Population data is aggregated by sub-district
4. Each region is colored based on its total population
5. Hover over regions to see detailed information

## Fallback Behavior

If sub-district boundaries are not available for a district:
- Falls back to simple point visualization
- Shows informational message to user
- Still displays all other map layers (facilities, etc.)

## Data Requirements

### Required Files
- `data/boundaries/malawi_level3.geojson` - Sub-district boundaries (auto-downloaded)
- `data/mwi_general_2020.csv` - Population data points

### Python Packages
- `geopandas` - Spatial operations
- `shapely` - Geometry handling
- `pydeck` - Map visualization
- `streamlit` - Web interface

## Downloading Boundaries

To re-download or update sub-district boundaries:

```bash
python download_subdistrict_boundaries.py
```

This script:
1. Downloads level 3 boundaries from GADM for Malawi
2. Saves to `data/boundaries/malawi_level3.geojson`
3. Shows summary of available properties

## Comparison: Heatmap vs Choropleth

| Feature | Heatmap | Choropleth |
|---------|---------|-----------|
| Boundaries | No boundaries | Clear regional boundaries |
| Color | Continuous gradient | Discrete color bins |
| Data Points | Visible as dots | Aggregated by region |
| Interpretation | Shows density hotspots | Shows population by admin unit |
| Performance | Faster rendering | More computational aggregation |
| Use Case | Quick density overview | Detailed regional analysis |

## Benefits of Choropleth

1. **Administrative Relevance** - Maps align with actual administrative units
2. **Clearer Boundaries** - Easy to see which areas have high/low population
3. **Quantifiable** - Can see exact population numbers per region
4. **Policy-Friendly** - Easier to make decisions based on administrative units
5. **Comparison** - Easy to compare different sub-districts

## Future Enhancements

Possible improvements:
- Add region selection to filter data
- Show multiple metrics (population, facilities, coverage)
- Export choropleth data as CSV/shapefile
- Add time-series animation if historical data available
- Allow custom color scales and binning strategies

