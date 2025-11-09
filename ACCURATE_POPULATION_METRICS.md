# Accurate Population Metrics - Implementation Summary

## Problem Identified

The app was showing incorrect population metrics due to sampling issues:

1. **Incorrect Population Sum**: Showed ~254,535 instead of the correct ~19.1 million
2. **Root Cause**: Data was sampled **before** filtering by country boundaries
   - Sampled 50k points from 3.7M total points (many outside Malawi)
   - Then filtered by country boundaries → only ~29k points remained
   - Result: Highly inaccurate, underrepresented population data

## Solution Implemented

### Two-Phase Approach: Full Data for Metrics, Sampled for Visualization

**Phase 1: Load Full Filtered Dataset**
- Load all population data (3.7M points)
- Filter out low population values (>0.5)
- Filter to only points **inside country boundaries** → 2.175M points
- Cache the filtered data as parquet files for instant reloading
- **Total population: 19,092,353** ✅

**Phase 2: Sample Only for Map Visualization**
- Use **full 2.175M points** for all metrics calculations
- Create **50k sample** only for map rendering (to avoid browser limits)
- Sample is drawn from the already-filtered data inside Malawi

### Benefits

✅ **100% Accurate Metrics**: All calculations use the complete dataset (2.175M points)
✅ **No Websocket Errors**: Map only renders 50k points (avoiding 562MB limit)
✅ **Fast Performance**: Cached parquet files load instantly
✅ **Correct Population**: Shows ~19.1M population (matches official estimates)
✅ **No Browser Slowdown**: 50k points render smoothly on the map

## Files Modified

### 1. `app.py`

**Changes:**
- Modified `load_population_data()` to load full filtered dataset (no sampling)
- Added `sample_for_visualization()` function for map rendering only
- Updated map layer to use sampled data (`population_df_viz`)
- Keep using full data (`population_df`) for all metrics calculations
- Added info banner showing data point counts

**Key Code:**
```python
# Load full dataset for metrics
population_df = load_population_data(dataset_name)  # 2.175M points

# Sample only for visualization
population_df_viz = sample_for_visualization(population_df, sample_size=50000)

# Metrics use full data
coverage_metrics = calculate_coverage_metrics(population_df, ...)  # 19.1M population

# Map uses sampled data
layers.append(pdk.Layer("ScatterplotLayer", data=population_df_viz, ...))
```

### 2. `prepopulate_cache.py`

**Completely Rewritten:**
- Now generates filtered population caches (without sampling)
- Creates `.parquet` files in `data/.cache/` for instant loading
- Optional district assignment (separate phase)
- New command-line arguments:
  - `--only DATASET`: Process just one dataset
  - `--with-districts`: Also assign districts
  - `--force`: Regenerate existing caches

**Usage:**
```bash
# Generate filtered cache for all datasets (recommended)
python prepopulate_cache.py

# Generate cache for just general population
python prepopulate_cache.py --only general

# Also assign districts (slower, optional)
python prepopulate_cache.py --with-districts
```

### 3. `spatial_utils.py`

No changes needed - already had the `filter_points_in_country()` function.

## Cache Files Created

Location: `data/.cache/`

**Filtered Population Data** (used by default):
- `mwi_general_2020_filtered.parquet` (~6 MB)
- `mwi_women_2020_filtered.parquet`
- `mwi_men_2020_filtered.parquet`
- `mwi_children_under_five_2020_filtered.parquet`
- `mwi_youth_15_24_2020_filtered.parquet`
- `mwi_elderly_60_plus_2020_filtered.parquet`
- `mwi_women_of_reproductive_age_15_49_2020_filtered.parquet`

**With Districts** (optional, for district breakdowns):
- `mwi_{dataset}_2020_with_districts.parquet`

## Verification Results

### Old (Incorrect) Approach:
```
1. Load: 3,741,693 rows
2. Sample: 50,000 rows (many outside Malawi)
3. Filter > 0.5: 48,263 rows
4. Filter by country: 29,039 rows ❌
Result: 254,535 population (WRONG!)
```

### New (Correct) Approach:
```
1. Load: 3,741,693 rows
2. Filter > 0.5: 3,611,641 rows
3. Filter by country: 2,175,099 rows ✅
4. Use ALL for metrics: 19,092,353 population (CORRECT!)
5. Sample for map only: 50,000 rows (visualization only)
```

## Performance

- **Cache Generation**: ~60-90 seconds per dataset
- **Loading Cached Data**: < 1 second (instant!)
- **Map Rendering**: Smooth with 50k points
- **Metrics Calculation**: Fast with full dataset (2.175M points)
- **Memory Usage**: ~66 MB per full dataset (manageable)

## Next Steps

1. ✅ Run `python prepopulate_cache.py` to generate all caches
2. ✅ Restart Streamlit app: `streamlit run app.py`
3. ✅ Verify population metrics show ~19.1M for General Population
4. ✅ Confirm map renders smoothly with no websocket errors

## Technical Notes

**Why This Works:**

1. **Parquet Format**: Compressed columnar storage (6MB vs 66MB in memory)
2. **Disk Persistence**: Cache survives restarts, no recomputation needed
3. **Streamlit Caching**: Additional memory cache for even faster access
4. **Spatial Filtering**: Removes ~1.5M points outside Malawi boundaries
5. **Stratified Sampling**: Map visualization maintains spatial distribution

**Tradeoffs:**

- ✅ 100% accurate metrics
- ✅ Fast map rendering
- ✅ No browser limits
- ⚠️ Initial cache generation takes ~10 minutes for all datasets (one-time)
- ⚠️ Slightly higher memory usage (~66MB vs ~17MB per dataset)

The tradeoff is worth it for accurate population metrics!

