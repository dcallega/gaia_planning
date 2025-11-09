# Caching Guide

## Overview

The GAIA Planning app uses a multi-layered caching strategy to provide fast performance for expensive operations like spatial joins and coverage calculations.

## Caching Layers

### 1. **Parquet Cache Files** (Persistent, Manual)
- **Location**: `data/.cache/*.parquet`
- **Purpose**: Pre-computed population data with district assignments
- **Persistence**: Survives app restarts, git-ignored
- **Speed**: Fastest (~10-100x faster than recomputing)

### 2. **Streamlit Cache** (Persistent, Automatic)
- **Location**: `.streamlit/cache/`
- **Purpose**: Coverage calculations and metrics
- **Persistence**: Survives app restarts
- **Speed**: Very fast (instant for cached results)

## Quick Start

### First-Time Setup (Recommended)

Pre-populate all caches before running the app:

```bash
# Run the pre-population script
python prepopulate_cache.py
```

This will:
1. Load all 7 population datasets
2. Assign districts to ~50k population points per dataset
3. Save to `.parquet` files for instant loading
4. Takes ~15-30 minutes total (one-time operation)

Then start the app normally:

```bash
streamlit run app.py
```

### Without Pre-population

You can start the app without pre-populating:

```bash
streamlit run app.py
```

The first time you select each population dataset:
- Districts will be assigned on-the-fly (~2-5 minutes per dataset)
- Results are cached automatically
- Subsequent loads are instant

## Performance Comparison

| Operation | Without Cache | With Parquet Cache | With Full Cache |
|-----------|--------------|-------------------|-----------------|
| Load population data | ~5-10s | ~0.5-1s | ~0.5-1s |
| Assign districts | ~2-5 min | N/A (pre-assigned) | N/A |
| Calculate coverage | ~3-5s | ~3-5s | ~0.1s |
| Total first load | ~3-6 min | ~5-10s | ~1s |
| Subsequent loads | ~1s | ~1s | ~0.1s |

## Cache Management

### View Cache Status

Check what's cached:

```bash
# Check parquet cache
ls -lh data/.cache/

# Check size
du -sh data/.cache/
```

### Clear Cache

If you need to rebuild the cache (e.g., after updating boundaries):

```bash
# Clear parquet cache
rm -rf data/.cache/

# Clear Streamlit cache
rm -rf .streamlit/cache/
```

Then re-run `prepopulate_cache.py` or let the app rebuild on next run.

### Selective Cache Clearing

Clear cache for a specific dataset:

```bash
# Example: Clear just the general population cache
rm data/.cache/mwi_general_2020_with_districts.parquet
```

## What Gets Cached

### Parquet Cache Files
- **Population points** with coordinates
- **Assigned districts** (most expensive operation)
- **Population values** for selected demographic
- ~5-15 MB per dataset

### Streamlit Cache
- **Coverage metrics**: total population, covered population, percentiles
- **District breakdowns**: metrics per district
- **Facility combinations**: cached per facility selection

## When to Rebuild Cache

Rebuild the cache when:

1. **District boundaries change**
   - Download new boundaries
   - Run `prepopulate_cache.py`

2. **Population data updates**
   - Clear old cache: `rm -rf data/.cache/`
   - Run `prepopulate_cache.py`

3. **Sample size changes**
   - Update `SAMPLE_SIZE` in both `app.py` and `prepopulate_cache.py`
   - Rebuild cache

4. **Cache corruption**
   - Clear all caches
   - Restart app or run prepopulate script

## Technical Details

### Caching Strategy

1. **Population Data Loading** (`load_population_data`)
   ```python
   @st.cache_data(persist="disk", show_spinner=False)
   ```
   - Checks for `.parquet` cache first
   - Falls back to CSV + district assignment
   - Auto-saves result to parquet
   - Persists across app restarts

2. **Coverage Calculations** (`calculate_coverage_metrics`)
   ```python
   @st.cache_data(persist="disk", show_spinner=False, max_entries=20)
   ```
   - Caches by: population data, facilities, column, radius
   - Keeps last 20 calculations
   - Persists across restarts

3. **District Metrics** (`calculate_district_coverage_metrics`)
   ```python
   @st.cache_data(persist="disk", show_spinner=False, max_entries=20)
   ```
   - Caches by: population data, facilities, column, radius
   - Builds spatial index once
   - Persists across restarts

### Cache Keys

Streamlit generates cache keys from function arguments:
- DataFrames: hash of contents (row-order independent)
- Scalars: direct value
- Changes to any input = new cache entry

### Storage Requirements

- **Parquet cache**: ~50-100 MB total (all datasets)
- **Streamlit cache**: ~100-500 MB (depends on usage)
- **Total**: ~150-600 MB

## Troubleshooting

### "Calculating for the first time"

**Symptom**: Metrics take 3-5 minutes to calculate

**Solution**: 
- Run `prepopulate_cache.py` to pre-compute
- Or wait once, then it's cached

### "Cache not persisting"

**Symptom**: Cache rebuilds every app restart

**Check**:
1. Parquet files exist: `ls data/.cache/`
2. Streamlit cache dir: `ls .streamlit/cache/`
3. File permissions (should be writable)

**Solution**:
- Ensure `data/.cache/` directory exists and is writable
- Check `@st.cache_data(persist="disk")` decorators

### "Out of disk space"

**Symptom**: Cache filling up disk

**Solution**:
```bash
# Clear Streamlit cache (safe, will rebuild)
rm -rf .streamlit/cache/

# Reduce max_entries in cache decorators
# In app.py, change: max_entries=20 to max_entries=5
```

### "Wrong results after data update"

**Symptom**: Metrics don't reflect new data

**Solution**:
```bash
# Clear all caches
rm -rf data/.cache/ .streamlit/cache/

# Rebuild
python prepopulate_cache.py
```

## Best Practices

1. **Pre-populate in production**
   - Always run `prepopulate_cache.py` before deploying
   - Include in deployment scripts

2. **Version cache with data**
   - Clear cache when data changes
   - Document data versions

3. **Monitor cache size**
   - Periodically check: `du -sh data/.cache/ .streamlit/cache/`
   - Set alerts if >1 GB

4. **Test without cache**
   - Occasionally test fresh load
   - Verify cache performance gains

5. **Document changes**
   - If changing sample size or boundaries, document in changelog
   - Notify team to rebuild cache

## Advanced: Custom Cache Configuration

### Change Sample Size

In `app.py` and `prepopulate_cache.py`:

```python
# Increase for more accuracy, decrease for speed
sample_size = min(50000, len(df))  # Default: 50k
sample_size = min(100000, len(df))  # More data: 100k
sample_size = min(25000, len(df))  # Faster: 25k
```

After changing, rebuild cache:
```bash
rm -rf data/.cache/
python prepopulate_cache.py
```

### Change Cache Limits

In `app.py`:

```python
# Increase to cache more facility combinations
@st.cache_data(persist="disk", max_entries=20)  # Default: 20
@st.cache_data(persist="disk", max_entries=50)  # More: 50
```

### Disable Caching (Debug Only)

Remove decorators temporarily:

```python
# Before (cached):
@st.cache_data(persist="disk")
def calculate_coverage_metrics(...):

# After (not cached):
def calculate_coverage_metrics(...):
```

**Warning**: App will be very slow without caching!

## Summary

✅ **Do**:
- Run `prepopulate_cache.py` before first use
- Keep cache files in `.gitignore`
- Clear cache after data updates
- Monitor disk space

❌ **Don't**:
- Commit cache files to git
- Manually edit cache files
- Forget to rebuild after data changes
- Disable caching in production

