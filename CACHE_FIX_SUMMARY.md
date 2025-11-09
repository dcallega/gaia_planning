# Cache Hash Fix Summary

## Problem
The District Analysis page was failing with:
```
TypeError: unhashable type: 'numpy.ndarray'
```

This was caused by cached parquet files containing a `color` column with numpy array values, which pandas cannot hash for Streamlit's caching system.

## Solution

### 1. Added Cache Helper Functions (`data_utils.py`)

Created centralized utilities to:
- Strip unhashable columns (like `color`) from DataFrames
- Provide deterministic cache keys based on data shape and population totals
- Share hash functions across all population caching operations

```python
def hash_dataframe_for_cache(df: pd.DataFrame) -> str:
    """Stable cache hash using attrs metadata instead of DataFrame content"""
    
def prepare_population_dataframe(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Remove unhashable columns and attach cache key"""
    
POPULATION_CACHE_HASH_FUNCS = {pd.DataFrame: hash_dataframe_for_cache}
```

### 2. Updated Population Loading Functions

Both `app.py` and `pages/3_District_Analysis.py` now:
- Call `prepare_population_dataframe()` before returning cached data
- Use `hash_funcs=POPULATION_CACHE_HASH_FUNCS` in `@st.cache_data` decorators
- Automatically clean problematic columns on load

### 3. Fixed Deprecation Warnings

- Replaced all `use_container_width=True` with `width="stretch"`
- Added `observed=False` to pandas groupby operations to silence warnings

### 4. Cleaned Existing Cache

Ran cleanup script to remove `color` column from all cached parquet files.

## How to Use

### For Users
1. **No action needed** - the fix is automatic
2. If you see issues, restart the Streamlit app
3. Navigate to the app through `http://localhost:8501` (not direct page URLs)

### For Developers
When adding new cached population functions:
```python
from data_utils import POPULATION_CACHE_HASH_FUNCS, prepare_population_dataframe

@st.cache_data(
    persist="disk",
    show_spinner=False,
    hash_funcs=POPULATION_CACHE_HASH_FUNCS,  # ← Add this
)
def my_population_function(population_df, ...):
    # Your logic here
    df = prepare_population_dataframe(df, dataset_name)  # ← Call before returning
    return df
```

## Navigation Note

The 404 errors (`/District_Analysis/_stcore/...`) occur when:
- Directly accessing page URLs instead of using the main app
- The page has a Python error that prevents proper initialization

**Solution**: Always access the app through `http://localhost:8501` and use the navigation menu.

## Files Changed

- `data_utils.py` - Added cache helper functions
- `app.py` - Updated population loading to use helpers
- `pages/3_District_Analysis.py` - Updated population loading to use helpers
- `pages/2_Coverage_Analysis.py` - Fixed deprecation warnings
- `pages/visit_logs.py` - Fixed deprecation warnings

## Testing

Verified that:
- Cached parquet files no longer contain `color` column
- Cache keys are properly set using `.attrs['_gaia_cache_key']`
- No linter errors in updated files

