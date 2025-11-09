#!/usr/bin/env python3
"""
Pre-populate cache for GAIA Planning app

This script pre-computes two types of caches:

1. FILTERED POPULATION DATA (RECOMMENDED):
   - Filters population data to only points within country boundaries
   - Removes sampling for accurate population metrics
   - Makes app load much faster on subsequent runs
   - Creates: data/.cache/mwi_{dataset}_2020_filtered.parquet

2. DISTRICT ASSIGNMENTS (OPTIONAL):
   - Assigns districts to population points
   - Only needed if you frequently use district breakdowns
   - Creates: data/.cache/mwi_{dataset}_2020_with_districts.parquet

Usage:
    python prepopulate_cache.py                    # Filter all datasets (recommended)
    python prepopulate_cache.py --with-districts   # Also assign districts
    python prepopulate_cache.py --only general     # Process only one dataset
"""

import os
import sys
import time
import argparse
import pandas as pd
from spatial_utils import assign_districts_to_dataframe, filter_points_in_country

# Population datasets to process
POPULATION_DATASETS = {
    "General Population": "general",
    "Women": "women",
    "Men": "men",
    "Children (Under 5)": "children_under_five",
    "Youth (15-24)": "youth_15_24",
    "Elderly (60+)": "elderly_60_plus",
    "Women of Reproductive Age (15-49)": "women_of_reproductive_age_15_49",
}


def filter_population_dataset(dataset_name, force=False):
    """
    Filter population dataset to only points within country boundaries.
    This is the PRIMARY cache that the app uses.
    
    Args:
        dataset_name: Internal name of the dataset (e.g., 'general', 'women')
        force: If True, overwrite existing cache
        
    Returns:
        DataFrame if successful, None otherwise
    """
    print(f"\n{'='*70}")
    print(f"PHASE 1: Filtering {dataset_name} to country boundaries")
    print(f"{'='*70}")
    
    # File paths
    input_file = f"data/mwi_{dataset_name}_2020.csv"
    output_file = f"data/.cache/mwi_{dataset_name}_2020_filtered.parquet"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return None
    
    # Check if output already exists
    if os.path.exists(output_file) and not force:
        print(f"✅ Cache already exists: {output_file}")
        print(f"   Use --force to regenerate")
        # Load and return for stats
        try:
            return pd.read_parquet(output_file)
        except:
            pass
    
    try:
        start_time = time.time()
        
        # Load population data
        print(f"📂 Loading {input_file}...")
        df = pd.read_csv(input_file)
        print(f"   Loaded {len(df):,} rows")
        
        # Filter out zero or very low population values
        pop_column = f"mwi_{dataset_name}_2020"
        initial_count = len(df)
        df = df[df[pop_column] > 0.5]
        filtered_low = initial_count - len(df)
        if filtered_low > 0:
            print(f"   Filtered out {filtered_low:,} low-population points")
        
        # Filter to points inside country boundary
        print(f"🗺️  Filtering points within country boundary...")
        initial_count = len(df)
        df = filter_points_in_country(df, lat_col='latitude', lon_col='longitude')
        filtered_outside = initial_count - len(df)
        if filtered_outside > 0:
            print(f"   Filtered out {filtered_outside:,} points outside boundary")
        
        # Statistics
        total_population = df[pop_column].sum()
        print(f"\n📊 Filtered Dataset Statistics:")
        print(f"   • Total points: {len(df):,}")
        print(f"   • Total population: {total_population:,.0f}")
        print(f"   • Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Save to parquet (compressed)
        print(f"\n💾 Saving to {output_file}...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_parquet(output_file, index=False, compression='snappy')
        
        # Verify file was created
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        elapsed = time.time() - start_time
        print(f"   ✅ Saved {file_size_mb:.2f} MB (took {elapsed:.1f}s)")
        
        return df
        
    except Exception as e:
        print(f"❌ Error filtering {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def assign_districts_to_dataset(dataset_name, filtered_df=None, force=False):
    """
    Assign districts to filtered population data.
    This is OPTIONAL - only needed for district breakdowns.
    
    Args:
        dataset_name: Internal name of the dataset (e.g., 'general', 'women')
        filtered_df: Pre-loaded filtered dataframe (optional)
        force: If True, overwrite existing cache
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"PHASE 2: Assigning districts for {dataset_name}")
    print(f"{'='*70}")
    
    # File paths
    filtered_file = f"data/.cache/mwi_{dataset_name}_2020_filtered.parquet"
    output_file = f"data/.cache/mwi_{dataset_name}_2020_with_districts.parquet"
    
    # Check if output already exists
    if os.path.exists(output_file) and not force:
        print(f"✅ Cache already exists: {output_file}")
        print(f"   Use --force to regenerate")
        return True
    
    try:
        start_time = time.time()
        
        # Load filtered data if not provided
        if filtered_df is None:
            if not os.path.exists(filtered_file):
                print(f"❌ Filtered data not found: {filtered_file}")
                print(f"   Run Phase 1 first (filtering)")
                return False
            
            print(f"📂 Loading filtered data from {filtered_file}...")
            filtered_df = pd.read_parquet(filtered_file)
            print(f"   Loaded {len(filtered_df):,} rows")
        
        # Assign districts (expensive operation)
        print(f"🗺️  Assigning districts to {len(filtered_df):,} population points...")
        print("   (This uses geopandas spatial join with R-tree index - fast!)")
        df = assign_districts_to_dataframe(filtered_df, lat_col='latitude', lon_col='longitude')
        
        # Count assignments
        assigned = df['assigned_district'].notna().sum()
        print(f"   ✅ Successfully assigned {assigned:,} points to districts")
        if assigned < len(df):
            print(f"   ⚠️  {len(df) - assigned:,} points could not be assigned")
        
        # Save to parquet
        print(f"\n💾 Saving to {output_file}...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_parquet(output_file, index=False, compression='snappy')
        
        # Verify file was created
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        elapsed = time.time() - start_time
        print(f"   ✅ Saved {file_size_mb:.2f} MB (took {elapsed:.1f}s)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error assigning districts for {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to process datasets"""
    parser = argparse.ArgumentParser(description='Pre-populate cache for GAIA Planning app')
    parser.add_argument('--with-districts', action='store_true', 
                       help='Also assign districts (slower, optional)')
    parser.add_argument('--only', type=str, metavar='DATASET',
                       help='Process only one dataset (e.g., general, women)')
    parser.add_argument('--force', action='store_true',
                       help='Force regeneration of existing caches')
    parser.add_argument('--skip-filter', action='store_true',
                       help='Skip filtering phase (only assign districts)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("GAIA Planning Cache Pre-population")
    print("="*70)
    
    # Check dependencies
    try:
        import geopandas
        import shapely
        import pyarrow
        print("✅ Required packages available")
    except ImportError as e:
        print(f"❌ Missing required packages: {e}")
        print("   Please install:")
        print("   pip install geopandas shapely pyarrow")
        sys.exit(1)
    
    # Check if boundary files exist
    if not os.path.exists('data/boundaries/malawi_country.geojson'):
        print("❌ Country boundary not found!")
        print("   Please run: python download_boundaries.py")
        sys.exit(1)
    
    if args.with_districts and not os.path.exists('data/boundaries/malawi_districts.geojson'):
        print("❌ District boundaries not found!")
        print("   Please run: python download_boundaries.py")
        sys.exit(1)
    
    print("✅ Boundary files found")
    
    # Determine which datasets to process
    if args.only:
        if args.only not in POPULATION_DATASETS.values():
            print(f"❌ Unknown dataset: {args.only}")
            print(f"   Available: {', '.join(POPULATION_DATASETS.values())}")
            sys.exit(1)
        datasets_to_process = {k: v for k, v in POPULATION_DATASETS.items() if v == args.only}
    else:
        datasets_to_process = POPULATION_DATASETS
    
    print(f"\n📋 Will process {len(datasets_to_process)} dataset(s):")
    for display_name, internal_name in datasets_to_process.items():
        print(f"   • {display_name} ({internal_name})")
    
    if args.with_districts:
        print(f"\n🗺️  District assignment: ENABLED")
    else:
        print(f"\n🗺️  District assignment: DISABLED (use --with-districts to enable)")
    
    print("="*70)
    
    # Process each dataset
    filter_results = {}
    district_results = {}
    
    for display_name, internal_name in datasets_to_process.items():
        # Phase 1: Filter to country boundaries
        if not args.skip_filter:
            filtered_df = filter_population_dataset(internal_name, force=args.force)
            filter_results[display_name] = filtered_df is not None
        else:
            filtered_df = None
            filter_results[display_name] = True  # Assume success if skipped
        
        # Phase 2: Assign districts (optional)
        if args.with_districts:
            success = assign_districts_to_dataset(internal_name, filtered_df, force=args.force)
            district_results[display_name] = success
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\n📊 Phase 1 - Filtered Population Data:")
    successful_filters = sum(1 for v in filter_results.values() if v)
    print(f"   ✅ Successful: {successful_filters}/{len(filter_results)}")
    if successful_filters < len(filter_results):
        print(f"   ❌ Failed:")
        for name, success in filter_results.items():
            if not success:
                print(f"      • {name}")
    
    if args.with_districts:
        print("\n🗺️  Phase 2 - District Assignments:")
        successful_districts = sum(1 for v in district_results.values() if v)
        print(f"   ✅ Successful: {successful_districts}/{len(district_results)}")
        if successful_districts < len(district_results):
            print(f"   ❌ Failed:")
            for name, success in district_results.items():
                if not success:
                    print(f"      • {name}")
    
    print("\n" + "="*70)
    if successful_filters == len(filter_results):
        print("✅ All caches created successfully!")
        print("\n🚀 You can now run the Streamlit app with full population data:")
        print("   streamlit run app.py")
        print("\n💡 The app will now show accurate population metrics without sampling!")
    else:
        print("⚠️  Some operations failed. Check errors above.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
