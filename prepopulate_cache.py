#!/usr/bin/env python3
"""
Pre-populate cache for GAIA Planning app

This script pre-computes and caches expensive operations:
1. Loads all population datasets
2. Assigns districts to population points
3. Saves cached files for fast app startup

Run this script before starting the Streamlit app for the first time,
or after updating population data or district boundaries.

Usage:
    python prepopulate_cache.py
"""

import os
import sys
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

# Sample size (same as in app.py)
SAMPLE_SIZE = 50000


def process_population_dataset(dataset_name):
    """
    Process and cache a single population dataset
    
    Args:
        dataset_name: Internal name of the dataset (e.g., 'general', 'women')
    """
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*60}")
    
    # File paths
    input_file = f"data/mwi_{dataset_name}_2020.csv"
    output_file = f"data/.cache/mwi_{dataset_name}_2020_with_districts.parquet"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return False
    
    # Check if output already exists
    if os.path.exists(output_file):
        response = input(f"⚠️  Cache file already exists: {output_file}\n   Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("   Skipping...")
            return True
    
    try:
        # Load population data
        print(f"📂 Loading {input_file}...")
        df = pd.read_csv(input_file)
        print(f"   Loaded {len(df):,} rows")
        
        # Sample to reduce points (same as app.py)
        sample_size = min(SAMPLE_SIZE, len(df))
        if len(df) > sample_size:
            print(f"   Sampling {sample_size:,} points...")
            df = df.sample(n=sample_size, random_state=42)
        
        # Filter out zero or very low population values
        pop_column = f"mwi_{dataset_name}_2020"
        initial_count = len(df)
        df = df[df[pop_column] > 0.5]
        if len(df) < initial_count:
            print(f"   Filtered out {initial_count - len(df):,} low-population points")
        
        # Filter out population points outside the country boundary
        print(f"   Filtering points within country boundary...")
        initial_count = len(df)
        df = filter_points_in_country(df, lat_col='latitude', lon_col='longitude')
        if len(df) < initial_count:
            print(f"   Filtered out {initial_count - len(df):,} points outside boundary")
        
        # Assign districts (expensive operation)
        print(f"🗺️  Assigning districts to {len(df):,} population points...")
        print("   (This may take several minutes...)")
        df = assign_districts_to_dataframe(df, lat_col='latitude', lon_col='longitude')
        
        # Count assignments
        assigned = df['assigned_district'].notna().sum()
        print(f"   ✅ Successfully assigned {assigned:,} points to districts")
        if assigned < len(df):
            print(f"   ⚠️  {len(df) - assigned:,} points could not be assigned")
        
        # Save to parquet
        print(f"💾 Saving to {output_file}...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_parquet(output_file, index=False)
        
        # Verify file was created
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"   ✅ Saved {file_size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to process all datasets"""
    print("\n" + "="*60)
    print("GAIA Planning Cache Pre-population")
    print("="*60)
    
    # Check dependencies
    try:
        import shapely
        print("✅ Required packages available")
    except ImportError:
        print("❌ Missing required packages. Please install:")
        print("   pip install shapely pandas pyarrow")
        sys.exit(1)
    
    # Check if boundary files exist
    if not os.path.exists('data/boundaries/malawi_districts.geojson'):
        print("❌ District boundaries not found!")
        print("   Please run: python download_boundaries.py")
        sys.exit(1)
    
    if not os.path.exists('data/boundaries/malawi_country.geojson'):
        print("❌ Country boundary not found!")
        print("   Please run: python download_boundaries.py")
        sys.exit(1)
    
    print("✅ Boundary files found")
    
    # Process each dataset
    print(f"\nWill process {len(POPULATION_DATASETS)} datasets:")
    for display_name, internal_name in POPULATION_DATASETS.items():
        print(f"  • {display_name} ({internal_name})")
    
    print()
    response = input("Continue? (Y/n): ")
    if response.lower() == 'n':
        print("Cancelled.")
        sys.exit(0)
    
    # Process all datasets
    results = {}
    for display_name, internal_name in POPULATION_DATASETS.items():
        success = process_population_dataset(internal_name)
        results[display_name] = success
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    successful = sum(1 for v in results.values() if v)
    failed = len(results) - successful
    
    print(f"\n✅ Successful: {successful}/{len(results)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(results)}")
        print("\nFailed datasets:")
        for name, success in results.items():
            if not success:
                print(f"  • {name}")
    
    print("\n" + "="*60)
    if failed == 0:
        print("✅ All datasets cached successfully!")
        print("\nYou can now run the Streamlit app:")
        print("   streamlit run app.py")
    else:
        print("⚠️  Some datasets failed. Check errors above.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

