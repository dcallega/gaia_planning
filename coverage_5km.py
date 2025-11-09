#!/usr/bin/env python3
"""
Coverage Analysis - 5km radius
Computes the percentage of population living within 5km of a health facility.

Usage:
    python coverage_5km.py

This script uses:
- MHFR Facilities data (data/MHFR_Facilities.csv)
- General population data (data/mwi_general_2020.csv)
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# Constants
EARTH_RADIUS_KM = 6371.0088
RADIUS_KM = 5.0
RADIUS_RAD = RADIUS_KM / EARTH_RADIUS_KM

def load_facilities(path: str) -> BallTree:
    """
    Load health facilities and build a BallTree for fast spatial queries.
    
    Args:
        path: Path to facilities CSV file (expects LATITUDE, LONGITUDE columns)
    
    Returns:
        BallTree with facility locations in radians
    """
    print(f"Loading facilities from {path}...")
    
    # MHFR uses uppercase column names
    fac = pd.read_csv(path, usecols=["LATITUDE", "LONGITUDE"]).rename(
        columns={"LATITUDE": "lat", "LONGITUDE": "lon"}
    )
    
    # Convert to numeric and drop rows with missing/invalid coordinates
    fac["lat"] = pd.to_numeric(fac["lat"], errors="coerce")
    fac["lon"] = pd.to_numeric(fac["lon"], errors="coerce")
    fac = fac.dropna()
    
    if fac.empty:
        raise ValueError("No facilities found (empty file after dropping NaNs).")
    
    print(f"Loaded {len(fac)} facilities with valid coordinates")
    
    # Convert to radians for haversine distance
    X_fac_rad = np.deg2rad(fac[["lat", "lon"]].to_numpy(dtype=np.float64))
    
    # Build BallTree with haversine metric (works on sphere)
    tree = BallTree(X_fac_rad, metric="haversine")
    
    return tree

def compute_coverage(pop_csv: str, tree: BallTree, chunksize: int = 200_000):
    """
    Compute population coverage within RADIUS_KM of any facility.
    
    Args:
        pop_csv: Path to population CSV (expects latitude, longitude, and density columns)
        tree: BallTree of facility locations
        chunksize: Number of rows to process at once
    
    Returns:
        Tuple of (covered_people, total_people, coverage_percentage)
    """
    print(f"\nAnalyzing population coverage from {pop_csv}...")
    print(f"Using radius: {RADIUS_KM} km")
    
    covered_people = 0.0
    total_people = 0.0
    chunks_processed = 0
    
    # Population file columns: longitude, latitude, mwi_general_2020
    cols = ["latitude", "longitude", "mwi_general_2020"]
    
    # Process in chunks for memory efficiency
    for chunk in pd.read_csv(pop_csv, usecols=cols, chunksize=chunksize):
        chunks_processed += 1
        
        # Drop rows with missing coordinates or population
        chunk = chunk.dropna(subset=["latitude", "longitude", "mwi_general_2020"])
        
        if chunk.empty:
            continue
        
        # Get coordinates and convert to radians
        coords_rad = np.deg2rad(chunk[["latitude", "longitude"]].to_numpy(dtype=np.float64))
        
        # Find nearest facility distance for each population point (in radians)
        dist_rad, _ = tree.query(coords_rad, k=1)
        dist_rad = dist_rad.ravel()
        
        # Get population density for each point
        people = chunk["mwi_general_2020"].to_numpy(dtype=float)
        
        # Accumulate totals
        total_people += people.sum()
        covered_people += people[dist_rad <= RADIUS_RAD].sum()
        
        if chunks_processed % 10 == 0:
            print(f"  Processed {chunks_processed} chunks ({chunks_processed * chunksize:,} rows)...")
    
    # Calculate coverage percentage
    coverage_pct = (covered_people / total_people * 100.0) if total_people > 0 else 0.0
    
    print(f"\nProcessed {chunks_processed} total chunks")
    
    return covered_people, total_people, coverage_pct

def main():
    """Main execution function."""
    
    # File paths
    facilities_path = "data/MHFR_Facilities.csv"
    population_path = "data/mwi_general_2020.csv"
    
    try:
        # Load facilities and build spatial index
        tree = load_facilities(facilities_path)
        
        # Compute coverage
        covered, total, pct = compute_coverage(population_path, tree, chunksize=200_000)
        
        # Display results
        print("\n" + "="*60)
        print("COVERAGE ANALYSIS RESULTS")
        print("="*60)
        print(f"People covered within {RADIUS_KM} km: {covered:,.0f}")
        print(f"Total people:                        {total:,.0f}")
        print(f"Coverage:                            {pct:.2f}%")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\nError: File not found - {e}")
        print("\nMake sure the following files exist:")
        print(f"  - {facilities_path}")
        print(f"  - {population_path}")
    except Exception as e:
        print(f"\nError: {e}")
        raise

if __name__ == "__main__":
    main()

