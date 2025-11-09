#!/usr/bin/env python3
"""
Coverage Overlap Analysis
Identifies facilities with overlapping service areas and analyzes population coverage redundancy.

This analysis helps answer:
1. Which facilities serve overlapping populations?
2. How many people are covered by multiple facilities?
3. Which areas have redundant coverage vs. gaps?
4. Where would new facilities (or mobile clinics) add the most value?

Usage:
    python coverage_overlap_analysis.py
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from collections import defaultdict

# Constants
EARTH_RADIUS_KM = 6371.0088
RADIUS_KM = 5.0
RADIUS_RAD = RADIUS_KM / EARTH_RADIUS_KM

def load_facilities_with_info(path: str) -> pd.DataFrame:
    """
    Load health facilities with their metadata.
    
    Args:
        path: Path to facilities CSV file
    
    Returns:
        DataFrame with facility information and valid coordinates
    """
    print(f"Loading facilities from {path}...")
    
    # Load relevant columns
    fac = pd.read_csv(path, usecols=[
        "CODE", "COMMON NAME", "LATITUDE", "LONGITUDE", 
        "TYPE", "OWNERSHIP", "STATUS", "DISTRICT"
    ]).rename(columns={
        "LATITUDE": "lat", 
        "LONGITUDE": "lon",
        "COMMON NAME": "name",
        "TYPE": "type",
        "OWNERSHIP": "ownership",
        "STATUS": "status",
        "DISTRICT": "district",
        "CODE": "code"
    })
    
    # Convert coordinates to numeric
    fac["lat"] = pd.to_numeric(fac["lat"], errors="coerce")
    fac["lon"] = pd.to_numeric(fac["lon"], errors="coerce")
    
    # Drop rows with missing coordinates
    fac = fac.dropna(subset=["lat", "lon"])
    
    # Filter out facilities without a type
    initial_count = len(fac)
    fac = fac[fac["type"].notna() & (fac["type"].astype(str).str.strip() != "")]
    if len(fac) < initial_count:
        print(f"  Filtered out {initial_count - len(fac)} facilities without a type")
    
    # Add facility index for tracking
    fac = fac.reset_index(drop=True)
    fac['facility_id'] = fac.index
    
    if fac.empty:
        raise ValueError("No facilities found (empty file after dropping NaNs).")
    
    print(f"Loaded {len(fac)} facilities with valid coordinates")
    
    return fac

def build_balltree(facilities_df: pd.DataFrame) -> BallTree:
    """Build BallTree from facilities dataframe."""
    X_fac_rad = np.deg2rad(facilities_df[["lat", "lon"]].to_numpy(dtype=np.float64))
    return BallTree(X_fac_rad, metric="haversine")

def analyze_facility_proximity(facilities_df: pd.DataFrame, tree: BallTree, radius_km: float = 10.0):
    """
    Find which facilities are close to each other.
    
    Args:
        facilities_df: DataFrame with facility information
        tree: BallTree of facility locations
        radius_km: Radius to consider facilities as "nearby" (default 10km = 2x service radius)
    
    Returns:
        DataFrame with facility pairs and their distances
    """
    print(f"\nAnalyzing facility proximity (within {radius_km} km)...")
    
    radius_rad = radius_km / EARTH_RADIUS_KM
    coords_rad = np.deg2rad(facilities_df[["lat", "lon"]].to_numpy(dtype=np.float64))
    
    # Find all neighbors within radius
    indices = tree.query_radius(coords_rad, r=radius_rad)
    
    # Build list of facility pairs
    pairs = []
    for i, neighbors in enumerate(indices):
        for j in neighbors:
            # Only count each pair once, exclude self
            if i < j:
                # Calculate actual distance
                dist_rad = tree.query([[coords_rad[i][0], coords_rad[i][1]]], k=1, return_distance=True)[0][0][0]
                dist_km = np.arccos(np.cos(coords_rad[i][0] - coords_rad[j][0]) * 
                                   np.cos(coords_rad[i][1] - coords_rad[j][1])) * EARTH_RADIUS_KM
                
                # Get distance between the two facilities
                pt1 = coords_rad[i]
                pt2 = coords_rad[j]
                dist_km = haversine_distance(pt1, pt2) * EARTH_RADIUS_KM
                
                pairs.append({
                    'facility_1_id': i,
                    'facility_1_name': facilities_df.iloc[i]['name'],
                    'facility_1_type': facilities_df.iloc[i]['type'],
                    'facility_1_district': facilities_df.iloc[i]['district'],
                    'facility_2_id': j,
                    'facility_2_name': facilities_df.iloc[j]['name'],
                    'facility_2_type': facilities_df.iloc[j]['type'],
                    'facility_2_district': facilities_df.iloc[j]['district'],
                    'distance_km': dist_km
                })
    
    pairs_df = pd.DataFrame(pairs)
    print(f"Found {len(pairs_df)} facility pairs within {radius_km} km")
    
    return pairs_df

def haversine_distance(coord1, coord2):
    """Calculate haversine distance between two points in radians."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return c

def analyze_population_overlap(pop_csv: str, tree: BallTree, facilities_df: pd.DataFrame, 
                               chunksize: int = 200_000):
    """
    Analyze how many facilities cover each population point.
    
    Args:
        pop_csv: Path to population CSV
        tree: BallTree of facility locations
        facilities_df: DataFrame with facility information
        chunksize: Number of rows to process at once
    
    Returns:
        Dictionary with coverage statistics
    """
    print(f"\nAnalyzing population overlap (populations covered by multiple facilities)...")
    print(f"Service radius: {RADIUS_KM} km")
    
    # Track coverage statistics
    coverage_counts = defaultdict(int)  # How many people covered by N facilities
    total_people = 0.0
    chunks_processed = 0
    
    # Track per-facility coverage
    facility_coverage = defaultdict(float)  # Total population each facility covers
    facility_unique_coverage = defaultdict(float)  # Population each facility uniquely covers
    
    cols = ["latitude", "longitude", "mwi_general_2020"]
    
    for chunk in pd.read_csv(pop_csv, usecols=cols, chunksize=chunksize):
        chunks_processed += 1
        
        chunk = chunk.dropna(subset=["latitude", "longitude", "mwi_general_2020"])
        
        if chunk.empty:
            continue
        
        coords_rad = np.deg2rad(chunk[["latitude", "longitude"]].to_numpy(dtype=np.float64))
        people = chunk["mwi_general_2020"].to_numpy(dtype=float)
        
        # Find ALL facilities within radius for each point (not just nearest)
        indices_list = tree.query_radius(coords_rad, r=RADIUS_RAD)
        
        # Process each population point
        for i, (indices, pop) in enumerate(zip(indices_list, people)):
            n_facilities = len(indices)
            total_people += pop
            coverage_counts[n_facilities] += pop
            
            # Track which facilities cover this population
            for facility_idx in indices:
                facility_coverage[facility_idx] += pop
            
            # If only one facility covers this point, it's unique coverage
            if n_facilities == 1:
                facility_unique_coverage[indices[0]] += pop
        
        if chunks_processed % 10 == 0:
            print(f"  Processed {chunks_processed} chunks ({chunks_processed * chunksize:,} rows)...")
    
    print(f"\nProcessed {chunks_processed} total chunks")
    
    # Compile results
    results = {
        'total_population': total_people,
        'coverage_by_count': coverage_counts,
        'facility_coverage': facility_coverage,
        'facility_unique_coverage': facility_unique_coverage,
        'facilities_df': facilities_df
    }
    
    return results

def print_overlap_summary(results: dict):
    """Print summary statistics about coverage overlap."""
    
    total_pop = results['total_population']
    coverage = results['coverage_by_count']
    
    print("\n" + "="*70)
    print("POPULATION COVERAGE OVERLAP ANALYSIS")
    print("="*70)
    
    # Overall coverage
    uncovered = coverage.get(0, 0)
    covered = total_pop - uncovered
    
    print(f"\nTotal Population:              {total_pop:,.0f}")
    print(f"Covered Population:            {covered:,.0f} ({covered/total_pop*100:.2f}%)")
    print(f"Uncovered Population:          {uncovered:,.0f} ({uncovered/total_pop*100:.2f}%)")
    
    # Coverage by number of facilities
    print(f"\n{'Facilities':<20} {'Population':<20} {'Percentage':<15}")
    print("-" * 55)
    
    max_facilities = max(coverage.keys()) if coverage else 0
    for n in range(0, max_facilities + 1):
        pop = coverage.get(n, 0)
        pct = (pop / total_pop * 100) if total_pop > 0 else 0
        print(f"{n:<20} {pop:>18,.0f} {pct:>12.2f}%")
    
    # Redundant coverage statistics
    multiple_coverage = sum(coverage.get(n, 0) for n in range(2, max_facilities + 1))
    if covered > 0:
        print(f"\nPopulation with Multiple Coverage: {multiple_coverage:,.0f} ({multiple_coverage/covered*100:.2f}% of covered)")
    
    # Facility-level statistics
    print("\n" + "="*70)
    print("FACILITY COVERAGE STATISTICS")
    print("="*70)
    
    fac_df = results['facilities_df']
    fac_coverage = results['facility_coverage']
    fac_unique = results['facility_unique_coverage']
    
    # Build facility summary
    facility_summary = []
    for idx, row in fac_df.iterrows():
        total_covered = fac_coverage.get(idx, 0)
        unique_covered = fac_unique.get(idx, 0)
        overlap_pct = ((total_covered - unique_covered) / total_covered * 100) if total_covered > 0 else 0
        
        facility_summary.append({
            'name': row['name'],
            'type': row['type'],
            'district': row['district'],
            'status': row['status'],
            'total_coverage': total_covered,
            'unique_coverage': unique_covered,
            'overlap_coverage': total_covered - unique_covered,
            'overlap_pct': overlap_pct
        })
    
    summary_df = pd.DataFrame(facility_summary)
    
    # Top facilities by total coverage
    print("\nTop 10 Facilities by Total Population Covered:")
    print("-" * 70)
    top_total = summary_df.nlargest(10, 'total_coverage')[['name', 'type', 'district', 'total_coverage']]
    for idx, row in top_total.iterrows():
        print(f"{row['name'][:40]:<40} {row['type']:<15} {row['total_coverage']:>12,.0f}")
    
    # Facilities with highest redundant coverage
    print("\nTop 10 Facilities with Most Overlapping Coverage:")
    print("-" * 70)
    top_overlap = summary_df.nlargest(10, 'overlap_pct')[['name', 'type', 'district', 'overlap_pct', 'total_coverage']]
    for idx, row in top_overlap.iterrows():
        print(f"{row['name'][:40]:<40} {row['overlap_pct']:>6.1f}% {row['total_coverage']:>12,.0f}")
    
    # Facilities with most unique coverage (most critical)
    print("\nTop 10 Facilities with Most Unique Coverage (Most Critical):")
    print("-" * 70)
    top_unique = summary_df.nlargest(10, 'unique_coverage')[['name', 'type', 'district', 'unique_coverage']]
    for idx, row in top_unique.iterrows():
        print(f"{row['name'][:40]:<40} {row['type']:<15} {row['unique_coverage']:>12,.0f}")
    
    # Save detailed results to CSV
    summary_df.to_csv('coverage_overlap_results.csv', index=False)
    print("\n✅ Detailed results saved to: coverage_overlap_results.csv")
    
    return summary_df

def main():
    """Main execution function."""
    
    # File paths
    facilities_path = "data/MHFR_Facilities.csv"
    population_path = "data/mwi_general_2020.csv"
    
    try:
        # Load facilities
        facilities_df = load_facilities_with_info(facilities_path)
        
        # Build spatial index
        tree = build_balltree(facilities_df)
        
        # Analyze facility proximity (which facilities are close to each other)
        proximity_df = analyze_facility_proximity(facilities_df, tree, radius_km=10.0)
        
        if len(proximity_df) > 0:
            proximity_df.to_csv('facility_proximity.csv', index=False)
            print(f"✅ Facility proximity analysis saved to: facility_proximity.csv")
            
            print(f"\nSample of nearby facility pairs:")
            print(proximity_df.head(10)[['facility_1_name', 'facility_2_name', 'distance_km']])
        
        # Analyze population coverage overlap
        results = analyze_population_overlap(population_path, tree, facilities_df)
        
        # Print and save summary
        summary_df = print_overlap_summary(results)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        
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

