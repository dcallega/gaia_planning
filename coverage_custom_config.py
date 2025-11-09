#!/usr/bin/env python3
"""
Custom Coverage Configuration Example

Demonstrates how to configure coverage analysis with custom parameters:
- Filter by ownership (free sites only)
- Custom service radius (4.5 km)
- Filter by status (functional only)

Usage:
    python coverage_custom_config.py
"""

from coverage_lib import CoverageAnalyzer
import pandas as pd


def main():
    """Run coverage analysis with custom configuration."""
    
    print("="*70)
    print("CUSTOM COVERAGE ANALYSIS")
    print("Configuration:")
    print("  - Service radius: 4.5 km")
    print("  - Facilities: FREE functional sites only")
    print("="*70)
    
    # Initialize analyzer with custom service radius
    analyzer = CoverageAnalyzer(service_radius_km=4.5)
    
    # Load all facilities first
    print("\nStep 1: Loading facilities...")
    analyzer.load_facilities("data/MHFR_Facilities.csv", filter_functional=True)
    
    # Check unique ownership types to understand the data
    print("\nAvailable ownership types:")
    ownership_types = analyzer.facilities_df['ownership'].value_counts()
    print(ownership_types)
    
    # Filter to free sites only
    # In Malawi, free sites are typically "Government" owned
    # Adjust this based on your data
    print("\nStep 2: Filtering to FREE sites only...")
    
    # You can customize this filter based on your definition of "free"
    free_ownership_types = [
        'Government',  # Government facilities are typically free
        # Add other ownership types that provide free services if applicable
    ]
    
    initial_count = len(analyzer.facilities_df)
    analyzer.facilities_df = analyzer.facilities_df[
        analyzer.facilities_df['ownership'].isin(free_ownership_types)
    ].copy()
    
    print(f"  Filtered from {initial_count} to {len(analyzer.facilities_df)} facilities")
    
    # Build spatial index with filtered facilities
    print("\nStep 3: Building spatial index...")
    analyzer.build_spatial_index()
    
    # Compute coverage with custom configuration
    print("\nStep 4: Computing coverage (4.5 km radius, free sites only)...")
    coverage = analyzer.compute_basic_coverage("data/mwi_general_2020.csv")
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS - FREE SITES (4.5 km radius)")
    print("="*70)
    print(f"Facilities analyzed:     {len(analyzer.facilities_df)}")
    print(f"Total Population:        {coverage['total']:,.0f}")
    print(f"Covered:                 {coverage['covered']:,.0f} ({coverage['coverage_pct']:.2f}%)")
    print(f"Uncovered:               {coverage['uncovered']:,.0f}")
    print("="*70)
    
    # Optional: Compare with all functional facilities at 5km
    print("\n" + "="*70)
    print("COMPARISON: Run with ALL functional facilities at 5 km")
    print("="*70)
    
    # Create a new analyzer for comparison
    analyzer_all = CoverageAnalyzer(service_radius_km=5.0)
    analyzer_all.load_facilities("data/MHFR_Facilities.csv", filter_functional=True)
    analyzer_all.build_spatial_index()
    coverage_all = analyzer_all.compute_basic_coverage("data/mwi_general_2020.csv")
    
    print(f"\nAll Functional Facilities (5 km):")
    print(f"  Facilities:  {len(analyzer_all.facilities_df)}")
    print(f"  Coverage:    {coverage_all['coverage_pct']:.2f}%")
    print(f"  Covered:     {coverage_all['covered']:,.0f} people")
    
    print(f"\nFree Facilities Only (4.5 km):")
    print(f"  Facilities:  {len(analyzer.facilities_df)}")
    print(f"  Coverage:    {coverage['coverage_pct']:.2f}%")
    print(f"  Covered:     {coverage['covered']:,.0f} people")
    
    print(f"\nDifference:")
    coverage_diff = coverage_all['coverage_pct'] - coverage['coverage_pct']
    people_diff = coverage_all['covered'] - coverage['covered']
    print(f"  Coverage:    -{coverage_diff:.2f} percentage points")
    print(f"  People:      -{people_diff:,.0f} people")
    
    # Save facility list for reference
    analyzer.facilities_df.to_csv('free_functional_facilities.csv', index=False)
    print(f"\n✅ Free facility list saved to: free_functional_facilities.csv")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

