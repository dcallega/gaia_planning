#!/usr/bin/env python3
"""
Mobile Clinic Planning Workflow Example

Demonstrates the complete workflow:
1. Analyze current facility coverage
2. Identify redundant facilities (over-served areas)
3. Identify critical facilities (must preserve)
4. Find coverage gaps
5. Plan mobile clinic routes from hospitals

Usage:
    python planning_workflow_example.py
"""

from coverage_lib import CoverageAnalyzer
from mobile_clinic_planner import MobileClinicPlanner, print_planning_summary
import pandas as pd
from data_utils import ensure_population_csv

# Configuration
FACILITIES_PATH = "data/MHFR_Facilities.csv"
POPULATION_PATH = str(ensure_population_csv("general"))
SERVICE_RADIUS_KM = 5.0
MAX_MOBILE_TEAMS = 10  # Maximum number of mobile clinic teams to plan


def main():
    """Run the complete planning workflow."""
    
    print("="*70)
    print("GAIA MOBILE CLINIC PLANNING WORKFLOW")
    print("="*70)
    
    # Step 1: Initialize analyzer and load data
    print("\n" + "="*70)
    print("STEP 1: LOAD FACILITIES AND BUILD SPATIAL INDEX")
    print("="*70)
    
    analyzer = CoverageAnalyzer(service_radius_km=SERVICE_RADIUS_KM)
    analyzer.load_facilities(FACILITIES_PATH, filter_functional=True)
    analyzer.build_spatial_index()
    
    # Step 2: Compute basic coverage
    print("\n" + "="*70)
    print("STEP 2: ANALYZE CURRENT COVERAGE")
    print("="*70)
    
    basic_coverage = analyzer.compute_basic_coverage(POPULATION_PATH)
    
    print(f"\nCurrent Coverage Statistics:")
    print(f"  Total Population:       {basic_coverage['total']:,.0f}")
    print(f"  Covered:                {basic_coverage['covered']:,.0f} ({basic_coverage['coverage_pct']:.2f}%)")
    print(f"  Uncovered:              {basic_coverage['uncovered']:,.0f}")
    
    # Step 3: Analyze overlap (identify redundancy)
    print("\n" + "="*70)
    print("STEP 3: ANALYZE FACILITY OVERLAP AND REDUNDANCY")
    print("="*70)
    
    overlap_results = analyzer.compute_overlap_analysis(POPULATION_PATH)
    
    # Identify redundant facilities
    redundant = analyzer.identify_redundant_facilities(overlap_results, redundancy_threshold=0.95)
    
    print(f"\nRedundancy Analysis:")
    print(f"  Facilities with >95% overlap: {len(redundant)}")
    
    if len(redundant) > 0:
        print(f"\nTop 10 Most Redundant Facilities:")
        print("-" * 70)
        for idx, row in redundant.head(10).iterrows():
            print(f"  {row['name'][:40]:<40} {row['redundancy_pct']:>6.1f}% overlap")
        
        # Save redundant facilities list
        redundant.to_csv('redundant_facilities.csv', index=False)
        print(f"\n✅ Full list saved to: redundant_facilities.csv")
    
    # Identify critical facilities
    critical = analyzer.identify_critical_facilities(overlap_results, min_unique_coverage=10000)
    
    print(f"\nCritical Facilities (unique coverage >10,000):")
    print(f"  Count: {len(critical)}")
    
    if len(critical) > 0:
        print(f"\nTop 10 Most Critical Facilities:")
        print("-" * 70)
        for idx, row in critical.head(10).iterrows():
            print(f"  {row['name'][:40]:<40} {row['unique_coverage']:>10,.0f} people")
        
        critical.to_csv('critical_facilities.csv', index=False)
        print(f"\n✅ Critical facilities saved to: critical_facilities.csv")
    
    # Step 4: Find coverage gaps
    print("\n" + "="*70)
    print("STEP 4: IDENTIFY COVERAGE GAPS")
    print("="*70)
    
    gap_points = analyzer.find_coverage_gaps(POPULATION_PATH, max_distance_km=10.0)
    
    if len(gap_points) > 0:
        total_gap_population = gap_points['mwi_general_2020'].sum()
        avg_distance = gap_points['distance_to_nearest_km'].mean()
        
        print(f"\nGap Analysis:")
        print(f"  Population points >10km from facility: {len(gap_points):,}")
        print(f"  Total underserved population:           {total_gap_population:,.0f}")
        print(f"  Average distance to nearest facility:   {avg_distance:.1f} km")
        
        # Save gap points for further analysis
        gap_points.to_csv('coverage_gaps.csv', index=False)
        print(f"\n✅ Gap locations saved to: coverage_gaps.csv")
    else:
        print("\n⚠️  No significant coverage gaps found (all population within 10km)")
        print("Consider reducing max_distance_km threshold for more granular analysis")
        return
    
    # Step 5: Plan mobile clinic routes
    print("\n" + "="*70)
    print("STEP 5: PLAN MOBILE CLINIC ROUTES")
    print("="*70)
    
    planner = MobileClinicPlanner(analyzer)
    planner.identify_deployment_hospitals()
    
    # Plan routes from hospitals
    routes = planner.plan_mobile_clinic_network(
        gap_points,
        max_teams=MAX_MOBILE_TEAMS,
        min_population_per_route=15000
    )
    
    if len(routes) > 0:
        # Save proposed routes
        routes.to_csv('proposed_mobile_clinic_routes.csv', index=False)
        print(f"\n✅ Proposed routes saved to: proposed_mobile_clinic_routes.csv")
        
        # Estimate coverage impact
        print("\n" + "="*70)
        print("STEP 6: ESTIMATE COVERAGE IMPACT")
        print("="*70)
        
        coverage_impact = planner.estimate_new_coverage(routes, POPULATION_PATH)
        
        # Print summary
        print_planning_summary(routes, coverage_impact)
        
        # Generate route summary by hospital
        summary = planner.generate_route_summary(routes)
        summary.to_csv('mobile_clinic_summary.csv', index=False)
        print(f"\n✅ Route summary saved to: mobile_clinic_summary.csv")
        
        # Calculate overall impact
        new_total_coverage = basic_coverage['covered'] + coverage_impact['new_coverage']
        new_coverage_pct = (new_total_coverage / basic_coverage['total'] * 100)
        
        print("\n" + "="*70)
        print("OVERALL IMPACT")
        print("="*70)
        print(f"Current coverage:        {basic_coverage['coverage_pct']:.2f}%")
        print(f"With mobile clinics:     {new_coverage_pct:.2f}%")
        print(f"Coverage increase:       {new_coverage_pct - basic_coverage['coverage_pct']:.2f} percentage points")
        print(f"Additional people:       {coverage_impact['new_coverage']:,.0f}")
        
    else:
        print("\n⚠️  No viable mobile clinic routes found")
        print("Try adjusting min_population_per_route or max_distance parameters")
    
    print("\n" + "="*70)
    print("PLANNING WORKFLOW COMPLETE")
    print("="*70)
    
    print("\nGenerated Files:")
    print("  - redundant_facilities.csv       (facilities with high overlap)")
    print("  - critical_facilities.csv        (irreplaceable facilities)")
    print("  - coverage_gaps.csv              (underserved population points)")
    print("  - proposed_mobile_clinic_routes.csv  (planned clinic stops)")
    print("  - mobile_clinic_summary.csv      (summary by hospital)")


if __name__ == "__main__":
    main()

