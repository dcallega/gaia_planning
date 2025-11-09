#!/usr/bin/env python3
"""
Mobile Clinic Planning - Scenario Comparison

Demonstrates different planning configurations side-by-side:
1. Default scenario (30km, 5 stops, 15k population threshold)
2. Conservative scenario (20km, 3 stops, 20k threshold)
3. Aggressive scenario (40km, 5 stops, 10k threshold)
4. District Hospital only scenario

Usage:
    python planning_scenarios_comparison.py
"""

from coverage_lib import CoverageAnalyzer
from mobile_clinic_planner import MobileClinicPlanner
import mobile_clinic_planner
import pandas as pd


def run_planning_scenario(scenario_name: str, 
                         max_distance_km: float,
                         stops_per_week: int,
                         min_population: float,
                         hospital_filter=None):
    """
    Run a complete planning scenario with custom parameters.
    
    Args:
        scenario_name: Name for this scenario
        max_distance_km: Maximum travel distance from hospital
        stops_per_week: Number of clinic stops per week
        min_population: Minimum population to justify a route
        hospital_filter: Optional function to filter hospitals
    
    Returns:
        Dictionary with results
    """
    print("\n" + "="*70)
    print(f"SCENARIO: {scenario_name}")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Max travel distance: {max_distance_km} km")
    print(f"  - Stops per week: {stops_per_week}")
    print(f"  - Min population: {min_population:,.0f}")
    
    # Configure parameters
    mobile_clinic_planner.MAX_TRAVEL_DISTANCE_KM = max_distance_km
    mobile_clinic_planner.STOPS_PER_WEEK = stops_per_week
    
    # Set up analyzer (reuse existing)
    # Note: In production, you'd want to cache the analyzer
    
    # Create planner
    planner = MobileClinicPlanner(analyzer)
    planner.identify_deployment_hospitals()
    
    # Apply hospital filter if provided
    initial_hospitals = len(planner.hospitals)
    if hospital_filter:
        planner.hospitals = hospital_filter(planner.hospitals)
        print(f"  - Hospitals: {len(planner.hospitals)} (filtered from {initial_hospitals})")
    else:
        print(f"  - Hospitals: {len(planner.hospitals)}")
    
    # Plan routes
    routes = planner.plan_mobile_clinic_network(
        gaps,
        max_teams=10,
        min_population_per_route=min_population
    )
    
    # Compile results
    results = {
        'scenario_name': scenario_name,
        'max_distance_km': max_distance_km,
        'stops_per_week': stops_per_week,
        'min_population': min_population,
        'n_hospitals': len(planner.hospitals),
        'n_routes': 0,
        'n_stops': 0,
        'total_population': 0,
        'avg_distance': 0,
        'routes_df': routes
    }
    
    if len(routes) > 0:
        results['n_routes'] = routes['route_id'].nunique()
        results['n_stops'] = len(routes)
        results['total_population'] = routes['population'].sum()
        results['avg_distance'] = routes['distance_from_hospital'].mean()
        
        print(f"\nResults:")
        print(f"  ✓ Routes planned: {results['n_routes']}")
        print(f"  ✓ Total stops: {results['n_stops']}")
        print(f"  ✓ Target population: {results['total_population']:,.0f}")
        print(f"  ✓ Avg distance from hospital: {results['avg_distance']:.1f} km")
        
        # Save scenario results
        filename = f"routes_{scenario_name.lower().replace(' ', '_')}.csv"
        routes.to_csv(filename, index=False)
        print(f"  ✓ Saved to: {filename}")
    else:
        print(f"\n  ✗ No viable routes found with these parameters")
    
    return results


def main():
    """Run all planning scenarios and compare results."""
    
    print("="*70)
    print("MOBILE CLINIC PLANNING - SCENARIO COMPARISON")
    print("="*70)
    
    # Set up coverage analyzer (shared across scenarios)
    print("\nInitializing coverage analysis...")
    global analyzer, gaps
    
    analyzer = CoverageAnalyzer(service_radius_km=5.0)
    analyzer.load_facilities("data/MHFR_Facilities.csv", filter_functional=True)
    analyzer.build_spatial_index()
    
    # Find coverage gaps
    print("Finding coverage gaps...")
    gaps = analyzer.find_coverage_gaps("data/mwi_general_2020.csv", max_distance_km=10.0)
    print(f"Found {len(gaps):,} population points in gaps")
    
    # ==========================================
    # SCENARIO 1: Default (Balanced)
    # ==========================================
    scenario1 = run_planning_scenario(
        scenario_name="Default Balanced",
        max_distance_km=30.0,
        stops_per_week=5,
        min_population=15000,
        hospital_filter=None
    )
    
    # ==========================================
    # SCENARIO 2: Conservative (Limited Resources)
    # ==========================================
    scenario2 = run_planning_scenario(
        scenario_name="Conservative Limited",
        max_distance_km=20.0,  # Shorter travel distance
        stops_per_week=3,       # Only 3 days per week
        min_population=20000,   # Higher population requirement
        hospital_filter=None
    )
    
    # ==========================================
    # SCENARIO 3: Aggressive Coverage
    # ==========================================
    scenario3 = run_planning_scenario(
        scenario_name="Aggressive Coverage",
        max_distance_km=40.0,  # Extended travel distance
        stops_per_week=5,       # Full week
        min_population=10000,   # Lower population threshold
        hospital_filter=None
    )
    
    # ==========================================
    # SCENARIO 4: District Hospitals Only
    # ==========================================
    def district_hospital_filter(hospitals_df):
        return hospitals_df[
            hospitals_df['type'].isin(['District Hospital', 'Central Hospital'])
        ].copy()
    
    scenario4 = run_planning_scenario(
        scenario_name="District Hospitals Only",
        max_distance_km=30.0,
        stops_per_week=5,
        min_population=15000,
        hospital_filter=district_hospital_filter
    )
    
    # ==========================================
    # SCENARIO 5: Free Facilities Only (Gov't)
    # ==========================================
    print("\n" + "="*70)
    print("SCENARIO: Free Facilities Only")
    print("="*70)
    print("Re-analyzing with government facilities only...")
    
    analyzer_free = CoverageAnalyzer(service_radius_km=4.5)
    analyzer_free.load_facilities("data/MHFR_Facilities.csv", filter_functional=True)
    analyzer_free.facilities_df = analyzer_free.facilities_df[
        analyzer_free.facilities_df['ownership'] == 'Government'
    ].copy()
    analyzer_free.build_spatial_index()
    
    gaps_free = analyzer_free.find_coverage_gaps("data/mwi_general_2020.csv", max_distance_km=10.0)
    print(f"Found {len(gaps_free):,} population points in gaps (free facilities only)")
    
    # Temporarily update global variables for this scenario
    analyzer_orig = analyzer
    gaps_orig = gaps
    analyzer = analyzer_free
    gaps = gaps_free
    
    scenario5 = run_planning_scenario(
        scenario_name="Free Facilities 4.5km",
        max_distance_km=25.0,
        stops_per_week=5,
        min_population=15000,
        hospital_filter=lambda df: df[df['ownership'] == 'Government'].copy()
    )
    
    # Restore original
    analyzer = analyzer_orig
    gaps = gaps_orig
    
    # ==========================================
    # COMPARISON SUMMARY
    # ==========================================
    
    scenarios = [scenario1, scenario2, scenario3, scenario4, scenario5]
    comparison_df = pd.DataFrame([
        {
            'Scenario': s['scenario_name'],
            'Max Distance (km)': s['max_distance_km'],
            'Stops/Week': s['stops_per_week'],
            'Min Population': f"{s['min_population']:,.0f}",
            'Hospitals': s['n_hospitals'],
            'Routes': s['n_routes'],
            'Total Stops': s['n_stops'],
            'Population': f"{s['total_population']:,.0f}",
            'Avg Distance (km)': f"{s['avg_distance']:.1f}" if s['avg_distance'] > 0 else 'N/A'
        }
        for s in scenarios
    ])
    
    print("\n" + "="*70)
    print("SCENARIO COMPARISON SUMMARY")
    print("="*70)
    print()
    print(comparison_df.to_string(index=False))
    print()
    
    # Save comparison
    comparison_df.to_csv('planning_scenarios_comparison.csv', index=False)
    print("✅ Comparison saved to: planning_scenarios_comparison.csv")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    best_coverage = max(scenarios, key=lambda s: s['total_population'])
    most_routes = max(scenarios, key=lambda s: s['n_routes'])
    most_efficient = min([s for s in scenarios if s['total_population'] > 0], 
                        key=lambda s: s['avg_distance'])
    
    print(f"\n📊 Best Population Coverage:")
    print(f"   {best_coverage['scenario_name']}")
    print(f"   - Reaches {best_coverage['total_population']:,.0f} people")
    print(f"   - {best_coverage['n_routes']} routes, {best_coverage['n_stops']} stops")
    
    print(f"\n🏥 Most Routes Deployed:")
    print(f"   {most_routes['scenario_name']}")
    print(f"   - {most_routes['n_routes']} routes planned")
    
    print(f"\n⚡ Most Efficient (Shortest Travel):")
    print(f"   {most_efficient['scenario_name']}")
    print(f"   - Average {most_efficient['avg_distance']:.1f} km from hospital")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - routes_default_balanced.csv")
    print("  - routes_conservative_limited.csv")
    print("  - routes_aggressive_coverage.csv")
    print("  - routes_district_hospitals_only.csv")
    print("  - routes_free_facilities_4.5km.csv")
    print("  - planning_scenarios_comparison.csv")
    
    print("\n💡 Tip: Review each scenario's routes and choose based on:")
    print("  - Available resources (vehicles, staff)")
    print("  - Population impact goals")
    print("  - Operational feasibility")
    print("  - Budget constraints")


if __name__ == "__main__":
    main()

