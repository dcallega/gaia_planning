"""
Mobile Clinic Planning Module

Plans mobile clinic routes based on:
- Hospital-based deployment (teams deploy from main hospitals)
- Weekly schedule (Monday-Friday, 5 clinic stops)
- Off-road travel constraints (~1 hour drive from hospital)
- Maximizing coverage of underserved populations

Each mobile clinic = one team + one vehicle + 5 weekly stops
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from coverage_lib import CoverageAnalyzer, haversine_distance_km, get_hospitals, EARTH_RADIUS_KM

# Planning constants
OFF_ROAD_SPEED_KMH = 30  # Average off-road driving speed
MAX_TRAVEL_TIME_HOURS = 1.0  # Maximum travel time from hospital to clinic
MAX_TRAVEL_DISTANCE_KM = OFF_ROAD_SPEED_KMH * MAX_TRAVEL_TIME_HOURS  # 30 km
STOPS_PER_WEEK = 5  # Monday-Friday
SERVICE_RADIUS_KM = 5.0  # Service radius of each clinic stop


class MobileClinicPlanner:
    """Plans mobile clinic routes from hospital bases."""
    
    def __init__(self, coverage_analyzer: CoverageAnalyzer):
        """
        Initialize planner with a coverage analyzer.
        
        Args:
            coverage_analyzer: CoverageAnalyzer instance with loaded facilities
        """
        self.analyzer = coverage_analyzer
        self.hospitals = None
        self.gap_points = None
        
    def identify_deployment_hospitals(self, min_nearby_gaps: int = 5) -> pd.DataFrame:
        """
        Identify hospitals suitable for mobile clinic deployment.
        
        Args:
            min_nearby_gaps: Minimum number of gap areas near hospital
            
        Returns:
            DataFrame of hospitals with gap information
        """
        if self.analyzer.facilities_df is None:
            raise ValueError("Facilities not loaded")
        
        print("\nIdentifying deployment hospitals...")
        
        # Get all hospitals
        self.hospitals = get_hospitals(self.analyzer.facilities_df)
        print(f"Found {len(self.hospitals)} hospitals")
        
        return self.hospitals
    
    def find_optimal_clinic_locations(self, hospital_row: pd.Series, 
                                     gap_points: pd.DataFrame,
                                     n_stops: int = STOPS_PER_WEEK,
                                     min_population: float = 5000) -> pd.DataFrame:
        """
        Find optimal mobile clinic stop locations around a hospital.
        
        Uses clustering to identify high-density underserved areas within
        travel distance of the hospital.
        
        Args:
            hospital_row: Row from hospitals DataFrame
            gap_points: DataFrame with coverage gaps (from find_coverage_gaps)
            n_stops: Number of clinic stops to plan (default 5)
            min_population: Minimum population to justify a stop
            
        Returns:
            DataFrame with proposed clinic stop locations
        """
        hospital_lat = hospital_row['lat']
        hospital_lon = hospital_row['lon']
        hospital_name = hospital_row['name']
        
        print(f"\nPlanning route from {hospital_name}...")
        
        # Filter gap points within travel distance
        gap_points = gap_points.copy()
        gap_points['distance_from_hospital'] = gap_points.apply(
            lambda row: haversine_distance_km(hospital_lat, hospital_lon, 
                                             row['latitude'], row['longitude']),
            axis=1
        )
        
        nearby_gaps = gap_points[gap_points['distance_from_hospital'] <= MAX_TRAVEL_DISTANCE_KM].copy()
        
        if len(nearby_gaps) == 0:
            print(f"  No gaps within {MAX_TRAVEL_DISTANCE_KM} km")
            return pd.DataFrame()
        
        print(f"  Found {len(nearby_gaps):,} population points within {MAX_TRAVEL_DISTANCE_KM} km")
        print(f"  Total underserved population: {nearby_gaps['mwi_general_2020'].sum():,.0f}")
        
        # Cluster nearby gaps to find high-density areas
        coords = nearby_gaps[['latitude', 'longitude']].values
        
        # Use DBSCAN for clustering (density-based)
        # eps is in radians for ball_tree metric
        eps_km = 5.0  # Cluster points within 5km
        eps_rad = eps_km / EARTH_RADIUS_KM
        
        coords_rad = np.deg2rad(coords)
        clustering = DBSCAN(eps=eps_rad, min_samples=50, metric='haversine').fit(coords_rad)
        
        nearby_gaps['cluster'] = clustering.labels_
        
        # Analyze clusters (exclude noise points with label -1)
        clusters = nearby_gaps[nearby_gaps['cluster'] >= 0].groupby('cluster').agg({
            'latitude': 'mean',
            'longitude': 'mean',
            'mwi_general_2020': 'sum',
            'distance_from_hospital': 'mean'
        }).reset_index()
        
        clusters.columns = ['cluster_id', 'lat', 'lon', 'population', 'distance_from_hospital']
        
        # Sort by population and filter by minimum
        clusters = clusters[clusters['population'] >= min_population]
        clusters = clusters.sort_values('population', ascending=False)
        
        print(f"  Identified {len(clusters)} high-density gap clusters")
        
        # Select top N clusters for clinic stops
        selected_stops = clusters.head(n_stops).copy()
        
        # Add metadata
        selected_stops['hospital_code'] = hospital_row['code']
        selected_stops['hospital_name'] = hospital_name
        selected_stops['hospital_lat'] = hospital_lat
        selected_stops['hospital_lon'] = hospital_lon
        selected_stops['stop_number'] = range(1, len(selected_stops) + 1)
        selected_stops['estimated_coverage'] = selected_stops['population']  # Rough estimate
        
        return selected_stops
    
    def estimate_new_coverage(self, proposed_stops: pd.DataFrame, 
                             pop_csv: str,
                             chunksize: int = 200_000) -> Dict:
        """
        Estimate the coverage impact of proposed mobile clinic stops.
        
        Args:
            proposed_stops: DataFrame with proposed clinic locations
            pop_csv: Path to population CSV
            chunksize: Rows to process at once
            
        Returns:
            Dictionary with coverage estimates
        """
        if len(proposed_stops) == 0:
            return {'new_coverage': 0, 'population': 0}
        
        print(f"\nEstimating coverage impact of {len(proposed_stops)} proposed stops...")
        
        # Build BallTree for proposed stops
        coords_rad = np.deg2rad(proposed_stops[['lat', 'lon']].to_numpy(dtype=np.float64))
        new_tree = BallTree(coords_rad, metric='haversine')
        
        service_radius_rad = SERVICE_RADIUS_KM / EARTH_RADIUS_KM
        
        # Compute coverage from proposed stops
        new_coverage = 0.0
        total_checked = 0.0
        chunks_processed = 0
        
        cols = ["latitude", "longitude", "mwi_general_2020"]
        
        for chunk in pd.read_csv(pop_csv, usecols=cols, chunksize=chunksize):
            chunks_processed += 1
            chunk = chunk.dropna(subset=["latitude", "longitude", "mwi_general_2020"])
            
            if chunk.empty:
                continue
            
            coords_rad = np.deg2rad(chunk[["latitude", "longitude"]].to_numpy(dtype=np.float64))
            people = chunk["mwi_general_2020"].to_numpy(dtype=float)
            
            # Check if covered by existing facilities
            dist_existing, _ = self.analyzer.tree.query(coords_rad, k=1)
            covered_by_existing = dist_existing.ravel() <= self.analyzer.service_radius_rad
            
            # Check if covered by new clinics
            dist_new, _ = new_tree.query(coords_rad, k=1)
            covered_by_new = dist_new.ravel() <= service_radius_rad
            
            # Count people who would be newly covered (not covered before, covered now)
            newly_covered = ~covered_by_existing & covered_by_new
            
            new_coverage += people[newly_covered].sum()
            total_checked += people.sum()
            
            if chunks_processed % 10 == 0:
                print(f"  Processed {chunks_processed} chunks...")
        
        return {
            'new_coverage': new_coverage,
            'total_population': total_checked,
            'new_coverage_pct': (new_coverage / total_checked * 100) if total_checked > 0 else 0
        }
    
    def plan_mobile_clinic_network(self, gap_points: pd.DataFrame,
                                   max_teams: Optional[int] = None,
                                   min_population_per_route: float = 10000) -> pd.DataFrame:
        """
        Plan a network of mobile clinics across multiple hospitals.
        
        Args:
            gap_points: DataFrame with coverage gaps
            max_teams: Maximum number of teams to deploy (None = no limit)
            min_population_per_route: Minimum population to justify a route
            
        Returns:
            DataFrame with all proposed mobile clinic routes
        """
        if self.hospitals is None:
            self.identify_deployment_hospitals()
        
        print("\n" + "="*70)
        print("PLANNING MOBILE CLINIC NETWORK")
        print("="*70)
        
        all_routes = []
        
        # For each hospital, find optimal stops
        for idx, hospital in self.hospitals.iterrows():
            stops = self.find_optimal_clinic_locations(hospital, gap_points)
            
            if len(stops) > 0:
                # Check if route serves enough population
                total_pop = stops['population'].sum()
                
                if total_pop >= min_population_per_route:
                    all_routes.append(stops)
                    print(f"  ✓ Route planned: {len(stops)} stops, ~{total_pop:,.0f} people")
                else:
                    print(f"  ✗ Route rejected: only {total_pop:,.0f} people (< {min_population_per_route:,.0f})")
            
            # Check if we've reached max teams
            if max_teams and len(all_routes) >= max_teams:
                print(f"\n  Reached maximum of {max_teams} teams")
                break
        
        if all_routes:
            routes_df = pd.concat(all_routes, ignore_index=True)
            routes_df['route_id'] = routes_df.groupby('hospital_code').ngroup()
            return routes_df
        else:
            return pd.DataFrame()
    
    def generate_route_summary(self, routes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for planned routes.
        
        Args:
            routes_df: DataFrame with planned routes
            
        Returns:
            Summary DataFrame by hospital
        """
        if len(routes_df) == 0:
            return pd.DataFrame()
        
        summary = routes_df.groupby(['hospital_code', 'hospital_name']).agg({
            'stop_number': 'count',
            'population': 'sum',
            'distance_from_hospital': 'mean'
        }).reset_index()
        
        summary.columns = ['hospital_code', 'hospital_name', 'n_stops', 
                          'total_population', 'avg_distance_km']
        
        return summary


def print_planning_summary(routes_df: pd.DataFrame, coverage_impact: Dict):
    """
    Print formatted summary of mobile clinic planning results.
    
    Args:
        routes_df: DataFrame with planned routes
        coverage_impact: Dictionary from estimate_new_coverage()
    """
    print("\n" + "="*70)
    print("MOBILE CLINIC PLANNING SUMMARY")
    print("="*70)
    
    if len(routes_df) == 0:
        print("No routes planned")
        return
    
    n_routes = routes_df['route_id'].nunique()
    n_stops = len(routes_df)
    total_pop = routes_df['population'].sum()
    
    print(f"\nProposed Mobile Clinic Teams: {n_routes}")
    print(f"Total Clinic Stops:           {n_stops}")
    print(f"Target Population:            {total_pop:,.0f}")
    
    if coverage_impact:
        print(f"\nEstimated New Coverage:")
        print(f"  Additional people covered:  {coverage_impact['new_coverage']:,.0f}")
        print(f"  Coverage increase:          {coverage_impact['new_coverage_pct']:.2f}%")
    
    print("\n" + "="*70)
    print("ROUTE DETAILS")
    print("="*70)
    
    for route_id in sorted(routes_df['route_id'].unique()):
        route = routes_df[routes_df['route_id'] == route_id]
        hospital_name = route.iloc[0]['hospital_name']
        n_stops = len(route)
        pop = route['population'].sum()
        
        print(f"\nRoute {route_id + 1}: {hospital_name}")
        print(f"  Stops: {n_stops}")
        print(f"  Population: {pop:,.0f}")
        
        for _, stop in route.iterrows():
            print(f"    Stop {stop['stop_number']}: ({stop['lat']:.4f}, {stop['lon']:.4f}) "
                  f"- {stop['population']:,.0f} people - {stop['distance_from_hospital']:.1f} km")

