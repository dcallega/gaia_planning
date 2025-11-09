"""
Coverage Analysis Library

Core functions for analyzing healthcare facility coverage and planning mobile clinic routes.
Provides reusable methods for:
- Loading facilities and population data
- Building spatial indices
- Computing coverage and overlap
- Identifying gaps and opportunities
- Planning mobile clinic deployments

Author: GAIA Planning Team
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from typing import Tuple, Dict, List, Optional
from collections import defaultdict

# Constants
EARTH_RADIUS_KM = 6371.0088
DEFAULT_SERVICE_RADIUS_KM = 5.0


class CoverageAnalyzer:
    """Main class for healthcare coverage analysis."""
    
    def __init__(self, service_radius_km: float = DEFAULT_SERVICE_RADIUS_KM):
        """
        Initialize the coverage analyzer.
        
        Args:
            service_radius_km: Service radius in kilometers (default 5km)
        """
        self.service_radius_km = service_radius_km
        self.service_radius_rad = service_radius_km / EARTH_RADIUS_KM
        self.facilities_df = None
        self.tree = None
        
    def load_facilities(self, path: str, filter_functional: bool = True) -> pd.DataFrame:
        """
        Load health facilities from CSV.
        
        Args:
            path: Path to facilities CSV file
            filter_functional: If True, only include functional facilities
            
        Returns:
            DataFrame with facility information
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
        
        # Filter by status if requested
        if filter_functional:
            fac = fac[fac["status"] == "Functional"]
            print(f"  Filtered to functional facilities only")
        
        # Add facility index
        fac = fac.reset_index(drop=True)
        fac['facility_id'] = fac.index
        
        print(f"Loaded {len(fac)} facilities")
        
        self.facilities_df = fac
        return fac
    
    def build_spatial_index(self, facilities_df: Optional[pd.DataFrame] = None) -> BallTree:
        """
        Build BallTree spatial index for fast queries.
        
        Args:
            facilities_df: DataFrame with facilities (uses self.facilities_df if None)
            
        Returns:
            BallTree spatial index
        """
        if facilities_df is None:
            facilities_df = self.facilities_df
        
        if facilities_df is None:
            raise ValueError("No facilities loaded. Call load_facilities() first.")
        
        coords_rad = np.deg2rad(facilities_df[["lat", "lon"]].to_numpy(dtype=np.float64))
        self.tree = BallTree(coords_rad, metric="haversine")
        
        return self.tree
    
    def compute_basic_coverage(self, pop_csv: str, chunksize: int = 200_000) -> Dict:
        """
        Compute basic coverage statistics (% population within service radius).
        
        Args:
            pop_csv: Path to population CSV file
            chunksize: Rows to process at once
            
        Returns:
            Dictionary with coverage statistics
        """
        if self.tree is None:
            raise ValueError("Spatial index not built. Call build_spatial_index() first.")
        
        print(f"\nComputing coverage (radius: {self.service_radius_km} km)...")
        
        covered_people = 0.0
        total_people = 0.0
        chunks_processed = 0
        
        cols = ["latitude", "longitude", "mwi_general_2020"]
        
        for chunk in pd.read_csv(pop_csv, usecols=cols, chunksize=chunksize):
            chunks_processed += 1
            chunk = chunk.dropna(subset=["latitude", "longitude", "mwi_general_2020"])
            
            if chunk.empty:
                continue
            
            coords_rad = np.deg2rad(chunk[["latitude", "longitude"]].to_numpy(dtype=np.float64))
            people = chunk["mwi_general_2020"].to_numpy(dtype=float)
            
            # Find nearest facility distance
            dist_rad, _ = self.tree.query(coords_rad, k=1)
            dist_rad = dist_rad.ravel()
            
            total_people += people.sum()
            covered_people += people[dist_rad <= self.service_radius_rad].sum()
            
            if chunks_processed % 10 == 0:
                print(f"  Processed {chunks_processed} chunks...")
        
        coverage_pct = (covered_people / total_people * 100) if total_people > 0 else 0
        
        return {
            'covered': covered_people,
            'uncovered': total_people - covered_people,
            'total': total_people,
            'coverage_pct': coverage_pct
        }
    
    def compute_overlap_analysis(self, pop_csv: str, chunksize: int = 200_000) -> Dict:
        """
        Analyze coverage overlap (how many facilities cover each population point).
        
        Args:
            pop_csv: Path to population CSV file
            chunksize: Rows to process at once
            
        Returns:
            Dictionary with overlap statistics
        """
        if self.tree is None:
            raise ValueError("Spatial index not built. Call build_spatial_index() first.")
        
        print(f"\nAnalyzing coverage overlap...")
        
        coverage_counts = defaultdict(float)
        facility_coverage = defaultdict(float)
        facility_unique_coverage = defaultdict(float)
        total_people = 0.0
        chunks_processed = 0
        
        cols = ["latitude", "longitude", "mwi_general_2020"]
        
        for chunk in pd.read_csv(pop_csv, usecols=cols, chunksize=chunksize):
            chunks_processed += 1
            chunk = chunk.dropna(subset=["latitude", "longitude", "mwi_general_2020"])
            
            if chunk.empty:
                continue
            
            coords_rad = np.deg2rad(chunk[["latitude", "longitude"]].to_numpy(dtype=np.float64))
            people = chunk["mwi_general_2020"].to_numpy(dtype=float)
            
            # Find ALL facilities within radius
            indices_list = self.tree.query_radius(coords_rad, r=self.service_radius_rad)
            
            for indices, pop in zip(indices_list, people):
                n_facilities = len(indices)
                total_people += pop
                coverage_counts[n_facilities] += pop
                
                for facility_idx in indices:
                    facility_coverage[facility_idx] += pop
                
                if n_facilities == 1:
                    facility_unique_coverage[indices[0]] += pop
            
            if chunks_processed % 10 == 0:
                print(f"  Processed {chunks_processed} chunks...")
        
        return {
            'total_population': total_people,
            'coverage_by_count': coverage_counts,
            'facility_coverage': facility_coverage,
            'facility_unique_coverage': facility_unique_coverage
        }
    
    def identify_redundant_facilities(self, overlap_results: Dict, 
                                     redundancy_threshold: float = 0.9) -> pd.DataFrame:
        """
        Identify facilities with high redundancy (most coverage overlaps with others).
        
        Args:
            overlap_results: Results from compute_overlap_analysis()
            redundancy_threshold: Threshold for considering a facility redundant (0-1)
            
        Returns:
            DataFrame of redundant facilities sorted by redundancy percentage
        """
        fac_coverage = overlap_results['facility_coverage']
        fac_unique = overlap_results['facility_unique_coverage']
        
        redundant = []
        for idx, row in self.facilities_df.iterrows():
            total = fac_coverage.get(idx, 0)
            unique = fac_unique.get(idx, 0)
            
            if total > 0:
                redundancy_pct = (total - unique) / total
                
                if redundancy_pct >= redundancy_threshold:
                    redundant.append({
                        'facility_id': idx,
                        'code': row['code'],
                        'name': row['name'],
                        'type': row['type'],
                        'district': row['district'],
                        'lat': row['lat'],
                        'lon': row['lon'],
                        'total_coverage': total,
                        'unique_coverage': unique,
                        'redundancy_pct': redundancy_pct * 100
                    })
        
        return pd.DataFrame(redundant).sort_values('redundancy_pct', ascending=False)
    
    def identify_critical_facilities(self, overlap_results: Dict, 
                                    min_unique_coverage: float = 10000) -> pd.DataFrame:
        """
        Identify critical facilities (serve populations no one else does).
        
        Args:
            overlap_results: Results from compute_overlap_analysis()
            min_unique_coverage: Minimum unique population to be considered critical
            
        Returns:
            DataFrame of critical facilities sorted by unique coverage
        """
        fac_coverage = overlap_results['facility_coverage']
        fac_unique = overlap_results['facility_unique_coverage']
        
        critical = []
        for idx, row in self.facilities_df.iterrows():
            unique = fac_unique.get(idx, 0)
            
            if unique >= min_unique_coverage:
                critical.append({
                    'facility_id': idx,
                    'code': row['code'],
                    'name': row['name'],
                    'type': row['type'],
                    'district': row['district'],
                    'lat': row['lat'],
                    'lon': row['lon'],
                    'unique_coverage': unique,
                    'total_coverage': fac_coverage.get(idx, 0)
                })
        
        return pd.DataFrame(critical).sort_values('unique_coverage', ascending=False)
    
    def find_coverage_gaps(self, pop_csv: str, 
                          max_distance_km: float = 10.0,
                          chunksize: int = 200_000) -> pd.DataFrame:
        """
        Identify population points far from any facility (coverage gaps).
        
        Args:
            pop_csv: Path to population CSV file
            max_distance_km: Consider gaps beyond this distance
            chunksize: Rows to process at once
            
        Returns:
            DataFrame with gap locations and population density
        """
        if self.tree is None:
            raise ValueError("Spatial index not built. Call build_spatial_index() first.")
        
        print(f"\nIdentifying coverage gaps (>{max_distance_km} km from any facility)...")
        
        max_distance_rad = max_distance_km / EARTH_RADIUS_KM
        gaps = []
        chunks_processed = 0
        
        cols = ["latitude", "longitude", "mwi_general_2020"]
        
        for chunk in pd.read_csv(pop_csv, usecols=cols, chunksize=chunksize):
            chunks_processed += 1
            chunk = chunk.dropna(subset=["latitude", "longitude", "mwi_general_2020"])
            
            if chunk.empty:
                continue
            
            coords_rad = np.deg2rad(chunk[["latitude", "longitude"]].to_numpy(dtype=np.float64))
            
            # Find distance to nearest facility
            dist_rad, _ = self.tree.query(coords_rad, k=1)
            dist_rad = dist_rad.ravel()
            dist_km = dist_rad * EARTH_RADIUS_KM
            
            # Filter to gaps
            gap_mask = dist_rad > max_distance_rad
            
            if gap_mask.any():
                gap_chunk = chunk[gap_mask].copy()
                gap_chunk['distance_to_nearest_km'] = dist_km[gap_mask]
                gaps.append(gap_chunk)
            
            if chunks_processed % 10 == 0:
                print(f"  Processed {chunks_processed} chunks...")
        
        if gaps:
            gaps_df = pd.concat(gaps, ignore_index=True)
            print(f"Found {len(gaps_df):,} population points in gaps")
            return gaps_df
        else:
            return pd.DataFrame()


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate haversine distance between two points in kilometers.
    
    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)
        
    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1_rad, lon1_rad = np.deg2rad([lat1, lon1])
    lat2_rad, lon2_rad = np.deg2rad([lat2, lon2])
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return c * EARTH_RADIUS_KM


def get_hospitals(facilities_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract hospital facilities (potential mobile clinic deployment bases).
    
    Args:
        facilities_df: DataFrame with all facilities
        
    Returns:
        DataFrame with only hospitals
    """
    hospital_types = ['Hospital', 'District Hospital', 'Central Hospital']
    hospitals = facilities_df[facilities_df['type'].isin(hospital_types)].copy()
    return hospitals

