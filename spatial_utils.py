"""
Spatial utility functions for associating points with districts
"""
import json
import pandas as pd
from shapely.geometry import Point, shape
from shapely.ops import unary_union

def load_district_boundaries():
    """Load district boundaries from GeoJSON"""
    with open('data/boundaries/malawi_districts.geojson', 'r') as f:
        return json.load(f)

def load_country_boundary():
    """Load country boundary from GeoJSON"""
    with open('data/boundaries/malawi_country.geojson', 'r') as f:
        return json.load(f)

def point_to_district(lat, lon, district_geojson):
    """
    Find which district a point (lat, lon) belongs to
    
    Args:
        lat: Latitude
        lon: Longitude
        district_geojson: GeoJSON data with district boundaries
        
    Returns:
        District name or None if not found
    """
    if pd.isna(lat) or pd.isna(lon):
        return None
    
    point = Point(lon, lat)  # shapely uses (x, y) = (lon, lat)
    
    for feature in district_geojson['features']:
        polygon = shape(feature['geometry'])
        if polygon.contains(point):
            # Return the district name
            return feature['properties'].get('shapeName', None)
    
    return None

def assign_districts_to_dataframe(df, lat_col='latitude', lon_col='longitude'):
    """
    Assign districts to all rows in a dataframe based on coordinates
    
    Args:
        df: DataFrame with latitude and longitude columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        
    Returns:
        DataFrame with added 'assigned_district' column
    """
    district_geojson = load_district_boundaries()
    
    df = df.copy()
    df['assigned_district'] = df.apply(
        lambda row: point_to_district(row[lat_col], row[lon_col], district_geojson),
        axis=1
    )
    
    return df

def get_district_stats(df, district_col='district'):
    """
    Get statistics about data points by district
    
    Args:
        df: DataFrame with district information
        district_col: Name of district column
        
    Returns:
        DataFrame with district statistics
    """
    stats = df.groupby(district_col).agg({
        district_col: 'count'
    }).rename(columns={district_col: 'count'})
    
    return stats.sort_values('count', ascending=False)

def point_in_country(lat, lon, country_geojson):
    """
    Check if a point (lat, lon) is within the country boundary
    
    Args:
        lat: Latitude
        lon: Longitude
        country_geojson: GeoJSON data with country boundary
        
    Returns:
        True if point is within country, False otherwise
    """
    if pd.isna(lat) or pd.isna(lon):
        return False
    
    point = Point(lon, lat)  # shapely uses (x, y) = (lon, lat)
    
    # Create a union of all features in the country boundary
    polygons = []
    for feature in country_geojson['features']:
        geom = shape(feature['geometry'])
        polygons.append(geom)
    
    if not polygons:
        return False
    
    # Union all polygons to create a single boundary
    country_boundary = unary_union(polygons)
    
    return country_boundary.contains(point)

# Cache the country boundary polygon for performance
_country_boundary_cache = None

def _get_country_boundary():
    """Get cached country boundary polygon"""
    global _country_boundary_cache
    if _country_boundary_cache is None:
        country_geojson = load_country_boundary()
        polygons = []
        for feature in country_geojson['features']:
            geom = shape(feature['geometry'])
            polygons.append(geom)
        if polygons:
            _country_boundary_cache = unary_union(polygons)
        else:
            _country_boundary_cache = None
    return _country_boundary_cache

def filter_points_in_country(df, lat_col='latitude', lon_col='longitude'):
    """
    Filter a DataFrame to only include points within the country boundary
    
    Args:
        df: DataFrame with latitude and longitude columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        
    Returns:
        Filtered DataFrame with only points within the country
    """
    country_boundary = _get_country_boundary()
    
    if country_boundary is None:
        # If no boundary available, return original dataframe
        return df.copy()
    
    # Create a mask for points within the country
    def check_point(row):
        if pd.isna(row[lat_col]) or pd.isna(row[lon_col]):
            return False
        point = Point(row[lon_col], row[lat_col])
        return country_boundary.contains(point)
    
    mask = df.apply(check_point, axis=1)
    
    return df[mask].copy()

