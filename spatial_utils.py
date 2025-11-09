"""
Spatial utility functions for associating points with districts
"""
import json
import pandas as pd
import geopandas as gpd
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
    
    OPTIMIZED: Uses geopandas spatial join with R-tree index (10-100x faster)
    
    Args:
        df: DataFrame with latitude and longitude columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        
    Returns:
        DataFrame with added 'assigned_district' column
    """
    # Load district boundaries as GeoDataFrame (with spatial index)
    districts_gdf = gpd.read_file('data/boundaries/malawi_districts.geojson')
    
    # Convert points to GeoDataFrame
    geometry = gpd.points_from_xy(df[lon_col], df[lat_col])
    points_gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=districts_gdf.crs)
    
    # Spatial join (uses R-tree spatial index - MUCH faster!)
    # For 50k points: ~5-30 seconds instead of 2-5 minutes
    joined = gpd.sjoin(points_gdf, districts_gdf[['geometry', 'shapeName']], 
                       how='left', predicate='within')
    
    # Add the assigned district column
    result = joined.drop(columns=['geometry', 'index_right'], errors='ignore')
    result = result.rename(columns={'shapeName': 'assigned_district'})
    
    return result

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
    
    OPTIMIZED: Uses vectorized geopandas operations
    
    Args:
        df: DataFrame with latitude and longitude columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        
    Returns:
        Filtered DataFrame with only points within the country
    """
    try:
        # Load country boundary as GeoDataFrame
        country_gdf = gpd.read_file('data/boundaries/malawi_country.geojson')
        country_boundary = country_gdf.unary_union  # Single polygon
        
        # Convert points to GeoDataFrame (vectorized)
        geometry = gpd.points_from_xy(df[lon_col], df[lat_col])
        points_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=country_gdf.crs)
        
        # Vectorized contains check (MUCH faster than row-by-row)
        mask = points_gdf.within(country_boundary)
        
        return df[mask].copy()
        
    except Exception as e:
        print(f"Warning: Could not filter by country boundary: {e}")
        # If boundary not available, return original dataframe
        return df.copy()

