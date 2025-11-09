"""
Download Malawi administrative boundaries from geoBoundaries or GADM
"""
import requests
import json
import os

def download_malawi_boundaries():
    """Download Malawi district boundaries from geoBoundaries"""
    
    # Create boundaries directory if it doesn't exist
    os.makedirs('data/boundaries', exist_ok=True)
    
    print("Downloading Malawi administrative boundaries...")
    
    # Option 1: geoBoundaries (ADM1 = districts/regions, ADM0 = country)
    # ADM0 - Country boundary
    adm0_url = "https://www.geoboundaries.org/api/current/gbOpen/MWI/ADM0/"
    
    # ADM1 - First-level administrative divisions (regions)
    adm1_url = "https://www.geoboundaries.org/api/current/gbOpen/MWI/ADM1/"
    
    # ADM2 - Second-level administrative divisions (districts)
    adm2_url = "https://www.geoboundaries.org/api/current/gbOpen/MWI/ADM2/"
    
    try:
        # Get ADM0 (Country)
        print("Fetching country boundary metadata...")
        response = requests.get(adm0_url)
        response.raise_for_status()
        metadata = response.json()
        
        # Download the actual GeoJSON
        geojson_url = metadata['gjDownloadURL']
        print(f"Downloading country boundary from {geojson_url}")
        geojson_response = requests.get(geojson_url)
        geojson_response.raise_for_status()
        
        with open('data/boundaries/malawi_country.geojson', 'w') as f:
            json.dump(geojson_response.json(), f)
        print("✓ Country boundary saved to data/boundaries/malawi_country.geojson")
        
        # Get ADM2 (Districts)
        print("\nFetching district boundaries metadata...")
        response = requests.get(adm2_url)
        response.raise_for_status()
        metadata = response.json()
        
        # Download the actual GeoJSON
        geojson_url = metadata['gjDownloadURL']
        print(f"Downloading district boundaries from {geojson_url}")
        geojson_response = requests.get(geojson_url)
        geojson_response.raise_for_status()
        
        with open('data/boundaries/malawi_districts.geojson', 'w') as f:
            json.dump(geojson_response.json(), f)
        print("✓ District boundaries saved to data/boundaries/malawi_districts.geojson")
        
        # Also get ADM1 (Regions) for completeness
        print("\nFetching region boundaries metadata...")
        response = requests.get(adm1_url)
        response.raise_for_status()
        metadata = response.json()
        
        geojson_url = metadata['gjDownloadURL']
        print(f"Downloading region boundaries from {geojson_url}")
        geojson_response = requests.get(geojson_url)
        geojson_response.raise_for_status()
        
        with open('data/boundaries/malawi_regions.geojson', 'w') as f:
            json.dump(geojson_response.json(), f)
        print("✓ Region boundaries saved to data/boundaries/malawi_regions.geojson")
        
        print("\n✅ All boundaries downloaded successfully!")
        
        # Print district names for reference
        districts_data = geojson_response.json()
        print(f"\nFound {len(districts_data['features'])} districts:")
        for feature in districts_data['features']:
            name = feature['properties'].get('shapeName', 'Unknown')
            print(f"  - {name}")
            
    except Exception as e:
        print(f"❌ Error downloading boundaries: {e}")
        print("\nAlternative: You can manually download from:")
        print("  - https://data.humdata.org/dataset/cod-ab-mwi")
        print("  - https://www.geoboundaries.org/")

if __name__ == "__main__":
    download_malawi_boundaries()

