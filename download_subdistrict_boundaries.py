"""
Download sub-district (Traditional Authority) boundaries for Malawi from GADM.

GADM provides administrative boundaries at multiple levels:
- Level 0: Country
- Level 1: Regions  
- Level 2: Districts
- Level 3: Traditional Authorities (sub-district level)

This script downloads level 3 boundaries for creating choropleth maps.
"""

import urllib.request
import json
import os
import ssl


def download_gadm_boundaries(country_code="MWI", level=3, output_dir="data/boundaries"):
    """
    Download administrative boundaries from GADM.
    
    Args:
        country_code: ISO 3166-1 alpha-3 country code (MWI for Malawi)
        level: Administrative level (0=country, 1=regions, 2=districts, 3=TAs)
        output_dir: Directory to save the downloaded file
    """
    # GADM download URL (version 4.1)
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{country_code}_{level}.json"
    
    output_file = os.path.join(output_dir, f"malawi_level{level}.geojson")
    
    print(f"Downloading GADM level {level} boundaries for {country_code}...")
    print(f"URL: {url}")
    
    try:
        # Create SSL context that doesn't verify certificates (for compatibility)
        context = ssl._create_unverified_context()
        
        with urllib.request.urlopen(url, timeout=60, context=context) as response:
            # Parse and save as GeoJSON
            data = json.loads(response.read())
        
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(data, f)
        
        print(f"✓ Successfully downloaded to: {output_file}")
        
        # Print summary
        if 'features' in data:
            num_features = len(data['features'])
            print(f"  Contains {num_features} administrative units")
            
            # Show sample feature properties
            if num_features > 0:
                sample_props = data['features'][0]['properties']
                print(f"  Available properties: {list(sample_props.keys())}")
        
        return output_file
        
    except Exception as e:
        print(f"✗ Error downloading: {e}")
        print("\nAlternative: You can manually download from:")
        print(f"  https://gadm.org/download_country.html")
        print(f"  Select Malawi, format: GeoJSON, and level {level}")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("GADM Boundary Downloader for Malawi")
    print("=" * 60)
    
    # Download level 3 (Traditional Authorities - sub-district)
    level3_file = download_gadm_boundaries(country_code="MWI", level=3)
    
    if level3_file:
        print("\n✓ Download complete!")
        print(f"\nYou can now use choropleth maps with sub-district boundaries.")
    else:
        print("\n✗ Download failed. Please check your internet connection.")
        print("   You can also try downloading manually from gadm.org")

