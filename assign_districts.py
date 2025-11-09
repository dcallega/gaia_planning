"""
Batch assign districts to facilities and clinic data using spatial joins
"""
import pandas as pd
from data_utils import ensure_population_csv
from spatial_utils import assign_districts_to_dataframe

def parse_gps_coordinates(gps_string):
    """Parse GPS string format: 'lat lon elevation accuracy'"""
    try:
        parts = str(gps_string).strip().split()
        if len(parts) >= 2:
            lat = float(parts[0])
            lon = float(parts[1])
            return lat, lon
    except:
        return None, None
    return None, None

def assign_districts_to_facilities():
    """Assign districts to MHFR facilities"""
    print("Loading MHFR facilities...")
    df = pd.read_csv('data/MHFR_Facilities.csv')
    
    # Rename columns
    df = df.rename(columns={
        'LATITUDE': 'latitude', 
        'LONGITUDE': 'longitude',
        'DISTRICT': 'district'
    })
    
    # Convert coordinates to numeric
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    # Find facilities without district or coordinates
    no_coords = df[(df['latitude'].isna()) | (df['longitude'].isna())]
    with_coords = df[(df['latitude'].notna()) & (df['longitude'].notna())]
    
    print(f"Total facilities: {len(df)}")
    print(f"  - With coordinates: {len(with_coords)}")
    print(f"  - Without coordinates: {len(no_coords)}")
    
    # Assign districts using spatial join
    print("\nAssigning districts using spatial join...")
    assigned_df = assign_districts_to_dataframe(with_coords, 'latitude', 'longitude')
    
    # Compare with existing district data
    has_district = assigned_df['district'].notna()
    has_assigned = assigned_df['assigned_district'].notna()
    
    print(f"\nResults:")
    print(f"  - Facilities with existing district: {has_district.sum()}")
    print(f"  - Facilities with assigned district: {has_assigned.sum()}")
    
    # Show mismatches (where existing doesn't match assigned)
    mismatches = assigned_df[
        (assigned_df['district'].notna()) & 
        (assigned_df['assigned_district'].notna()) &
        (assigned_df['district'].str.lower() != assigned_df['assigned_district'].str.lower())
    ]
    
    if len(mismatches) > 0:
        print(f"\n⚠️  Found {len(mismatches)} potential mismatches between existing and assigned districts:")
        print(mismatches[['NAME', 'district', 'assigned_district', 'latitude', 'longitude']].head(10))
    
    # Save to new file
    output_file = 'data/MHFR_Facilities_with_districts.csv'
    assigned_df.to_csv(output_file, index=False)
    print(f"\n✅ Saved results to {output_file}")
    
    return assigned_df

def assign_districts_to_clinics():
    """Assign districts to GAIA clinic stops"""
    print("\n" + "="*60)
    print("Loading GAIA clinic stops...")
    df = pd.read_csv('data/GAIA MHC Clinic Stops GPS.xlsx - Clinic stops GPS.csv')
    
    # Parse GPS coordinates
    df[['latitude', 'longitude']] = df['collect_gps_coordinates'].apply(
        lambda x: pd.Series(parse_gps_coordinates(x))
    )
    
    # Remove invalid coordinates
    df = df.dropna(subset=['latitude', 'longitude'])
    
    print(f"Total clinic stops with valid coordinates: {len(df)}")
    
    # Assign districts
    print("\nAssigning districts using spatial join...")
    assigned_df = assign_districts_to_dataframe(df, 'latitude', 'longitude')
    
    successful = assigned_df['assigned_district'].notna().sum()
    print(f"  - Successfully assigned: {successful} out of {len(df)}")
    
    # Show distribution by district
    print("\nClinic stops by district:")
    district_counts = assigned_df['assigned_district'].value_counts()
    for district, count in district_counts.items():
        print(f"  - {district}: {count}")
    
    # Save to new file
    output_file = 'data/GAIA_Clinics_with_districts.csv'
    assigned_df.to_csv(output_file, index=False)
    print(f"\n✅ Saved results to {output_file}")
    
    return assigned_df

def assign_districts_to_population():
    """Example: Assign districts to population data (on a sample)"""
    print("\n" + "="*60)
    print("Loading population data (sample)...")
    
    # Load a smaller sample
    csv_path = ensure_population_csv('general')
    df = pd.read_csv(csv_path, nrows=1000)
    
    print(f"Loaded {len(df)} population points (sample)")
    
    # Assign districts
    print("\nAssigning districts using spatial join...")
    assigned_df = assign_districts_to_dataframe(df, 'latitude', 'longitude')
    
    successful = assigned_df['assigned_district'].notna().sum()
    print(f"  - Successfully assigned: {successful} out of {len(df)}")
    
    # Show sample
    print("\nSample with district assignments:")
    print(assigned_df[['latitude', 'longitude', 'mwi_general_2020', 'assigned_district']].head(10))
    
    return assigned_df

if __name__ == "__main__":
    print("="*60)
    print("DISTRICT ASSIGNMENT TOOL")
    print("="*60)
    
    # Assign districts to facilities
    facilities_df = assign_districts_to_facilities()
    
    # Assign districts to clinics
    clinics_df = assign_districts_to_clinics()
    
    # Example with population (optional - commented out by default as it's large)
    # population_df = assign_districts_to_population()
    
    print("\n" + "="*60)
    print("✅ All assignments complete!")
    print("="*60)

