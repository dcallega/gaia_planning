#!/usr/bin/env python3
"""
Setup script to download and extract Malawi population data from Google Drive.
"""

import os
import sys
import zipfile
from pathlib import Path

# File ID from Google Drive link
GDRIVE_FILE_ID = "1BfIEe_35somT16UrIJPU4RUtknU4Fq_R"
ZIP_FILENAME = "malawi_population_data.zip"
DATA_DIR = Path("data")

# Expected CSV files
EXPECTED_FILES = [
    "mwi_general_2020.csv",
    "mwi_women_2020.csv",
    "mwi_men_2020.csv",
    "mwi_children_under_five_2020.csv",
    "mwi_youth_15_24_2020.csv",
    "mwi_elderly_60_plus_2020.csv",
    "mwi_women_of_reproductive_age_15_49_2020.csv",
]


def check_files_exist():
    """Check if all expected CSV files already exist."""
    missing_files = []
    for filename in EXPECTED_FILES:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            missing_files.append(filename)
    return missing_files


def download_from_gdrive(file_id, destination):
    """Download file from Google Drive."""
    try:
        import gdown
    except ImportError:
        print("❌ gdown library not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
    
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"📥 Downloading from Google Drive...")
    print(f"   File size: ~159MB (this may take a few minutes)")
    
    try:
        gdown.download(url, destination, quiet=False)
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """Extract zip file to destination directory."""
    print(f"📦 Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("  GAIA Planning - Data Setup")
    print("=" * 60)
    print()
    
    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(exist_ok=True)
    
    # Check if files already exist
    print("🔍 Checking for existing data files...")
    missing_files = check_files_exist()
    
    if not missing_files:
        print("✅ All data files are already present!")
        print("   No download needed.")
        return 0
    
    print(f"📋 Missing {len(missing_files)} out of {len(EXPECTED_FILES)} files")
    print()
    
    # Download zip file
    zip_path = DATA_DIR / ZIP_FILENAME
    
    if zip_path.exists():
        print(f"📦 Found existing {ZIP_FILENAME}")
        use_existing = input("   Use existing file? (y/n): ").strip().lower()
        if use_existing != 'y':
            print("   Removing old file...")
            zip_path.unlink()
    
    if not zip_path.exists():
        success = download_from_gdrive(GDRIVE_FILE_ID, str(zip_path))
        if not success:
            print()
            print("⚠️  Automated download failed. Please download manually:")
            print("   1. Visit: https://drive.google.com/file/d/1BfIEe_35somT16UrIJPU4RUtknU4Fq_R/view")
            print(f"   2. Download the file to: {zip_path}")
            print("   3. Run this script again")
            return 1
    
    print()
    
    # Extract zip file
    success = extract_zip(zip_path, DATA_DIR)
    if not success:
        return 1
    
    # Verify extraction
    print()
    print("🔍 Verifying extracted files...")
    missing_files = check_files_exist()
    
    if not missing_files:
        print("✅ All files extracted successfully!")
        print()
        print("📊 Dataset includes:")
        for filename in EXPECTED_FILES:
            filepath = DATA_DIR / filename
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"   ✓ {filename} ({size_mb:.1f} MB)")
        
        # Optionally remove zip file
        print()
        remove_zip = input(f"Remove {ZIP_FILENAME}? (y/n): ").strip().lower()
        if remove_zip == 'y':
            zip_path.unlink()
            print(f"   Removed {ZIP_FILENAME}")
        
        print()
        print("🎉 Setup complete! You can now run the app:")
        print("   streamlit run app.py")
        return 0
    else:
        print(f"⚠️  Warning: Still missing {len(missing_files)} files:")
        for filename in missing_files:
            print(f"   - {filename}")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(1)

