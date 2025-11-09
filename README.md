# Gaia Planning

Data-driven resource allocation for equitable healthcare in Malawi. Interactive visualization and analysis tool for healthcare facility coverage and population density.

📍 [GitHub Repository](https://github.com/dcallega/gaia_planning) | 📋 [Problem Definition](https://hackforsocialimpact.notion.site/PS7-GAIA-Health-Data-Driven-Resource-Allocation-for-Equitable-Healthcare-in-Malawi-2a5ed247254d80db9741d5ceefa5a66a)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/dcallega/gaia_planning.git
cd gaia_planning

# Install dependencies
pip install -r requirements.txt

# Download and setup data (automated)
python setup_data.py

# Run the app
streamlit run app.py
```

The app will open at http://localhost:8501

## Current Status Estimate 
There are mobile clinics deployed in Mulanje, Phalombe, and Mangochi.

## Interactive Map Visualization

This project includes an interactive Streamlit app to visualize clinic locations and population density data across Malawi.

### Features

**Main Map Page:**
- 🗺️ Interactive map with clinic locations
- 📊 Multiple population density datasets (general, women, men, children, youth, elderly, women of reproductive age)
- 🎨 Color-coded population density visualization
- 🔍 Filter by specific clinic locations
- 🏥 MHFR facilities with filtering by status, ownership, and type
- 📈 Statistics and metrics for each dataset
- 🗺️ **NEW: District boundary visualization** - View all 28 districts of Malawi
- 🎯 **NEW: District assignment** - Associate facilities and clinics with districts using spatial joins

**Coverage Analysis Page:**
- 🎯 Calculate population coverage by healthcare facilities
- 📏 Adjustable coverage radius (1-50 km)
- 🏛️ Compare coverage by facility ownership (Government vs CHAM vs Private vs GAIA)
- 📊 Detailed metrics showing covered vs uncovered populations
- 🗺️ Visual map showing coverage areas and gaps
- 💾 Export coverage data and facility information to CSV
- 🔍 Filter by facility type, status, and ownership
- 📉 Distance distribution statistics

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Population data:
   
   - The repository includes compressed population datasets under `data/mwi_*_2020.csv.zip` (≈24 MB each).  
     They are extracted automatically the first time the app or helper scripts need them.
   - If you need to refresh from the upstream source, run the automated downloader:
     ```bash
     python setup_data.py
     ```
     This fetches the original Google Drive archive (~160 MB) and recreates the raw CSVs (~1.5 GB total).

### Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default web browser at http://localhost:8501

### District Boundaries & Spatial Analysis

**Setup District Boundaries:**
```bash
# Download district boundary files (GeoJSON format)
python download_boundaries.py
```

This downloads administrative boundaries for Malawi (districts, regions, and country) from [geoBoundaries](https://www.geoboundaries.org/).

**Batch Assign Districts:**
```bash
# Assign districts to all facilities and clinics
python assign_districts.py
```

This creates two new files:
- `data/MHFR_Facilities_with_districts.csv` - All health facilities with district assignments
- `data/GAIA_Clinics_with_districts.csv` - All clinic stops with district assignments

**Interactive District Assignment:**
- Enable **"🗺️ District Boundaries"** in the sidebar
- Navigate to **"🗺️ District Analysis"** section
- Use the **"🔍 Auto-assign Districts"** buttons to interactively assign districts to your data

📖 For detailed information, see [DISTRICT_BOUNDARIES_GUIDE.md](DISTRICT_BOUNDARIES_GUIDE.md)

### Data Sources
- **GAIA Mobile Health Clinic GPS locations**: Clinic stops with GPS coordinates
- **Malawi Health Facility Registry (MHFR)**: Comprehensive health facility database with locations, types, and ownership
- **High Resolution Population Density Maps**: Meta Data for Good - Population density data for Malawi (2020)
- **Administrative Boundaries**: [geoBoundaries](https://www.geoboundaries.org/) - Open-source political administrative boundaries for Malawi (28 districts)

