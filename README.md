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

2. Download and extract the population data:
   
   **Option A - Automated setup (recommended):**
   ```bash
   python setup_data.py
   ```
   
   **Option B - Manual setup:**
   - Download `malawi_population_data.zip` from [Google Drive](https://drive.google.com/file/d/1BfIEe_35somT16UrIJPU4RUtknU4Fq_R/view?usp=drive_link)
   - Extract the zip file into the `data/` directory:
   ```bash
   unzip malawi_population_data.zip -d data/
   ```
   
   The zip file contains 7 population density CSV files (~159MB compressed, ~1.5GB uncompressed):
   - `mwi_general_2020.csv` - General population density
   - `mwi_women_2020.csv` - Women population density
   - `mwi_men_2020.csv` - Men population density
   - `mwi_children_under_five_2020.csv` - Children under 5 population density
   - `mwi_youth_15_24_2020.csv` - Youth (15-24) population density
   - `mwi_elderly_60_plus_2020.csv` - Elderly (60+) population density
   - `mwi_women_of_reproductive_age_15_49_2020.csv` - Women of reproductive age (15-49) population density

### Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default web browser at http://localhost:8501

### Data Sources
- **GAIA Mobile Health Clinic GPS locations**: Clinic stops with GPS coordinates
- **High Resolution Population Density Maps**: Meta Data for Good - Population density data for Malawi (2020)

