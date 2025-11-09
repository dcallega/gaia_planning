# Mobile Clinic Planning - Detailed Guide

## How the Algorithm Works

### Current Implementation

The mobile clinic planner follows these steps:

```
For each hospital:
  1. Find all population gaps within MAX_TRAVEL_DISTANCE (30km = 1hr at 30km/h)
  2. Use DBSCAN clustering to identify high-density underserved areas
  3. Select top N clusters (up to 5) with highest population
  4. Create clinic stops at cluster centroids
  5. Validate minimum population threshold
```

### Key Constraints (Currently Implemented)

✅ **Travel Distance Constraint**
- MAX_TRAVEL_DISTANCE_KM = 30km (1 hour at 30 km/h off-road)
- Only considers gaps within this radius from hospital

✅ **Weekly Schedule**
- STOPS_PER_WEEK = 5 (Monday-Friday)
- Algorithm tries to find up to 5 stops per hospital

✅ **Minimum Population**
- Default: 15,000 people per route
- Ensures routes are worth the resource investment

### What's NOT Fully Enforced Yet

⚠️ **Exactly 5 Stops**
- Algorithm finds "up to 5" stops based on clustering
- Might return 1, 2, or 3 stops if only that many high-density clusters exist
- Doesn't force 5 stops if population is too sparse

⚠️ **Stop Spacing**
- Doesn't ensure stops are well-distributed across the week
- Doesn't optimize travel sequence between stops

⚠️ **Road Network**
- Uses straight-line distance (haversine)
- Doesn't account for actual road networks or terrain

## How to Customize Planning Parameters

### 1. Change Travel Distance (Distance Constraint)

```python
import mobile_clinic_planner

# Extend to 40km (if vehicles/roads allow)
mobile_clinic_planner.MAX_TRAVEL_DISTANCE_KM = 40.0

# Or restrict to 20km (conservative)
mobile_clinic_planner.MAX_TRAVEL_DISTANCE_KM = 20.0

# Then create planner
planner = MobileClinicPlanner(analyzer)
```

### 2. Change Number of Stops Per Week

```python
# If you only deploy 3 days per week
mobile_clinic_planner.STOPS_PER_WEEK = 3

# Or if you want 6 stops (including Saturday)
mobile_clinic_planner.STOPS_PER_WEEK = 6
```

### 3. Adjust Clustering Sensitivity

```python
# In mobile_clinic_planner.py, find_optimal_clinic_locations method
# Modify these parameters:

# Cluster radius (how close points must be to form a cluster)
eps_km = 5.0  # Default - points within 5km form a cluster
eps_km = 3.0  # Tighter clusters (more stops, smaller coverage each)
eps_km = 8.0  # Looser clusters (fewer stops, larger coverage each)

# Minimum points to form a cluster
min_samples = 50  # Default
min_samples = 100  # More conservative (only high-density areas)
min_samples = 20  # More permissive (more potential stops)
```

### 4. Change Minimum Population Threshold

```python
# When calling plan_mobile_clinic_network:

# Higher threshold (only plan routes serving many people)
routes = planner.plan_mobile_clinic_network(
    gaps,
    max_teams=10,
    min_population_per_route=25000  # Increased from 15,000
)

# Lower threshold (plan routes even for smaller populations)
routes = planner.plan_mobile_clinic_network(
    gaps,
    max_teams=10,
    min_population_per_route=10000  # Decreased from 15,000
)
```

### 5. Filter to Specific Hospitals

```python
# Plan only from certain hospitals
planner.identify_deployment_hospitals()

# Filter to specific districts
planner.hospitals = planner.hospitals[
    planner.hospitals['district'].isin(['Lilongwe', 'Mzuzu', 'Blantyre'])
]

# Or filter to specific hospital types
planner.hospitals = planner.hospitals[
    planner.hospitals['type'] == 'District Hospital'
]

# Then plan routes
routes = planner.plan_mobile_clinic_network(gaps, max_teams=10)
```

## Complete Customized Planning Example

Here's a complete example with custom constraints:

```python
from coverage_lib import CoverageAnalyzer
from mobile_clinic_planner import MobileClinicPlanner, print_planning_summary
import mobile_clinic_planner

# ==========================================
# STEP 1: Configure Coverage Analysis
# ==========================================

# Use free facilities only at 4.5km
analyzer = CoverageAnalyzer(service_radius_km=4.5)
analyzer.load_facilities("data/MHFR_Facilities.csv", filter_functional=True)
analyzer.facilities_df = analyzer.facilities_df[
    analyzer.facilities_df['ownership'] == 'Government'
].copy()
analyzer.build_spatial_index()

# Find gaps
gaps = analyzer.find_coverage_gaps("data/mwi_general_2020.csv", max_distance_km=10.0)

# ==========================================
# STEP 2: Configure Mobile Clinic Planning
# ==========================================

# CUSTOMIZE: Travel distance (conservative approach)
mobile_clinic_planner.MAX_TRAVEL_DISTANCE_KM = 25.0  # Only 25km from hospital

# CUSTOMIZE: Only 3 stops per week (if limited resources)
mobile_clinic_planner.STOPS_PER_WEEK = 3

# CUSTOMIZE: Service radius for mobile clinics
mobile_clinic_planner.SERVICE_RADIUS_KM = 4.5  # Match coverage analysis

# ==========================================
# STEP 3: Plan Routes with Custom Settings
# ==========================================

planner = MobileClinicPlanner(analyzer)
planner.identify_deployment_hospitals()

# CUSTOMIZE: Only plan from District Hospitals
planner.hospitals = planner.hospitals[
    planner.hospitals['type'].isin(['District Hospital', 'Central Hospital'])
]

print(f"Planning from {len(planner.hospitals)} hospitals")

# CUSTOMIZE: Higher population threshold
routes = planner.plan_mobile_clinic_network(
    gaps,
    max_teams=5,  # Only 5 teams available
    min_population_per_route=20000  # Must serve at least 20k people
)

# Estimate impact
if len(routes) > 0:
    impact = planner.estimate_new_coverage(routes, "data/mwi_general_2020.csv")
    print_planning_summary(routes, impact)
    routes.to_csv('custom_mobile_clinic_routes.csv', index=False)
```

## Understanding Clustering Parameters

The algorithm uses DBSCAN to find high-density population clusters. Here's how parameters affect results:

### eps_km (Cluster Radius)

```
eps_km = 3.0  →  Many small clusters  →  More potential stops (smaller coverage each)
eps_km = 5.0  →  Moderate clusters    →  Balanced (default)
eps_km = 8.0  →  Few large clusters   →  Fewer stops (larger coverage each)
```

### min_samples (Minimum Cluster Size)

```
min_samples = 20  →  Easy to form clusters  →  More stops (some may be small)
min_samples = 50  →  Moderate threshold     →  Balanced (default)
min_samples = 100 →  Hard to form clusters  →  Fewer stops (only high-density)
```

### Example: Conservative Planning

```python
# Modify in mobile_clinic_planner.py line ~111:
eps_km = 4.0  # Tighter clustering
min_samples = 75  # Higher density required

# Result: Fewer stops, but each serves a denser population
```

### Example: Aggressive Coverage

```python
# Modify in mobile_clinic_planner.py line ~111:
eps_km = 7.0  # Looser clustering
min_samples = 30  # Lower density acceptable

# Result: More stops, reaching more dispersed populations
```

## Planning for Exactly 5 Stops Per Hospital

The current algorithm doesn't guarantee exactly 5 stops. Here's how to modify for stricter control:

### Option 1: Force 5 Stops (if enough clusters exist)

Modify `find_optimal_clinic_locations` in `mobile_clinic_planner.py`:

```python
# After line ~132, change:
selected_stops = clusters.head(n_stops).copy()

# To:
if len(clusters) >= n_stops:
    selected_stops = clusters.head(n_stops).copy()
elif len(clusters) > 0:
    # If fewer clusters than needed, duplicate/split largest ones
    selected_stops = clusters.copy()
    print(f"  ⚠️ Only {len(clusters)} clusters found, need {n_stops}")
else:
    selected_stops = pd.DataFrame()
```

### Option 2: Grid-based Stop Selection

Instead of clustering, place stops on a grid around the hospital:

```python
def plan_grid_stops(hospital_row, n_stops=5):
    """
    Place n_stops evenly spaced around hospital in a circle.
    """
    hospital_lat = hospital_row['lat']
    hospital_lon = hospital_row['lon']
    
    stops = []
    for i in range(n_stops):
        # Place stops in a circle at MAX_TRAVEL_DISTANCE
        angle = (2 * np.pi * i) / n_stops
        distance_km = MAX_TRAVEL_DISTANCE_KM * 0.8  # 80% of max
        
        # Calculate lat/lon offset
        delta_lat = (distance_km / 111.32) * np.cos(angle)  # ~111km per degree
        delta_lon = (distance_km / (111.32 * np.cos(np.radians(hospital_lat)))) * np.sin(angle)
        
        stops.append({
            'lat': hospital_lat + delta_lat,
            'lon': hospital_lon + delta_lon,
            'stop_number': i + 1
        })
    
    return pd.DataFrame(stops)
```

## Real-World Planning Considerations

### 1. Road Network Constraints

**Current:** Uses straight-line distance  
**Reality:** Roads may not be direct

**Solution:** Multiply distance by a "tortuosity factor"

```python
ROAD_TORTUOSITY = 1.3  # Roads are 30% longer than straight-line
effective_distance = straight_line_distance * ROAD_TORTUOSITY
```

### 2. Seasonal Accessibility

**Consideration:** Some areas inaccessible in rainy season

**Solution:** Add seasonal flags to planning

```python
# Filter gaps to dry-season accessible areas only
gaps_accessible = gaps[gaps['elevation'] > threshold]  # If you have elevation data
```

### 3. Population Density Validation

**Issue:** Some "gaps" may be uninhabited areas (forests, parks)

**Solution:** Set minimum density threshold

```python
# When finding gaps, filter low-density points
gaps = gaps[gaps['mwi_general_2020'] > 1.0]  # At least 1 person per grid cell
```

### 4. Security Considerations

**Issue:** Some areas may be unsafe for deployment

**Solution:** Manually filter out unsafe regions

```python
# Load unsafe districts (you'd need to maintain this list)
unsafe_districts = ['District1', 'District2']  # Example

# Filter hospitals
planner.hospitals = planner.hospitals[
    ~planner.hospitals['district'].isin(unsafe_districts)
]
```

## Workflow: From Analysis to Deployment

```
1. Run Coverage Analysis
   ↓
2. Identify Gaps (underserved populations)
   ↓
3. Configure Planning Parameters
   - Travel distance
   - Number of stops
   - Population threshold
   ↓
4. Run Mobile Clinic Planner
   ↓
5. Review Proposed Routes
   - Check feasibility
   - Validate locations
   ↓
6. Adjust Parameters & Re-run
   ↓
7. Export Final Routes for Field Teams
```

## Validation Checklist

Before deploying, validate:

- [ ] All stops are within travel distance of hospital
- [ ] Routes serve sufficient population to justify cost
- [ ] Locations are accessible by vehicle
- [ ] No stops in uninhabited areas (lakes, forests)
- [ ] Security situation is acceptable
- [ ] Communities are aware and supportive
- [ ] Staff and supplies can be sustained weekly

## Next Steps

1. **Run default planning:** See what algorithm suggests
2. **Review on map:** Plot proposed stops
3. **Validate with local knowledge:** Check accessibility
4. **Adjust parameters:** Refine based on constraints
5. **Iterate:** Re-run until routes are optimal
6. **Pilot test:** Deploy one route as proof of concept
7. **Scale up:** Expand to more routes based on learnings

---

**Key Takeaway:** The algorithm provides data-driven suggestions, but local knowledge and operational constraints should guide final decisions!

