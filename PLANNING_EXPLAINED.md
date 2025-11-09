# Mobile Clinic Planning - How It Works

## Your Question: Does it consider the 5 stops within 1 hour constraint?

### Short Answer: **Partially ✅ / ⚠️**

The algorithm enforces:
- ✅ **1 hour travel constraint** - Only searches within 30km (1 hour at 30 km/h)
- ✅ **5 stops per week** - Tries to find up to 5 stops per hospital
- ⚠️ **But**: Doesn't guarantee exactly 5 stops if density is too low

## How the Algorithm Works

### Step-by-Step Process

```
For Each Hospital:

1. FILTER GAPS
   └─ Find all underserved population within 30km radius
   └─ This enforces the "1 hour drive" constraint

2. CLUSTER GAPS  
   └─ Use DBSCAN to find high-density underserved areas
   └─ Parameters: points within 5km cluster together
   └─ Minimum 50 population points to form a cluster

3. SELECT STOPS
   └─ Sort clusters by population (largest first)
   └─ Pick top 5 clusters (or fewer if less clusters found)
   └─ Place clinic stop at each cluster's centroid

4. VALIDATE ROUTE
   └─ Check if total population > minimum threshold (15,000)
   └─ If YES: Accept route
   └─ If NO: Reject route
```

### Visual Example

```
Hospital (H) at center
30km radius (1 hour drive) ─────────────────┐
                                             │
    [Gap Cluster 1]  ← 50,000 people        │
                                             │
           [Gap Cluster 2]  ← 30,000         │  Within
                                             │  30km
       [Gap Cluster 3]  ← 20,000             │
                                             │
    [Gap Cluster 4]  ← 15,000                │
                                             │
[Gap Cluster 5]  ← 10,000                    │
────────────────────────────────────────────┘

Algorithm selects: Clusters 1-5 (all within range)
Result: 1 route with 5 stops serving ~125,000 people
```

### What if there are only 2 clusters?

```
Hospital (H) at center
30km radius ─────────────────────────────────┐
                                             │
    [Large Gap Cluster]  ← 80,000 people     │
                                             │
                                             │
           [Small Cluster]  ← 25,000         │
                                             │
    (No other dense clusters found)          │
                                             │
────────────────────────────────────────────┘

Algorithm selects: Only 2 clusters
Result: 1 route with 2 stops (not 5!)
⚠️ You have 3 days without clinic stops
```

## Key Parameters You Can Customize

### 1. Travel Distance Constraint

```python
import mobile_clinic_planner

# Default: 30km (1 hour at 30 km/h off-road)
mobile_clinic_planner.MAX_TRAVEL_DISTANCE_KM = 30.0

# More aggressive (if roads are good)
mobile_clinic_planner.MAX_TRAVEL_DISTANCE_KM = 40.0

# More conservative (difficult terrain)
mobile_clinic_planner.MAX_TRAVEL_DISTANCE_KM = 20.0
```

**Effect:** Larger radius = more potential stops, but longer travel

### 2. Stops Per Week

```python
# Default: 5 stops (Monday-Friday)
mobile_clinic_planner.STOPS_PER_WEEK = 5

# Only 3 days (limited resources)
mobile_clinic_planner.STOPS_PER_WEEK = 3

# 6 days (including Saturday)
mobile_clinic_planner.STOPS_PER_WEEK = 6
```

**Effect:** Algorithm tries to find this many stops (but may find fewer)

### 3. Clustering Sensitivity

In `mobile_clinic_planner.py`, line ~111:

```python
# Default settings
eps_km = 5.0          # Cluster radius
min_samples = 50      # Min points to form cluster

# TIGHTER clustering (more stops, smaller each)
eps_km = 3.0
min_samples = 75

# LOOSER clustering (fewer stops, larger each)
eps_km = 7.0
min_samples = 30
```

**Effect:**
- Tighter → More clusters found → More potential stops
- Looser → Fewer clusters found → Fewer stops

### 4. Population Threshold

```python
# When planning routes
routes = planner.plan_mobile_clinic_network(
    gaps,
    max_teams=10,
    min_population_per_route=15000  # Change this
)

# Higher threshold (20k)
min_population_per_route=20000  # Only plan high-impact routes

# Lower threshold (10k)
min_population_per_route=10000  # Plan routes even for smaller populations
```

**Effect:** Higher threshold = fewer routes (only high-impact ones)

## Complete Custom Planning Example

```python
from coverage_lib import CoverageAnalyzer
from mobile_clinic_planner import MobileClinicPlanner
import mobile_clinic_planner

# ====================================
# YOUR CUSTOM CONFIGURATION
# ====================================

# 1. Configure travel distance
mobile_clinic_planner.MAX_TRAVEL_DISTANCE_KM = 25.0  # 25km max (conservative)

# 2. Configure stops per week  
mobile_clinic_planner.STOPS_PER_WEEK = 3  # Only 3 days/week (limited staff)

# 3. Configure service radius for mobile clinics
mobile_clinic_planner.SERVICE_RADIUS_KM = 4.5  # Each stop serves 4.5km radius

# ====================================
# RUN ANALYSIS
# ====================================

# Set up coverage analysis (free sites only)
analyzer = CoverageAnalyzer(service_radius_km=4.5)
analyzer.load_facilities("data/MHFR_Facilities.csv", filter_functional=True)
analyzer.facilities_df = analyzer.facilities_df[
    analyzer.facilities_df['ownership'] == 'Government'
].copy()
analyzer.build_spatial_index()

# Find gaps
gaps = analyzer.find_coverage_gaps("data/mwi_general_2020.csv", max_distance_km=10.0)

# Plan routes
planner = MobileClinicPlanner(analyzer)
planner.identify_deployment_hospitals()

# OPTIONAL: Filter to specific hospitals
planner.hospitals = planner.hospitals[
    planner.hospitals['district'].isin(['Lilongwe', 'Mzuzu'])
]

# Plan with custom thresholds
routes = planner.plan_mobile_clinic_network(
    gaps,
    max_teams=5,  # Max 5 teams available
    min_population_per_route=20000  # Must serve at least 20k
)

# Save results
if len(routes) > 0:
    routes.to_csv('my_custom_routes.csv', index=False)
    print(f"Planned {routes['route_id'].nunique()} routes")
    print(f"Total stops: {len(routes)}")
    print(f"Target population: {routes['population'].sum():,.0f}")
```

## Understanding the Results

### Route Output Columns

When you get planned routes, each row is one clinic stop:

```csv
route_id,hospital_name,hospital_lat,hospital_lon,stop_number,lat,lon,population,distance_from_hospital
0,Lilongwe Hospital,-13.98,33.78,1,-14.05,33.65,45000,22.3
0,Lilongwe Hospital,-13.98,33.78,2,-13.92,33.91,38000,18.7
0,Lilongwe Hospital,-13.98,33.78,3,-14.12,33.72,29000,15.8
```

**Interpretation:**
- `route_id`: 0 means this is the first route (team)
- `hospital_name`: Base hospital for this team
- `stop_number`: Day of week (1=Monday, 2=Tuesday, etc.)
- `lat,lon`: Where to set up the mobile clinic
- `population`: Estimated people within 5km of this stop
- `distance_from_hospital`: How far from base (must be <30km)

### Route Summary

```csv
hospital_name,n_stops,total_population,avg_distance_km
Lilongwe Hospital,5,187000,20.1
Mzuzu Hospital,3,95000,24.5
```

**Interpretation:**
- Lilongwe route has 5 stops (full week)
- Mzuzu route has only 3 stops (sparse population)

## Common Questions

### Q: Why doesn't every route have exactly 5 stops?

**A:** The algorithm only creates stops where there's sufficient population density. If an area is too sparse, it may only find 2-3 viable locations within the 30km radius.

**Solution:** Lower the clustering thresholds or extend travel distance.

### Q: Can I force exactly 5 stops per route?

**A:** Yes, but you'd need to modify the code. Current algorithm is demand-driven (finds density). You could add a grid-based approach that places 5 stops evenly regardless of density.

### Q: How do I ensure stops are well-distributed geographically?

**A:** The DBSCAN clustering naturally spreads them out (eps_km=5.0 means stops are at least 5km apart). To spread more, increase eps_km.

### Q: What if I want to consider actual roads, not straight-line distance?

**A:** You'd need to integrate a routing API or road network data. Current implementation uses haversine (straight-line) distance. A rough approximation is to multiply by 1.3 (roads are ~30% longer).

## Recommendations

### Start Conservative

```python
MAX_TRAVEL_DISTANCE_KM = 20.0  # Short distances
STOPS_PER_WEEK = 3             # Fewer days
min_population_per_route = 25000  # High threshold
```

→ Fewer routes, but highly feasible

### Then Expand

Once first routes are successful:

```python
MAX_TRAVEL_DISTANCE_KM = 30.0  # Extend reach
STOPS_PER_WEEK = 5             # Full week
min_population_per_route = 15000  # Lower threshold
```

→ More routes, greater coverage

### Validate Everything

Before deploying:
1. Plot stops on a map
2. Check road accessibility
3. Validate with local teams
4. Consider seasonal factors
5. Pilot test one route first

---

**Bottom Line:** The algorithm finds optimal locations based on population density and travel constraints, but you control all the parameters!

