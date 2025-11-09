# Mobile Clinic Planning - Quick Reference

## ❓ Your Question

**"Does the algorithm consider having 5 deployments within 1 hour driving from the same hospital?"**

## ✅ Answer: YES (with caveats)

### What's Enforced:

✅ **1-hour travel constraint**
- Searches only within 30km radius (30 km/h off-road)
- All stops MUST be within this distance

✅ **Weekly schedule (5 stops)**
- Algorithm tries to find up to 5 high-density locations
- One stop per day (Monday-Friday)

### What's NOT Guaranteed:

⚠️ **Exactly 5 stops**
- May find fewer if population is too sparse
- Example: Might return 2-3 stops if only 2-3 dense clusters exist

⚠️ **Stop spacing/sequence**
- Doesn't optimize which stop to visit which day
- Doesn't plan the driving route between stops

---

## 🎯 How to Customize Planning

### Quick Configuration Template

```python
from coverage_lib import CoverageAnalyzer
from mobile_clinic_planner import MobileClinicPlanner
import mobile_clinic_planner

# ============ CUSTOMIZE HERE ============

# Travel distance (how far from hospital)
mobile_clinic_planner.MAX_TRAVEL_DISTANCE_KM = 30.0  # km

# Stops per week (how many days)
mobile_clinic_planner.STOPS_PER_WEEK = 5  # days

# Service radius (how far people will walk to clinic)
mobile_clinic_planner.SERVICE_RADIUS_KM = 5.0  # km

# Population threshold (minimum people to justify route)
MIN_POPULATION_PER_ROUTE = 15000  # people

# ============ END CUSTOMIZATION ============

# Run analysis
analyzer = CoverageAnalyzer(service_radius_km=5.0)
analyzer.load_facilities("data/MHFR_Facilities.csv", filter_functional=True)
analyzer.build_spatial_index()

gaps = analyzer.find_coverage_gaps("data/mwi_general_2020.csv", max_distance_km=10.0)

planner = MobileClinicPlanner(analyzer)
planner.identify_deployment_hospitals()

routes = planner.plan_mobile_clinic_network(
    gaps,
    max_teams=10,
    min_population_per_route=MIN_POPULATION_PER_ROUTE
)

routes.to_csv('my_routes.csv', index=False)
```

---

## 📊 Parameter Effects

| Parameter | Lower Value | Higher Value |
|-----------|------------|--------------|
| **MAX_TRAVEL_DISTANCE** | Fewer potential stops | More potential stops |
| | Easier logistics | Longer travel time |
| **STOPS_PER_WEEK** | Fewer stops needed | More stops needed |
| | Easier to staff | Better coverage |
| **MIN_POPULATION** | More routes planned | Fewer routes (high-impact only) |
| | Lower impact each | Higher impact each |
| **SERVICE_RADIUS** | People walk less | People walk more |
| | More clinics needed | Fewer clinics needed |

---

## 🚀 Common Scenarios

### Scenario 1: Conservative (Limited Resources)

```python
mobile_clinic_planner.MAX_TRAVEL_DISTANCE_KM = 20.0  # Short trips
mobile_clinic_planner.STOPS_PER_WEEK = 3  # 3 days/week
MIN_POPULATION_PER_ROUTE = 20000  # High threshold
```

**Result:** Fewer routes, highly feasible, high impact

### Scenario 2: Standard (Recommended)

```python
mobile_clinic_planner.MAX_TRAVEL_DISTANCE_KM = 30.0  # Default
mobile_clinic_planner.STOPS_PER_WEEK = 5  # Full week
MIN_POPULATION_PER_ROUTE = 15000  # Moderate threshold
```

**Result:** Balanced coverage and feasibility

### Scenario 3: Aggressive (Maximum Coverage)

```python
mobile_clinic_planner.MAX_TRAVEL_DISTANCE_KM = 40.0  # Extended
mobile_clinic_planner.STOPS_PER_WEEK = 5  # Full week  
MIN_POPULATION_PER_ROUTE = 10000  # Lower threshold
```

**Result:** Maximum routes, challenging logistics

### Scenario 4: Free Facilities Only

```python
# Before building spatial index:
analyzer.facilities_df = analyzer.facilities_df[
    analyzer.facilities_df['ownership'] == 'Government'
].copy()

mobile_clinic_planner.MAX_TRAVEL_DISTANCE_KM = 25.0
mobile_clinic_planner.STOPS_PER_WEEK = 5
MIN_POPULATION_PER_ROUTE = 15000
```

**Result:** Plans based on government facilities only

---

## 📂 Understanding Output

### Route File Structure

```csv
route_id | hospital_name | stop_number | lat | lon | population | distance_from_hospital
---------|---------------|-------------|-----|-----|------------|----------------------
0        | Lilongwe Hosp | 1           |-14.05|33.65| 45,000    | 22.3 km
0        | Lilongwe Hosp | 2           |-13.92|33.91| 38,000    | 18.7 km
0        | Lilongwe Hosp | 3           |-14.12|33.72| 29,000    | 15.8 km
1        | Mzuzu Hosp    | 1           |-11.45|34.02| 52,000    | 28.1 km
1        | Mzuzu Hosp    | 2           |-11.38|34.15| 41,000    | 19.4 km
```

### What Each Row Means:

- **route_id = 0**: First mobile clinic team
- **stop_number = 1**: Monday clinic location
- **stop_number = 2**: Tuesday clinic location
- **population**: Estimated people within 5km of this stop
- **distance_from_hospital**: Travel distance from base

### Route Analysis:

- **Route 0** (Lilongwe): 3 stops planned
  - Monday: 45k people (22.3km away)
  - Tuesday: 38k people (18.7km away)
  - Wednesday: 29k people (15.8km away)
  - Thursday: NO STOP (no dense population found)
  - Friday: NO STOP

- **Route 1** (Mzuzu): 2 stops planned
  - Monday: 52k people (28.1km away)
  - Tuesday: 41k people (19.4km away)
  - Wednesday-Friday: NO STOPS

---

## ⚡ Quick Checklist

Before deploying routes, verify:

- [ ] All stops are within MAX_TRAVEL_DISTANCE of hospital
- [ ] Each route serves > MIN_POPULATION people
- [ ] Locations are accessible by vehicle
- [ ] Roads are passable year-round (or note seasonal restrictions)
- [ ] Security situation is acceptable
- [ ] Hospital has capacity to deploy team(s)
- [ ] Staff and supplies available
- [ ] Communities are informed and supportive

---

## 🎓 Next Steps

1. **Run default planning:**
   ```bash
   python planning_workflow_example.py
   ```

2. **Review results:**
   - Check `proposed_mobile_clinic_routes.csv`
   - Plot locations on map

3. **Customize parameters:**
   - Adjust based on your resources
   - Re-run analysis

4. **Validate:**
   - Check with local teams
   - Verify accessibility

5. **Deploy:**
   - Start with 1-2 pilot routes
   - Scale based on learnings

---

## 📚 Full Documentation

- `PLANNING_EXPLAINED.md` - Detailed algorithm explanation
- `MOBILE_CLINIC_PLANNING_GUIDE.md` - Complete configuration guide
- `COVERAGE_LIBRARY_README.md` - Library API documentation

---

**Key Takeaway:** The algorithm provides data-driven suggestions considering your constraints. You control all parameters to match your operational reality!

