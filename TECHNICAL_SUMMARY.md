# GAIA Planning - Technical Stack Summary

**One-Page Technical Overview for Judges**

---

## 🎯 The Problem
In Malawi, **20-30% of the population lives >10km from healthcare** → Distance equals death in rural areas. GAIA deploys mobile clinics but needs data-driven optimization.

---

## 💡 Our Solution
**Interactive geospatial analytics platform** that visualizes coverage gaps and plans optimal mobile clinic deployment using ML-driven algorithms.

---

## 📊 Data Stack

### Input Data (2.8 Million Data Points)
- **Population:** 7 demographic datasets × 400K points each (Meta Data for Good, 100m resolution)
- **Facilities:** 1,400 healthcare facilities (Malawi Health Facility Registry)
- **Boundaries:** 28 districts + 200+ sub-districts (geoBoundaries)
- **Validation:** 50+ actual GAIA mobile clinic GPS locations

### Processing
- **Parquet caching** → Instant load times (<1 sec)
- **Chunked processing** → Constant memory usage
- **Vectorized NumPy operations** → 10-100x speedup

---

## 🧮 Core Algorithms

### 1. Coverage Analysis → **O(m log n)**
```
BallTree spatial index + Haversine distance
→ 400K population points vs 1,400 facilities in <10 seconds
→ 100-1000x faster than brute force O(n·m)
```

### 2. District Assignment → **O(m log n)**
```
R-tree spatial join (GeoPandas)
→ 50,000 points assigned to districts in 5-30 seconds
→ 10-100x faster than point-in-polygon loops
```

### 3. Mobile Clinic Route Planning → **O(c·m log n)**
```
DBSCAN clustering + Greedy optimization
→ Identifies high-density underserved clusters
→ Hospital-based deployment (30km range, 5 stops/week)
→ Non-overlapping routes for maximum coverage
```

### 4. Equity Analysis
```
Gini coefficient → Quantify healthcare access inequality
Vulnerability scoring → High population + low access = priority
```

---

## 🏗️ Tech Stack

| Layer | Technology | Why? |
|-------|-----------|------|
| **Frontend** | Streamlit | Rapid interactive UI, Python-native |
| **Mapping** | PyDeck (deck.gl) | WebGL-powered, 50K+ points |
| **Spatial** | GeoPandas + Shapely | R-tree indexing, spatial joins |
| **ML** | scikit-learn | BallTree, DBSCAN clustering |
| **Data** | Pandas + NumPy | Vectorized operations |
| **Cache** | Parquet + Streamlit | 3-tier: memory → disk → pre-compute |

---

## 🚀 Key Innovations

### 1. **Dual-Layer Population Strategy**
- **Full dataset (400K)** for metrics → Accuracy
- **Sampled dataset (50K)** for visualization → Performance
- **Result:** Precise calculations + smooth rendering

### 2. **Three-Tier Caching**
- **Runtime:** Function-level memoization
- **Disk:** Parquet files survive restarts
- **Deploy:** Pre-computed cache files
- **Result:** 10-15s → <1s load time

### 3. **ML-Driven Route Planning**
- **DBSCAN:** Finds natural population clusters (not arbitrary grids)
- **Greedy selection:** Maximizes coverage per stop
- **Hospital-based:** Operationally realistic deployment
- **Result:** +5-10% coverage increase (~500K-1M people)

### 4. **Real-Time Interactive Analysis**
- Dynamic filtering (facility type, ownership, status)
- Live metric recalculation
- Choropleth maps by sub-district
- **Result:** Explore "what-if" scenarios instantly

---

## 📈 Performance Metrics

| Operation | Time | Scale |
|-----------|------|-------|
| Coverage calculation | <10 sec | 400K population points |
| District assignment | 5-30 sec | 50K points → 28 districts |
| Spatial query | ~2 sec | 1M distance calculations |
| Interactive response | <1 sec | Cached operations |
| Mobile clinic planning | 30-60 sec | 10 crews × 5 stops |

---

## 🎯 Impact

### Current State (Malawi)
- **Population:** 19 million
- **Functional facilities:** 1,400
- **Coverage (5km):** 60-70%
- **Healthcare deserts:** 20-30% of population

### With Mobile Clinics (Our Plan)
- **10 crews × 5 stops** = 50 new clinic locations
- **Additional coverage:** +5-10 percentage points
- **People reached:** ~500K-1M additional people
- **GPS coordinates provided** → Ready for immediate deployment

---

## 💎 Technical Elegance

### Algorithmic Sophistication
✅ **BallTree (O(log n) queries)** instead of brute force (O(n))  
✅ **R-tree spatial indexing** instead of nested loops  
✅ **Vectorized NumPy** instead of Python loops  
✅ **DBSCAN clustering** instead of arbitrary grids  

### Architecture Excellence
✅ **Separation of concerns:** Computation vs. visualization  
✅ **Progressive enhancement:** Works offline after first load  
✅ **Scalable design:** Millions of points, sub-second response  
✅ **Production-ready:** Cached, optimized, deployable  

### Code Quality
✅ **Modular libraries:** `coverage_lib.py`, `spatial_utils.py`, `mobile_clinic_planner.py`  
✅ **Documented:** Docstrings, type hints, READMEs  
✅ **Testable:** Separate logic from UI  
✅ **Open source:** Reproducible, extensible  

---

## 🔑 Key Differentiators

### 1. **Data-Driven Precision**
Not just visualization—quantitative coverage metrics with confidence intervals (P50, P75, P95 distances)

### 2. **Actionable Insights**
Specific GPS coordinates for mobile clinic deployment, not vague recommendations

### 3. **Equity Focus**
Gini coefficient quantifies healthcare inequality, vulnerability scoring identifies most at-risk populations

### 4. **Operational Realism**
Hospital-based deployment, 30km travel constraint, 5-stop weekly schedule → immediately implementable

### 5. **Interactive Exploration**
Not a static report—real-time "what-if" analysis for resource allocation decisions

---

## 🏆 Why This Matters

> **"Every percentage point of coverage increase means thousands of people gaining access to healthcare."**

### Technical Achievement
- Sophisticated spatial algorithms (BallTree, R-tree, DBSCAN)
- Vectorized operations for massive datasets
- Production-grade caching and optimization
- Interactive WebGL visualization

### Real-World Impact
- Identifies exact locations for mobile clinic deployment
- Quantifies equity gaps in healthcare access
- Provides district-level analysis for targeted resource allocation
- Scalable solution applicable to other regions/countries

### Social Good
- **Lives saved:** Earlier access to healthcare in rural areas
- **Equity improved:** Prioritizes most underserved populations
- **Resources optimized:** Maximum impact per mobile clinic crew
- **Transparency:** Open-source, reproducible methodology

---

## 🎤 Elevator Pitch

**"We built a geospatial analytics platform that combines 2.8 million population data points with ML-driven algorithms to plan optimal mobile clinic deployment in Malawi. Using BallTree spatial indexing, DBSCAN clustering, and three-tier caching, we identify exact GPS coordinates for 50 mobile clinic stops that would increase healthcare coverage by 5-10%, reaching an additional 500,000-1,000,000 people. The system processes 400,000 population points in under 10 seconds and provides interactive, district-level analysis for resource allocation decisions. Technical elegance meets social impact: every algorithm choice—from R-tree spatial joins to vectorized NumPy operations—was made to ensure that distance no longer equals death in rural Malawi."**

---

## 📞 Links

**GitHub:** https://github.com/dcallega/gaia_planning  
**Live Demo:** `streamlit run app.py`  
**Full Technical Docs:** `TECHNICAL_PRESENTATION.md`

---

*Elegant algorithms. Real-world impact. Lives saved.*

