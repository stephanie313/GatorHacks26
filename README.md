# Public Health Dashboard Map · SDOH Advocate

A nationwide ZIP-level health equity dashboard that surfaces **Social Determinants of Health (SDOH)** burdens, clusters communities with machine learning, and links each area to evidence-based next steps.

> **SDOH** = **Social Determinants of Health** — the conditions where people live, work, learn, and play (income, housing, food access, coverage, neighborhood resources) that shape health outcomes beyond clinical care alone.

---

## Live demo

| Page | URL |
|------|-----|
| **Home** | https://stephanie313.github.io/GatorHacks26/ |
| **Interactive map** | https://stephanie313.github.io/GatorHacks26/health_map.html |
| **Action guide** | https://stephanie313.github.io/GatorHacks26/action_guide.html |
| **K-Means visualization** | https://stephanie313.github.io/GatorHacks26/kmeans_viz.html |

The map page is a large standalone HTML file (~5 MB) and may take a few seconds to load on first visit.

---

## What it does

1. **Detect** — K-Means clustering on six SDOH health rates across ~30,000 U.S. ZIP codes assigns each community to one of four equity tiers.
2. **Explore** — Pan, zoom, and search by ZIP, city, or state. Click any marker for a localized health snapshot.
3. **Prioritize** — Each ZIP popup highlights the **top two local concerns** above the U.S. median (or the highest local rates when fewer qualify).
4. **Act** — Evidence-based suggested actions, cross-metric SDOH insights, and curated resource links help residents and advocates take practical next steps.

---

## Six health indicators

| Indicator | What it tracks |
|-----------|----------------|
| **Asthma** | Respiratory burden |
| **Diabetes** | Chronic metabolic disease |
| **Uninsured** | Health coverage gaps |
| **Obesity** | Weight-related cardiometabolic risk |
| **Depression** | Mental health burden |
| **Food Insecurity** | Access to reliable nutrition |

**Equity tiers (K-Means):** Low Risk · Moderate Risk · High Risk · Severe Disparity

Clustering uses all six rates in 6-dimensional space (scikit-learn K-Means). Tiers are ranked by the sum of each cluster's average rates.

---

## Features

- **Nationwide interactive map** — Folium/Leaflet map with state, city-cluster, and ZIP-level views
- **Search & filter** — Find locations; color markers by metric or overall burden; filter above U.S. median
- **ZIP popups** — Six-metric grid, equity tier summary, top concerns, and suggested actions modal
- **Action guide** — Standalone printable reference with evidence-based steps per indicator
- **K-Means viz** — D3 animation of Lloyd's algorithm on a stratified ZIP sample
- **Navigation** — Sticky top bar with **Home** and **Go back** on all subpages

---

## Tech stack

| Layer | Tools |
|-------|-------|
| **Language** | Python 3.10+ |
| **Data** | pandas, numpy |
| **ML** | scikit-learn (K-Means) |
| **Map** | Folium → standalone HTML |
| **Viz** | D3.js (K-Means animation) |
| **Deploy** | GitHub Pages (static HTML) |

---

## Project structure

```
GatorHack26/
├── index.html              # Landing page
├── health_map.html         # Generated interactive map (run main.py)
├── action_guide.html       # Generated evidence-based action guide
├── kmeans_viz.html         # K-Means D3 dashboard
├── kmeans_viz_data.json    # Sample data for K-Means viz (generated)
├── main.py                 # Source of truth — data, ML, HTML generation
├── data/
│   └── final_app_data.csv  # ~30k ZIPs × 6 SDOH metrics
├── requirements.txt
├── PRD.md                  # Product requirements
└── README.md
```

---

## Run locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate the map and guides

```bash
python3 main.py --zip 98101 --output health_map.html
```

This also writes `action_guide.html` and `kmeans_viz_data.json`.

### 3. Serve locally

```bash
python3 -m http.server 8765
```

Open:

- http://127.0.0.1:8765/ — landing page
- http://127.0.0.1:8765/health_map.html — map
- http://127.0.0.1:8765/action_guide.html — action guide
- http://127.0.0.1:8765/kmeans_viz.html — K-Means viz

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--zip` | `98101` | 5-digit U.S. ZIP to center the map |
| `--output` | `health_map.html` | Output path for the interactive map |
| `--action-guide` | `action_guide.html` | Output path for the action guide |
| `--kmeans-viz-data` | `kmeans_viz_data.json` | JSON sample for the K-Means dashboard |

---

## GatorHack 2026 track

- **The Prioneer (AI for Social Good)** — ML-driven detection of subnational health inequities and an advocacy layer that turns data into community action.

---

## Evidence sources

Suggested actions in the map and action guide are grounded in curated clinical and community sources, including:

- Chan et al., *eClinicalMedicine* 2025 (asthma)
- Galaviz et al., *American Journal of Lifestyle Medicine* 2018 (diabetes)
- Davis, *BMJ* 2007 (uninsured)
- Lean et al., *BMJ* 2006 (obesity)
- Dunlop et al., *Mental Health in Family Medicine* 2013 (depression)
- PovertyUSA.org (food insecurity)

Resource links include the [APA Psychologist Locator](https://locator.apa.org/) and [USDA Food & Nutrition Service](https://www.fns.usda.gov/).

---

## License & data

Built for **GatorHack 2026**. Health metrics are derived from merged public ZIP-level datasets (`final_app_data.csv`). See `PRD.md` for full product requirements.

**Repository:** https://github.com/stephanie313/GatorHacks26

