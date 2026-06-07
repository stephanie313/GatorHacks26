# PulseMap SDOH Advocate · GatorHack 2026

Nationwide ZIP-level health equity map with K-Means clustering and evidence-based action guides.

## Live demo

- **Map:** https://stephanie313.github.io/GatorHacks26/health_map.html
- **Action guide:** https://stephanie313.github.io/GatorHacks26/action_guide.html
- **K-Means viz:** https://stephanie313.github.io/GatorHacks26/kmeans_viz.html

The site root redirects to the map automatically.

## Run locally

```bash
pip install -r requirements.txt
python3 main.py --zip 98101 --output health_map.html
python3 -m http.server 8765
```

Open http://127.0.0.1:8765/health_map.html

## Track

The Prioneer (AI for Social Good) · Women in Tech
