# Product Requirements Document (PRD)

## Project Name: PulseMap SDOH Advocate

## 1. Target Tracks (GatorHack 2026)

- **The Prioneer (AI for Social Good):** Addresses subnational health inequities by exposing hidden environmental and economic health drivers in vulnerable neighborhoods.
- **Women in Tech (WiT):** Surfaces systemic care barriers and provides accessible, jargon-free medical equity advocacy for families navigating local health risks.

## 2. Product Overview & User Flow

PulseMap SDOH Advocate is a local Python script that maps regional health disparities.

1. The user inputs a location (a 5-digit Zip Code).
2. The system reads local merged datasets containing population numbers and core health metrics (Asthma, Diabetes, Insurance access).
3. The system calls an LLM to generate localized, empathetic community health advice.
4. The system outputs an interactive, standalone HTML map (`health_map.html`).

## 3. Technical Requirements (The Tech Stack)

- **Language:** Python 3.10+
- **Data Analysis:** `pandas`
- **Mapping UI:** `folium` (Generates interactive Leaflet maps as static HTML files)
- **Intelligence:** LLM API (to compile raw numbers into clear, plain-language text alerts)

## 4. Minimum Viable Product (MVP) Features

- **Data Ingestion Engine:** Matches the user's location input to rows in a master `final_app_data.csv` file.
- **Dynamic Folium Map Generation:** Creates an HTML map centered on the input location with a clickable geographic marker.
- **AI Advocate Pop-ups:** Clicking on a region or pin displays an LLM-generated popup box translating raw data into practical lifestyle advice.