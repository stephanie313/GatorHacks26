#!/usr/bin/env python3
"""Generate an interactive health equity map for a given ZIP code."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

DATA_PATH = Path(__file__).parent / "data" / "final_app_data.csv"
USZIPS_PATH = Path(__file__).parent / "uszips.csv"
DEFAULT_OUTPUT = Path(__file__).parent / "health_map.html"
DEFAULT_ACTION_GUIDE = Path(__file__).parent / "action_guide.html"
KMEANS_VIZ_DATA_PATH = Path(__file__).parent / "kmeans_viz_data.json"
KMEANS_VIZ_SAMPLE_SIZE = 1200
DEFAULT_ZIP = "98101"  # Seattle, WA
PRODUCT_NAME = "Public Health Dashboard Map"
PRODUCT_TAGLINE = "Public Health Dashboard Map · SDOH Advocate"
HOME_PAGE = "index.html"
MAX_VISIBLE_MARKERS = 1500
MAX_CITY_CLUSTERS = 1000
STATE_ZOOM_THRESHOLD = 6
CLUSTER_ZOOM_THRESHOLD = 10
ZIP_DETAIL_ZOOM = 12
ML_FEATURE_COLUMNS = [
    "AsthmaRate",
    "DiabetesRate",
    "NoInsuranceRate",
    "ObesityRate",
    "DepressionRate",
    "FoodInsecurityRate",
]
ML_FEATURE_EXPORT_KEYS = ["a", "d", "u", "o", "dp", "fi"]
EQUITY_TIER_NAMES = [
    "Low Risk",
    "Moderate Risk",
    "High Risk",
    "Severe Disparity",
]
EQUITY_TIER_REPORTS: dict[str, str] = {
    "Low Risk": (
        "The Public Health Dashboard Map's local K-Means model classifies {city} in the "
        "{tier} equity tier based on asthma, diabetes, uninsured, obesity, "
        "depression, and food insecurity rates across ~30,000 U.S. ZIP codes. "
        "Baseline indicators in this cluster "
        "are relatively stable compared with higher-burden peers nationwide. "
        "Continued investment in preventive care and community wellness programs "
        "can help preserve these outcomes."
    ),
    "Moderate Risk": (
        "The Public Health Dashboard Map's local K-Means model places {city} in the "
        "{tier} equity tier, signaling modest but meaningful SDOH pressure. "
        "Multiple SDOH and health burden indicators here sit above the lowest-risk "
        "cluster but below the most acute disparities. Targeted screening, "
        "nutrition access, and local clinic partnerships are recommended to "
        "prevent escalation."
    ),
    "High Risk": (
        "The Public Health Dashboard Map's local K-Means model assigns {city} to the "
        "{tier} equity tier, where multiple health drivers cluster at "
        "elevated levels. Residents face compounded challenges across respiratory "
        "health, chronic disease, and insurance access that warrant coordinated "
        "response. Community health workers, expanded clinic hours, and "
        "cross-agency referral networks should be prioritized."
    ),
    "Severe Disparity": (
        "The Public Health Dashboard Map's local K-Means model identifies {city} in the "
        "{tier} tier—the highest-risk cluster in our four-group segmentation. "
        "Compounding chronic, mental health, coverage, and nutrition burdens point to systemic "
        "barriers that amplify one another across this community. Immediate "
        "intervention—coverage navigation, chronic-disease management, and "
        "cross-sector coalition building—is strongly advised."
    ),
}
NATIONAL_MEDIANS: dict[str, float] = {
    "Asthma": 10.7,
    "Diabetes": 12.7,
    "Uninsured": 8.6,
    "Obesity": 36.7,
    "Depression": 23.0,
    "Food Insecurity": 13.5,
}
METRIC_SPECS: list[dict[str, str | float]] = [
    {"col": "AsthmaRate", "js": "a", "emoji": "🫁", "label": "Asthma", "high": 15, "medium": 10},
    {"col": "DiabetesRate", "js": "d", "emoji": "🩸", "label": "Diabetes", "high": 12, "medium": 8},
    {"col": "NoInsuranceRate", "js": "u", "emoji": "🛡️", "label": "Uninsured", "high": 20, "medium": 12},
    {"col": "ObesityRate", "js": "o", "emoji": "⚖️", "label": "Obesity", "high": 35, "medium": 25},
    {"col": "DepressionRate", "js": "dp", "emoji": "🧠", "label": "Depression", "high": 22, "medium": 15},
    {"col": "FoodInsecurityRate", "js": "fi", "emoji": "🍎", "label": "Food Insecurity", "high": 15, "medium": 10},
]
TOP_CONCERNS_COUNT = 2
METRIC_SOLUTIONS: dict[str, dict[str, list[str] | str]] = {
    "Asthma": {
        "summary": (
            "Structured asthma pathways combine symptom tracking, trigger reduction, "
            "and a written action plan so attacks are caught early."
        ),
        "actions": [
            "Track symptoms and rescue inhaler use; seek care if rescue use increases.",
            "Work with your clinician on trigger awareness (smoke, pollution, allergens).",
            "Follow a written asthma action plan and keep controller meds on schedule.",
            "Ask about home peak flow monitoring if your care team recommends it.",
        ],
        "source": "Chan et al., eClinicalMedicine 2025 (digital asthma journey)",
    },
    "Diabetes": {
        "summary": (
            "Lifestyle programs can prevent or delay type 2 diabetes. Screening, "
            "weight loss, activity, and diet changes work in clinics and communities."
        ),
        "actions": [
            "Ask about diabetes screening (A1C, fasting glucose, or glucose tolerance) if you are overweight or have risk factors—many prediabetes cases go undiagnosed.",
            "If at risk, join a lifestyle program targeting about 7% weight loss and at least 150 minutes of moderate activity per week.",
            "Improve diet quality: more fiber and whole grains, less saturated fat; reduce sugar-sweetened drinks.",
            "Ask for brief counseling, group diabetes-prevention sessions, or a referral to a local program (clinic, YMCA, or telehealth).",
        ],
        "resource_link": {
            "label": "USDA Food & Nutrition Service (nutrition programs & diabetes resources)",
            "url": "https://www.fns.usda.gov/",
        },
        "source": "Galaviz et al., American Journal of Lifestyle Medicine 2018",
    },
    "Uninsured": {
        "summary": (
            "Coverage gaps drive inequitable access, avoidable harm, and poorly managed "
            "chronic conditions. State and community programs can expand affordable options."
        ),
        "actions": [
            "Check eligibility for Medicaid, CHIP, or state programs that cover low-income children—and whether your state covers low-income adults.",
            "If you work without employer insurance, ask about state insurance pools that subsidize premiums for small businesses and the self-employed.",
            "Prioritize a regular primary care source; being uninsured is linked to missed preventive care and poorly managed chronic conditions.",
            "Connect with local enrollment help to compare individual or marketplace coverage and reduce gaps that force emergency-only care.",
        ],
        "source": "Davis, BMJ 2007 (Uninsured in America: problems and possible solutions)",
    },
    "Obesity": {
        "summary": (
            "Modest, sustained changes to diet and physical activity support healthy "
            "weight loss and lower cardiometabolic risk."
        ),
        "actions": [
            "Combine modest diet changes with regular walking or activity most days.",
            "Reduce sugary drinks and high-fat processed foods; add fruits and vegetables.",
            "Use smaller portions and limit long stretches of sitting or screen time.",
            "Ask your clinic about structured nutrition or weight-support programs nearby.",
        ],
        "source": "Lean et al., BMJ 2006 (ABC of Obesity)",
    },
    "Depression": {
        "summary": (
            "Evidence-based psychotherapy and sustained follow-up toward remission "
            "reduce long-term burden; crisis support is available when needed."
        ),
        "actions": [
            "Ask about evidence-based therapy such as CBT or interpersonal therapy (IPT).",
            "Keep follow-up visits until symptoms clearly improve—treat toward remission.",
            "Tell your doctor if treatment is not helping after about six weeks.",
            "Be open about alcohol or drug use; call or text 988 if you are in crisis.",
        ],
        "resource_link": {
            "label": "Find a psychologist near you (APA locator)",
            "url": "https://locator.apa.org/",
        },
        "source": "Dunlop et al., Mental Health in Family Medicine 2013",
    },
    "Food Insecurity": {
        "summary": (
            "Poverty is the primary driver—not grocery distance alone. When food costs "
            "compete with rent, energy, and healthcare, households face trade-offs that "
            "raise risk for diabetes, obesity, and depression."
        ),
        "connects_with": ["Diabetes", "Obesity", "Depression", "Uninsured"],
        "actions": [
            "Apply for SNAP and ask about WIC or school meal programs if affording quality food is difficult.",
            "Connect with local food banks, co-ops, or mutual aid networks—not only when stores are far away.",
            "When food, rent, and medical costs compete, combine nutrition help with coverage navigation in the same plan.",
            "Volunteer, donate, or advocate locally for stronger SNAP benefits and universal school lunch access.",
        ],
        "resource_link": {
            "label": "USDA Food & Nutrition Service (SNAP, WIC, school meals)",
            "url": "https://www.fns.usda.gov/",
        },
        "source": 'PovertyUSA.org, "What Causes Food Insecurity and What are Solutions to It?"',
    },
}
SDOH_CONNECTION_INSIGHTS: list[dict[str, list[str] | str]] = [
    {
        "requires": ["Food Insecurity", "Diabetes"],
        "text": (
            "Food-insecure adults face higher chronic-disease risk, including diabetes. "
            "Pair nutrition access with screening and lifestyle prevention steps."
        ),
    },
    {
        "requires": ["Food Insecurity", "Obesity"],
        "text": (
            "Food insecurity often means lower diet quality—not just less food—which "
            "can worsen obesity risk. Address affordability and nutrition support together."
        ),
    },
    {
        "requires": ["Food Insecurity", "Depression"],
        "text": (
            "Food stress and mental health burdens frequently overlap in the same "
            "communities. Coordinate food programs with depression care and follow-up."
        ),
    },
    {
        "requires": ["Food Insecurity", "Uninsured"],
        "text": (
            "Households often choose between food, housing, and healthcare. SNAP/WIC "
            "and coverage enrollment can relieve the same budget pressure."
        ),
    },
    {
        "requires": ["Uninsured", "Diabetes"],
        "text": (
            "Coverage gaps make prediabetes screening and prevention harder to reach. "
            "Prioritize enrollment help alongside community diabetes-prevention programs."
        ),
    },
    {
        "requires": ["Uninsured", "Depression"],
        "text": (
            "Without insurance, therapy and follow-up care are harder to sustain. "
            "Pair coverage navigation with evidence-based mental health treatment access."
        ),
    },
]
SDOH_CONNECTION_NARRATIVE = (
    "The Public Health Dashboard Map treats these six indicators as a system—not isolated statistics. "
    "Poverty drives food insecurity; food stress raises diabetes, obesity, and depression risk; "
    "and being uninsured makes every chronic condition harder to manage. K-Means clustering "
    "surfaces where these burdens stack up, and the Action Guide maps each concern to steps "
    "from your source library."
)
POPUP_HIGHLIGHT_BG = "#EBEBEB"
POPUP_HIGHLIGHT_BORDER = "#D4D4D4"
POPUP_DIVIDER = "#E5E1DB"
POPUP_ACCENT = "#36454F"
US_STATE_NAMES: dict[str, str] = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "District of Columbia",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Generate an interactive {PRODUCT_TAGLINE} health equity map."
    )
    parser.add_argument(
        "--zip",
        default=DEFAULT_ZIP,
        metavar="ZIP",
        help=f"5-digit U.S. ZIP code to center the map (default: {DEFAULT_ZIP}, Seattle WA)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output HTML file path (default: {DEFAULT_OUTPUT.name})",
    )
    parser.add_argument(
        "--action-guide",
        default=str(DEFAULT_ACTION_GUIDE),
        help=f"Evidence-based action guide HTML (default: {DEFAULT_ACTION_GUIDE.name})",
    )
    parser.add_argument(
        "--kmeans-viz-data",
        default=str(KMEANS_VIZ_DATA_PATH),
        help=f"JSON sample for kmeans_viz.html (default: {KMEANS_VIZ_DATA_PATH.name})",
    )
    return parser.parse_args()


def validate_zip(zip_code: str) -> int:
    zip_code = zip_code.strip()
    if not zip_code.isdigit():
        raise ValueError(f"ZIP code must contain only digits, got '{zip_code}'.")
    if len(zip_code) != 5:
        raise ValueError(f"ZIP code must be exactly 5 digits, got '{zip_code}'.")
    return int(zip_code)


def load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found at '{data_path}'. "
            "Ensure final_app_data.csv is in the data/ directory."
        )
    return pd.read_csv(data_path, dtype={"ZipCode": str})


def normalize_zip_series(df: pd.DataFrame) -> pd.Series:
    return df["ZipCode"].astype(str).str.lstrip("0").replace("", "0").astype(int)


def lookup_zip(df: pd.DataFrame, zip_code: int) -> pd.Series:
    matches = df[normalize_zip_series(df) == zip_code]
    if matches.empty:
        raise KeyError(f"No data found for ZIP code {zip_code:05d}.")
    return matches.iloc[0]


def format_population(value: str | int | float) -> str:
    if isinstance(value, str):
        return value
    return f"{int(value):,}"


def prepare_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the six-feature matrix used for K-Means (median-impute missing values)."""
    features = df[ML_FEATURE_COLUMNS].astype(float).copy()
    return features.fillna(features.median(numeric_only=True))


def export_kmeans_viz_data(
    df: pd.DataFrame,
    output_path: Path,
    sample_size: int = KMEANS_VIZ_SAMPLE_SIZE,
) -> int:
    """Export a stratified ZIP sample for the D3 K-Means animation dashboard."""
    features = prepare_ml_features(df)
    count = min(sample_size, len(features))
    sample_index = features.sample(n=count, random_state=42).index
    points = []
    for idx in sample_index:
        row = df.loc[idx]
        values = features.loc[idx]
        point = {
            "id": str(row["ZipCode"]).zfill(5),
            "city": str(row["city"]),
            "tier": str(row["EquityVulnerabilityTier"]),
        }
        for key, column in zip(ML_FEATURE_EXPORT_KEYS, ML_FEATURE_COLUMNS, strict=True):
            point[key] = round(float(values[column]), 2)
        points.append(point)
    payload = {
        "ml_features": ML_FEATURE_EXPORT_KEYS,
        "feature_axes": {
            "x": "AsthmaRate",
            "y": "DiabetesRate",
            "plot_note": "2D plot; clustering uses all six SDOH rates in 6D space.",
        },
        "tier_labels": ["Low", "Moderate", "High", "Severe"],
        "sample_size": count,
        "points": points,
    }
    output_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return count


def assign_equity_vulnerability_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """Fit K-Means on six SDOH rates; label each ZIP with an equity tier."""
    features = prepare_ml_features(df)

    model = KMeans(n_clusters=4, n_init=10, random_state=42)
    cluster_labels = model.fit_predict(features)

    center_sums = model.cluster_centers_.sum(axis=1)
    ordered_clusters = np.argsort(center_sums)
    cluster_to_tier = {
        int(cluster_id): EQUITY_TIER_NAMES[rank]
        for rank, cluster_id in enumerate(ordered_clusters)
    }

    enriched = df.copy()
    enriched["EquityCluster"] = cluster_labels
    enriched["EquityVulnerabilityTier"] = enriched["EquityCluster"].map(cluster_to_tier)
    return enriched


def load_state_lookup() -> dict[str, str]:
    if not USZIPS_PATH.exists():
        return {}
    uszips = pd.read_csv(USZIPS_PATH, dtype={"zip": str})
    normalized = uszips["zip"].astype(str).str.zfill(5)
    return dict(zip(normalized, uszips["state_id"], strict=False))


def prepare_zip_records(
    df: pd.DataFrame,
    state_lookup: dict[str, str] | None = None,
) -> list[dict[str, float | int | str]]:
    state_lookup = state_lookup or {}
    records: list[dict[str, float | int | str]] = []
    for row in df.itertuples(index=False):
        zip_int = int(str(row.ZipCode).lstrip("0") or "0")
        zip_label = str(row.ZipCode).zfill(5)
        records.append(
            {
                "z": zip_int,
                "zip": zip_label,
                "c": row.city,
                "st": state_lookup.get(zip_label, ""),
                "lat": float(row.lat),
                "lng": float(row.lng),
                "a": float(row.AsthmaRate),
                "d": float(row.DiabetesRate),
                "u": float(row.NoInsuranceRate),
                "o": None if pd.isna(row.ObesityRate) else float(row.ObesityRate),
                "dp": None if pd.isna(row.DepressionRate) else float(row.DepressionRate),
                "fi": None if pd.isna(row.FoodInsecurityRate) else float(row.FoodInsecurityRate),
                "p": format_population(row.TotalPopulation),
                "t": str(getattr(row, "EquityVulnerabilityTier", "Moderate Risk")),
            }
        )
    return records


def add_map_legend(
    health_map: folium.Map,
    center_city: str,
    center_zip: int,
    total_zips: int,
) -> None:
    metric_options = (
        '<option value="overall">Overall (vs US median)</option>'
        + "".join(
            f'<option value="{spec["js"]}">{spec["label"]}</option>'
            for spec in METRIC_SPECS
        )
    )
    legend_html = f"""
    <nav id="pulsemap-page-nav" aria-label="Page navigation" style="
        position: fixed; top: 0; left: 0; right: 0; z-index: 1002;
        display: flex; align-items: center; gap: 10px; padding: 10px 14px;
        background: rgba(255, 255, 255, 0.96); backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(0, 0, 0, 0.08);
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
        font-family: Inter, system-ui, sans-serif;
    ">
        <a href="{HOME_PAGE}" aria-label="Go to home page" style="
            display: inline-flex; align-items: center; padding: 8px 14px;
            border-radius: 999px; font-size: 12px; font-weight: 600;
            text-decoration: none; cursor: pointer;
            border: 1px solid rgba(0, 0, 0, 0.1); background: #fff; color: #36454F;
        ">← Home</a>
        <button type="button" id="pulsemap-back-btn" aria-label="Go back to previous page" style="
            display: inline-flex; align-items: center; padding: 8px 14px;
            border-radius: 999px; font-size: 12px; font-weight: 600;
            cursor: pointer; border: none; background: #5c6b73; color: #fff;
        ">← Go back</button>
        <span style="margin-left: auto; font-size: 11px; font-weight: 600; color: #36454F;">
            {html.escape(PRODUCT_NAME)}
        </span>
    </nav>
    <div id="pulsemap-controls" style="
        position: fixed; top: 54px; left: 10px; z-index: 1001;
        width: 258px; background: white; padding: 12px 14px;
        border-radius: 10px; box-shadow: 0 4px 16px rgba(0,0,0,0.12);
        font-family: Inter, system-ui, sans-serif; font-size: 12px;
        line-height: 1.45; border: 1px solid rgba(0,0,0,0.08);
    ">
        <div style="font-weight: 700; color: #36454F; margin-bottom: 8px;">Search &amp; Filter</div>
        <div style="margin-bottom: 10px;">
            <div style="display: flex; gap: 6px; align-items: stretch;">
                <div style="position: relative; flex: 1;">
                    <input id="location-search" type="text" placeholder="ZIP, city, or state"
                        autocomplete="off" spellcheck="false"
                        style="width:100%; box-sizing:border-box; padding:7px 30px 7px 9px;
                        border:1px solid #d4d4d4; border-radius:8px; font-size:12px; outline:none;" />
                    <button id="location-search-clear" type="button" aria-label="Clear search"
                        style="display:none; position:absolute; right:6px; top:50%; transform:translateY(-50%);
                        width:20px; height:20px; border:none; border-radius:10px; background:#ebebeb;
                        color:#666; font-size:14px; line-height:1; cursor:pointer; padding:0;">×</button>
                    <div id="search-suggestions" style="
                        display:none; position:absolute; top:calc(100% + 4px); left:0; right:0;
                        background:#fff; border:1px solid #d4d4d4; border-radius:8px;
                        box-shadow:0 8px 20px rgba(0,0,0,0.12); max-height:280px; overflow-y:auto; z-index:1002;
                    "></div>
                </div>
                <button id="location-search-btn" type="button"
                    style="padding:7px 12px; border:none; border-radius:8px; background:#36454F;
                    color:#fff; font-size:12px; font-weight:600; cursor:pointer;">Go</button>
            </div>
        </div>
        <label for="metric-filter" style="display:block; font-size:11px; color:#666; margin-bottom:4px;">
            Color markers by
        </label>
        <select id="metric-filter"
            style="width:100%; padding:7px 9px; border:1px solid #d4d4d4; border-radius:8px; font-size:12px;">
            {metric_options}
        </select>
        <label style="display:flex; align-items:center; gap:7px; margin-top:10px; font-size:11px; color:#444;">
            <input id="above-national-filter" type="checkbox" />
            Show only above US median
        </label>
        <div id="search-status" style="color:#b45309; font-size:11px; margin-top:8px; min-height:14px;"></div>
    </div>
    <div id="pulsemap-legend" style="
        position: fixed; bottom: 28px; left: 28px; z-index: 9999;
        background: white; padding: 12px 14px; border-radius: 10px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
        font-family: Inter, system-ui, sans-serif; font-size: 12px;
        line-height: 1.5; border: 1px solid rgba(0,0,0,0.08);
    ">
        <div style="font-weight: 700; margin-bottom: 6px;">{PRODUCT_TAGLINE}</div>
        <div style="color: #555; margin-bottom: 8px;">
            {total_zips:,} ZIPs nationwide &mdash; drag &amp; zoom to explore<br>
            Home ZIP: {html.escape(center_city)} ({center_zip:05d})
        </div>
        <div id="view-mode-badge" style="
            display: inline-block; margin-bottom: 8px; padding: 3px 10px;
            border-radius: 999px; background: #36454F; color: #fff;
            font-size: 10px; font-weight: 700; letter-spacing: 0.3px;
        ">Exploring</div>
        <div id="visible-zip-count" style="color: #555; margin-bottom: 8px;"></div>
        <div style="color: #888; font-size: 11px; margin-bottom: 8px;">
            State overview &lt; zoom 6 &middot; city clusters 6&ndash;9 &middot; ZIPs 10+
        </div>
        <div id="legend-scale" style="font-size: 11px;">
            <div style="color:#666; margin-bottom:6px;">
                Marker color: <strong id="legend-metric-name">Asthma</strong>
            </div>
            <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
                <span style="width:10px; height:10px; border-radius:50%; background:#16a34a; display:inline-block;"></span>
                <span id="legend-low-label">Asthma &le; 10%</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
                <span style="width:10px; height:10px; border-radius:50%; background:#d97706; display:inline-block;"></span>
                <span id="legend-mid-label">Asthma 10&ndash;15%</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
                <span style="width:10px; height:10px; border-radius:50%; background:#dc2626; display:inline-block;"></span>
                <span id="legend-high-label">Asthma &gt; 15%</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px; margin-top:4px; color:#888;">
                <span style="width:10px; height:10px; border-radius:50%; background:#9ca3af; display:inline-block;"></span>
                No data
            </div>
        </div>
    </div>
    <div id="pulsemap-tools" style="position:fixed; top:54px; right:10px; z-index:1001; display:flex; gap:6px; align-items:flex-start;">
    <div id="pulsemap-guide" style="position:relative;">
        <button id="pulsemap-guide-btn" type="button" aria-label="Action guide"
            aria-expanded="false" aria-controls="pulsemap-guide-panel"
            style="
                height:32px; padding:0 11px; border:none; border-radius:999px;
                background:#36454F; color:#fff; font-family:system-ui,sans-serif;
                font-size:11px; font-weight:700; line-height:1; cursor:pointer;
                box-shadow:0 4px 14px rgba(0,0,0,0.18); letter-spacing:0.2px;
            ">Guide</button>
        <div id="pulsemap-guide-panel" style="
            display:none; position:absolute; top:40px; right:0;
            box-sizing:border-box; flex-direction:column;
            background:#fff; border:1px solid rgba(0,0,0,0.08); border-radius:10px;
            box-shadow:0 8px 24px rgba(0,0,0,0.14); padding:12px 14px 16px;
            font-family:Inter,system-ui,sans-serif; font-size:11.5px;
            line-height:1.55; color:#444; overflow:hidden;
        ">
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; flex-shrink:0;">
                <div style="font-weight:700; font-size:12px; color:#36454F;">Action Guide</div>
                <button id="pulsemap-guide-close" type="button" aria-label="Close action guide"
                    style="
                        width:22px; height:22px; border:none; border-radius:11px;
                        background:#ebebeb; color:#666; font-size:14px; cursor:pointer;
                        line-height:1; padding:0;
                    ">×</button>
            </div>
            <div style="
                margin-bottom:10px; padding:9px 10px; border-radius:8px;
                background:#f4f7f4; border:1px solid #dfe8df; color:#3d4f3d;
                font-size:11px; line-height:1.55; flex-shrink:0;
            ">
                The Public Health Dashboard Map uses <strong>machine learning</strong> to flag high-burden ZIP codes, then links each
                community's top concerns with <strong>evidence-based steps</strong> from your six sources—showing
                how food, coverage, and chronic-disease risks often compound together.
            </div>
            <div id="pulsemap-guide-sections" style="flex:1; min-height:0; overflow:auto;"></div>
            <div style="margin-top:10px; padding-top:8px; border-top:1px solid #eee; font-size:10px; color:#888; flex-shrink:0;">
                Open <a href="action_guide.html" target="_blank" rel="noopener"
                    style="color:#36454F; font-weight:600;">action_guide.html</a> for a printable reference.
            </div>
            <div id="pulsemap-guide-resize" role="separator" aria-orientation="both"
                aria-label="Resize action guide panel" title="Drag to resize"
                style="
                    position:absolute; left:0; bottom:0; width:18px; height:18px;
                    cursor:nesw-resize; z-index:2; border-bottom-left-radius:10px;
                    background:linear-gradient(135deg, transparent 50%, rgba(54,69,79,0.24) 50%);
                "></div>
        </div>
    </div>
    <div id="pulsemap-help" style="position:relative;">
        <button id="pulsemap-help-btn" type="button" aria-label="How to read this map"
            aria-expanded="false" aria-controls="pulsemap-help-panel"
            style="
                width:32px; height:32px; border:none; border-radius:999px;
                background:#36454F; color:#fff; font-family:system-ui,sans-serif;
                font-size:16px; font-weight:700; line-height:1; cursor:pointer;
                box-shadow:0 4px 14px rgba(0,0,0,0.18);
            ">?</button>
        <div id="pulsemap-help-panel" style="
            display:none; position:absolute; top:40px; right:0; width:290px;
            background:#fff; border:1px solid rgba(0,0,0,0.08); border-radius:10px;
            box-shadow:0 8px 24px rgba(0,0,0,0.14); padding:12px 14px;
            font-family:Inter,system-ui,sans-serif; font-size:11.5px;
            line-height:1.55; color:#444;
        ">
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
                <div style="font-weight:700; font-size:12px; color:#36454F;">How to read this map</div>
                <button id="pulsemap-help-close" type="button" aria-label="Close help"
                    style="
                        width:22px; height:22px; border:none; border-radius:11px;
                        background:#ebebeb; color:#666; font-size:14px; cursor:pointer;
                        line-height:1; padding:0;
                    ">×</button>
            </div>
            <div style="margin-bottom:10px;">
                <div style="font-weight:600; color:#36454F; margin-bottom:4px;">Highlighted metrics</div>
                <div>Gray cells with a dark left bar are the <strong>top 2 concerns</strong>. We prefer metrics <strong>above the US median</strong>; if fewer than 2 qualify, we show the highest local rates instead.</div>
            </div>
            <div style="margin-bottom:10px;">
                <div style="font-weight:600; color:#36454F; margin-bottom:4px;">Green / orange / red numbers</div>
                <div>Number color uses <strong>fixed severity bands</strong> per metric (not the national median). Green = lower risk, orange = moderate, red = high.</div>
            </div>
            <div style="margin-bottom:10px;">
                <div style="font-weight:600; color:#36454F; margin-bottom:4px;">Map marker colors</div>
                <div>Match the legend (bottom left). Choose <strong>Overall</strong> to color by how many metrics are above the US median, or pick a single metric.</div>
            </div>
            <div style="margin-bottom:10px;">
                <div style="font-weight:600; color:#36454F; margin-bottom:4px;">Equity tier (K-Means)</div>
                <div>Each ZIP is placed in one of <strong>4 clusters</strong> using scikit-learn K-Means on six rates: Asthma, Diabetes, Uninsured, Obesity, Depression, and Food Insecurity. Missing food insecurity values are filled with the dataset median. Clusters are ranked by the <strong>sum of their six average rates</strong> (lowest = Low Risk, highest = Severe Disparity).</div>
            </div>
            <div style="margin-bottom:10px;">
                <div style="font-weight:600; color:#36454F; margin-bottom:4px;">Suggested actions</div>
                <div>Click <strong>View suggested actions</strong> in a ZIP popup to open a centered guide with evidence-based steps for that community's top concerns.</div>
            </div>
            <div style="margin-bottom:10px;">
                <div style="font-weight:600; color:#36454F; margin-bottom:4px;">Health Advocate</div>
                <div>The popup summary reflects the six-metric K-Means tier assignment for that ZIP.</div>
            </div>
            <div style="color:#888; font-size:10.5px;">
                Search by ZIP, city, or state (e.g. <em>WA</em>, <em>Washington</em>, <em>Auburn, WA</em>).
            </div>
        </div>
    </div>
    </div>
    <div id="pulsemap-action-modal" aria-hidden="true" role="presentation">
        <div class="pulsemap-action-modal-backdrop" data-close-modal="true"></div>
        <div class="pulsemap-action-modal-dialog" role="dialog" aria-modal="true" aria-labelledby="pulsemap-action-modal-title">
            <div style="background:#36454F; padding:16px 18px 14px; border-radius:14px 14px 0 0; position:relative;">
                <button id="pulsemap-action-modal-close" type="button" class="pulsemap-action-modal-close"
                    aria-label="Close suggested actions" style="position:absolute; top:12px; right:12px;">×</button>
                <div style="font-family:system-ui,sans-serif; font-size:9px; font-weight:700; letter-spacing:0.8px; text-transform:uppercase; color:#E2E8F0; margin-bottom:6px;">
                    Suggested Actions
                </div>
                <div id="pulsemap-action-modal-title" style="font-family:Georgia,serif; font-size:1.35rem; font-weight:700; color:#fff; line-height:1.2; padding-right:28px;">
                    Community actions
                </div>
                <div id="pulsemap-action-modal-subtitle" style="margin-top:5px; font-family:system-ui,sans-serif; font-size:11px; color:#CBD5E1;"></div>
            </div>
            <div id="pulsemap-action-modal-body" style="padding:14px 16px 10px; font-family:Inter,system-ui,sans-serif; font-size:12px; line-height:1.55; color:#444;"></div>
            <div style="padding:0 16px 16px; display:flex; gap:8px; flex-wrap:wrap;">
                <button id="pulsemap-action-modal-guide-btn" type="button" style="
                    flex:1; min-width:180px; padding:8px 12px; border:1px solid #d4cfc7;
                    border-radius:8px; background:#fff; color:#36454F;
                    font-family:system-ui,sans-serif; font-size:11px; font-weight:600; cursor:pointer;
                ">Browse full Action Guide</button>
                <button id="pulsemap-action-modal-dismiss-btn" type="button" style="
                    padding:8px 12px; border:1px solid #d4cfc7;
                    border-radius:8px; background:#ebebeb; color:#444;
                    font-family:system-ui,sans-serif; font-size:11px; font-weight:600; cursor:pointer;
                ">Close</button>
            </div>
        </div>
    </div>
    """
    health_map.get_root().html.add_child(folium.Element(legend_html))


def add_viewport_marker_script(
    health_map: folium.Map,
    zip_records: list[dict[str, float | int | str]],
    center_zip: int,
) -> None:
    health_map.get_root().header.add_child(
        folium.Element(
            """
            <style>
              .pulsemap-popup .leaflet-popup-content-wrapper {
                background: transparent;
                box-shadow: none;
                border-radius: 0;
                padding: 0;
              }
              .pulsemap-popup .leaflet-popup-content {
                margin: 0;
                line-height: inherit;
              }
              .pulsemap-popup .leaflet-popup-tip {
                background: #faf8f5;
                box-shadow: none;
                border: 1px solid #e8e0d5;
                border-top: none;
                border-left: none;
              }
              .pulsemap-popup a.leaflet-popup-close-button {
                color: #f8fafc;
                font-size: 18px;
                font-weight: 700;
                width: 24px;
                height: 24px;
                top: 10px;
                right: 10px;
                padding: 0;
                line-height: 24px;
                z-index: 10;
              }
              .pulsemap-popup a.leaflet-popup-close-button:hover {
                color: #ffffff;
              }
              .pulsemap-popup.popup-below .leaflet-popup-tip-container {
                top: 0;
                bottom: auto;
                margin-top: -1px;
              }
              .pulsemap-popup.popup-below .leaflet-popup-tip {
                margin: -10px auto 0;
                transform: rotate(180deg);
                border-top: 1px solid #e8e0d5;
                border-bottom: none;
                border-left: none;
              }
              .pulsemap-popup.popup-left .leaflet-popup-tip-container,
              .pulsemap-popup.popup-right .leaflet-popup-tip-container {
                display: none;
              }
              .pulsemap-popup.leaflet-popup {
                z-index: 10050 !important;
                transition: opacity 0.12s ease;
              }
              .pulsemap-popup.pulsemap-popup--positioning {
                opacity: 0 !important;
                pointer-events: none;
              }
              .leaflet-top.leaflet-left {
                top: auto !important;
                left: auto !important;
                bottom: 26px !important;
                right: 6px !important;
              }
              .leaflet-top.leaflet-left .leaflet-control-zoom {
                margin: 0 !important;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
                border-radius: 8px;
                overflow: hidden;
              }
              .leaflet-top.leaflet-left .leaflet-control-zoom a {
                width: 32px;
                height: 32px;
                line-height: 32px;
              }
              #pulsemap-action-modal {
                display: none;
                position: fixed;
                inset: 0;
                z-index: 50000;
                align-items: center;
                justify-content: center;
                padding: 20px;
                pointer-events: none;
              }
              #pulsemap-action-modal.is-open {
                display: flex !important;
                pointer-events: auto;
              }
              #pulsemap-action-modal .pulsemap-action-modal-backdrop {
                position: absolute;
                inset: 0;
                background: rgba(22, 28, 32, 0.52);
              }
              #pulsemap-action-modal .pulsemap-action-modal-dialog {
                position: relative;
                width: min(520px, 100%);
                max-height: min(85vh, 720px);
                overflow: auto;
                background: #faf8f5;
                border: 1px solid #e8e0d5;
                border-radius: 14px;
                box-shadow: 0 18px 48px rgba(0, 0, 0, 0.22);
              }
              #pulsemap-action-modal .pulsemap-action-modal-close {
                width: 28px;
                height: 28px;
                border: none;
                border-radius: 14px;
                background: rgba(255,255,255,0.16);
                color: #fff;
                font-size: 18px;
                line-height: 1;
                cursor: pointer;
              }
              body.pulsemap-modal-open {
                overflow: hidden;
              }
            </style>
            """
        )
    )
    map_var = health_map.get_name()
    zip_json = json.dumps(zip_records, separators=(",", ":"))
    equity_tier_reports_json = json.dumps(EQUITY_TIER_REPORTS)
    national_medians_json = json.dumps(NATIONAL_MEDIANS)
    metric_specs_json = json.dumps(METRIC_SPECS)
    metric_solutions_json = json.dumps(METRIC_SOLUTIONS)
    sdoh_connections_json = json.dumps(SDOH_CONNECTION_INSIGHTS)
    state_names_json = json.dumps(US_STATE_NAMES)

    script = f"""
    (function() {{
        const ZIP_DATA = {zip_json};
        const CENTER_ZIP = {center_zip};
        const EQUITY_TIER_REPORTS = {equity_tier_reports_json};
        const MAX_MARKERS = {MAX_VISIBLE_MARKERS};
        const MAX_CITY_CLUSTERS = {MAX_CITY_CLUSTERS};
        const STATE_ZOOM = {STATE_ZOOM_THRESHOLD};
        const CLUSTER_ZOOM = {CLUSTER_ZOOM_THRESHOLD};
        const ZIP_DETAIL_ZOOM = {ZIP_DETAIL_ZOOM};
        const NATIONAL_MEDIANS = {national_medians_json};
        const METRIC_SPECS = {metric_specs_json};
        const METRIC_SOLUTIONS = {metric_solutions_json};
        const SDOH_CONNECTION_INSIGHTS = {sdoh_connections_json};
        const US_STATE_NAMES = {state_names_json};
        const TOP_CONCERNS_COUNT = {TOP_CONCERNS_COUNT};
        const POPUP_HIGHLIGHT_BG = "{POPUP_HIGHLIGHT_BG}";
        const POPUP_HIGHLIGHT_BORDER = "{POPUP_HIGHLIGHT_BORDER}";
        const POPUP_DIVIDER = "{POPUP_DIVIDER}";

        const filterState = {{
            metric: "a",
            aboveNationalOnly: false,
        }};

        function boot() {{
            if (typeof {map_var} === "undefined") {{
                setTimeout(boot, 50);
                return;
            }}

            const map = {map_var};

        function activeMetricSpec() {{
            if (filterState.metric === "overall") return null;
            return METRIC_SPECS.find((spec) => spec.js === filterState.metric) || METRIC_SPECS[0];
        }}

        function metricsAboveNationalCount(d) {{
            return METRIC_SPECS.reduce((count, spec) => {{
                const value = d[spec.js];
                if (value == null) return count;
                const median = NATIONAL_MEDIANS[spec.label];
                return value > median ? count + 1 : count;
            }}, 0);
        }}

        function markerColorForMetric(d) {{
            if (filterState.metric === "overall") {{
                const count = metricsAboveNationalCount(d);
                if (count >= 3) return "#dc2626";
                if (count >= 1) return "#d97706";
                return "#16a34a";
            }}

            const spec = activeMetricSpec();
            const rate = d[spec.js];
            if (rate == null) return "#9ca3af";
            if (rate > spec.high) return "#dc2626";
            if (rate > spec.medium) return "#d97706";
            return "#16a34a";
        }}

        function passesFilter(d) {{
            if (!filterState.aboveNationalOnly) return true;
            if (filterState.metric === "overall") {{
                return metricsAboveNationalCount(d) > 0;
            }}
            const spec = activeMetricSpec();
            const value = d[spec.js];
            if (value == null) return false;
            const median = NATIONAL_MEDIANS[spec.label];
            return value > median;
        }}

        function updateLegendScale() {{
            const nameEl = document.getElementById("legend-metric-name");
            const lowEl = document.getElementById("legend-low-label");
            const midEl = document.getElementById("legend-mid-label");
            const highEl = document.getElementById("legend-high-label");
            if (filterState.metric === "overall") {{
                if (nameEl) nameEl.textContent = "Overall burden";
                if (lowEl) lowEl.textContent = "0 metrics above US median";
                if (midEl) midEl.textContent = "1–2 metrics above US median";
                if (highEl) highEl.textContent = "3+ metrics above US median";
                return;
            }}

            const spec = activeMetricSpec();
            if (nameEl) nameEl.textContent = spec.label;
            if (lowEl) lowEl.textContent = `${{spec.label}} ≤ ${{spec.medium}}%`;
            if (midEl) midEl.textContent = `${{spec.label}} ${{spec.medium}}–${{spec.high}}%`;
            if (highEl) highEl.textContent = `${{spec.label}} > ${{spec.high}}%`;
        }}

        let cityIndex = null;
        let stateIndex = null;
        let activeSuggestions = [];
        let suggestionIndex = -1;
        let suggestTimer = null;
        let suppressBlurHide = false;

        function locationLabel(d) {{
            return d.st ? `${{d.c}}, ${{d.st}}` : d.c;
        }}

        function setSearchStatus(message, isError = false) {{
            const statusEl = document.getElementById("search-status");
            if (!statusEl) return;
            statusEl.textContent = message || "";
            statusEl.style.color = isError ? "#b45309" : "#666";
        }}

        function updateClearButton() {{
            const clearBtn = document.getElementById("location-search-clear");
            const input = document.getElementById("location-search");
            if (!clearBtn || !input) return;
            clearBtn.style.display = input.value.trim() ? "block" : "none";
        }}

        function clearSearch() {{
            const input = document.getElementById("location-search");
            if (input) {{
                input.value = "";
                input.focus();
            }}
            hideSuggestions();
            setSearchStatus("");
            updateClearButton();
            updateVisibleMarkers();
        }}

        function scheduleSuggestions() {{
            const searchInput = document.getElementById("location-search");
            if (!searchInput) return;
            if (suggestTimer) clearTimeout(suggestTimer);
            suggestTimer = setTimeout(() => {{
                const query = searchInput.value.trim();
                updateClearButton();
                if (!query) {{
                    hideSuggestions();
                    setSearchStatus("");
                    return;
                }}
                if (query.length < 2 && !/^\\d/.test(query)) {{
                    hideSuggestions();
                    return;
                }}
                renderSuggestions(getSuggestions(searchInput.value));
                setSearchStatus("");
            }}, 140);
        }}

        function hideSuggestions() {{
            const box = document.getElementById("search-suggestions");
            if (!box) return;
            box.style.display = "none";
            box.innerHTML = "";
            activeSuggestions = [];
            suggestionIndex = -1;
        }}

        function buildCityIndex() {{
            if (cityIndex) return cityIndex;
            const groups = new Map();
            for (const d of ZIP_DATA) {{
                const key = `${{d.c.toLowerCase()}}|${{d.st || ""}}`;
                if (!groups.has(key)) {{
                    groups.set(key, {{ city: d.c, state: d.st || "", zips: [] }});
                }}
                groups.get(key).zips.push(d);
            }}
            cityIndex = Array.from(groups.values()).map((entry) => {{
                const [lat, lng] = centroid(entry.zips);
                const label = entry.state ? `${{entry.city}}, ${{entry.state}}` : entry.city;
                return {{
                    city: entry.city,
                    state: entry.state,
                    label,
                    count: entry.zips.length,
                    lat,
                    lng,
                    zips: entry.zips,
                }};
            }});
            cityIndex.sort((a, b) => a.label.localeCompare(b.label));
            return cityIndex;
        }}

        function buildStateIndex() {{
            if (stateIndex) return stateIndex;
            const groups = new Map();
            for (const d of ZIP_DATA) {{
                if (!d.st) continue;
                if (!groups.has(d.st)) groups.set(d.st, []);
                groups.get(d.st).push(d);
            }}
            stateIndex = Array.from(groups.entries()).map(([state, zips]) => {{
                const [lat, lng] = centroid(zips);
                const name = US_STATE_NAMES[state] || state;
                return {{
                    state,
                    name,
                    label: `${{name}} (${{state}})`,
                    count: zips.length,
                    cityCount: new Set(zips.map((z) => z.c)).size,
                    lat,
                    lng,
                    zips,
                }};
            }});
            stateIndex.sort((a, b) => a.name.localeCompare(b.name));
            return stateIndex;
        }}

        function isLikelyStateCode(query) {{
            const code = query.trim().toUpperCase();
            return /^[A-Z]{{2}}$/.test(code) && Boolean(US_STATE_NAMES[code]);
        }}

        function parseLocationQuery(query) {{
            const trimmed = query.trim();
            const commaMatch = trimmed.match(/^(.+),\\s*([a-zA-Z]{{2}})$/);
            if (commaMatch) {{
                return {{
                    city: commaMatch[1].trim(),
                    state: commaMatch[2].toUpperCase(),
                }};
            }}
            const spaceMatch = trimmed.match(/^(.+)\\s+([a-zA-Z]{{2}})$/);
            if (spaceMatch) {{
                return {{
                    city: spaceMatch[1].trim(),
                    state: spaceMatch[2].toUpperCase(),
                }};
            }}
            return {{ city: trimmed, state: "" }};
        }}

        function scoreCityEntry(entry, parsed) {{
            const cityLower = entry.city.toLowerCase();
            const qCity = parsed.city.toLowerCase();
            let score = 0;

            if (cityLower === qCity) score += 100;
            else if (cityLower.startsWith(qCity)) score += 60;
            else if (cityLower.includes(qCity)) score += 25;
            else return -1;

            if (parsed.state) {{
                if (entry.state === parsed.state) score += 200;
                else return -1;
            }} else if (entry.label.toLowerCase().includes(parsed.city.toLowerCase())) {{
                score += 10;
            }}

            const homeRecord = ZIP_DATA.find((d) => d.z === CENTER_ZIP);
            if (homeRecord && homeRecord.st && entry.state === homeRecord.st) {{
                score += 30;
            }}
            return score;
        }}

        function scoreStateEntry(entry, query, parsed) {{
            const q = query.trim().toLowerCase();
            const qUpper = query.trim().toUpperCase();
            if (parsed.state && parsed.state !== entry.state) return -1;

            let score = 0;
            if (entry.state === qUpper && q.length === 2) score += 320;
            else if (entry.state.toLowerCase() === q) score += 300;

            const nameLower = entry.name.toLowerCase();
            if (nameLower === q) score += 280;
            else if (nameLower.startsWith(q)) score += 200;
            else if (nameLower.includes(q)) score += 90;
            else if (
                q.length >= 2 &&
                nameLower.split(/\\s+/).some((word) => word.startsWith(q))
            ) {{
                score += 70;
            }} else if (score === 0) {{
                return -1;
            }}

            const homeRecord = ZIP_DATA.find((d) => d.z === CENTER_ZIP);
            if (homeRecord && homeRecord.st && entry.state === homeRecord.st) {{
                score += 25;
            }}
            return score;
        }}

        function getSuggestions(query) {{
            const q = query.trim().toLowerCase();
            if (!q) return [];
            const results = [];

            if (/^\\d/.test(q)) {{
                for (const d of ZIP_DATA) {{
                    if (d.zip.startsWith(q) || d.zip.includes(q)) {{
                        results.push({{
                            type: "zip",
                            label: `${{d.zip}} · ${{locationLabel(d)}}`,
                            zipData: d,
                        }});
                    }}
                    if (results.length >= 10) break;
                }}
                return results;
            }}

            if (isLikelyStateCode(query)) {{
                const entry = buildStateIndex().find(
                    (item) => item.state === query.trim().toUpperCase()
                );
                if (entry) {{
                    return [{{
                        type: "state",
                        label: `${{entry.label}} · ${{entry.count}} ZIPs`,
                        entry,
                    }}];
                }}
            }}

            const parsed = parseLocationQuery(query);
            const ranked = [
                ...buildStateIndex().map((entry) => ({{
                    type: "state",
                    entry,
                    score: scoreStateEntry(entry, query, parsed),
                }})),
                ...buildCityIndex().map((entry) => ({{
                    type: "city",
                    entry,
                    score: scoreCityEntry(entry, parsed),
                }})),
            ]
                .filter((item) => item.score >= 0)
                .sort(
                    (a, b) =>
                        b.score - a.score ||
                        (a.type === "state" && b.type !== "state" ? -1 : 0) ||
                        (a.type !== "state" && b.type === "state" ? 1 : 0) ||
                        a.entry.label.localeCompare(b.entry.label)
                )
                .slice(0, 15);

            for (const item of ranked) {{
                if (item.type === "state") {{
                    results.push({{
                        type: "state",
                        label: `${{item.entry.label}} · ${{item.entry.count}} ZIPs`,
                        entry: item.entry,
                    }});
                }} else {{
                    results.push({{
                        type: "city",
                        label: `${{item.entry.label}} · ${{item.entry.count}} ZIPs`,
                        entry: item.entry,
                    }});
                }}
            }}
            return results;
        }}

        function highlightSuggestionButtons() {{
            const box = document.getElementById("search-suggestions");
            if (!box) return;
            box.querySelectorAll(".search-suggestion-item").forEach((btn, idx) => {{
                btn.style.background = idx === suggestionIndex ? "#EBEBEB" : "#fff";
            }});
        }}

        function zipSuggestionsForCity(entry) {{
            return entry.zips
                .slice()
                .sort((a, b) => a.zip.localeCompare(b.zip))
                .map((d) => ({{
                    type: "zip",
                    label: `${{d.zip}} · ${{locationLabel(d)}}`,
                    zipData: d,
                }}));
        }}

        function renderSuggestions(items, options = {{}}) {{
            const box = document.getElementById("search-suggestions");
            if (!box) return;
            activeSuggestions = items;
            suggestionIndex = -1;
            if (!items.length) {{
                hideSuggestions();
                return;
            }}
            const header = options.title
                ? `<div style="
                    padding:8px 10px; font-size:11px; font-weight:600; color:#666;
                    background:#f8f8f8; border-bottom:1px solid #eee;
                  ">${{options.title}}</div>`
                : "";
            box.innerHTML = header + items.map((item, idx) => `
                <button type="button" data-idx="${{idx}}" class="search-suggestion-item" style="
                    display:block; width:100%; text-align:left; border:none; background:#fff;
                    padding:8px 10px; font-size:12px; color:#36454F; cursor:pointer;
                    border-bottom:1px solid #f0f0f0;
                ">${{item.label}}</button>
            `).join("");
            box.style.display = "block";
            box.querySelectorAll(".search-suggestion-item").forEach((btn) => {{
                btn.addEventListener("mousedown", (event) => {{
                    event.preventDefault();
                    suppressBlurHide = true;
                    applySuggestion(activeSuggestions[Number(btn.dataset.idx)]);
                    suppressBlurHide = false;
                    updateClearButton();
                }});
            }});
        }}

        function showCityZipPicker(entry) {{
            const input = document.getElementById("location-search");
            if (input) input.value = entry.label;
            focusCity(entry);
            renderSuggestions(zipSuggestionsForCity(entry), {{
                title: `Pick a ZIP in ${{entry.label}}`,
            }});
            setSearchStatus(`Choose one of ${{entry.count}} ZIP codes below.`);
        }}

        function focusZip(d) {{
            const targetZoom = Math.max(map.getZoom(), ZIP_DETAIL_ZOOM);
            map.flyTo([d.lat, d.lng], targetZoom, {{ duration: 0.75 }});
            map.once("moveend", () => {{
                markerLayer.clearLayers();
                addIndividualMarker(d);
                markerLayer.eachLayer((layer) => {{
                    if (typeof layer.openPopup === "function") layer.openPopup();
                }});
            }});
        }}

        function focusCity(entry) {{
            const targetZoom = entry.count > 12 ? CLUSTER_ZOOM : CLUSTER_ZOOM + 1;
            map.flyTo([entry.lat, entry.lng], targetZoom, {{ duration: 0.75 }});
            setSearchStatus(`Showing ${{entry.label}} (${{entry.count}} ZIPs).`);
        }}

        function zoomIntoCluster(lat, lng, cityLabel) {{
            map.closePopup();
            const targetZoom = Math.max(map.getZoom() + 1, CLUSTER_ZOOM + 1);
            map.flyTo([lat, lng], targetZoom, {{ duration: 0.75 }});
            setSearchStatus(
                `Zoomed into ${{cityLabel}} — click a ZIP dot for its community profile.`
            );
        }}

        function findZipRecordFromButton(btn) {{
            if (!btn) return null;
            if (btn.dataset.z) {{
                const z = Number(btn.dataset.z);
                if (Number.isFinite(z)) {{
                    return ZIP_DATA.find((row) => row.z === z) || null;
                }}
            }}
            if (btn.dataset.zip) {{
                return ZIP_DATA.find((row) => String(row.zip) === String(btn.dataset.zip)) || null;
            }}
            return null;
        }}

        function closeZipPopup() {{
            if (typeof map !== "undefined" && map && typeof map.closePopup === "function") {{
                map.closePopup();
            }}
        }}

        function handleSuggestedActionsClick(event) {{
            const btn = event.target.closest(".pulsemap-open-actions-btn");
            if (!btn) return;

            event.preventDefault();
            event.stopPropagation();
            if (typeof event.stopImmediatePropagation === "function") {{
                event.stopImmediatePropagation();
            }}

            const zipData = findZipRecordFromButton(btn);
            if (!zipData) return;
            openActionModal(getTopConcerns(zipData), zipData);
        }}

        function bindSuggestedActionButtons(root) {{
            if (!root) return;
            root.querySelectorAll(".pulsemap-open-actions-btn").forEach((btn) => {{
                if (btn.dataset.actionsBound === "true") return;
                btn.dataset.actionsBound = "true";
                if (typeof L !== "undefined" && L.DomEvent) {{
                    L.DomEvent.disableClickPropagation(btn);
                    L.DomEvent.disableScrollPropagation(btn);
                }}
            }});
        }}

        function wirePopupActions(e) {{
            const el = e.popup.getElement();
            if (!el) return;

            const zoomBtn = el.querySelector(".pulsemap-zoom-cluster-btn");
            if (zoomBtn) {{
                zoomBtn.addEventListener("click", (event) => {{
                    event.preventDefault();
                    const lat = Number(zoomBtn.dataset.lat);
                    const lng = Number(zoomBtn.dataset.lng);
                    const city = decodeURIComponent(zoomBtn.dataset.city || "");
                    if (!Number.isFinite(lat) || !Number.isFinite(lng)) return;
                    zoomIntoCluster(lat, lng, city || "this area");
                }});
            }}

            bindSuggestedActionButtons(el);
        }}

        function updateViewModeBadge(zoom) {{
            const badge = document.getElementById("view-mode-badge");
            if (!badge) return;

            if (zoom < STATE_ZOOM) {{
                badge.textContent = "State overview";
                badge.style.background = "#36454F";
            }} else if (zoom < CLUSTER_ZOOM) {{
                badge.textContent = "City clusters";
                badge.style.background = "#4a5f6f";
            }} else {{
                badge.textContent = "ZIP detail";
                badge.style.background = "#2f6b3a";
            }}
        }}

        function markerRadiusForZoom() {{
            const zoom = map.getZoom();
            if (zoom >= 13) return 10;
            if (zoom >= ZIP_DETAIL_ZOOM) return 9;
            if (zoom >= CLUSTER_ZOOM) return 8;
            return 7;
        }}

        function focusState(entry) {{
            const bounds = L.latLngBounds(entry.zips.map((d) => [d.lat, d.lng]));
            map.flyToBounds(bounds, {{
                padding: [48, 48],
                duration: 0.75,
                maxZoom: 7,
            }});
            setSearchStatus(
                `Showing ${{entry.label}} — ${{entry.count}} ZIPs across ${{entry.cityCount}} cities.`
            );
        }}

        function applySuggestion(suggestion) {{
            if (suggestion.type === "state") {{
                hideSuggestions();
                const input = document.getElementById("location-search");
                if (input) input.value = suggestion.entry.label;
                updateClearButton();
                focusState(suggestion.entry);
                return;
            }}

            if (suggestion.type === "zip") {{
                hideSuggestions();
                const input = document.getElementById("location-search");
                if (input) input.value = `${{suggestion.zipData.zip}} · ${{locationLabel(suggestion.zipData)}}`;
                updateClearButton();
                const d = suggestion.zipData;
                setSearchStatus(`Showing ${{locationLabel(d)}} (${{d.zip}}).`);
                focusZip(d);
                return;
            }}

            const entry = suggestion.entry;
            if (entry.count === 1) {{
                applySuggestion({{
                    type: "zip",
                    label: `${{entry.zips[0].zip}} · ${{locationLabel(entry.zips[0])}}`,
                    zipData: entry.zips[0],
                }});
                return;
            }}
            showCityZipPicker(entry);
        }}

        function runSearch() {{
            const input = document.getElementById("location-search");
            if (!input) return;
            const query = input.value.trim();
            if (!query) {{
                setSearchStatus("Enter a ZIP code, city, or state.", true);
                return;
            }}

            if (suggestionIndex >= 0 && activeSuggestions[suggestionIndex]) {{
                applySuggestion(activeSuggestions[suggestionIndex]);
                return;
            }}

            const suggestions = getSuggestions(query);
            if (suggestions.length === 1) {{
                applySuggestion(suggestions[0]);
                return;
            }}
            if (suggestions.length > 1) {{
                renderSuggestions(suggestions);
                setSearchStatus("Pick a location from the list.", true);
                return;
            }}
            hideSuggestions();
            setSearchStatus(`No matches for "${{query}}".`, true);
        }}

        function setHelpPanelOpen(open) {{
            const helpBtn = document.getElementById("pulsemap-help-btn");
            const helpPanel = document.getElementById("pulsemap-help-panel");
            if (!helpBtn || !helpPanel) return;
            helpPanel.style.display = open ? "block" : "none";
            helpBtn.setAttribute("aria-expanded", open ? "true" : "false");
            helpBtn.style.background = open ? "#2f6b3a" : "#36454F";
            if (open) setGuidePanelOpen(false);
        }}

        const GUIDE_PANEL_SIZE_KEY = "pulsemap-guide-panel-size";
        const GUIDE_PANEL_DEFAULT = {{ width: 340, height: 520 }};
        const GUIDE_PANEL_MIN = {{ width: 280, height: 240 }};
        const GUIDE_PANEL_MAX = {{ width: 720, widthRatio: 0.9, heightRatio: 0.9 }};

        function clampGuidePanelSize(width, height) {{
            const maxWidth = Math.min(
                GUIDE_PANEL_MAX.width,
                Math.round(window.innerWidth * GUIDE_PANEL_MAX.widthRatio)
            );
            const maxHeight = Math.round(window.innerHeight * GUIDE_PANEL_MAX.heightRatio);
            return {{
                width: Math.max(GUIDE_PANEL_MIN.width, Math.min(maxWidth, Math.round(width))),
                height: Math.max(GUIDE_PANEL_MIN.height, Math.min(maxHeight, Math.round(height))),
            }};
        }}

        function loadGuidePanelSize() {{
            try {{
                const raw = localStorage.getItem(GUIDE_PANEL_SIZE_KEY);
                if (!raw) return {{ ...GUIDE_PANEL_DEFAULT }};
                const parsed = JSON.parse(raw);
                return clampGuidePanelSize(
                    parsed.width || GUIDE_PANEL_DEFAULT.width,
                    parsed.height || GUIDE_PANEL_DEFAULT.height
                );
            }} catch (error) {{
                return {{ ...GUIDE_PANEL_DEFAULT }};
            }}
        }}

        function applyGuidePanelSize(panel) {{
            if (!panel) return;
            const size = loadGuidePanelSize();
            panel.style.width = `${{size.width}}px`;
            panel.style.height = `${{size.height}}px`;
        }}

        function saveGuidePanelSize(panel) {{
            if (!panel) return;
            const size = clampGuidePanelSize(panel.offsetWidth, panel.offsetHeight);
            panel.style.width = `${{size.width}}px`;
            panel.style.height = `${{size.height}}px`;
            localStorage.setItem(GUIDE_PANEL_SIZE_KEY, JSON.stringify(size));
        }}

        function isGuidePanelOpen() {{
            const guidePanel = document.getElementById("pulsemap-guide-panel");
            if (!guidePanel) return false;
            return guidePanel.style.display !== "none" && guidePanel.style.display !== "";
        }}

        function setGuidePanelOpen(open, highlightLabel) {{
            const guideBtn = document.getElementById("pulsemap-guide-btn");
            const guidePanel = document.getElementById("pulsemap-guide-panel");
            if (!guideBtn || !guidePanel) return;
            guidePanel.style.display = open ? "flex" : "none";
            guideBtn.setAttribute("aria-expanded", open ? "true" : "false");
            guideBtn.style.background = open ? "#2f6b3a" : "#36454F";
            if (open) {{
                applyGuidePanelSize(guidePanel);
                setHelpPanelOpen(false);
                if (highlightLabel) highlightGuideSection(highlightLabel);
            }}
        }}

        function wireGuidePanelResize() {{
            const panel = document.getElementById("pulsemap-guide-panel");
            const handle = document.getElementById("pulsemap-guide-resize");
            if (!panel || !handle) return;

            applyGuidePanelSize(panel);

            let startX = 0;
            let startY = 0;
            let startWidth = 0;
            let startHeight = 0;

            const onMove = (event) => {{
                const deltaX = event.clientX - startX;
                const deltaY = event.clientY - startY;
                const next = clampGuidePanelSize(startWidth - deltaX, startHeight + deltaY);
                panel.style.width = `${{next.width}}px`;
                panel.style.height = `${{next.height}}px`;
            }};

            const onUp = () => {{
                document.removeEventListener("mousemove", onMove);
                document.removeEventListener("mouseup", onUp);
                document.body.style.cursor = "";
                document.body.style.userSelect = "";
                saveGuidePanelSize(panel);
            }};

            handle.addEventListener("mousedown", (event) => {{
                event.preventDefault();
                event.stopPropagation();
                startX = event.clientX;
                startY = event.clientY;
                startWidth = panel.offsetWidth;
                startHeight = panel.offsetHeight;
                document.body.style.cursor = "nesw-resize";
                document.body.style.userSelect = "none";
                document.addEventListener("mousemove", onMove);
                document.addEventListener("mouseup", onUp);
            }});

            window.addEventListener("resize", () => {{
                if (!isGuidePanelOpen()) return;
                saveGuidePanelSize(panel);
            }});
        }}

        function highlightGuideSection(label) {{
            const panel = document.getElementById("pulsemap-guide-panel");
            if (!panel || !label) return;
            const section = panel.querySelector(`[data-guide-metric="${{label}}"]`);
            if (!section) return;
            panel.querySelectorAll("[data-guide-metric]").forEach((node) => {{
                node.style.boxShadow = "";
                node.style.borderColor = "#ede8e1";
            }});
            section.style.borderColor = "#36454F";
            section.style.boxShadow = "0 0 0 1px #36454F";
            section.scrollIntoView({{ block: "nearest", behavior: "smooth" }});
        }}

        function buildResourceLinkHtml(sol) {{
            const link = sol && sol.resource_link;
            if (!link || !link.url) return "";
            const label = link.label || link.url;
            return `
      <div style="margin-top:8px; font-size:10.5px; line-height:1.45;">
        <a href="${{escapeHtml(link.url)}}" target="_blank" rel="noopener"
           style="color:#2f6b3a; font-weight:600; text-decoration:none;">
          ${{escapeHtml(label)}} ↗
        </a>
      </div>`;
        }}

        function buildGuideSectionsHtml() {{
            return METRIC_SPECS.map((spec) => {{
                const sol = METRIC_SOLUTIONS[spec.label];
                if (!sol) return "";

                const actions = (sol.actions || [])
                    .map(
                        (action) =>
                            `<li style="margin-bottom:4px;">${{escapeHtml(action)}}</li>`
                    )
                    .join("");
                const summary = sol.summary
                    ? `<div style="margin-bottom:6px; color:#555;">${{escapeHtml(sol.summary)}}</div>`
                    : "";
                const connects = Array.isArray(sol.connects_with) && sol.connects_with.length
                    ? `<div style="margin-bottom:6px; font-size:10px; color:#6b7280;">
                        Connects with: ${{sol.connects_with.map((item) => escapeHtml(item)).join(", ")}}
                      </div>`
                    : "";

                return `
    <section data-guide-metric="${{escapeHtml(spec.label)}}" style="
      margin-bottom:10px; padding:10px 11px; border:1px solid #ede8e1;
      border-radius:8px; background:#faf8f5;
    ">
      <div style="font-weight:700; color:#36454F; margin-bottom:5px;">
        ${{spec.emoji}} ${{escapeHtml(spec.label)}}
      </div>
      ${{summary}}
      ${{connects}}
      <ul style="margin:0; padding-left:16px; color:#4a2c10;">${{actions}}</ul>
      ${{buildResourceLinkHtml(sol)}}
      <div style="margin-top:6px; font-size:9.5px; color:#a89880; line-height:1.45;">
        Source: ${{escapeHtml(sol.source || "")}}
      </div>
    </section>`;
            }}).join("");
        }}

        function populateGuidePanel() {{
            const root = document.getElementById("pulsemap-guide-sections");
            if (!root) return;
            root.innerHTML = buildGuideSectionsHtml();
        }}

        function openActionGuide(highlightLabel) {{
            closeZipPopup();
            setGuidePanelOpen(true, highlightLabel || "");
            const guidePanel = document.getElementById("pulsemap-guide-panel");
            if (guidePanel) {{
                guidePanel.scrollIntoView({{ block: "nearest", behavior: "smooth" }});
            }}
        }}

        function stopModalClick(event) {{
            event.preventDefault();
            event.stopPropagation();
            if (typeof event.stopImmediatePropagation === "function") {{
                event.stopImmediatePropagation();
            }}
        }}

        function buildConnectionInsightHtml(concerns, forModal = false) {{
            if (!concerns.length) return "";

            const labels = new Set(concerns.map((item) => item.label));
            const matches = SDOH_CONNECTION_INSIGHTS.filter((entry) =>
                (entry.requires || []).every((label) => labels.has(label))
            );
            if (!matches.length) return "";

            const lines = matches
                .slice(0, 2)
                .map(
                    (entry) =>
                        `<li style="margin-bottom:4px;">${{escapeHtml(entry.text || "")}}</li>`
                )
                .join("");

            const wrapPad = forModal ? "0 0 14px" : "0 14px 10px";
            return `
  <div style="padding: ${{wrapPad}};">
    <div style="
      background:#f4f7f4; border:1px solid #dfe8df; border-left:3px solid #2f6b3a;
      border-radius:0 8px 8px 0; padding:9px 11px;
    ">
      <div style="
        font-family:system-ui,sans-serif; font-size:8.5px; font-weight:700;
        letter-spacing:0.8px; text-transform:uppercase; color:#2f6b3a; margin-bottom:5px;
      ">Connected SDOH insight</div>
      <ul style="
        margin:0; padding-left:16px;
        font-family:system-ui,sans-serif; font-size:10px; line-height:1.5; color:#3d4f3d;
      ">${{lines}}</ul>
    </div>
  </div>`;
        }}

        function buildMetricActionBlock(concern) {{
            const sol = METRIC_SOLUTIONS[concern.label];
            if (!sol || !sol.actions || !sol.actions.length) return "";

            const actions = sol.actions
                .map(
                    (action) => `<li style="margin-bottom:4px;">${{escapeHtml(action)}}</li>`
                )
                .join("");
            const summary = sol.summary
                ? `<div style="margin-bottom:6px; color:#555; font-size:11px;">${{escapeHtml(sol.summary)}}</div>`
                : "";
            const connects = Array.isArray(sol.connects_with) && sol.connects_with.length
                ? `<div style="margin-bottom:6px; font-size:10px; color:#6b7280;">Connects with: ${{sol.connects_with.map((item) => escapeHtml(item)).join(", ")}}</div>`
                : "";

            return `
    <section style="
      margin-bottom:12px; padding:11px 12px; border:1px solid #ede8e1;
      border-radius:8px; background:#fff;
    ">
      <div style="font-weight:700; color:#36454F; margin-bottom:5px; font-size:12px;">
        ${{concern.emoji}} ${{escapeHtml(concern.label)}} · <span style="color:#78350f;">${{escapeHtml(concern.rate)}}</span>
      </div>
      ${{summary}}
      ${{connects}}
      <ul style="margin:0; padding-left:18px; color:#4a2c10; font-size:11px; line-height:1.55;">${{actions}}</ul>
      ${{buildResourceLinkHtml(sol)}}
      <div style="margin-top:6px; font-size:9.5px; color:#a89880; line-height:1.45;">
        Source: ${{escapeHtml(sol.source || "")}}
      </div>
    </section>`;
        }}

        function buildActionModalBodyHtml(concerns) {{
            if (!concerns.length) {{
                return `<div style="color:#666;">No prioritized concerns available for this community.</div>`;
            }}

            const connectionHtml = buildConnectionInsightHtml(concerns, true);
            const blocks = concerns.map((concern) => buildMetricActionBlock(concern)).filter(Boolean).join("");
            return `${{connectionHtml}}${{blocks}}`;
        }}

        function ensureActionModalRoot() {{
            const modal = document.getElementById("pulsemap-action-modal");
            if (!modal) return null;
            if (modal.parentElement !== document.body || modal !== document.body.lastElementChild) {{
                document.body.appendChild(modal);
            }}
            return modal;
        }}

        function openActionModal(concerns, zipData) {{
            closeZipPopup();
            const modal = ensureActionModalRoot();
            const title = document.getElementById("pulsemap-action-modal-title");
            const subtitle = document.getElementById("pulsemap-action-modal-subtitle");
            const body = document.getElementById("pulsemap-action-modal-body");
            if (!modal || !title || !subtitle || !body) return;

            const concernLabels = concerns
                .map((item) => `${{item.emoji}} ${{item.label}}`)
                .join(" · ");
            title.textContent = zipData && zipData.c ? `${{zipData.c}}` : "Community actions";
            subtitle.textContent = concernLabels
                ? `ZIP ${{zipData?.zip || ""}} · Top concerns: ${{concernLabels}}`
                : `ZIP ${{zipData?.zip || ""}}`;
            body.innerHTML = buildActionModalBodyHtml(concerns);
            modal.classList.add("is-open");
            modal.style.display = "flex";
            modal.setAttribute("aria-hidden", "false");
            document.body.classList.add("pulsemap-modal-open");
            modal.dataset.highlightMetric = concerns[0]?.label || "";
        }}

        function closeActionModal() {{
            const modal = document.getElementById("pulsemap-action-modal");
            if (!modal) return;
            modal.classList.remove("is-open");
            modal.style.display = "";
            modal.setAttribute("aria-hidden", "true");
            document.body.classList.remove("pulsemap-modal-open");
        }}

        function wireActionModalEvents() {{
            ensureActionModalRoot();
            const modal = document.getElementById("pulsemap-action-modal");
            if (!modal || modal.dataset.wired === "true") return;
            modal.dataset.wired = "true";

            document.addEventListener("click", handleSuggestedActionsClick, true);

            const closeBtn = document.getElementById("pulsemap-action-modal-close");
            const dismissBtn = document.getElementById("pulsemap-action-modal-dismiss-btn");
            const guideBtn = document.getElementById("pulsemap-action-modal-guide-btn");
            const backdrop = modal.querySelector("[data-close-modal]");

            if (closeBtn) {{
                closeBtn.addEventListener("click", (event) => {{
                    stopModalClick(event);
                    closeActionModal();
                }});
            }}
            if (dismissBtn) {{
                dismissBtn.addEventListener("click", (event) => {{
                    stopModalClick(event);
                    closeActionModal();
                }});
            }}
            if (backdrop) {{
                backdrop.addEventListener("click", (event) => {{
                    stopModalClick(event);
                    closeActionModal();
                }});
            }}
            if (guideBtn) {{
                guideBtn.addEventListener("click", (event) => {{
                    stopModalClick(event);
                    const highlight = modal.dataset.highlightMetric || "";
                    closeActionModal();
                    requestAnimationFrame(() => openActionGuide(highlight));
                }});
            }}
            modal.querySelector(".pulsemap-action-modal-dialog")?.addEventListener("click", stopModalClick);
        }}

        function wireControlEvents() {{
            const searchBtn = document.getElementById("location-search-btn");
            const searchInput = document.getElementById("location-search");
            const clearBtn = document.getElementById("location-search-clear");
            const metricSelect = document.getElementById("metric-filter");
            const aboveNationalToggle = document.getElementById("above-national-filter");
            const backBtn = document.getElementById("pulsemap-back-btn");
            const helpBtn = document.getElementById("pulsemap-help-btn");
            const helpPanel = document.getElementById("pulsemap-help-panel");
            const helpClose = document.getElementById("pulsemap-help-close");
            const guideBtn = document.getElementById("pulsemap-guide-btn");
            const guidePanel = document.getElementById("pulsemap-guide-panel");
            const guideClose = document.getElementById("pulsemap-guide-close");

            populateGuidePanel();
            wireGuidePanelResize();
            wireActionModalEvents();

            if (backBtn) {{
                backBtn.addEventListener("click", () => {{
                    if (window.history.length > 1) {{
                        history.back();
                        return;
                    }}
                    window.location.href = "{HOME_PAGE}";
                }});
            }}

            if (searchBtn) {{
                searchBtn.addEventListener("mousedown", (event) => {{
                    event.preventDefault();
                    suppressBlurHide = true;
                }});
                searchBtn.addEventListener("click", () => {{
                    suppressBlurHide = false;
                    runSearch();
                }});
            }}
            if (clearBtn) {{
                clearBtn.addEventListener("mousedown", (event) => {{
                    event.preventDefault();
                    suppressBlurHide = true;
                }});
                clearBtn.addEventListener("click", () => {{
                    suppressBlurHide = false;
                    clearSearch();
                }});
            }}
            if (searchInput) {{
                searchInput.addEventListener("input", scheduleSuggestions);
                searchInput.addEventListener("focus", () => {{
                    suppressBlurHide = false;
                    updateClearButton();
                    const query = searchInput.value.trim();
                    if (query.length >= 2 || /^\\d/.test(query)) {{
                        scheduleSuggestions();
                    }}
                }});
                searchInput.addEventListener("keydown", (event) => {{
                    if (event.key === "ArrowDown") {{
                        if (!activeSuggestions.length) return;
                        event.preventDefault();
                        suggestionIndex = Math.min(
                            suggestionIndex + 1,
                            activeSuggestions.length - 1
                        );
                        highlightSuggestionButtons();
                        return;
                    }}
                    if (event.key === "ArrowUp") {{
                        if (!activeSuggestions.length) return;
                        event.preventDefault();
                        suggestionIndex = Math.max(suggestionIndex - 1, 0);
                        highlightSuggestionButtons();
                        return;
                    }}
                    if (event.key === "Escape") {{
                        hideSuggestions();
                        return;
                    }}
                    if (event.key === "Enter") runSearch();
                }});
                searchInput.addEventListener("blur", () => {{
                    setTimeout(() => {{
                        if (suppressBlurHide) return;
                        hideSuggestions();
                    }}, 120);
                }});
            }}
            if (metricSelect) {{
                metricSelect.addEventListener("change", () => {{
                    filterState.metric = metricSelect.value;
                    updateLegendScale();
                    updateVisibleMarkers();
                }});
            }}
            if (aboveNationalToggle) {{
                aboveNationalToggle.addEventListener("change", () => {{
                    filterState.aboveNationalOnly = aboveNationalToggle.checked;
                    updateVisibleMarkers();
                }});
            }}
            if (helpBtn && helpPanel) {{
                helpBtn.addEventListener("click", (event) => {{
                    event.stopPropagation();
                    const isOpen = helpPanel.style.display === "block";
                    setHelpPanelOpen(!isOpen);
                }});
            }}
            if (helpClose) {{
                helpClose.addEventListener("click", (event) => {{
                    event.stopPropagation();
                    setHelpPanelOpen(false);
                }});
            }}
            if (guideBtn && guidePanel) {{
                guideBtn.addEventListener("click", (event) => {{
                    event.stopPropagation();
                    setGuidePanelOpen(!isGuidePanelOpen());
                }});
            }}
            if (guideClose) {{
                guideClose.addEventListener("click", (event) => {{
                    event.stopPropagation();
                    setGuidePanelOpen(false);
                }});
            }}
            document.addEventListener("click", (event) => {{
                if (event.target.closest("#pulsemap-action-modal")) return;

                const toolsRoot = document.getElementById("pulsemap-tools");
                const helpPanelEl = document.getElementById("pulsemap-help-panel");
                const helpOpen = helpPanelEl && helpPanelEl.style.display === "block";
                const guideOpen = isGuidePanelOpen();
                if (!helpOpen && !guideOpen) return;
                if (toolsRoot && !toolsRoot.contains(event.target)) {{
                    if (helpOpen) setHelpPanelOpen(false);
                    if (guideOpen) setGuidePanelOpen(false);
                }}
            }});
            document.addEventListener("keydown", (event) => {{
                if (event.key === "Escape") {{
                    const modal = document.getElementById("pulsemap-action-modal");
                    if (modal && modal.classList.contains("is-open")) {{
                        closeActionModal();
                        return;
                    }}
                    setHelpPanelOpen(false);
                    setGuidePanelOpen(false);
                }}
            }});
            updateLegendScale();
            updateClearButton();
        }}

        function metricColors(rate, high, medium) {{
            if (rate > high) return ["#7f1d1d", "#fef2f2", "#fca5a5"];
            if (rate > medium) return ["#78350f", "#fffbeb", "#fcd34d"];
            return ["#14532d", "#f0fdf4", "#86efac"];
        }}

        function formatRate(value) {{
            return value == null ? "N/A" : `${{value}}%`;
        }}

        function metricColorsFromRate(rateStr, high, medium) {{
            if (rateStr === "N/A") return ["#6b7280", "#f3f4f6", "#d1d5db"];
            return metricColors(parseFloat(rateStr), high, medium);
        }}

        function getTopConcerns(d) {{
            const scored = METRIC_SPECS.map((spec) => {{
                const value = d[spec.js];
                if (value == null) return null;
                const median = NATIONAL_MEDIANS[spec.label];
                const elevation = median ? ((value - median) / median) * 100 : 0;
                return {{
                    emoji: spec.emoji,
                    label: spec.label,
                    value,
                    rate: formatRate(value),
                    elevation,
                    aboveNational: elevation > 0,
                }};
            }}).filter(Boolean);

            const aboveNational = scored
                .filter((item) => item.aboveNational)
                .sort((a, b) => b.elevation - a.elevation);
            if (aboveNational.length >= TOP_CONCERNS_COUNT) {{
                return aboveNational.slice(0, TOP_CONCERNS_COUNT).map((item) => ({{
                    ...item,
                    mode: "national",
                }}));
            }}

            return scored
                .sort((a, b) => b.value - a.value)
                .slice(0, TOP_CONCERNS_COUNT)
                .map((item) => ({{
                    ...item,
                    mode: "local",
                }}));
        }}

        function topConcernsBanner(concerns) {{
            if (!concerns.length) return "";
            const title = concerns[0].mode === "national"
                ? "Top Concerns vs National"
                : "Highest Local Rates";
            const pills = concerns.map((item) => {{
                if (item.mode === "national") {{
                    return `<span style="display:inline-flex;align-items:center;gap:4px;background:#fff;border:1px solid #f5dbb8;border-radius:14px;padding:3px 9px;font-family:system-ui,sans-serif;font-size:10px;color:#4a2c10;">${{item.emoji}} ${{item.label}} <strong>${{item.rate}}</strong> <span style="color:#c46b39;">+${{item.elevation.toFixed(0)}}% vs US</span></span>`;
                }}
                return `<span style="display:inline-flex;align-items:center;gap:4px;background:#fff;border:1px solid #f5dbb8;border-radius:14px;padding:3px 9px;font-family:system-ui,sans-serif;font-size:10px;color:#4a2c10;">${{item.emoji}} ${{item.label}} <strong>${{item.rate}}</strong></span>`;
            }}).join("");
            return `<div style="padding:9px 12px 10px;background:${{POPUP_HIGHLIGHT_BG}};border-bottom:1px dashed ${{POPUP_HIGHLIGHT_BORDER}};">
              <div style="font-family:system-ui,sans-serif;font-size:8.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.9px;color:#36454F;margin-bottom:6px;">${{title}}</div>
              <div style="display:flex;flex-wrap:wrap;gap:6px;">${{pills}}</div>
            </div>`;
        }}

        function metricCell(emoji, label, rateStr, colors, highlight = false) {{
            const textColor = colors[0];
            const cellStyle = highlight
                ? `background:${{POPUP_HIGHLIGHT_BG}};border:1px solid ${{POPUP_HIGHLIGHT_BORDER}};border-left:3px solid {POPUP_ACCENT};`
                : "background:#fff;border:1px solid #ede8e1;box-shadow:0 1px 2px rgba(0,0,0,0.04);";
            return `<div style="
              ${{cellStyle}}border-radius:8px;padding:8px;
            ">
              <div style="
                font-family:system-ui,sans-serif;font-size:10px;font-weight:600;
                color:#2d1b0e;margin-bottom:5px;
              ">${{emoji}} ${{label}}</div>
              <div style="
                font-family:system-ui,sans-serif;font-size:13px;font-weight:700;
                color:${{textColor}};line-height:1.2;
              ">${{rateStr}}</div>
            </div>`;
        }}

        function avgRate(zips, field) {{
            const vals = zips.map((d) => d[field]).filter((v) => v != null);
            if (!vals.length) return "N/A";
            return `${{(vals.reduce((s, v) => s + v, 0) / vals.length).toFixed(1)}}%`;
        }}

        function buildMlReport(city, tier) {{
            const template = EQUITY_TIER_REPORTS[tier] || EQUITY_TIER_REPORTS["Moderate Risk"];
            return template
                .replace(/\\{{city\\}}/g, city)
                .replace(/\\{{tier\\}}/g, tier);
        }}

        function escapeHtml(text) {{
            return String(text)
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;");
        }}

        function buildSolutionsCtaHtml(concerns, zipRecord) {{
            if (!concerns.length) return "";

            const labels = concerns
                .map((item) => `${{item.emoji}} ${{escapeHtml(item.label)}}`)
                .join(" · ");

            return `
  <div style="
    margin: 0 14px;
    border: none;
    border-top: 1px dashed {POPUP_DIVIDER};
  "></div>
  <div style="padding: 10px 14px 4px;">
    <button type="button" class="pulsemap-open-actions-btn"
      data-z="${{escapeHtml(String(zipRecord.z))}}"
      data-zip="${{escapeHtml(String(zipRecord.zip))}}"
      style="
        width:100%; padding:10px 12px; border:none; border-radius:8px;
        background:#36454F; color:#fff; cursor:pointer; text-align:left;
        box-shadow:0 2px 8px rgba(54,69,79,0.22);
      ">
      <div style="
        font-family:system-ui,sans-serif; font-size:11px; font-weight:700;
        margin-bottom:3px;
      ">View suggested actions →</div>
      <div style="
        font-family:system-ui,sans-serif; font-size:9.5px; font-weight:500;
        color:#E2E8F0; line-height:1.45;
      ">Evidence-based steps for ${{labels}}</div>
    </button>
  </div>`;
        }}

        function buildPopup(d) {{
            const aiReport = buildMlReport(d.c, d.t || "Moderate Risk");
            const topConcerns = getTopConcerns(d);
            const highlightLabels = new Set(topConcerns.map((item) => item.label));
            const topConcernsHtml = topConcernsBanner(topConcerns);
            const solutionsCtaHtml = buildSolutionsCtaHtml(topConcerns, d);
            const metricCells = [
                metricCell("🫁", "Asthma", formatRate(d.a), metricColorsFromRate(formatRate(d.a), 15, 10), highlightLabels.has("Asthma")),
                metricCell("🩸", "Diabetes", formatRate(d.d), metricColorsFromRate(formatRate(d.d), 12, 8), highlightLabels.has("Diabetes")),
                metricCell("🛡️", "Uninsured", formatRate(d.u), metricColorsFromRate(formatRate(d.u), 20, 12), highlightLabels.has("Uninsured")),
                metricCell("⚖️", "Obesity", formatRate(d.o), metricColorsFromRate(formatRate(d.o), 35, 25), highlightLabels.has("Obesity")),
                metricCell("🧠", "Depression", formatRate(d.dp), metricColorsFromRate(formatRate(d.dp), 22, 15), highlightLabels.has("Depression")),
                metricCell("🍎", "Food Insecurity", formatRate(d.fi), metricColorsFromRate(formatRate(d.fi), 15, 10), highlightLabels.has("Food Insecurity")),
            ].join("");

            return `
<div style="
  font-family: 'Georgia', 'Times New Roman', serif;
  width: 320px;
  background: #faf8f5;
  border-radius: 12px;
  box-shadow:
    0 2px 4px rgba(0,0,0,0.04),
    0 6px 18px rgba(0,0,0,0.08),
    0 1px 0px #e8e0d5 inset;
  overflow: hidden;
  border: 1px solid #e8e0d5;
">
  <div style="
    background: #36454F;
    padding: 14px 34px 11px 14px;
    position: relative;
    overflow: hidden;
  ">
    <div style="
      position: absolute; inset: 0;
      background-image:
        radial-gradient(ellipse at 80% 0%, rgba(255,255,255,0.1) 0%, transparent 60%),
        radial-gradient(ellipse at 10% 100%, rgba(255,255,255,0.05) 0%, transparent 50%);
      pointer-events: none;
    "></div>
    <div style="
      display: inline-flex; align-items: center; gap: 5px;
      background: rgba(255,255,255,0.14);
      border: 1px solid rgba(255,255,255,0.3);
      border-radius: 16px;
      padding: 2px 8px 2px 6px;
      margin-bottom: 7px;
    ">
      <span style="font-size:9px; line-height:1;">📍</span>
      <span style="
        font-family: system-ui, sans-serif;
        font-size: 9px; font-weight: 700;
        letter-spacing: 0.7px; text-transform: uppercase;
        color: #F8FAFC;
      ">Community Profile</span>
    </div>
    <div style="
      font-size: 20px; font-weight: 700; line-height: 1.1;
      color: #FFFFFF;
      letter-spacing: -0.4px;
      margin-bottom: 3px;
    ">${{d.c}}</div>
    <div style="
      font-family: system-ui, sans-serif;
      font-size: 10px; font-weight: 500; color: #E2E8F0;
      letter-spacing: 0.2px;
    ">ZIP ${{d.zip}} · ${{d.st || "US"}} · Pop. ${{d.p}}</div>
  </div>
  <div style="border-top: 1px dashed {POPUP_DIVIDER};"></div>
  ${{topConcernsHtml}}
  <div style="padding: 12px 12px 8px;">
    <div style="
      font-family: system-ui, sans-serif;
      font-size: 8.5px; font-weight: 700;
      text-transform: uppercase; letter-spacing: 1px;
      color: #36454F; margin-bottom: 10px;
    ">Key Health Indicators</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
      ${{metricCells}}
    </div>
  </div>
  ${{solutionsCtaHtml}}
  <div style="
    margin: 0 14px;
    border: none;
    border-top: 1px dashed {POPUP_DIVIDER};
  "></div>
  <div style="padding: 10px 14px 14px;">
    <div style="
      display: flex; align-items: center; gap: 6px;
      margin-bottom: 8px;
    ">
      <div style="
        width:20px; height:20px;
        background:#36454F;
        color:#FFFFFF;
        border-radius:5px;
        display:flex; align-items:center; justify-content:center;
        font-size:11px; line-height:1;
        flex-shrink:0;
      ">♡</div>
      <span style="
        font-family: system-ui, sans-serif;
        font-size:9px; font-weight:700;
        letter-spacing:0.8px; text-transform:uppercase;
        color:#36454F;
      ">Health Advocate · AI</span>
      <div style="flex:1; height:1px; background:{POPUP_DIVIDER};"></div>
    </div>
    <div style="
      background: {POPUP_HIGHLIGHT_BG};
      border: 1px solid {POPUP_HIGHLIGHT_BORDER};
      border-left: 3px solid {POPUP_ACCENT};
      border-radius: 0 8px 8px 0;
      padding: 9px 11px;
      position: relative;
    ">
      <div style="
        position:absolute; top:4px; right:8px;
        font-size:28px; line-height:1;
        color:{POPUP_HIGHLIGHT_BORDER}; font-family:Georgia,serif;
        pointer-events:none; user-select:none;
      ">&ldquo;</div>
      <div style="
        font-family: Georgia, serif;
        font-size: 11.5px;
        line-height: 1.6;
        color: #4a2c10;
        position: relative;
        z-index: 1;
      ">${{aiReport}}</div>
    </div>
    <div style="
      margin-top: 8px;
      text-align: right;
      font-family: system-ui, sans-serif;
      font-size: 8.5px;
      color: #c4b5a0;
      letter-spacing: 0.2px;
    ">{PRODUCT_TAGLINE} · GatorHack 2026</div>
  </div>
</div>`;
        }}

        function groupByCity(zips) {{
            const groups = {{}};
            for (const d of zips) {{
                if (!groups[d.c]) groups[d.c] = [];
                groups[d.c].push(d);
            }}
            return groups;
        }}

        function groupByState(zips) {{
            const groups = {{}};
            for (const d of zips) {{
                const key = d.st || "—";
                if (!groups[key]) groups[key] = [];
                groups[key].push(d);
            }}
            return groups;
        }}

        function centroid(zips) {{
            const lat = zips.reduce((sum, d) => sum + d.lat, 0) / zips.length;
            const lng = zips.reduce((sum, d) => sum + d.lng, 0) / zips.length;
            return [lat, lng];
        }}

        function buildClusterIcon(count) {{
            const size = count > 99 ? 44 : count > 9 ? 38 : 34;
            return L.divIcon({{
                className: "",
                html: `<div style="
                    width:${{size}}px;height:${{size}}px;
                    background:linear-gradient(145deg,#2d1b0e 0%,#4a2c10 100%);
                    border:2.5px solid #e8a97e;
                    border-radius:50%;
                    display:flex;align-items:center;justify-content:center;
                    color:#fdf6ee;
                    font-family:system-ui,sans-serif;
                    font-weight:800;
                    font-size:${{count > 99 ? 11 : 13}}px;
                    box-shadow:0 3px 10px rgba(45,27,14,0.35);
                    letter-spacing:-0.3px;
                ">${{count}}</div>`,
                iconSize: [size, size],
                iconAnchor: [size / 2, size / 2],
            }});
        }}

        function buildClusterPopup(city, zips, lat, lng) {{
            const count = zips.length;
            const avgAsthma = avgRate(zips, "a");
            const avgDiabetes = avgRate(zips, "d");
            const avgUninsured = avgRate(zips, "u");
            const avgObesity = avgRate(zips, "o");
            const avgDepression = avgRate(zips, "dp");
            const avgFoodInsecurity = avgRate(zips, "fi");
            const zipPreview = zips.slice(0, 5).map((d) => d.zip).join(", ");
            const zipExtra = count > 5 ? `, +${{count - 5}} more` : "";

            return `
<div style="
  font-family:'Georgia','Times New Roman',serif;
  width:248px;
  background:#faf8f5;
  border-radius:12px;
  box-shadow:0 2px 4px rgba(0,0,0,0.04),0 6px 18px rgba(0,0,0,0.08);
  overflow:hidden;
  border:1px solid #e8e0d5;
">
  <div style="background:#36454F;padding:14px 34px 11px 14px;position:relative;overflow:hidden;">
    <div style="
      position:absolute;inset:0;
      background-image:radial-gradient(ellipse at 80% 0%,rgba(255,255,255,0.1) 0%,transparent 60%);
      pointer-events:none;
    "></div>
    <div style="
      display:inline-flex;align-items:center;gap:5px;
      background:rgba(255,255,255,0.14);
      border:1px solid rgba(255,255,255,0.3);
      border-radius:16px;padding:2px 8px 2px 6px;margin-bottom:7px;
    ">
      <span style="font-size:9px;">📍</span>
      <span style="
        font-family:system-ui,sans-serif;font-size:9px;font-weight:700;
        letter-spacing:0.7px;text-transform:uppercase;color:#F8FAFC;
      ">City Cluster</span>
    </div>
    <div style="font-size:20px;font-weight:700;color:#FFFFFF;line-height:1.1;margin-bottom:3px;">
      ${{city}}
    </div>
    <div style="
      font-family:system-ui,sans-serif;font-size:10px;font-weight:500;color:#E2E8F0;
    ">${{count}} ZIP codes in this area</div>
  </div>
  <div style="border-top:1px dashed {POPUP_DIVIDER};"></div>
  <div style="padding:12px 14px;">
    <div style="
      font-family:system-ui,sans-serif;font-size:8.5px;font-weight:700;
      text-transform:uppercase;letter-spacing:1px;color:#a89880;margin-bottom:10px;
    ">Average Health Indicators</div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:10px;">
      <div style="background:#fff;border:1px solid #ede8e1;border-radius:8px;padding:8px;text-align:center;">
        <div style="font-size:9px;color:#a89880;">Asthma</div>
        <div style="font-size:12px;font-weight:800;color:#2d1b0e;">${{avgAsthma}}</div>
      </div>
      <div style="background:#fff;border:1px solid #ede8e1;border-radius:8px;padding:8px;text-align:center;">
        <div style="font-size:9px;color:#a89880;">Diabetes</div>
        <div style="font-size:12px;font-weight:800;color:#2d1b0e;">${{avgDiabetes}}</div>
      </div>
      <div style="background:#fff;border:1px solid #ede8e1;border-radius:8px;padding:8px;text-align:center;">
        <div style="font-size:9px;color:#a89880;">Uninsured</div>
        <div style="font-size:12px;font-weight:800;color:#2d1b0e;">${{avgUninsured}}</div>
      </div>
      <div style="background:#fff;border:1px solid #ede8e1;border-radius:8px;padding:8px;text-align:center;">
        <div style="font-size:9px;color:#a89880;">Obesity</div>
        <div style="font-size:12px;font-weight:800;color:#2d1b0e;">${{avgObesity}}</div>
      </div>
      <div style="background:#fff;border:1px solid #ede8e1;border-radius:8px;padding:8px;text-align:center;">
        <div style="font-size:9px;color:#a89880;">Depression</div>
        <div style="font-size:12px;font-weight:800;color:#2d1b0e;">${{avgDepression}}</div>
      </div>
      <div style="background:#fff;border:1px solid #ede8e1;border-radius:8px;padding:8px;text-align:center;">
        <div style="font-size:9px;color:#a89880;">Food Insec.</div>
        <div style="font-size:12px;font-weight:800;color:#2d1b0e;">${{avgFoodInsecurity}}</div>
      </div>
    </div>
    <div style="
      background:{POPUP_HIGHLIGHT_BG};
      border:1px solid {POPUP_HIGHLIGHT_BORDER};border-left:3px solid {POPUP_ACCENT};
      border-radius:0 8px 8px 0;padding:9px 11px;
      font-family:Georgia,serif;font-size:11px;line-height:1.6;color:#4a2c10;
    ">
      <strong style="color:#c46b39;">ZIPs:</strong> ${{zipPreview}}${{zipExtra}}
    </div>
    <button type="button" class="pulsemap-zoom-cluster-btn"
      data-lat="${{lat}}" data-lng="${{lng}}"
      data-city="${{encodeURIComponent(city)}}"
      style="
        display:block; width:100%; margin-top:10px; padding:9px 12px;
        border:none; border-radius:8px; background:#36454F; color:#fff;
        font-family:system-ui,sans-serif; font-size:12px; font-weight:700;
        cursor:pointer;
      ">Zoom into ${{city}}</button>
  </div>
</div>`;
        }}

        function uniqueCityCount(zips) {{
            return new Set(zips.map((d) => d.c)).size;
        }}

        function buildStateIcon(state, zipCount) {{
            const size = zipCount > 500 ? 48 : zipCount > 100 ? 42 : 36;
            return L.divIcon({{
                className: "",
                html: `<div style="
                    min-width:${{size}}px;height:${{size}}px;padding:0 8px;
                    background:linear-gradient(145deg,#36454F 0%,#4a5f6f 100%);
                    border:2.5px solid #cbd5e1;
                    border-radius:999px;
                    display:flex;align-items:center;justify-content:center;
                    color:#f8fafc;
                    font-family:system-ui,sans-serif;
                    font-weight:800;
                    font-size:12px;
                    box-shadow:0 3px 10px rgba(54,69,79,0.35);
                    letter-spacing:0.4px;
                ">${{state}}</div>`,
                iconSize: [size, size],
                iconAnchor: [size / 2, size / 2],
            }});
        }}

        function buildStatePopup(state, zips) {{
            const zipCount = zips.length;
            const cityCount = uniqueCityCount(zips);
            const avgAsthma = avgRate(zips, "a");
            const avgDiabetes = avgRate(zips, "d");
            const avgUninsured = avgRate(zips, "u");
            const avgObesity = avgRate(zips, "o");
            const avgDepression = avgRate(zips, "dp");
            const avgFoodInsecurity = avgRate(zips, "fi");

            return `
<div style="
  font-family:'Georgia','Times New Roman',serif;
  width:248px;
  background:#faf8f5;
  border-radius:12px;
  box-shadow:0 2px 4px rgba(0,0,0,0.04),0 6px 18px rgba(0,0,0,0.08);
  overflow:hidden;
  border:1px solid #e8e0d5;
">
  <div style="background:#36454F;padding:14px 34px 11px 14px;position:relative;overflow:hidden;">
    <div style="
      display:inline-flex;align-items:center;gap:5px;
      background:rgba(255,255,255,0.14);
      border:1px solid rgba(255,255,255,0.3);
      border-radius:16px;padding:2px 8px 2px 6px;margin-bottom:7px;
    ">
      <span style="font-size:9px;">🗺️</span>
      <span style="
        font-family:system-ui,sans-serif;font-size:9px;font-weight:700;
        letter-spacing:0.7px;text-transform:uppercase;color:#F8FAFC;
      ">State Overview</span>
    </div>
    <div style="font-size:20px;font-weight:700;color:#FFFFFF;line-height:1.1;margin-bottom:3px;">
      ${{state}}
    </div>
    <div style="
      font-family:system-ui,sans-serif;font-size:10px;font-weight:500;color:#E2E8F0;
    ">${{zipCount.toLocaleString()}} ZIPs · ${{cityCount.toLocaleString()}} cities in view</div>
  </div>
  <div style="border-top:1px dashed {POPUP_DIVIDER};"></div>
  <div style="padding:12px 14px;">
    <div style="
      font-family:system-ui,sans-serif;font-size:8.5px;font-weight:700;
      text-transform:uppercase;letter-spacing:1px;color:#a89880;margin-bottom:10px;
    ">Average Health Indicators</div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:10px;">
      <div style="background:#fff;border:1px solid #ede8e1;border-radius:8px;padding:8px;text-align:center;">
        <div style="font-size:9px;color:#a89880;">Asthma</div>
        <div style="font-size:12px;font-weight:800;color:#2d1b0e;">${{avgAsthma}}</div>
      </div>
      <div style="background:#fff;border:1px solid #ede8e1;border-radius:8px;padding:8px;text-align:center;">
        <div style="font-size:9px;color:#a89880;">Diabetes</div>
        <div style="font-size:12px;font-weight:800;color:#2d1b0e;">${{avgDiabetes}}</div>
      </div>
      <div style="background:#fff;border:1px solid #ede8e1;border-radius:8px;padding:8px;text-align:center;">
        <div style="font-size:9px;color:#a89880;">Uninsured</div>
        <div style="font-size:12px;font-weight:800;color:#2d1b0e;">${{avgUninsured}}</div>
      </div>
      <div style="background:#fff;border:1px solid #ede8e1;border-radius:8px;padding:8px;text-align:center;">
        <div style="font-size:9px;color:#a89880;">Obesity</div>
        <div style="font-size:12px;font-weight:800;color:#2d1b0e;">${{avgObesity}}</div>
      </div>
      <div style="background:#fff;border:1px solid #ede8e1;border-radius:8px;padding:8px;text-align:center;">
        <div style="font-size:9px;color:#a89880;">Depression</div>
        <div style="font-size:12px;font-weight:800;color:#2d1b0e;">${{avgDepression}}</div>
      </div>
      <div style="background:#fff;border:1px solid #ede8e1;border-radius:8px;padding:8px;text-align:center;">
        <div style="font-size:9px;color:#a89880;">Food Insec.</div>
        <div style="font-size:12px;font-weight:800;color:#2d1b0e;">${{avgFoodInsecurity}}</div>
      </div>
    </div>
    <div style="
      background:{POPUP_HIGHLIGHT_BG};
      border:1px solid {POPUP_HIGHLIGHT_BORDER};border-left:3px solid {POPUP_ACCENT};
      border-radius:0 8px 8px 0;padding:9px 11px;
      font-family:Georgia,serif;font-size:11px;line-height:1.6;color:#4a2c10;
    ">
      <span style="color:#a89880;font-family:system-ui,sans-serif;font-size:10px;">
        State-level averages for this view. Zoom in for city clusters and ZIP detail.
      </span>
    </div>
  </div>
</div>`;
        }}

        const POPUP_OPTIONS = {{
            maxWidth: 380,
            autoPan: false,
            className: "pulsemap-popup",
        }};
        const POPUP_EDGE_PAD = 14;
        const POPUP_FALLBACK_HEIGHT = 420;
        const POPUP_FALLBACK_WIDTH = 320;
        const SIDE_PREFERENCE = {{ right: 0, above: 1, below: 2, left: 3 }};
        const UI_OBSTACLE_SELECTORS = [
            "#pulsemap-page-nav",
            "#pulsemap-tools",
            "#pulsemap-controls",
            "#pulsemap-legend",
        ];

        function markerScreenPad(source) {{
            if (source && typeof source.getRadius === "function") {{
                return source.getRadius() + 8;
            }}
            if (source && source.options && source.options.icon) {{
                return 20;
            }}
            return 12;
        }}

        function measurePopup(el) {{
            const rect = el.getBoundingClientRect();
            const h = Math.max(rect.height, el.offsetHeight, el.scrollHeight);
            const w = Math.max(rect.width, el.offsetWidth, el.scrollWidth);
            return {{
                h: h > 0 ? h : POPUP_FALLBACK_HEIGHT,
                w: w > 0 ? w : POPUP_FALLBACK_WIDTH,
            }};
        }}

        function rectsOverlap(a, b) {{
            return a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;
        }}

        function popupRectInContainer(el, map) {{
            const mapRect = map.getContainer().getBoundingClientRect();
            const rect = el.getBoundingClientRect();
            return {{
                left: rect.left - mapRect.left,
                top: rect.top - mapRect.top,
                right: rect.right - mapRect.left,
                bottom: rect.bottom - mapRect.top,
            }};
        }}

        function getUiObstacleRects(map) {{
            const mapRect = map.getContainer().getBoundingClientRect();
            const pad = POPUP_EDGE_PAD;
            const obstacles = [];

            for (const selector of UI_OBSTACLE_SELECTORS) {{
                const node = document.querySelector(selector);
                if (!node) continue;
                const style = window.getComputedStyle(node);
                if (style.display === "none" || style.visibility === "hidden") continue;

                const rect = node.getBoundingClientRect();
                if (rect.width <= 0 || rect.height <= 0) continue;
                if (
                    rect.bottom < mapRect.top ||
                    rect.top > mapRect.bottom ||
                    rect.right < mapRect.left ||
                    rect.left > mapRect.right
                ) {{
                    continue;
                }}

                obstacles.push({{
                    left: rect.left - mapRect.left - pad,
                    top: rect.top - mapRect.top - pad,
                    right: rect.right - mapRect.left + pad,
                    bottom: rect.bottom - mapRect.top + pad,
                }});
            }}

            const suggestions = document.getElementById("search-suggestions");
            if (suggestions && suggestions.style.display !== "none") {{
                const rect = suggestions.getBoundingClientRect();
                if (rect.width > 0 && rect.height > 0) {{
                    obstacles.push({{
                        left: rect.left - mapRect.left - pad,
                        top: rect.top - mapRect.top - pad,
                        right: rect.right - mapRect.left + pad,
                        bottom: rect.bottom - mapRect.top + pad,
                    }});
                }}
            }}

            return obstacles;
        }}

        function popupRectForSide(markerPoint, side, popupH, popupW, markerPad) {{
            const halfW = popupW / 2;
            const halfH = popupH / 2;

            if (side === "above") {{
                const bottom = markerPoint.y - markerPad;
                return {{
                    left: markerPoint.x - halfW,
                    top: bottom - popupH,
                    right: markerPoint.x + halfW,
                    bottom,
                }};
            }}
            if (side === "below") {{
                const top = markerPoint.y + markerPad;
                return {{
                    left: markerPoint.x - halfW,
                    top,
                    right: markerPoint.x + halfW,
                    bottom: top + popupH,
                }};
            }}
            if (side === "left") {{
                const right = markerPoint.x - markerPad;
                return {{
                    left: right - popupW,
                    top: markerPoint.y - halfH,
                    right,
                    bottom: markerPoint.y + halfH,
                }};
            }}

            const left = markerPoint.x + markerPad;
            return {{
                left,
                top: markerPoint.y - halfH,
                right: left + popupW,
                bottom: markerPoint.y + halfH,
            }};
        }}

        function scorePopupPlacement(rect, mapSize, obstacles, pad, markerPoint) {{
            let overflow = 0;
            overflow += Math.max(0, pad - rect.left);
            overflow += Math.max(0, pad - rect.top);
            overflow += Math.max(0, rect.right - (mapSize.x - pad));
            overflow += Math.max(0, rect.bottom - (mapSize.y - pad));

            for (const obstacle of obstacles) {{
                if (!rectsOverlap(rect, obstacle)) continue;
                const overlapW = Math.min(rect.right, obstacle.right) - Math.max(rect.left, obstacle.left);
                const overlapH = Math.min(rect.bottom, obstacle.bottom) - Math.max(rect.top, obstacle.top);
                overflow += overlapW + overlapH + overlapW * overlapH * 0.02;
            }}

            if (popupCoversMarker(rect, markerPoint)) {{
                overflow += 10000;
            }}

            return {{ fits: overflow === 0, overflow }};
        }}

        function popupCoversMarker(rect, markerPoint) {{
            return (
                markerPoint.x >= rect.left &&
                markerPoint.x <= rect.right &&
                markerPoint.y >= rect.top &&
                markerPoint.y <= rect.bottom
            );
        }}

        function requiredMarkerShift(side, markerPoint, popupH, popupW, markerPad, mapSize, obstacles, pad) {{
            let shiftX = 0;
            let shiftY = 0;

            for (let attempt = 0; attempt < 8; attempt += 1) {{
                const point = {{
                    x: markerPoint.x + shiftX,
                    y: markerPoint.y + shiftY,
                }};
                const rect = popupRectForSide(point, side, popupH, popupW, markerPad);
                let dx = 0;
                let dy = 0;

                if (rect.left < pad) dx = pad - rect.left;
                else if (rect.right > mapSize.x - pad) dx = (mapSize.x - pad) - rect.right;

                if (rect.top < pad) dy = pad - rect.top;
                else if (rect.bottom > mapSize.y - pad) dy = (mapSize.y - pad) - rect.bottom;

                for (const obstacle of obstacles) {{
                    if (!rectsOverlap(rect, obstacle)) continue;
                    dx = Math.max(dx, obstacle.right - rect.left + pad);
                    dy = Math.max(dy, obstacle.bottom - rect.top + pad);
                }}

                if (popupCoversMarker(rect, point)) {{
                    if (side === "above") dy -= markerPad + 12;
                    else if (side === "below") dy += markerPad + 12;
                    else if (side === "left") dx -= markerPad + 12;
                    else dx += markerPad + 12;
                }}

                if (!dx && !dy) break;
                shiftX += dx;
                shiftY += dy;
            }}

            return {{ shiftX, shiftY }};
        }}

        function sidePanBias(side) {{
            if (side === "right") return -60;
            if (side === "left") return 100;
            return 0;
        }}

        function pickBestSide(markerPoint, map, popupH, popupW, markerPad) {{
            const mapSize = map.getSize();
            const obstacles = getUiObstacleRects(map);
            const pad = POPUP_EDGE_PAD;
            const candidates = ["above", "below", "right", "left"].map((side) => {{
                const rect = popupRectForSide(markerPoint, side, popupH, popupW, markerPad);
                const score = scorePopupPlacement(
                    rect,
                    mapSize,
                    obstacles,
                    pad,
                    markerPoint
                );
                const shift = requiredMarkerShift(
                    side,
                    markerPoint,
                    popupH,
                    popupW,
                    markerPad,
                    mapSize,
                    obstacles,
                    pad
                );
                const panCost =
                    Math.abs(shift.shiftX) +
                    Math.abs(shift.shiftY) +
                    sidePanBias(side);
                return {{ side, ...score, panCost, shift }};
            }});

            candidates.sort((a, b) => {{
                if (a.fits !== b.fits) return Number(b.fits) - Number(a.fits);
                if (a.overflow !== b.overflow) return a.overflow - b.overflow;
                if (a.panCost !== b.panCost) return a.panCost - b.panCost;
                return SIDE_PREFERENCE[a.side] - SIDE_PREFERENCE[b.side];
            }});
            return candidates[0];
        }}

        function resetPopupSideClasses(el) {{
            el.classList.remove("popup-below", "popup-left", "popup-right");
        }}

        function popupOffsetForSide(el, side, popupH, popupW, markerPad) {{
            let offsetX = 0;
            let offsetY = 0;

            if (side === "below") {{
                el.classList.add("popup-below");
                offsetY = markerPad + popupH;
            }} else if (side === "left") {{
                el.classList.add("popup-left");
                offsetX = -(markerPad + Math.round(popupW / 2));
                offsetY = Math.round(popupH / 2);
            }} else if (side === "right") {{
                el.classList.add("popup-right");
                offsetX = markerPad + Math.round(popupW / 2);
                offsetY = Math.round(popupH / 2);
            }} else {{
                offsetY = -markerPad;
            }}

            return L.point(Math.round(offsetX), Math.round(offsetY));
        }}

        function placementNeedsPan(candidate, markerPoint, popupH, popupW, markerPad, map) {{
            const mapSize = map.getSize();
            const obstacles = getUiObstacleRects(map);
            const pad = POPUP_EDGE_PAD;
            const rect = popupRectForSide(
                markerPoint,
                candidate.side,
                popupH,
                popupW,
                markerPad
            );
            const score = scorePopupPlacement(
                rect,
                mapSize,
                obstacles,
                pad,
                markerPoint
            );
            return (
                !score.fits ||
                Math.abs(candidate.shift.shiftX) > 2 ||
                Math.abs(candidate.shift.shiftY) > 2
            );
        }}

        let popupPanning = false;

        function setPopupPositioning(el, active) {{
            if (!el) return;
            el.classList.toggle("pulsemap-popup--positioning", active);
        }}

        function finalizePopupPlacement(e) {{
            const popup = e.popup;
            const map = e.target;
            const el = popup.getElement();
            const latlng = popup.getLatLng();
            if (!el || !latlng) return;

            const markerPoint = map.latLngToContainerPoint(latlng);
            const markerPad = markerScreenPad(popup._source);
            const {{ h: popupH, w: popupW }} = measurePopup(el);
            const candidate = pickBestSide(markerPoint, map, popupH, popupW, markerPad);

            resetPopupSideClasses(el);
            popup.options.offset = popupOffsetForSide(
                el,
                candidate.side,
                popupH,
                popupW,
                markerPad
            );
            popup.update();
            requestAnimationFrame(() => setPopupPositioning(el, false));
        }}

        function placePopupBesidePoint(e, options = {{}}) {{
            const popup = e.popup;
            const map = e.target;
            const el = popup.getElement();
            const latlng = popup.getLatLng();
            if (!el || !latlng) return;

            if (!options.keepHidden) {{
                setPopupPositioning(el, true);
            }}

            const markerPoint = map.latLngToContainerPoint(latlng);
            const markerPad = markerScreenPad(popup._source);
            const {{ h: popupH, w: popupW }} = measurePopup(el);

            if (popupH < 50 && !options.layoutRetry) {{
                requestAnimationFrame(() => {{
                    placePopupBesidePoint(e, {{ ...options, layoutRetry: true, keepHidden: true }});
                }});
                return;
            }}

            const candidate = pickBestSide(markerPoint, map, popupH, popupW, markerPad);

            if (
                !options.skipPan &&
                !popupPanning &&
                placementNeedsPan(candidate, markerPoint, popupH, popupW, markerPad, map)
            ) {{
                const maxPan = 320;
                const panX = Math.max(-maxPan, Math.min(maxPan, -candidate.shift.shiftX));
                const panY = Math.max(-maxPan, Math.min(maxPan, -candidate.shift.shiftY));

                if (Math.abs(panX) > 2 || Math.abs(panY) > 2) {{
                    popupPanning = true;
                    map.once("moveend", () => {{
                        popupPanning = false;
                        placePopupBesidePoint(e, {{ skipPan: true, keepHidden: true }});
                    }});
                    map.panBy([panX, panY], {{ animate: true, duration: 0.35 }});
                    return;
                }}
            }}

            finalizePopupPlacement(e);
        }}

        function schedulePopupPlacement(e) {{
            requestAnimationFrame(() => placePopupBesidePoint(e));
        }}

        function addIndividualMarker(d) {{
            const popup = L.popup(POPUP_OPTIONS).setContent(buildPopup(d));
            const tooltip = `${{d.zip}} · ${{locationLabel(d)}}`;

            if (d.z === CENTER_ZIP) {{
                L.marker([d.lat, d.lng], {{
                    icon: L.AwesomeMarkers.icon({{
                        icon: "star",
                        markerColor: "red",
                        prefix: "fa",
                    }}),
                }})
                    .bindPopup(popup)
                    .bindTooltip(`${{tooltip}} — home ZIP`, {{ sticky: true }})
                    .addTo(markerLayer);
            }} else {{
                const color = markerColorForMetric(d);
                L.circleMarker([d.lat, d.lng], {{
                    radius: markerRadiusForZoom(),
                    color,
                    fillColor: color,
                    fillOpacity: 0.85,
                    weight: 1,
                }})
                    .bindPopup(popup)
                    .bindTooltip(tooltip, {{ sticky: true }})
                    .addTo(markerLayer);
            }}
        }}

        const markerLayer = L.layerGroup().addTo(map);
        let updateTimer = null;
        let popupOpen = false;

        function sampleItemKey(item) {{
            if (item.z != null) return `z:${{item.z}}`;
            if (item.city) return `c:${{item.city}}:${{item.lat}}:${{item.lng}}`;
            return `p:${{item.lat}}:${{item.lng}}`;
        }}

        function pickBestForBucket(bucket, map) {{
            let best = bucket.items[0];
            let bestDist = Infinity;
            for (const item of bucket.items) {{
                const point = map.latLngToContainerPoint([item.lat, item.lng]);
                const dx = point.x - bucket.cx;
                const dy = point.y - bucket.cy;
                const dist = dx * dx + dy * dy;
                if (dist < bestDist) {{
                    bestDist = dist;
                    best = item;
                }}
            }}
            return best;
        }}

        function sampleByViewportGrid(items, map, limit) {{
            if (items.length <= limit) return items;

            const size = map.getSize();
            const aspect = size.x / Math.max(size.y, 1);
            const cols = Math.max(8, Math.ceil(Math.sqrt(limit * aspect * 0.65)));
            const rows = Math.max(6, Math.ceil(limit / cols));
            const cellW = size.x / cols;
            const cellH = size.y / rows;
            const buckets = new Map();

            for (const d of items) {{
                const point = map.latLngToContainerPoint([d.lat, d.lng]);
                if (point.x < 0 || point.x > size.x || point.y < 0 || point.y > size.y) {{
                    continue;
                }}

                const col = Math.min(
                    cols - 1,
                    Math.max(0, Math.floor(point.x / cellW))
                );
                const row = Math.min(
                    rows - 1,
                    Math.max(0, Math.floor(point.y / cellH))
                );
                const key = `${{row}}:${{col}}`;
                if (!buckets.has(key)) {{
                    buckets.set(key, {{
                        row,
                        col,
                        cx: (col + 0.5) * cellW,
                        cy: (row + 0.5) * cellH,
                        items: [],
                    }});
                }}
                buckets.get(key).items.push(d);
            }}

            const picked = new Set();
            const result = [];

            function tryAdd(item) {{
                if (result.length >= limit) return false;
                const key = sampleItemKey(item);
                if (picked.has(key)) return false;
                picked.add(key);
                result.push(item);
                return true;
            }}

            for (let col = 0; col < cols; col += 1) {{
                const columnBuckets = Array.from(buckets.values())
                    .filter((bucket) => bucket.col === col)
                    .sort((a, b) => a.row - b.row);
                if (!columnBuckets.length) continue;
                const middle = columnBuckets[Math.floor(columnBuckets.length / 2)];
                tryAdd(pickBestForBucket(middle, map));
            }}

            const orderedCells = Array.from(buckets.values()).sort(
                (a, b) => a.col - b.col || a.row - b.row
            );
            for (const bucket of orderedCells) {{
                if (result.length >= limit) break;
                tryAdd(pickBestForBucket(bucket, map));
            }}

            if (result.length < limit) {{
                const byColumn = Array.from({{ length: cols }}, () => []);
                for (const bucket of buckets.values()) {{
                    for (const item of bucket.items) {{
                        if (!picked.has(sampleItemKey(item))) {{
                            byColumn[bucket.col].push(item);
                        }}
                    }}
                }}

                let col = 0;
                let safety = items.length * 2;
                while (result.length < limit && safety > 0) {{
                    safety -= 1;
                    if (byColumn[col].length) {{
                        tryAdd(byColumn[col].shift());
                    }}
                    col = (col + 1) % cols;
                }}
            }}

            if (result.length < limit) {{
                for (const bucket of orderedCells) {{
                    for (const item of bucket.items) {{
                        if (result.length >= limit) break;
                        tryAdd(item);
                    }}
                    if (result.length >= limit) break;
                }}
            }}

            return result;
        }}

        function cityEntriesFromZips(zips) {{
            const groups = groupByCity(zips);
            return Object.entries(groups).map(([city, cityZips]) => {{
                const [lat, lng] = centroid(cityZips);
                return {{ city, zips: cityZips, lat, lng }};
            }});
        }}

        function addCityCluster(entry) {{
            if (entry.zips.length === 1) {{
                addIndividualMarker(entry.zips[0]);
                return;
            }}

            const clusterPopupOptions = {{ ...POPUP_OPTIONS, maxWidth: 270 }};
            const clusterPopup = L.popup(clusterPopupOptions).setContent(
                buildClusterPopup(entry.city, entry.zips, entry.lat, entry.lng)
            );

            L.marker([entry.lat, entry.lng], {{ icon: buildClusterIcon(entry.zips.length) }})
                .bindPopup(clusterPopup)
                .bindTooltip(`${{entry.city}} — ${{entry.zips.length}} ZIPs`, {{ sticky: true }})
                .addTo(markerLayer);

            const homeZip = entry.zips.find((d) => d.z === CENTER_ZIP);
            if (homeZip && passesFilter(homeZip)) addIndividualMarker(homeZip);
        }}

        function updateVisibleMarkers() {{
            if (updateTimer) clearTimeout(updateTimer);
            updateTimer = setTimeout(() => {{
                const zoom = map.getZoom();
                updateViewModeBadge(zoom);
                if (popupOpen) return;

                const bounds = map.getBounds();
                const inView = ZIP_DATA.filter((d) => bounds.contains([d.lat, d.lng]));
                const visible = inView.filter(passesFilter);
                const filterNote = filterState.aboveNationalOnly
                    ? ` · ${{visible.length}} above US median`
                    : "";

                markerLayer.clearLayers();

                if (zoom < STATE_ZOOM) {{
                    const stateGroups = groupByState(visible);
                    const states = Object.keys(stateGroups).filter((state) => state !== "—");
                    for (const state of states) {{
                        const zips = stateGroups[state];
                        const [lat, lng] = centroid(zips);
                        const statePopupOptions = {{ ...POPUP_OPTIONS, maxWidth: 270 }};
                        const statePopup = L.popup(statePopupOptions).setContent(
                            buildStatePopup(state, zips)
                        );

                        L.marker([lat, lng], {{ icon: buildStateIcon(state, zips.length) }})
                            .bindPopup(statePopup)
                            .bindTooltip(`${{state}} — ${{zips.length}} ZIPs`, {{ sticky: true }})
                            .addTo(markerLayer);
                    }}

                    const homeZip = visible.find((d) => d.z === CENTER_ZIP);
                    if (homeZip) addIndividualMarker(homeZip);

                    const countEl = document.getElementById("visible-zip-count");
                    if (countEl) {{
                        countEl.textContent =
                            `${{states.length}} states · ${{visible.length}} ZIPs${{filterNote}} — state overview`;
                    }}
                }} else if (zoom < CLUSTER_ZOOM) {{
                    const entries = cityEntriesFromZips(visible);
                    const showing = sampleByViewportGrid(entries, map, MAX_CITY_CLUSTERS);
                    for (const entry of showing) addCityCluster(entry);

                    const countEl = document.getElementById("visible-zip-count");
                    if (countEl) {{
                        const suffix = entries.length > MAX_CITY_CLUSTERS
                            ? ` (showing ${{showing.length}} of ${{entries.length}} cities)`
                            : "";
                        countEl.textContent =
                            `${{showing.length}} cities · ${{visible.length}} ZIPs${{suffix}}${{filterNote}} — click a cluster to zoom in`;
                    }}
                }} else {{
                    const showing = sampleByViewportGrid(visible, map, MAX_MARKERS);
                    for (const d of showing) addIndividualMarker(d);

                    const countEl = document.getElementById("visible-zip-count");
                    if (countEl) {{
                        const suffix = visible.length > MAX_MARKERS
                            ? ` (${{visible.length}} in view, showing ${{showing.length}})`
                            : "";
                        countEl.textContent = `${{showing.length}} ZIPs in this view${{suffix}}${{filterNote}}`;
                    }}
                }}
            }}, 120);
        }}

            map.on("popupopen", (e) => {{
                popupOpen = true;
                setPopupPositioning(e.popup.getElement(), true);
                schedulePopupPlacement(e);
                wirePopupActions(e);
            }});
            map.on("popupclose", () => {{
                popupOpen = false;
                popupPanning = false;
                updateVisibleMarkers();
            }});

            wireControlEvents();
            map.whenReady(updateVisibleMarkers);
            map.on("moveend", updateVisibleMarkers);
            map.on("zoomend", updateVisibleMarkers);
        }}

        boot();
    }})();
    """
    health_map.get_root().script.add_child(folium.Element(script))


def write_action_guide_html(output_path: Path) -> None:
    metric_cards: list[str] = []
    for spec in METRIC_SPECS:
        label = str(spec["label"])
        sol = METRIC_SOLUTIONS.get(label)
        if not sol:
            continue
        emoji = html.escape(str(spec["emoji"]))
        safe_label = html.escape(label)
        summary = html.escape(str(sol.get("summary", "")))
        actions = sol.get("actions", [])
        if not isinstance(actions, list):
            actions = []
        action_items = "".join(
            f"<li>{html.escape(str(action))}</li>" for action in actions
        )
        source = html.escape(str(sol.get("source", "")))
        resource_link = sol.get("resource_link")
        resource_html = ""
        if isinstance(resource_link, dict) and resource_link.get("url"):
            link_url = html.escape(str(resource_link["url"]))
            link_label = html.escape(
                str(resource_link.get("label", resource_link["url"]))
            )
            resource_html = (
                f'<p class="resource-line">'
                f'<a href="{link_url}" target="_blank" rel="noopener">{link_label}</a>'
                f"</p>"
            )
        summary_html = (
            f'<p class="card-summary">{summary}</p>' if summary else ""
        )
        connects = sol.get("connects_with", [])
        connects_html = ""
        if isinstance(connects, list) and connects:
            joined = ", ".join(html.escape(str(item)) for item in connects)
            connects_html = f'<p class="connects-line"><strong>Connects with:</strong> {joined}</p>'
        metric_cards.append(
            f"""
    <article class="metric-card" id="{safe_label.lower().replace(' ', '-')}">
      <h2>{emoji} {safe_label}</h2>
      {summary_html}
      {connects_html}
      <ol class="action-list">{action_items}</ol>
      {resource_html}
      <p class="source-line"><strong>Source:</strong> {source}</p>
    </article>"""
        )

    connection_bullets = "".join(
        f"<li>{html.escape(str(entry.get('text', '')))}</li>"
        for entry in SDOH_CONNECTION_INSIGHTS
    )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(PRODUCT_NAME)} Action Guide — Evidence-Based SDOH Steps</title>
  <style>
    :root {{
      --ink: #36454F;
      --paper: #faf8f5;
      --muted: #6b7280;
      --accent: #2f6b3a;
      --line: #e8e0d5;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, system-ui, sans-serif;
      color: #333;
      background: linear-gradient(180deg, #f5f2ed 0%, #faf8f5 240px);
      line-height: 1.6;
    }}
    .wrap {{ max-width: 820px; margin: 0 auto; padding: 28px 20px 48px; }}
    .hero {{
      background: var(--ink);
      color: #fff;
      border-radius: 14px;
      padding: 22px 24px;
      margin-bottom: 22px;
      box-shadow: 0 10px 28px rgba(0,0,0,0.12);
    }}
    .hero h1 {{ margin: 0 0 8px; font-size: 1.55rem; }}
    .hero p {{ margin: 0; opacity: 0.92; font-size: 0.98rem; }}
    .hero .track {{
      display: inline-block;
      margin-top: 12px;
      padding: 4px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.14);
      font-size: 0.78rem;
      letter-spacing: 0.2px;
    }}
    .flow {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 22px;
    }}
    .flow-step {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px;
      font-size: 0.88rem;
    }}
    .flow-step strong {{
      display: block;
      color: var(--ink);
      margin-bottom: 4px;
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.4px;
    }}
    .metric-card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 16px 18px;
      margin-bottom: 14px;
    }}
    .metric-card h2 {{ margin: 0 0 8px; color: var(--ink); font-size: 1.15rem; }}
    .card-summary {{ margin: 0 0 10px; color: #555; font-size: 0.95rem; }}
    .connects-line {{ margin: 0 0 10px; color: #6b7280; font-size: 0.88rem; }}
    .action-list {{ margin: 0; padding-left: 20px; color: #4a2c10; }}
    .action-list li {{ margin-bottom: 6px; }}
    .resource-line {{
      margin: 10px 0 0;
      font-size: 0.9rem;
    }}
    .resource-line a {{
      color: var(--accent);
      font-weight: 600;
      text-decoration: none;
    }}
    .resource-line a:hover {{ text-decoration: underline; }}
    .source-line {{
      margin: 12px 0 0;
      padding-top: 10px;
      border-top: 1px dashed var(--line);
      font-size: 0.82rem;
      color: #a89880;
    }}
    .connected {{
      background: #fff;
      border: 1px solid var(--line);
      border-left: 4px solid var(--accent);
      border-radius: 10px;
      padding: 14px 16px;
      margin-bottom: 22px;
    }}
    .connected h2 {{
      margin: 0 0 8px;
      color: var(--ink);
      font-size: 1.05rem;
    }}
    .connected p {{ margin: 0 0 10px; color: #555; }}
    .connected ul {{ margin: 0; padding-left: 20px; color: #3d4f3d; }}
    .connected li {{ margin-bottom: 6px; }}
    .footer {{
      margin-top: 24px;
      font-size: 0.88rem;
      color: var(--muted);
    }}
    .footer a {{ color: var(--ink); font-weight: 600; }}
    .page-nav {{
      position: sticky;
      top: 0;
      z-index: 50;
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 12px 20px;
      background: rgba(250, 248, 245, 0.94);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--line);
    }}
    .page-nav a,
    .page-nav button {{
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 8px 14px;
      border-radius: 999px;
      font-family: Inter, system-ui, sans-serif;
      font-size: 12px;
      font-weight: 600;
      text-decoration: none;
      cursor: pointer;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
    }}
    .page-nav button {{
      background: #5c6b73;
      color: #fff;
      border-color: #5c6b73;
    }}
    .page-nav a.map-link {{
      margin-left: auto;
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }}
    @media (max-width: 720px) {{
      .flow {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <nav class="page-nav" aria-label="Page navigation">
    <a href="{HOME_PAGE}">← Home</a>
    <button type="button" id="guide-back-btn">← Go back</button>
    <a href="health_map.html" class="map-link">Open map</a>
  </nav>
  <div class="wrap">
    <header class="hero">
      <h1>{html.escape(PRODUCT_NAME)} Action Guide</h1>
      <p>
        Practical, plain-language steps for the six SDOH health indicators {html.escape(PRODUCT_NAME)} tracks.
        Every section is grounded in a source you provided—from clinical papers to community policy guides.
      </p>
      <span class="track">The Prioneer · AI for Social Good · underserved communities</span>
    </header>

    <section class="flow" aria-label="How {html.escape(PRODUCT_NAME)} connects data to action">
      <div class="flow-step">
        <strong>1 · Detect</strong>
        K-Means clusters ~30,000 ZIP codes on six burden rates to surface equity tiers.
      </div>
      <div class="flow-step">
        <strong>2 · Prioritize</strong>
        Each community popup highlights the top two local concerns above the U.S. median.
      </div>
      <div class="flow-step">
        <strong>3 · Act</strong>
        Evidence-based steps below help residents, advocates, and clinics take next steps.
      </div>
    </section>

    <section class="connected" aria-label="How the six metrics connect">
      <h2>How the six metrics connect</h2>
      <p>{html.escape(SDOH_CONNECTION_NARRATIVE)}</p>
      <ul>{connection_bullets}</ul>
    </section>

    {''.join(metric_cards)}

    <p class="footer">
      <a href="{HOME_PAGE}">← Home</a>
      ·
      <a href="health_map.html">Open interactive map</a>
      · ZIP popups show the top two local concerns plus connected SDOH insights when burdens overlap.
    </p>
  </div>
  <script>
    document.getElementById("guide-back-btn")?.addEventListener("click", () => {{
      if (window.history.length > 1) {{
        history.back();
        return;
      }}
      window.location.href = {json.dumps(HOME_PAGE)};
    }});
  </script>
</body>
</html>
"""
    output_path.write_text(page, encoding="utf-8")


def build_nationwide_map(
    center_row: pd.Series,
    df: pd.DataFrame,
) -> folium.Map:
    center_lat = float(center_row["lat"])
    center_lng = float(center_row["lng"])
    center_city = center_row["city"]
    center_zip = int(normalize_zip_series(pd.DataFrame([center_row])).iloc[0])
    zip_records = prepare_zip_records(df, load_state_lookup())

    health_map = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=11,
        min_zoom=4,
        tiles="CartoDB positron",
    )

    add_viewport_marker_script(health_map, zip_records, center_zip)
    add_map_legend(health_map, center_city, center_zip, len(zip_records))
    return health_map


def main() -> int:
    args = parse_args()

    try:
        zip_code = validate_zip(args.zip)
        df = assign_equity_vulnerability_tiers(load_data(DATA_PATH))
        center_row = lookup_zip(df, zip_code)
        health_map = build_nationwide_map(center_row, df)
        output_path = Path(args.output)
        health_map.save(str(output_path))
        guide_path = Path(args.action_guide)
        write_action_guide_html(guide_path)
        viz_path = Path(args.kmeans_viz_data)
        viz_count = export_kmeans_viz_data(df, viz_path)
        print(
            f"Map saved to {output_path.resolve()} "
            f"({len(df):,} ZIPs nationwide, centered on "
            f"{center_row['city']} / {zip_code:05d})"
        )
        print(f"Action guide saved to {guide_path.resolve()}")
        print(
            f"K-Means viz data saved to {viz_path.resolve()} "
            f"({viz_count:,} sample points — open kmeans_viz.html)"
        )
        return 0
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
