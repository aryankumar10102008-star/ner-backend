"""
NER Smart Logistics — Route Risk Prediction API

Run:
    python -m uvicorn risk_api:app --reload --port 8000
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import joblib
import pandas as pd
import requests

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# ENVIRONMENT / SUPABASE
# ============================================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
from supabase import create_client

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="NER Smart Logistics — Risk Prediction API"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "risk_model.pkl"

_bundle = None


# ============================================================
# DATA MODELS
# ============================================================

class RouteFeatures(BaseModel):

    rainfall_last_24h_mm: float = Field(..., ge=0)

    rainfall_forecast_6h_mm: float = Field(..., ge=0)

    wind_speed_kmh: float = Field(..., ge=0)

    elevation_m: float = Field(..., ge=0)

    slope_deg: float = Field(..., ge=0, le=90)

    landslide_prone_zone: int = Field(..., ge=0, le=1)

    past_closures_90d: int = Field(..., ge=0)

    road_type_highway: int = Field(..., ge=0, le=1)

    is_monsoon_month: int = Field(..., ge=0, le=1)


class RiskResponse(BaseModel):

    risk_score: float

    risk_category: str

    model_probability: float

    safety_override_applied: bool

    top_factors: list[str]


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "NER Smart Logistics API"
    }


# ============================================================
# LOAD ML MODEL
# ============================================================

def load_model():

    global _bundle

    if _bundle is None:

        try:

            _bundle = joblib.load(MODEL_PATH)

        except FileNotFoundError:

            raise HTTPException(
                status_code=503,
                detail="Model not trained yet — run train_risk_model.py first."
            )

    return _bundle


# ============================================================
# SAFETY OVERRIDE
# ============================================================

def apply_safety_override(
    features: RouteFeatures,
    ml_probability: float
):

    override_applied = False

    probability = ml_probability


    if (
        features.rainfall_last_24h_mm > 150
        or
        features.rainfall_forecast_6h_mm > 80
    ):

        probability = max(
            probability,
            0.85
        )

        override_applied = True


    if (
        features.landslide_prone_zone == 1
        and
        features.slope_deg > 35
    ):

        probability = max(
            probability,
            0.75
        )

        override_applied = True


    if features.past_closures_90d >= 5:

        probability = max(
            probability,
            0.70
        )

        override_applied = True


    return probability, override_applied


# ============================================================
# EXPLAIN RISK
# ============================================================

def explain_top_factors(
    features: RouteFeatures,
    feature_importances: dict
):

    baseline = {

        "rainfall_last_24h_mm": 10,

        "rainfall_forecast_6h_mm": 5,

        "wind_speed_kmh": 15,

        "elevation_m": 500,

        "slope_deg": 5,

        "landslide_prone_zone": 0,

        "past_closures_90d": 0,

        "road_type_highway": 1,

        "is_monsoon_month": 0,
    }


    values = features.dict()

    contribution = {}


    for feat, importance in feature_importances.items():

        deviation = abs(
            values[feat] - baseline[feat]
        )

        contribution[feat] = (
            importance * deviation
        )


    ranked = sorted(
        contribution.items(),
        key=lambda x: x[1],
        reverse=True
    )


    labels = {

        "rainfall_last_24h_mm":
            "Heavy rainfall in the last 24h",

        "rainfall_forecast_6h_mm":
            "Heavy rainfall forecasted soon",

        "wind_speed_kmh":
            "High wind speed",

        "elevation_m":
            "High elevation segment",

        "slope_deg":
            "Steep terrain",

        "landslide_prone_zone":
            "Known landslide-prone zone",

        "past_closures_90d":
            "History of recent closures",

        "road_type_highway":
            "Non-highway / rural road",

        "is_monsoon_month":
            "Monsoon season",
    }


    return [
        labels[f]
        for f, _ in ranked[:3]
        if contribution[f] > 0
    ]


# ============================================================
# PREDICT RISK
# ============================================================

@app.post(
    "/predict-risk",
    response_model=RiskResponse
)
def predict_risk(features: RouteFeatures):

    bundle = load_model()

    model = bundle["model"]

    feature_order = bundle["features"]


    X = pd.DataFrame(
        [features.dict()]
    )[feature_order]


    ml_probability = float(
        model.predict_proba(X)[0, 1]
    )


    final_probability, override_applied = (
        apply_safety_override(
            features,
            ml_probability
        )
    )


    risk_score = round(
        final_probability * 100,
        1
    )


    if risk_score < 30:

        category = "green"

    elif risk_score < 65:

        category = "yellow"

    else:

        category = "red"


    importances = dict(
        zip(
            feature_order,
            model.feature_importances_
        )
    )


    top_factors = explain_top_factors(
        features,
        importances
    )


    return RiskResponse(

        risk_score=risk_score,

        risk_category=category,

        model_probability=round(
            ml_probability,
            3
        ),

        safety_override_applied=
            override_applied,

        top_factors=top_factors
    )


# ============================================================
# INCIDENT STORAGE
# ============================================================

UPLOAD_DIR = "incident_photos"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# Temporary in-memory incidents
# We will replace this with Supabase next.
incident_reports = []


# ============================================================
# WEATHER
# ============================================================

_weather_cache = {}
_elevation_cache = {}

def _cache_key(lat, lng):
    return (round(lat, 2), round(lng, 2))

def get_weather(lat, lng):

    key = _cache_key(lat, lng)
    if key in _weather_cache:
        return _weather_cache[key]

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}"
        f"&longitude={lng}"
        "&current=precipitation,wind_speed_10m"
        "&hourly=precipitation"
    )


    r = requests.get(
        url,
        timeout=8
    )


    r.raise_for_status()

    data = r.json()


    rain_now = (
        data["current"]["precipitation"]
    )


    wind = (
        data["current"]["wind_speed_10m"]
    )


    rain_6h = sum(
        data["hourly"]["precipitation"][:6]
    )


    result = (
        rain_now,
        rain_6h,
        wind
    )
    _weather_cache[key] = result
    return result


# ============================================================
# ELEVATION
# ============================================================

def get_elevation(lat, lng):

    key = _cache_key(lat, lng)
    if key in _elevation_cache:
        return _elevation_cache[key]

    url = (
        "https://api.open-meteo.com/v1/elevation"
        f"?latitude={lat}"
        f"&longitude={lng}"
    )


    r = requests.get(
        url,
        timeout=8
    )


    r.raise_for_status()

    data = r.json()

    elevation = data["elevation"][0]
    _elevation_cache[key] = elevation
    return elevation


# ============================================================
# BUILD ROUTE FEATURES
# ============================================================

def build_route_features(lat, lng):

    rain_now, rain_6h, wind = (
        get_weather(lat, lng)
    )


    elevation = get_elevation(
        lat,
        lng
    )


    month = time.localtime().tm_mon


    return RouteFeatures(

        rainfall_last_24h_mm=rain_now,

        rainfall_forecast_6h_mm=rain_6h,

        wind_speed_kmh=wind,

        elevation_m=elevation,

        slope_deg=5,

        landslide_prone_zone=0,

        past_closures_90d=0,

        road_type_highway=1,

        is_monsoon_month=
            1 if 6 <= month <= 9 else 0
    )


# ============================================================
# FIND NEARBY INCIDENT
# ============================================================

def nearby_incident(
    lat,
    lng,
    radius_km=5
):

    result = (
        supabase.table("active_incidents")
        .select("lat, lng, incident_type, severity")
        .eq("status", "Active")
        .execute()
    )

    for report in result.data:

        if report["lat"] is None or report["lng"] is None:
            continue

        dist = (
            (
                report["lat"] - lat
            ) ** 2
            +
            (
                report["lng"] - lng
            ) ** 2
        ) ** 0.5 * 111


        if dist < radius_km:

            return report


    return None


# ============================================================
# SCORE ROUTE POINT
# ============================================================

def score_point(lat, lng):

    features = build_route_features(
        lat,
        lng
    )


    bundle = load_model()

    model = bundle["model"]

    feature_order = bundle["features"]


    X = pd.DataFrame(
        [features.dict()]
    )[feature_order]


    ml_probability = float(
        model.predict_proba(X)[0, 1]
    )


    final_probability, _ = (
        apply_safety_override(
            features,
            ml_probability
        )
    )


    hit = nearby_incident(
        lat,
        lng
    )


    if hit:

        final_probability = max(
            final_probability,
            0.95
        )


    return (
        final_probability,
        hit is not None
    )


# ============================================================
# UPLOAD INCIDENT
# ============================================================

@app.post("/upload-incident")
async def upload_incident(
    lat: float,
    lng: float,
    file: UploadFile = File(...)
):

    report_id = str(
        uuid.uuid4()
    )


    filepath = (
        f"{UPLOAD_DIR}/{report_id}.jpg"
    )


    with open(
        filepath,
        "wb"
    ) as f:

        f.write(
            await file.read()
        )


    supabase.table("active_incidents").insert({

        "report_id": report_id,

        "lat": lat,

        "lng": lng,

        "filename": filepath,

        "status": "Active"
    }).execute()


    return {

        "status": "reported",

        "id": report_id
    }


# ============================================================
# GET ACTIVE INCIDENTS
# ============================================================

@app.get("/active-incidents")
def get_active_incidents():

    result = (
        supabase.table("active_incidents")
        .select("*")
        .eq("status", "Active")
        .order("created_at", desc=True)
        .execute()
    )

    return {
        "incidents": result.data
    }


# ============================================================
# RESOLVE INCIDENT
# ============================================================

@app.delete(
    "/resolve-incident/{report_id}"
)
def resolve_incident(
    report_id: str
):

    global incident_reports


    incident_reports = [
        r
        for r in incident_reports
        if r["id"] != report_id
    ]


    return {
        "status": "resolved"
    }


# ============================================================
# ROUTE RISK
# ============================================================

@app.get("/route-risk")
def route_risk(
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float
):

    url = (
        "https://router.project-osrm.org"
        "/route/v1/driving/"
        f"{start_lng},{start_lat};"
        f"{end_lng},{end_lat}"
        "?alternatives=true"
        "&overview=full"
        "&geometries=geojson"
    )


    response = requests.get(
        url,
        timeout=10
    )


    response.raise_for_status()

    data = response.json()


    routes = data["routes"]


    # Collect every route's sample points first, so we can score
    # ALL of them in parallel (much faster than one-by-one).
    route_sample_points = []

    for route in routes:
        coords = route["geometry"]["coordinates"]
        sample_points = [
            coords[0],
            coords[len(coords) // 2],
            coords[-1],
        ]
        route_sample_points.append(sample_points)

    flat_points = [
        (lat, lng)
        for sample_points in route_sample_points
        for lng, lat in sample_points
    ]

    with ThreadPoolExecutor(max_workers=12) as executor:
        flat_scores = list(
            executor.map(lambda p: score_point(p[0], p[1]), flat_points)
        )

    results = []
    idx = 0

    for i, route in enumerate(routes):
        coords = route["geometry"]["coordinates"]
        sample_points = route_sample_points[i]

        max_risk = 0
        incident_hit = False

        for _ in sample_points:
            risk, hit = flat_scores[idx]
            idx += 1
            max_risk = max(max_risk, risk)
            incident_hit = incident_hit or hit

        results.append({
            "coordinates": coords,
            "risk_score": round(max_risk * 100, 1),
            "incident_reported": incident_hit,
        })


    results.sort(
        key=lambda r:
            r["risk_score"]
    )


    return {

        "routes": results,

        "recommended": results[0]
    }

# ============================================================
# GOV DISASTER ALERTS (NDMA SACHET — real government data)
# ============================================================

import xml.etree.ElementTree as ET

NDMA_FEED_URL = "https://sachet.ndma.gov.in/cap_public_website/rss/rss_india.xml"

NER_STATES = [
    "Assam",
    "Arunachal Pradesh",
    "Meghalaya",
    "Manipur",
    "Mizoram",
    "Nagaland",
    "Tripura",
    "Sikkim",
]

NER_AGENCY_HINTS = [
    "ASDMA",
    "IMD Guwahati",
    "IMD Shillong",
    "IMD Agartala",
    "IMD Itanagar",
    "IMD Aizawl",
    "IMD Imphal",
    "IMD Kohima",
    "IMD Gangtok",
    "SDMA, IMD, Shillong",
]
AGENCY_TO_STATE ={
    "ASDMA": "Assam",
    "IMD Guwahati": "Assam",
    "IMD Shillong": "Meghalaya",
    "SDMA, IMD, Shillong": "Meghalaya",
    "IMD Agartala": "Tripura",
    "IMD Itanagar": "Arunachal Pradesh",
    "IMD Aizawl": "Mizoram",
    "IMD Imphal": "Manipur",
    "IMD Kohima": "Nagaland",
    "IMD Gangtok": "Sikkim",
}
def fetch_ner_gov_alerts():

    response = requests.get(
        NDMA_FEED_URL,
        timeout=15
    )

    response.raise_for_status()

    root = ET.fromstring(response.text)

    items = root.findall(".//item")

    ner_alerts = []

    for item in items:

        title = item.findtext("title", default="")
        link = item.findtext("link", default="")
        category = item.findtext("category", default="")
        author = item.findtext("author", default="")
        guid = item.findtext("guid", default="")
        pub_date = item.findtext("pubDate", default="")

        combined_text = f"{title} {author}"

        matched_state = None

        for state in NER_STATES:
            if state.lower() in combined_text.lower():
                matched_state = state
                break

        matched_agency = None

        for hint in NER_AGENCY_HINTS:
            if hint.lower() in combined_text.lower():
                matched_agency = hint
                break

        if matched_state or matched_agency:

            ner_alerts.append({
                "title": title,
                "raw_description": title,
                "state_matched": matched_state,
                "district_matched": None,
                "source_agency": author,
                "category": category,
                "published_at": pub_date,
                "link": link,
                "ndma_guid": guid,
            })

    return ner_alerts

@app.post("/sync-gov-alerts")
def sync_gov_alerts():

    alerts = fetch_ner_gov_alerts()

    inserted_count = 0
    skipped_count = 0

    for alert in alerts:

        try:

            supabase.table("gov_alerts").insert(alert).execute()

            inserted_count += 1

        except Exception:

            skipped_count += 1

    return {
        "total_ner_alerts_found": len(alerts),
        "inserted": inserted_count,
        "skipped_duplicates": skipped_count
    }


@app.get("/gov-alerts")
def get_gov_alerts():

    result = (
        supabase.table("gov_alerts")
        .select("*")
        .order("published_at", desc=True)
        .limit(50)
        .execute()
    )

    return {"alerts": result.data}
@app.post("/sync-gov-alerts")
def sync_gov_alerts():

    alerts = fetch_ner_gov_alerts()

    inserted_count = 0
    skipped_count = 0

    for alert in alerts:

        try:

            supabase.table("gov_alerts").insert(alert).execute()

            inserted_count += 1

        except Exception:

            skipped_count += 1

    return {
        "total_ner_alerts_found": len(alerts),
        "inserted": inserted_count,
        "skipped_duplicates": skipped_count
    }


@app.get("/gov-alerts")
def get_gov_alerts():

    result = (
        supabase.table("gov_alerts")
        .select("*")
        .order("published_at", desc=True)
        .limit(50)
        .execute()
    )

    return {"alerts": result.data}