
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import joblib
import json


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "model.joblib"
FARE_PATH = BASE_DIR / "data" / "domestic_fares_clean.csv"
LOG_PATH = BASE_DIR / "logs" / "predictions.log"
HTML_PATH = BASE_DIR / "static" / "index.html"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


# ==========================================
# LOAD ML MODEL AND FARE DATASET
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Run train.py first."
    )

if not FARE_PATH.exists():
    raise FileNotFoundError(
        "Cleaned fare dataset not found."
    )

model = joblib.load(MODEL_PATH)

fare_data = pd.read_csv(FARE_PATH)


# ==========================================
# EXISTING ML PRICE CATEGORY RANGES
# ==========================================

PRICE_RANGES = {
    "Low": {
        "min": 1998,
        "max": 4501,
        "currency": "INR"
    },

    "Medium": {
        "min": 4501,
        "max": 6450,
        "currency": "INR"
    },

    "High": {
        "min": 6450,
        "max": 35145,
        "currency": "INR"
    }
}


# ==========================================
# ROUTE-BASED FARE CLASSIFICATION
# Based on historical dataset quartiles
# ==========================================

ROUTE_FARE_THRESHOLDS = {
    "low": 6260,
    "high": 11476
}


def classify_route_fare(median_fare):

    if median_fare < ROUTE_FARE_THRESHOLDS["low"]:
        return "Low"

    elif median_fare <= ROUTE_FARE_THRESHOLDS["high"]:
        return "Medium"

    else:
        return "High"


# ==========================================
# EXTRACT CITY NAMES FROM FARE DATA
# ==========================================

route_parts = fare_data["route"].str.split(
    " to ",
    n=1,
    expand=True
)

fare_data["_origin_city"] = route_parts[0].str.strip()

fare_data["_destination_city"] = route_parts[1].str.strip()


available_cities = sorted(
    set(fare_data["_origin_city"].dropna()) |
    set(fare_data["_destination_city"].dropna())
)


# ==========================================
# CITY NAME ALIASES
# ==========================================

CITY_ALIASES = {
    "delhi": "New Delhi",
    "bangalore": "Bengaluru"
}


def normalize_city(city: str):

    city = city.strip()

    return CITY_ALIASES.get(
        city.casefold(),
        city
    )


print("FlySmart ML model loaded successfully.")
print(f"Loaded {len(fare_data)} domestic routes.")
print(f"Loaded {len(available_cities)} searchable cities.")


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="FlySmart API",
    description=(
        "Flight Price Classification and "
        "Domestic Route Fare Comparison System"
    ),
    version="3.3.0"
)


app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static"
)


# ==========================================
# FLIGHT INPUT VALIDATION
# ==========================================

class FlightInput(BaseModel):

    model_config = ConfigDict(populate_by_name=True)

    airline: Literal[
        "AirAsia",
        "Air_India",
        "GO_FIRST",
        "Indigo",
        "SpiceJet",
        "Vistara"
    ]

    source_city: Literal[
        "Bangalore",
        "Chennai",
        "Delhi",
        "Hyderabad",
        "Kolkata",
        "Mumbai"
    ]

    departure_time: Literal[
        "Afternoon",
        "Early_Morning",
        "Evening",
        "Late_Night",
        "Morning",
        "Night"
    ]

    stops: Literal[
        "zero",
        "one",
        "two_or_more"
    ]

    arrival_time: Literal[
        "Afternoon",
        "Early_Morning",
        "Evening",
        "Late_Night",
        "Morning",
        "Night"
    ]

    destination_city: Literal[
        "Bangalore",
        "Chennai",
        "Delhi",
        "Hyderabad",
        "Kolkata",
        "Mumbai"
    ]

    flight_class: Literal[
        "Business",
        "Economy"
    ] = Field(alias="class")

    duration: float = Field(
        gt=0,
        le=100
    )

    days_left: int = Field(
        ge=1,
        le=49
    )


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "application": "FlySmart",
        "model_loaded": model is not None,
        "fare_dataset_loaded": fare_data is not None,
        "available_routes": len(fare_data),
        "available_cities": len(available_cities)
    }


# ==========================================
# GET AVAILABLE CITIES
# ==========================================

@app.get("/cities")
def get_cities():

    return {
        "total": len(available_cities),
        "cities": available_cities
    }


# ==========================================
# FORMAT HISTORICAL FARE RESULTS
# ==========================================

def format_fare_results(
    matches,
    requested_route=None,
    matched_direction="requested"
):

    results = []

    for _, row in matches.iterrows():

        results.append({

            "route": row["route"],

            "origin": row["origin"],

            "destination": row["destination"],

            "observations": int(row["observations"]),

            "median_fare": int(row["median"]),

            "p25": int(row["p25"]),

            "p75": int(row["p75"]),

            "recent_low": int(row["low_30d"]),

            "currency": row["currency"],

            "last_updated": row["updated"],

            "source_url": row["url"]

        })


    if matched_direction == "reverse":

        notice = (
            f"Historical fare information for {requested_route} "
            "is unavailable. Showing historical information for "
            f"the reverse route: {results[0]['route']}. "
            "These fares apply to the reverse direction only."
        )

    else:

        notice = None


    return {

        "total": len(results),

        "fare_type": "Historical route fare estimates",

        "requested_route": requested_route,

        "matched_direction": matched_direction,

        "notice": notice,

        "results": results

    }


# ==========================================
# AIRPORT CODE ROUTE SEARCH
# ==========================================

@app.get("/routes")
def search_routes(

    origin: str = Query(
        min_length=3,
        max_length=3,
        pattern="^[A-Za-z]{3}$"
    ),

    destination: str = Query(
        min_length=3,
        max_length=3,
        pattern="^[A-Za-z]{3}$"
    )

):

    origin = origin.upper()
    destination = destination.upper()

    if origin == destination:

        raise HTTPException(
            status_code=422,
            detail="Origin and destination cannot be the same."
        )

    matches = fare_data[
        (fare_data["origin"] == origin) &
        (fare_data["destination"] == destination)
    ]

    if matches.empty:

        raise HTTPException(
            status_code=404,
            detail="No fare information found for this route."
        )

    return format_fare_results(matches)


# ==========================================
# CITY NAME ROUTE SEARCH
# WITH REVERSE DIRECTION FALLBACK
# ==========================================

@app.get("/routes/search")
def search_routes_by_city(

    origin_city: str = Query(
        min_length=2,
        max_length=60
    ),

    destination_city: str = Query(
        min_length=2,
        max_length=60
    )

):

    origin = normalize_city(origin_city)
    destination = normalize_city(destination_city)

    if origin.casefold() == destination.casefold():

        raise HTTPException(
            status_code=422,
            detail="Origin and destination cannot be the same."
        )

    origin_match = fare_data["_origin_city"].str.casefold().eq(
        origin.casefold()
    )

    destination_match = fare_data["_destination_city"].str.casefold().eq(
        destination.casefold()
    )

    # Search exact requested direction

    matches = fare_data[
        origin_match & destination_match
    ]

    requested_route = f"{origin} to {destination}"

    if not matches.empty:

        return format_fare_results(
            matches,
            requested_route=requested_route,
            matched_direction="requested"
        )

    # Search reverse direction if exact route is unavailable

    reverse_origin_match = fare_data[
        "_origin_city"
    ].str.casefold().eq(destination.casefold())

    reverse_destination_match = fare_data[
        "_destination_city"
    ].str.casefold().eq(origin.casefold())

    reverse_matches = fare_data[
        reverse_origin_match & reverse_destination_match
    ]

    if not reverse_matches.empty:

        return format_fare_results(
            reverse_matches,
            requested_route=requested_route,
            matched_direction="reverse"
        )

    raise HTTPException(
        status_code=404,
        detail=(
            f"No fare information found for {requested_route} "
            "or its reverse route."
        )
    )


# ==========================================
# ROUTE-BASED FARE CLASSIFICATION
# ==========================================

@app.get("/routes/classify")
def classify_route(

    origin_city: str = Query(
        min_length=2,
        max_length=60
    ),

    destination_city: str = Query(
        min_length=2,
        max_length=60
    )

):

    origin = normalize_city(origin_city)
    destination = normalize_city(destination_city)

    if origin.casefold() == destination.casefold():

        raise HTTPException(
            status_code=422,
            detail="Origin and destination cannot be the same."
        )

    requested_route = f"{origin} to {destination}"

    # Find exact route

    matches = fare_data[
        fare_data["_origin_city"].str.casefold().eq(
            origin.casefold()
        ) &
        fare_data["_destination_city"].str.casefold().eq(
            destination.casefold()
        )
    ]

    matched_direction = "requested"

    # Try reverse route if exact route is unavailable

    if matches.empty:

        matches = fare_data[
            fare_data["_origin_city"].str.casefold().eq(
                destination.casefold()
            ) &
            fare_data["_destination_city"].str.casefold().eq(
                origin.casefold()
            )
        ]

        matched_direction = "reverse"

    if matches.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                f"No fare information found for {requested_route} "
                "or its reverse route."
            )
        )

    # Get the matching route record

    row = matches.iloc[0]

    median_fare = float(row["median"])

    fare_category = classify_route_fare(median_fare)

    return {

        "requested_route": requested_route,

        "matched_route": row["route"],

        "matched_direction": matched_direction,

        "notice": (
            "Showing reverse-direction historical fare information. "
            "It is not an estimate for the requested direction."
            if matched_direction == "reverse"
            else None
        ),

        "predicted_category": fare_category,

        "historical_median_fare": round(median_fare),

        "historical_fare_range": {

            "p25": round(float(row["p25"])),

            "p75": round(float(row["p75"])),

            "currency": row["currency"]

        },

        "observations": int(row["observations"]),

        "last_updated": row["updated"],

        "source_url": row["url"],

        "classification_method": (
            "Historical route median fare using dataset quartile thresholds"
        )

    }


# ==========================================
# SMART RECOMMENDATION FUNCTION
# ==========================================

def generate_tips(flight):

    tips = []

    if flight.days_left >= 20:

        tips.append(
            "You are booking early. Compare available fares before booking."
        )

    elif flight.days_left <= 7:

        tips.append(
            "Your travel date is close. Check alternative timings and airlines."
        )

    else:

        tips.append(
            "Compare different airlines to find a suitable fare."
        )

    if flight.stops != "zero":

        tips.append(
            "Consider non-stop flights if you prefer a shorter journey."
        )

    if flight.flight_class == "Business":

        tips.append(
            "Compare Economy and Business fares if your travel needs allow."
        )

    if flight.duration > 5:

        tips.append(
            "For longer journeys, compare arrival timings and total travel duration."
        )

    return tips[:3]


# ==========================================
# EXISTING ML FLIGHT PRICE PREDICTION
# ==========================================

@app.post("/predict")
def predict_flight(flight: FlightInput):

    # Convert user input into a DataFrame

    input_data = pd.DataFrame([
        flight.model_dump(by_alias=True)
    ])

    # Generate prediction using trained ML model

    prediction = model.predict(input_data)[0]

    predicted_category = str(prediction)

    # Get corresponding price category range

    category_range = PRICE_RANGES.get(
        predicted_category
    )

    # Generate recommendations

    tips = generate_tips(flight)

    # Prepare prediction log

    log_entry = {

        "timestamp": datetime.now(timezone.utc).isoformat(),

        "input": flight.model_dump(by_alias=True),

        "prediction": predicted_category,

        "category_price_range": category_range

    }

    # Save prediction history

    with open(
        LOG_PATH,
        "a",
        encoding="utf-8"
    ) as log_file:

        log_file.write(
            json.dumps(log_entry) + "\n"
        )

    # Return prediction and price range

    return {

        "predicted_category": predicted_category,

        "category_price_range": category_range,

        "recommendations": tips

    }


# ==========================================
# HOMEPAGE
# ==========================================

@app.get("/")
def home():

    if not HTML_PATH.exists():

        raise HTTPException(
            status_code=404,
            detail="Homepage not found."
        )

    return FileResponse(HTML_PATH)

# ==========================================
# AI SMART OFFERS MODULE
# ==========================================

class OfferRequest(BaseModel):
    origin: str = Field(min_length=2, max_length=60)
    destination: str = Field(min_length=2, max_length=60)
    ticket_price: float = Field(gt=0, le=1000000)
    is_student: bool = False
    bank: str = "None"
    card_type: str = "None"
    loyalty_member: bool = False


# DEMO OFFER CATALOGUE
# These are illustrative offers, not live promotions.

OFFER_CATALOGUE = [

    {
        "id": "STUDENT01",
        "name": "Student Travel Benefit",
        "category": "Student",
        "description": "Sample student discount",
        "discount_type": "percentage",
        "discount_value": 8,
        "max_discount": 500,
        "minimum_fare": 2000,
        "requires_student": True,
        "bank": None,
        "loyalty_required": False
    },

    {
        "id": "BANK01",
        "name": "HDFC Card Savings",
        "category": "Bank & Card",
        "description": "Sample HDFC card discount",
        "discount_type": "percentage",
        "discount_value": 10,
        "max_discount": 750,
        "minimum_fare": 3000,
        "requires_student": False,
        "bank": "HDFC",
        "loyalty_required": False
    },

    {
        "id": "BANK02",
        "name": "ICICI Card Savings",
        "category": "Bank & Card",
        "description": "Sample ICICI card discount",
        "discount_type": "percentage",
        "discount_value": 7,
        "max_discount": 600,
        "minimum_fare": 2500,
        "requires_student": False,
        "bank": "ICICI",
        "loyalty_required": False
    },

    {
        "id": "SEASON01",
        "name": "Seasonal Travel Savings",
        "category": "Seasonal & Special",
        "description": "Sample seasonal discount",
        "discount_type": "percentage",
        "discount_value": 5,
        "max_discount": 300,
        "minimum_fare": 1500,
        "requires_student": False,
        "bank": None,
        "loyalty_required": False
    },

    {
        "id": "LOYAL01",
        "name": "Loyalty Member Benefit",
        "category": "Loyalty & Rewards",
        "description": "Sample loyalty member discount",
        "discount_type": "percentage",
        "discount_value": 5,
        "max_discount": 400,
        "minimum_fare": 2000,
        "requires_student": False,
        "bank": None,
        "loyalty_required": True
    }
]


def calculate_offer_savings(ticket_price, offer):

    discount = (
        ticket_price * offer["discount_value"] / 100
    )

    discount = min(
        discount,
        offer["max_discount"],
        ticket_price
    )

    return round(discount, 2)


def recommend_offers(request: OfferRequest):

    recommendations = []

    for offer in OFFER_CATALOGUE:

        # Check minimum booking amount
        if request.ticket_price < offer["minimum_fare"]:
            continue

        # Check student eligibility
        if offer["requires_student"] and not request.is_student:
            continue

        # Check bank eligibility
        if offer["bank"] is not None:

            if request.bank.strip().casefold() != offer["bank"].casefold():
                continue

            if request.card_type.casefold() == "none":
                continue

        # Check loyalty eligibility
        if offer["loyalty_required"] and not request.loyalty_member:
            continue

        # Calculate eligible discount
        savings = calculate_offer_savings(
            request.ticket_price,
            offer
        )

        final_price = round(
            request.ticket_price - savings,
            2
        )

        # Explain why the offer matches
        reasons = []

        if offer["requires_student"]:
            reasons.append("Student eligibility verified from user input")

        if offer["bank"]:
            reasons.append(f"Selected bank matches {offer['bank']}")

        if offer["loyalty_required"]:
            reasons.append("User selected loyalty membership")

        if not reasons:
            reasons.append("Meets the sample offer's minimum fare condition")

        recommendations.append({

            "offer_id": offer["id"],
            "name": offer["name"],
            "category": offer["category"],
            "description": offer["description"],
            "discount_percentage": offer["discount_value"],
            "savings": savings,
            "original_price": request.ticket_price,
            "final_price": final_price,
            "reasons": reasons,
            "offer_status": "DEMO - NOT A LIVE PROMOTION"

        })

    # Rank offers by calculated savings
    recommendations.sort(
        key=lambda item: item["savings"],
        reverse=True
    )

    return {

        "route": f"{request.origin} to {request.destination}",

        "original_price": request.ticket_price,

        "eligible_offers": len(recommendations),

        "recommendations": recommendations,

        "notice": (
            "These are illustrative academic project offers. "
            "They are not verified or redeemable live promotions."
        )

    }


@app.post("/offers/recommend")
def get_offer_recommendations(request: OfferRequest):

    if request.origin.strip().casefold() == request.destination.strip().casefold():
        raise HTTPException(
            status_code=400,
            detail="Origin and destination cannot be the same."
        )

    return recommend_offers(request)
