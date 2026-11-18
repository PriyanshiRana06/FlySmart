
document.addEventListener("DOMContentLoaded", () => {

    // ==========================================
    // EXISTING HISTORICAL FLIGHT SEARCH ELEMENTS
    // ==========================================

    const citySuggestions = document.getElementById("citySuggestions");
    const originInput = document.getElementById("originCity");
    const destinationInput = document.getElementById("destinationCity");
    const departureDate = document.getElementById("departureDate");

    const searchForm = document.getElementById("routeSearchForm");
    const searchButton = document.getElementById("searchFlightsButton");

    const resultsSection = document.getElementById("fareResults");
    const fareCards = document.getElementById("fareCards");

    const searchError = document.getElementById("searchError");
    const searchedRoute = document.getElementById("searchedRoute");
    const searchedDate = document.getElementById("searchedDate");
    const fareSource = document.getElementById("fareSource");


    // ==========================================
    // ROUTE-BASED FARE CLASSIFICATION ELEMENTS
    // ==========================================

    const classifyRouteButton = document.getElementById("classifyRouteButton");

    const classificationError = document.getElementById("classificationError");

    const routeClassificationResults = document.getElementById(
        "routeClassificationResults"
    );

    const classificationRoute = document.getElementById("classificationRoute");
    const classificationDirection = document.getElementById("classificationDirection");

    const classificationMedian = document.getElementById("classificationMedian");
    const classificationCategory = document.getElementById("classificationCategory");

    const classificationObservations = document.getElementById(
        "classificationObservations"
    );

    const classificationP25 = document.getElementById("classificationP25");
    const classificationP75 = document.getElementById("classificationP75");

    const classificationUpdated = document.getElementById("classificationUpdated");
    const classificationNotice = document.getElementById("classificationNotice");
    const classificationSource = document.getElementById("classificationSource");


    // ==========================================
    // ML PREDICTION ELEMENTS
    // ==========================================

    const predictionForm = document.getElementById("predictionForm");
    const predictionButton = document.getElementById("predictFlightButton");

    const predictionError = document.getElementById("predictionError");
    const predictionResults = document.getElementById("predictionResults");

    const predictionCategory = document.getElementById("predictionCategory");
    const predictionRecommendations = document.getElementById(
        "predictionRecommendations"
    );


    if (!searchForm) return;


    // ==========================================
    // SET MINIMUM DEPARTURE DATE
    // ==========================================

    const today = new Date();

    const localToday = new Date(
        today.getTime() - today.getTimezoneOffset() * 60000
    ).toISOString().split("T")[0];

    departureDate.min = localToday;


    // ==========================================
    // LOAD CITIES FROM BACKEND
    // ==========================================

    async function loadCities() {

        try {

            const response = await fetch("/cities");

            if (!response.ok) {
                throw new Error("Unable to load cities.");
            }

            const data = await response.json();

            citySuggestions.innerHTML = "";

            data.cities.forEach(city => {

                const option = document.createElement("option");
                option.value = city;

                citySuggestions.appendChild(option);

            });

            console.log(`${data.total} cities loaded successfully.`);

        } catch (error) {

            searchError.textContent =
                "Unable to load city suggestions. Please refresh the page.";

            console.error(error);

        }
    }


    // ==========================================
    // CREATE HISTORICAL FARE CARD
    // ==========================================

    function createFareCard(label, value, description) {

        const card = document.createElement("div");
        card.className = "fare-card";

        const heading = document.createElement("h3");
        heading.textContent = label;

        const amount = document.createElement("p");
        amount.className = "fare-amount";
        amount.textContent = value;

        const detail = document.createElement("span");
        detail.textContent = description;

        card.append(heading, amount, detail);

        return card;
    }


    // ==========================================
    // DISPLAY HISTORICAL FARE RESULTS
    // ==========================================

    function displayResults(data, origin, destination, date) {

        fareCards.innerHTML = "";

        const result = data.results[0];

        searchedRoute.textContent = result.route;

        searchedDate.textContent =
            `Selected departure date: ${date}`;


        // Reverse-direction warning

        if (data.matched_direction === "reverse") {

            const directionNotice = document.createElement("div");

            directionNotice.className = "route-direction-notice";
            directionNotice.textContent = data.notice;

            fareCards.appendChild(directionNotice);

        }


        // Historical fare cards

        fareCards.append(
            createFareCard(
                "Median Fare",
                `₹${result.median_fare.toLocaleString("en-IN")}`,
                "Middle observed historical fare"
            ),

            createFareCard(
                "Recent Low",
                `₹${result.recent_low.toLocaleString("en-IN")}`,
                "Reported low over the recent 30 days"
            ),

            createFareCard(
                "Lower Fare Range",
                `₹${result.p25.toLocaleString("en-IN")}`,
                "25th percentile"
            ),

            createFareCard(
                "Upper Fare Range",
                `₹${result.p75.toLocaleString("en-IN")}`,
                "75th percentile"
            )
        );


        // Dataset information

        const observationInfo = document.createElement("p");

        observationInfo.textContent =
            `Based on ${result.observations} historical observations.`;


        const updatedInfo = document.createElement("p");

        updatedInfo.textContent =
            `Dataset last updated: ${result.last_updated}`;


        const sourceLink = document.createElement("a");

        sourceLink.href = result.source_url;
        sourceLink.target = "_blank";
        sourceLink.rel = "noopener noreferrer";
        sourceLink.textContent = "View original fare source";


        fareSource.replaceChildren(
            observationInfo,
            updatedInfo,
            sourceLink
        );


        resultsSection.hidden = false;

        resultsSection.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    }


    // ==========================================
    // HISTORICAL ROUTE SEARCH SUBMIT
    // ==========================================

    searchForm.addEventListener("submit", async (event) => {

        event.preventDefault();

        searchError.textContent = "";
        resultsSection.hidden = true;

        const origin = originInput.value.trim();
        const destination = destinationInput.value.trim();
        const date = departureDate.value;


        if (!origin || !destination || !date) {

            searchError.textContent =
                "Please enter both cities and select a departure date.";

            return;
        }


        if (origin.toLowerCase() === destination.toLowerCase()) {

            searchError.textContent =
                "Origin and destination cannot be the same.";

            return;
        }


        searchButton.disabled = true;
        searchButton.textContent = "Searching...";


        try {

            const params = new URLSearchParams({
                origin_city: origin,
                destination_city: destination
            });


            const response = await fetch(
                `/routes/search?${params.toString()}`
            );


            const data = await response.json();


            if (!response.ok) {

                throw new Error(
                    data.detail ||
                    "No fare information found for this route."
                );

            }


            if (!data.results || data.results.length === 0) {

                throw new Error("No fare information available.");

            }


            displayResults(data, origin, destination, date);


        } catch (error) {

            searchError.textContent = error.message;

            console.error("Flight search error:", error);


        } finally {

            searchButton.disabled = false;
            searchButton.textContent = "Search Flights →";

        }

    });


    // ==========================================
    // ROUTE-BASED FARE CLASSIFICATION
    // ==========================================

    if (classifyRouteButton) {

        classifyRouteButton.addEventListener("click", async () => {

            classificationError.textContent = "";
            routeClassificationResults.hidden = true;

            const origin = originInput.value.trim();
            const destination = destinationInput.value.trim();


            // Validate city inputs

            if (!origin || !destination) {

                classificationError.textContent =
                    "Please enter both origin and destination in the Flight Search section.";

                return;
            }


            if (origin.toLowerCase() === destination.toLowerCase()) {

                classificationError.textContent =
                    "Origin and destination cannot be the same.";

                return;
            }


            // Disable button while fetching data

            classifyRouteButton.disabled = true;
            classifyRouteButton.textContent = "Classifying Route...";


            try {

                const params = new URLSearchParams({
                    origin_city: origin,
                    destination_city: destination
                });


                const response = await fetch(
                    `/routes/classify?${params.toString()}`
                );


                const data = await response.json();


                if (!response.ok) {

                    throw new Error(
                        data.detail || "Unable to classify this route."
                    );

                }


                // Format prices in Indian currency format

                function formatPrice(value) {

                    return `₹${Number(value).toLocaleString("en-IN")}`;

                }


                // Display requested route

                classificationRoute.textContent =
                    data.requested_route;


                // Display direction information

                if (data.matched_direction === "reverse") {

                    classificationDirection.textContent =
                        `Fare data matched from the reverse route: ${data.matched_route}`;

                } else {

                    classificationDirection.textContent =
                        "Exact route match found.";

                }


                // Display historical median fare

                classificationMedian.textContent =
                    formatPrice(data.historical_median_fare);


                // Display predicted category

                classificationCategory.textContent =
                    data.predicted_category;


                // Display historical observations

                classificationObservations.textContent =
                    `${data.observations} historical observations`;


                // Display historical fare range

                classificationP25.textContent =
                    formatPrice(data.historical_fare_range.p25);

                classificationP75.textContent =
                    formatPrice(data.historical_fare_range.p75);


                // Display dataset update date

                classificationUpdated.textContent =
                    `Dataset last updated: ${data.last_updated}`;


                // Display reverse-route warning, if applicable

                if (data.notice) {

                    classificationNotice.textContent = data.notice;
                    classificationNotice.hidden = false;

                } else {

                    classificationNotice.textContent = "";
                    classificationNotice.hidden = true;

                }


                // Display original source link

                classificationSource.replaceChildren();

                if (data.source_url) {

                    const sourceLink = document.createElement("a");

                    sourceLink.href = data.source_url;
                    sourceLink.target = "_blank";
                    sourceLink.rel = "noopener noreferrer";
                    sourceLink.textContent = "View original fare source";

                    classificationSource.appendChild(sourceLink);

                }


                // Show classification results

                routeClassificationResults.hidden = false;

                routeClassificationResults.scrollIntoView({
                    behavior: "smooth",
                    block: "center"
                });


                console.log("Route classification successful:", data);


            } catch (error) {

                classificationError.textContent =
                    error.message ||
                    "Something went wrong while classifying the route.";

                console.error("Route classification error:", error);


            } finally {

                classifyRouteButton.disabled = false;
                classifyRouteButton.textContent = "Classify Route →";

            }

        });

    }


    // ==========================================
    // CONVERT CITY NAMES FOR ML MODEL
    // ==========================================

    function convertCityForModel(city) {

        const cityMap = {

            "bangalore": "Bangalore",
            "bengaluru": "Bangalore",

            "chennai": "Chennai",

            "delhi": "Delhi",
            "new delhi": "Delhi",

            "hyderabad": "Hyderabad",

            "kolkata": "Kolkata",

            "mumbai": "Mumbai"

        };

        return cityMap[city.trim().toLowerCase()] || null;

    }


    // ==========================================
    // CALCULATE DAYS LEFT UNTIL DEPARTURE
    // ==========================================

    function calculateDaysLeft(selectedDate) {

        const [year, month, day] = selectedDate
            .split("-")
            .map(Number);

        const [todayYear, todayMonth, todayDay] = localToday
            .split("-")
            .map(Number);

        const selectedUTC = Date.UTC(year, month - 1, day);

        const todayUTC = Date.UTC(
            todayYear,
            todayMonth - 1,
            todayDay
        );

        return Math.round(
            (selectedUTC - todayUTC) / (1000 * 60 * 60 * 24)
        );

    }


    // ==========================================
    // DISPLAY ML PREDICTION
    // ==========================================

    function displayPrediction(data) {

        predictionCategory.innerHTML = "";
        predictionRecommendations.innerHTML = "";


        // ======================================
        // PREDICTION CATEGORY CARD
        // ======================================

        const categoryCard = document.createElement("div");

        categoryCard.className = "prediction-category-card";


        const categoryHeading = document.createElement("h3");

        categoryHeading.textContent = "Predicted Price Category";


        const categoryValue = document.createElement("p");

        categoryValue.className = "prediction-category-value";

        categoryValue.textContent = data.predicted_category;


        categoryCard.append(
            categoryHeading,
            categoryValue
        );


        predictionCategory.appendChild(categoryCard);


        // ======================================
        // HISTORICAL TRAINING DATASET PRICE RANGE
        // ======================================

        if (data.category_price_range) {

            const rangeCard = document.createElement("div");

            rangeCard.className = "prediction-category-card";


            const rangeHeading = document.createElement("h3");

            rangeHeading.textContent =
                "Estimated Historical Price Range";


            const rangeValue = document.createElement("p");

            rangeValue.className = "prediction-category-value";


            const minPrice = Number(
                data.category_price_range.min
            ).toLocaleString("en-IN");


            const maxPrice = Number(
                data.category_price_range.max
            ).toLocaleString("en-IN");


            rangeValue.textContent =
                `₹${minPrice} – ₹${maxPrice}`;


            const rangeNote = document.createElement("span");

            rangeNote.textContent =
                "Based on historical training dataset price bands. Not a live ticket price.";


            rangeCard.append(
                rangeHeading,
                rangeValue,
                rangeNote
            );


            predictionCategory.appendChild(rangeCard);

        }


        // ======================================
        // SMART RECOMMENDATIONS
        // ======================================

        const recommendationHeading = document.createElement("h3");

        recommendationHeading.textContent =
            "Smart Recommendations";


        const recommendationList = document.createElement("ul");


        if (
            data.recommendations &&
            data.recommendations.length > 0
        ) {

            data.recommendations.forEach(recommendation => {

                const item = document.createElement("li");

                item.textContent = recommendation;

                recommendationList.appendChild(item);

            });

        } else {

            const item = document.createElement("li");

            item.textContent =
                "No additional recommendations available.";

            recommendationList.appendChild(item);

        }


        predictionRecommendations.append(
            recommendationHeading,
            recommendationList
        );


        // ======================================
        // SHOW PREDICTION RESULTS
        // ======================================

        predictionResults.hidden = false;

        predictionResults.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });

    }


    // ==========================================
    // ML PRICE PREDICTION FORM
    // ==========================================

    if (predictionForm) {

        predictionForm.addEventListener("submit", async (event) => {

            // IMPORTANT: Prevent page refresh

            event.preventDefault();

            predictionError.textContent = "";
            predictionResults.hidden = true;


            // ======================================
            // GET ROUTE DETAILS FROM SEARCH SECTION
            // ======================================

            const origin = originInput.value.trim();
            const destination = destinationInput.value.trim();
            const date = departureDate.value;


            if (!origin || !destination || !date) {

                predictionError.textContent =
                    "Please enter your origin, destination and departure date in the Flight Search section first.";

                return;

            }


            if (origin.toLowerCase() === destination.toLowerCase()) {

                predictionError.textContent =
                    "Origin and destination cannot be the same.";

                return;

            }


            // ======================================
            // CONVERT CITIES INTO ML-SUPPORTED NAMES
            // ======================================

            const sourceCity = convertCityForModel(origin);

            const destinationCity = convertCityForModel(destination);


            if (!sourceCity || !destinationCity) {

                predictionError.textContent =
                    "ML predictions currently support only Bangalore, Chennai, Delhi, Hyderabad, Kolkata and Mumbai.";

                return;

            }


            // ======================================
            // CALCULATE DAYS LEFT
            // ======================================

            const daysLeft = calculateDaysLeft(date);


            if (daysLeft < 1 || daysLeft > 49) {

                predictionError.textContent =
                    "For ML prediction, select a departure date between 1 and 49 days from today.";

                return;

            }


            // ======================================
            // READ REMAINING FLIGHT DETAILS
            // ======================================

            const airline = document.getElementById("airline").value;

            const departureTime =
                document.getElementById("departureTime").value;

            const arrivalTime =
                document.getElementById("arrivalTime").value;

            const stops = document.getElementById("stops").value;

            const flightClass =
                document.getElementById("flightClass").value;

            const duration = Number(
                document.getElementById("flightDuration").value
            );


            if (
                !airline ||
                !departureTime ||
                !arrivalTime ||
                !stops ||
                !flightClass ||
                !duration
            ) {

                predictionError.textContent =
                    "Please fill in all flight details.";

                return;

            }


            if (!Number.isFinite(duration) || duration <= 0 || duration > 100) {

                predictionError.textContent =
                    "Flight duration must be greater than 0 and no more than 100 hours.";

                return;

            }


            // ======================================
            // PREPARE DATA FOR FASTAPI
            // ======================================

            const predictionData = {

                airline: airline,

                source_city: sourceCity,

                departure_time: departureTime,

                stops: stops,

                arrival_time: arrivalTime,

                destination_city: destinationCity,

                "class": flightClass,

                duration: duration,

                days_left: daysLeft

            };


            console.log("Sending ML prediction data:", predictionData);


            // ======================================
            // DISABLE BUTTON DURING PREDICTION
            // ======================================

            predictionButton.disabled = true;

            predictionButton.textContent = "Predicting...";


            try {

                // Send request to ML prediction API

                const response = await fetch("/predict", {

                    method: "POST",

                    headers: {
                        "Content-Type": "application/json"
                    },

                    body: JSON.stringify(predictionData)

                });


                const data = await response.json();


                if (!response.ok) {

                    let errorMessage = "Unable to generate prediction.";

                    if (typeof data.detail === "string") {

                        errorMessage = data.detail;

                    } else if (Array.isArray(data.detail)) {

                        errorMessage = data.detail
                            .map(item => item.msg)
                            .join(", ");

                    }

                    throw new Error(errorMessage);

                }


                // ======================================
                // DISPLAY PREDICTION RESULT
                // ======================================

                displayPrediction(data);

                console.log("ML prediction successful:", data);


            } catch (error) {

                predictionError.textContent =
                    error.message ||
                    "Something went wrong while predicting the flight price.";

                console.error("ML prediction error:", error);


            } finally {

                predictionButton.disabled = false;

                predictionButton.textContent =
                    "Predict Flight Price →";

            }

        });

    }


    // ==========================================
    // INITIALIZE PAGE
    // ==========================================

    loadCities();

});
