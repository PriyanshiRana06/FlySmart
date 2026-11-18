
# FlySmart — AI-Powered Flight Intelligence & Savings Platform

## Live Deployment

FlySmart is publicly deployed using Render and Docker.

**Live Website:** https://flysmart-5kqe.onrender.com

**API Documentation:** https://flysmart-5kqe.onrender.com/docs

**Health Check:** https://flysmart-5kqe.onrender.com/health

Deployment Platform: Render
Containerization: Docker

FlySmart is an academic flight intelligence platform that combines machine learning-based flight price classification, historical domestic fare insights, and smart savings recommendations.

The platform helps users understand flight price categories, explore historical route fares, and discover applicable illustrative offers through a simple web interface.

## Features

### 1. Flight Price Predictor
- Predicts flight price categories: Low, Medium, and High.
- Uses a trained Random Forest classification model.
- Accepts airline, source city, destination city, departure time, arrival time, number of stops, travel class, flight duration, and days left before departure.
- Displays predicted price category, indicative price range, and recommendations.

### 2. Historical Fare Insights
- Explore historical domestic flight fares.
- Search routes using airport codes or city names.
- View historical fare statistics.
- Supports reverse-direction route lookup when an exact direction is unavailable.

### 3. Smart Offers
- Recommends illustrative student, bank/card, seasonal, and loyalty offers.
- Calculates estimated savings based on eligibility rules.
- Sorts eligible offers by estimated savings.
- Uses a rule-based recommendation system.

**Note:** Smart Offers are illustrative academic demonstrations and are not verified or redeemable live promotions.

## Technology Stack

| Component | Technologies |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, FastAPI |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Model Persistence | Joblib |
| Experiment Tracking | MLflow, SQLite |
| Testing | Pytest |
| Containerization | Docker |
| Continuous Integration | GitHub Actions |
| Large File Management | Git LFS |

## Machine Learning Methodology

### Dataset

The model is trained using `data/Dataset.csv`.

The continuous flight price column is converted into three categories using quantile-based discretization with Pandas `qcut()`:

- Low
- Medium
- High

### Input Features

**Categorical Features**
- Airline
- Source City
- Departure Time
- Stops
- Arrival Time
- Destination City
- Travel Class

**Numerical Features**
- Flight Duration
- Days Left Before Departure

### Data Preprocessing

The following preprocessing techniques are applied:

- SimpleImputer with most-frequent strategy for categorical features.
- OneHotEncoder for categorical feature encoding.
- SimpleImputer with median strategy for numerical features.
- StandardScaler for numerical feature scaling.
- ColumnTransformer to combine preprocessing pipelines.
- Scikit-learn Pipeline to combine preprocessing and classification.

### Model Training

Three machine learning experiments are performed:

1. Logistic Regression (C = 1.0)
2. Logistic Regression (C = 0.1)
3. Random Forest Classifier

The dataset is divided into:

| Dataset | Percentage |
|---|---:|
| Training | 60% |
| Validation | 20% |
| Testing | 20% |

Models are compared using validation Macro F1-score. The model with the highest validation F1-score is selected and evaluated on a separate test dataset.

### Model Evaluation

The selected Random Forest experiment achieved the following recorded results:

| Metric | Score |
|---|---:|
| Validation Macro F1-score | 0.9430 |
| Test Macro F1-score | 0.9432 |

Other evaluation metrics include:
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Classification Report

The final trained pipeline is saved as `model/model.joblib`.

## MLflow Experiment Tracking

MLflow is used to track model training experiments.

- Tracking URI: `sqlite:///mlflow.db`
- Experiment Name: `FlySmart_Flight_Classification`
- Tracks model parameters, validation metrics, and trained model artifacts.
- Stores final test metrics for the selected model.

## Project Structure

```text
FlySmart/
│
├── app.py
├── train.py
├── train_route_model.py
├── requirements.txt
├── requirements-serve.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── .gitattributes
│
├── data/
│   ├── Dataset.csv
│   ├── airports_reference.csv
│   ├── domestic_fares_clean.csv
│   └── fare-index.csv
│
├── model/
│   ├── model.joblib
│   └── route_model.joblib
│
├── static/
│
├── tests/
│
└── .github/
    └── workflows/
        └── ci.yml
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Opens the FlySmart homepage |
| GET | `/health` | Checks application and model status |
| GET | `/cities` | Retrieves supported cities |
| POST | `/predict` | Predicts flight price category |
| GET | `/routes` | Searches historical fares using airport codes |
| GET | `/routes/search` | Searches routes using city names |
| GET | `/routes/classify` | Classifies historical route fare levels |
| POST | `/offers/recommend` | Generates eligible illustrative offers |
| GET | `/docs` | Opens interactive API documentation |

## Installation and Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/PriyanshiRana06/FlySmart.git
cd FlySmart
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

### 3. Activate the Environment

For Linux / Ubuntu / WSL:

```bash
source .venv/bin/activate
```

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements-serve.txt
```

### 5. Run the Application

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open the application in your browser:

`http://localhost:8000`

Interactive API documentation:

`http://localhost:8000/docs`

Health check:

`http://localhost:8000/health`

## Running with Docker

### Build the Docker Image

```bash
docker build -t flysmart .
```

### Run the Container

```bash
docker run --rm --name flysmart-test -p 8000:8000 flysmart
```

Open:

`http://localhost:8000`

Docker packages the application and its serving dependencies into a container for consistent execution.

## Testing

FlySmart uses Pytest for automated testing.

Run the tests using:

```bash
python -m pytest -v
```

The test suite covers:

- Application health check.
- Homepage response.
- Valid flight prediction.
- Invalid flight input handling.

The current test suite contains four automated tests.

## Continuous Integration

GitHub Actions is configured to automatically run the test suite.

The workflow:
1. Checks out the GitHub repository.
2. Sets up Python 3.11.
3. Installs application and testing dependencies.
4. Runs automated Pytest tests.

The workflow is triggered on pushes and pull requests targeting the `main` branch.

## Limitations

- Flight price prediction provides price categories and indicative ranges, not guaranteed live ticket prices.
- Historical fare insights depend on the available historical dataset.
- Smart Offers are illustrative rule-based recommendations, not live verified promotions.
- Prediction results depend on the training dataset and supported input values.
- Docker provides containerization but does not automatically make the application publicly accessible.

## GitHub Repository

[FlySmart — GitHub Repository](https://github.com/Priy​​anshiRana06/FlySmart)

## Project Objective

The objective of FlySmart is to demonstrate the practical integration of machine learning, data preprocessing, REST API development, historical fare analysis, experiment tracking, automated testing, containerization, and continuous integration in an academic software project.
