# Bicycle Theft Prediction API

An advanced machine learning API that predicts whether a stolen bicycle is likely to be recovered or remain stolen, based on historical bicycle theft data.

## Features

- **Theft Prediction**: Get probability scores for bicycle recovery using machine learning
- **Location Services**: Convert latitude/longitude to neighborhood information
- **Field Options**: Access all valid options for bike makes, models, colors, and more
- **Data Preprocessing**: Automatic handling of data formatting and encoding

## API Endpoints

### 1. Prediction Endpoint (`/predict`)
Make predictions about bicycle theft outcomes:
```bash
POST /predict
```
Required fields:
- `BIKE_TYPE`: Type of bicycle (REGULAR, MOUNTAIN, or RACING)
- `PREMISES_TYPE`: Type of premises where incident occurred
- `LOCATION_TYPE`: Type of location
- `OCC_DATE`: Date of occurrence (YYYY-MM-DD)
- `OCC_DOW`: Day of week
- `OCC_HOUR`: Hour of occurrence (0-23)
- `OCC_DOY`: Day of year (1-366)
- `REPORT_DATE`: Date reported (YYYY-MM-DD)
- `HOOD_140`: Neighborhood code
- `NEIGHBOURHOOD_140`: Neighborhood name

Optional fields:
- `BIKE_MAKE`: Make of the bicycle
- `BIKE_MODEL`: Model of the bicycle
- `BIKE_SPEED`: Number of speeds
- `BIKE_COLOUR`: Color of the bicycle
- `BIKE_COST`: Cost of the bicycle

### 2. Neighborhood Lookup (`/neighbourhood`)
Get neighborhood information from coordinates:
```bash
POST /neighbourhood
```
Required fields:
- `latitude`: Latitude of the location
- `longitude`: Longitude of the location

### 3. Field Options (`/options`)
Get all valid options for input fields:
```bash
GET /options
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Mindful-Developer/BicycleTheftPrediction.git
cd BicycleTheftPrediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python >= 3.8
- Dependencies:
  - numpy >= 1.21.0
  - pandas >= 1.3.0
  - scikit-learn >= 0.24.2
  - xgboost >= 1.4.2
  - imbalanced-learn >= 0.8.1
  - matplotlib >= 3.4.3
  - seaborn >= 0.11.2
  - flask >= 2.0.1
  - flask-restx >= 1.1.0
  - joblib >= 1.0.1
  - requests >= 2.26.0

## Usage

1. Start the API server:
```bash
cd api/
python -m flask --app app.py run
```

2. Access the API documentation:
```
http://localhost:5000/docs
```

3. Make a prediction:
```python
import requests

data = {
    "BIKE_TYPE": "REGULAR",
    "PREMISES_TYPE": "House",
    "LOCATION_TYPE": "Single Home",
    "OCC_DATE": "2023-01-01",
    "OCC_DOW": "Sunday",
    "OCC_HOUR": 14,
    "OCC_DOY": 1,
    "REPORT_DATE": "2023-01-02",
    "HOOD_140": "080",
    "NEIGHBOURHOOD_140": "Palmerston-Little Italy (80)"
}

response = requests.post("http://localhost:5000/predict", json=data)
print(response.json())
```

## Group Members

- Hodan Ahmed Yusuf
- Dimitrios Avgerakis
- Chris Busse
- Gabriel Normand
- Nabanita Saha