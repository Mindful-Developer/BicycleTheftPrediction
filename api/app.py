from tkinter import N
from tkinter.scrolledtext import example

from flask import Flask
from flask_restx import Api, Resource, fields
import pandas as pd
import joblib
import os
import sys
import json

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_preprocessing import preprocess_features
from rtree_search import RTree

app = Flask(__name__)
api = Api(app, version='1.0', title='Bicycle Theft Prediction API',
          description='An API for predicting bicycle theft outcomes',
          doc='/docs')


# Initialize R-tree search
rtree = RTree()


# Define the models for request/response
prediction_input = api.model('PredictionInput', {
    'BIKE_MAKE': fields.String(required=False, description='Make of the bicycle', example='TREK'),
    'BIKE_MODEL': fields.String(required=False, description='Model of the bicycle', example='FX3'),
    'BIKE_TYPE': fields.String(required=True, description='Type of bicycle (REGULAR, MOUNTAIN, or RACING)', example='REGULAR'),
    'BIKE_SPEED': fields.Integer(required=False, description='Number of speeds', example=21),
    'BIKE_COLOUR': fields.String(required=False, description='Color of the bicycle (e.g., BLK, WHI, BLU)', example='BLK'),
    'BIKE_COST': fields.Float(required=False, description='Cost of the bicycle', example=800.0),
    'PREMISES_TYPE': fields.String(required=True, description='Type of premises (e.g., House, Commercial, Apartment)', example='House'),
    'LOCATION_TYPE': fields.String(required=True, description='Detailed location type (e.g., Single Home, Apartment, Commercial)', example='Apartment (Rooming House, Condo)'),
    'OCC_DATE': fields.String(required=True, description='Date of occurrence (YYYY-MM-DD)', example='2023-01-01'),
    'OCC_DOW': fields.String(required=True, description='Day of week', example='Sunday'),
    'OCC_HOUR': fields.Integer(required=True, description='Hour of occurrence (0-23)', example=14),
    'OCC_DOY': fields.Integer(required=True, description='Day of year (1-366)', example=1),
    'REPORT_DATE': fields.String(required=True, description='Date reported (YYYY-MM-DD)', example='2023-01-02'),
    'HOOD_140': fields.String(required=True, description='Neighborhood code (e.g., 080)', example='080'),
    'NEIGHBOURHOOD_140': fields.String(required=True, description='Full neighborhood name with code', example='Palmerston-Little Italy (80)')
})

prediction_output = api.model('PredictionOutput', {
    'status': fields.String(description='Predicted status (RECOVERED or STOLEN)'),
    'probability_recovered': fields.Float(description='Probability of recovery'),
    'probability_stolen': fields.Float(description='Probability of remaining stolen')
})

neighbourhood_input = api.model('NeighbourhoodInput', {
    'latitude': fields.Float(required=True, description='Latitude of the location', example=43.693449),
    'longitude': fields.Float(required=True, description='Longitude of the location', example=-79.433288)
})

neighbourhood_output = api.model('NeighbourhoodOutput', {
    'HOOD_140': fields.String(description='Neighborhood code'),
    'NEIGHBOURHOOD_140': fields.String(description='Neighborhood name')
})

error_output = api.model('ErrorOutput', {
    'error': fields.String(description='Error message')
})

# Load unique values
unique_values_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unique_values.json')
with open(unique_values_path, 'r') as f:
    unique_values = json.load(f)

# Load model and feature order
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'models', 'best_model.pkl')
feature_order_path = os.path.join(base_dir, 'models', 'feature_order.pkl')

# Ensure models directory exists
models_dir = os.path.dirname(model_path)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Load model and feature order if they exist
if os.path.exists(model_path) and os.path.exists(feature_order_path):
    model = joblib.load(model_path)
    feature_order = joblib.load(feature_order_path)
else:
    print("Warning: Model files not found. Please train the model first.")
    model = None
    feature_order = None

# Define namespaces
ns = api.namespace('', description='Prediction operations')

@ns.route('/health')
class Health(Resource):
    @api.doc(responses={200: 'API is healthy'})
    def get(self):
        """Check API health status"""
        return {'status': 'healthy'}


@ns.route('/predict')
class PredictTheft(Resource):
    @api.expect(prediction_input)
    @api.response(200, 'Success', prediction_output)
    @api.response(400, 'Validation Error', error_output)
    @api.response(503, 'Model Not Available', error_output)
    def post(self):
        """Predict bicycle theft outcome

        Returns the probability of a bicycle being recovered or remaining stolen based on the input features.
        """
        if model is None or feature_order is None:
            return {'error': "Model not loaded. Please train the model first."}, 503
        try:
            data = api.payload

            for field in ['BIKE_COST', 'BIKE_MAKE', 'BIKE_MODEL', 'BIKE_SPEED', 'BIKE_COLOUR']: # Optional fields
                if field not in data:
                    data[field] = None
                    
            df = pd.DataFrame([data])
            X = preprocess_features(df)
            X = X[feature_order]

            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0]

            status = 'RECOVERED' if prediction == 1 else 'STOLEN'
            prob_recovered = float(probability[1])

            return {
                'status': status,
                'probability_recovered': round(prob_recovered, 2),
                'probability_stolen': round(float(probability[0]), 2)
            }
        except Exception as e:
            return {'error': str(e)}, 400

@ns.route('/neighbourhood')
class Neighbourhood(Resource):
    @api.expect(neighbourhood_input)
    @api.response(200, 'Success', neighbourhood_output)
    @api.response(400, 'Validation Error', error_output)
    def post(self):
        """Get the neighborhood code and name for a given latitude and longitude"""
        try:
            lat = api.payload['latitude']
            lon = api.payload['longitude']

            # Call the R-tree search function here
            h140, n140 = rtree.search(lat, lon)

            return {
                'HOOD_140': h140,
                'NEIGHBOURHOOD_140': n140
            }
        except Exception as e:
            return {'error': str(e)}, 400

@ns.route('/options')
class Options(Resource):
    @api.doc(responses={200: 'Success', 500: 'Internal Server Error'})
    def get(self):
        """Get all available options for each field in the prediction input"""
        try:
            return unique_values
        except Exception as e:
            return {'error': str(e)}, 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
