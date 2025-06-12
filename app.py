import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)

# --- Global Variables ---
DATA_DIR = 'data'
MODELS_DIR = 'models'
WEATHER_DATA_PATH = os.path.join(DATA_DIR, 'weather.csv')
CROP_DATA_PATH = os.path.join(DATA_DIR, 'crop.csv')
WEATHER_MODEL_PATH = os.path.join(MODELS_DIR, 'weather_model.pkl')
WATER_MODEL_PATH = os.path.join(MODELS_DIR, 'water_model.pkl')

# --- Machine Learning Model Training ---

def train_models_if_not_exist():
    """
    Checks for the existence of model files. If they don't exist, it trains them
    from the CSV data files and saves them.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- Train Weather Prediction Model ---
    if not os.path.exists(WEATHER_MODEL_PATH):
        print("Weather model not found. Training a new one...")
        if not os.path.exists(WEATHER_DATA_PATH):
            raise FileNotFoundError(f"'{WEATHER_DATA_PATH}' not found. Please create it.")
        
        weather_df = pd.read_csv(WEATHER_DATA_PATH)
        weather_df['weather_simple'] = weather_df['weather'].apply(lambda x: 'RAINY' if x in ['rain', 'drizzle', 'snow'] else 'SUNNY')
        X_weather = weather_df[['precipitation', 'temp_max', 'temp_min', 'wind']]
        y_weather = weather_df['weather_simple']
        weather_model = RandomForestClassifier(n_estimators=100, random_state=42)
        weather_model.fit(X_weather, y_weather)
        joblib.dump(weather_model, WEATHER_MODEL_PATH)
        print("Weather model trained and saved.")

    # --- Train Crop Water Requirement Model ---
    if not os.path.exists(WATER_MODEL_PATH):
        print("Water requirement model not found. Training a new, more robust one...")
        if not os.path.exists(CROP_DATA_PATH):
             raise FileNotFoundError(f"'{CROP_DATA_PATH}' not found. Please create it.")
        
        crop_df = pd.read_csv(CROP_DATA_PATH)
        crop_df = crop_df[crop_df['WEATHER CONDITION'].isin(['SUNNY', 'RAINY'])]
        
        X_crop = crop_df.drop(columns=['WATER REQUIREMENT', 'TEMPERATURE'])
        y_crop = crop_df['WATER REQUIREMENT']
        
        # **ERROR FIX:** Define all possible categories that can appear in the form.
        # This makes the model aware of all dropdown options, even if they aren't in crop.csv.
        crop_type_categories = ['RICE', 'BANANA', 'SOYBEAN', 'CABBAGE', 'POTATO', 'MELON', 'MAIZE', 'CITRUS', 'BEAN', 'WHEAT', 'MUSTARD', 'COTTON', 'SUGARCANE', 'TOMATO', 'ONION']
        soil_type_categories = ['DRY', 'HUMID', 'WET']
        region_categories = ['DESSERT', 'SEMI ARID', 'SEMI HUMID', 'HUMID']
        weather_condition_categories = ['SUNNY', 'RAINY']
        
        categorical_features = ['CROP TYPE', 'SOIL TYPE', 'REGION', 'WEATHER CONDITION']
        
        # Create the OneHotEncoder and explicitly provide it with all possible categories.
        encoder = OneHotEncoder(
            categories=[
                crop_type_categories,
                soil_type_categories,
                region_categories,
                weather_condition_categories
            ],
            handle_unknown='ignore' # Ignores any unexpected values not in the lists
        )
        
        preprocessor = ColumnTransformer(
            transformers=[('cat', encoder, categorical_features)],
            remainder='passthrough'
        )
        
        water_model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        water_model_pipeline.fit(X_crop, y_crop)
        joblib.dump(water_model_pipeline, WATER_MODEL_PATH)
        print("Water requirement model (pipeline) trained and saved.")


# --- Flask API Endpoints ---
@app.route('/')
def home():
    """Serves the main HTML page from the root directory."""
    return send_from_directory('.', 'index.html')

@app.route('/predict_weather', methods=['POST'])
def predict_weather():
    """Predicts the weather based on sensor data."""
    try:
        data = request.get_json()
        sensor_data = pd.DataFrame({
            'precipitation': [float(data['precipitation'])],
            'temp_max': [float(data['temp_max'])],
            'temp_min': [float(data['temp_min'])],
            'wind': [float(data['wind'])]
        })
        weather_model = joblib.load(WEATHER_MODEL_PATH)
        prediction = weather_model.predict(sensor_data)
        return jsonify({'weather': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_water', methods=['POST'])
def predict_water():
    """Predicts the crop water requirement."""
    try:
        data = request.get_json()
        farm_data = pd.DataFrame({
            'CROP TYPE': [data['crop_type']],
            'SOIL TYPE': [data['soil_type']],
            'REGION': [data['region']],
            'WEATHER CONDITION': [data['weather']]
        })
        water_model_pipeline = joblib.load(WATER_MODEL_PATH)
        water_prediction = water_model_pipeline.predict(farm_data)
        return jsonify({'water_requirement': round(water_prediction[0], 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# --- Main Execution ---
if __name__ == '__main__':
    train_models_if_not_exist()
    app.run(debug=True)

