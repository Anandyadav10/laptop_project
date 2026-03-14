from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load and prepare the model
def load_model():
    try:
        # Try to load existing model
        with open('laptop_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
            return model_data['model'], model_data['label_encoders'], model_data['scaler']
    except FileNotFoundError:
        # Create model if it doesn't exist
        return create_model()

def create_model():
    # Load dataset
    df = pd.read_csv('laptopPrice.csv')
    
    # Data preprocessing
    df = df.drop(['rating', 'Number of Ratings', 'Number of Reviews'], axis=1)
    
    # Handle missing values
    df['processor_gnrtn'] = df['processor_gnrtn'].fillna('Not Available')
    
    # Convert columns with units to numeric values
    def extract_numeric(value):
        if pd.isna(value):
            return 0
        if isinstance(value, str):
            return int(''.join(filter(str.isdigit, value)) or 0)
        return int(value)
    
    # Convert columns with units (GB) to numeric
    df['ram_gb'] = df['ram_gb'].apply(extract_numeric)
    df['ssd'] = df['ssd'].apply(extract_numeric)
    df['hdd'] = df['hdd'].apply(extract_numeric)
    df['graphic_card_gb'] = df['graphic_card_gb'].apply(extract_numeric)
    
    # Separate features and target
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    # Label encoding for categorical variables
    categorical_cols = ['brand', 'processor_brand', 'processor_name', 'processor_gnrtn', 
                         'ram_type', 'os', 'os_bit', 'weight', 'warranty', 
                         'Touchscreen', 'msoffice']
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Save model
    model_data = {
        'model': model,
        'label_encoders': label_encoders,
        'scaler': scaler
    }
    
    with open('laptop_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    return model, label_encoders, scaler

# Load model with error handling
try:
    model, label_encoders, scaler = load_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model, label_encoders, scaler = None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model is loaded
        if model is None or label_encoders is None or scaler is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded properly. Please check server logs.'
            })
        
        # Get form data
        data = request.json
        
        # Prepare input data
        input_data = {
            'brand': data['brand'],
            'processor_brand': data['processor_brand'],
            'processor_name': data['processor_name'],
            'processor_gnrtn': data['processor_gnrtn'],
            'ram_gb': data['ram_gb'],
            'ram_type': data['ram_type'],
            'ssd': int(data['ssd']),
            'hdd': int(data['hdd']),
            'os': data['os'],
            'os_bit': data['os_bit'],
            'graphic_card_gb': int(data['graphic_card_gb']),
            'weight': data['weight'],
            'warranty': data['warranty'],
            'Touchscreen': data['touchscreen'],
            'msoffice': data['msoffice']
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Apply label encoding
        for col in label_encoders:
            if col in input_df.columns:
                try:
                    input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
                except ValueError:
                    # Handle unknown categories
                    input_df[col] = 0
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        return jsonify({
            'success': True,
            'predicted_price': round(prediction, 2),
            'message': f'Predicted laptop price: ₹{round(prediction, 2):,}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/get_options')
def get_options():
    try:
        df = pd.read_csv('laptopPrice.csv')
        
        # Handle missing values
        df['processor_gnrtn'] = df['processor_gnrtn'].fillna('Not Available')
        
        options = {
            'brands': sorted(df['brand'].unique().tolist()),
            'processor_brands': sorted(df['processor_brand'].unique().tolist()),
            'processor_names': sorted(df['processor_name'].unique().tolist()),
            'processor_generations': sorted(df['processor_gnrtn'].dropna().unique().tolist()),
            'ram_types': sorted(df['ram_type'].unique().tolist()),
            'operating_systems': sorted(df['os'].unique().tolist()),
            'os_bits': sorted(df['os_bit'].unique().tolist()),
            'weights': sorted(df['weight'].unique().tolist()),
            'warranties': sorted(df['warranty'].unique().tolist()),
            'touchscreen_options': sorted(df['Touchscreen'].unique().tolist()),
            'msoffice_options': sorted(df['msoffice'].unique().tolist())
        }
        
        return jsonify(options)
        
    except Exception as e:
        return jsonify({'error': str(e)})

def initialize_app():
    """Initialize the application and ensure model is loaded"""
    global model, label_encoders, scaler
    try:
        model, label_encoders, scaler = load_model()
        print("✅ Application initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize application: {e}")
        return False

if __name__ == '__main__':
    if initialize_app():
        app.run(debug=True)
    else:
        print("Failed to start application due to initialization errors.")