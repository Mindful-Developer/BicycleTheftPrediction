import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_features(df):
    df = df.copy()

    # Convert dates to datetime
    df['OCC_DATE'] = pd.to_datetime(df['OCC_DATE'])
    df['REPORT_DATE'] = pd.to_datetime(df['REPORT_DATE'])

    # Create hour bins
    df['OCC_HOUR_BIN'] = pd.cut(df['OCC_HOUR'], 
                               bins=[-1,6,12,18,24],
                               labels=['Night','Morning','Afternoon','Evening']).astype(str)
    
    # Create season feature
    month = pd.to_datetime(df['OCC_DATE']).dt.month
    conditions = [
        (month >= 3) & (month <= 5),
        (month >= 6) & (month <= 8),
        (month >= 9) & (month <= 11)
    ]
    choices = ['Spring', 'Summer', 'Fall']
    df['SEASON'] = np.select(conditions, choices, default='Winter')

    # Create cost categories
    conditions = [
        df['BIKE_COST'].astype(float) <= 500,
        (df['BIKE_COST'].astype(float) > 500) & (df['BIKE_COST'].astype(float) <= 1000),
        (df['BIKE_COST'].astype(float) > 1000) & (df['BIKE_COST'].astype(float) <= 2000),
        (df['BIKE_COST'].astype(float) > 2000) & (df['BIKE_COST'].astype(float) <= 5000)
    ]
    choices = ['Very Low', 'Low', 'Medium', 'High']
    df['COST_CATEGORY'] = np.select(conditions, choices, default='Very High')
    
    # Create cyclical features for day of year
    df['x'] = np.cos(2 * np.pi * df['OCC_DOY'] / 365)
    df['y'] = np.sin(2 * np.pi * df['OCC_DOY'] / 365)
    
    # Create binary features
    df['IS_NIGHT'] = ((df['OCC_HOUR'] >= 20) | (df['OCC_HOUR'] <= 5)).astype(int)
    df['IS_WEEKEND'] = df['OCC_DOW'].isin(['Saturday', 'Sunday']).astype(int)
    
    # Extract date components
    df['OCC_DAY'] = pd.to_datetime(df['OCC_DATE']).dt.day
    df['OCC_YEAR'] = pd.to_datetime(df['OCC_DATE']).dt.year
    df['REPORT_DAY'] = pd.to_datetime(df['REPORT_DATE']).dt.day
    
    # Fill missing numerical values
    df['BIKE_SPEED'] = pd.to_numeric(df['BIKE_SPEED'], errors='coerce').fillna(0)
    df['BIKE_COST'] = pd.to_numeric(df['BIKE_COST'], errors='coerce').fillna(0)

    # Handle categorical features
    categorical_columns = [
        'BIKE_MAKE', 'BIKE_MODEL', 'BIKE_TYPE', 'BIKE_COLOUR',
        'PREMISES_TYPE', 'OCC_DOW', 'LOCATION_TYPE',
        'HOOD_140', 'NEIGHBOURHOOD_140', 'OCC_HOUR_BIN',
        'SEASON', 'COST_CATEGORY'
    ]
    
    # Fill missing categorical values and encode
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Select features in consistent order
    feature_columns = [
        'BIKE_SPEED', 'BIKE_COST', 'OCC_HOUR', 'OCC_DOW',
        'OCC_DOY', 'OCC_DAY', 'OCC_YEAR', 'REPORT_DAY',
        'BIKE_MAKE', 'BIKE_MODEL', 'BIKE_TYPE', 'BIKE_COLOUR',
        'PREMISES_TYPE', 'LOCATION_TYPE', 'HOOD_140',
        'NEIGHBOURHOOD_140', 'OCC_HOUR_BIN', 'SEASON',
        'COST_CATEGORY', 'x', 'y', 'IS_NIGHT', 'IS_WEEKEND'
    ]
    
    # Ensure all columns exist, fill with default values if missing
    for col in feature_columns:
        if col not in df.columns:
            if col in ['BIKE_SPEED', 'BIKE_COST', 'OCC_HOUR', 'OCC_DOY', 'OCC_DAY', 'OCC_YEAR', 'REPORT_DAY', 'x', 'y']:
                df[col] = 0
            elif col in ['IS_NIGHT', 'IS_WEEKEND']:
                df[col] = 0
            else:
                df[col] = 'Unknown'
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
    
    return df[feature_columns]

def prepare_data_for_training(df, target_column='STATUS', test_size=0.2, random_state=42):
    """Prepare data for model training by splitting into features and target."""
    X = preprocess_features(df)
    # Convert STATUS to binary (1 for RECOVERED, 0 for STOLEN)
    y = pd.Series(0, index=df.index)  # Default to STOLEN (0)
    if target_column in df.columns:
        y = (df[target_column].str.upper() == 'RECOVERED').astype(int)
    return X, y
