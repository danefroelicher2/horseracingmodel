import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

class HorseRacePreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, filepath):
        """Load the CSV data"""
        try:
            df = pd.read_csv(filepath)
            print(f"✅ Loaded {len(df)} records from {filepath}")
            print(f"📊 Data shape: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def explore_data(self, df):
        """Quick data exploration"""
        print("\n📈 DATA EXPLORATION")
        print("=" * 40)
        
        print("Column info:")
        print(df.info())
        
        print("\nFirst few rows:")
        print(df.head())
        
        print("\nMissing values:")
        print(df.isnull().sum())
        
        print("\nUnique values in key columns:")
        categorical_cols = ['track_name', 'surface', 'track_condition', 'jockey', 'trainer']
        for col in categorical_cols:
            if col in df.columns:
                print(f"{col}: {df[col].nunique()} unique values")
        
        return df
    
    def convert_time_to_seconds(self, time_str):
        """Convert time format '1:10.23' to total seconds"""
        try:
            if ':' in str(time_str):
                parts = str(time_str).split(':')
                minutes = int(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            else:
                return float(time_str)
        except:
            return np.nan
    
    def convert_distance_to_furlongs(self, distance_str):
        """Convert distance to furlongs (standard unit)"""
        try:
            dist_str = str(distance_str).lower()
            if 'f' in dist_str:
                return float(dist_str.replace('f', ''))
            elif 'm' in dist_str:
                miles = float(dist_str.replace('m', ''))
                return miles * 8  # 1 mile = 8 furlongs
            else:
                return float(dist_str)
        except:
            return np.nan
    
    def clean_data(self, df):
        """Clean and prepare the data"""
        print("\n🧹 CLEANING DATA")
        print("=" * 40)
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Convert time to seconds
        if 'final_time' in df_clean.columns:
            df_clean['final_time_seconds'] = df_clean['final_time'].apply(self.convert_time_to_seconds)
            print(f"✅ Converted final_time to seconds")
        
        # Convert distance to furlongs
        if 'distance' in df_clean.columns:
            df_clean['distance_furlongs'] = df_clean['distance'].apply(self.convert_distance_to_furlongs)
            print(f"✅ Converted distance to furlongs")
        
        # Clean odds (remove any non-numeric characters)
        if 'odds' in df_clean.columns:
            df_clean['odds'] = pd.to_numeric(df_clean['odds'], errors='coerce')
            print(f"✅ Cleaned odds column")
        
        # Clean margin (extract numeric value)
        if 'margin' in df_clean.columns:
            df_clean['margin_lengths'] = df_clean['margin'].str.extract(r'(\d+\.?\d*)').astype(float)
            df_clean['margin_lengths'] = df_clean['margin_lengths'].fillna(0)  # Winners get 0
            print(f"✅ Converted margin to lengths")
        
        # Handle missing values
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                print(f"✅ Filled missing values in {col}")
        
        print(f"📊 Clean data shape: {df_clean.shape}")
        return df_clean
    
    def encode_categorical_features(self, df):
        """Convert categorical variables to numbers"""
        print("\n🔢 ENCODING CATEGORICAL FEATURES")
        print("=" * 40)
        
        df_encoded = df.copy()
        
        categorical_cols = ['track_name', 'surface', 'track_condition', 'jockey', 'trainer', 'horse_name']
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                # Create or use existing encoder
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                
                df_encoded[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                print(f"✅ Encoded {col} ({df_encoded[col].nunique()} unique values)")
        
        return df_encoded
    
    def create_features(self, df):
        """Create additional features for ML"""
        print("\n⚙️ CREATING FEATURES")
        print("=" * 40)
        
        df_features = df.copy()
        
        # Speed per furlong (if we have time and distance)
        if 'final_time_seconds' in df_features.columns and 'distance_furlongs' in df_features.columns:
            df_features['speed_per_furlong'] = df_features['final_time_seconds'] / df_features['distance_furlongs']
            print("✅ Created speed_per_furlong feature")
        
        # Odds categories (favorite, longshot, etc.)
        if 'odds' in df_features.columns:
            df_features['odds_category'] = pd.cut(df_features['odds'], 
                                                bins=[0, 3, 6, 10, float('inf')], 
                                                labels=['favorite', 'second_choice', 'medium_odds', 'longshot'])
            df_features['odds_category_encoded'] = LabelEncoder().fit_transform(df_features['odds_category'])
            print("✅ Created odds_category feature")
        
        # Post position advantage (inside posts often have advantage)
        if 'post_position' in df_features.columns:
            df_features['inside_post'] = (df_features['post_position'] <= 3).astype(int)
            print("✅ Created inside_post feature")
        
        return df_features
    
    def prepare_for_ml(self, df, target_column='speed_rating'):
        """Prepare final dataset for machine learning"""
        print(f"\n🎯 PREPARING FOR ML (Target: {target_column})")
        print("=" * 40)
        
        # Select features for ML
        feature_columns = [
            'distance_furlongs', 'post_position', 'weight', 'odds',
            'track_condition_encoded', 'surface_encoded', 'jockey_encoded',
            'speed_per_furlong', 'odds_category_encoded', 'inside_post'
        ]
        
        # Only use columns that exist
        available_features = [col for col in feature_columns if col in df.columns]
        
        if target_column not in df.columns:
            print(f"❌ Target column '{target_column}' not found!")
            return None, None
        
        # Create feature matrix (X) and target (y)
        X = df[available_features]
        y = df[target_column]
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        print(f"✅ Features prepared: {list(X_scaled.columns)}")
        print(f"✅ Target prepared: {target_column}")
        print(f"📊 Final shape: X={X_scaled.shape}, y={y.shape}")
        
        return X_scaled, y
    
    def save_processed_data(self, X, y, filename_prefix="processed"):
        """Save processed data"""
        os.makedirs('data', exist_ok=True)
        
        X_path = f"data/{filename_prefix}_X.csv"
        y_path = f"data/{filename_prefix}_y.csv"
        
        X.to_csv(X_path, index=False)
        y.to_csv(y_path, index=False)
        
        print(f"✅ Saved processed data:")
        print(f"   Features: {X_path}")
        print(f"   Target: {y_path}")
        
        return X_path, y_path

# Main execution
if __name__ == "__main__":
    print("🏇 HORSE RACING DATA PREPROCESSOR")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = HorseRacePreprocessor()
    
    # Load data
    df = preprocessor.load_data('data/sample_race_data.csv')
    
    if df is not None:
        # Explore data
        df = preprocessor.explore_data(df)
        
        # Clean data
        df_clean = preprocessor.clean_data(df)
        
        # Encode categorical features
        df_encoded = preprocessor.encode_categorical_features(df_clean)
        
        # Create additional features
        df_features = preprocessor.create_features(df_encoded)
        
        # Prepare for ML
        X, y = preprocessor.prepare_for_ml(df_features, target_column='speed_rating')
        
        if X is not None and y is not None:
            # Save processed data
            preprocessor.save_processed_data(X, y)
            
            print("\n🎉 SUCCESS! Data is ready for machine learning!")
            print("\nNext step: Run train_model.py to build your first ML model")
        else:
            print("❌ Failed to prepare data for ML")
    else:
        print("❌ Failed to load data")