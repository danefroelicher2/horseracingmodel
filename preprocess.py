import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

class WinProbabilityPreprocessor:
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
        """Quick data exploration focused on WIN ANALYSIS"""
        print("\n📈 WIN PROBABILITY DATA EXPLORATION")
        print("=" * 50)
        
        print("Column info:")
        print(df.info())
        
        print("\nFirst few rows:")
        print(df.head())
        
        print("\nFinish position distribution:")
        print(df['finish_position'].value_counts().sort_index())
        
        print("\nMissing values:")
        print(df.isnull().sum())
        
        return df
    
    def add_win_targets(self, df):
        """
        🎯 KEY CHANGE: Create win/place/show targets instead of speed rating
        This is the fundamental shift from regression to classification!
        """
        print("\n🎯 CREATING WIN PROBABILITY TARGETS")
        print("=" * 50)
        
        df_targets = df.copy()
        
        # Create binary targets for machine learning
        df_targets['won'] = (df_targets['finish_position'] == 1).astype(int)
        df_targets['placed'] = (df_targets['finish_position'] <= 2).astype(int) 
        df_targets['showed'] = (df_targets['finish_position'] <= 3).astype(int)
        
        # Add field size (crucial for win probability - harder to win with more horses)
        df_targets['field_size'] = df_targets.groupby(['race_date', 'track_name', 'race_number'])['horse_name'].transform('count')
        
        print(f"✅ Win rate in data: {df_targets['won'].mean()*100:.1f}%")
        print(f"✅ Place rate in data: {df_targets['placed'].mean()*100:.1f}%") 
        print(f"✅ Show rate in data: {df_targets['showed'].mean()*100:.1f}%")
        print(f"✅ Average field size: {df_targets['field_size'].mean():.1f}")
        
        return df_targets
    
    def create_performance_features(self, df):
        """
        ⚡ Create jockey/trainer performance features
        These are MUCH more important for win prediction than for speed ratings!
        """
        print("\n⚡ CREATING PERFORMANCE FEATURES")
        print("=" * 50)
        
        df_perf = df.copy()
        
        # Jockey performance (this is HUGE for win prediction)
        jockey_stats = df_perf.groupby('jockey').agg({
            'won': 'mean',
            'placed': 'mean',
            'showed': 'mean',
            'horse_name': 'count'  # Number of rides
        }).rename(columns={'horse_name': 'jockey_rides'})
        jockey_stats.columns = ['jockey_win_rate', 'jockey_place_rate', 'jockey_show_rate', 'jockey_rides']
        
        # Merge back to main dataframe
        df_perf = df_perf.merge(jockey_stats, left_on='jockey', right_index=True, how='left')
        
        # Trainer performance
        trainer_stats = df_perf.groupby('trainer').agg({
            'won': 'mean',
            'placed': 'mean', 
            'showed': 'mean',
            'horse_name': 'count'
        }).rename(columns={'horse_name': 'trainer_starts'})
        trainer_stats.columns = ['trainer_win_rate', 'trainer_place_rate', 'trainer_show_rate', 'trainer_starts']
        
        df_perf = df_perf.merge(trainer_stats, left_on='trainer', right_index=True, how='left')
        
        # Track/surface specific performance
        track_surface_stats = df_perf.groupby(['track_name', 'surface']).agg({
            'won': 'mean'
        })['won'].reset_index()
        track_surface_stats.columns = ['track_name', 'surface', 'track_surface_win_rate']
        
        df_perf = df_perf.merge(track_surface_stats, on=['track_name', 'surface'], how='left')
        
        print(f"✅ Added jockey performance features")
        print(f"✅ Added trainer performance features") 
        print(f"✅ Added track/surface specific features")
        
        return df_perf
    
    def create_odds_features(self, df):
        """
        💰 Create advanced odds-based features
        These are CRITICAL for finding value bets and win prediction!
        """
        print("\n💰 CREATING ODDS FEATURES")
        print("=" * 50)
        
        df_odds = df.copy()
        
        # Favorite identification (within each race)
        df_odds['favorite'] = df_odds.groupby(['race_date', 'track_name', 'race_number'])['odds'].transform(
            lambda x: (x == x.min()).astype(int)
        )
        
        # Odds ranking within race (1 = lowest odds/favorite)
        df_odds['odds_rank'] = df_odds.groupby(['race_date', 'track_name', 'race_number'])['odds'].rank()
        
        # Log odds (better for ML algorithms)
        df_odds['log_odds'] = np.log(df_odds['odds'])
        
        # Implied probability from odds
        df_odds['implied_probability'] = 1 / df_odds['odds']
        
        # Odds categories
        df_odds['odds_category'] = pd.cut(df_odds['odds'], 
                                        bins=[0, 3, 6, 10, float('inf')], 
                                        labels=['favorite', 'second_choice', 'medium_odds', 'longshot'])
        
        print(f"✅ Created favorite indicator")
        print(f"✅ Created odds ranking")
        print(f"✅ Created implied probability features")
        print(f"✅ Favorite win rate: {df_odds[df_odds['favorite']==1]['won'].mean()*100:.1f}%")
        
        return df_odds
    
    def create_post_position_features(self, df):
        """
        🏃 Post position is CRUCIAL for win probability
        """
        print("\n🏃 CREATING POST POSITION FEATURES")
        print("=" * 50)
        
        df_post = df.copy()
        
        # Inside post advantage (posts 1-3 often have advantage)
        df_post['inside_post'] = (df_post['post_position'] <= 3).astype(int)
        df_post['outside_post'] = (df_post['post_position'] >= 8).astype(int)
        
        # Post position win rate by track
        post_track_stats = df_post.groupby(['track_name', 'post_position']).agg({
            'won': 'mean'
        })['won'].reset_index()
        post_track_stats.columns = ['track_name', 'post_position', 'post_win_rate']
        
        df_post = df_post.merge(post_track_stats, on=['track_name', 'post_position'], how='left')
        
        print(f"✅ Created inside/outside post indicators")
        print(f"✅ Inside post win rate: {df_post[df_post['inside_post']==1]['won'].mean()*100:.1f}%")
        print(f"✅ Outside post win rate: {df_post[df_post['outside_post']==1]['won'].mean()*100:.1f}%")
        
        return df_post
    
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
        """Convert distance to furlongs"""
        try:
            dist_str = str(distance_str).lower()
            if 'f' in dist_str:
                return float(dist_str.replace('f', ''))
            elif 'm' in dist_str:
                miles = float(dist_str.replace('m', ''))
                return miles * 8
            else:
                return float(dist_str)
        except:
            return np.nan
    
    def clean_data(self, df):
        """Clean and prepare the data - enhanced for win prediction"""
        print("\n🧹 CLEANING DATA FOR WIN PREDICTION")
        print("=" * 40)
        
        df_clean = df.copy()
        
        # Convert time to seconds
        if 'final_time' in df_clean.columns:
            df_clean['final_time_seconds'] = df_clean['final_time'].apply(self.convert_time_to_seconds)
            print(f"✅ Converted final_time to seconds")
        
        # Convert distance to furlongs
        if 'distance' in df_clean.columns:
            df_clean['distance_furlongs'] = df_clean['distance'].apply(self.convert_distance_to_furlongs)
            print(f"✅ Converted distance to furlongs")
        
        # Clean odds
        if 'odds' in df_clean.columns:
            df_clean['odds'] = pd.to_numeric(df_clean['odds'], errors='coerce')
            print(f"✅ Cleaned odds column")
        
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
        
        categorical_cols = ['track_name', 'surface', 'track_condition', 'jockey', 'trainer', 'odds_category']
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                
                df_encoded[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                print(f"✅ Encoded {col} ({df_encoded[col].nunique()} unique values)")
        
        return df_encoded
    
    def prepare_for_win_prediction(self, df, target='won'):
        """
        🎯 Prepare final dataset for WIN PROBABILITY prediction
        This completely replaces your speed rating preparation!
        """
        print(f"\n🎯 PREPARING FOR WIN PREDICTION (Target: {target})")
        print("=" * 50)
        
        # Select features that matter for WIN prediction
        feature_columns = [
            # Basic race info
            'distance_furlongs', 'post_position', 'weight', 'field_size',
            
            # Odds features (VERY important for win prediction)
            'odds', 'log_odds', 'favorite', 'odds_rank', 'implied_probability',
            
            # Performance features (the secret sauce)
            'jockey_win_rate', 'jockey_place_rate', 'jockey_rides',
            'trainer_win_rate', 'trainer_place_rate', 'trainer_starts',
            'track_surface_win_rate',
            
            # Post position features
            'inside_post', 'outside_post', 'post_win_rate',
            
            # Encoded categorical features
            'surface_encoded', 'track_condition_encoded', 'odds_category_encoded'
        ]
        
        # Only use columns that exist
        available_features = [col for col in feature_columns if col in df.columns]
        
        if target not in df.columns:
            print(f"❌ Target column '{target}' not found!")
            return None, None
        
        # Create feature matrix (X) and target (y)
        X = df[available_features]
        y = df[target]
        
        # Fill any remaining missing values
        X = X.fillna(0)
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        print(f"✅ Features prepared: {list(X_scaled.columns)}")
        print(f"✅ Target prepared: {target}")
        print(f"✅ Win rate in target: {y.mean()*100:.1f}%")
        print(f"📊 Final shape: X={X_scaled.shape}, y={y.shape}")
        
        return X_scaled, y
    
    def save_processed_data(self, X, y, filename_prefix="win_prediction"):
        """Save processed data for win prediction"""
        os.makedirs('data', exist_ok=True)
        
        X_path = f"data/{filename_prefix}_X.csv"
        y_path = f"data/{filename_prefix}_y.csv"
        
        X.to_csv(X_path, index=False)
        y.to_csv(y_path, index=False)
        
        print(f"✅ Saved win prediction data:")
        print(f"   Features: {X_path}")
        print(f"   Target: {y_path}")
        
        return X_path, y_path

# Main execution
if __name__ == "__main__":
    print("🏇 WIN PROBABILITY PREPROCESSOR")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = WinProbabilityPreprocessor()
    
    # Load data
    df = preprocessor.load_data('data/sample_race_data.csv')
    
    if df is not None:
        # Explore data
        df = preprocessor.explore_data(df)
        
        # Add win targets (KEY CHANGE!)
        df_targets = preprocessor.add_win_targets(df)
        
        # Clean data
        df_clean = preprocessor.clean_data(df_targets)
        
        # Create performance features (NEW!)
        df_performance = preprocessor.create_performance_features(df_clean)
        
        # Create odds features (NEW!)
        df_odds = preprocessor.create_odds_features(df_performance)
        
        # Create post position features (NEW!)
        df_post = preprocessor.create_post_position_features(df_odds)
        
        # Encode categorical features
        df_encoded = preprocessor.encode_categorical_features(df_post)
        
        # Prepare for WIN prediction (not speed rating!)
        X, y = preprocessor.prepare_for_win_prediction(df_encoded, target='won')
        
        if X is not None and y is not None:
            # Save processed data
            preprocessor.save_processed_data(X, y)
            
            print("\n🎉 SUCCESS! Data is ready for WIN PREDICTION!")
            print("\nNext step: Run train_win_model.py to build your WIN PROBABILITY model")
        else:
            print("❌ Failed to prepare data for ML")
    else:
        print("❌ Failed to load data")