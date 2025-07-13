import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

class WinProbabilityPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, filepath='data/horse_racing_data_1000.csv'):
        """Load the CSV data"""
        try:
            df = pd.read_csv(filepath)
            print(f"SUCCESS: Loaded {len(df)} records from {filepath}")
            print(f"Data shape: {df.shape}")
            print(f"Date range: {df['race_date'].min()} to {df['race_date'].max()}")
            return df
        except Exception as e:
            print(f"ERROR loading data: {e}")
            print("Make sure you've run data_generator.py first!")
            return None
    
    def explore_data(self, df):
        """Quick data exploration focused on WIN ANALYSIS"""
        print("\n=== WIN PROBABILITY DATA EXPLORATION ===")
        print("=" * 50)
        
        print("Dataset overview:")
        print(f"Total horses: {len(df)}")
        print(f"Unique races: {len(df.groupby(['race_date', 'track_name', 'race_number']))}")
        print(f"Unique tracks: {df['track_name'].nunique()}")
        print(f"Unique jockeys: {df['jockey'].nunique()}")
        print(f"Unique trainers: {df['trainer'].nunique()}")
        print(f"Average field size: {df.groupby(['race_date', 'track_name', 'race_number']).size().mean():.1f}")
        
        print("\nFinish position distribution:")
        finish_dist = df['finish_position'].value_counts().sort_index()
        for pos, count in finish_dist.head(8).items():
            print(f"  {pos}: {count} horses ({count/len(df)*100:.1f}%)")
        
        print(f"\nMissing values:")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("  No missing values found!")
        
        return df
    
    def add_win_targets(self, df):
        """
        Create win/place/show targets for classification
        """
        print("\n=== CREATING WIN PROBABILITY TARGETS ===")
        print("=" * 50)
        
        df_targets = df.copy()
        
        # Create binary targets for machine learning
        df_targets['won'] = (df_targets['finish_position'] == 1).astype(int)
        df_targets['placed'] = (df_targets['finish_position'] <= 2).astype(int) 
        df_targets['showed'] = (df_targets['finish_position'] <= 3).astype(int)
        
        # Add field size (crucial for win probability)
        df_targets['field_size'] = df_targets.groupby(['race_date', 'track_name', 'race_number'])['horse_name'].transform('count')
        
        print(f"SUCCESS: Win rate in data: {df_targets['won'].mean()*100:.1f}%")
        print(f"SUCCESS: Place rate in data: {df_targets['placed'].mean()*100:.1f}%") 
        print(f"SUCCESS: Show rate in data: {df_targets['showed'].mean()*100:.1f}%")
        print(f"SUCCESS: Average field size: {df_targets['field_size'].mean():.1f}")
        
        # Validate realistic win rates
        expected_win_rate = 1.0 / df_targets['field_size'].mean()
        actual_win_rate = df_targets['won'].mean()
        print(f"Expected win rate (1/field_size): {expected_win_rate:.1%}")
        print(f"Actual win rate: {actual_win_rate:.1%}")
        
        return df_targets
    
    def create_performance_features(self, df):
        """
        Create jockey/trainer performance features with proper statistical handling
        """
        print("\n=== CREATING PERFORMANCE FEATURES ===")
        print("=" * 50)
        
        df_perf = df.copy()
        
        # Jockey performance with minimum ride requirements
        jockey_stats = df_perf.groupby('jockey').agg({
            'won': ['mean', 'sum', 'count'],
            'placed': 'mean',
            'showed': 'mean'
        }).round(3)
        
        # Flatten column names
        jockey_stats.columns = ['jockey_win_rate', 'jockey_wins', 'jockey_rides', 
                               'jockey_place_rate', 'jockey_show_rate']
        
        # Only use jockey stats for jockeys with 10+ rides (statistical significance)
        min_rides = 10
        jockey_stats.loc[jockey_stats['jockey_rides'] < min_rides, 'jockey_win_rate'] = df_perf['won'].mean()
        jockey_stats.loc[jockey_stats['jockey_rides'] < min_rides, 'jockey_place_rate'] = df_perf['placed'].mean()
        jockey_stats.loc[jockey_stats['jockey_rides'] < min_rides, 'jockey_show_rate'] = df_perf['showed'].mean()
        
        # Merge back to main dataframe
        df_perf = df_perf.merge(jockey_stats, left_on='jockey', right_index=True, how='left')
        
        # Trainer performance with minimum starts requirements
        trainer_stats = df_perf.groupby('trainer').agg({
            'won': ['mean', 'sum', 'count'],
            'placed': 'mean',
            'showed': 'mean'
        }).round(3)
        
        trainer_stats.columns = ['trainer_win_rate', 'trainer_wins', 'trainer_starts',
                                'trainer_place_rate', 'trainer_show_rate']
        
        # Only use trainer stats for trainers with 15+ starts
        min_starts = 15
        trainer_stats.loc[trainer_stats['trainer_starts'] < min_starts, 'trainer_win_rate'] = df_perf['won'].mean()
        trainer_stats.loc[trainer_stats['trainer_starts'] < min_starts, 'trainer_place_rate'] = df_perf['placed'].mean()
        trainer_stats.loc[trainer_stats['trainer_starts'] < min_starts, 'trainer_show_rate'] = df_perf['showed'].mean()
        
        df_perf = df_perf.merge(trainer_stats, left_on='trainer', right_index=True, how='left')
        
        # Track/surface specific performance
        track_surface_stats = df_perf.groupby(['track_name', 'surface']).agg({
            'won': 'mean',
            'horse_name': 'count'
        }).round(3)
        track_surface_stats.columns = ['track_surface_win_rate', 'track_surface_races']
        
        # Only use track/surface stats with 20+ races
        min_track_races = 20
        track_surface_stats.loc[track_surface_stats['track_surface_races'] < min_track_races, 'track_surface_win_rate'] = df_perf['won'].mean()
        
        track_surface_stats = track_surface_stats[['track_surface_win_rate']].reset_index()
        df_perf = df_perf.merge(track_surface_stats, on=['track_name', 'surface'], how='left')
        
        print(f"SUCCESS: Added jockey performance features")
        print(f"  Jockeys with 10+ rides: {(jockey_stats['jockey_rides'] >= min_rides).sum()}")
        print(f"SUCCESS: Added trainer performance features") 
        print(f"  Trainers with 15+ starts: {(trainer_stats['trainer_starts'] >= min_starts).sum()}")
        print(f"SUCCESS: Added track/surface specific features")
        
        return df_perf
    
    def create_odds_features(self, df):
        """
        Create advanced odds-based features
        """
        print("\n=== CREATING ODDS FEATURES ===")
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
        
        # Normalize implied probabilities to sum to 1 (remove track takeout)
        race_totals = df_odds.groupby(['race_date', 'track_name', 'race_number'])['implied_probability'].transform('sum')
        df_odds['true_probability'] = df_odds['implied_probability'] / race_totals
        
        # Odds categories
        df_odds['odds_category'] = pd.cut(df_odds['odds'], 
                                        bins=[0, 3, 6, 10, float('inf')], 
                                        labels=['favorite', 'second_choice', 'medium_odds', 'longshot'])
        
        print(f"SUCCESS: Created favorite indicator")
        print(f"SUCCESS: Created odds ranking and probabilities")
        print(f"VALIDATION: Favorite win rate: {df_odds[df_odds['favorite']==1]['won'].mean()*100:.1f}%")
        print(f"VALIDATION: Longshot win rate: {df_odds[df_odds['odds_category']=='longshot']['won'].mean()*100:.1f}%")
        
        return df_odds
    
    def create_post_position_features(self, df):
        """
        Post position analysis with statistical validation
        """
        print("\n=== CREATING POST POSITION FEATURES ===")
        print("=" * 50)
        
        df_post = df.copy()
        
        # Inside post advantage (posts 1-3)
        df_post['inside_post'] = (df_post['post_position'] <= 3).astype(int)
        df_post['outside_post'] = (df_post['post_position'] >= 8).astype(int)
        
        # Post position win rate by track (with minimum sample sizes)
        post_track_stats = df_post.groupby(['track_name', 'post_position']).agg({
            'won': ['mean', 'count']
        }).round(3)
        post_track_stats.columns = ['post_win_rate', 'post_count']
        
        # Only use post position stats with 10+ races
        min_post_races = 10
        post_track_stats.loc[post_track_stats['post_count'] < min_post_races, 'post_win_rate'] = df_post['won'].mean()
        
        post_track_stats = post_track_stats[['post_win_rate']].reset_index()
        df_post = df_post.merge(post_track_stats, on=['track_name', 'post_position'], how='left')
        
        print(f"SUCCESS: Created inside/outside post indicators")
        print(f"VALIDATION: Inside post win rate: {df_post[df_post['inside_post']==1]['won'].mean()*100:.1f}%")
        print(f"VALIDATION: Outside post win rate: {df_post[df_post['outside_post']==1]['won'].mean()*100:.1f}%")
        print(f"VALIDATION: Overall win rate: {df_post['won'].mean()*100:.1f}%")
        
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
        """Clean and prepare the data for win prediction"""
        print("\n=== CLEANING DATA FOR WIN PREDICTION ===")
        print("=" * 40)
        
        df_clean = df.copy()
        
        # Convert time to seconds
        if 'final_time' in df_clean.columns:
            df_clean['final_time_seconds'] = df_clean['final_time'].apply(self.convert_time_to_seconds)
            print(f"SUCCESS: Converted final_time to seconds")
        
        # Convert distance to furlongs
        if 'distance' in df_clean.columns:
            df_clean['distance_furlongs'] = df_clean['distance'].apply(self.convert_distance_to_furlongs)
            print(f"SUCCESS: Converted distance to furlongs")
        
        # Clean odds
        if 'odds' in df_clean.columns:
            df_clean['odds'] = pd.to_numeric(df_clean['odds'], errors='coerce')
            print(f"SUCCESS: Cleaned odds column")
        
        # Create speed per furlong
        if 'final_time_seconds' in df_clean.columns and 'distance_furlongs' in df_clean.columns:
            df_clean['speed_per_furlong'] = df_clean['final_time_seconds'] / df_clean['distance_furlongs']
            print(f"SUCCESS: Created speed per furlong feature")
        
        # Handle missing values
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                print(f"SUCCESS: Filled missing values in {col}")
        
        print(f"Final data shape: {df_clean.shape}")
        return df_clean
    
    def encode_categorical_features(self, df):
        """Convert categorical variables to numbers"""
        print("\n=== ENCODING CATEGORICAL FEATURES ===")
        print("=" * 40)
        
        df_encoded = df.copy()
        
        categorical_cols = ['track_name', 'surface', 'track_condition', 'jockey', 'trainer', 'odds_category']
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                
                df_encoded[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                print(f"SUCCESS: Encoded {col} ({df_encoded[col].nunique()} unique values)")
        
        return df_encoded
    
    def prepare_for_win_prediction(self, df, target='won'):
        """
        Prepare final dataset for WIN PROBABILITY prediction
        """
        print(f"\n=== PREPARING FOR WIN PREDICTION (Target: {target}) ===")
        print("=" * 50)
        
        # Select features that matter for WIN prediction
        feature_columns = [
            # Basic race info
            'distance_furlongs', 'post_position', 'weight', 'field_size',
            
            # Odds features (VERY important for win prediction)
            'odds', 'log_odds', 'favorite', 'odds_rank', 'implied_probability', 'true_probability',
            
            # Performance features (the secret sauce)
            'jockey_win_rate', 'jockey_place_rate', 'jockey_rides',
            'trainer_win_rate', 'trainer_place_rate', 'trainer_starts',
            'track_surface_win_rate',
            
            # Post position features
            'inside_post', 'outside_post', 'post_win_rate',
            
            # Speed features
            'speed_per_furlong', 'speed_rating',
            
            # Encoded categorical features
            'surface_encoded', 'track_condition_encoded', 'odds_category_encoded'
        ]
        
        # Only use columns that exist
        available_features = [col for col in feature_columns if col in df.columns]
        
        if target not in df.columns:
            print(f"ERROR: Target column '{target}' not found!")
            return None, None
        
        # Create feature matrix (X) and target (y)
        X = df[available_features]
        y = df[target]
        
        # Fill any remaining missing values
        X = X.fillna(X.median())
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        print(f"SUCCESS: Features prepared: {len(available_features)} features")
        print(f"SUCCESS: Target prepared: {target}")
        print(f"SUCCESS: Win rate in target: {y.mean()*100:.1f}%")
        print(f"Final shape: X={X_scaled.shape}, y={y.shape}")
        
        # Show feature summary
        print(f"\nFEATURE CATEGORIES:")
        odds_features = [f for f in available_features if 'odds' in f.lower() or 'favorite' in f.lower() or 'probability' in f.lower()]
        performance_features = [f for f in available_features if 'win_rate' in f or 'place_rate' in f]
        position_features = [f for f in available_features if 'post' in f.lower()]
        print(f"  Odds-related: {len(odds_features)} features")
        print(f"  Performance: {len(performance_features)} features") 
        print(f"  Position: {len(position_features)} features")
        print(f"  Other: {len(available_features) - len(odds_features) - len(performance_features) - len(position_features)} features")
        
        return X_scaled, y
    
    def save_processed_data(self, X, y, filename_prefix="win_prediction"):
        """Save processed data for win prediction"""
        os.makedirs('data', exist_ok=True)
        
        X_path = f"data/{filename_prefix}_X.csv"
        y_path = f"data/{filename_prefix}_y.csv"
        
        X.to_csv(X_path, index=False)
        y.to_csv(y_path, index=False)
        
        print(f"SUCCESS: Saved win prediction data:")
        print(f"   Features: {X_path}")
        print(f"   Target: {y_path}")
        
        return X_path, y_path

# Main execution
if __name__ == "__main__":
    print("HORSE RACING WIN PROBABILITY PREPROCESSOR - 1000+ HORSES")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = WinProbabilityPreprocessor()
    
    # Load data (tries multiple file names)
    data_files = ['data/horse_racing_data_1000.csv', 'data/sample_race_data.csv']
    df = None
    
    for filepath in data_files:
        if os.path.exists(filepath):
            df = preprocessor.load_data(filepath)
            break
    
    if df is None:
        print("ERROR: No data file found!")
        print("Run data_generator.py first to generate 1000+ horses")
        exit()
    
    if df is not None:
        # Explore data
        df = preprocessor.explore_data(df)
        
        # Add win targets
        df_targets = preprocessor.add_win_targets(df)
        
        # Clean data
        df_clean = preprocessor.clean_data(df_targets)
        
        # Create performance features
        df_performance = preprocessor.create_performance_features(df_clean)
        
        # Create odds features
        df_odds = preprocessor.create_odds_features(df_performance)
        
        # Create post position features
        df_post = preprocessor.create_post_position_features(df_odds)
        
        # Encode categorical features
        df_encoded = preprocessor.encode_categorical_features(df_post)
        
        # Prepare for WIN prediction
        X, y = preprocessor.prepare_for_win_prediction(df_encoded, target='won')
        
        if X is not None and y is not None:
            # Save processed data
            preprocessor.save_processed_data(X, y)
            
            print("\nSUCCESS! 1000+ horses ready for WIN PREDICTION!")
            print("Next step: Run train_model.py to build your enhanced model")
        else:
            print("ERROR: Failed to prepare data for ML")
    else:
        print("ERROR: Failed to load data")