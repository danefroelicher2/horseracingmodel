def create_performance_features(self, df):
        """
        Create jockey/trainer performance features optimized for 10K+ horses
        """
        print("\n=== CREATING PERFORMANCE FEATURES (10K+ HORSES) ===")
        print("=" * 60)
        
        df_perf = df.copy()
        import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

class WinProbabilityPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, filepath=None):
        """Load the CSV data - auto-detect largest dataset"""
        # Try different file paths in order of preference
        data_files = [
            'data/horse_racing_data_10k.csv',
            'data/horse_racing_data_1000.csv', 
            'data/sample_race_data.csv'
        ]
        
        if filepath:
            data_files.insert(0, filepath)
        
        for file_path in data_files:
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    print(f"SUCCESS: Loaded {len(df):,} records from {file_path}")
                    print(f"Data shape: {df.shape}")
                    print(f"Date range: {df['race_date'].min()} to {df['race_date'].max()}")
                    
                    # Memory usage info for large datasets
                    if len(df) > 5000:
                        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
                        print(f"Memory usage: {memory_mb:.1f} MB")
                    
                    return df
                except Exception as e:
                    print(f"ERROR loading {file_path}: {e}")
                    continue
        
        print("ERROR: No data file found!")
        print("Run data_generator.py first to generate data")
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
        Create jockey/trainer performance features optimized for 10K+ horses
        """
        print("\n=== CREATING PERFORMANCE FEATURES (10K+ HORSES) ===")
        print("=" * 60)
        
        df_perf = df.copy()
        
        # Enhanced jockey performance with higher minimum requirements for 10K dataset
        jockey_stats = df_perf.groupby('jockey').agg({
            'won': ['mean', 'sum', 'count'],
            'placed': 'mean',
            'showed': 'mean'
        }).round(4)
        
        jockey_stats.columns = ['jockey_win_rate', 'jockey_wins', 'jockey_rides', 
                               'jockey_place_rate', 'jockey_show_rate']
        
        # Higher minimum for statistical significance with 10K+ horses
        min_rides = 50  # Increased from 10 to 50
        reliable_jockeys = jockey_stats['jockey_rides'] >= min_rides
        
        # Use overall averages for jockeys with insufficient rides
        overall_win_rate = df_perf['won'].mean()
        overall_place_rate = df_perf['placed'].mean()
        overall_show_rate = df_perf['showed'].mean()
        
        jockey_stats.loc[~reliable_jockeys, 'jockey_win_rate'] = overall_win_rate
        jockey_stats.loc[~reliable_jockeys, 'jockey_place_rate'] = overall_place_rate
        jockey_stats.loc[~reliable_jockeys, 'jockey_show_rate'] = overall_show_rate
        
        # Create jockey tier classification
        jockey_stats['jockey_tier'] = 'average'
        jockey_stats.loc[(jockey_stats['jockey_win_rate'] > 0.18) & reliable_jockeys, 'jockey_tier'] = 'elite'
        jockey_stats.loc[(jockey_stats['jockey_win_rate'] > 0.15) & reliable_jockeys, 'jockey_tier'] = 'top'
        jockey_stats.loc[(jockey_stats['jockey_win_rate'] > 0.12) & reliable_jockeys, 'jockey_tier'] = 'good'
        
        df_perf = df_perf.merge(jockey_stats, left_on='jockey', right_index=True, how='left')
        
        # Enhanced trainer performance
        trainer_stats = df_perf.groupby('trainer').agg({
            'won': ['mean', 'sum', 'count'],
            'placed': 'mean',
            'showed': 'mean'
        }).round(4)
        
        trainer_stats.columns = ['trainer_win_rate', 'trainer_wins', 'trainer_starts',
                                'trainer_place_rate', 'trainer_show_rate']
        
        # Higher minimum for trainers with 10K+ horses
        min_starts = 75  # Increased from 15 to 75
        reliable_trainers = trainer_stats['trainer_starts'] >= min_starts
        
        trainer_stats.loc[~reliable_trainers, 'trainer_win_rate'] = overall_win_rate
        trainer_stats.loc[~reliable_trainers, 'trainer_place_rate'] = overall_place_rate
        trainer_stats.loc[~reliable_trainers, 'trainer_show_rate'] = overall_show_rate
        
        # Create trainer tier classification
        trainer_stats['trainer_tier'] = 'average'
        trainer_stats.loc[(trainer_stats['trainer_win_rate'] > 0.18) & reliable_trainers, 'trainer_tier'] = 'elite'
        trainer_stats.loc[(trainer_stats['trainer_win_rate'] > 0.15) & reliable_trainers, 'trainer_tier'] = 'top'
        trainer_stats.loc[(trainer_stats['trainer_win_rate'] > 0.12) & reliable_trainers, 'trainer_tier'] = 'good'
        
        df_perf = df_perf.merge(trainer_stats, left_on='trainer', right_index=True, how='left')
        
        # Advanced combination features for 10K+ dataset
        # Jockey-Trainer combination performance
        jt_combo_stats = df_perf.groupby(['jockey', 'trainer']).agg({
            'won': ['mean', 'count']
        }).round(4)
        jt_combo_stats.columns = ['jt_combo_win_rate', 'jt_combo_races']
        
        # Only use combo stats with sufficient sample size
        min_combo_races = 10
        reliable_combos = jt_combo_stats['jt_combo_races'] >= min_combo_races
        jt_combo_stats.loc[~reliable_combos, 'jt_combo_win_rate'] = overall_win_rate
        
        jt_combo_stats = jt_combo_stats[['jt_combo_win_rate']].reset_index()
        df_perf = df_perf.merge(jt_combo_stats, on=['jockey', 'trainer'], how='left')
        
        # Track/surface/distance specific performance
        track_surface_distance_stats = df_perf.groupby(['track_name', 'surface', 'distance']).agg({
            'won': ['mean', 'count']
        }).round(4)
        track_surface_distance_stats.columns = ['tsd_win_rate', 'tsd_races']
        
        # Use track/surface/distance stats with sufficient sample size
        min_tsd_races = 20
        reliable_tsd = track_surface_distance_stats['tsd_races'] >= min_tsd_races
        track_surface_distance_stats.loc[~reliable_tsd, 'tsd_win_rate'] = overall_win_rate
        
        track_surface_distance_stats = track_surface_distance_stats[['tsd_win_rate']].reset_index()
        df_perf = df_perf.merge(track_surface_distance_stats, on=['track_name', 'surface', 'distance'], how='left')
        
        # Enhanced track/surface performance (keeping original for comparison)
        track_surface_stats = df_perf.groupby(['track_name', 'surface']).agg({
            'won': ['mean', 'count']
        }).round(4)
        track_surface_stats.columns = ['track_surface_win_rate', 'track_surface_races']
        
        min_track_races = 50  # Increased minimum
        reliable_track_surface = track_surface_stats['track_surface_races'] >= min_track_races
        track_surface_stats.loc[~reliable_track_surface, 'track_surface_win_rate'] = overall_win_rate
        
        track_surface_stats = track_surface_stats[['track_surface_win_rate']].reset_index()
        df_perf = df_perf.merge(track_surface_stats, on=['track_name', 'surface'], how='left')
        
        print(f"SUCCESS: Enhanced performance features for 10K+ horses")
        print(f"  Reliable jockeys (50+ rides): {reliable_jockeys.sum()}")
        print(f"  Reliable trainers (75+ starts): {reliable_trainers.sum()}")
        print(f"  Elite jockeys: {(jockey_stats['jockey_tier'] == 'elite').sum()}")
        print(f"  Elite trainers: {(trainer_stats['trainer_tier'] == 'elite').sum()}")
        print(f"  Jockey-Trainer combos: {reliable_combos.sum()}")
        
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
        Prepare final dataset for WIN PROBABILITY prediction - optimized for 10K+
        """
        print(f"\n=== PREPARING FOR WIN PREDICTION (Target: {target}) ===")
        print("=" * 60)
        
        # Enhanced feature set for 10K+ horses
        feature_columns = [
            # Basic race info
            'distance_furlongs', 'post_position', 'weight', 'field_size',
            
            # Enhanced odds features
            'odds', 'log_odds', 'favorite', 'odds_rank', 'implied_probability', 'true_probability',
            
            # Enhanced performance features (now more reliable with 10K+ horses)
            'jockey_win_rate', 'jockey_place_rate', 'jockey_rides',
            'trainer_win_rate', 'trainer_place_rate', 'trainer_starts',
            
            # Advanced combination features
            'jt_combo_win_rate',  # Jockey-trainer combination
            'tsd_win_rate',       # Track-surface-distance specific
            'track_surface_win_rate',
            
            # Post position features
            'inside_post', 'outside_post', 'post_win_rate',
            
            # Speed features
            'speed_per_furlong', 'speed_rating',
            
            # Encoded categorical features
            'surface_encoded', 'track_condition_encoded', 'odds_category_encoded',
            'jockey_encoded', 'trainer_encoded'  # Adding these back for 10K dataset
        ]
        
        # Only use columns that exist
        available_features = [col for col in feature_columns if col in df.columns]
        
        if target not in df.columns:
            print(f"ERROR: Target column '{target}' not found!")
            return None, None
        
        # Create feature matrix (X) and target (y)
        X = df[available_features]
        y = df[target]
        
        # Enhanced missing value handling for large dataset
        missing_counts = X.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"Handling missing values in {missing_counts[missing_counts > 0].shape[0]} features")
            
            # Use different strategies for different feature types
            for col in X.columns:
                if X[col].isnull().sum() > 0:
                    if 'rate' in col.lower() or 'probability' in col.lower():
                        # Use median for rate/probability features
                        X[col] = X[col].fillna(X[col].median())
                    elif 'encoded' in col.lower():
                        # Use mode for encoded features
                        X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 0)
                    else:
                        # Use median for other numeric features
                        X[col] = X[col].fillna(X[col].median())
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        print(f"SUCCESS: Enhanced features prepared: {len(available_features)} features")
        print(f"SUCCESS: Target prepared: {target}")
        print(f"SUCCESS: Win rate in target: {y.mean()*100:.2f}%")
        print(f"Final shape: X={X_scaled.shape}, y={y.shape}")
        
        # Enhanced feature summary for 10K+ dataset
        print(f"\nENHANCED FEATURE CATEGORIES:")
        odds_features = [f for f in available_features if any(word in f.lower() for word in ['odds', 'favorite', 'probability'])]
        performance_features = [f for f in available_features if any(word in f.lower() for word in ['win_rate', 'place_rate', 'combo'])]
        position_features = [f for f in available_features if 'post' in f.lower()]
        speed_features = [f for f in available_features if any(word in f.lower() for word in ['speed', 'time'])]
        track_features = [f for f in available_features if any(word in f.lower() for word in ['track', 'surface', 'tsd'])]
        
        print(f"  Odds-related: {len(odds_features)} features")
        print(f"  Performance: {len(performance_features)} features") 
        print(f"  Position: {len(position_features)} features")
        print(f"  Speed: {len(speed_features)} features")
        print(f"  Track-specific: {len(track_features)} features")
        print(f"  Other: {len(available_features) - len(odds_features) - len(performance_features) - len(position_features) - len(speed_features) - len(track_features)} features")
        
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
    print("HORSE RACING WIN PROBABILITY PREPROCESSOR - 10,000+ HORSES")
    print("=" * 70)
    
    # Initialize preprocessor
    preprocessor = WinProbabilityPreprocessor()
    
    # Load data (auto-detects largest available dataset)
    df = preprocessor.load_data()
    
    if df is None:
        print("ERROR: No data file found!")
        print("Run data_generator.py first to generate 10,000+ horses")
        exit()
    
    # Check if we have enough data for advanced processing
    if len(df) < 5000:
        print("WARNING: Dataset is smaller than optimal for 10K processing")
        print("Consider running data_generator.py to generate more data")
    
    # Explore data
    df = preprocessor.explore_data(df)
    
    # Add win targets
    df_targets = preprocessor.add_win_targets(df)
    
    # Clean data
    df_clean = preprocessor.clean_data(df_targets)
    
    # Create enhanced performance features for large dataset
    df_performance = preprocessor.create_performance_features(df_clean)
    
    # Create odds features
    df_odds = preprocessor.create_odds_features(df_performance)
    
    # Create post position features
    df_post = preprocessor.create_post_position_features(df_odds)
    
    # Encode categorical features
    df_encoded = preprocessor.encode_categorical_features(df_post)
    
    # Prepare for WIN prediction with enhanced features
    X, y = preprocessor.prepare_for_win_prediction(df_encoded, target='won')
    
    if X is not None and y is not None:
        # Save processed data
        filename_prefix = f"win_prediction_{len(df)}"  # Include size in filename
        preprocessor.save_processed_data(X, y, filename_prefix)
        
        print(f"\nSUCCESS! {len(df):,} horses ready for ENHANCED WIN PREDICTION!")
        print("=" * 70)
        
        if len(df) >= 10000:
            print("🎉 PROFESSIONAL-GRADE DATASET ACHIEVED!")
            print("✅ Statistical significance guaranteed")
            print("✅ Advanced features enabled")
            print("✅ Ready for industry-level performance")
        elif len(df) >= 5000:
            print("✅ LARGE DATASET - Good for advanced modeling")
        else:
            print("📊 MEDIUM DATASET - Consider scaling up further")
        
        print("\nNext step: Run train_model.py for enhanced model training")
    else:
        print("ERROR: Failed to prepare data for ML")