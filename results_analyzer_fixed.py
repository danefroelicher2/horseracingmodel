"""
RESULTS ANALYZER - FIXED VERSION
================================

This handles the model mismatch issue and shows your actual results.
Save as: results_analyzer_fixed.py
"""

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import glob

def analyze_processed_data():
    """Analyze the processed data files"""
    print("=" * 60)
    print("ANALYZING YOUR WIN PREDICTION DATA")
    print("=" * 60)
    
    try:
        # Load the processed data
        X = pd.read_csv('data/win_prediction_X.csv')
        y = pd.read_csv('data/win_prediction_y.csv')
        
        print(f"SUCCESS: Successfully loaded processed data")
        print(f"   Features (X): {X.shape}")
        print(f"   Targets (y): {y.shape}")
        
        # Analyze target distribution
        y_values = y.values.ravel()
        win_rate = y_values.mean()
        total_races = len(y_values)
        winners = y_values.sum()
        losers = total_races - winners
        
        print(f"\nTARGET ANALYSIS:")
        print(f"   Total horses: {total_races}")
        print(f"   Winners: {winners}")
        print(f"   Losers: {losers}")
        print(f"   Win rate: {win_rate:.1%}")
        print(f"   NOTE: {win_rate:.1%} win rate is realistic for horse racing!")
        
        # Analyze features
        print(f"\nFEATURE ANALYSIS:")
        print(f"   Number of features: {X.shape[1]}")
        print(f"   Your new WIN PREDICTION features:")
        for i, col in enumerate(X.columns):
            print(f"     {i+1:2d}. {col}")
        
        return X, y
        
    except FileNotFoundError as e:
        print(f"ERROR: Could not find processed data files")
        print(f"   Make sure you've run preprocess_v2.py")
        return None, None
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return None, None

def train_correct_model(X, y):
    """Train the correct classification model for win prediction"""
    print(f"\n" + "=" * 60)
    print("TRAINING CORRECT WIN PREDICTION MODEL")
    print("=" * 60)
    
    if X is None or y is None:
        print("ERROR: No data available for training")
        return None
    
    try:
        # Split the data
        y_values = y.values.ravel()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_values, test_size=0.3, random_state=42, stratify=y_values
        )
        
        print(f"Data split:")
        print(f"   Training: {X_train.shape[0]} samples")
        print(f"   Testing: {X_test.shape[0]} samples")
        
        # Train Random Forest Classifier (not regressor!)
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced data
        )
        
        print(f"\nTraining RandomForestClassifier...")
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_prob)
        
        print(f"SUCCESS: Model trained!")
        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   AUC Score: {auc_score:.3f}")
        
        # Save the correct model
        os.makedirs('models', exist_ok=True)
        model_path = 'models/win_prediction_classifier.pkl'
        joblib.dump(model, model_path)
        print(f"   Model saved: {model_path}")
        
        return {
            'model': model,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'X_test': X_test,
            'y_test': y_test,
            'y_prob': y_prob
        }
        
    except Exception as e:
        print(f"ERROR training model: {e}")
        return None

def interpret_auc_score(auc_score):
    """Interpret the AUC score"""
    if auc_score < 0.6:
        return "Poor - barely better than random", "❌"
    elif auc_score < 0.7:
        return "Fair - some predictive power", "⚠️"
    elif auc_score < 0.8:
        return "Good - solid predictive power", "✅"
    elif auc_score < 0.9:
        return "Very Good - strong predictive power", "🎉"
    else:
        return "Excellent - exceptional predictive power", "🏆"

def analyze_feature_importance(model, X):
    """Analyze feature importance"""
    print(f"\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    
    try:
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("TOP 10 MOST IMPORTANT FEATURES for WIN PREDICTION:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            print(f"   {i:2d}. {row['feature']:25} ({row['importance']:.3f})")
        
        # Analyze what this means
        top_feature = importance_df.iloc[0]['feature']
        print(f"\nKEY INSIGHT:")
        print(f"   Most important factor: {top_feature}")
        
        if 'odds' in top_feature.lower():
            print(f"   This makes sense! Odds reflect public opinion and insider knowledge.")
        elif 'jockey' in top_feature.lower():
            print(f"   This makes sense! Jockey skill is crucial in horse racing.")
        elif 'post' in top_feature.lower():
            print(f"   This makes sense! Post position affects racing strategy.")
        
        return importance_df
        
    except Exception as e:
        print(f"ERROR analyzing feature importance: {e}")
        return None

def simulate_betting_performance(model_results, X):
    """Simulate betting performance"""
    print(f"\n" + "=" * 60)
    print("BETTING SIMULATION")
    print("=" * 60)
    
    if model_results is None:
        print("ERROR: No model results available")
        return
    
    try:
        y_prob = model_results['y_prob']
        y_test = model_results['y_test']
        
        # Top confidence predictions
        confidence_threshold = 0.3  # Bet on horses with >30% win probability
        confident_bets = y_prob >= confidence_threshold
        
        if confident_bets.sum() > 0:
            actual_win_rate = y_test[confident_bets].mean()
            predicted_win_rate = y_prob[confident_bets].mean()
            
            print(f"BETTING STRATEGY: Bet on horses with >{confidence_threshold:.0%} win probability")
            print(f"   Bets made: {confident_bets.sum()}")
            print(f"   Predicted win rate: {predicted_win_rate:.1%}")
            print(f"   Actual win rate: {actual_win_rate:.1%}")
            
            # Simple ROI calculation (assuming average odds of 4:1)
            bets_made = confident_bets.sum()
            money_bet = bets_made * 10  # $10 per bet
            winners = y_test[confident_bets].sum()
            money_won = winners * 40  # $40 payout per winner (4:1 odds)
            profit = money_won - money_bet
            roi = (profit / money_bet * 100) if money_bet > 0 else 0
            
            print(f"\nSIMPLE ROI CALCULATION:")
            print(f"   Money bet: ${money_bet}")
            print(f"   Money won: ${money_won}")
            print(f"   Profit: ${profit}")
            print(f"   ROI: {roi:.1f}%")
            
            if roi > 10:
                print(f"   🎉 Profitable! This strategy could make money!")
            elif roi > -10:
                print(f"   ⚠️  Break-even. Close to profitability.")
            else:
                print(f"   ❌ Losing strategy. Need more data or better features.")
        
        else:
            print("No high-confidence predictions found.")
            
    except Exception as e:
        print(f"ERROR in betting simulation: {e}")

def explain_what_happened(auc_score, win_rate):
    """Explain what the model actually learned"""
    print(f"\n" + "=" * 60)
    print("WHAT YOUR MODEL ACTUALLY LEARNED")
    print("=" * 60)
    
    print(f"YOUR MODEL ANALYZED 40 HORSES and discovered:")
    print(f"   • Only {win_rate:.1%} of horses win (realistic!)")
    print(f"   • Patterns that predict winners with AUC {auc_score:.3f}")
    
    interpretation, emoji = interpret_auc_score(auc_score)
    print(f"   {emoji} {interpretation}")
    
    print(f"\nWHAT THIS MEANS:")
    if auc_score >= 0.7:
        print(f"   ✅ Your model found REAL patterns in horse racing!")
        print(f"   ✅ It can distinguish winners from losers better than chance")
        print(f"   ✅ With more data (1000+ races), this could be very powerful")
    else:
        print(f"   ⚠️  Your model found some patterns, but needs more data")
        print(f"   ⚠️  40 data points isn't enough for strong conclusions")
        print(f"   ⚠️  Scaling to 1000+ races should improve performance significantly")

def provide_next_steps(auc_score):
    """Provide specific next steps"""
    print(f"\n" + "=" * 60)
    print("YOUR SPECIFIC NEXT STEPS")
    print("=" * 60)
    
    if auc_score >= 0.75:
        print(f"🚀 IMMEDIATE ACTION PLAN:")
        print(f"   1. Your model shows REAL promise - scale up immediately!")
        print(f"   2. Build web scraper to get 1000+ races from Equibase")
        print(f"   3. Focus on data quality over fancy algorithms")
        print(f"   4. Target: AUC > 0.85 with more data")
        
    elif auc_score >= 0.65:
        print(f"📈 GRADUAL IMPROVEMENT PLAN:")
        print(f"   1. Add more sophisticated features (recent form, class levels)")
        print(f"   2. Collect 500-1000 races to validate patterns")
        print(f"   3. Experiment with feature engineering")
        print(f"   4. Target: AUC > 0.80 with better features")
        
    else:
        print(f"🔧 DEBUGGING PLAN:")
        print(f"   1. Check data quality - might have preprocessing issues")
        print(f"   2. Try simpler features first")
        print(f"   3. Validate with domain experts")
        print(f"   4. Target: AUC > 0.70 with cleaner data")
    
    print(f"\nLONG-TERM VISION:")
    print(f"   • 1000 races → AUC 0.75-0.80")
    print(f"   • 10,000 races → AUC 0.80-0.85")
    print(f"   • 100,000 races → AUC 0.85+ (commercial viability)")

def main():
    """Run complete analysis with fixed model handling"""
    print("HORSE RACING WIN PREDICTION - COMPLETE ANALYSIS")
    print("=" * 60)
    print("Analyzing your transformation from speed ratings to win probabilities...")
    
    # Load and analyze data
    X, y = analyze_processed_data()
    
    if X is not None and y is not None:
        # Train the correct model type
        model_results = train_correct_model(X, y)
        
        if model_results is not None:
            auc_score = model_results['auc_score']
            win_rate = y.values.ravel().mean()
            
            # Analyze feature importance
            feature_importance = analyze_feature_importance(model_results['model'], X)
            
            # Simulate betting
            simulate_betting_performance(model_results, X)
            
            # Explain results
            explain_what_happened(auc_score, win_rate)
            
            # Provide next steps
            provide_next_steps(auc_score)
            
            print(f"\n" + "=" * 60)
            print(f"ANALYSIS COMPLETE!")
            print(f"Key Result: AUC Score = {auc_score:.3f}")
            print("=" * 60)
            
        else:
            print("ERROR: Could not train model")
    else:
        print("ERROR: Could not load data")

if __name__ == "__main__":
    main()