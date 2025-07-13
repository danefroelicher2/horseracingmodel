"""
RESULTS ANALYZER
================

This script analyzes your win prediction results and shows you 
the key metrics that matter for your horse racing ML system.

Save as: results_analyzer.py
"""

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, classification_report
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
        
        print(f"✅ Successfully loaded processed data")
        print(f"   Features (X): {X.shape}")
        print(f"   Targets (y): {y.shape}")
        
        # Analyze target distribution
        y_values = y.values.ravel()
        win_rate = y_values.mean()
        total_races = len(y_values)
        winners = y_values.sum()
        losers = total_races - winners
        
        print(f"\n📊 TARGET ANALYSIS:")
        print(f"   Total horses: {total_races}")
        print(f"   Winners: {winners}")
        print(f"   Losers: {losers}")
        print(f"   Win rate: {win_rate:.1%}")
        
        # Analyze features
        print(f"\n🔧 FEATURE ANALYSIS:")
        print(f"   Number of features: {X.shape[1]}")
        print(f"   Feature names:")
        for i, col in enumerate(X.columns):
            print(f"     {i+1:2d}. {col}")
        
        # Check for missing values
        missing_features = X.isnull().sum()
        if missing_features.sum() > 0:
            print(f"\n⚠️  MISSING VALUES FOUND:")
            for col, missing in missing_features.items():
                if missing > 0:
                    print(f"     {col}: {missing} missing")
        else:
            print(f"\n✅ No missing values in features")
        
        return X, y
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find processed data files")
        print(f"   Make sure you've run the preprocessing step")
        return None, None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None

def find_and_analyze_model():
    """Find and analyze the trained model"""
    print(f"\n" + "=" * 60)
    print("ANALYZING YOUR TRAINED MODEL")
    print("=" * 60)
    
    # Look for model files
    model_files = glob.glob('models/*.pkl')
    
    if not model_files:
        print("❌ No trained models found in models/ directory")
        return None
    
    # Use the most recent model
    latest_model = max(model_files, key=os.path.getctime)
    print(f"📁 Found model: {latest_model}")
    
    try:
        model = joblib.load(latest_model)
        print(f"✅ Successfully loaded model: {type(model).__name__}")
        
        # Analyze model properties
        if hasattr(model, 'n_estimators'):
            print(f"   Estimators: {model.n_estimators}")
        if hasattr(model, 'max_depth'):
            print(f"   Max depth: {model.max_depth}")
        if hasattr(model, 'feature_importances_'):
            print(f"   Has feature importance: Yes")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def evaluate_model_performance(model, X, y):
    """Evaluate the model performance"""
    print(f"\n" + "=" * 60)
    print("MODEL PERFORMANCE EVALUATION")
    print("=" * 60)
    
    if model is None or X is None or y is None:
        print("❌ Cannot evaluate - missing model or data")
        return
    
    try:
        # Make predictions
        y_values = y.values.ravel()
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]  # Probability of winning
        
        # Calculate key metrics
        accuracy = (y_pred == y_values).mean()
        auc_score = roc_auc_score(y_values, y_prob)
        
        print(f"🎯 KEY PERFORMANCE METRICS:")
        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   AUC Score: {auc_score:.3f}")
        
        # Interpret AUC score
        if auc_score < 0.6:
            interpretation = "Poor - barely better than random"
            emoji = "❌"
        elif auc_score < 0.7:
            interpretation = "Fair - some predictive power"
            emoji = "⚠️"
        elif auc_score < 0.8:
            interpretation = "Good - solid predictive power"
            emoji = "✅"
        elif auc_score < 0.9:
            interpretation = "Very Good - strong predictive power"
            emoji = "🎉"
        else:
            interpretation = "Excellent - exceptional (possibly overfitted)"
            emoji = "🏆"
        
        print(f"   {emoji} AUC Interpretation: {interpretation}")
        
        # Analyze predictions by confidence
        print(f"\n📈 PREDICTION CONFIDENCE ANALYSIS:")
        
        # Top 10% most confident predictions
        top_10_threshold = np.percentile(y_prob, 90)
        top_10_mask = y_prob >= top_10_threshold
        top_10_actual_win_rate = y_values[top_10_mask].mean()
        top_10_predicted_win_rate = y_prob[top_10_mask].mean()
        
        print(f"   Top 10% predictions:")
        print(f"     Predicted win rate: {top_10_predicted_win_rate:.1%}")
        print(f"     Actual win rate: {top_10_actual_win_rate:.1%}")
        print(f"     Selections: {top_10_mask.sum()}")
        
        # Betting simulation
        print(f"\n💰 SIMPLE BETTING SIMULATION:")
        print(f"   If you bet $1 on every top 10% prediction:")
        
        bets_made = top_10_mask.sum()
        money_bet = bets_made * 1.0
        # Assume average odds of 5:1 for top picks (conservative)
        average_payout = 5.0
        money_won = y_values[top_10_mask].sum() * average_payout
        profit = money_won - money_bet
        roi = (profit / money_bet * 100) if money_bet > 0 else 0
        
        print(f"     Bets made: {bets_made}")
        print(f"     Money bet: ${money_bet:.0f}")
        print(f"     Money won: ${money_won:.0f}")
        print(f"     Profit: ${profit:.0f}")
        print(f"     ROI: {roi:.1f}%")
        
        return {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'top_10_win_rate': top_10_actual_win_rate,
            'roi': roi
        }
        
    except Exception as e:
        print(f"❌ Error evaluating model: {e}")
        return None

def analyze_feature_importance(model, X):
    """Analyze which features matter most"""
    print(f"\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    
    if model is None or X is None:
        print("❌ Cannot analyze - missing model or data")
        return
    
    try:
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("🔝 TOP 10 MOST IMPORTANT FEATURES:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                print(f"   {i:2d}. {row['feature']:25} ({row['importance']:.3f})")
            
            # Categorize features
            print(f"\n📊 FEATURE CATEGORIES:")
            
            odds_features = [f for f in importance_df['feature'] if 'odds' in f.lower() or 'favorite' in f.lower()]
            performance_features = [f for f in importance_df['feature'] if 'win_rate' in f or 'place_rate' in f]
            position_features = [f for f in importance_df['feature'] if 'post' in f.lower()]
            
            print(f"   Odds-related features: {len(odds_features)}")
            print(f"   Performance features: {len(performance_features)}")
            print(f"   Position features: {len(position_features)}")
            
            return importance_df
            
        else:
            print("❌ Model doesn't support feature importance analysis")
            return None
            
    except Exception as e:
        print(f"❌ Error analyzing feature importance: {e}")
        return None

def check_visualization_files():
    """Check if visualization files were created"""
    print(f"\n" + "=" * 60)
    print("CHECKING VISUALIZATION FILES")
    print("=" * 60)
    
    viz_file = 'plots/win_prediction_analysis.png'
    
    if os.path.exists(viz_file):
        print(f"✅ Visualization file found: {viz_file}")
        print(f"   File size: {os.path.getsize(viz_file)} bytes")
        print(f"   📊 Open this file to see your model's performance charts!")
    else:
        print(f"❌ Visualization file not found: {viz_file}")
        print(f"   The training script may not have completed successfully")

def provide_next_steps(performance_metrics):
    """Provide personalized next steps based on results"""
    print(f"\n" + "=" * 60)
    print("YOUR PERSONALIZED NEXT STEPS")
    print("=" * 60)
    
    if performance_metrics is None:
        print("❌ Cannot provide recommendations without performance metrics")
        return
    
    auc_score = performance_metrics['auc_score']
    roi = performance_metrics['roi']
    
    print(f"Based on your AUC score of {auc_score:.3f}:")
    
    if auc_score >= 0.75:
        print(f"🎉 EXCELLENT START! Your model shows real predictive power.")
        print(f"   Next steps:")
        print(f"   1. Scale to 1,000+ races immediately")
        print(f"   2. Start building web scraper for Equibase")
        print(f"   3. Focus on feature engineering improvements")
        
    elif auc_score >= 0.65:
        print(f"✅ GOOD FOUNDATION! Your model beats random guessing.")
        print(f"   Next steps:")
        print(f"   1. Add more sophisticated features")
        print(f"   2. Collect 500-1000 races to improve performance")
        print(f"   3. Experiment with different model types")
        
    else:
        print(f"⚠️  NEEDS IMPROVEMENT. Model is close to random guessing.")
        print(f"   Next steps:")
        print(f"   1. Debug feature engineering - something may be wrong")
        print(f"   2. Check data quality and preprocessing")
        print(f"   3. Try simpler models first")
    
    if roi > 10:
        print(f"\n💰 PROFIT POTENTIAL: {roi:.1f}% ROI suggests this could be profitable!")
    elif roi > 0:
        print(f"\n💰 BREAK-EVEN: {roi:.1f}% ROI - close to profitability")
    else:
        print(f"\n💰 NEEDS WORK: {roi:.1f}% ROI - not profitable yet")

def main():
    """Run complete results analysis"""
    print("HORSE RACING WIN PREDICTION - RESULTS ANALYSIS")
    print("=" * 60)
    print("Let's see how well your model performed!")
    
    # Analyze processed data
    X, y = analyze_processed_data()
    
    # Find and analyze model
    model = find_and_analyze_model()
    
    # Evaluate performance
    performance_metrics = evaluate_model_performance(model, X, y)
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(model, X)
    
    # Check visualizations
    check_visualization_files()
    
    # Provide next steps
    provide_next_steps(performance_metrics)
    
    print(f"\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("Check plots/win_prediction_analysis.png for detailed charts!")

if __name__ == "__main__":
    main()