import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve, 
                           precision_recall_curve, confusion_matrix, accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class WinProbabilityTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        
    def load_processed_data(self, X_path='data/win_prediction_X.csv', y_path='data/win_prediction_y.csv'):
        """Load the preprocessed win prediction data"""
        try:
            X = pd.read_csv(X_path)
            y = pd.read_csv(y_path).values.ravel()  # Convert to 1D array
            
            print(f"✅ Loaded win prediction data:")
            print(f"   Features shape: {X.shape}")
            print(f"   Target shape: {y.shape}")
            print(f"   Win rate: {y.mean()*100:.1f}%")
            print(f"   Feature columns: {list(X.columns)}")
            
            return X, y
        except Exception as e:
            print(f"❌ Error loading processed data: {e}")
            print("Make sure you've run preprocess_v2.py first!")
            return None, None
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets - stratified for win prediction"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"✅ Data split completed:")
        print(f"   Training set: {X_train.shape[0]} samples (Win rate: {y_train.mean()*100:.1f}%)")
        print(f"   Test set: {X_test.shape[0]} samples (Win rate: {y_test.mean()*100:.1f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self):
        """
        Initialize CLASSIFICATION models for win prediction
        🎯 KEY CHANGE: These predict WIN PROBABILITY, not speed ratings!
        """
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'  # Handle imbalanced data (few winners)
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=6,
                learning_rate=0.1
            )
        }
        
        print("✅ Initialized WIN PREDICTION models:")
        for name in self.models.keys():
            print(f"   - {name}")
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate WIN PREDICTION models
        🎯 Uses classification metrics instead of regression metrics!
        """
        print("\n🎯 TRAINING WIN PREDICTION MODELS")
        print("=" * 50)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Get prediction probabilities (this is what we really want!)
            y_prob_train = model.predict_proba(X_train)[:, 1]  # Probability of winning
            y_prob_test = model.predict_proba(X_test)[:, 1]
            
            # Calculate classification metrics
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            
            # AUC Score (Area Under ROC Curve) - KEY METRIC for win prediction
            train_auc = roc_auc_score(y_train, y_prob_train)
            test_auc = roc_auc_score(y_test, y_prob_test)
            
            # Cross-validation AUC
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                      scoring='roc_auc')
            cv_auc = cv_scores.mean()
            
            results[name] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'train_auc': train_auc,
                'test_auc': test_auc,
                'cv_auc': cv_auc,
                'y_pred_test': y_pred_test,
                'y_prob_test': y_prob_test
            }
            
            print(f"   Train Accuracy: {train_accuracy:.3f}")
            print(f"   Test Accuracy:  {test_accuracy:.3f}")
            print(f"   Test AUC:       {test_auc:.3f}")
            print(f"   CV AUC:         {cv_auc:.3f}")
        
        return results
    
    def select_best_model(self, results):
        """Select the best performing model based on AUC score"""
        # Select based on test AUC (higher is better for classification)
        best_name = max(results.keys(), key=lambda x: results[x]['test_auc'])
        self.best_model = results[best_name]['model']
        
        print(f"\n🏆 BEST WIN PREDICTION MODEL: {best_name}")
        print("=" * 40)
        print(f"Test AUC:      {results[best_name]['test_auc']:.3f}")
        print(f"Test Accuracy: {results[best_name]['test_accuracy']:.3f}")
        print(f"CV AUC:        {results[best_name]['cv_auc']:.3f}")
        
        return best_name, results[best_name]
    
    def analyze_feature_importance(self, X, best_model_name, best_result):
        """Analyze what features matter most for WIN PREDICTION"""
        model = best_result['model']
        
        if hasattr(model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n📊 WIN PREDICTION FEATURE IMPORTANCE")
            print("=" * 40)
            print("Top 10 most important features:")
            print(self.feature_importance.head(10))
            
            return self.feature_importance
        elif hasattr(model, 'coef_'):
            # For logistic regression, use coefficient magnitude
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': np.abs(model.coef_[0])
            }).sort_values('importance', ascending=False)
            
            print("\n📊 WIN PREDICTION FEATURE IMPORTANCE (Logistic Regression)")
            print("=" * 40)
            print("Top 10 most important features:")
            print(self.feature_importance.head(10))
            
            return self.feature_importance
        else:
            print(f"\n⚠️  {best_model_name} doesn't support feature importance")
            return None
    
    def create_win_prediction_visualizations(self, results, X, y_test, best_model_name):
        """
        Create visualizations specifically for WIN PREDICTION
        🎯 Different charts than regression - focuses on probabilities and classification
        """
        print("\n📈 Creating win prediction visualizations...")
        
        os.makedirs('plots', exist_ok=True)
        
        plt.figure(figsize=(15, 12))
        
        # 1. Model AUC comparison
        plt.subplot(2, 3, 1)
        models = list(results.keys())
        test_auc = [results[model]['test_auc'] for model in models]
        
        bars = plt.bar(models, test_auc, alpha=0.8, color=['skyblue', 'lightgreen', 'salmon'])
        plt.xlabel('Models')
        plt.ylabel('AUC Score')
        plt.title('Win Prediction Model Comparison (AUC)')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, auc in zip(bars, test_auc):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{auc:.3f}', ha='center', va='bottom')
        
        # 2. ROC Curve for best model
        plt.subplot(2, 3, 2)
        best_probs = results[best_model_name]['y_prob_test']
        fpr, tpr, _ = roc_curve(y_test, best_probs)
        auc_score = results[best_model_name]['test_auc']
        
        plt.plot(fpr, tpr, linewidth=2, label=f'{best_model_name} (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Win Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        plt.subplot(2, 3, 3)
        precision, recall, _ = precision_recall_curve(y_test, best_probs)
        
        plt.plot(recall, precision, linewidth=2, label=f'{best_model_name}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Probability distribution for winners vs losers
        plt.subplot(2, 3, 4)
        winners = best_probs[y_test == 1]
        losers = best_probs[y_test == 0]
        
        plt.hist(losers, bins=20, alpha=0.7, label='Losers', color='red', density=True)
        plt.hist(winners, bins=20, alpha=0.7, label='Winners', color='green', density=True)
        plt.xlabel('Predicted Win Probability')
        plt.ylabel('Density')
        plt.title('Probability Distribution: Winners vs Losers')
        plt.legend()
        
        # 5. Confusion Matrix
        plt.subplot(2, 3, 5)
        best_predictions = results[best_model_name]['y_pred_test']
        cm = confusion_matrix(y_test, best_predictions)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        # 6. Feature importance (if available)
        plt.subplot(2, 3, 6)
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(10)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title('Top 10 Feature Importance')
            plt.gca().invert_yaxis()
        else:
            plt.text(0.5, 0.5, 'Feature importance\nnot available\nfor this model', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Feature Importance')
        
        plt.tight_layout()
        plt.savefig('plots/win_prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Win prediction visualizations saved to plots/win_prediction_analysis.png")
    
    def analyze_betting_performance(self, results, best_model_name, y_test):
        """
        🎰 Analyze how well the model would perform for BETTING
        This is what you ultimately want - finding value bets!
        """
        print(f"\n🎰 BETTING PERFORMANCE ANALYSIS")
        print("=" * 50)
        
        best_probs = results[best_model_name]['y_prob_test']
        
        # Analyze performance at different probability thresholds
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
        
        print("Performance at different confidence thresholds:")
        print("Threshold | Selections | Win Rate | Expected Win Rate")
        print("-" * 55)
        
        for threshold in thresholds:
            selected = best_probs >= threshold
            if selected.sum() > 0:
                actual_win_rate = y_test[selected].mean()
                expected_win_rate = best_probs[selected].mean()
                print(f"{threshold:8.1%} | {selected.sum():10d} | {actual_win_rate:8.1%} | {expected_win_rate:14.1%}")
        
        # Find the best horses (top 10% by probability)
        top_10_pct = np.percentile(best_probs, 90)
        top_selections = best_probs >= top_10_pct
        
        if top_selections.sum() > 0:
            top_win_rate = y_test[top_selections].mean()
            print(f"\n🏆 TOP 10% PREDICTIONS:")
            print(f"   Selections: {top_selections.sum()}")
            print(f"   Win Rate: {top_win_rate:.1%}")
            print(f"   Expected: {best_probs[top_selections].mean():.1%}")
    
    def save_model(self, model, model_name, filename=None):
        """Save the trained win prediction model"""
        if filename is None:
            filename = f"models/win_prediction_model_{model_name.lower().replace(' ', '_')}.pkl"
        
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, filename)
        print(f"✅ Win prediction model saved to {filename}")
        return filename
    
    def predict_win_probabilities(self, race_features):
        """
        🎯 Predict WIN PROBABILITIES for new races
        This is your ultimate goal - live race prediction!
        """
        if self.best_model is None:
            print("❌ No trained model available. Train a model first!")
            return None
        
        win_probabilities = self.best_model.predict_proba(race_features)[:, 1]
        return win_probabilities
    
    def simulate_race_prediction(self, X, y, sample_size=8):
        """
        🏁 Simulate predicting a single race
        Shows what your live system will look like!
        """
        if self.best_model is None:
            print("❌ No trained model available!")
            return None
        
        # Take a random sample to simulate a race
        sample_indices = np.random.choice(len(X), size=min(sample_size, len(X)), replace=False)
        race_X = X.iloc[sample_indices]
        race_y = y[sample_indices]
        
        # Predict win probabilities
        win_probs = self.predict_win_probabilities(race_X)
        
        # Create race results
        race_results = pd.DataFrame({
            'Horse': [f"Horse_{i+1}" for i in range(len(race_X))],
            'Predicted_Win_Prob': win_probs,
            'Actually_Won': race_y
        })
        
        # Sort by predicted probability
        race_results = race_results.sort_values('Predicted_Win_Prob', ascending=False)
        race_results['Rank'] = range(1, len(race_results) + 1)
        
        print(f"\n🏁 SIMULATED RACE PREDICTION")
        print("=" * 50)
        print("Rank | Horse    | Win Prob | Actually Won")
        print("-" * 45)
        
        for _, row in race_results.iterrows():
            won_symbol = "🏆" if row['Actually_Won'] else "❌"
            print(f"{row['Rank']:4d} | {row['Horse']:8s} | {row['Predicted_Win_Prob']:8.1%} | {won_symbol}")
        
        # Check if we picked the winner
        top_pick_won = race_results.iloc[0]['Actually_Won']
        actual_winner_rank = race_results[race_results['Actually_Won'] == 1]['Rank'].iloc[0] if any(race_results['Actually_Won']) else "No winner in sample"
        
        print(f"\n📊 PREDICTION RESULTS:")
        print(f"   Top pick won: {'YES! 🎉' if top_pick_won else 'No 😞'}")
        print(f"   Actual winner ranked: {actual_winner_rank}")
        
        return race_results

# Main execution
if __name__ == "__main__":
    print("🏇 WIN PROBABILITY ML TRAINER")
    print("=" * 50)
    
    # Initialize trainer
    trainer = WinProbabilityTrainer()
    
    # Load processed data
    X, y = trainer.load_processed_data()
    
    if X is not None and y is not None:
        # Split data
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        
        # Initialize models
        trainer.initialize_models()
        
        # Train and evaluate models
        results = trainer.train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        # Select best model
        best_name, best_result = trainer.select_best_model(results)
        
        # Analyze feature importance
        trainer.analyze_feature_importance(X, best_name, best_result)
        
        # Create visualizations
        trainer.create_win_prediction_visualizations(results, X, y_test, best_name)
        
        # Analyze betting performance
        trainer.analyze_betting_performance(results, best_name, y_test)
        
        # Save the best model
        model_path = trainer.save_model(best_result['model'], best_name)
        
        # Simulate a race prediction
        trainer.simulate_race_prediction(X_test, y_test)
        
        print(f"\n🎉 WIN PREDICTION PIPELINE COMPLETE!")
        print(f"✅ Best model: {best_name}")
        print(f"✅ Model saved: {model_path}")
        print(f"✅ Visualizations: plots/win_prediction_analysis.png")
        
        print(f"\n💡 Next steps:")
        print(f"   1. Check your AUC score (>0.7 is good, >0.8 is excellent)")
        print(f"   2. Look at feature importance - what matters most?")
        print(f"   3. Scale up with 1000+ races from Equibase")
        print(f"   4. Build live prediction interface")
        
    else:
        print("❌ Failed to load processed data")
        print("Make sure you've run preprocess_v2.py first!")