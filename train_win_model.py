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
            
            print(f"SUCCESS: Loaded win prediction data:")
            print(f"   Features shape: {X.shape}")
            print(f"   Target shape: {y.shape}")
            print(f"   Win rate: {y.mean()*100:.1f}%")
            print(f"   Feature columns: {list(X.columns)}")
            
            return X, y
        except Exception as e:
            print(f"ERROR loading processed data: {e}")
            print("Make sure you've run preprocess_v2.py first!")
            return None, None
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets - stratified for win prediction"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"SUCCESS: Data split completed:")
        print(f"   Training set: {X_train.shape[0]} samples (Win rate: {y_train.mean()*100:.1f}%)")
        print(f"   Test set: {X_test.shape[0]} samples (Win rate: {y_test.mean()*100:.1f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self):
        """
        Initialize CLASSIFICATION models for win prediction
        KEY CHANGE: These predict WIN PROBABILITY, not speed ratings!
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
        
        print("SUCCESS: Initialized WIN PREDICTION models:")
        for name in self.models.keys():
            print(f"   - {name}")
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate WIN PREDICTION models
        Uses classification metrics instead of regression metrics!
        """
        print("\n=== TRAINING WIN PREDICTION MODELS ===")
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
        
        print(f"\n*** BEST WIN PREDICTION MODEL: {best_name} ***")
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
            
            print("\n=== WIN PREDICTION FEATURE IMPORTANCE ===")
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
            
            print("\n=== WIN PREDICTION FEATURE IMPORTANCE (Logistic Regression) ===")
            print("=" * 40)
            print("Top 10 most important features:")
            print(self.feature_importance.head(10))
            
            return self.feature_importance
        else:
            print(f"\nWARNING: {best_model_name} doesn't support feature importance")
            return None
    
    def create_win_prediction_visualizations(self, results, X, y_test, best_model_name):
        """
        Create visualizations specifically for WIN PREDICTION
        Different charts than regression - focuses on probabilities and classification
        """
        print("\n=== Creating win prediction visualizations ===")
        
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
        
        print("SUCCESS: Win prediction visualizations saved to plots/win_prediction_analysis.png")
    
    def analyze_betting_performance(self, results, best_model_name, y_test):
        """
        Analyze how well the model would perform for BETTING
        This is what you ultimately want - finding value bets!
        """
        print(f"\n=== BETTING PERFORMANCE ANALYSIS ===")
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