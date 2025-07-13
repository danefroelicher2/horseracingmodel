def initialize_models(self, dataset_size):
        """Initialize models optimized for dataset size"""
        
        if dataset_size >= 10000:
            # Professional-grade models for 10K+ horses
            self.models = {
                'Logistic Regression': LogisticRegression(
                    random_state=42,
                    max_iter=3000,
                    class_weight='balanced',
                    C=0.5,  # More regularization for large dataset
                    solver='lbfgs'
                ),
                'Random Forest': RandomForestClassifier(
                    n_estimators=300,  # More trees for large dataset
                    random_state=42,
                    max_depth=20,      # Deeper for complex patterns
                    min_samples_split=20,
                    min_samples_leaf=10,
                    class_weight='balanced',
                    n_jobs=-1  # Use all CPU cores
                ),
                'Gradient Boosting': GradientBoostingClassifier(
                    n_estimators=300,
                    random_state=42,
                    max_depth=10,
                    learning_rate=0.08,  # Slightly lower for stability
                    min_samples_split=20,
                    min_samples_leaf=10,
                    subsample=0.8  # Add stochasticity
                ),
                'Extra Trees': RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    max_depth=15,
                    min_samples_split=15,
                    min_samples_leaf=8,
                    class_weight='balanced',
                    bootstrap=False,  # Extra Trees variation
                    n_jobs=-1
                )
            }
            print("🏆 PROFESSIONAL-GRADE MODELS initialized for 10,000+ horses:")
            
        elif dataset_size >= 5000:
            # Enhanced models for large datasets
            self.models = {
                'Logistic Regression': LogisticRegression(
                    random_state=42,
                    max_iter=2000,
                    class_weight='balanced',
                    C=1.0 )}
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
        
    def load_processed_data(self, X_path=None, y_path=None):
        """Load the preprocessed win prediction data - auto-detect largest dataset"""
        
        # Auto-detect processed data files
        if X_path is None or y_path is None:
            import glob
            x_files = glob.glob('data/win_prediction_*_X.csv')
            y_files = glob.glob('data/win_prediction_*_y.csv')
            
            if x_files and y_files:
                # Sort by file size to get largest dataset
                x_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
                y_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
                X_path = x_files[0]
                y_path = y_files[0]
            else:
                # Fallback to default names
                X_path = 'data/win_prediction_X.csv'
                y_path = 'data/win_prediction_y.csv'
        
        try:
            X = pd.read_csv(X_path)
            y = pd.read_csv(y_path).values.ravel()
            
            print(f"SUCCESS: Loaded enhanced win prediction data:")
            print(f"   Dataset: {X_path}")
            print(f"   Features shape: {X.shape}")
            print(f"   Target shape: {y.shape}")
            print(f"   Win rate: {y.mean()*100:.2f}%")
            print(f"   Total winners: {y.sum():,}")
            print(f"   Total losers: {len(y) - y.sum():,}")
            
            # Dataset size classification
            if len(X) >= 10000:
                print(f"   🏆 PROFESSIONAL-GRADE DATASET ({len(X):,} horses)")
            elif len(X) >= 5000:
                print(f"   ✅ LARGE DATASET ({len(X):,} horses)")
            elif len(X) >= 1000:
                print(f"   📊 MEDIUM DATASET ({len(X):,} horses)")
            else:
                print(f"   ⚠️  SMALL DATASET ({len(X):,} horses)")
            
            return X, y
        except Exception as e:
            print(f"ERROR loading processed data: {e}")
            print("Make sure you've run preprocess.py first!")
            return None, None
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets with proper stratification"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"SUCCESS: Data split completed:")
        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"     - Winners: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
        print(f"     - Losers: {len(y_train) - y_train.sum()}")
        print(f"   Test set: {X_test.shape[0]} samples") 
        print(f"     - Winners: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
        print(f"     - Losers: {len(y_test) - y_test.sum()}")
        
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self):
        """Initialize classification models optimized for 1000+ data points"""
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=2000,
                class_weight='balanced',
                C=1.0
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200,  # More trees for better performance
                random_state=42,
                max_depth=15,      # Deeper trees for more complex patterns
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,  # More estimators for better learning
                random_state=42,
                max_depth=8,
                learning_rate=0.1,
                min_samples_split=10,
                min_samples_leaf=5
            )
        }
        
        print("SUCCESS: Initialized WIN PREDICTION models for 1000+ horses:")
        for name, model in self.models.items():
            if hasattr(model, 'n_estimators'):
                print(f"   - {name} ({model.n_estimators} estimators)")
            else:
                print(f"   - {name}")
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate models with proper metrics for large datasets"""
        print("\n=== TRAINING WIN PREDICTION MODELS ON 1000+ HORSES ===")
        print("=" * 60)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Get prediction probabilities
            y_prob_train = model.predict_proba(X_train)[:, 1]
            y_prob_test = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            train_auc = roc_auc_score(y_train, y_prob_train)
            test_auc = roc_auc_score(y_test, y_prob_test)
            
            # Cross-validation with more folds for better estimates
            cv_scores = cross_val_score(model, X_train, y_train, cv=10, 
                                      scoring='roc_auc', n_jobs=-1)
            cv_auc = cv_scores.mean()
            cv_std = cv_scores.std()
            
            results[name] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'train_auc': train_auc,
                'test_auc': test_auc,
                'cv_auc': cv_auc,
                'cv_std': cv_std,
                'y_pred_test': y_pred_test,
                'y_prob_test': y_prob_test
            }
            
            print(f"   Train Accuracy: {train_accuracy:.3f}")
            print(f"   Test Accuracy:  {test_accuracy:.3f}")
            print(f"   Train AUC:      {train_auc:.3f}")
            print(f"   Test AUC:       {test_auc:.3f}")
            print(f"   CV AUC:         {cv_auc:.3f} (+/- {cv_std*2:.3f})")
        
        return results
    
    def select_best_model(self, results):
        """Select best model based on cross-validation AUC"""
        best_name = max(results.keys(), key=lambda x: results[x]['cv_auc'])
        self.best_model = results[best_name]['model']
        
        print(f"\n*** BEST WIN PREDICTION MODEL: {best_name} ***")
        print("=" * 50)
        print(f"Test AUC:      {results[best_name]['test_auc']:.3f}")
        print(f"CV AUC:        {results[best_name]['cv_auc']:.3f} (+/- {results[best_name]['cv_std']*2:.3f})")
        print(f"Test Accuracy: {results[best_name]['test_accuracy']:.3f}")
        
        # Interpret the AUC score
        auc = results[best_name]['test_auc']
        if auc >= 0.85:
            interpretation = "Excellent - Commercial quality!"
        elif auc >= 0.75:
            interpretation = "Very Good - Strong predictive power"
        elif auc >= 0.65:
            interpretation = "Good - Solid foundation" 
        elif auc >= 0.55:
            interpretation = "Fair - Some predictive ability"
        else:
            interpretation = "Poor - Needs improvement"
        
        print(f"Interpretation: {interpretation}")
        
        return best_name, results[best_name]
    
    def analyze_feature_importance(self, X, best_model_name, best_result):
        """Analyze feature importance with statistical significance"""
        model = best_result['model']
        
        if hasattr(model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
            print("=" * 50)
            print("Top 15 most important features for WIN prediction:")
            for i, (_, row) in enumerate(self.feature_importance.head(15).iterrows(), 1):
                print(f"   {i:2d}. {row['feature']:25} ({row['importance']:.4f})")
            
            # Analyze feature categories
            print(f"\nFEATURE CATEGORY ANALYSIS:")
            
            odds_features = self.feature_importance[
                self.feature_importance['feature'].str.contains('odds|favorite|probability', case=False)
            ]
            performance_features = self.feature_importance[
                self.feature_importance['feature'].str.contains('win_rate|place_rate', case=False)
            ]
            position_features = self.feature_importance[
                self.feature_importance['feature'].str.contains('post', case=False)
            ]
            
            print(f"   Odds-related features: {len(odds_features)} features, avg importance: {odds_features['importance'].mean():.4f}")
            print(f"   Performance features: {len(performance_features)} features, avg importance: {performance_features['importance'].mean():.4f}")
            print(f"   Position features: {len(position_features)} features, avg importance: {position_features['importance'].mean():.4f}")
            
            return self.feature_importance
            
        elif hasattr(model, 'coef_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': np.abs(model.coef_[0])
            }).sort_values('importance', ascending=False)
            
            print("\n=== FEATURE IMPORTANCE (Logistic Regression Coefficients) ===")
            print("=" * 50)
            print("Top 15 most important features:")
            for i, (_, row) in enumerate(self.feature_importance.head(15).iterrows(), 1):
                print(f"   {i:2d}. {row['feature']:25} ({row['importance']:.4f})")
            
            return self.feature_importance
        else:
            print(f"\nWARNING: {best_model_name} doesn't support feature importance")
            return None
    
    def create_comprehensive_visualizations(self, results, X, y_test, best_model_name):
        """Create comprehensive visualizations for 1000+ horse analysis"""
        print("\n=== Creating comprehensive visualizations ===")
        
        os.makedirs('plots', exist_ok=True)
        
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Model comparison
        plt.subplot(3, 4, 1)
        models = list(results.keys())
        test_auc = [results[model]['test_auc'] for model in models]
        cv_auc = [results[model]['cv_auc'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, test_auc, width, label='Test AUC', alpha=0.8)
        plt.bar(x + width/2, cv_auc, width, label='CV AUC', alpha=0.8)
        plt.xlabel('Models')
        plt.ylabel('AUC Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, [m.replace(' ', '\n') for m in models], fontsize=9)
        plt.legend()
        plt.ylim(0.5, 1.0)
        
        # 2. ROC Curve
        plt.subplot(3, 4, 2)
        best_probs = results[best_model_name]['y_prob_test']
        fpr, tpr, _ = roc_curve(y_test, best_probs)
        auc_score = results[best_model_name]['test_auc']
        
        plt.plot(fpr, tpr, linewidth=2, label=f'{best_model_name}\n(AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        plt.subplot(3, 4, 3)
        precision, recall, _ = precision_recall_curve(y_test, best_probs)
        
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        
        # 4. Prediction probability distribution
        plt.subplot(3, 4, 4)
        winners = best_probs[y_test == 1]
        losers = best_probs[y_test == 0]
        
        plt.hist(losers, bins=30, alpha=0.7, label=f'Losers (n={len(losers)})', 
                color='red', density=True)
        plt.hist(winners, bins=30, alpha=0.7, label=f'Winners (n={len(winners)})', 
                color='green', density=True)
        plt.xlabel('Predicted Win Probability')
        plt.ylabel('Density')
        plt.title('Probability Distribution')
        plt.legend()
        
        # 5. Confusion Matrix
        plt.subplot(3, 4, 5)
        best_predictions = results[best_model_name]['y_pred_test']
        cm = confusion_matrix(y_test, best_predictions)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Predicted Lose', 'Predicted Win'],
                   yticklabels=['Actual Lose', 'Actual Win'])
        plt.title('Confusion Matrix')
        
        # 6. Feature Importance
        plt.subplot(3, 4, 6)
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(12)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), 
                      [f.replace('_', ' ').title() for f in top_features['feature']])
            plt.xlabel('Importance')
            plt.title('Top 12 Feature Importance')
            plt.gca().invert_yaxis()
        
        # 7. Performance by probability threshold
        plt.subplot(3, 4, 7)
        thresholds = np.arange(0.1, 0.9, 0.05)
        precisions = []
        recalls = []
        
        for threshold in thresholds:
            predictions = (best_probs >= threshold).astype(int)
            if predictions.sum() > 0:
                precision = (predictions & y_test).sum() / predictions.sum()
                recall = (predictions & y_test).sum() / y_test.sum()
            else:
                precision = 0
                recall = 0
            precisions.append(precision)
            recalls.append(recall)
        
        plt.plot(thresholds, precisions, label='Precision', marker='o')
        plt.plot(thresholds, recalls, label='Recall', marker='s')
        plt.xlabel('Probability Threshold')
        plt.ylabel('Score')
        plt.title('Precision/Recall vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. Calibration plot
        plt.subplot(3, 4, 8)
        from sklearn.calibration import calibration_curve
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, best_probs, n_bins=10
        )
        
        plt.plot(mean_predicted_value, fraction_of_positives, marker='o', 
                linewidth=2, label='Model')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 9. Model performance by field size
        plt.subplot(3, 4, 9)
        # This would require field_size data - placeholder for now
        plt.text(0.5, 0.5, 'Performance by\nField Size\n(Requires field_size data)', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Performance by Field Size')
        
        # 10. Cross-validation scores
        plt.subplot(3, 4, 10)
        model_names = list(results.keys())
        cv_means = [results[model]['cv_auc'] for model in model_names]
        cv_stds = [results[model]['cv_std'] for model in model_names]
        
        plt.bar(range(len(model_names)), cv_means, yerr=cv_stds, 
               capsize=5, alpha=0.8)
        plt.xticks(range(len(model_names)), 
                  [m.replace(' ', '\n') for m in model_names], fontsize=9)
        plt.ylabel('CV AUC Score')
        plt.title('Cross-Validation Results')
        plt.ylim(0.5, 1.0)
        
        # 11. Learning curve (simplified)
        plt.subplot(3, 4, 11)
        train_scores = [results[model]['train_auc'] for model in model_names]
        test_scores = [results[model]['test_auc'] for model in model_names]
        
        x = range(len(model_names))
        plt.plot(x, train_scores, 'o-', label='Training AUC')
        plt.plot(x, test_scores, 's-', label='Test AUC') 
        plt.xticks(x, [m.replace(' ', '\n') for m in model_names], fontsize=9)
        plt.ylabel('AUC Score')
        plt.title('Train vs Test Performance')
        plt.legend()
        plt.ylim(0.5, 1.0)
        
        # 12. Summary statistics
        plt.subplot(3, 4, 12)
        plt.text(0.1, 0.8, f'Dataset Size: {len(y_test) + len(results[best_model_name]["y_pred_test"])}', 
                transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.1, 0.7, f'Test Set: {len(y_test)}', transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.1, 0.6, f'Winners: {y_test.sum()}', transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.1, 0.5, f'Win Rate: {y_test.mean():.1%}', transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.1, 0.4, f'Best Model: {best_model_name}', transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.1, 0.3, f'Best AUC: {results[best_model_name]["test_auc"]:.3f}', 
                transform=plt.gca().transAxes, fontsize=12)
        plt.axis('off')
        plt.title('Summary Statistics')
        
        plt.tight_layout()
        plt.savefig('plots/comprehensive_win_prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("SUCCESS: Comprehensive visualizations saved to plots/comprehensive_win_prediction_analysis.png")
    
    def analyze_betting_performance(self, results, best_model_name, y_test):
        """Comprehensive betting performance analysis for 1000+ horses"""
        print(f"\n=== COMPREHENSIVE BETTING PERFORMANCE ANALYSIS ===")
        print("=" * 60)
        
        best_probs = results[best_model_name]['y_prob_test']
        
        print("PERFORMANCE AT DIFFERENT CONFIDENCE THRESHOLDS:")
        print("Threshold | Selections | Win Rate | Expected | ROI Est.")
        print("-" * 60)
        
        thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
        
        for threshold in thresholds:
            selected = best_probs >= threshold
            if selected.sum() > 0:
                actual_win_rate = y_test[selected].mean()
                expected_win_rate = best_probs[selected].mean()
                
                # Estimate ROI assuming average odds of 1/probability
                avg_implied_odds = 1 / expected_win_rate
                estimated_roi = (actual_win_rate * avg_implied_odds - 1) * 100
                
                print(f"{threshold:8.1%} | {selected.sum():10d} | {actual_win_rate:8.1%} | {expected_win_rate:8.1%} | {estimated_roi:6.1f}%")
            else:
                print(f"{threshold:8.1%} | {0:10d} | {'N/A':8s} | {'N/A':8s} | {'N/A':6s}")
        
        # Top decile analysis
        top_decile_threshold = np.percentile(best_probs, 90)
        top_quintile_threshold = np.percentile(best_probs, 80)
        
        top_decile_mask = best_probs >= top_decile_threshold
        top_quintile_mask = best_probs >= top_quintile_threshold
        
        print(f"\nTOP SELECTIONS ANALYSIS:")
        print(f"Top 10% (Decile):")
        print(f"   Threshold: {top_decile_threshold:.1%}")
        print(f"   Selections: {top_decile_mask.sum()}")
        print(f"   Win Rate: {y_test[top_decile_mask].mean():.1%}")
        print(f"   Expected: {best_probs[top_decile_mask].mean():.1%}")
        
        print(f"Top 20% (Quintile):")
        print(f"   Threshold: {top_quintile_threshold:.1%}")
        print(f"   Selections: {top_quintile_mask.sum()}")
        print(f"   Win Rate: {y_test[top_quintile_mask].mean():.1%}")
        print(f"   Expected: {best_probs[top_quintile_mask].mean():.1%}")
        
        # Statistical significance test
        from scipy import stats
        overall_win_rate = y_test.mean()
        top_decile_win_rate = y_test[top_decile_mask].mean()
        
        if top_decile_mask.sum() > 10:  # Minimum sample size
            # Chi-square test
            observed = [y_test[top_decile_mask].sum(), top_decile_mask.sum() - y_test[top_decile_mask].sum()]
            expected_rate = overall_win_rate
            expected = [top_decile_mask.sum() * expected_rate, top_decile_mask.sum() * (1 - expected_rate)]
            
            chi2, p_value = stats.chisquare(observed, expected)
            
            print(f"\nSTATISTICAL SIGNIFICANCE:")
            print(f"   Overall win rate: {overall_win_rate:.1%}")
            print(f"   Top decile win rate: {top_decile_win_rate:.1%}")
            print(f"   Improvement: {top_decile_win_rate - overall_win_rate:.1%}")
            print(f"   P-value: {p_value:.4f}")
            
            if p_value < 0.05:
                print(f"   ✅ STATISTICALLY SIGNIFICANT improvement!")
            else:
                print(f"   ⚠️  Not statistically significant (p > 0.05)")
    
    def save_model(self, model, model_name, filename=None):
        """Save the trained model with metadata"""
        if filename is None:
            filename = f"models/win_prediction_model_{model_name.lower().replace(' ', '_')}.pkl"
        
        os.makedirs('models', exist_ok=True)
        
        # Save model with metadata
        model_data = {
            'model': model,
            'model_name': model_name,
            'training_date': pd.Timestamp.now(),
            'feature_names': self.feature_importance['feature'].tolist() if self.feature_importance is not None else None
        }
        
        joblib.dump(model_data, filename)
        print(f"SUCCESS: Enhanced model saved to {filename}")
        return filename
    
    def predict_win_probabilities(self, race_features):
        """Predict win probabilities for new races"""
        if self.best_model is None:
            print("ERROR: No trained model available!")
            return None
        
        win_probabilities = self.best_model.predict_proba(race_features)[:, 1]
        return win_probabilities
    
    def simulate_race_prediction(self, X, y, sample_size=8):
        """Enhanced race simulation with realistic scenario"""
        if self.best_model is None:
            print("ERROR: No trained model available!")
            return None
        
        # Sample a complete race (8 horses from same race ideally)
        sample_indices = np.random.choice(len(X), size=min(sample_size, len(X)), replace=False)
        race_X = X.iloc[sample_indices]
        race_y = y[sample_indices]
        
        # Predict win probabilities
        win_probs = self.predict_win_probabilities(race_X)
        
        # Create detailed race results
        race_results = pd.DataFrame({
            'Horse': [f"Horse_{i+1}" for i in range(len(race_X))],
            'Predicted_Win_Prob': win_probs,
            'Predicted_Rank': range(1, len(win_probs) + 1),
            'Actually_Won': race_y,
            'Confidence': ['High' if p > 0.3 else 'Medium' if p > 0.15 else 'Low' for p in win_probs]
        })
        
        # Sort by predicted probability
        race_results = race_results.sort_values('Predicted_Win_Prob', ascending=False)
        race_results['Predicted_Rank'] = range(1, len(race_results) + 1)
        
        print(f"\n=== ENHANCED RACE SIMULATION ===")
        print("=" * 60)
        print("Rank | Horse    | Win Prob | Confidence | Actually Won")
        print("-" * 60)
        
        for _, row in race_results.iterrows():
            won_symbol = "🏆 YES" if row['Actually_Won'] else "❌ NO"
            print(f"{row['Predicted_Rank']:4d} | {row['Horse']:8s} | {row['Predicted_Win_Prob']:8.1%} | {row['Confidence']:10s} | {won_symbol}")
        
        # Detailed analysis
        top_pick_won = race_results.iloc[0]['Actually_Won']
        top_3_picks = race_results.head(3)['Actually_Won'].sum()
        actual_winner_rank = race_results[race_results['Actually_Won'] == 1]['Predicted_Rank'].iloc[0] if any(race_results['Actually_Won']) else "No winner"
        
        print(f"\n=== PREDICTION ANALYSIS ===")
        print(f"   Top pick won: {'✅ SUCCESS!' if top_pick_won else '❌ Missed'}")
        print(f"   Top 3 picks contained winner: {'✅ YES' if top_3_picks > 0 else '❌ NO'}")
        print(f"   Actual winner ranked: #{actual_winner_rank}")
        print(f"   Model confidence in top pick: {race_results.iloc[0]['Predicted_Win_Prob']:.1%}")
        
        return race_results

# Main execution
if __name__ == "__main__":
    print("HORSE RACING WIN PROBABILITY TRAINER - 1000+ HORSES")
    print("=" * 60)
    
    # Initialize trainer
    trainer = WinProbabilityTrainer()
    
    # Load processed data
    X, y = trainer.load_processed_data()
    
    if X is not None and y is not None:
        print(f"\nTraining on {len(X)} horses - this should give much better results!")
        
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
        
        # Create comprehensive visualizations
        trainer.create_comprehensive_visualizations(results, X, y_test, best_name)
        
        # Analyze betting performance
        trainer.analyze_betting_performance(results, best_name, y_test)
        
        # Save the best model
        model_path = trainer.save_model(best_result['model'], best_name)
        
        # Simulate race predictions
        trainer.simulate_race_prediction(X_test, y_test)
        
        print(f"\n*** 1000+ HORSE WIN PREDICTION COMPLETE! ***")
        print("=" * 60)
        print(f"✅ Best model: {best_name}")
        print(f"✅ Test AUC: {best_result['test_auc']:.3f}")
        print(f"✅ Model saved: {model_path}")
        print(f"✅ Visualizations: plots/comprehensive_win_prediction_analysis.png")
        
        # Provide realistic next steps
        auc_score = best_result['test_auc']
        if auc_score >= 0.75:
            print(f"\n🎉 EXCELLENT RESULTS! Your model is showing commercial-grade performance.")
            print(f"🚀 Ready to scale to 10,000+ races and build live prediction system!")
        elif auc_score >= 0.65:
            print(f"\n✅ GOOD RESULTS! Your model has solid predictive power.")
            print(f"📈 Next: Optimize features and scale to 10,000+ races")
        else:
            print(f"\n📊 LEARNING PROGRESS! Better than random, needs optimization.")
            print(f"🔧 Next: Feature engineering and data quality improvements")
        
    else:
        print("ERROR: Failed to load processed data")
        print("Make sure you've run data_generator.py and preprocess.py first!")