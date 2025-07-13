import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class HorseRaceMLTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        
    def load_processed_data(self, X_path='data/processed_X.csv', y_path='data/processed_y.csv'):
        """Load the preprocessed data"""
        try:
            X = pd.read_csv(X_path)
            y = pd.read_csv(y_path).values.ravel()  # Convert to 1D array
            
            print(f"✅ Loaded processed data:")
            print(f"   Features shape: {X.shape}")
            print(f"   Target shape: {y.shape}")
            print(f"   Feature columns: {list(X.columns)}")
            
            return X, y
        except Exception as e:
            print(f"❌ Error loading processed data: {e}")
            return None, None
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"✅ Data split completed:")
        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self):
        """Initialize different ML models to compare"""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=6
            )
        }
        
        print("✅ Initialized models:")
        for name in self.models.keys():
            print(f"   - {name}")
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all models"""
        print("\n🎯 TRAINING AND EVALUATING MODELS")
        print("=" * 50)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                      scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            results[name] = {
                'model': model,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'cv_rmse': cv_rmse,
                'y_pred_test': y_pred_test
            }
            
            print(f"   Train RMSE: {train_rmse:.3f}")
            print(f"   Test RMSE:  {test_rmse:.3f}")
            print(f"   Test R²:    {test_r2:.3f}")
            print(f"   Test MAE:   {test_mae:.3f}")
            print(f"   CV RMSE:    {cv_rmse:.3f}")
        
        return results
    
    def select_best_model(self, results):
        """Select the best performing model"""
        # Select based on test RMSE (lower is better)
        best_name = min(results.keys(), key=lambda x: results[x]['test_rmse'])
        self.best_model = results[best_name]['model']
        
        print(f"\n🏆 BEST MODEL: {best_name}")
        print("=" * 30)
        print(f"Test RMSE: {results[best_name]['test_rmse']:.3f}")
        print(f"Test R²:   {results[best_name]['test_r2']:.3f}")
        print(f"Test MAE:  {results[best_name]['test_mae']:.3f}")
        
        return best_name, results[best_name]
    
    def analyze_feature_importance(self, X, best_model_name, best_result):
        """Analyze feature importance for tree-based models"""
        model = best_result['model']
        
        if hasattr(model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n📊 FEATURE IMPORTANCE")
            print("=" * 30)
            print(self.feature_importance)
            
            return self.feature_importance
        else:
            print(f"\n⚠️  {best_model_name} doesn't support feature importance")
            return None
    
    def create_visualizations(self, results, X, y_test, best_model_name):
        """Create visualizations of model performance"""
        print("\n📈 Creating visualizations...")
        
        # Create plots directory
        os.makedirs('plots', exist_ok=True)
        
        # 1. Model comparison plot
        plt.figure(figsize=(12, 8))
        
        # Performance comparison
        plt.subplot(2, 2, 1)
        models = list(results.keys())
        test_rmse = [results[model]['test_rmse'] for model in models]
        test_r2 = [results[model]['test_r2'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, test_rmse, width, label='RMSE', alpha=0.8)
        plt.xlabel('Models')
        plt.ylabel('RMSE')
        plt.title('Model Performance Comparison (RMSE)')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        
        # R² comparison
        plt.subplot(2, 2, 2)
        plt.bar(x, test_r2, alpha=0.8, color='green')
        plt.xlabel('Models')
        plt.ylabel('R² Score')
        plt.title('Model Performance Comparison (R²)')
        plt.xticks(x, models, rotation=45)
        
        # Prediction vs Actual for best model
        plt.subplot(2, 2, 3)
        best_predictions = results[best_model_name]['y_pred_test']
        plt.scatter(y_test, best_predictions, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Speed Rating')
        plt.ylabel('Predicted Speed Rating')
        plt.title(f'{best_model_name}: Predictions vs Actual')
        
        # Feature importance (if available)
        plt.subplot(2, 2, 4)
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(8)
            plt.barh(top_features['feature'], top_features['importance'])
            plt.xlabel('Importance')
            plt.title('Top Feature Importance')
            plt.gca().invert_yaxis()
        else:
            plt.text(0.5, 0.5, 'Feature importance\nnot available\nfor this model', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Feature Importance')
        
        plt.tight_layout()
        plt.savefig('plots/model_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved to plots/model_analysis.png")
    
    def save_model(self, model, model_name, filename=None):
        """Save the trained model"""
        if filename is None:
            filename = f"models/best_horse_racing_model_{model_name.lower().replace(' ', '_')}.pkl"
        
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, filename)
        print(f"✅ Model saved to {filename}")
        return filename
    
    def hyperparameter_tuning(self, X_train, y_train, model_type='random_forest'):
        """Perform hyperparameter tuning for the best model type"""
        print(f"\n🔧 HYPERPARAMETER TUNING FOR {model_type.upper()}")
        print("=" * 50)
        
        if model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestRegressor(random_state=42)
        
        elif model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'min_samples_split': [2, 5, 10]
            }
            model = GradientBoostingRegressor(random_state=42)
        
        else:
            print(f"❌ Hyperparameter tuning not implemented for {model_type}")
            return None
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, 
            scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"✅ Best parameters: {grid_search.best_params_}")
        print(f"✅ Best CV score: {np.sqrt(-grid_search.best_score_):.3f}")
        
        return grid_search.best_estimator_
    
    def predict_speed_rating(self, race_features):
        """Make predictions for new race data"""
        if self.best_model is None:
            print("❌ No trained model available. Train a model first!")
            return None
        
        prediction = self.best_model.predict(race_features)
        return prediction

# Main execution
if __name__ == "__main__":
    print("🏇 HORSE RACING ML TRAINER")
    print("=" * 50)
    
    # Initialize trainer
    trainer = HorseRaceMLTrainer()
    
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
        trainer.create_visualizations(results, X, y_test, best_name)
        
        # Save the best model
        model_path = trainer.save_model(best_result['model'], best_name)
        
        print(f"\n🎉 SUCCESS! ML Pipeline Complete!")
        print(f"✅ Best model: {best_name}")
        print(f"✅ Model saved: {model_path}")
        print(f"✅ Visualizations: plots/model_analysis.png")
        
        # Optional: Perform hyperparameter tuning
        print(f"\n💡 Next steps:")
        print(f"   1. Check the visualizations in plots/")
        print(f"   2. Run hyperparameter tuning for better performance")
        print(f"   3. Create a prediction script for new races")
        
    else:
        print("❌ Failed to load processed data")
        print("Make sure you've run preprocess.py first!")