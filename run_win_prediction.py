"""
RUN WIN PREDICTION PIPELINE
===========================

This is your main script to run the complete win prediction pipeline.
Save this as: run_win_prediction.py

This script:
1. Runs preprocessing (preprocess_v2.py)
2. Runs model training (train_win_model.py) 
3. Shows you the results

This replaces running files individually.
"""

import subprocess
import sys
import os

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:")
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False
    except FileNotFoundError:
        print(f"❌ File {script_name} not found!")
        print(f"Make sure you've created {script_name} in the current directory.")
        return False

def check_requirements():
    """Check if required files exist"""
    required_files = [
        'data/sample_race_data.csv',
        'preprocess_v2.py', 
        'train_win_model.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    return True

def main():
    """Run the complete win prediction pipeline"""
    print("🏇 COMPLETE WIN PREDICTION PIPELINE")
    print("=" * 60)
    print("This will:")
    print("1. ✨ Preprocess data for win prediction")
    print("2. 🎯 Train win probability models") 
    print("3. 📊 Create performance visualizations")
    print("4. 🎰 Analyze betting performance")
    print("=" * 60)
    
    # Check if all required files exist
    if not check_requirements():
        print("\n❌ Pipeline cannot run - missing files!")
        return False
    
    # Step 1: Run preprocessing
    if not run_script('preprocess_v2.py', "STEP 1: Preprocessing for Win Prediction"):
        print("\n❌ Preprocessing failed! Cannot continue.")
        return False
    
    # Step 2: Run model training
    if not run_script('train_win_model.py', "STEP 2: Training Win Prediction Models"):
        print("\n❌ Model training failed!")
        return False
    
    # Success!
    print(f"\n{'='*60}")
    print("🎉 WIN PREDICTION PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    
    print("\n📊 Check your results:")
    print("   📁 data/win_prediction_X.csv - Processed features")
    print("   📁 data/win_prediction_y.csv - Win targets") 
    print("   📁 models/ - Trained win prediction model")
    print("   📁 plots/win_prediction_analysis.png - Performance charts")
    
    print("\n🎯 Key metrics to look for:")
    print("   • AUC Score > 0.65 = Your model beats random guessing")
    print("   • AUC Score > 0.75 = Good predictive power")
    print("   • AUC Score > 0.85 = Excellent (rare with 40 data points)")
    
    print("\n🚀 Next steps:")
    print("   1. Review the plots/win_prediction_analysis.png")
    print("   2. Check which features are most important")
    print("   3. Scale up with 1000+ races from Equibase")
    print("   4. Build live prediction system")
    
    return True

if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n💡 Troubleshooting tips:")
        print("   1. Make sure you have all required files")
        print("   2. Check that data/sample_race_data.csv exists")
        print("   3. Install required packages: pip install -r requirements.txt")
        print("   4. Run scripts individually to see specific errors")
        
    input("\nPress Enter to exit...")