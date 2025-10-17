#!/usr/bin/env python3
"""
Streamlit App Launcher for Churn Prediction
"""

import os
import sys
import subprocess

def main():
    """Launch the Streamlit app"""
    print("🚀 Starting Churn Prediction Dashboard...")
    
    # Add src to path
    src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
    sys.path.append(src_path)
    
    # Check if models exist
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    if not os.path.exists(models_dir):
        print("❌ Models directory not found. Please run the training first.")
        print("💡 Run: python run_project.py")
        return
    
    # Check if model file exists
    model_path = os.path.join(models_dir, 'best_model.pkl')
    if not os.path.exists(model_path):
        print("❌ Model file not found. Please train the model first.")
        print("💡 Run: python run_project.py")
        return
    
    # Launch Streamlit
    app_path = os.path.join(os.path.dirname(__file__), 'streamlit_app.py')
    
    try:
        subprocess.run([
            'streamlit', 'run', app_path, 
            '--server.port=8501', 
            '--server.address=0.0.0.0'
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error starting app: {e}")

if __name__ == "__main__":
    main()