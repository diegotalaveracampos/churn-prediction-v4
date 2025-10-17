#!/usr/bin/env python3
"""
Automation script for the Churn Prediction project
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import subprocess
import sys
import os
import webbrowser
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the path
sys.path.append('src')

def run_python_script(script_path, description):
    """Execute Python script directly"""
    print(f"\n{'='*60}")
    print(f"🚀 EXECUTING: {description}")
    print(f"{'='*60}")
    try:
        # Execute the script as a module
        with open(script_path, 'r', encoding='utf-8') as f:
            code = f.read()
        exec(code, globals())
        print(f"✅ {description} completed successfully")
        return True
    except Exception as e:
        print(f"❌ Error in {description}: {e}")
        return False

def check_requirements():
    """Verify that requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check if the dataset exists
    if not os.path.exists('data/telco_churn.csv'):
        print("❌ Dataset not found at data/telco_churn.csv")
        print("📥 Creating sample dataset...")
        create_sample_dataset()
    
    # Check that the necessary directories exist
    required_dirs = ['data', 'notebooks', 'src', 'models', 'app']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"📁 Creating directory: {dir_name}")
            os.makedirs(dir_name)
    
    # Check libraries
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        import streamlit as st
        print("✅ All libraries are available")
        return True
    except ImportError as e:
        print(f"❌ Error importing libraries: {e}")
        return False

def create_sample_dataset():
    """Create a sample dataset if the real one does not exist"""
    try:
        # More realistic sample data
        np.random.seed(42)
        n_samples = 1000
        
        sample_data = {
            'customerID': [f'0000-{i:04d}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples, p=[0.5, 0.5]),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
            'tenure': np.random.randint(1, 72, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.4, 0.5, 0.1]),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.4, 0.2]),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.4, 0.2]),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2]),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
            'PaymentMethod': np.random.choice([
                'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
            ], n_samples, p=[0.3, 0.2, 0.25, 0.25]),
            'MonthlyCharges': np.random.uniform(20, 120, n_samples).round(2),
            'TotalCharges': np.random.uniform(50, 8000, n_samples).round(2),
        }
        
        # Create target variable Churn with realistic patterns
        df = pd.DataFrame(sample_data)
        
        # Realistic patterns for churn
        churn_proba = np.zeros(n_samples)
        
        # Factors that increase churn
        churn_proba += (df['Contract'] == 'Month-to-month') * 0.3
        churn_proba += (df['tenure'] < 12) * 0.2
        churn_proba += (df['OnlineSecurity'] == 'No') * 0.1
        churn_proba += (df['TechSupport'] == 'No') * 0.1
        churn_proba += (df['PaymentMethod'] == 'Electronic check') * 0.1
        churn_proba += (df['MonthlyCharges'] > 80) * 0.1
        
        # Normalize and convert to probabilities
        churn_proba = np.clip(churn_proba, 0, 1)
        df['Churn'] = np.random.binomial(1, churn_proba)
        df['Churn'] = df['Churn'].map({1: 'Yes', 0: 'No'})
        
        # Adjust TotalCharges based on tenure and MonthlyCharges
        df['TotalCharges'] = df['tenure'] * df['MonthlyCharges'] * np.random.uniform(0.8, 1.2, n_samples)
        
        df.to_csv('data/telco_churn.csv', index=False)
        print("✅ Sample dataset created in data/telco_churn.csv")
        print(f"📊 Dimensions: {df.shape}")
        print(f"🎯 Churn Distribution: {df['Churn'].value_counts().to_dict()}")
        print("⚠️ This is a sample dataset. For better results, use the real Kaggle dataset")
        
    except Exception as e:
        print(f"❌ Error creating sample dataset: {e}")

def run_eda_analysis():
    """Execute exploratory analysis directly"""
    print("\n🔍 Running Exploratory Data Analysis...")
    
    try:
        # Import libraries for EDA
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Configuration
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Load data
        df = pd.read_csv('data/telco_churn.csv')
        
        print("=== DATASET INFORMATION ===")
        print(f"Dimensions: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Basic analysis
        print("\n=== CHURN DISTRIBUTION ===")
        churn_dist = df['Churn'].value_counts()
        print(churn_dist)
        
        # Create basic visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Churn Distribution
        churn_dist.plot(kind='bar', ax=axes[0,0], color=['skyblue', 'salmon'])
        axes[0,0].set_title('Churn Distribution')
        axes[0,0].set_xlabel('Churn')
        axes[0,0].set_ylabel('Count')
        
        # Tenure vs Churn
        df.boxplot(column='tenure', by='Churn', ax=axes[0,1])
        axes[0,1].set_title('Tenure by Churn Status')
        
        # MonthlyCharges vs Churn
        df.boxplot(column='MonthlyCharges', by='Churn', ax=axes[1,0])
        axes[1,0].set_title('Monthly Charges by Churn Status')
        
        # Contract type
        contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index')
        contract_churn.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Churn Rate by Contract Type')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('notebooks/eda_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save clean dataset
        df_clean = df.copy()
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        df_clean['TotalCharges'].fillna(0, inplace=True)
        df_clean.to_csv('data/telco_churn_clean.csv', index=False)
        
        print("✅ Exploratory analysis completed")
        print("💾 Graphs saved to 'notebooks/eda_visualizations.png'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in exploratory analysis: {e}")
        return False


def run_model_training():
    """Execute model training directly"""
    print("\n🤖 Training Machine Learning models...")
    
    try:
        # Import the processor directly
        from src.data_processing import DataProcessor, get_processed_data
        
        # Load and process data
        print("📥 Loading and processing data...")
        data_dict = get_processed_data('data/telco_churn.csv')
        
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        feature_names = data_dict['feature_names']
        processor = data_dict['processor']
        
        print(f"📊 Training data: {X_train.shape}")
        print(f"📈 Test data: {X_test.shape}")
        print(f"🎯 Features: {len(feature_names)}")
        print(f"📋 Feature names: {feature_names}")
        
        # Define and train models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=100)
        }
        
        results = {}
        print("\n🎯 TRAINING MODELS...")
        
        for name, model in models.items():
            print(f"🚀 Training {name}...")
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = model.score(X_test, y_test)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"  ✅ Accuracy: {accuracy:.4f}")
            print(f"  📊 AUC Score: {auc_score:.4f}")
            print(f"  📈 CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Select best model based on AUC
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
        best_model = results[best_model_name]['model']
        best_result = results[best_model_name]
        
        print(f"\n🌟 BEST MODEL: {best_model_name}")
        print(f"🎯 AUC Score: {best_result['auc_score']:.4f}")
        print(f"🎯 Accuracy: {best_result['accuracy']:.4f}")
        
        # Save the complete model package
        model_data = {
            'model': best_model,
            'processor': processor,
            'feature_names': feature_names,
            'results': results,
            'X_test': X_test,
            'y_test': y_test,
            'model_name': best_model_name,
            'training_date': pd.Timestamp.now()
        }
        
        with open('models/best_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save processor separately for easier access
        processor.save_processor('models/data_processor.pkl')
        
        print("💾 Model saved to 'models/best_model.pkl'")
        print("💾 Processor saved to 'models/data_processor.pkl'")
        
        # Create comprehensive results visualizations
        create_model_visualizations(results, best_model_name, X_test, y_test)
        
        print("✅ Model training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_model_visualizations(results, best_model_name, X_test, y_test):
    """Create comprehensive model comparison visualizations"""
    print("📊 Creating model visualizations...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model Comparison Bar Plot
        model_names = list(results.keys())
        auc_scores = [results[name]['auc_score'] for name in model_names]
        accuracies = [results[name]['accuracy'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0,0].bar(x - width/2, auc_scores, width, label='AUC Score', alpha=0.8)
        axes[0,0].bar(x + width/2, accuracies, width, label='Accuracy', alpha=0.8)
        axes[0,0].set_xlabel('Models')
        axes[0,0].set_ylabel('Scores')
        axes[0,0].set_title('Model Performance Comparison')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(model_names, rotation=45)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Confusion Matrix for Best Model
        best_result = results[best_model_name]
        cm = confusion_matrix(y_test, best_result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1],
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        axes[0,1].set_title(f'Confusion Matrix - {best_model_name}')
        axes[0,1].set_ylabel('True Label')
        axes[0,1].set_xlabel('Predicted Label')
        
        # 3. ROC Curves for All Models
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
            axes[1,0].plot(fpr, tpr, label=f'{name} (AUC = {result["auc_score"]:.3f})')
        
        axes[1,0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1,0].set_xlabel('False Positive Rate')
        axes[1,0].set_ylabel('True Positive Rate')
        axes[1,0].set_title('ROC Curves - All Models')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Feature Importance (if available)
        best_model = results[best_model_name]['model']
        if hasattr(best_model, 'feature_importances_'):
            feature_importances = best_model.feature_importances_
            feature_names = [f'Feature {i+1}' for i in range(len(feature_importances))]
            
            # Sort features by importance
            indices = np.argsort(feature_importances)[::-1]
            sorted_features = [feature_names[i] for i in indices]
            sorted_importances = feature_importances[indices]
            
            axes[1,1].barh(range(len(sorted_importances)), sorted_importances[:10])
            axes[1,1].set_yticks(range(len(sorted_importances[:10])))
            axes[1,1].set_yticklabels(sorted_features[:10])
            axes[1,1].set_xlabel('Importance')
            axes[1,1].set_title(f'Top 10 Feature Importances - {best_model_name}')
        
        plt.tight_layout()
        plt.savefig('notebooks/model_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved to 'notebooks/model_results.png'")
        
    except Exception as e:
        print(f"⚠️ Error creating visualizations: {e}")


def run_streamlit_app():
    """Execute Streamlit application"""
    print("\n🌐 Starting Streamlit application...")
    print("💡 The application will open in your browser")
    print("🛑 To stop the application, press Ctrl+C in this terminal")
    
    try:
        # Check that streamlit is installed
        import streamlit
        
        # Wait a moment
        time.sleep(2)
        
        # ❌ REMOVE THE LINE THAT OPENS THE FIRST TAB MANUALLY: 
        # webbrowser.open('http://localhost:8501')
        
        # Execute streamlit using the module directly
        from streamlit.web import cli as stcli
        
        # ✅ FIX: Remove the manual browser opening and rely on Streamlit's default.
        # We also remove '--server.headless=true' to ensure Streamlit does the opening.
        sys.argv = [
            "streamlit", 
            "run", 
            "app/streamlit_app.py", 
            "--server.port=8501", 
            "--server.address=localhost"
            # Removed: "--server.headless=true"
        ]
        
        stcli.main()
        
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        print("💡 You can start Streamlit manually with: python -m streamlit run app/streamlit_app.py")
        return False

def main():
    print("🎯 STARTING CHURN PREDICTION PROJECT")
    print("="*60)
    
    # Check requirements
    if not check_requirements():
        print("❌ System requirements are not met")
        return
    
    # Execute EDA
    if not run_eda_analysis():
        print("⚠️ Continuing without complete EDA...")
    
    # Execute model training
    if not run_model_training():
        print("❌ Error in model training. Stopping...")
        return
    
    # Ask if to start Streamlit
    print("\n" + "="*60)
    user_input = input("Do you want to start the Streamlit application? (y/n): ")
    
    if user_input.lower() in ['s', 'si', 'sí', 'y', 'yes']: # Keeping the Spanish options 's' and 'sí' just in case, but English user should use 'y' or 'yes'
        run_streamlit_app()
    else:
        print("\n🎉 Project configured successfully!")
        print("📊 You can view the results in:")
        print("   - notebooks/eda_visualizations.png (Exploratory Analysis)")
        print("   - notebooks/model_results.png (Model Results)")
        print("   - models/ (Trained Models)")
        print("\n🚀 To start Streamlit manually:")
        print("   python -m streamlit run app/streamlit_app.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Project stopped by the user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")