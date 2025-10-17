import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

class DataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.columns_to_drop = ['customerID']
        self.categorical_features = []
        self.numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    def load_and_clean_data(self, file_path):
        """Load and clean data - handles both files and file buffers"""
        print("📥 Loading data...")
        
        try:
            # If it's an uploaded file object (Streamlit UploadedFile)
            if hasattr(file_path, 'read'):
                df = pd.read_csv(file_path)
            else:
                # If it's a normal file path
                df = pd.read_csv(file_path)
            
            print(f"✅ Raw data loaded: {df.shape}")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            # Create empty DataFrame as fallback
            return pd.DataFrame()

        # Clean data - SIN inplace=True
        print("🧹 Cleaning data...")
        df_clean = df.copy()
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0)
        
        # Drop unnecessary columns
        df_clean = df_clean.drop(columns=self.columns_to_drop, errors='ignore')
        
        print(f"✅ Clean data: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
        return df_clean
        
    def load_uploaded_file(self, uploaded_file):
        """Load uploaded file from Streamlit - CORREGIDO"""
        try:
            if uploaded_file is not None:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
                
                print(f"✅ Uploaded file loaded: {df.shape}")
                
                # Apply cleaning
                return self.clean_data(df)
            else:
                raise ValueError("No file provided")
                
        except Exception as e:
            print(f"❌ Error loading uploaded file: {e}")
            raise

    def clean_data(self, df):
        """Clean data (separated for reusability) - CORREGIDO"""
        df_clean = df.copy()
        
        # Clean TotalCharges - SIN inplace=True
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0)
        
        # Drop unnecessary columns
        df_clean = df_clean.drop(columns=self.columns_to_drop, errors='ignore')
        
        return df_clean

    def preprocess_features(self, df, training=True):
        """Preprocess features for the model - CORREGIDO"""
        print("🔧 Preprocessing features...")
        df_processed = df.copy()
        
        # Only encode Churn if we are in training mode
        if training and 'Churn' in df_processed.columns:
            df_processed['Churn'] = df_processed['Churn'].map({'Yes': 1, 'No': 0})
        
        # Identify categorical features
        self.categorical_features = [
            col for col in df_processed.columns 
            if col not in self.numeric_features + ['Churn'] 
            and df_processed[col].dtype == 'object'
        ]
        
        print(f"📊 Categorical features: {self.categorical_features}")
        
        # Encode categorical variables using LabelEncoder - SIN inplace assignment
        for feature in self.categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
            
            if training:
                # Fit and transform for training
                encoded_values = self.label_encoders[feature].fit_transform(
                    df_processed[feature].astype(str)
                )
                df_processed[feature] = encoded_values
            else:
                # Transform for prediction, with error handling
                try:
                    encoded_values = self.label_encoders[feature].transform(
                        df_processed[feature].astype(str)
                    )
                    df_processed[feature] = encoded_values
                except ValueError as e:
                    print(f"⚠️ New value found in {feature}: {e}")
                    # Use -1 for unknown categories
                    df_processed[feature] = -1
        
        print("✅ Features preprocessed correctly")
        return df_processed, self.numeric_features, self.categorical_features

    def preprocess_for_prediction(self, df):
        """Preprocess data specifically for prediction"""
        print("🎯 Preprocessing for prediction...")
        
        # Clean data first
        df_clean = self.clean_data(df)
        
        # Preprocess without the Churn column
        df_processed, numeric_features, categorical_features = self.preprocess_features(
            df_clean, training=False
        )
        
        return df_processed, numeric_features, categorical_features
        
    def prepare_training_data(self, df_processed, numeric_features):
        """Prepare data for training - CORREGIDO"""
        print("🎯 Preparing data for training...")
        X = df_processed.drop('Churn', axis=1).copy()
        y = df_processed['Churn'].copy()
        
        # Scale numerical features - SIN inplace assignment
        X_scaled = X.copy()
        if numeric_features:  # Verificar que hay características numéricas
            scaled_values = self.scaler.fit_transform(X[numeric_features])
            # Asignación directa sin encadenamiento
            X_scaled[numeric_features] = scaled_values
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"📚 Training data: {X_train.shape[0]} samples")
        print(f"📊 Test data: {X_test.shape[0]} samples")
        if len(y_train) > 0:
            print(f"🎯 Class balance - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test, X.columns.tolist()
    
    def save_processor(self, file_path):
        """Save the processor for future use"""
        processor_data = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'categorical_features': self.categorical_features,
            'numeric_features': self.numeric_features
        }
        joblib.dump(processor_data, file_path)
        print(f"💾 Processor saved to: {file_path}")
    
    def load_processor(self, file_path):
        """Load processor from file"""
        processor_data = joblib.load(file_path)
        self.label_encoders = processor_data['label_encoders']
        self.scaler = processor_data['scaler']
        self.categorical_features = processor_data['categorical_features']
        self.numeric_features = processor_data['numeric_features']
        print(f"📂 Processor loaded from: {file_path}")

# Utility function to get processed data
def get_processed_data(file_path):
    """Main function to get processed data for training"""
    processor = DataProcessor()
    df_clean = processor.load_and_clean_data(file_path)
    df_processed, numeric_features, categorical_features = processor.preprocess_features(df_clean)
    
    X_train, X_test, y_train, y_test, feature_names = processor.prepare_training_data(
        df_processed, numeric_features
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'processor': processor,
        'df_clean': df_clean,
        'df_processed': df_processed
    }

def test_processor():
    """Test function for the DataProcessor"""
    try:
        processor = DataProcessor()
        
        # Test with sample data
        print("🧪 Testing DataProcessor...")
        
        # Create a small test dataset
        test_data = {
            'customerID': ['0001-Test', '0002-Test'],
            'gender': ['Male', 'Female'],
            'SeniorCitizen': [0, 1],
            'tenure': [12, 5],
            'MonthlyCharges': [70.5, 95.3],
            'TotalCharges': [846.0, 476.5],
            'Contract': ['Month-to-month', 'Two year'],
            'Churn': ['No', 'Yes']
        }
        
        df_test = pd.DataFrame(test_data)
        df_clean = processor.clean_data(df_test)
        
        print(f"✅ Clean test data: {df_clean.shape}")
        print("🎯 DataProcessor test completed successfully!")
        
    except Exception as e:
        print(f"❌ DataProcessor test failed: {e}")

if __name__ == "__main__":
    test_processor()