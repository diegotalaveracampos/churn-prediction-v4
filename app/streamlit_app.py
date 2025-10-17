import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve
import sys
import os
import joblib

# 🎯 SOLUCIÓN ESPECÍFICA PARA TU REPOSITORIO
try:
    # Ruta exacta para Streamlit Cloud
    sys.path.append('/mount/src/diegotalaveracampos/churn-prediction-v2/src')
    from data_processing import DataProcessor
    st.success("✅ DataProcessor imported successfully from src")
except ImportError as e:
    try:
        # Fallback: importación relativa
        current_dir = os.path.dirname(__file__)
        src_path = os.path.join(current_dir, '..', 'src')
        sys.path.append(src_path)
        from data_processing import DataProcessor
        st.success("✅ DataProcessor imported via relative path")
    except ImportError:
        # Fallback final: Mock
        st.warning("⚠️ Using mock DataProcessor")
        class DataProcessor:
            def __init__(self):
                self.scaler = None
                self.label_encoders = {}
                self.columns_to_drop = ['customerID']
                self.numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
                self.categorical_features = []

            def load_and_clean_data(self, file_path):
                try:
                    if hasattr(file_path, 'read'):
                        df = pd.read_csv(file_path)
                    else:
                        df = pd.read_csv(file_path)
                    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
                    df['TotalCharges'] = df['TotalCharges'].fillna(0)
                    return df.drop(columns=self.columns_to_drop, errors='ignore')
                except Exception as e:
                    st.error(f"Error loading data: {e}")
                    return pd.DataFrame()

            def clean_data(self, df):
                df_clean = df.copy()
                df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
                df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0)
                return df_clean.drop(columns=self.columns_to_drop, errors='ignore')

            def preprocess_features(self, df, training=False):
                df_processed = df.copy()
                if training and 'Churn' in df_processed.columns:
                    df_processed['Churn'] = df_processed['Churn'].map({'Yes': 1, 'No': 0})
                return df_processed, self.numeric_features, self.categorical_features

            def load_uploaded_file(self, uploaded_file):
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file)
                    return self.clean_data(df)
                except Exception as e:
                    st.error(f"Error loading uploaded file: {e}")
                    return pd.DataFrame()

            def save_processor(self, file_path):
                pass

            def load_processor(self, file_path):
                pass

# Configuración de la página
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎯 FUNCIONES DE CARGA OPTIMIZADAS PARA CLOUD
@st.cache_resource
def load_model():
    """Carga el modelo con paths específicos para TU repositorio"""
    try:
        # Rutas específicas para tu repositorio en Streamlit Cloud
        possible_paths = [
            '/mount/src/diegotalaveracampos/churn-prediction-v2/models/best_model.pkl',
            'models/best_model.pkl',
            '../models/best_model.pkl',
            './models/best_model.pkl',
            os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')
        ]
        
        model_data = None
        loaded_path = ""
        
        for path in possible_paths:
            try:
                with open(path, 'rb') as f:
                    model_data = pickle.load(f)
                loaded_path = path
                st.success(f"✅ Model loaded from: {path}")
                break
            except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
                st.warning(f"⚠️ Failed to load from {path}: {e}")
                continue
        
        if model_data is None:
            st.error("""
            ❌ Model file not found in any expected location.
            
            Please ensure:
            1. The model is trained and saved as 'best_model.pkl'
            2. The file is in the 'models/' directory
            3. You've run: python run_project.py
            """)
            return None
            
        # Asegurar que el processor esté disponible
        if 'processor' not in model_data:
            model_data['processor'] = DataProcessor()
            
        return model_data
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

@st.cache_data
def load_sample_data():
    """Carga datos de muestra para la app"""
    try:
        possible_paths = [
            'data/telco_churn.csv',
            '../data/telco_churn.csv',
            './data/telco_churn.csv',
            os.path.join(os.path.dirname(__file__), '..', 'data', 'telco_churn.csv')
        ]
        
        for path in possible_paths:
            try:
                sample_data = pd.read_csv(path)
                st.success(f"✅ Sample data loaded from: {path}")
                return sample_data.head(8)
            except FileNotFoundError:
                continue
                
        # Si no encuentra el archivo, crea datos de muestra
        st.warning("⚠️ Sample data file not found, using mock data")
        return create_sample_data()
        
    except Exception as e:
        st.warning(f"⚠️ Could not load sample data: {e}")
        return create_sample_data()

def create_sample_data():
    """Crea datos de muestra si no existe el archivo"""
    data = {
        'customerID': ['3668-QPYBK', '9237-HQITU', '9305-CDHLH', '1452-KNVPV', '6723-OGFNR'],
        'gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
        'SeniorCitizen': [0, 0, 0, 1, 0],
        'Partner': ['No', 'No', 'No', 'Yes', 'No'],
        'Dependents': ['No', 'No', 'No', 'Yes', 'No'],
        'tenure': [1, 2, 8, 45, 12],
        'PhoneService': ['No', 'Yes', 'Yes', 'Yes', 'Yes'],
        'MultipleLines': ['No phone service', 'No', 'Yes', 'No', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic', 'Fiber optic', 'DSL', 'Fiber optic'],
        'OnlineSecurity': ['No', 'No', 'No', 'Yes', 'No'],
        'OnlineBackup': ['Yes', 'Yes', 'No', 'Yes', 'No'],
        'DeviceProtection': ['No', 'No', 'Yes', 'Yes', 'No'],
        'TechSupport': ['No', 'No', 'No', 'Yes', 'No'],
        'StreamingTV': ['No', 'No', 'Yes', 'Yes', 'Yes'],
        'StreamingMovies': ['No', 'No', 'Yes', 'Yes', 'Yes'],
        'Contract': ['Month-to-month', 'Month-to-month', 'Month-to-month', 'Two year', 'One year'],
        'PaperlessBilling': ['Yes', 'Yes', 'Yes', 'No', 'Yes'],
        'PaymentMethod': ['Mailed check', 'Electronic check', 'Electronic check', 'Credit card (automatic)', 'Bank transfer (automatic)'],
        'MonthlyCharges': [29.85, 70.7, 99.65, 45.30, 89.50],
        'TotalCharges': [29.85, 151.65, 820.5, 2038.50, 1074.00],
        'Churn': ['No', 'Yes', 'Yes', 'No', 'No']
    }
    return pd.DataFrame(data)

# Título principal
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("""
This application uses Machine Learning to predict which customers are most likely to leave the service.
**Proactively identify the risk and take retention actions!**
""")

# Cargar modelo y datos
model_data = load_model()
sample_data = load_sample_data()

if model_data is None:
    st.error("""
    ❌ **Model not available**
    
    Please ensure:
    1. The model file exists in the `models/` directory
    2. The file is named `best_model.pkl`
    3. The model was trained successfully
    
    If running locally, execute `python run_project.py` first.
    """)
    st.stop()

model = model_data['model']
processor = model_data['processor']
feature_names = model_data.get('feature_names', [])
results = model_data.get('results', {})
X_test = model_data.get('X_test', pd.DataFrame())
y_test = model_data.get('y_test', np.array([]))

# Sidebar para navegación
st.sidebar.title("🔍 Navigation")
app_mode = st.sidebar.selectbox(
    "Select a section:",
    ["📈 Overview", "🔮 Predict Customer", "📋 Customer Batch", "🤖 Model Analysis", "📚 About the Project"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**💡 Tip:** Use the 'Predict Customer' section for individual analysis and 'Customer Batch' for bulk analysis.
""")

if app_mode == "📈 Overview":
    st.header("🎯 Project Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Project Goal")
        st.markdown("""
        - **🎯 Predict** which customers have a high probability of churning
        - **🔍 Identify** the main factors influencing churn
        - **💡 Provide** actionable insights for customer retention
        - **💰 Optimize** resources in retention campaigns
        """)
        
        st.subheader("🚀 Expected Benefits")
        st.markdown("""
        - Reduction in churn rate
        - Better resource allocation
        - More effective retention campaigns
        - Increase in Customer Lifetime Value
        """)
    
    with col2:
        st.subheader("📊 Model Metrics")
        if results:
            best_result_name = list(results.keys())[0]
            best_result = results[best_result_name]
            
            metric1, metric2, metric3 = st.columns(3)
            
            with metric1:
                st.metric("Accuracy", f"{best_result['accuracy']:.1%}")
            with metric2:
                st.metric("AUC Score", f"{best_result['auc_score']:.3f}")
            with metric3:
                st.metric("CV Score (Mean)", f"{best_result['cv_mean']:.3f}")
        
        st.subheader("🛠️ Tech Stack")
        st.markdown("""
        - **Language:** Python
        - **ML:** Scikit-learn, XGBoost
        - **Visualization:** Plotly, Matplotlib
        - **Dashboard:** Streamlit
        - **Explainability:** SHAP
        """)
    
    # Display sample dataset
    st.subheader("📋 Sample Dataset")
    st.dataframe(sample_data, use_container_width=True)
    st.caption("Sample of customer data used for training the model")

elif app_mode == "🔮 Predict Customer":
    st.header("🔮 Individual Customer Prediction")
    
    st.markdown("""
    Enter the customer's characteristics to predict their churn probability.
    The model will analyze the patterns and provide specific recommendations.
    """)
    
    # Divide into columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📅 Contract Information")
        tenure = st.slider("Months of Tenure", 0, 72, 12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        
        st.subheader("💳 Payment Information")
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0, step=5.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0, step=50.0)
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    
    with col2:
        st.subheader("🌐 Internet Services")
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    
    with col3:
        st.subheader("📺 Streaming Services")
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        
        st.subheader("👥 Demographic Information")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        
        st.subheader("📞 Phone Service")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    
    # Create input data
    input_data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    # Prediction button
    if st.button("🎯 Predict Churn Probability", type="primary", use_container_width=True):
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        try:
            # Preprocess
            df_processed, numeric_features, categorical_features = processor.preprocess_features(input_df, training=False)
            
            # Scale numerical features
            if hasattr(processor, 'scaler') and processor.scaler is not None:
                try:
                    df_processed[numeric_features] = processor.scaler.transform(df_processed[numeric_features])
                except Exception as e:
                    st.warning(f"⚠️ Scaling failed: {e}. Using unscaled features.")
            else:
                st.warning("⚠️ Scaler not available. Using unscaled features.")
            
            # Align columns to match the model's expected features
            missing_cols = set(feature_names) - set(df_processed.columns)
            for c in missing_cols:
                df_processed[c] = 0
            
            # Ensure the order of columns matches the training data
            df_final = df_processed[feature_names]

            # Predict
            probability = model.predict_proba(df_final)[0, 1]
            
            # Show result
            st.success("✅ Prediction completed successfully!")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Churn Probability", 'font': {'size': 24}},
                    delta = {'reference': 50, 'increasing': {'color': "red"}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': 'lightgreen'},
                            {'range': [30, 70], 'color': 'yellow'},
                            {'range': [70, 100], 'color': 'red'}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=300, font={'color': "darkblue", 'family': "Arial"})
                st.plotly_chart(fig, use_container_width=True)
            
            with col_result2:
                # Recommendations
                st.subheader("🎯 Recommendations")
                
                if probability < 0.3:
                    st.success("🟢 LOW RISK")
                    st.markdown("""
                    - **Stable customer**, low churn risk
                    - **Maintain** quality service
                    - **Consider** loyalty programs
                    - **Monitor** changes in usage patterns
                    """)
                    st.metric("Exact Probability", f"{probability:.1%}", delta="Low Risk", delta_color="off")
                    
                elif probability < 0.7:
                    st.warning("🟡 MODERATE RISK")
                    st.markdown("""
                    - **Customer at moderate risk**
                    - **Proactively contact**
                    - **Offer** retention incentives
                    - **Review** recent complaints
                    - **Propose** contract improvement
                    """)
                    st.metric("Exact Probability", f"{probability:.1%}", delta="Moderate Risk", delta_color="off")
                    
                else:
                    st.error("🔴 HIGH RISK")
                    st.markdown("""
                    - **High-risk customer**
                    - **Immediate contact** required
                    - **Offer** special discounts
                    - **Escalate** to retention team
                    - **Analyze** specific causes
                    - **Propose** long-term contract
                    """)
                    st.metric("Exact Probability", f"{probability:.1%}", delta="High Risk", delta_color="inverse")
            
            # Risk factor analysis
            st.subheader("🔍 Identified Risk Factors")
            
            risk_factors = []
            if contract == "Month-to-month":
                risk_factors.append("Month-to-month contract (high risk)")
            if tenure < 6:
                risk_factors.append("Tenure less than 6 months")
            if internet_service == "Fiber optic" and monthly_charges > 80:
                risk_factors.append("Fiber optic service with high charges")
            if payment_method == "Electronic check":
                risk_factors.append("Payment method: Electronic check")
            if online_security == "No" and internet_service != "No":
                risk_factors.append("Lack of online security")
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.info("No significant risk factors identified")
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("💡 Verify that all fields are correctly completed")

elif app_mode == "📋 Customer Batch":
    st.header("📋 Customer Batch Analysis")
    
    st.info("""
    **Upload a CSV file with customer data to analyze multiple predictions simultaneously.**
    
    📋 **Required columns:**
    customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, 
    MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, 
    TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, 
    PaymentMethod, MonthlyCharges, TotalCharges
    """)
    
    st.warning("""
    ⚠️ **IMPORTANT**: DO NOT include the 'Churn' column in your CSV file. 
    
    This is the variable to be predicted. Only include the customer characteristics.
    """)
    
    # Create sample file for download
    try:
        sample_data_no_churn = sample_data.drop('Churn', axis=1) if 'Churn' in sample_data.columns else sample_data
        sample_csv = sample_data_no_churn.head(5).to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Sample File (No Churn)",
            data=sample_csv,
            file_name="example_customers_no_churn.csv",
            mime="text/csv",
            help="Sample file with the correct format (no Churn column)"
        )
    except Exception as e:
        st.warning(f"Could not create sample file: {e}")
    
    uploaded_file = st.file_uploader("Upload CSV file", type="csv", help="CSV file with customer data")
    
    if uploaded_file is not None:
        try:
            # Load data
            uploaded_file.seek(0)
            batch_data = pd.read_csv(uploaded_file)
            
            # Check for Churn column
            if 'Churn' in batch_data.columns:
                st.error("""
                ❌ **ERROR**: Your file contains the 'Churn' column. 
                
                This column must be removed as it is the variable to be predicted.
                Please remove the 'Churn' column from your CSV file and upload it again.
                """)
                st.stop()
            
            st.success(f"✅ File loaded: {batch_data.shape[0]} customers, {batch_data.shape[1]} features")
            
            # Show preview
            with st.expander("📊 Data Preview"):
                st.dataframe(batch_data.head(), use_container_width=True)
            
            # Preprocess
            uploaded_file.seek(0)
            df_clean = processor.load_uploaded_file(uploaded_file)
            df_processed, numeric_features, categorical_features = processor.preprocess_features(df_clean, training=False)
            
            # Scale
            if hasattr(processor, 'scaler') and processor.scaler is not None:
                try:
                    df_processed[numeric_features] = processor.scaler.transform(df_processed[numeric_features])
                except Exception as e:
                    st.warning(f"⚠️ Scaling failed: {e}. Using unscaled features.")
            else:
                st.warning("⚠️ Scaler not available. Using unscaled features.")

            # Align columns
            missing_cols = set(feature_names) - set(df_processed.columns)
            for c in missing_cols:
                df_processed[c] = 0
            df_final = df_processed[feature_names]

            # Predict
            predictions = model.predict(df_final)
            probabilities = model.predict_proba(df_final)[:, 1]
            
            # Add results to DataFrame
            results_df = batch_data.copy()
            results_df['Churn_Probability'] = probabilities
            results_df['Churn_Prediction'] = ['High Risk' if p == 1 else 'Low Risk' for p in predictions]
            results_df['Risk_Level'] = pd.cut(probabilities, 
                                             bins=[0, 0.3, 0.7, 1.0], 
                                             labels=['Low', 'Medium', 'High'])
            
            # Show results
            st.subheader("📈 Analysis Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            high_risk_count = (results_df['Risk_Level'] == 'High').sum()
            medium_risk_count = (results_df['Risk_Level'] == 'Medium').sum()
            low_risk_count = (results_df['Risk_Level'] == 'Low').sum()
            total_customers = len(results_df)
            
            with col1:
                st.metric("Total Customers", total_customers)
            with col2:
                st.metric("High Risk", high_risk_count, 
                          delta=f"{high_risk_count/total_customers*100:.1f}%")
            with col3:
                st.metric("Medium Risk", medium_risk_count,
                          delta=f"{medium_risk_count/total_customers*100:.1f}%")
            with col4:
                st.metric("Low Risk", low_risk_count,
                          delta=f"{low_risk_count/total_customers*100:.1f}%")
            
            # Risk distribution
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                risk_counts = results_df['Risk_Level'].value_counts().reset_index()
                risk_counts.columns = ['Risk_Level', 'Count']
                risk_counts['Risk_Level'] = pd.Categorical(risk_counts['Risk_Level'], categories=['Low', 'Medium', 'High'], ordered=True)
                risk_counts = risk_counts.sort_values('Risk_Level')
                
                fig_pie = px.pie(risk_counts, names='Risk_Level', values='Count',
                               title='Distribution of Risk Levels',
                               color='Risk_Level',
                               color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'})
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_chart2:
                fig_hist = px.histogram(results_df, x='Churn_Probability', 
                                         title='Distribution of Churn Probabilities',
                                         nbins=20, color_discrete_sequence=['red'])
                fig_hist.add_vline(x=0.7, line_dash="dash", line_color="red", annotation_text="High Risk")
                fig_hist.add_vline(x=0.3, line_dash="dash", line_color="green", annotation_text="Low Risk")
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Results table
            st.subheader("📋 Prediction Details")
            
            # Filters
            col_filter1, col_filter2 = st.columns(2)
            with col_filter1:
                min_prob = st.slider("Minimum Probability", 0.0, 1.0, 0.0, 0.1)
            with col_filter2:
                risk_filter = st.multiselect("Filter by Risk Level", 
                                             ['Low', 'Medium', 'High'], 
                                             default=['Low', 'Medium', 'High'])
            
            filtered_df = results_df[
                (results_df['Churn_Probability'] >= min_prob) & 
                (results_df['Risk_Level'].isin(risk_filter))
            ].sort_values('Churn_Probability', ascending=False)
            
            st.dataframe(filtered_df, use_container_width=True)
            
            # Export results
            st.subheader("💾 Export Results")
            csv = results_df.to_csv(index=False).encode('utf-8')
            
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="📥 Download Full Results",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col_dl2:
                high_risk_csv = results_df[results_df['Risk_Level'] == 'High'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download High Risk Only",
                    data=high_risk_csv,
                    file_name="high_risk_customers.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("💡 Ensure the file has the correct format and all necessary columns")

elif app_mode == "🤖 Model Analysis":
    st.header("🤖 Machine Learning Model Analysis")
    
    if not results:
        st.warning("No model results available")
        st.stop()
    
    best_result_name = list(results.keys())[0]
    best_result = results[best_result_name]
    
    # Model Metrics
    st.subheader("📊 Performance Metrics")
    
    col_met1, col_met2, col_met3, col_met4 = st.columns(4)
    
    with col_met1:
        st.metric("Accuracy", f"{best_result['accuracy']:.3f}")
    with col_met2:
        st.metric("AUC Score", f"{best_result['auc_score']:.3f}")
    with col_met3:
        st.metric("CV Score (Mean)", f"{best_result['cv_mean']:.3f}")
    with col_met4:
        st.metric("CV Std", f"{best_result['cv_std']:.3f}")
    
    # Evaluation Charts
    if len(y_test) > 0 and len(best_result.get('y_pred', [])) > 0:
        col_eval1, col_eval2 = st.columns(2)
        
        with col_eval1:
            # Confusion Matrix
            st.subheader("📋 Confusion Matrix")
            try:
                cm = confusion_matrix(y_test, best_result['y_pred'])
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['No Churn', 'Churn'], 
                            yticklabels=['No Churn', 'Churn'])
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate Confusion Matrix: {e}")
        
        with col_eval2:
            # ROC Curve
            st.subheader("📈 ROC Curve")
            try:
                fpr, tpr, _ = roc_curve(y_test, best_result['y_pred_proba'])
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(fpr, tpr, linewidth=2, label=f'{best_result_name} (AUC = {best_result["auc_score"]:.3f})')
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate ROC Curve: {e}")
    else:
        st.warning("Test data or predictions not available for plotting.")
    
    # Feature Importance
    st.subheader("🔍 Feature Importance")
    
    if hasattr(model, 'feature_importances_') and feature_names:
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True).tail(15)
        
        fig = px.bar(feature_importance, 
                      x='importance', 
                      y='feature', 
                      orientation='h',
                      title='Top 15 Most Important Features',
                      color='importance',
                      color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Feature importance data is not available for this model.")
    
    # Model Information
    st.subheader("🔧 Technical Model Information")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
        **🎯 Algorithm Used:**
        - XGBoost Classifier
        
        **📊 Evaluation Metrics:**
        - **Accuracy:** Measure of correct predictions
        - **AUC Score:** Ability to distinguish between classes
        - **Cross-Validation:** Model robustness
        
        **⚙️ Preprocessing:**
        - Scaling of numerical features
        - Encoding of categorical variables
        - Handling of missing values
        """)
    
    with col_info2:
        st.markdown("""
        **🎯 Business Application:**
        - Proactive identification of at-risk customers
        - Optimization of retention campaigns
        - Personalization of offers
        - Reduction of acquisition costs
        
        **📈 Benefits:**
        - Better resource allocation
        - Higher ROI in retention
        - Data-driven decisions
        - Churn rate reduction
        """)

else:  # 📚 About the Project
    st.header("📚 About the Project")
    
    st.markdown("""
    ## 🎯 Customer Churn Prediction
    
    This project demonstrates a complete **Data Science** workflow applied to a real business problem: 
    predicting customer churn in the telecommunications sector.
    
    ### 📊 Main Objectives
    
    1. **🔍 Exploratory Analysis:** Understand the patterns and factors influencing churn
    2. **🤖 Predictive Modeling:** Develop an ML model to identify at-risk customers
    3. **📈 Interactive Dashboard:** Provide tools for data-driven decision making
    4. **💼 Business Impact:** Translate technical insights into commercial actions
    
    ### 🛠️ Methodology
    
    - **Dataset:** Telco Customer Churn (Kaggle)
    - **Algorithms:** Logistic Regression, Random Forest, XGBoost
    - **Evaluation:** AUC-ROC, Accuracy, Cross-Validation
    - **Explainability:** SHAP values for interpretability
    
    ### 💡 Practical Applications
    
    - **Retention Campaigns:** Focus resources on high-risk customers
    - **Personalized Discounts:** Offer incentives based on churn probability
    - **Service Improvement:** Identify problematic areas of the service
    - **Commercial Strategy:** Develop effective loyalty programs
    
    ### 🚀 Next Steps
    
    - Incorporate real-time data
    - Implement continuous learning
    - Integrate with CRM systems
    - Develop API for external consumption
    """)
    
    st.markdown("---")
    st.subheader("👨‍💻 Developed as a Data Science Portfolio Project")
    st.markdown("""
    **Technologies:** Python, Scikit-learn, XGBoost, Streamlit, SHAP, Plotly
    
    **Purpose:** Demonstrate comprehensive skills in the Data Science project lifecycle
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Developed as a Data Science portfolio project | 
        Using Streamlit, Scikit-learn, and XGBoost</p>
    </div>
    """,
    unsafe_allow_html=True
)