import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

def load_models():
    model = tf.keras.models.load_model('model.h5')
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, label_encoder_gender, onehot_encoder_geo, scaler

def predict_churn(data, model, scaler):
    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)
    return prediction[0][0]

def main():
    st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
    
    st.title('🔄 Customer Churn Prediction')
    st.markdown("---")

    # Load models
    model, label_encoder_gender, onehot_encoder_geo, scaler = load_models()

    # Create two columns for input fields
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Demographics")
        geography = st.selectbox('📍 Geography', onehot_encoder_geo.categories_[0])
        gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
        age = st.slider('🎂 Age', 18, 92, 30)
        
    with col2:
        st.subheader("Account Information")
        balance = st.number_input('💰 Balance', min_value=0.0, format="%.2f")
        credit_score = st.number_input('📊 Credit Score', min_value=300, max_value=850, value=650)
        estimated_salary = st.number_input('💵 Estimated Salary', min_value=0.0, format="%.2f")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Product Details")
        tenure = st.slider('📅 Tenure (years)', 0, 10, 5)
        num_of_products = st.slider('🏦 Number of Products', 1, 4, 1)
        
    with col4:
        st.subheader("Customer Status")
        has_cr_card = st.selectbox('💳 Has Credit Card', ['No', 'Yes'], format_func=lambda x: '✓ Yes' if x == 'Yes' else '✗ No')
        is_active_member = st.selectbox('✨ Is Active Member', ['No', 'Yes'], format_func=lambda x: '✓ Yes' if x == 'Yes' else '✗ No')

    # Convert Yes/No to 1/0
    has_cr_card = 1 if has_cr_card == 'Yes' else 0
    is_active_member = 1 if is_active_member == 'Yes' else 0

    # Prepare input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode Geography
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    if st.button('🔮 Predict Churn', use_container_width=True):
        prediction_proba = float(predict_churn(input_data, model, scaler))
        col1, col2 = st.columns([2, 1])
        with col1:
            st.progress(prediction_proba)
        with col2:
            st.metric("Churn Probability", f"{prediction_proba:.1%}")
        
        st.markdown("---")
        st.subheader("Prediction Results")
        
        # Create a progress bar for churn probability
        col1, col2 = st.columns([2, 1])
        with col1:
            st.progress(prediction_proba)
        with col2:
            st.metric("Churn Probability", f"{prediction_proba:.1%}")

        # Show recommendation based on prediction
        if prediction_proba > 0.5:
            st.error("⚠️ High Risk of Churn! Recommended Actions:")
            st.markdown("""
                - Contact customer for feedback
                - Review pricing and product offerings
                - Consider loyalty program enrollment
                - Offer personalized retention incentives
            """)
        else:
            st.success("✅ Low Risk of Churn! Recommended Actions:")
            st.markdown("""
                - Continue monitoring customer satisfaction
                - Consider upsell opportunities
                - Encourage product exploration
                - Maintain regular communication
            """)

if __name__ == "__main__":
    main()