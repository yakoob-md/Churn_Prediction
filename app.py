import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
import pickle

print("hellow wolrd")

# loading all the trained models and scaled models
model = load_model("model.h5")

with open("label_encoder_gender.pkl","rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geo.pkl","rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("scaler.pkl","rb") as file:
    scaler = pickle.load(file)

# creating ui for the streamlit app

st.title("Customer Churn Prediction")

# inputs
geography = st.selectbox("Geogrpahy",onehot_encoder_geo.categories_[0])
# for slect boc u need to give an iterbale like a list 

gender = st.selectbox("Gender",label_encoder_gender.classes_)

age = st.slider("Age",18,100)
balance = st.number_input("balance:")
credit_score = st.number_input("credit_score:")
est_salary = st.number_input("estimated_salary")
tenure =st.slider("tenure",0,10)
num_of_products = st.slider("num of products",1,4)
has_cr_card = st.selectbox("Has Credit Card",[0,1])
is_active = st.selectbox("is active member",[0,1])

# these are values u cannot use them directly to predict as u need to 
# convert it into a dataframe
# so first convert it in to a dic then to data frame

# Build dict
input_data = {
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active],
    "EstimatedSalary": [est_salary]
}

# One-hot encode geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded, 
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Create input_df
input_df = pd.DataFrame(input_data)
input_df = pd.concat([input_df, geo_encoded_df], axis=1)

# ✅ Reorder columns to match training order
input_df = input_df[scaler.feature_names_in_]

# Scale input
scaled_input_df = scaler.transform(input_df)

# Prediction
prediction = model.predict(scaled_input_df)
probability = prediction[0][0]

if probability > 0.5:
    st.write(f"🚨 The customer is likely to leave ({probability:.2f})")
else:
    st.write(f"✅ The customer is likely to stay ({probability:.2f})")










