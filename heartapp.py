import streamlit as st
import pickle
import numpy as np
import sklearn


# Load the trained model
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create a Streamlit UI
st.title('Stroke Prediction App')


gender = st.selectbox("Gender", [0, 1])
age = st.slider("Age", 0, 100, 50)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
smoking_status = st.selectbox("Smoking Status", [0, 1])
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
ever_married = st.selectbox("Ever Married", [0, 1])
avg_glucose_level = st.number_input("avg_glucose_level", min_value=0.0, max_value=300.0, value=150.0)

# Convert gender to binary (0 for Male, 1 for Female)
gender = 0 if gender == "Female" else 1
# if sex == "Female":
#     sex = 0
# else:
#     sex = 1
#ever_married_mapping = {"yes":1, "no":0}
#evermarried = ever_married_mapping[ever_married]
#smoking_status_mapping = {'formerly smoked':1, 'never smoked':2,'smokes':3}
#smoking_status = smoking_status_mapping[smoking_status]

inputs = [[gender,age,hypertension,heart_disease,smoking_status,bmi,ever_married,avg_glucose_level]]

# Make prediction
if st.button('Predict'):
    result = model.predict(inputs)
    updated_res = result.flatten().astype(int)
    if updated_res == 0:
       st.write("Not very Proabable you will have a stoke soon but still take good care of yourself regardless")
    else:
       st.write("It is Probable you might have a stroke soon therfore you should take better care of yourself")
   




