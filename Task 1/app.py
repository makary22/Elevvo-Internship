import streamlit as st 
import pandas as pd 
import pickle 
import numpy as np
import warnings
from sklearn.linear_model import LinearRegression





model = pickle.load(open("student_model.pkl", "rb"))
st.title("📊 Student Score Prediction App")

hours = st.number_input("Enter study hours:", min_value=0.0, max_value=44.0, step=0.5)
attendance = st.number_input("Enter attendance (%):", min_value=60.0, max_value=100.0, step=1.0)
previous = st.number_input("Enter previous scores:", min_value=50.0, max_value=100.0, step=1.0)
tutoring = st.number_input("Enter number of tutoring sessions:", min_value=0, max_value=8, step=1)

if st.button("Predict"):
    
    input_features = np.array([[hours, attendance, previous, tutoring]])
    prediction = model.predict(input_features)[0]
    st.write("Model used: **Linear Regression**")
    st.success(f"Predicted Score: {prediction:.2f}")





