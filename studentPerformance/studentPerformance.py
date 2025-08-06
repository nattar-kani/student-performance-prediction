import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

def load_model():
    with open("Student-Performance-LR.pkl", 'rb') as file:
        lr, scaler, le = pickle.load(file)
    return lr, scaler, le

def preprocessingInputData(data, scaler, le):
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])[0]
    df = pd.DataFrame([data])
    dftransformed = scaler.transform(df)
    return dftransformed

def predictData(data):
    lr, scaler, le = load_model()
    preprocessed_data = preprocessingInputData(data, scaler, le)
    result = lr.predict(preprocessed_data)
    return result

def main():
    st.title("Student performance prediction")
    st.write("enter your data to get a prediction about your performance")
    hours = st.number_input("Hours studied", min_value = 1, max_value = 15, value = 5)
    prevScore = st.number_input("Previous Score")
    extra = st.selectbox("Extracurricular Activities", ['Yes', 'No'])
    sleepingHours = st.number_input("Sleeping hours")
    qpSolved = st.number_input("Sample question papers solved")
    
    if st.button("Predict your Performance Score"):
        userData = {
            'Hours Studied': hours,
            'Previous Scores': prevScore,
            'Extracurricular Activities': extra,
            'Sleep Hours': sleepingHours,
            'Sample Question Papers Practiced': qpSolved    
        }
        prediction = predictData(userData)
        st.success(f"Prediction result is {prediction}")


if __name__ == "__main__":
    main()