# 💼 Employee Salary Predictor
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Employee Salary Predictor", page_icon="💰")

st.title("💼 Employee Salary Predictor")
st.write("Enter employee details below to get estimated salary!")

@st.cache_data
def load_real_data():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/Salaries.csv"
    response = requests.get(url)
    df = pd.read_csv(StringIO(response.text))
    return df

@st.cache_resource
def train_model(df):
    df = df[['rank', 'discipline', 'yrs.since.phd', 'yrs.service', 'sex', 'salary']]
    df.dropna(inplace=True)
    encoders = {}
    for col in ['rank', 'discipline', 'sex']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    X = df.drop('salary', axis=1)
    y = df['salary']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, encoders

df = load_real_data()
model, encoders = train_model(df)

rank = st.selectbox("Academic Rank", encoders['rank'].classes_)
discipline = st.selectbox("Discipline", encoders['discipline'].classes_)
sex = st.radio("Sex", encoders['sex'].classes_)
phd_years = st.slider("Years Since PhD", 0, 45, 10)
service_years = st.slider("Years of Service", 0, 50, 10)

if st.button("Predict Salary"):
    input_data = {
        'rank': encoders['rank'].transform([rank])[0],
        'discipline': encoders['discipline'].transform([discipline])[0],
        'yrs.since.phd': phd_years,
        'yrs.service': service_years,
        'sex': encoders['sex'].transform([sex])[0]
    }
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Salary: **${prediction:,.2f}**")
