#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Menampilkan direktori tempat file disimpan
st.sidebar.write(f"Current Directory: {os.getcwd()}")

# Load dataset
def load_data():
    df = pd.read_csv("data_final.csv")  # Sesuaikan dengan path dataset Anda
    return df

df = load_data()

# Mapping label asli dari data yang telah dienkode
label_mapping = {
    "Gender": {1: "Male", 0: "Female"},
    "Attrition": {0: "No", 1: "Yes"},
    "Department": {0: "Sales", 1: "R&D", 2: "HR"},
    "Overtime": {0: "No", 1: "Yes"},
    "MaritalStatus": {1: "Single", 0: "Married", 2: "Divorced"}
}

# Ubah kembali nilai yang sudah dienkode menjadi label asli
for column, mapping in label_mapping.items():
    if column in df.columns:
        df[column] = df[column].map(mapping)

# Sidebar
st.sidebar.title("HR Attrition Dashboard")
menu = st.sidebar.radio("Pilih Menu:", ["Home", "Department"])

st.title("HR Attrition Dashboard")

if menu == "Home":
    st.subheader("Dashboard Karyawan Berdasarkan Attrition")
    attrition_status = st.selectbox("Pilih Status Attrition:", df["Attrition"].unique())
    filtered_df = df[df["Attrition"] == attrition_status]
    
    # Visualisasi OverTime
    st.subheader("Distribusi OverTime")
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_df, x='Overtime', ax=ax, hue='Overtime', palette='coolwarm', legend=False)
    st.pyplot(fig)
    
    # Visualisasi Marital Status
    st.subheader("Distribusi Marital Status")
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_df, x='MaritalStatus', ax=ax, hue='MaritalStatus', palette='viridis', legend=False)
    st.pyplot(fig)
    
    # Visualisasi Total Working Years
    st.subheader("Distribusi Total Working Years")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df['TotalWorkingYears'], bins=20, kde=True, ax=ax, color='blue')
    st.pyplot(fig)
    
    # Visualisasi Age
    st.subheader("Distribusi Usia Karyawan")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df['Age'], bins=20, kde=True, ax=ax, color='green')
    st.pyplot(fig)
    
    # Visualisasi Job Satisfaction dengan nilai 4 berdasarkan attrition
    st.subheader("Karyawan dengan Job Satisfaction Level 4 per Attrition")
    job_satisfaction_4 = df[df['JobSatisfaction'] == 4].groupby('Attrition').size().reset_index(name='Total')
    st.dataframe(job_satisfaction_4)
    
    job_satisfaction_4_filtered = filtered_df[filtered_df['JobSatisfaction'] == 4]
    st.write(f"Total Karyawan dengan Job Satisfaction 4 ({attrition_status}): {len(job_satisfaction_4_filtered)}")
    st.dataframe(job_satisfaction_4_filtered[['Age', 'Department', 'Gender', 'MaritalStatus', 'TotalWorkingYears']])

elif menu == "Department":
    st.subheader("Attrition Berdasarkan Department")
    departments = df['Department'].unique()
    selected_department = st.selectbox("Pilih Departemen:", departments)
    filtered_df = df[df['Department'] == selected_department]
    
    # Grafik jumlah gender per department
    st.subheader("Distribusi Gender per Department")
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_df, x='Gender', ax=ax, palette='coolwarm')
    st.pyplot(fig)
    
    # Grafik rata-rata pendapatan bulanan per department
    st.subheader("Rata-rata Monthly Income per Department")
    avg_income = df.groupby('Department')['MonthlyIncome'].mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(data=avg_income, x='Department', y='MonthlyIncome', ax=ax, palette='viridis')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Grafik jumlah attrition per department
    st.subheader("Attrition per Department")
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_df, x='Attrition', ax=ax)
    st.pyplot(fig)

elif menu == "Prediction" :

    # Load Model Random Forest
    model_RF = joblib.load('best_model_catboost')
    
    # Streamlit App
    st.cache_data.clear()
    st.title('Employee Attrition Prediction')
    st.markdown(
        """
        <div style="background-color:#000000; padding:10px; border-radius:5px">
            <h4 style="color:#faf7f7;"> This app predicts employee attrition based on key work-related factors 🚀. Enter the details and get an instant prediction!</h4>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Encoding Mapping
    marital_status_options = ['Single', 'Married', 'Divorced']
    overtime_options = ['No', 'Yes']
    
    # Input Form
    with st.container():
        st.header('Enter Employee Details')
        col1, col2 = st.columns(2)
        with col1:
            marital = st.selectbox("Marital Status", marital_status_options)
            overtime_status = st.selectbox("Overtime", overtime_options)
            total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=5)
            avg_hours_per_day = st.number_input("Average Hours Per Day", min_value=0, max_value=24, value=8)
    
        with col2:
            age = st.number_input("Age", min_value=18, max_value=65, value=30)
            total_hours_per_year = st.number_input("Total Hours Per Year", min_value=0, max_value=8760, value=2000)
            years_at_company = st.number_input("Years At Company", min_value=0, max_value=40, value=5)
            monthly_income = st.number_input("Monthly Income", min_value=0, max_value=100000, value=5000)
            years_with_curr_manager = st.number_input("Years With Current Manager", min_value=0, max_value=40, value=3)
            job_satisfaction_4 = st.selectbox("Job Satisfaction (1 to 4)", [1, 2, 3, 4])
    
    # Encode Input Data
    encoded_input = {
        'MaritalStatus_1': [1 if marital == "Married" else 0],
        'Overtime': [1 if overtime_status == "Yes" else 0],
        'TotalWorkingYears': [total_working_years],
        'avg_hours_per_day': [avg_hours_per_day],
        'Age': [age],
        'total_hours_per_year': [total_hours_per_year],
        'YearsAtCompany': [years_at_company],
        'MonthlyIncome': [monthly_income],
        'YearsWithCurrManager': [years_with_curr_manager],
        'JobSatisfaction_4.0': [1 if job_satisfaction_4 == 4 else 0]
    }
    
    df = pd.DataFrame.from_dict(encoded_input)
    
    # Sidebar for Prediction
    with st.sidebar:
        st.write("# Employee Attrition Prediction")
        st.info("The prediction is based on a machine learning model")
        button = st.button("Predict", type='primary')
        
        if button:
            st.markdown("---")
            prediction = model_RF.predict(df)
            result = "Yes, employee is likely to leave" if prediction[0] == 1 else "No, employee is likely to stay"
            st.write(f"## Prediction: {result}")

st.sidebar.write("Made by ByteBrains")

