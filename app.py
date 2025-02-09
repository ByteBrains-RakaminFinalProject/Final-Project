#!/usr/bin/env python
# coding: utf-8

# In[19]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

st.sidebar.write("Made by ByteBrains")

