# Core Packages
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Preprocessing function
def preprocess_data(data):
    to_numeric = {
        'Male': 1, 'Female': 2,
        'Yes': 1, 'No': 2,
        'Graduate': 1, 'Not Graduate': 2,
        'Urban': 3, 'Semiurban': 2, 'Rural': 1,
        'Y': 1, 'N': 0,
        '3+': 3
    }
    data = data.applymap(lambda label: to_numeric.get(label) if label in to_numeric else label)
    data['Dependents'] = pd.to_numeric(data['Dependents'])
    return data

# Load dataset
data = pd.read_csv('Microfinance_Dataset.csv')

# Data cleaning
data['LoanAmount'].fillna(data['LoanAmount'].mean(), inplace=True)
data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
data['Married'].fillna(data['Married'].mode()[0], inplace=True)
data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)
data = data.drop(['Loan_ID'], axis=1)

# Preprocess dataset
data = preprocess_data(data)

# Split the data
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
RF = RandomForestClassifier(random_state=42)
RF.fit(X_train, y_train)
y_predict = RF.predict(X_test)

# Calculate accuracy
RF_SC = accuracy_score(y_predict, y_test)
classification_rep = classification_report(y_test, y_predict)

# Function to predict loan status
def predict_loan_status(data):
    data = np.array(data).reshape(1, -1)
    prediction = RF.predict(data)
    return 'Approved' if prediction[0] == 1 else 'Rejected'

# Dashboard Page
def show_dashboard():
    st.title("Dashboard")

    # Metrics
    total_data = len(data)
    total_loan_amount = data['LoanAmount'].sum()
    total_approved = len(data[data['Loan_Status'] == 1])
    total_rejected = len(data[data['Loan_Status'] == 0])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Number of Data", total_data)
    with col2:
        st.metric("Total Loan Amount", f"${total_loan_amount:,.2f}")
    with col3:
        st.metric("Total Approved", total_approved)
    with col4:
        st.metric("Total Rejected", total_rejected)

    # Bar charts
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    sns.countplot(x="Gender", data=data, ax=axs[0, 0], palette="viridis")
    axs[0, 0].set_title("Gender Distribution")

    sns.countplot(x="Married", data=data, ax=axs[0, 1], palette="viridis")
    axs[0, 1].set_title("Marital Status Distribution")

    sns.countplot(x="Dependents", data=data, ax=axs[1, 0], palette="viridis")
    axs[1, 0].set_title("Dependents Distribution")

    sns.countplot(x="Education", data=data, ax=axs[1, 1], palette="viridis")
    axs[1, 1].set_title("Education Distribution")

    st.pyplot(fig)

    col5, col6 = st.columns((1, 2))

    with col5:
        # Pie chart
        fig, ax = plt.subplots()
        property_counts = data['Property_Area'].value_counts()
        ax.pie(property_counts, labels=property_counts.index, autopct='%1.1f%%', colors=sns.color_palette("viridis", len(property_counts)))
        ax.set_title("Property Area Distribution")
        st.pyplot(fig)

    with col6:
        # Summary of specific columns
        st.write("Summary of Applicant and Coapplicant Income, Loan Amount, and Loan Term")
        st.write(data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']].describe())

    # Display model accuracy
    st.write(f"Model Accuracy: {round(RF_SC*100, 2)}%")

# Prediction Page
def show_prediction():
    st.title("Loan Status Prediction")

    # Input fields
    gender = st.selectbox("Gender", ("Male", "Female"))
    married = st.selectbox("Married", ("Yes", "No"))
    dependents = st.selectbox("Dependents", ("0", "1", "2", "3+"))
    education = st.selectbox("Education", ("Graduate", "Not Graduate"))
    self_employed = st.selectbox("Self Employed", ("Yes", "No"))
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_amount_term = st.number_input("Loan Amount Term", min_value=0)
    credit_history = st.selectbox("Credit History", (1, 0))
    property_area = st.selectbox("Property Area", ("Urban", "Semiurban", "Rural"))

    # Convert input to numerical format
    gender = 1 if gender == 'Male' else 2
    married = 1 if married == 'Yes' else 2
    education = 1 if education == 'Graduate' else 2
    self_employed = 1 if self_employed == 'Yes' else 2
    dependents = 3 if dependents == '3+' else int(dependents)
    property_area = 3 if property_area == 'Urban' else (2 if property_area == 'Semiurban' else 1)

    # Collect user input into a list
    user_input = [gender, married, dependents, education, self_employed, applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history, property_area]

    # Predict button
    if st.button("Predict"):
        result = predict_loan_status(user_input)
        st.write(f"The loan status is: {result}")

# Navigation Menu
with st.sidebar:
    selected = option_menu(
        "Main Menu",
        ["Dashboard", "Prediction"],
        icons=["house", "bar-chart-line", "clipboard-data"],
        menu_icon="cast",
        default_index=0,
    )

# Show the selected page
if selected == "Dashboard":
    show_dashboard()
elif selected == "Prediction":
    show_prediction()
