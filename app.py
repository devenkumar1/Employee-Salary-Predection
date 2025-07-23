import streamlit as st
import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("Employee Salary Prediction App")
st.write("Fill the details below to predict estimated annual salary.")

# Input fields
age = st.number_input("Age", 18, 65, 30)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
experience = st.slider("Years of Experience", 0, 40, 5)
job_title = st.selectbox("Job Title", ["Software Engineer", "Manager", "Analyst", "Clerk", "Other"])
department = st.selectbox("Department", ["IT", "HR", "Finance", "Admin", "Sales"])
location = st.selectbox("Location", ["Delhi", "Mumbai", "Bangalore", "Chennai", "Other"])
company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])
performance_rating = st.slider("Performance Rating", 1, 5, 3)
work_hours = st.slider("Work Hours per Week", 20, 80, 40)
distance = st.slider("Distance from Home (in km)", 0, 100, 10)
training_hours = st.slider("Training Hours per Year", 0, 200, 20)
# certifications = st.slider("No. of Certifications", 0, 10, 1)

# Label Encoding map (must match training)
gender_map = {"Male": 0, "Female": 1, "Other": 2}
education_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
job_map = {"Software Engineer": 0, "Manager": 1, "Analyst": 2, "Clerk": 3, "Other": 4}
dept_map = {"IT": 0, "HR": 1, "Finance": 2, "Admin": 3, "Sales": 4}
loc_map = {"Delhi": 0, "Mumbai": 1, "Bangalore": 2, "Chennai": 3, "Other": 4}
company_map = {"Small": 0, "Medium": 1, "Large": 2}

# Encode user input
input_data = np.array([[
    age,
    gender_map[gender],
    education_map[education],
    experience,
    job_map[job_title],
    dept_map[department],
    loc_map[location],
    company_map[company_size],
    performance_rating,
    work_hours,
    distance,
    training_hours
]])

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Salary"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"Estimated Annual Salary: ${int(prediction):,}")
