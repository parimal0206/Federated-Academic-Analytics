import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Student Academic Outcome Predictor",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "random_forest_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "..", "models", "preprocessor.pkl")

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

st.title("🎓 Student Academic Outcome Predictor")
st.write("Enter a few simple student details to predict the student's academic outcome.")
st.info("The model predicts one of three outcomes: Dropout, Enrolled, or Graduate.")

st.divider()

st.subheader("👨‍🎓 Student Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age at Enrollment", min_value=15, max_value=80, value=20)

    gender_label = st.selectbox("Gender", ["Female", "Male"])
    gender = 0 if gender_label == "Female" else 1

    attendance_label = st.selectbox("Attendance", ["Daytime", "Evening"])
    attendance = 1 if attendance_label == "Daytime" else 0

    admission_grade = st.number_input(
        "Admission Grade", min_value=0.0, max_value=200.0, value=120.0
    )

    international_label = st.selectbox("International Student", ["No", "Yes"])
    international = 1 if international_label == "Yes" else 0

with col2:
    first_sem_grade = st.number_input(
        "1st Semester Average Grade",
        min_value=0.0,
        max_value=20.0,
        value=12.0
    )

    second_sem_grade = st.number_input(
        "2nd Semester Average Grade",
        min_value=0.0,
        max_value=20.0,
        value=12.0
    )

    tuition_label = st.selectbox("Tuition Fees Paid", ["Yes", "No"])
    tuition_fees = 1 if tuition_label == "Yes" else 0

    scholarship_label = st.selectbox("Scholarship Holder", ["No", "Yes"])
    scholarship = 1 if scholarship_label == "Yes" else 0

    debtor_label = st.selectbox("Has Outstanding Debt", ["No", "Yes"])
    debtor = 1 if debtor_label == "Yes" else 0

st.caption("ℹ️ Technical dataset features are handled automatically by the application.")

st.divider()

if st.button("🔮 Predict Student Outcome", use_container_width=True):

    # The trained model expects all original UCI features.
    # Advanced features are kept internally with standard default values.
    input_data = pd.DataFrame([{
        "Marital status": 1,
        "Application mode": 1,
        "Application order": 1,
        "Course": 171,
        "Daytime/evening attendance\t": attendance,
        "Previous qualification": 1,
        "Previous qualification (grade)": 120.0,
        "Nacionality": 1,
        "Mother's qualification": 1,
        "Father's qualification": 1,
        "Mother's occupation": 0,
        "Father's occupation": 0,
        "Admission grade": admission_grade,
        "Displaced": 0,
        "Educational special needs": 0,
        "Debtor": debtor,
        "Tuition fees up to date": tuition_fees,
        "Gender": gender,
        "Scholarship holder": scholarship,
        "Age at enrollment": age,
        "International": international,
        "Curricular units 1st sem (credited)": 0,
        "Curricular units 1st sem (enrolled)": 6,
        "Curricular units 1st sem (evaluations)": 6,
        "Curricular units 1st sem (approved)": 5,
        "Curricular units 1st sem (grade)": first_sem_grade,
        "Curricular units 1st sem (without evaluations)": 0,
        "Curricular units 2nd sem (credited)": 0,
        "Curricular units 2nd sem (enrolled)": 6,
        "Curricular units 2nd sem (evaluations)": 6,
        "Curricular units 2nd sem (approved)": 5,
        "Curricular units 2nd sem (grade)": second_sem_grade,
        "Curricular units 2nd sem (without evaluations)": 0,
        "Unemployment rate": 10.0,
        "Inflation rate": 1.0,
        "GDP": 1.5
    }])

    processed_input = preprocessor.transform(input_data)
    prediction = model.predict(processed_input)[0]

    st.subheader("📊 Prediction Result")

    if prediction == "Dropout":
        st.error("⚠️ Predicted Outcome: DROPOUT")
    elif prediction == "Enrolled":
        st.warning("📚 Predicted Outcome: ENROLLED")
    else:
        st.success("🎓 Predicted Outcome: GRADUATE")

    st.write("**Predicted class:**", prediction)
