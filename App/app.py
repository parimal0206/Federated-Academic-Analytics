import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Student Academic Outcome Predictor",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------------------------
# Load model and preprocessing pipeline
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "random_forest_model.pkl"
)
PREPROCESSOR_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "preprocessor.pkl"
)
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🎓 Student Academic Outcome Predictor")

st.write(
    "This application predicts a student's academic outcome "
    "using the UCI Student Dropout & Academic Success dataset."
)

st.info(
    "Prediction classes: Dropout, Enrolled, Graduate"
)

st.divider()

# --------------------------------------------------
# Student input
# --------------------------------------------------

st.subheader("Enter Student Information")

col1, col2 = st.columns(2)

with col1:
    marital_status = st.number_input(
        "Marital Status",
        min_value=1,
        value=1
    )

    application_mode = st.number_input(
        "Application Mode",
        min_value=1,
        value=1
    )

    application_order = st.number_input(
        "Application Order",
        min_value=0,
        value=1
    )

    course = st.number_input(
        "Course",
        min_value=1,
        value=171
    )

    attendance = st.number_input(
        "Daytime/Evening Attendance",
        min_value=0,
        max_value=1,
        value=1
    )

    previous_qualification = st.number_input(
        "Previous Qualification",
        min_value=1,
        value=1
    )

    previous_qualification_grade = st.number_input(
        "Previous Qualification Grade",
        min_value=0.0,
        value=120.0
    )

    admission_grade = st.number_input(
        "Admission Grade",
        min_value=0.0,
        value=120.0
    )

    age = st.number_input(
        "Age at Enrollment",
        min_value=15,
        max_value=80,
        value=20
    )

with col2:
    displaced = st.number_input(
        "Displaced",
        min_value=0,
        max_value=1,
        value=0
    )

    educational_special_needs = st.number_input(
        "Educational Special Needs",
        min_value=0,
        max_value=1,
        value=0
    )

    debtor = st.number_input(
        "Debtor",
        min_value=0,
        max_value=1,
        value=0
    )

    tuition_fees = st.number_input(
        "Tuition Fees Up to Date",
        min_value=0,
        max_value=1,
        value=1
    )

    gender = st.number_input(
        "Gender",
        min_value=0,
        max_value=1,
        value=1
    )

    scholarship = st.number_input(
        "Scholarship Holder",
        min_value=0,
        max_value=1,
        value=0
    )

    international = st.number_input(
        "International",
        min_value=0,
        max_value=1,
        value=0
    )

    mothers_qualification = st.number_input(
        "Mother's Qualification",
        min_value=1,
        value=1
    )

    fathers_qualification = st.number_input(
        "Father's Qualification",
        min_value=1,
        value=1
    )

    mothers_occupation = st.number_input(
        "Mother's Occupation",
        min_value=0,
        value=0
    )

    fathers_occupation = st.number_input(
        "Father's Occupation",
        min_value=0,
        value=0
    )

st.divider()

# --------------------------------------------------
# Academic performance
# --------------------------------------------------

st.subheader("Academic Performance")

col3, col4 = st.columns(2)

with col3:
    curricular_1_credited = st.number_input(
        "1st Sem - Credited Units",
        min_value=0,
        value=0
    )

    curricular_1_enrolled = st.number_input(
        "1st Sem - Enrolled Units",
        min_value=0,
        value=6
    )

    curricular_1_evaluations = st.number_input(
        "1st Sem - Evaluations",
        min_value=0,
        value=6
    )

    curricular_1_approved = st.number_input(
        "1st Sem - Approved Units",
        min_value=0,
        value=5
    )

    curricular_1_grade = st.number_input(
        "1st Sem - Grade",
        min_value=0.0,
        max_value=20.0,
        value=12.0
    )

    curricular_1_without_evaluation = st.number_input(
        "1st Sem - Without Evaluation",
        min_value=0,
        value=0
    )

with col4:
    curricular_2_credited = st.number_input(
        "2nd Sem - Credited Units",
        min_value=0,
        value=0
    )

    curricular_2_enrolled = st.number_input(
        "2nd Sem - Enrolled Units",
        min_value=0,
        value=6
    )

    curricular_2_evaluations = st.number_input(
        "2nd Sem - Evaluations",
        min_value=0,
        value=6
    )

    curricular_2_approved = st.number_input(
        "2nd Sem - Approved Units",
        min_value=0,
        value=5
    )

    curricular_2_grade = st.number_input(
        "2nd Sem - Grade",
        min_value=0.0,
        max_value=20.0,
        value=12.0
    )

    curricular_2_without_evaluation = st.number_input(
        "2nd Sem - Without Evaluation",
        min_value=0,
        value=0
    )

st.divider()

# --------------------------------------------------
# Economic indicators
# --------------------------------------------------

st.subheader("Economic Indicators")

col5, col6 = st.columns(2)

with col5:
    unemployment_rate = st.number_input(
        "Unemployment Rate",
        value=10.0
    )

    inflation_rate = st.number_input(
        "Inflation Rate",
        value=1.0
    )

with col6:
    gdp = st.number_input(
        "GDP",
        value=1.5
    )

# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.divider()

if st.button("🔮 Predict Student Outcome", use_container_width=True):

    input_data = pd.DataFrame([{
        'Marital status': marital_status,
        'Application mode': application_mode,
        'Application order': application_order,
        'Course': course,
        'Daytime/evening attendance\t': attendance,
        'Previous qualification': previous_qualification,
        'Previous qualification (grade)': previous_qualification_grade,
        'Nacionality': 1,
        "Mother's qualification": mothers_qualification,
        "Father's qualification": fathers_qualification,
        "Mother's occupation": mothers_occupation,
        "Father's occupation": fathers_occupation,
        'Admission grade': admission_grade,
        'Displaced': displaced,
        'Educational special needs': educational_special_needs,
        'Debtor': debtor,
        'Tuition fees up to date': tuition_fees,
        'Gender': gender,
        'Scholarship holder': scholarship,
        'Age at enrollment': age,
        'International': international,

        'Curricular units 1st sem (credited)': curricular_1_credited,
        'Curricular units 1st sem (enrolled)': curricular_1_enrolled,
        'Curricular units 1st sem (evaluations)': curricular_1_evaluations,
        'Curricular units 1st sem (approved)': curricular_1_approved,
        'Curricular units 1st sem (grade)': curricular_1_grade,
        'Curricular units 1st sem (without evaluations)': curricular_1_without_evaluation,

        'Curricular units 2nd sem (credited)': curricular_2_credited,
        'Curricular units 2nd sem (enrolled)': curricular_2_enrolled,
        'Curricular units 2nd sem (evaluations)': curricular_2_evaluations,
        'Curricular units 2nd sem (approved)': curricular_2_approved,
        'Curricular units 2nd sem (grade)': curricular_2_grade,
        'Curricular units 2nd sem (without evaluations)': curricular_2_without_evaluation,

        'Unemployment rate': unemployment_rate,
        'Inflation rate': inflation_rate,
        'GDP': gdp
    }])

    # Preprocess input
    processed_input = preprocessor.transform(input_data)

    # Prediction
    prediction = model.predict(processed_input)[0]

    # Display result
    st.subheader("Prediction Result")

    if prediction == "Dropout":
        st.error("⚠️ Predicted Outcome: DROPOUT")

    elif prediction == "Enrolled":
        st.warning("📚 Predicted Outcome: ENROLLED")

    else:
        st.success("🎓 Predicted Outcome: GRADUATE")

    st.write("**Predicted class:**", prediction)