import streamlit as st
import pickle
import numpy as np

# Load saved models
with open("loan_approval_model.pkl", "rb") as f:
    log_model = pickle.load(f)

with open("loan_approval_dt_model.pkl", "rb") as f:
    dt_model = pickle.load(f)

with open("loan_approval_rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

st.title("🏦 Loan Approval Prediction App")

st.write("Fill in the details below to check if your loan will be **Approved** or **Rejected**")

# Layout with 2 columns
col1, col2 = st.columns(2)

with col1:
    income_annum = st.number_input("Annual Income ", min_value=10000, max_value=10000000, step=1000)
    loan_amount = st.number_input("Loan Amount ", min_value=1000, max_value=5000000, step=500)
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 60, 120, 180, 240, 300, 360])

with col2:
    cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900, step=1)
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, step=1)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

# Encode categorical features
education_val = 1 if education == "Graduate" else 0
self_employed_val = 1 if self_employed == "Yes" else 0

# Model selection
model_choice = st.selectbox("Select Model", ["Logistic Regression", "Decision Tree", "Random Forest"])
if model_choice == "Logistic Regression":
    model = log_model
elif model_choice == "Decision Tree":
    model = dt_model
else:
    model = rf_model




# Prediction
if st.button("🔍 Predict Loan Status"):
    input_data = np.array([[income_annum, loan_amount, loan_term, cibil_score,
                            no_of_dependents, education_val, self_employed_val] + [0]*4])  
    
    prediction = model.predict(input_data)[0]
    
    if prediction == 0:
        st.error("❌ Loan Rejected")
    else:
        st.success("✅ Loan Approved")


# Remove side padding
st.markdown(
    """
    <style>
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            
            padding-bottom: 1rem;
            max-width: 80%;
        }
    </style>
    """,
    unsafe_allow_html=True
)
