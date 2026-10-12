
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("credit_risk_before_loan.pkl")

model = load_model()

st.title("💳 Credit Risk Management")
st.subheader("Pre-Loan Default Risk Prediction")

st.write(
    "Estimate a borrower's default risk using information "
    "intended to be available before loan approval."
)

st.info(
    "This tool estimates default risk. It does not automatically "
    "approve or reject a loan."
)

with st.form("borrower_form"):

    st.header("Borrower Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age (years)", 18, 100, 25
        )

        income = st.number_input(
            "Annual income ($)",
            min_value=1,
            value=50000,
            step=1000
        )

        employment = st.number_input(
            "Employment length (years)",
            min_value=0.0,
            max_value=60.0,
            value=3.0,
            step=1.0
        )

        home = st.selectbox(
            "Home ownership",
            ["RENT", "OWN", "MORTGAGE", "OTHER"]
        )

        credit_history = st.number_input(
            "Credit history length (years)",
            min_value=0,
            max_value=60,
            value=5
        )

    with col2:
        purpose = st.selectbox(
            "Loan purpose",
            [
                "PERSONAL",
                "EDUCATION",
                "MEDICAL",
                "VENTURE",
                "HOMEIMPROVEMENT",
                "DEBTCONSOLIDATION"
            ]
        )

        loan_amount = st.number_input(
            "Requested loan amount ($)",
            min_value=500,
            value=10000,
            step=500
        )

        previous_default = st.selectbox(
            "Previous default on credit file?",
            ["N", "Y"]
        )

        st.caption(
            "Loan-to-income ratio is calculated automatically."
        )

    submitted = st.form_submit_button(
        "Estimate Default Risk",
        type="primary",
        use_container_width=True
    )


if submitted:
    loan_percent_income = loan_amount / income

    input_data = pd.DataFrame([{
        "person_age": age,
        "person_income": income,
        "person_home_ownership": home,
        "person_emp_length": employment,
        "loan_intent": purpose,
        "loan_amnt": loan_amount,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": previous_default,
        "cb_person_cred_hist_length": credit_history
    }])

    try:
        probabilities = model.predict_proba(input_data)[0]

        # Probability of each class
        default_probability = probabilities[1] * 100
        repayment_probability = probabilities[0] * 100

        st.divider()
        st.header("💳 Loan Repayment Assessment")

        st.metric(
            "Chance of Repaying the Loan",
            f"{repayment_probability:.1f}%"
        )

        st.progress(float(repayment_probability / 100))

        st.success(
            f"The user has an estimated "
            f"{repayment_probability:.1f}% probability "
            f"of being able to repay the loan."
        )

        st.caption(
            f"Estimated probability of default: "
            f"{default_probability:.1f}%"
        )

        st.warning(
            "This is a model-based estimate, not a guarantee "
            "of repayment or a loan approval decision."
        )

        with st.expander("View Borrower Details"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
