from pathlib import Path

import joblib

import pandas as pd

import streamlit as st

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(

    page_title="Bank Customer Churn",

    page_icon="🏦",

    layout="wide"

)

ROOT = Path(__file__).resolve().parent

MODEL_DIR = ROOT / "Models"

MODEL_FILES = {

    "Logistic Regression": "logistic_regression.joblib",

    "Decision Tree": "decision_tree.joblib",

    "Random Forest": "random_forest.joblib",

    "Extra Trees": "extra_trees.joblib"

}

# ---------------- DESIGN ----------------

st.markdown("""

<style>

.stApp {

    background: linear-gradient(135deg, #F3F6FB, #F9F1F3, #EEF5F3);

}

.block-container {

    max-width: 1150px;

    padding-top: 2rem;

    padding-bottom: 2rem;

}

.hero {

    background: linear-gradient(120deg, #172D4D, #315A83);

    padding: 32px;

    border-radius: 20px;

    color: white;

    margin-bottom: 25px;

    box-shadow: 0 8px 25px rgba(23,45,77,0.12);

}

.hero h1 {

    color: white;

    font-size: 34px;

    margin-bottom: 8px;

}

.hero p {

    color: #DCE8F5;

    font-size: 16px;

}

.section-title {

    color: #203B5B;

    font-size: 23px;

    font-weight: 700;

    margin: 12px 0 15px 0;

}

div[data-testid="stVerticalBlockBorderWrapper"] {

    background: rgba(255,255,255,0.82);

    border: 1px solid #E1E8F0;

    border-radius: 17px;

    padding: 15px;

}

div[data-testid="stTextInput"] input,

div[data-testid="stNumberInput"] input {

    border-radius: 10px;

}

div[data-testid="stSelectbox"] > div > div {

    border-radius: 10px;

}

div[data-testid="stSlider"] {

    padding-top: 8px;

}

.stButton > button,

.stFormSubmitButton > button {

    background: linear-gradient(110deg, #315A83, #477CA8);

    color: white;

    border: none;

    border-radius: 12px;

    padding: 12px 25px;

    font-weight: 700;

    width: 100%;

    transition: 0.2s;

}

.stButton > button:hover,

.stFormSubmitButton > button:hover {

    background: #203B5B;

    color: white;

    border: none;

}

div[data-testid="stMetric"] {

    background: #F0F5FA;

    border-radius: 13px;

    padding: 16px;

}

.footer {

    text-align: center;

    color: #718096;

    font-size: 13px;

    padding-top: 25px;

}

</style>

""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------

@st.cache_resource

def load_models():

    return {

        name: joblib.load(MODEL_DIR / filename)

        for name, filename in MODEL_FILES.items()

        if (MODEL_DIR / filename).exists()

    }

def engineer(data):

    data = data.copy()

    data["Age_Group"] = pd.cut(

        data["Age"],

        [0, 25, 40, 55, 120],

        labels=["18-24", "25-39", "40-54", "55+"],

        right=False

    )

    data["Tenure_Group"] = pd.cut(

        data["Tenure"],

        [-1, 2, 5, 10],

        labels=["0-2", "3-5", "6-10"]

    )

    data["HasBalance"] = (data["Balance"] > 0).astype(int)

    return data

models = load_models()

if not models:

    st.error("Saved models are missing. Run Notebook 2 first.")

    st.stop()

# ---------------- HEADER ----------------

st.html("""

<div class="hero">

    <h1> Bank Customer Churn</h1>

    <p>Smart Customer Prediction System</p>

    <p>Understand customer behavior using Machine Learning.</p>

</div>

""")

# ---------------- CUSTOMER INFORMATION ----------------

st.markdown(

    '<div class="section-title">👤 Customer Information</div>',

    unsafe_allow_html=True

)

with st.form("customer_form"):

    # Personal information

    with st.container(border=True):

        st.markdown("###  Personal Information")

        col1, col2 = st.columns(2)

        with col1:

            customer_name = st.text_input(

                "Customer Name",

                placeholder="Enter customer name"

            )

            geography = st.selectbox(

                "Geography",

                ["France", "Germany", "Spain"]

            )

            gender = st.selectbox(

                "Gender",["Female", "Male"]

            )

        with col2:

            age = st.slider("Age", 18, 92, 38)

            credit = st.slider(

                "Credit Score",

                300,

                850,

                650

            )

            tenure = st.slider(

                "Tenure (Years)",

                0,

                10,

                5

            )

    st.write("")

    # Financial information

    with st.container(border=True):

        st.markdown("###  Financial Information")

        col1, col2 = st.columns(2)

        with col1:

            balance = st.number_input(

                "Account Balance",

                min_value=0.0,

                max_value=300000.0,

                value=50000.0,

                step=1000.0

            )

            salary = st.number_input(

                "Estimated Salary",

                min_value=0.0,

                max_value=250000.0,

                value=100000.0,

                step=1000.0

            )

        with col2:

            products = st.selectbox(

                "Number of Products",

                [1, 2, 3, 4],

                index=1

            )

            card = st.selectbox(

                "Has Credit Card?",

                ["Yes", "No"]

            )

            active = st.selectbox(

                "Is Active Member?",

                ["Yes", "No"]

            )

    st.write("")

    # Model selection

    with st.container(border=True):

        st.markdown("###  Prediction Settings")

        model_name = st.selectbox(

            "Choose a Machine Learning Model",

            list(models.keys())

        )

    st.write("")

    submitted = st.form_submit_button(

        "✨ Predict Customer Churn"

    )

# ---------------- PREDICTION ----------------

if submitted:

    row = pd.DataFrame([{

        "CreditScore": credit,

        "Geography": geography,

        "Gender": gender,

        "Age": age,

        "Tenure": tenure,

        "Balance": balance,

        "NumOfProducts": products,

        "HasCrCard": int(card == "Yes"),

        "IsActiveMember": int(active == "Yes"),

        "EstimatedSalary": salary

    }])

    row = engineer(row)

    model = models[model_name]

    prediction = int(model.predict(row)[0])

    probability = float(

        model.predict_proba(row)[0, 1]

    )

    # Prediction result

    st.write("")

    st.markdown(

        '<div class="section-title">📊 Prediction Result</div>',

        unsafe_allow_html=True

    )

    with st.container(border=True):

        if customer_name.strip():

            st.markdown(f"### Result for {customer_name.strip()}")

        else:

            st.markdown("### Customer Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(

                "Churn Probability",

                f"{probability:.1%}"

            )

        with col2:

            st.metric(

                "Selected Model",

                model_name

            )

        if prediction == 1:

            st.error("⚠️ The customer may leave the bank.")

        else:

            st.success("✅ The customer is likely to stay.")

        st.progress(probability)

        st.caption(

            "The probability is a model estimate, not a certainty."

        )

    # Model comparison

    st.write("")

    st.markdown(

        '<div class="section-title"> Compare All Models</div>',

        unsafe_allow_html=True

    )

    results = []

    for name, current_model in models.items():

        current_prediction = int(

            current_model.predict(row)[0]

        )

        current_probability = float(

            current_model.predict_proba(row)[0, 1]

        )

        results.append({

            "Model": name,

            "Prediction": (

                "May Leave"

                if current_prediction == 1

                else "Likely to Stay"

            ),

            "Churn Probability": (

                f"{current_probability:.2%}")

        })

    with st.container(border=True):

        st.dataframe(

            pd.DataFrame(results),

            use_container_width=True,

            hide_index=True

        )

# ---------------- FOOTER ----------------

st.html("""

<div class="footer">

    Bank Customer Churn Classification |

    Machine Learning Project<br>

    Educational Demonstration Only

</div>

""")