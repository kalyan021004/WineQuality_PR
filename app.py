import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- Page Configuration (Optional but good for premium feel) ---
st.set_page_config(
    page_title="VitiPredict AI",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS for luxury/professional look (basic simulation) ---
st.markdown("""
<style>
    /* Dark background, white text */
    body {
        color: #fff;
        background-color: #1a1a1a; /* Dark background */
    }
    .stApp {
        background-color: #1a1a1a;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: #d4af37; /* Gold accents for titles */
        font-family: 'Inter', sans-serif;
    }
    /* Input fields and select boxes */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > button,
    .stTextInput > div > div > input {
        background-color: #2c2c2c; /* Slightly lighter dark for inputs */
        color: #fff;
        border: 1px solid #722f37; /* Deep wine red border */
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stSelectbox > div > div > button:hover {
        border-color: #d4af37; /* Gold on hover */
    }
    /* Button styling (Predict) */
    .stButton > button {
        background: linear-gradient(to right, #722f37, #d4af37); /* Gradient for Predict button */
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    /* Metric cards styling */
    .stMetric {
        background-color: #2c2c2c;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .stMetric > div > div:first-child {
        color: #d4af37; /* Gold for metric label */
        font-size: 0.9rem;
    }
    .stMetric > div > div:nth-child(2) {
        color: #fff; /* White for metric value */
        font-size: 1.8rem;
        font-weight: bold;
    }
    /* Success message */
    .stSuccess {
        background-color: #1f3b20; /* Darker green for success */
        color: #4caf50;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #4caf50;
    }
    /* General text */
    .stText, .stMarkdown {
        font-family: 'Inter', sans-serif;
        color: #ffffff;
    }
    /* Specifically for the predicted quality */
    .predicted-quality {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-top: 1rem;
    }
    .low-quality { color: #f44336; } /* Red */
    .medium-quality { color: #ff9800; } /* Orange */
    .high-quality { color: #d4af37; } /* Gold */
</style>
""", unsafe_allow_html=True)


st.title("🍷 VitiPredict AI: Wine Quality Prediction")
st.markdown("### Predict whether wine is **Low**, **Medium**, or **High** quality based on chemical properties.")

# --- Model Performance Data ---
performance = {
    "model_personA_multiclass.joblib": {"Accuracy": 0.6177, "Macro-F1": 0.5945},
    "model_personB_multiclass.joblib": {"Accuracy": 0.7163, "Macro-F1": 0.7109},
    "model_personC_multiclass.joblib": {"Accuracy": 0.5889, "Macro-F1": 0.5596},
    "model_personD_multiclass.joblib": {"Accuracy": 0.5680, "Macro-F1": 0.5564}
}

# --- Layout for Prediction Form (Left 50% for inputs, Right 50% for results) ---
col_inputs, col_results = st.columns([0.5, 0.5])

with col_inputs:
    st.subheader("🔬 Analyze Your Wine")

    # Model Selection (placed subtly)
    st.markdown("---")
    st.markdown("#### Model Selection")
    model_file = st.selectbox(
        "Choose an AI Model",
        list(performance.keys()),
        help="Select a trained machine learning model for prediction."
    )
    st.write(f"**Model Accuracy:** `{performance[model_file]['Accuracy']:.2%}`")
    st.write(f"**Macro-F1 Score:** `{performance[model_file]['Macro-F1']:.2%}`")
    st.markdown("---")

    # Load selected model and transformers
    try:
        model_data = joblib.load(model_file)
        model = model_data["model"]
        scaler = model_data["scaler"]
        poly = model_data.get("poly", None)
        selector = model_data.get("selector", None)
        lda = model_data.get("lda", None)
        pca = model_data.get("pca", None)
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_file}' not found. Please ensure all .joblib files are in the script directory.")
        st.stop()


    st.markdown("#### Enter Wine Chemical Measurements")

    # Grouped Input Fields (using columns for 3-column layout)
    # Acidity Properties
    st.markdown("##### Acidity Properties")
    col1, col2, col3 = st.columns(3)
    with col1:
        fixed = st.number_input("Fixed Acidity", 0.0, 20.0, 7.4, help="tartaric acid [g/dm^3]")
    with col2:
        volatile = st.number_input("Volatile Acidity", 0.0, 2.0, 0.7, help="acetic acid [g/dm^3]")
    with col3:
        citric = st.number_input("Citric Acid", 0.0, 2.0, 0.0, help="[g/dm^3]")

    # Sulfur Content
    st.markdown("##### Sulfur Content")
    col1, col2, col3 = st.columns(3)
    with col1:
        free_s = st.number_input("Free Sulfur Dioxide", 0.0, 200.0, 11.0, help="[mg/dm^3]")
    with col2:
        total_s = st.number_input("Total Sulfur Dioxide", 0.0, 500.0, 34.0, help="[mg/dm^3]")
    with col3:
        sulphates = st.number_input("Sulphates", 0.0, 2.0, 0.56, help="potassium sulphate [g/dm^3]")

    # Alcohol & Density
    st.markdown("##### Alcohol & Density")
    col1, col2 = st.columns(2)
    with col1:
        alcohol = st.number_input("Alcohol %", 0.0, 20.0, 9.4, help="[volume %]")
    with col2:
        density = st.number_input("Density", 0.9900, 1.0100, 0.9978, format="%.4f", help="[g/cm^3]")

    # Other Properties
    st.markdown("##### Other Properties")
    col1, col2, col3 = st.columns(3)
    with col1:
        residual = st.number_input("Residual Sugar", 0.0, 40.0, 1.9, help="[g/dm^3]")
    with col2:
        chlorides = st.number_input("Chlorides", 0.0, 1.0, 0.076, help="sodium chloride [g/dm^3]")
    with col3:
        ph = st.number_input("pH", 2.5, 4.5, 3.51, help="pH value")

    # Wine Type (can be placed centrally or with other properties)
    wine_type = st.selectbox("Wine Type", ["red", "white"], help="Is this a Red or White wine?")
    wine_code = 1 if wine_type == "red" else 0

    # Prepare DataFrame for prediction
    df = pd.DataFrame([[fixed, volatile, citric, residual, chlorides, free_s, total_s,
                        density, ph, sulphates, alcohol, wine_code]],
                      columns=["fixed acidity", "volatile acidity", "citric acid",
                               "residual sugar", "chlorides", "free sulfur dioxide",
                               "total sulfur dioxide", "density", "pH", "sulphates",
                               "alcohol", "wine_type_code"])

    # --- FEATURE ENGINEERING MATCHING PERSON MODELS ---
    if "personA" in model_file:
        df["acidity_ratio"] = df["fixed acidity"] / (df["volatile acidity"] + 1e-6)
        df["sulfur_ratio"] = df["total sulfur dioxide"] / (df["free sulfur dioxide"] + 1e-6)
        df["sweetness_index"] = df["residual sugar"] * df["density"]

    if "personB" in model_file:
        base_cols = ['alcohol','sulphates','density','pH','residual sugar']
        for i in range(len(base_cols)):
            for j in range(i+1, len(base_cols)):
                a = base_cols[i]; b = base_cols[j]
                df[f"{a}_x_{b}"] = df[a] * df[b]

    if "personC" in model_file:
        df["alc_over_density"] = df["alcohol"] / (df["density"] + 1e-6)
        df["log_res_sugar"] = np.log1p(df["residual sugar"])
        df["norm_sulphates"] = (df["sulphates"] - df["sulphates"].mean()) / (df["sulphates"].std() + 1e-6)
        df["sulfur_balance"] = df["total sulfur dioxide"] - df["free sulfur dioxide"]

    if "personD" in model_file:
        df["acid_balance"] = (df["fixed acidity"] + df["citric acid"]) / (df["volatile acidity"] + 1e-6)

    # Handle potential NaNs introduced by feature engineering (e.g., division by zero if not handled with 1e-6)
    # Using df.mean() here is a simple placeholder. In a real app, you'd use the mean from training data.
    df = df.fillna(df.mean(numeric_only=True))

    # --- Prediction Button ---
    st.markdown("---")
    predict_button = st.button("Predict Quality")


with col_results:
    st.subheader("📊 Prediction Results")

    if predict_button:
        # --- APPLY SAME TRANSFORMERS USED DURING TRAINING ---
        try:
            X = scaler.transform(df.values)
            if poly:
                X = poly.transform(X)
            if selector:
                X = selector.transform(X)
            if lda:
                X = lda.transform(X)
            if pca:
                X = pca.transform(X)
        except Exception as e:
            st.error(f"Error during feature transformation: {e}. Please check your input data or model setup.")
            st.stop()


        # --- PREDICTION ---
        pred_label_map = {0:"Low Quality 🍂", 1:"Medium Quality 🙂", 2:"High Quality 🌟"}
        class_color_map = {0:"low-quality", 1:"medium-quality", 2:"high-quality"}
        quality_grade_map = {0:"Poor Grade", 1:"Good Grade", 2:"Excellent Grade"}

        try:
            prediction_index = model.predict(X)[0]
            predicted_quality_text = pred_label_map[prediction_index]
            predicted_quality_grade = quality_grade_map[prediction_index]
            predicted_quality_color_class = class_color_map[prediction_index]

            st.markdown(f"<p class='predicted-quality {predicted_quality_color_class}'>{predicted_quality_text}</p>", unsafe_allow_html=True)

            # Confidence (requires predict_proba)
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)[0]
                confidence = probabilities[prediction_index] * 100
                st.metric("Confidence", f"{confidence:.2f}%")
                st.progress(confidence / 100)
            else:
                st.info("Model does not support probability prediction for confidence score.")

            st.markdown(f"**Quality Grade:** <span style='background-color:#722f37; color:white; padding: 5px 10px; border-radius: 5px;'>{predicted_quality_grade}</span>", unsafe_allow_html=True)


            st.markdown("---")
            st.markdown("#### Probability Distribution")
            # Create a simple bar chart for probabilities (if available)
            if hasattr(model, 'predict_proba'):
                prob_df = pd.DataFrame({
                    'Quality': ['Low', 'Medium', 'High'],
                    'Probability': probabilities
                })
                # Using Streamlit's built-in chart for simplicity
                st.bar_chart(prob_df.set_index('Quality'))
            else:
                st.write("Probability distribution not available for this model.")

            st.markdown("---")
            st.markdown("#### Actionable Insights (Example)")
            if prediction_index == 2: # High Quality
                st.success("🌟 This wine has high potential! Consider premium bottling or extended aging.")
            elif prediction_index == 1: # Medium Quality
                st.info("🙂 Good quality. Optimal for general market distribution or blending options.")
            else: # Low Quality
                st.warning("🍂 Suggest reviewing vineyard practices or considering alternative uses (e.g., vinegar).")

            st.markdown("---")
            st.button("Export Prediction Details", help="Download current prediction details as CSV/PDF")


        except Exception as e:
            st.error(f"An error occurred during prediction: {e}. Please check model compatibility or input values.")
    else:
        st.info("Enter the wine parameters on the left and click 'Predict Quality' to see results.")

st.markdown("---")
st.markdown("<footer><p style='text-align:center; color:#722f37;'>VitiPredict AI © 2023</p></footer>", unsafe_allow_html=True)