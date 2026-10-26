import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st


# ============================================================
# 1. DASHBOARD CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Telecom Churn Intelligence",
    page_icon="📊",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "churn_model.joblib"
CONFIG_PATH = PROJECT_DIR / "models" / "model_config.json"
DATA_PATH = PROJECT_DIR / "data" / "raw" / "Telco_customer_churn.xlsx"

FEATURE_COLUMNS = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Monthly Charges",
    "Total Charges",
]

NUMERIC_COLUMNS = [
    "Senior Citizen",
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
]


# ============================================================
# 2. LOAD MODEL AND CONFIGURATION
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Model configuration not found: {CONFIG_PATH}"
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def get_background_data():
    """Load valid representative rows for SHAP explanations."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_excel(DATA_PATH)

    missing_columns = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    background = df[FEATURE_COLUMNS].copy()

    # Senior Citizen may be stored as Yes/No in some dataset copies,
    # although the trained model expects numeric 1/0 values.
    senior_values = background["Senior Citizen"].astype(str).str.strip().str.lower()
    senior_values = senior_values.replace({
        "yes": "1", "y": "1", "true": "1",
        "no": "0", "n": "0", "false": "0",
    })
    background["Senior Citizen"] = pd.to_numeric(senior_values, errors="coerce")

    for column in [c for c in NUMERIC_COLUMNS if c != "Senior Citizen"]:
        background[column] = pd.to_numeric(background[column], errors="coerce")

    categorical_columns = [c for c in FEATURE_COLUMNS if c not in NUMERIC_COLUMNS]
    for column in categorical_columns:
        background[column] = background[column].where(
            background[column].notna(), np.nan
        )

    # Fill missing values rather than dropping every row that has any missing field.
    for column in NUMERIC_COLUMNS:
        median = background[column].median()
        if pd.isna(median):
            if column == "Senior Citizen":
                median = 0.0
            else:
                sample_values = df[column].dropna().astype(str).head(5).tolist()
                raise ValueError(
                    f"No usable numeric values found for '{column}'. "
                    f"Example source values: {sample_values}"
                )
        background[column] = background[column].fillna(median)

    for column in categorical_columns:
        modes = background[column].mode(dropna=True)
        fill_value = modes.iloc[0] if not modes.empty else "Unknown"
        background[column] = background[column].fillna(fill_value).astype(str)

    if background.empty:
        raise ValueError("The dataset contains no rows for SHAP background data.")

    return background.sample(
        n=min(50, len(background)),
        random_state=42,
    ).reset_index(drop=True)


model = load_model()
config = load_config()
THRESHOLD = float(config["decision_threshold"])

if "Yes" not in list(model.classes_):
    st.error("The saved model does not contain the expected 'Yes' class.")
    st.stop()


# ============================================================
# 3. SHAP EXPLANATION FUNCTION
# ============================================================

def explain_churn(model, customer, background):
    """
    Explain a customer's churn probability using SHAP on the model's
    numerically transformed features, then group contributions by input field.
    """
    if "preprocessor" not in model.named_steps or "model" not in model.named_steps:
        raise ValueError(
            "Expected a scikit-learn pipeline with 'preprocessor' and 'model' steps."
        )

    customer_input = customer[FEATURE_COLUMNS].copy()
    background_input = background[FEATURE_COLUMNS].copy()

    # Normalize Senior Citizen whether it arrives as 1/0 or Yes/No.
    for frame in (customer_input, background_input):
        senior_values = frame["Senior Citizen"].astype(str).str.strip().str.lower()
        senior_values = senior_values.replace({
            "yes": "1", "y": "1", "true": "1",
            "no": "0", "n": "0", "false": "0",
        })
        frame["Senior Citizen"] = pd.to_numeric(senior_values, errors="coerce")
        for column in [c for c in NUMERIC_COLUMNS if c != "Senior Citizen"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    categorical_columns = [c for c in FEATURE_COLUMNS if c not in NUMERIC_COLUMNS]
    for column in categorical_columns:
        customer_input[column] = customer_input[column].astype(str)
        background_input[column] = background_input[column].astype(str)

    if customer_input[NUMERIC_COLUMNS].isna().any().any():
        raise ValueError("The customer profile contains missing numeric values.")
    if background_input.empty:
        raise ValueError("The SHAP background dataset is empty.")

    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]

    background_encoded = preprocessor.transform(background_input)
    customer_encoded = preprocessor.transform(customer_input)

    # SHAP's tree explainer expects numeric encoded features, not raw strings.
    if hasattr(background_encoded, "toarray"):
        background_encoded = background_encoded.toarray()
    if hasattr(customer_encoded, "toarray"):
        customer_encoded = customer_encoded.toarray()

    transformed_names = list(preprocessor.get_feature_names_out())
    background_encoded = pd.DataFrame(
        background_encoded, columns=transformed_names
    )
    customer_encoded = pd.DataFrame(
        customer_encoded, columns=transformed_names
    )

    # A small background sample keeps the probability-based explanation manageable.
    explainer = shap.TreeExplainer(
        estimator,
        data=background_encoded,
        feature_perturbation="interventional",
        model_output="probability",
    )
    explanation = explainer(customer_encoded)

    values = np.asarray(explanation.values)
    if values.ndim == 3:
        # Some SHAP versions return [rows, features, classes].
        class_index = list(model.classes_).index("Yes")
        values = values[0, :, class_index]
    elif values.ndim == 2:
        values = values[0]
    else:
        values = values.reshape(-1)

    if len(values) != len(transformed_names):
        raise ValueError(
            "SHAP returned an unexpected number of feature contributions."
        )

    # Combine one-hot encoded columns back into the original user-facing fields.
    grouped = {column: 0.0 for column in FEATURE_COLUMNS}
    for transformed_name, contribution in zip(transformed_names, values):
        matched_column = None
        for original_column in FEATURE_COLUMNS:
            if transformed_name == f"num__{original_column}":
                matched_column = original_column
                break
            if transformed_name.startswith(f"cat__{original_column}_"):
                matched_column = original_column
                break
        if matched_column is None:
            # Handle transformers whose naming convention omits num__/cat__.
            for original_column in FEATURE_COLUMNS:
                if transformed_name.endswith(original_column):
                    matched_column = original_column
                    break
        if matched_column is not None:
            grouped[matched_column] += float(contribution)

    results = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "SHAP contribution": [grouped[c] for c in FEATURE_COLUMNS],
        "Customer value": customer_input.iloc[0][FEATURE_COLUMNS].astype(str).values,
    })

    results["Direction"] = np.where(
        results["SHAP contribution"] > 0,
        "Higher predicted risk",
        np.where(
            results["SHAP contribution"] < 0,
            "Lower predicted risk",
            "Little or no contribution",
        ),
    )
    results["Absolute contribution"] = results["SHAP contribution"].abs()
    return results.sort_values(
        "Absolute contribution", ascending=False
    ).reset_index(drop=True)


# ============================================================
# 4. DASHBOARD HEADER
# ============================================================

st.title("📊 Telecom Churn Intelligence Platform")

st.write(
    "An interactive machine-learning dashboard for estimating "
    "telecom customer churn risk and understanding model predictions."
)

st.caption(
    "Academic project using historical telecom customer data. "
    "Predictions are estimates, not guarantees or proof of causation."
)

st.divider()


# ============================================================
# 5. CUSTOMER INPUT FORM
# ============================================================

st.header("👤 Customer Profile")

st.write(
    "Enter a hypothetical customer's service, contract, and billing details."
)

with st.form("customer_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender", ["Male", "Female"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen", [0, 1]
        )

        partner = st.selectbox(
            "Has Partner?", ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Has Dependents?", ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12,
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"],
        )

    with col2:
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"],
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"],
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"],
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"],
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"],
        )

    with col3:
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"],
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"],
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"],
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=1.0,
        )

        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=20000.0,
            value=840.0,
            step=10.0,
        )

    submitted = st.form_submit_button(
        "Predict Churn Risk",
        type="primary",
        width="stretch",
    )


# ============================================================
# 6. PREDICTION AND EXPLANATION RESULTS
# ============================================================

if submitted:

    customer = pd.DataFrame([{
        "Gender": gender,
        "Senior Citizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure Months": tenure,
        "Phone Service": phone_service,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet_service,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Contract": contract,
        "Paperless Billing": paperless_billing,
        "Payment Method": payment_method,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges,
    }], columns=FEATURE_COLUMNS)

    # Predict churn probability.
    churn_index = list(model.classes_).index("Yes")

    probability = float(
        model.predict_proba(customer)[0, churn_index]
    )

    prediction = (
        "Yes" if probability >= THRESHOLD else "No"
    )

    # Display prediction.
    st.divider()
    st.header("🔍 Prediction Results")

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric(
            "Estimated Churn Probability",
            f"{probability:.1%}",
        )

    with metric2:
        st.metric(
            "Decision Threshold",
            f"{THRESHOLD:.0%}",
        )

    st.progress(probability)

    if prediction == "Yes":
        st.error(
            "⚠️ The model classifies this customer as AT RISK of churning."
        )
    else:
        st.success(
            "✅ The model classifies this customer as NOT AT RISK "
            "under the current threshold."
        )

    st.caption(
        "The risk label is determined by comparing the estimated "
        "probability with the configured decision threshold."
    )

    with st.expander("View customer data sent to the model"):
        st.dataframe(
            customer,
            hide_index=True,
            width="stretch",
        )

    # --------------------------------------------------------
    # SHAP EXPLANATIONS
    # --------------------------------------------------------

    st.divider()
    st.header("🧠 Why did the model make this prediction?")

    st.write(
        "SHAP estimates how each customer attribute contributes "
        "to the model's predicted churn probability relative to "
        "a representative background sample."
    )

    st.caption(
        "A positive contribution pushes predicted churn probability "
        "higher; a negative contribution pushes it lower. These are "
        "model explanations, not proof of real-world causes."
    )

    with st.spinner(
        "Calculating individual customer explanations..."
    ):
        try:
            background = get_background_data()

            explanation_results = explain_churn(
                model,
                customer,
                background,
            )

            st.subheader("Strongest contributing features")

            display_results = explanation_results.head(8).copy()

            display_results["SHAP contribution"] = (
                display_results["SHAP contribution"].map(
                    lambda value: f"{value:+.4f}"
                )
            )

            display_results = display_results[
                [
                    "Feature",
                    "Customer value",
                    "SHAP contribution",
                    "Direction",
                ]
            ]

            st.dataframe(
                display_results,
                hide_index=True,
                width="stretch",
            )

            higher_risk = explanation_results[
                explanation_results["SHAP contribution"] > 0
            ].head(3)

            lower_risk = explanation_results[
                explanation_results["SHAP contribution"] < 0
            ].head(3)

            higher_col, lower_col = st.columns(2)

            with higher_col:
                st.subheader("🔺 Pushes risk higher")

                if higher_risk.empty:
                    st.write(
                        "No positive SHAP contributions were identified."
                    )
                else:
                    for _, row in higher_risk.iterrows():
                        st.write(
                            f"**{row['Feature']}**: "
                            f"{row['Customer value']}"
                        )
                        st.caption(
                            f"Contribution: "
                            f"{row['SHAP contribution']:+.4f}"
                        )

            with lower_col:
                st.subheader("🔻 Pushes risk lower")

                if lower_risk.empty:
                    st.write(
                        "No negative SHAP contributions were identified."
                    )
                else:
                    for _, row in lower_risk.iterrows():
                        st.write(
                            f"**{row['Feature']}**: "
                            f"{row['Customer value']}"
                        )
                        st.caption(
                            f"Contribution: "
                            f"{row['SHAP contribution']:+.4f}"
                        )

            st.info(
                "Interpret the contributions relative to the background "
                "sample. They describe how the model responds to these "
                "inputs, not why a customer will actually leave."
            )

        except Exception as error:
            st.error(
                "The churn prediction succeeded, but SHAP explanations "
                "could not be calculated."
            )

            st.write(
                "Error details are shown below so the explanation "
                "can be debugged."
            )

            st.code(
                f"{type(error).__name__}: {error}"
            )

            st.caption(
                "The prediction above remains the model's result; "
                "the explanation has not been generated successfully."
            )

# ============================================================
# 7. GEOGRAPHIC CUSTOMER CHURN MAP
# ============================================================

@st.cache_data
def load_geographic_data():
    """Load customer locations and historical churn outcomes."""
    geo_df = pd.read_excel(DATA_PATH)

    required_columns = [
        "Latitude",
        "Longitude",
        "City",
        "State",
        "Churn Label",
    ]

    missing = [
        column for column in required_columns
        if column not in geo_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing geographic columns: {missing}"
        )

    geo_df = geo_df[required_columns].copy()

    geo_df["Latitude"] = pd.to_numeric(
        geo_df["Latitude"], errors="coerce"
    )
    geo_df["Longitude"] = pd.to_numeric(
        geo_df["Longitude"], errors="coerce"
    )

    geo_df = geo_df.dropna(
        subset=["Latitude", "Longitude"]
    )

    # Remove coordinates outside valid geographic ranges.
    geo_df = geo_df[
        geo_df["Latitude"].between(-90, 90)
        & geo_df["Longitude"].between(-180, 180)
    ]

    geo_df["City"] = geo_df["City"].fillna("Unknown")
    geo_df["State"] = geo_df["State"].fillna("Unknown")
    geo_df["Churn Label"] = (
        geo_df["Churn Label"].fillna("Unknown").astype(str)
    )

    return geo_df


st.divider()
st.header("🌎 Geographic Customer Churn Map")

st.write(
    "Explore historical customer locations and churn outcomes. "
    "This map describes the dataset; it does not predict churn "
    "from geography or establish that location causes churn."
)

try:
    import pydeck as pdk

    geo_df = load_geographic_data()

    if geo_df.empty:
        st.warning(
            "No valid latitude and longitude coordinates were found."
        )
    else:
        st.metric(
            "Customers with valid coordinates",
            f"{len(geo_df):,}"
        )

        churn_filter = st.selectbox(
            "Filter historical customers",
            [
                "All customers",
                "Churned customers",
                "Retained customers",
            ],
            key="geo_churn_filter",
        )

        if churn_filter == "Churned customers":
            map_df = geo_df[
                geo_df["Churn Label"].str.lower() == "yes"
            ].copy()
        elif churn_filter == "Retained customers":
            map_df = geo_df[
                geo_df["Churn Label"].str.lower() == "no"
            ].copy()
        else:
            map_df = geo_df.copy()

        if map_df.empty:
            st.info("No customers match this filter.")
        else:
            # Color historical churned customers red and retained
            # customers blue. Unknown outcomes are gray.
            def customer_color(label):
                label = str(label).strip().lower()

                if label == "yes":
                    return [220, 60, 60, 170]
                if label == "no":
                    return [40, 120, 220, 150]
                return [130, 130, 130, 150]

            map_df["color"] = map_df["Churn Label"].apply(
                customer_color
            )

            # Aggregate nearby records so overlapping customers
            # remain easier to explore on the map.
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position="[Longitude, Latitude]",
                get_fill_color="color",
                get_radius=100,
                radius_min_pixels=3,
                radius_max_pixels=12,
                pickable=True,
                auto_highlight=True,
            )

            view_state = pdk.ViewState(
                latitude=float(map_df["Latitude"].median()),
                longitude=float(map_df["Longitude"].median()),
                zoom=3,
                pitch=0,
            )

            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={
                    "html": (
                        "<b>City:</b> {City}<br/>"
                        "<b>State:</b> {State}<br/>"
                        "<b>Historical churn:</b> {Churn Label}"
                    ),
                    "style": {"color": "white"},
                },
                map_style=None,
            )

            st.pydeck_chart(
                deck,
                use_container_width=True,
            )

            st.caption(
                "Red = historically churned; blue = historically retained; "
                "gray = unknown outcome. Each point represents a customer "
                "record. Overlapping points may obscure one another."
            )

            st.subheader("Historical churn summary by state")

            state_summary = (
                map_df.groupby("State")
                .agg(
                    Customer_Records=("Churn Label", "size"),
                    Churned=("Churn Label", lambda s: (
                        s.str.lower() == "yes"
                    ).sum()),
                )
                .reset_index()
            )

            state_summary["Churn Rate (%)"] = (
                state_summary["Churned"]
                / state_summary["Customer_Records"]
                * 100
            ).round(1)

            state_summary = state_summary.sort_values(
                "Customer_Records", ascending=False
            )

            st.dataframe(
                state_summary,
                hide_index=True,
                use_container_width=True,
            )

except Exception as error:
    st.error("The geographic visualization could not be loaded.")
    st.code(f"{type(error).__name__}: {error}")
