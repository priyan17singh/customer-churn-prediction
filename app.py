import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉")

st.title("📉 Customer Churn Prediction")
st.markdown("Predict whether a customer is likely to churn using a trained Deep Learning model.")

# ----------------------------
# Load Artifacts
# ----------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("model.h5")

    with open("./Pickle/label_encoder_gender.pkl", "rb") as f:
        label_encoder_gender = pickle.load(f)

    with open("./Pickle/onehot_encoder_geo.pkl", "rb") as f:
        onehot_encoder_geo = pickle.load(f)

    with open("./Pickle/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, label_encoder_gender, onehot_encoder_geo, scaler


model, label_encoder_gender, onehot_encoder_geo, scaler = load_artifacts()

# ----------------------------
# Initialize Session State
# ----------------------------
default_state = {
    "geography": onehot_encoder_geo.categories_[0][0],
    "gender": label_encoder_gender.classes_[0],
    "age": 27,
    "tenure": 3,
    "num_products": 1,
    "credit_score": 650,
    "balance": 5000.0,
    "salary": 30000.0,
    "has_card": "Yes",
    "active_member": "Yes",
    "show_popup": False,
    "prediction": None
}

for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ----------------------------
# Input Form
# ----------------------------
st.subheader("🔹 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    st.selectbox(
        "Geography",
        onehot_encoder_geo.categories_[0],
        key="geography"
    )
    st.selectbox(
        "Gender",
        label_encoder_gender.classes_,
        key="gender"
    )
    st.slider("Age", 18, 92, key="age")
    st.slider("Tenure", 0, 10, key="tenure")
    st.slider("Number of Products", 1, 4, key="num_products")

with col2:
    st.number_input("Credit Score", 300, 900, key="credit_score")
    st.number_input("Balance", min_value=0.0, key="balance")
    st.number_input("Estimated Salary", min_value=0.0, key="salary")
    st.selectbox("Has Credit Card", ["Yes", "No"], key="has_card")
    st.selectbox("Active Member", ["Yes", "No"], key="active_member")

# ----------------------------
# Predict Button
# ----------------------------
if st.button("🔮 Predict"):
    gender_encoded = label_encoder_gender.transform([st.session_state.gender])[0]

    input_df = pd.DataFrame({
        "CreditScore": [st.session_state.credit_score],
        "Gender": [gender_encoded],
        "Age": [st.session_state.age],
        "Tenure": [st.session_state.tenure],
        "Balance": [st.session_state.balance],
        "NumOfProducts": [st.session_state.num_products],
        "HasCrCard": [1 if st.session_state.has_card == "Yes" else 0],
        "IsActiveMember": [1 if st.session_state.active_member == "Yes" else 0],
        "EstimatedSalary": [st.session_state.salary]
    })

    geo_encoded = onehot_encoder_geo.transform(
        [[st.session_state.geography]]
    ).toarray()

    geo_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
    )

    final_input = pd.concat([input_df, geo_df], axis=1)
    final_scaled = scaler.transform(final_input)

    st.session_state.prediction = model.predict(final_scaled)[0][0]
    st.session_state.show_popup = True

# ----------------------------
# Popup Modal
# ----------------------------
if st.session_state.show_popup:

    @st.dialog("📊 Prediction Result")
    def prediction_dialog():
        prob = st.session_state.prediction

        st.metric("Churn Probability", f"{prob:.2%}")

        if prob > 0.5:
            st.error("⚠️ Customer is likely to churn")
        else:
            st.success("✅ Customer is not likely to churn")

        if st.button("Close"):
            # 🔥 RESET ALL INPUTS
            for key in default_state:
                st.session_state[key] = default_state[key]

            st.rerun()

    prediction_dialog()



