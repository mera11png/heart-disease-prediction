import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# تحميل الموديل والـ scaler وأسماء الأعمدة
# ----------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("❤️ Heart Disease Prediction")
st.write("أدخل بيانات المريض عشان نتوقع احتمال وجود مرض قلب.")

# ----------------------------
# نموذج إدخال البيانات
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    dataset = st.selectbox("Dataset (Source)", ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"])
    cp = st.selectbox("Chest Pain Type", ["typical angina", "asymptomatic", "non-anginal", "atypical angina"])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=50, max_value=250, value=130)
    chol = st.number_input("Cholesterol (chol)", min_value=50, max_value=700, value=240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
    restecg = st.selectbox("Resting ECG", ["normal", "lv hypertrophy", "st-t abnormality"])

with col2:
    thalch = st.number_input("Max Heart Rate Achieved (thalch)", min_value=50, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina", ["True", "False"])
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=-3.0, max_value=7.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment", ["upsloping", "flat", "downsloping"])
    ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])

# ----------------------------
# زرار التوقع
# ----------------------------
if st.button("توقع (Predict)"):

    # بناء صف واحد بنفس شكل الداتا الخام
    raw_input = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "dataset": dataset,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs == "True",
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang == "True",
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }])

    # نفس خطوات الـ Encoding اللي حصلت وقت التدريب
    raw_input["sex"] = raw_input["sex"].map({"Male": 1, "Female": 0})
    raw_input["fbs"] = raw_input["fbs"].astype(int)
    raw_input["exang"] = raw_input["exang"].astype(int)

    nominal_cols = ["cp", "restecg", "slope", "thal", "dataset"]
    input_encoded = pd.get_dummies(raw_input, columns=nominal_cols, drop_first=False)

    # محاذاة الأعمدة مع الأعمدة اللي اتدرب عليها الموديل بالظبط
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    # Scaling
    input_scaled = scaler.transform(input_encoded)

    # التوقع
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ النتيجة: احتمال وجود مرض قلب (Probability: {probability:.1%})")
    else:
        st.success(f"✅ النتيجة: مفيش مؤشر لمرض قلب (Probability: {probability:.1%})")

    st.caption("⚠️ ده مجرد نموذج تعليمي، مش بديل عن استشارة طبيب.")
