import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load("modelo_diabetes.pkl")

modelo = load_model()

st.title("🔬 Predicción de Diabetes — IA Cloud Demo")

st.markdown(
    "Introduce los valores del paciente y presiona **Predecir** "
    "para estimar el riesgo de diabetes."
)

# ── Entradas ────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    Pregnancies  = st.number_input("Embarazos",      min_value=0, step=1)
    Glucose      = st.number_input("Glucosa",        min_value=0)
    BloodPressure= st.number_input("Presión arterial", min_value=0)
with col2:
    SkinThickness= st.number_input("Grosor de piel", min_value=0)
    Insulin      = st.number_input("Insulina",       min_value=0)
    BMI          = st.number_input("IMC",            min_value=0.0, format="%.1f")
with col3:
    DPF          = st.number_input("Pedigree Func.", min_value=0.0, format="%.3f")
    Age          = st.number_input("Edad",           min_value=0, step=1)

# ── Predicción ──────────────────────────────────────────────
if st.button("Predecir"):
    datos = pd.DataFrame(
        [[Pregnancies, Glucose, BloodPressure, SkinThickness,
          Insulin, BMI, DPF, Age]],
        columns=[
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
    )
    pred = modelo.predict(datos)[0]
    prob = modelo.predict_proba(datos)[0][1]

    st.subheader("Resultado")
    if pred == 1:
        st.error(f"⚠️ Riesgo alto de diabetes (prob. {prob:.2%})")
    else:
        st.success(f"✅ Riesgo bajo de diabetes (prob. {prob:.2%})")

    st.markdown("### Datos utilizados")
    st.dataframe(datos.style.format(precision=2))