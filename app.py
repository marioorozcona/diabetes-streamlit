# ── IMPORTS ─────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import shap
from fpdf import FPDF
import io

st.set_page_config(page_title="Predicción de Diabetes", page_icon="🩺", layout="centered")

# ── CARGA DE MODELO Y EXPLAINER (cache) ─────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("modelo_diabetes.pkl")

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

model = load_model()
explainer = load_explainer(model)

# ── TÍTULO ──────────────────────────────────────────────────────────────
st.title("🧪 Predicción de Diabetes — IA Cloud Demo")
st.markdown(
    "Introduce los valores del paciente y presiona **Predecir** "
    "para estimar el **riesgo** de diabetes."
)

# ── ENTRADAS DE USUARIO ────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    Pregnancies  = st.number_input("Embarazos", min_value=0, step=1)
    Glucose      = st.number_input("Glucosa",   min_value=0)
    BloodPressure= st.number_input("Presión arterial", min_value=0)
with col2:
    SkinThickness= st.number_input("Grosor de piel",  min_value=0)
    Insulin      = st.number_input("Insulina",        min_value=0)
    BMI          = st.number_input("IMC",             min_value=0.0, format="%.1f")
with col3:
    DPF          = st.number_input("Pedigree Func.",  min_value=0.0, format="%.3f")
    Age          = st.number_input("Edad",            min_value=0,  step=1)

# ── BOTÓN DE PREDICCIÓN ────────────────────────────────────────────────
if st.button("Predecir"):
    # DataFrame con una sola fila
    X = pd.DataFrame(
        [[Pregnancies, Glucose, BloodPressure, SkinThickness,
          Insulin, BMI, DPF, Age]],
        columns=[
            "Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age"
        ]
    )

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    # ─ Resultado textual
    st.subheader("Resultado")
    if pred == 1:
        st.error(f"⚠️ **Riesgo alto** de diabetes (probabilidad {prob:.2%})")
    else:
        st.success(f"✅ **Riesgo bajo** de diabetes (probabilidad {prob:.2%})")

    # ─ Gráfico de probabilidad (matplotlib)
    st.subheader("Probabilidad estimada")
    fig, ax = plt.subplots(figsize=(4, 0.5))
    ax.barh([0], prob, color="#EF553B")
    ax.barh([0], 1 - prob, left=prob, color="#DDDDDD")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([])
    ax.set_xlabel("Probabilidad de diabetes")
    st.pyplot(fig)

    # Versión interactiva (Plotly)
    fig_px = px.bar(
        x=[prob, 1 - prob],
        y=["Riesgo", ""],
        orientation="h",
        labels={"x": "Probabilidad"},
        text=[f"{prob:.2%}", ""],
        color_discrete_sequence=["#EF553B", "#DDDDDD"],
        height=150
    )
    fig_px.update_layout(yaxis_visible=False, xaxis_range=[0,1])
    st.plotly_chart(fig_px, use_container_width=True)

    # ─ Explicación SHAP global
    with st.expander("Ver importancia global de características (SHAP)"):
        shap_vals = explainer.shap_values(X.sample(1, replace=True))[0]
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.summary_plot(shap_vals, feature_names=X.columns, plot_type="bar", show=False)
        st.pyplot(bbox_inches="tight")

    # ─ Descargar CSV y PDF
    st.markdown("### Descargar resultado")
    csv_bytes = X.assign(Prediccion=pred, Probabilidad=prob).to_csv(index=False).encode()
    st.download_button("⬇️ CSV", csv_bytes, "resultado_diabetes.csv", "text/csv")

    def make_pdf(df, p, pr):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0,10,"Reporte de Predicción de Diabetes", ln=1)
        pdf.ln(4)
        for col,val in df.iloc[0].items():
            pdf.cell(0,8,f"{col}: {val}", ln=1)
        pdf.ln(4)
        pdf.cell(0,10,f"Predicción: {'Diabetes' if p==1 else 'No diabetes'}",ln=1)
        pdf.cell(0,10,f"Probabilidad: {pr:.2%}",ln=1)
        return pdf.output(dest="S").encode("latin-1")

    pdf_bytes = make_pdf(X, pred, prob)
    st.download_button("⬇️ PDF", pdf_bytes, "resultado_diabetes.pdf", "application/pdf")
