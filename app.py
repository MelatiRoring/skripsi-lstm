import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

st.set_page_config(page_title="Dashboard LSTM", layout="wide")

# ================= LOGIN =================
def login():
    st.title("🔐 Login Admin")
    user = st.text_input("Username")
    pw = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pw == "123":
            st.session_state["login"] = True
        else:
            st.error("Login gagal")

if "login" not in st.session_state:
    st.session_state["login"] = False

if not st.session_state["login"]:
    login()
    st.stop()

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    from tensorflow.keras.models import load_model
    return load_model("model_lstm.h5")

model = load_model()

# ================= DASHBOARD =================
st.title("📊 Dashboard Prediksi Pengangguran")

menu = st.sidebar.selectbox("Menu", ["Upload", "Visualisasi", "Prediksi"])

# ================= UPLOAD =================
if menu == "Upload":
    file = st.file_uploader("Upload Dataset", type=["xlsx"])
    if file:
        df = pd.read_excel(file)
        df['Tahun'] = pd.to_datetime(df['Tahun'])
        df.set_index('Tahun', inplace=True)
        st.session_state["data"] = df
        st.success("Upload berhasil")
        st.dataframe(df)

# ================= VISUAL =================
elif menu == "Visualisasi":
    df = st.session_state.get("data")
    if df is not None:
        fig = px.line(df, x=df.index, y="Total_Pengangguran")
        st.plotly_chart(fig)
    else:
        st.warning("Upload data dulu")

# ================= PREDIKSI =================
elif menu == "Prediksi":
    df = st.session_state.get("data")

    if df is not None:
        if st.button("Prediksi"):
            data = df.values
            last = data[-5:]

            pred = []
            for _ in range(5):
                p = model.predict(last.reshape(1,5,last.shape[1]))[0][0]
                pred.append(p)
                last = np.vstack([last[1:], last[-1]])

            years = [2026,2027,2028,2029,2030]
            result = pd.DataFrame({"Tahun": years, "Prediksi": pred})

            st.dataframe(result)

            fig = px.line(result, x="Tahun", y="Prediksi", markers=True)
            st.plotly_chart(fig)

            st.download_button("Download CSV", result.to_csv(index=False), "prediksi.csv")

    else:
        st.warning("Upload data dulu")
