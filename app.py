import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf

st.set_page_config(page_title="Prediksi Pengangguran", layout="wide")

# ================= UI HEADER =================
st.markdown("""
<h1 style='text-align: center; color: #2c3e50;'>
📊 Dashboard Prediksi Pengangguran Sulawesi Utara
</h1>
<p style='text-align: center;'>Model: LSTM Multivariat</p>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_lstm.h5", compile=False)

model = load_model()

# ================= SIDEBAR =================
st.sidebar.title("⚙️ Pengaturan")
file = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx"])

# ================= MAIN =================
if file is not None:
    if file.name.endswith(".xlsx"):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)

    df['Tahun'] = pd.to_datetime(df['Tahun'])
    df.set_index('Tahun', inplace=True)

    st.subheader("📄 Data")
    st.dataframe(df.tail())

    # ================= NORMALISASI =================
    scaler = MinMaxScaler()
    scaled_df = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns,
        index=df.index
    )

    # ================= FEATURE SELECTION =================
    target_col = 'Total_Pengangguran'
    corr = scaled_df.corr()
    selected = corr[target_col].drop(target_col)
    selected_features = selected[abs(selected) >= 0.3].index.tolist()
    selected_features.append(target_col)

    scaled_df = scaled_df[selected_features]

    st.success(f"Fitur digunakan: {selected_features}")

    # ================= VISUAL =================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Tren Data")
        fig, ax = plt.subplots()
        scaled_df.plot(ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("🔥 Korelasi")
        fig2, ax2 = plt.subplots()
        sns.heatmap(scaled_df.corr(), annot=True, cmap='viridis', ax=ax2)
        st.pyplot(fig2)

    # ================= LSTM =================
    def create_seq(data, target, window):
        X, y = [], []
        for i in range(len(data) - window):
            X.append(data.iloc[i:i+window].drop(columns=[target]).values)
            y.append(data.iloc[i+window][target])
        return np.array(X), np.array(y)

    window = 5
    X, y = create_seq(scaled_df, target_col, window)

    split = int(len(X)*0.8)
    X_test, y_test = X[split:], y[split:]

    y_pred = model.predict(X_test)

    # ================= INVERSE =================
    def inverse(y_scaled):
        dummy = np.zeros((len(y_scaled), scaler.n_features_in_))
        idx = scaled_df.columns.get_loc(target_col)
        dummy[:, idx] = y_scaled
        return scaler.inverse_transform(dummy)[:, idx]

    y_test_inv = inverse(y_test)
    y_pred_inv = inverse(y_pred.flatten())

    # ================= METRICS =================
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mape = np.mean(np.abs((y_test_inv - y_pred_inv)/y_test_inv))*100

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:,.0f}")
    c2.metric("RMSE", f"{rmse:,.0f}")
    c3.metric("MAPE (%)", f"{mape:.2f}")

    # ================= GRAPH =================
    st.subheader("📉 Actual vs Predicted")
    tahun = df.index[-len(y_test_inv):].year

    fig3, ax3 = plt.subplots()
    ax3.plot(tahun, y_test_inv, label="Aktual")
    ax3.plot(tahun, y_pred_inv, label="Prediksi")
    ax3.legend()
    st.pyplot(fig3)

    # DOWNLOAD PNG
    fig3.savefig("grafik.png")
    with open("grafik.png", "rb") as f:
        st.download_button("📥 Download Grafik", f, file_name="grafik.png")

    # ================= FORECAST =================
    st.subheader("🔮 Prediksi 5 Tahun ke Depan")

    last_window = scaled_df.drop(columns=[target_col]).values[-window:]
    future = []

    for _ in range(5):
        pred = model.predict(last_window.reshape(1, window, -1))[0][0]
        future.append(pred)
        last_window = np.vstack([last_window[1:], last_window[-1]])

    future_inv = inverse(np.array(future))
    future_years = [df.index.year[-1] + i for i in range(1, 6)]

    forecast_df = pd.DataFrame({
        "Tahun": future_years,
        "Prediksi": future_inv
    })

    st.dataframe(forecast_df)

    # DOWNLOAD CSV
    st.download_button(
        "📥 Download Hasil Prediksi",
        forecast_df.to_csv(index=False),
        "prediksi.csv"
    )

else:
    st.info("Upload dataset untuk mulai.")