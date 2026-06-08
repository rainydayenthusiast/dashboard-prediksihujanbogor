
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Dashboard Prediksi Curah Hujan", layout="wide")

# =========================
# PATH FILE
# =========================
BASE_DATA_PATH = "df_utama.csv"
MODEL_PATH = "model_final.h5"
SCALER_PATH = "scaler_multifeature.save"

FEATURES = ['RR', 'RH_AVG', 'TAVG', 'SS', 'FF_AVG']

HORIZON_OPTIONS = {
    "3 hari": 3,
    "7 hari": 7,
    "14 hari": 14,
    "30 hari": 30,
    "90 hari": 90,
    "180 hari": 180,
    "365 hari": 365
}

LOOK_BACK = 14

# =========================
# LOAD MODEL / SCALER / DATA
# =========================
@st.cache_resource
def load_assets():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

@st.cache_data
def load_base_data():
    df = pd.read_csv(BASE_DATA_PATH)
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], format='mixed', dayfirst=True)
    df = df.sort_values('TANGGAL')
    return df

model, scaler = load_assets()
df_base = load_base_data()

# =========================
# FUNGSI BANTU
# =========================
def standardize_extra_data(df_extra: pd.DataFrame) -> pd.DataFrame:
    df_extra = df_extra.copy()
    
    # 1. Pastikan semua nama kolom kapital dan bersih dari spasi tak terlihat
    df_extra.columns = [str(c).upper().strip() for c in df_extra.columns]

    rename_map = {
        'DATE': 'TANGGAL',
        'RH_AVG': 'RH_AVG',
        'TAVG': 'TAVG',
        'SS': 'SS',
        'FF_AVG': 'FF_AVG'
    }
    df_extra = df_extra.rename(columns=rename_map)

    required_cols = ['TANGGAL'] + FEATURES
    missing = [c for c in required_cols if c not in df_extra.columns]
    
    if missing:
        kolom_ditemukan = list(df_extra.columns)
        raise ValueError(f"Kolom hilang: {missing}. Kolom yang terdeteksi di file: {kolom_ditemukan}")

    df_extra = df_extra[required_cols].copy()

    # 2. PERBAIKAN TANGGAL & CATATAN KAKI BMKG
    # Gunakan errors='coerce' agar tulisan "KETERANGAN:" berubah menjadi NaT (Not a Time)
    df_extra['TANGGAL'] = pd.to_datetime(df_extra['TANGGAL'], format='mixed', dayfirst=True, errors='coerce')
    
    # Buang baris yang tanggalnya NaT (baris metadata/catatan kaki di bagian bawah file)
    df_extra = df_extra.dropna(subset=['TANGGAL'])

    # cleaning dasar
    for col in FEATURES:
        df_extra[col] = (
            df_extra[col]
            .astype(str)
            .str.replace(',', '.', regex=False)
            .replace({'TTU': np.nan, '8888': np.nan, '9999': np.nan, 'NAN': np.nan, 'NONE': np.nan})
        )

    # khusus RR, simbol '-' dianggap 0 mm
    df_extra['RR'] = df_extra['RR'].replace('-', '0')

    # kolom lain: '-' dianggap missing
    for col in ['RH_AVG', 'TAVG', 'SS', 'FF_AVG']:
        df_extra[col] = df_extra[col].replace('-', np.nan)

    # konversi ke numerik
    for col in FEATURES:
        df_extra[col] = pd.to_numeric(df_extra[col], errors='coerce')

    # heuristik SS: kalau masih persen, ubah jadi jam (basis 8 jam)
    if df_extra['SS'].dropna().shape[0] > 0 and df_extra['SS'].dropna().max() > 24:
        df_extra['SS'] = (df_extra['SS'] / 100.0) * 8.0

    # imputasi sederhana sesuai pipeline utama
    other_cols = ['RH_AVG', 'TAVG', 'SS', 'FF_AVG']
    df_extra[other_cols] = df_extra[other_cols].interpolate(method='linear').ffill().bfill()
    df_extra['RR'] = df_extra['RR'].interpolate(limit=3).fillna(0)

    return df_extra

def recursive_forecast_multifeature(model, scaler, df_final, look_back, horizon):
    """
    Prediksi multi-step untuk 5 fitur sekaligus.
    Output model = [RR, RH_AVG, TAVG, SS, FF_AVG]
    """
    df_features = df_final[FEATURES].copy()

    # scaling
    scaled = scaler.transform(df_features)

    n_features = len(FEATURES)
    current_window = scaled[-look_back:].copy()
    preds_scaled = []

    for _ in range(horizon):
        x_input = current_window.reshape(1, look_back, n_features)
        pred_scaled = model.predict(x_input, verbose=0)[0]
        preds_scaled.append(pred_scaled)
        current_window = np.vstack([current_window[1:], pred_scaled])

    preds_scaled = np.array(preds_scaled)
    preds = scaler.inverse_transform(preds_scaled)

    # RR tidak boleh negatif
    idx_rr = FEATURES.index('RR')
    preds[:, idx_rr] = np.maximum(preds[:, idx_rr], 0)

    return preds

def make_hover_chart(df_plot, y_col, title, y_label):
    fig = px.line(
        df_plot,
        x='TANGGAL',
        y=y_col,
        markers=True,
        title=title
    )

    fig.update_traces(
        hovertemplate=(
            "<b>Tanggal:</b> %{x|%d-%m-%Y}<br>"
            f"<b>{y_label}:</b> " + "%{y:.3f}<extra></extra>"
        )
    )

    fig.update_layout(
        xaxis_title="Tanggal",
        yaxis_title=y_label,
        hovermode="x unified"
    )

    return fig

# =========================
# UI
# =========================
col_logo, col_title = st.columns([1, 6])

with col_logo:
    try:
        st.image("logo.png", width=120)
    except:
        pass

with col_title:
    st.title("Dashboard Prediksi Curah Hujan")
    st.write(
        "Model Vanilla LSTM Multi-Output untuk Prediksi Parameter Klimatologi Harian"
    )
    
try:
    st.sidebar.image("logo.png", width=150)
except:
    pass
    
st.sidebar.header("Pengaturan Prediksi")
horizon_label = st.sidebar.selectbox(
    "Pilih horizon prediksi",
    list(HORIZON_OPTIONS.keys()),
    index=2
)
horizon = HORIZON_OPTIONS[horizon_label]

# UBAHAN KUNCI: accept_multiple_files=True
uploaded_files = st.sidebar.file_uploader(
    "Upload data tambahan (.csv / .xlsx)",
    type=["csv", "xlsx"],
    accept_multiple_files=True 
)

# =========================
# GABUNG DATA (SUPER AMAN & MULTI-FILE)
# =========================
df_used = df_base.copy()

if uploaded_files: # Mengecek apakah list tidak kosong
    for uploaded_file in uploaded_files: # Melakukan iterasi untuk setiap file yang diunggah
        try:
            # Membaca file dengan deteksi separator otomatis (sep=None)
            if uploaded_file.name.lower().endswith(".csv"):
                df_extra_raw = pd.read_csv(uploaded_file, header=None, sep=None, engine='python')
            else:
                df_extra_raw = pd.read_excel(uploaded_file, header=None)

            # Mencari letak baris yang mengandung tulisan "TANGGAL"
            header_idx = 0
            for i in range(min(20, len(df_extra_raw))): 
                row_vals = [str(x).upper().strip() for x in df_extra_raw.iloc[i].values]
                if any("TANGGAL" in val or "DATE" in val for val in row_vals):
                    header_idx = i
                    break
            
            # Jadikan baris yang ditemukan sebagai header asli dan bersihkan spasi
            new_cols = [str(c).upper().strip() for c in df_extra_raw.iloc[header_idx].values]
            df_extra_raw.columns = new_cols
            
            # Buang baris-baris metadata di atasnya
            df_extra = df_extra_raw.iloc[header_idx + 1:].copy().reset_index(drop=True)

            # Lakukan standarisasi menggunakan fungsi yang sudah ada
            df_extra = standardize_extra_data(df_extra)

            df_used = pd.concat([df_used, df_extra], ignore_index=True)
            
            st.success(f"Data tambahan ({uploaded_file.name}) berhasil diproses dan digabung.")
        except Exception as e:
            st.error(f"Gagal memproses data {uploaded_file.name}: {e}")

# standarisasi final (menggabungkan dan mengurutkan keseluruhan data)
df_used = standardize_extra_data(df_used)

df_used = (
    df_used
    .sort_values("TANGGAL")
    .drop_duplicates(
        subset="TANGGAL",
        keep="last" # Jika ada tanggal yang sama, gunakan data dari file terbaru
    )
    .reset_index(drop=True)
)

# =========================
# PREVIEW DATA
# =========================
st.subheader("Data yang Digunakan")
st.write(f"Jumlah data: {len(df_used)} baris")
if uploaded_files:
    st.info(
        f"Dataset saat ini terdiri dari {len(df_used)} baris setelah proses penggabungan {len(uploaded_files)} file tambahan."
    )
st.write(f"Rentang tanggal: {df_used['TANGGAL'].min().date()} s.d. {df_used['TANGGAL'].max().date()}")
st.dataframe(df_used.tail(10), use_container_width=True)

# =========================
# PREDIKSI
# =========================
if st.button("Jalankan Prediksi"):
    try:
        preds = recursive_forecast_multifeature(model, scaler, df_used, LOOK_BACK, horizon)

        start_date = df_used['TANGGAL'].max() + pd.Timedelta(days=1)
        future_dates = pd.date_range(start=start_date, periods=horizon, freq='D')

        hasil = pd.DataFrame(preds, columns=FEATURES)
        hasil.insert(0, 'TANGGAL', future_dates)

        st.subheader("Hasil Prediksi")
        st.dataframe(hasil, use_container_width=True)

        # Grafik utama RR interaktif
        st.subheader("Grafik Prediksi Curah Hujan (RR)")
        fig_rr = make_hover_chart(
            hasil,
            y_col='RR',
            title=f"Prediksi Curah Hujan - {horizon_label}",
            y_label='Curah Hujan (mm/hari)'
        )
        st.plotly_chart(fig_rr, use_container_width=True)

        # Grafik semua parameter interaktif
        st.subheader("Grafik Prediksi Semua Parameter")
        label_map = {
            'RR': 'Curah Hujan (mm/hari)',
            'RH_AVG': 'Kelembapan Rata-rata (%)',
            'TAVG': 'Temperatur Rata-rata (°C)',
            'SS': 'Lama Penyinaran (jam)',
            'FF_AVG': 'Kecepatan Angin Rata-rata'
        }

        for col in FEATURES:
            fig_param = make_hover_chart(
                hasil,
                y_col=col,
                title=f"Prediksi {col}",
                y_label=label_map.get(col, col)
            )
            st.plotly_chart(fig_param, use_container_width=True)

        csv = hasil.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download hasil prediksi (CSV)",
            data=csv,
            file_name="hasil_prediksi_multifeature.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Prediksi gagal dijalankan: {e}")
