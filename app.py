import os
import sys

# Auto-runner: jika dijalankan dengan 'py app.py' atau 'python app.py', otomatis aktifkan server Streamlit!
from streamlit.runtime.scriptrunner import get_script_run_ctx
if get_script_run_ctx() is None:
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
    sys.exit(stcli.main())

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Klasifikasi Kelayakan Pernikahan - KUA",
    page_icon="💍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan elegan & modern
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .card-header-suami {
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .card-header-istri {
        background: linear-gradient(135deg, #DB2777 0%, #F472B6 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .card-body {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 0 0 8px 8px;
        padding: 18px 20px 20px 20px;
        margin-bottom: 20px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1E40AF 0%, #1D4ED8 100%);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. LOAD MODEL DAN PREPROCESSOR (.PKL)
# ==============================================================================
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    
    model = joblib.load(os.path.join(models_dir, 'naive_bayes.pkl'))
    encoder = joblib.load(os.path.join(models_dir, 'preprocessor.pkl'))
    features = joblib.load(os.path.join(models_dir, 'features.pkl'))
    return model, encoder, features

try:
    nb_model, encoder, feature_cols = load_artifacts()
except Exception as e:
    st.error(f"Gagal memuat artefak model (.pkl). Pastikan folder models/ tersedia. Error: {e}")
    st.stop()

# ==============================================================================
# 3. FUNGSI FEATURE ENGINEERING & LOGIKA DOMAIN
# ==============================================================================
def kategori_usia(u):
    if u < 19:
        return '< 19 (Di Bawah Umur)'
    elif u <= 24:
        return '19-24 (Muda)'
    elif u <= 30:
        return '25-30 (Matang)'
    else:
        return '> 30 (Dewasa)'

def kategori_pekerjaan_suami(p):
    p = str(p).upper().strip()
    if any(k in p for k in ['PNS', 'POLISI', 'TNI', 'TENTARA', 'BUMD', 'BUMN', 'DOKTER', 'DOSEN', 'GURU', 'PERAWAT', 'PERANGKAT DESA', 'PPPK', 'KARYAWAN SWASTA']):
        return 'Tetap/Mapan'
    elif any(k in p for k in ['WIRASWASTA', 'PEDAGANG', 'PERDAGANGAN', 'PETANI', 'PETERNAK', 'PETAMBAK', 'TAMBANG']):
        return 'Mandiri/Wiraswasta'
    elif any(k in p for k in ['BURUH', 'SOPIR', 'HONORER', 'SENI']):
        return 'Tidak Tetap/Buruh'
    else:
        return 'Tidak Bekerja'

def kategori_pekerjaan_istri(p):
    p = str(p).upper().strip()
    if 'MENGURUS RUMAH TANGGA' in p:
        return 'Mengurus Rumah Tangga'
    elif any(k in p for k in ['PNS', 'POLISI', 'TNI', 'BUMD', 'BUMN', 'DOKTER', 'DOSEN', 'PERAWAT', 'BIDAN', 'KARYAWAN SWASTA', 'WIRASWASTA']) or p in ['GURU', 'GURU HONORER']:
        return 'Bekerja/Berpenghasilan'
    elif 'BURUH' in p:
        return 'Pekerja Tidak Tetap'
    else:
        return 'Belum Bekerja/Pelajar'

def standarisasi_pendidikan(p):
    if pd.isnull(p):
        return p
    p = str(p).upper().strip()
    if 'STRATA II' in p or 'S2' in p or 'S3' in p:
        return 'STRATA II'
    elif 'STRATA I' in p or 'DIPLOMA IV' in p or 'S1' in p:
        return 'DIPLOMA IV/STRATA I'
    elif 'DIPLOMA III' in p or 'S. MUDA' in p or 'D3' in p:
        return 'AKADEMI/DIPLOMA III/S. MUDA'
    elif 'DIPLOMA I' in p or 'DIPLOMA II' in p:
        return 'DIPLOMA I/II'
    elif 'SLTA' in p or 'SMA' in p or 'SMK' in p or 'ALIYAH' in p:
        return 'SLTA/SEDERAJAT'
    elif 'SLTP' in p or 'SMP' in p or 'MTS' in p:
        return 'SLTP/SEDERAJAT'
    elif 'TIDAK TAMAT SD' in p or 'BELUM SEKOLAH' in p:
        return 'TIDAK TAMAT SD/SEDERAJAT'
    elif 'SD' in p:
        return 'TAMAT SD/SEDERAJAT'
    return p

# ==============================================================================
# 4. SIDEBAR - INFORMASI & PRESET KASUS UJI
# ==============================================================================
with st.sidebar:
    # st.image("https://cdn-icons-png.flaticon.com/512/8724/8724227.png", width=75)
    st.title("Informasi Model")
    st.caption("Implementasi Skripsi Teknik Informatika")
    
    st.markdown("---")
    st.markdown("### 📊 Performa Model")
    st.metric(label="Akurasi Data Uji", value="96.32%")
    st.metric(label="10-Fold CV (Mean)", value="97.67%", delta="± 0.97%")
    st.markdown("""
    - **Algoritma:** Multinomial Naive Bayes ($\\alpha=1.0$)
    - **Dataset Bersih:** 1.761 Baris (KUA)
    - **Dimensi Fitur:** 32 Kolom Biner OHE
    """)
    
    st.markdown("---")
    st.markdown("### ⚡ Muat Contoh Kasus Uji")
    preset = st.radio(
        "Pilih simulasi kasus:",
        ["Input Bebas (Manual)", 
         "Kasus 1: Usia < 19 & Buruh (Tidak Layak)", 
         "Kasus 2: Usia Mapan & Karyawan Swasta + IRT (Layak)", 
         "Kasus 3: Usia Cukup & Suami Buruh + IRT (Tidak Layak)", 
         "Kasus 4: Usia Dewasa & Pasangan PNS (Layak)"],
        index=0
    )

# Konfigurasi nilai default berdasarkan preset (Default: Kosong untuk Input Bebas)
defaults = {
    "u_s": None, "p_s": None, "pend_s": None,
    "u_i": None, "p_i": None, "pend_i": None
}

if preset == "Kasus 1: Usia < 19 & Buruh (Tidak Layak)":
    defaults = {"u_s": 17, "p_s": "BURUH HARIAN LEPAS", "pend_s": "SLTA/SEDERAJAT", "u_i": 15, "p_i": "MENGURUS RUMAH TANGGA", "pend_i": "SLTP/SEDERAJAT"}
elif preset == "Kasus 2: Usia Mapan & Karyawan Swasta + IRT (Layak)":
    defaults = {"u_s": 27, "p_s": "KARYAWAN SWASTA", "pend_s": "SLTA/SEDERAJAT", "u_i": 24, "p_i": "MENGURUS RUMAH TANGGA", "pend_i": "SLTA/SEDERAJAT"}
elif preset == "Kasus 3: Usia Cukup & Suami Buruh + IRT (Tidak Layak)":
    defaults = {"u_s": 25, "p_s": "BURUH HARIAN LEPAS", "pend_s": "SLTA/SEDERAJAT", "u_i": 22, "p_i": "MENGURUS RUMAH TANGGA", "pend_i": "SLTA/SEDERAJAT"}
elif preset == "Kasus 4: Usia Dewasa & Pasangan PNS (Layak)":
    defaults = {"u_s": 28, "p_s": "PEGAWAI NEGERI SIPIL (PNS)", "pend_s": "DIPLOMA IV/STRATA I", "u_i": 26, "p_i": "PEGAWAI NEGERI SIPIL (PNS)", "pend_i": "DIPLOMA IV/STRATA I"}

# ==============================================================================
# 5. KONTEN UTAMA - FORM INPUT DATA PASANGAN
# ==============================================================================
st.markdown('<div class="main-header">Sistem Klasifikasi Kelayakan Pernikahan</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Pemeriksaan Keselarasan Regulasi UU No. 16/2019 & Kesiapan Sosial-Ekonomi Pasangan Berbasis Naive Bayes</div>', unsafe_allow_html=True)

# Pilihan Opsi Dropdown
pekerjaan_suami_options = [
    'KARYAWAN SWASTA', 'WIRASWASTA', 'BURUH HARIAN LEPAS', 'PETANI / PEKEBUN',
    'PEDAGANG', 'PEGAWAI NEGERI SIPIL (PNS)', 'GURU', 'GURU HONORER',
    'SOPIR', 'PERANGKAT DESA', 'TNI / POLISI', 'PELAJAR / MAHASISWA',
    'BELUM / TIDAK BEKERJA'
]
if defaults["p_s"] is not None and defaults["p_s"] not in pekerjaan_suami_options:
    pekerjaan_suami_options.insert(0, defaults["p_s"])

pekerjaan_istri_options = [
    'MENGURUS RUMAH TANGGA', 'BELUM / TIDAK BEKERJA', 'PELAJAR / MAHASISWA',
    'KARYAWAN SWASTA', 'WIRASWASTA', 'GURU', 'GURU HONORER',
    'BIDAN', 'PERAWAT', 'PEGAWAI NEGERI SIPIL (PNS)', 'BURUH HARIAN LEPAS'
]
if defaults["p_i"] is not None and defaults["p_i"] not in pekerjaan_istri_options:
    pekerjaan_istri_options.insert(0, defaults["p_i"])

pendidikan_options = [
    'SLTA/SEDERAJAT', 'SLTP/SEDERAJAT', 'TAMAT SD/SEDERAJAT',
    'DIPLOMA IV/STRATA I', 'AKADEMI/DIPLOMA III/S. MUDA', 'DIPLOMA I/II',
    'STRATA II', 'TIDAK TAMAT SD/SEDERAJAT'
]

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="card-header-suami">👨 Data Calon Suami (Kepala Keluarga)</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-body">', unsafe_allow_html=True)
    nama_suami = st.text_input("Nama Lengkap Calon Suami (Opsional):", placeholder="Contoh: Ahmad Fauzi")
    usia_suami = st.number_input("Usia Calon Suami (Tahun):", min_value=15, max_value=85, value=defaults["u_s"], step=1, placeholder="Masukkan usia calon suami...")
    
    idx_p_s = pekerjaan_suami_options.index(defaults["p_s"]) if defaults["p_s"] is not None else None
    pekerjaan_suami = st.selectbox("Pekerjaan Calon Suami:", options=pekerjaan_suami_options, index=idx_p_s, placeholder="-- Pilih Pekerjaan Calon Suami --")
    
    idx_pend_s = pendidikan_options.index(defaults["pend_s"]) if defaults["pend_s"] is not None else None
    pendidikan_suami = st.selectbox("Pendidikan Terakhir Suami:", options=pendidikan_options, index=idx_pend_s, placeholder="-- Pilih Pendidikan Terakhir Suami --")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card-header-istri">👩 Data Calon Istri (Mitra Pasangan)</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-body">', unsafe_allow_html=True)
    nama_istri = st.text_input("Nama Lengkap Calon Istri (Opsional):", placeholder="Contoh: Siti Nurhaliza")
    usia_istri = st.number_input("Usia Calon Istri (Tahun):", min_value=15, max_value=85, value=defaults["u_i"], step=1, placeholder="Masukkan usia calon istri...")
    
    idx_p_i = pekerjaan_istri_options.index(defaults["p_i"]) if defaults["p_i"] is not None else None
    pekerjaan_istri = st.selectbox("Pekerjaan Calon Istri:", options=pekerjaan_istri_options, index=idx_p_i, placeholder="-- Pilih Pekerjaan Calon Istri --")
    
    idx_pend_i = pendidikan_options.index(defaults["pend_i"]) if defaults["pend_i"] is not None else None
    pendidikan_istri = st.selectbox("Pendidikan Terakhir Istri:", options=pendidikan_options, index=idx_pend_i, placeholder="-- Pilih Pendidikan Terakhir Istri --")
    st.markdown('</div>', unsafe_allow_html=True)

# Tombol Klasifikasi
btn_klasifikasi = st.button("🔍 PROSES KLASIFIKASI KELAYAKAN PERNIKAHAN")

# ==============================================================================
# 6. PEMROSESAN KLASIFIKASI DAN PENAMPILAN HASIL
# ==============================================================================
if btn_klasifikasi:
    # Validasi kelengkapan data input
    if (usia_suami is None or usia_istri is None or 
        pekerjaan_suami is None or pekerjaan_istri is None or 
        pendidikan_suami is None or pendidikan_istri is None):
        st.warning("⚠️ Silakan lengkapi seluruh data (Usia, Pekerjaan, dan Pendidikan) untuk Calon Suami dan Calon Istri terlebih dahulu!")
        st.stop()
    # 1. Feature Engineering Input
    kat_u_s = kategori_usia(usia_suami)
    kat_u_i = kategori_usia(usia_istri)
    kat_p_s = kategori_pekerjaan_suami(pekerjaan_suami)
    kat_p_i = kategori_pekerjaan_istri(pekerjaan_istri)
    pend_s = standarisasi_pendidikan(pendidikan_suami)
    pend_i = standarisasi_pendidikan(pendidikan_istri)

    input_df = pd.DataFrame([{
        'Kategori_Usia_Suami': kat_u_s,
        'Kategori_Usia_Istri': kat_u_i,
        'Kategori_Pekerjaan_Suami': kat_p_s,
        'Kategori_Pekerjaan_Istri': kat_p_i,
        'Pendidikan Suami': pend_s,
        'Pendidikan Istri': pend_i
    }])

    # 2. Transformasi ke 32 Kolom Biner One-Hot Encoding
    X_input = encoder.transform(input_df[feature_cols])

    # 3. Prediksi Klasifikasi & Probabilitas Posterior
    hasil_klasifikasi = nb_model.predict(X_input)[0]
    probabilitas = dict(zip(nb_model.classes_, [round(float(p) * 100, 2) for p in nb_model.predict_proba(X_input)[0]]))

    st.markdown("---")
    st.subheader("📋 Hasil Analisis & Keputusan Klasifikasi")

    # Banner Hasil Klasifikasi
    if hasil_klasifikasi == "Layak":
        st.success(f"""
        ### 🟢 STATUS: LAYAK MENIKAH
        Calon pasangan pengantin **{nama_suami if nama_suami else 'Calon Suami'}** dan **{nama_istri if nama_istri else 'Calon Istri'}** memenuhi persyaratan hukum dan kesiapan ekonomi keluarga.
        """)
    else:
        st.error(f"""
        ### 🔴 STATUS: TIDAK LAYAK MENIKAH
        Calon pasangan pengantin **{nama_suami if nama_suami else 'Calon Suami'}** dan **{nama_istri if nama_istri else 'Calon Istri'}** dinyatakan belum memenuhi persyaratan kelayakan pernikahan.
        """)

    # Matriks Indikator & Alasan
    res_col1, res_col2 = st.columns([1.2, 1])

    with res_col1:
        st.markdown("#### 🔍 Evaluasi Indikator Penentu:")
        
        # Cek Usia Hukum
        if usia_suami >= 19 and usia_istri >= 19:
            st.markdown(f"✅ **Syarat Usia Hukum (UU No. 16/2019):** Terpenuhi (Suami: {usia_suami} th, Istri: {usia_istri} th $\\ge 19$ th).")
        else:
            alasan_usia = []
            if usia_suami < 19:
                alasan_usia.append(f"Suami berusia {usia_suami} tahun (< 19 tahun)")
            if usia_istri < 19:
                alasan_usia.append(f"Istri berusia {usia_istri} tahun (< 19 tahun)")
            st.markdown(f"❌ **Syarat Usia Hukum (UU No. 16/2019):** **Melanggar Batas Minimal Usia** ({', '.join(alasan_usia)}).")

        # Cek Kesiapan Nafkah
        if kat_p_s in ['Tetap/Mapan', 'Mandiri/Wiraswasta']:
            st.markdown(f"✅ **Kesiapan Ekonomi Kepala Keluarga:** Mandiri/Mapan ({pekerjaan_suami} masuk kategori *{kat_p_s}*).")
        elif kat_p_s == 'Tidak Tetap/Buruh' and kat_p_i == 'Bekerja/Berpenghasilan':
            st.markdown(f"✅ **Sinergi Ekonomi Pasangan:** Dual-income keluarga saling menopang ({pekerjaan_suami} ditopang oleh istri yang berpenghasilan sebagai {pekerjaan_istri}).")
        elif kat_p_s == 'Tidak Tetap/Buruh':
            st.markdown(f"⚠️ **Kesiapan Ekonomi Kepala Keluarga:** **Rentan** ({pekerjaan_suami} tidak tetap, sedangkan istri {pekerjaan_istri}).")
        else:
            st.markdown(f"❌ **Kesiapan Ekonomi Kepala Keluarga:** **Belum Siap** (Suami {pekerjaan_suami} belum memiliki penghasilan mandiri).")

    with res_col2:
        st.markdown("#### 📈 Probabilitas Naive Bayes:")
        p_layak = probabilitas.get('Layak', 0.0)
        p_tidak = probabilitas.get('Tidak Layak', 0.0)

        st.write(f"Peluang Layak: **{p_layak:.2f}%**")
        st.progress(p_layak / 100.0)

        st.write(f"Peluang Tidak Layak: **{p_tidak:.2f}%**")
        st.progress(p_tidak / 100.0)

    # Expander Detail Debug Fitur
    with st.expander("🛠️ Lihat Vektor Fitur & Detail Teknis One-Hot Encoding"):
        st.write("**Fitur Input Kategorikal:**", input_df)
        st.write(f"**Vektor Biner (32 Dimensi Kolom):** Nilai aktif (1) pada kelas input.")
        active_features = [col for col, val in zip(encoder.get_feature_names_out(feature_cols), X_input[0]) if val == 1]
        st.write("**Fitur Aktif Biner (Nilai 1):**", active_features)
