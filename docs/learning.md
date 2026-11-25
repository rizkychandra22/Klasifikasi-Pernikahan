# Dokumentasi & Panduan Pembelajaran Pipeline Machine Learning
## Klasifikasi Kelayakan Pernikahan Berdasarkan Data KUA (Nyalindung & Jampangtengah)
**Program Studi Teknik Informatika — Masket-TI22**

---

## Daftar Isi
1. [Pendahuluan & Arsitektur Sistem](#1-pendahuluan--arsitektur-sistem)
2. [Bagian I: Ekstraksi & Konsolidasi Dataset (ekstraksi_dataset.ipynb)](#2-bagian-i-ekstraksi--konsolidasi-dataset-ekstraksi_datasetipynb)
   - [2.1. Latar Belakang Masalah Data Excel Mentah](#21-latar-belakang-masalah-data-excel-mentah)
   - [2.2. Bedah Kode Cell 1: Inisialisasi Path & Pustaka](#22-bedah-kode-cell-1-inisialisasi-path--pustaka)
   - [2.3. Bedah Kode Cell 2: Fungsi Ekstraksi & Pembersihan Awal](#23-bedah-kode-cell-2-fungsi-ekstraksi--pembersihan-awal)
   - [2.4. Bedah Kode Cell 3: Batch Processing File Tahunan](#24-bedah-kode-cell-3-batch-processing-file-tahunan)
   - [2.5. Bedah Kode Cell 4: Konsolidasi ke Master Dataset (index_data.csv)](#25-bedah-kode-cell-4-konsolidasi-ke-master-dataset-index_datacsv)
   - [2.6. Bedah Kode Cell 5: Verifikasi Integritas Data Master](#26-bedah-kode-cell-5-verifikasi-integritas-data-master)
3. [Bagian II: Eksplorasi & Pembersihan Cerdas (model_klasifikasi.ipynb Cell 1–3)](#3-bagian-ii-eksplorasi--pembersihan-cerdas-model_klasifikasiipynb-cell-13)
   - [3.1. Cell 1: Import Library Machine Learning & Load Data](#31-cell-1-import-library-machine-learning--load-data)
   - [3.2. Cell 2: Data Understanding (EDA & Deteksi Anomali)](#32-cell-2-data-understanding-eda--deteksi-anomali)
   - [3.3. Cell 3: Data Cleaning & Heuristic Imputation (clean_data.csv)](#33-cell-3-data-cleaning--heuristic-imputation-clean_datacsv)
4. [Bagian III: Rekayasa Fitur & Pelabelan Ilmiah (Cell 4–5)](#4-bagian-iii-rekayasa-fitur--pelabelan-ilmiah-cell-45)
   - [4.1. Cell 4: Binning Usia, Kategorisasi Pekerjaan & Ground Truth Labeling](#41-cell-4-binning-usia-kategorisasi-pekerjaan--ground-truth-labeling)
   - [4.2. Cell 5: Stratified Split & One-Hot Encoding (Anti-Data Leakage)](#42-cell-5-stratified-split--one-hot-encoding-anti-data-leakage)
5. [Bagian IV: Pelatihan & Landasan Matematis Naive Bayes (Cell 6)](#5-bagian-iv-pelatihan--landasan-matematis-naive-bayes-cell-6)
   - [5.1. Teorema Bayes & Asumsi Independensi](#51-teorema-bayes--asumsi-independensi)
   - [5.2. Mengapa Multinomial Naive Bayes (Bukan Gaussian Naive Bayes)?](#52-mengapa-multinomial-naive-bayes-bukan-gaussian-naive-bayes)
   - [5.3. Laplace Smoothing (alpha = 1.0)](#53-laplace-smoothing-alpha--10)
6. [Bagian V: Evaluasi & Pengujian Model (Cell 7–9)](#6-bagian-v-evaluasi--pengujian-model-cell-79)
   - [6.1. Cell 7: Metrik Evaluasi (Akurasi, Presisi, Recall, F1-Score)](#61-cell-7-metrik-evaluasi-akurasi-presisi-recall-f1-score)
   - [6.2. Cell 8: Visualisasi Confusion Matrix Heatmap](#62-cell-8-visualisasi-confusion-matrix-heatmap)
   - [6.3. Cell 9: 10-Fold Cross-Validation](#63-cell-9-10-fold-cross-validation)
7. [Bagian VI: Simulasi Klasifikasi & Serialisasi Model (Cell 10–12)](#7-bagian-vi-simulasi-klasifikasi--serialisasi-model-cell-1012)
   - [7.1. Cell 10: Simulasi Klasifikasi Kasus Nyata](#71-cell-10-simulasi-klasifikasi-kasus-nyata)
   - [7.2. Cell 11: Penyimpanan Model Artifacts (.pkl)](#72-cell-11-penyimpanan-model-artifacts-pkl)
   - [7.3. Cell 12: Pengujian Load Model dari File .pkl](#73-cell-12-pengujian-load-model-dari-file-pkl)
   - [7.4. Skrip Inspeksi Pipeline & Artefak Model (check_pipeline.py)](#74-skrip-inspeksi-pipeline--artefak-model-check_pipelinepy)
8. [Panduan Pertanyaan Sidang Skripsi (FAQ Dosen Penguji)](#8-panduan-pertanyaan-sidang-skripsi-faq-dosen-penguji)

---

## 1. Pendahuluan & Arsitektur Sistem

Penelitian ini bertujuan untuk mengotomatisasi **Klasifikasi Kelayakan Pernikahan** calon pasangan pengantin berdasarkan data riil dari Kantor Urusan Agama (KUA). Klasifikasi ini berlandaskan pada dua pilar utama:
1. **Pilar Hukum Positif Indonesia:** Undang-Undang Republik Indonesia Nomor 16 Tahun 2019 tentang Perubahan atas Undang-Undang Nomor 1 Tahun 1974 tentang Perkawinan, yang menetapkan batas minimal usia pernikahan bagi pria maupun wanita adalah **19 tahun**.
2. **Pilar Kesiapan Sosial-Ekonomi Pasangan:** Kesiapan nafkah keluarga dengan mempertimbangkan peran suami sebagai pencari nafkah utama (*primary provider*) serta peran istri sebagai mitra domestik maupun penopang ekonomi keluarga (*dual-income supporter*).

### Arsitektur Aliran Data (Data Pipeline Flow)
```
[ Data-Excel/ (Laporan Bulanan 2021-2026) ]
                    │
                    ▼
       [ ekstraksi_dataset.ipynb ]
   (Normalisasi Kolom, Regex Usia, Filter)
                    │
                    ▼
     [ dataset/index_data.csv (1.765 baris) ]  <-- Data Master Mentah
                    │
                    ▼
      [ model_klasifikasi.ipynb (Cell 1-3) ]
   (Data Understanding & Heuristic Imputation)
                    │
                    ▼
     [ dataset/clean_data.csv (1.761 baris) ]  <-- Data Master Bersih
                    │
                    ▼
      [ Feature Engineering & Labeling ]
   (Binning Usia, Kategorisasi Pekerjaan, Ground Truth)
                    │
                    ▼
   [ Stratified Split (80:20) & One-Hot Encoding ]
                    │
                    ▼
       [ Multinomial Naive Bayes ]
                    │
                    ▼
   [ Evaluasi: Akurasi (96.32%), 10-Fold CV (97.67%) ]
                    │
                    ▼
   [ Folder models/: .pkl (Model, Preprocessor, Features) ]
```

---

## 2. Bagian I: Ekstraksi & Konsolidasi Dataset (`ekstraksi_dataset.ipynb`)

Notebook ini bertugas mengotomatisasi pengumpulan, penyeragaman, dan penggabungan ratusan file Excel laporan bulanan peristiwa nikah dari KUA Nyalindung menjadi satu dataset terstruktur.

### 2.1. Latar Belakang Masalah Data Excel Mentah
Data mentah yang diterima dari KUA memiliki beberapa kendala teknis:
- **Tersebar dalam banyak file:** Laporan tersimpan per bulan dalam folder tahun (`Nyalindung 2021` hingga `Nyalindung 2026`).
- **Inkonsistensi Nama Kolom:** Pada sebagian tahun kolom usia tertulis sebagai `"Umur Suami" / "Umur Istri"`, sedangkan pada tahun lain tertulis sebagai `"Usia Suami" / "Usia Istri"`.
- **Format Nilai Teks Campuran:** Kolom umur terkadang mengandung teks tambahan (misal `"24 Tahun"` atau spasi acak), sehingga harus diekstraksi nilai angkanya.
- **Nilai Kosong Bertipe String:** Terdapat sel kosong yang terbaca sebagai teks literal `"nan"` atau `"None"`.

---

### 2.2. Bedah Kode Cell 1: Inisialisasi Path & Pustaka
```python
import glob
import os
import numpy as np
import pandas as pd

raw_dir = '..' if os.path.exists(os.path.join('..', 'Data-Excel')) else '.'
data_excel = (
    os.path.join(raw_dir, 'Data-Excel')
    if os.path.exists(os.path.join(raw_dir, 'Data-Excel'))
    else raw_dir
)
dataset_dir = (
    'dataset' if os.path.exists('dataset') else os.path.join('..', 'dataset')
)
os.makedirs(dataset_dir, exist_ok=True)
```
- **Tujuan:** Menyiapkan pustaka `pandas`, `numpy`, `os`, dan `glob`, serta mendeteksi direktori kerja secara dinamis (relatif terhadap posisi eksekusi notebook) agar kode tidak error saat dijalankan di lingkungan yang berbeda.
- `os.makedirs(dataset_dir, exist_ok=True)`: Memastikan folder `dataset/` dibuat secara otomatis jika belum ada.

---

### 2.3. Bedah Kode Cell 2: Fungsi Ekstraksi & Pembersihan Awal
```python
def extract_clean(file_paths, year):
  dfs = []
  for f in sorted(file_paths):
    df = pd.read_excel(f)
    sub = pd.DataFrame()

    sub['Nama Suami'] = df['Nama Suami'].astype(str).str.strip()
    sub['Nama Istri'] = df['Nama Istri'].astype(str).str.strip()

    col_us = 'Umur Suami' if 'Umur Suami' in df.columns else 'Usia Suami'
    col_ui = 'Umur Istri' if 'Umur Istri' in df.columns else 'Usia Istri'
    sub['Usia Suami'] = (
        df[col_us].astype(str).str.extract(r'(\d+)')[0].astype(int)
    )
    sub['Usia Istri'] = (
        df[col_ui].astype(str).str.extract(r'(\d+)')[0].astype(int)
    )

    sub['Pendidikan Suami'] = (
        df['Pendidikan Suami']
        .astype(str)
        .str.strip()
        .replace({'nan': np.nan, 'None': np.nan})
    )
    sub['Pendidikan Istri'] = (
        df['Pendidikan Istri']
        .astype(str)
        .str.strip()
        .replace({'nan': np.nan, 'None': np.nan})
    )

    sub['Pekerjaan Suami'] = df['Pekerjaan Suami'].astype(str).str.strip()
    sub['Pekerjaan Istri'] = df['Pekerjaan Istri'].astype(str).str.strip()
    sub['Tahun'] = year
    dfs.append(sub)

  return pd.concat(dfs, ignore_index=True)
```
- **Penanganan Kolom Dinamis:**
  ```python
  col_us = 'Umur Suami' if 'Umur Suami' in df.columns else 'Usia Suami'
```
  Memeriksa keberadaan kolom terlebih dahulu sehingga skrip tidak mengalami `KeyError`.
- **Ekstraksi Regular Expression (Regex):**
  ```python
  sub['Usia Suami'] = (
      df[col_us].astype(str).str.extract(r'(\d+)')[0].astype(int)
  )
```
  Pola `r'(\d+)'` mengambil digit angka pertama yang ditemukan dari teks dan mengonversinya langsung menjadi bilangan bulat (*integer*).
- **Normalisasi Nilai Kosong:** Nilai teks `'nan'` dan `'None'` digantikan menjadi objek `np.nan` asli agar dikenali oleh pustaka statistik sebagai *missing value*.

---

### 2.4. Bedah Kode Cell 3: Batch Processing File Tahunan
```python
tasks = [
    ('Nyalindung 2021', 'nyalindung_21.csv', 2021),
    ('Nyalindung 2022', 'nyalindung_22.csv', 2022),
    ('Nyalindung 2023', 'nyalindung_23.csv', 2023),
    ('Nyalindung 2024', 'nyalindung_24.csv', 2024),
    ('Nyalindung 2025', 'nyalindung_25.csv', 2025),
    ('Nyalindung 2026', 'nyalindung_26.csv', 2026),
]

for folder, outfile, year in tasks:
  files = sorted(glob.glob(os.path.join(data_excel, folder, '*.xlsx')))
  if files:
    df_clean = extract_clean(files, year)
    df_clean.to_csv(
        os.path.join(dataset_dir, outfile), index=False, encoding='utf-8'
    )
    print(f'{outfile}: {len(df_clean)} rows')
```
- `glob.glob(...)`: Mencari seluruh berkas berekstensi `.xlsx` dalam masing-masing folder tahun.
- Menghasilkan 6 berkas CSV tahunan:
  - `nyalindung_21.csv` : 453 baris
  - `nyalindung_22.csv` : 378 baris
  - `nyalindung_23.csv` : 405 baris
  - `nyalindung_24.csv` : 59 baris
  - `nyalindung_25.csv` : 268 baris
  - `nyalindung_26.csv` : 202 baris

---

### 2.5. Bedah Kode Cell 4: Konsolidasi ke Master Dataset (`index_data.csv`)
```python
csv_files = [
    'nyalindung_21.csv',
    'nyalindung_22.csv',
    'nyalindung_23.csv',
    'nyalindung_24.csv',
    'nyalindung_25.csv',
    'nyalindung_26.csv',
]

dfs = [
    pd.read_csv(os.path.join(dataset_dir, f))
    for f in csv_files
    if os.path.exists(os.path.join(dataset_dir, f))
]
index_data = pd.concat(dfs, ignore_index=True)
index_data.to_csv(
    os.path.join(dataset_dir, 'index_data.csv'), index=False, encoding='utf-8'
)
print(f'index_data.csv: {len(index_data)} rows')
```
- Menggabungkan (*concatenation*) seluruh baris data tahunan menjadi satu kesatuan data master: **1.765 baris**.

---

### 2.6. Bedah Kode Cell 5: Verifikasi Integritas Data Master
```python
index_data.info()
index_data.head()
```
- Menampilkan metadata dataset master: 9 kolom standar (`Nama Suami`, `Nama Istri`, `Usia Suami`, `Usia Istri`, `Pendidikan Suami`, `Pendidikan Istri`, `Pekerjaan Suami`, `Pekerjaan Istri`, `Tahun`).

---

## 3. Bagian II: Eksplorasi & Pembersihan Cerdas (`model_klasifikasi.ipynb` Cell 1–3)

Bagian ini merupakan tahap krusial dalam metodologi data science (*CRISP-DM / KDD*) sebelum data masuk ke tahap rekayasa fitur dan pemodelan machine learning.

### 3.1. Cell 1: Import Library Machine Learning & Load Data
```python
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder

dataset_path = (
    'dataset/index_data.csv'
    if os.path.exists('dataset/index_data.csv')
    else '../dataset/index_data.csv'
)
df_raw = pd.read_csv(dataset_path)
print(
    f'Dataset mentah berhasil dimuat: {df_raw.shape[0]} baris,'
    f' {df_raw.shape[1]} kolom'
)
df_raw.head()
```
- Memuat pustaka `sklearn` untuk pemodelan dan evaluasi, `joblib` untuk serialisasi model, dan dataset mentah `index_data.csv`.

---

### 3.2. Cell 2: Data Understanding (EDA & Deteksi Anomali)
```python
print("=== INFORMASI STRUKTUR DATASET ===")
df_raw.info()

print("\n=== STATISTIK DESKRIPTIF USIA ===")
print(df_raw[['Usia Suami', 'Usia Istri']].describe())

print("\n=== PENGECEKAN MISSING VALUES ===")
print(df_raw.isnull().sum())

print("\n=== PENGECEKAN DATA DUPLIKAT ===")
print(f"Jumlah baris duplikat: {df_raw.duplicated().sum()}")

# Pengecekan Anomali Nilai
anomali_usia = df_raw[(df_raw['Usia Suami'] <= 0) | (df_raw['Usia Istri'] <= 0)]
anomali_pekerjaan = df_raw[
    (df_raw['Pekerjaan Suami'].str.strip() == '-')
    | (df_raw['Pekerjaan Istri'].str.strip() == '-')
]

print(f"\nDeteksi Anomali:")
print(f" - Baris dengan Usia <= 0 tahun  : {len(anomali_usia)} baris")
print(f" - Baris dengan Pekerjaan '-'    : {len(anomali_pekerjaan)} baris")
print(
    f" - Baris Pendidikan Kosong (NaN) :"
    f" {df_raw['Pendidikan Suami'].isnull().sum()} Suami,"
    f" {df_raw['Pendidikan Istri'].isnull().sum()} Istri"
)
```
- **Tujuan Akademis:** Menganalisis kondisi riil data sebelum manipulasi.
- **Temuan Data Understanding:**
  1. Terdapat **1 baris anomali usia**: Usia Istri bernilai `0` tahun pada baris index 438 (kesalahan pencatatan administratif di KUA).
  2. Terdapat **3 baris anomali pekerjaan**: Pekerjaan Istri hanya berupa simbol minus `"-"` tanpa keterangan.
  3. Terdapat **76 nilai kosong (NaN)** pada kolom `Pendidikan Suami` dan `Pendidikan Istri`.
  4. Baris duplikat bernilai `0` (tidak ada catatan ganda).

---

### 3.3. Cell 3: Data Cleaning & Heuristic Imputation (`clean_data.csv`)
```python
# A. Hapus baris anomali pada fitur primer (Usia <= 0 dan Pekerjaan '-')
df_clean = df_raw[(df_raw['Usia Suami'] > 0) & (df_raw['Usia Istri'] > 0)].copy()
df_clean = df_clean[
    (df_clean['Pekerjaan Suami'].str.strip() != '-')
    & (df_clean['Pekerjaan Istri'].str.strip() != '-')
].copy()


# B. Standarisasi Teks Jenjang Pendidikan
def standarisasi_pendidikan(p):
  if pd.isnull(p):
    return p
  p = str(p).upper().strip()
  if (
      'STRATA II' in p
      or 'STRATA-II' in p
      or 'STRATA III' in p
      or 'S2' in p
      or 'S3' in p
  ):
    return 'STRATA II'
  elif 'STRATA I' in p or 'STRATA-I' in p or 'DIPLOMA IV' in p or 'S1' in p:
    return 'DIPLOMA IV/STRATA I'
  elif (
      'DIPLOMA III' in p
      or 'S. MUDA' in p
      or 'SARJANA MUDA' in p
      or 'D3' in p
  ):
    return 'AKADEMI/DIPLOMA III/S. MUDA'
  elif 'DIPLOMA I' in p or 'DIPLOMA II' in p or 'D1' in p or 'D2' in p:
    return 'DIPLOMA I/II'
  elif 'SLTA' in p or 'SMA' in p or 'SMK' in p or 'ALIYAH' in p:
    return 'SLTA/SEDERAJAT'
  elif 'SLTP' in p or 'SMP' in p or 'MTS' in p or 'TSANAWIYAH' in p:
    return 'SLTP/SEDERAJAT'
  elif (
      'TIDAK TAMAT SD' in p
      or 'TIDAK/BELUM SEKOLAH' in p
      or 'TIDAK/BLM SEKOLAH' in p
  ):
    return 'TIDAK TAMAT SD/SEDERAJAT'
  elif 'SD' in p or 'IBTIDAIYAH' in p:
    return 'TAMAT SD/SEDERAJAT'
  return p


df_clean['Pendidikan Suami'] = df_clean['Pendidikan Suami'].apply(
    standarisasi_pendidikan
)
df_clean['Pendidikan Istri'] = df_clean['Pendidikan Istri'].apply(
    standarisasi_pendidikan
)


# C. Imputasi Cerdas Pendidikan Berdasarkan Korelasi Usia & Pekerjaan (Domain-Knowledge Heuristic)
def imputasi_pendidikan(pendidikan, usia, pekerjaan):
  if (
      pd.notnull(pendidikan)
      and str(pendidikan).strip() not in ['', 'NAN', 'NONE', 'TIDAK DIKETAHUI']
  ):
    return pendidikan
  p = str(pekerjaan).upper().strip()
  u = int(usia)

  # 1. Profesi formal berjenjang tinggi
  if any(
      k in p for k in ['PNS', 'DOKTER', 'DOSEN', 'BIDAN', 'PERAWAT', 'GURU']
  ):
    return 'DIPLOMA IV/STRATA I'
  # 2. Profesi mapan / swasta / wiraswasta
  if any(
      k in p
      for k in [
          'KARYAWAN SWASTA',
          'BUMD',
          'BUMN',
          'PERANGKAT DESA',
          'PPPK',
          'WIRASWASTA',
          'PEDAGANG',
          'POLISI',
          'TNI',
      ]
  ):
    return 'SLTA/SEDERAJAT'
  # 3. Profesi buruh / petani / sopir
  if any(k in p for k in ['BURUH', 'PETANI', 'PETERNAK', 'SOPIR']):
    return 'TAMAT SD/SEDERAJAT' if u >= 35 else 'SLTP/SEDERAJAT'
  # 4. Pelajar / IRT / Belum bekerja
  if u < 19:
    return 'SLTP/SEDERAJAT'
  else:
    return 'SLTA/SEDERAJAT'


df_clean['Pendidikan Suami'] = df_clean.apply(
    lambda r: imputasi_pendidikan(
        r['Pendidikan Suami'], r['Usia Suami'], r['Pekerjaan Suami']
    ),
    axis=1,
)
df_clean['Pendidikan Istri'] = df_clean.apply(
    lambda r: imputasi_pendidikan(
        r['Pendidikan Istri'], r['Usia Istri'], r['Pekerjaan Istri']
    ),
    axis=1,
)

# Simpan ke dataset/clean_data.csv
clean_dir = 'dataset' if os.path.exists('dataset') else '../dataset'
clean_path = os.path.join(clean_dir, 'clean_data.csv')
df_clean.to_csv(clean_path, index=False)
```

#### Alasan Ilmiah Pembersihan Data:
1. **Listwise Deletion untuk Fitur Primer:** Fitur Usia dan Pekerjaan adalah variabel inti status kelayakan. Mengisi usia 0 dengan tebakan dapat menghasilkan bias. Mengingat hanya 4 baris yang anomali ($< 0.23\%$ dari total data), penghapusan baris adalah tindakan paling objektif.
2. **Pencegahan Jebakan Substring pada Standarisasi:**
   Kata `"TAMAT SD/SEDERAJATNYA"` memiliki substring `"MA"` di dalam kata `"TAMAT"`. Jika pengecekan Madrasah Aliyah (`MA`) dilakukan dengan pencarian substring biasa, teks tamat SD akan salah terpetakan menjadi SLTA. Oleh karena itu, pengecekan kata dibuat spesifik (`'ALIYAH'`, `'SMA'`, `'SMK'`).
3. **Domain-Knowledge Heuristic Imputation:**
   Alih-alih mengisi 76 data kosong dengan *modus* (yang bias ke nilai terbanyak) atau `'TIDAK DIKETAHUI'` (yang merusak pola fitur), kita memanfaatkan korelasi sosiologis antara usia dan profesi. Seseorang dengan profesi PNS/Bidan/Guru secara struktural mempersyaratkan ijazah D4/S1, sementara pekerja buruh di daerah usia matang umumnya berpendidikan dasar.
4. **Hasil Ekspor Data Bersih:** Dihasilkan file `clean_data.csv` berisi **1.761 baris, 9 kolom, dengan 0 missing value**.

---

## 4. Bagian III: Rekayasa Fitur & Pelabelan Ilmiah (Cell 4–5)

### 4.1. Cell 4: Binning Usia, Kategorisasi Pekerjaan & Ground Truth Labeling
```python
df = pd.read_csv(
    'dataset/clean_data.csv'
    if os.path.exists('dataset/clean_data.csv')
    else '../dataset/clean_data.csv'
)


# A. Kategori Kemapanan Pekerjaan Suami (Nafkah Utama)
def kategori_pekerjaan_suami(p):
  p = str(p).upper().strip()
  if any(
      k in p
      for k in [
          'PNS',
          'POLISI',
          'TNI',
          'TENTARA',
          'BUMD',
          'BUMN',
          'DOKTER',
          'DOSEN',
          'GURU',
          'PERAWAT',
          'PERANGKAT DESA',
          'PPPK',
          'KARYAWAN SWASTA',
      ]
  ):
    return 'Tetap/Mapan'
  elif any(
      k in p
      for k in [
          'WIRASWASTA',
          'PEDAGANG',
          'PERDAGANGAN',
          'PETANI',
          'PETERNAK',
          'PETAMBAK',
          'TAMBANG',
      ]
  ):
    return 'Mandiri/Wiraswasta'
  elif any(k in p for k in ['BURUH', 'SOPIR', 'HONORER', 'SENI']):
    return 'Tidak Tetap/Buruh'
  else:
    return 'Tidak Bekerja'


# B. Kategori Peran Ekonomi Istri (Penopang Sekunder / Mitra Domestik)
def kategori_pekerjaan_istri(p):
  p = str(p).upper().strip()
  if 'MENGURUS RUMAH TANGGA' in p:
    return 'Mengurus Rumah Tangga'
  elif any(
      k in p
      for k in [
          'PNS',
          'POLISI',
          'TNI',
          'BUMD',
          'BUMN',
          'DOKTER',
          'DOSEN',
          'PERAWAT',
          'BIDAN',
          'KARYAWAN SWASTA',
          'WIRASWASTA',
      ]
  ) or p in ['GURU', 'GURU HONORER']:
    return 'Bekerja/Berpenghasilan'
  elif 'BURUH' in p:
    return 'Pekerja Tidak Tetap'
  else:
    return 'Belum Bekerja/Pelajar'


# C. Binning Usia Berdasarkan Kategori Hukum & Kematangan
def kategori_usia(u):
  if u < 19:
    return '< 19 (Di Bawah Umur)'
  elif u <= 24:
    return '19-24 (Muda)'
  elif u <= 30:
    return '25-30 (Matang)'
  else:
    return '> 30 (Dewasa)'


df['Kategori_Pekerjaan_Suami'] = df['Pekerjaan Suami'].apply(
    kategori_pekerjaan_suami
)
df['Kategori_Pekerjaan_Istri'] = df['Pekerjaan Istri'].apply(
    kategori_pekerjaan_istri
)
df['Kategori_Usia_Suami'] = df['Usia Suami'].apply(kategori_usia)
df['Kategori_Usia_Istri'] = df['Usia Istri'].apply(kategori_usia)

# D. Penentuan Target Label Kelayakan (Ground Truth)
syarat_usia_sah = (df['Usia Suami'] >= 19) & (df['Usia Istri'] >= 19)

suami_mapan = df['Kategori_Pekerjaan_Suami'].isin(
    ['Tetap/Mapan', 'Mandiri/Wiraswasta']
)
suami_buruh_istri_bekerja = (
    df['Kategori_Pekerjaan_Suami'] == 'Tidak Tetap/Buruh'
) & (df['Kategori_Pekerjaan_Istri'] == 'Bekerja/Berpenghasilan')

syarat_ekonomi_layak = suami_mapan | suami_buruh_istri_bekerja
df['Status_Kelayakan'] = np.where(
    syarat_usia_sah & syarat_ekonomi_layak, 'Layak', 'Tidak Layak'
)
```

#### Logika Formal Ground Truth Labeling:
Status kelayakan pernikahan dimodelkan secara matematis dengan logika proposisional:

$$\text{Status Kelayakan} = \text{Syarat Usia Sah} \land \text{Syarat Ekonomi Layak}$$

Di mana:
1. $\text{Syarat Usia Sah} = (\text{Usia Suami} \ge 19) \land (\text{Usia Istri} \ge 19)$
2. $\text{Syarat Ekonomi Layak} = (\text{Pekerjaan Suami} \in \{\text{Mapan, Mandiri}\}) \lor (\text{Pekerjaan Suami} = \text{Buruh} \land \text{Pekerjaan Istri} = \text{Berpenghasilan})$

**Tabel Keputusan Kelayakan Sinergi Pasangan:**
| Usia Pasangan | Pekerjaan Suami | Pekerjaan Istri | Status Kelayakan | Alasan Logis |
| :---: | :---: | :---: | :---: | :--- |
| Salah satu $< 19$ | Apa saja | Apa saja | **Tidak Layak** | Melanggar batas usia UU No. 16/2019 (Pernikahan Dini) |
| Keduanya $\ge 19$ | Tetap / Mapan | Mengurus Rumah Tangga | **Layak** | Suami mapan menafkahi istri sebagai ibu rumah tangga (Ideal) |
| Keduanya $\ge 19$ | Mandiri / Wiraswasta | Bekerja / Berpenghasilan | **Layak** | Keluarga mapan dengan *dual-income* |
| Keduanya $\ge 19$ | Tidak Tetap / Buruh | Bekerja / Berpenghasilan | **Layak** | Penghasilan istri menopang kerentanan nafkah suami buruh |
| Keduanya $\ge 19$ | Tidak Tetap / Buruh | Mengurus Rumah Tangga | **Tidak Layak** | Suami tidak tetap & tidak ada penopang finansial lain |
| Keduanya $\ge 19$ | Tidak Bekerja | Apa saja | **Tidak Layak** | Kepala keluarga belum mandiri menafkahi rumah tangga |

---

### 4.2. Cell 5: Stratified Split & One-Hot Encoding (Anti-Data Leakage)
```python
feature_cols = [
    'Kategori_Usia_Suami',
    'Kategori_Usia_Istri',
    'Kategori_Pekerjaan_Suami',
    'Kategori_Pekerjaan_Istri',
    'Pendidikan Suami',
    'Pendidikan Istri',
]

X_raw = df[feature_cols]
y = df['Status_Kelayakan']

# A. Stratified Split Data terlebih dahulu (80% Latih, 20% Uji)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)

# B. Fit One-Hot Encoder HANYA pada data latih, lalu transform data uji
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train = encoder.fit_transform(X_train_raw)
X_test = encoder.transform(X_test_raw)
```

#### Prinsip Metodologi: Mencegah Kebocoran Data (*Data Leakage*)
- **Kesalahan Umum:** Melakukan `encoder.fit_transform(X)` pada *seluruh* dataset sebelum pemisahan data latih/uji. Hal ini menyebabkan informasi sebaran kategori pada data uji ikut terbaca saat pembentukan encoder (*data leakage*).
- **Prosedur Baku:** 
  1. Data dipecah terlebih dahulu dengan `train_test_split(..., stratify=y)`. Parameter `stratify=y` memastikan rasio kelas Layak (21.3%) dan Tidak Layak (78.7%) seimbang di data latih maupun uji.
  2. Encoder hanya mempelajari kamus kategori dari data latih (`fit_transform(X_train_raw)`).
  3. Data uji diperlakukan murni sebagai data unseen yang hanya diubah bentuknya (`transform(X_test_raw)`).

---

## 5. Bagian IV: Pelatihan & Landasan Matematis Naive Bayes (Cell 6)

```python
# 6. Training Model Multinomial Naive Bayes
nb_model = MultinomialNB(alpha=1.0)  # alpha=1.0 adalah Laplace Smoothing
nb_model.fit(X_train, y_train)
```

### 5.1. Teorema Bayes & Asumsi Independensi
Algoritma Naive Bayes mengklasifikasikan data ke dalam kelas $C_k \in \{\text{Layak}, \text{Tidak Layak}\}$ yang memaksimalkan probabilitas *posterior* berdasarkan vektor fitur biner $X = (x_1, x_2, \dots, x_n)$:

$$P(C_k | X) = \frac{P(C_k) \prod_{i=1}^{n} P(x_i | C_k)}{P(X)}$$

Karena penyebut $P(X)$ bernilai konstan untuk semua kelas, keputusan klasifikasi diambil menggunakan aturan argmax:

$$\hat{y} = \arg\max_{C_k} \left[ P(C_k) \prod_{i=1}^{n} P(x_i | C_k) \right]$$

### 5.2. Mengapa Multinomial Naive Bayes (Bukan Gaussian Naive Bayes)?
- **Kegagalan Gaussian Naive Bayes pada Kasus Ini (The Bell Curve Trap):**  
  Gaussian Naive Bayes mengasumsikan fitur berdistribusi kontinu normal (kurva lonceng Gauss). Ketika usia 15 atau 17 tahun dimasukkan ke Gaussian NB, variansi data usia pada kelas "Layak" yang cukup lebar memberikan nilai densitas probabilitas non-nol $P(\text{Usia}=17 | \text{Layak}) > 0$. Akibatnya, pasangan di bawah umur dengan suami karyawan swasta justru terprediksi "Layak" secara keliru (*false positive*).
- **Keunggulan Multinomial Naive Bayes:**  
  Dengan melakukan *binning* usia dan *One-Hot Encoding*, semua fitur berubah menjadi variabel frekuensi diskrit (kolom biner 0 dan 1). Multinomial NB menghitung probabilitas kemunculan fitur diskrit secara murni:
  
  $$P(x_i | C_k) = \frac{N_{ki} + \alpha}{N_k + \alpha d}$$
  
  Di mana $N_{ki}$ adalah frekuensi fitur $x_i$ pada kelas $C_k$, $N_k$ total frekuensi semua fitur di kelas $C_k$, $\alpha$ parameter smoothing, dan $d$ jumlah fitur. Fitur `Kategori_Usia = < 19` memiliki frekuensi 0 pada kelas Layak, sehingga probabilitas Layak menjadi mendekati nol secara pasti.

### 5.3. Laplace Smoothing ($\alpha = 1.0$)
Jika ada kombinasi kategori yang tidak pernah muncul pada data latih di kelas tertentu ($N_{ki} = 0$), perkalian probabilitas akan bernilai nol mutlak (*Zero Probability Problem*). Laplace smoothing ($\alpha = 1.0$) memberikan bobot kecil sehingga model tetap mampu menghitung probabilitas secara stabil tanpa pembagian dengan nol.

---

## 6. Bagian V: Evaluasi & Pengujian Model (Cell 7–9)

### 6.1. Cell 7: Metrik Evaluasi (Akurasi, Presisi, Recall, F1-Score)
```python
y_pred = nb_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=['Layak', 'Tidak Layak'])
print(classification_report(y_test, y_pred))
```

#### Hasil Pengujian Data Uji (353 Pasangan):
- **Akurasi Total:** **96.32%**
- **Matriks Konfusi (Confusion Matrix):**
  $$\begin{pmatrix} 72 & 3 \\ 10 & 268 \end{pmatrix}$$
  - **True Positive (TP = 72):** Pasangan Layak yang tepat terklasifikasi Layak.
  - **False Negative (FN = 3):** Pasangan Layak yang terklasifikasi Tidak Layak.
  - **False Positive (FP = 10):** Pasangan Tidak Layak yang terklasifikasi Layak.
  - **True Negative (TN = 268):** Pasangan Tidak Layak yang tepat terklasifikasi Tidak Layak.

- **Rincian Metrik Per Kelas:**
  - **Kelas Layak:** Presisi = 0.88 | Recall = 0.96 | F1-Score = 0.92
  - **Kelas Tidak Layak:** Presisi = 0.99 | Recall = 0.96 | F1-Score = 0.98

---

### 6.2. Cell 8: Visualisasi Confusion Matrix Heatmap
```python
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Layak', 'Tidak Layak'],
    yticklabels=['Layak', 'Tidak Layak'],
)
plt.title('Confusion Matrix - Multinomial Naive Bayes (Clean Data)')
plt.xlabel('Klasifikasi Model')
plt.ylabel('Nilai Aktual')
plt.tight_layout()
plt.show()
```
- Menampilkan grafik matriks konfusi dengan sumbu horizontal berlabel **"Klasifikasi Model"** (bukan prediksi) sesuai kaidah klasifikasi biner.

---

### 6.3. Cell 9: 10-Fold Cross-Validation
```python
X_all = encoder.transform(X_raw)
cv_scores = cross_val_score(nb_model, X_all, y, cv=10, scoring='accuracy')
```
- **Tujuan:** Menguji apakah model mengalami *overfitting* atau sensitif terhadap pembagian data tertentu. Dataset dibagi menjadi 10 lipatan (*folds*), di mana setiap lipatan bergantian menjadi data uji sementara 9 lipatan lainnya menjadi data latih.
- **Hasil 10-Fold CV:**
  - Fold 1: 98.87% | Fold 2: 97.16% | Fold 3: 97.73% | Fold 4: 97.16% | Fold 5: 98.30%
  - Fold 6: 98.86% | Fold 7: 98.30% | Fold 8: 97.16% | Fold 9: 97.73% | Fold 10: 95.45%
  - **Rata-rata Akurasi (Mean CV):** **97.67% ($\pm 0.97\%$)**
- **Kesimpulan:** Standar deviasi sangat kecil ($< 1\%$), membuktikan model sangat stabil dan memiliki kemampuan generalisasi (*generalisability*) yang kuat pada data baru.

---

## 7. Bagian VI: Simulasi Klasifikasi & Serialisasi Model (Cell 10–12)

### 7.1. Cell 10: Simulasi Klasifikasi Kasus Nyata
```python
def klasifikasi_kelayakan(
    usia_suami,
    usia_istri,
    pekerjaan_suami,
    pekerjaan_istri,
    pendidikan_suami='SLTA/SEDERAJAT',
    pendidikan_istri='SLTA/SEDERAJAT',
):
  input_raw = pd.DataFrame([{
      'Kategori_Usia_Suami': kategori_usia(usia_suami),
      'Kategori_Usia_Istri': kategori_usia(usia_istri),
      'Kategori_Pekerjaan_Suami': kategori_pekerjaan_suami(pekerjaan_suami),
      'Kategori_Pekerjaan_Istri': kategori_pekerjaan_istri(pekerjaan_istri),
      'Pendidikan Suami': standarisasi_pendidikan(pendidikan_suami),
      'Pendidikan Istri': standarisasi_pendidikan(pendidikan_istri),
  }])
  input_encoded = encoder.transform(input_raw[feature_cols])
  hasil = nb_model.predict(input_encoded)[0]
  prob = {
      str(k): round(float(v) * 100, 2)
      for k, v in zip(nb_model.classes_, nb_model.predict_proba(input_encoded)[0])
  }
  return hasil, prob
```

#### Hasil 4 Kasus Pengujian:
1. **Kasus 1 (Usia 17 & 15, Suami Buruh Lepas, Istri IRT):**  
   - Output: **Tidak Layak** (Probabilitas: Tidak Layak **99.11%** | Layak 0.89%)  
   - *Analisis:* Usia di bawah 19 tahun dan suami belum mapan secara mutlak diklasifikasikan Tidak Layak.
2. **Kasus 2 (Usia 27 & 24, Suami Karyawan Swasta, Istri IRT):**  
   - Output: **Layak** (Probabilitas: Layak **95.28%** | Tidak Layak 4.72%)  
   - *Analisis:* Usia dewasa dan suami mapan menafkahi istri sebagai ibu rumah tangga.
3. **Kasus 3 (Usia 25 & 22, Suami Buruh Lepas, Istri IRT):**  
   - Output: **Tidak Layak** (Probabilitas: Tidak Layak **93.88%** | Layak 6.12%)  
   - *Analisis:* Walaupun usia sah secara hukum, status suami buruh lepas tanpa penghasilan tambahan istri dinilai belum mapan secara ekonomi.
4. **Kasus 4 (Usia 28 & 26, Suami PNS, Istri PNS):**  
   - Output: **Layak** (Probabilitas: Layak **99.94%** | Tidak Layak 0.06%)  
   - *Analisis:* Keluarga dengan kemapanan karir ganda (*dual career*).

---

### 7.2. Cell 11: Penyimpanan Model Artifacts (`.pkl`)
```python
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

joblib.dump(nb_model, os.path.join(models_dir, 'naive_bayes.pkl'))
joblib.dump(encoder, os.path.join(models_dir, 'preprocessor.pkl'))
joblib.dump(feature_cols, os.path.join(models_dir, 'features.pkl'))
```
- Menyimpan 3 berkas biner serialisasi:
  1. `naive_bayes.pkl`: Struktur bobot model probabilitas Multinomial Naive Bayes.
  2. `preprocessor.pkl`: Objek OneHotEncoder yang menyimpan matriks kolom biner hasil transformasi.
  3. `features.pkl`: Daftar urutan nama kolom fitur (`feature_cols`) agar input data baru tidak tertukar posisinya.

---

### 7.3. Cell 12: Pengujian Load Model dari File `.pkl`
```python
loaded_model = joblib.load(os.path.join(models_dir, 'naive_bayes.pkl'))
loaded_encoder = joblib.load(os.path.join(models_dir, 'preprocessor.pkl'))
loaded_features = joblib.load(os.path.join(models_dir, 'features.pkl'))

test_X = loaded_encoder.transform(test_input[loaded_features])
pred_pkl = loaded_model.predict(test_X)[0]
print(f'Verifikasi Hasil Klasifikasi dari .pkl: {pred_pkl}')
```
- Memastikan bahwa model yang disimpan ke dalam file `.pkl` dapat dimuat kembali (*load*) secara independen dan menghasilkan klasifikasi yang sama persis saat diimplementasikan ke aplikasi web nanti.

---

### 7.4. Skrip Inspeksi Pipeline & Artefak Model (`check_pipeline.py`)

Untuk membedah, memverifikasi struktur biner, dan melihat isi matematis di dalam ketiga file `.pkl` tanpa harus membuka file notebook, telah disediakan skrip mandiri [check_pipeline.py](../check_pipeline.py).

#### Cara Menjalankan:
```bash
python check_pipeline.py
```

#### Struktur & Informasi yang Dibongkar oleh Skrip:
1. **Pemeriksaan `features.pkl`:**
   - Menyimpan objek bertipe `list` berisi 6 string urutan fitur:
     `['Kategori_Usia_Suami', 'Kategori_Usia_Istri', 'Kategori_Pekerjaan_Suami', 'Kategori_Pekerjaan_Istri', 'Pendidikan Suami', 'Pendidikan Istri']`.
   - **Tujuan Teknis:** Memastikan urutan kolom data input dari antarmuka web selalu selaras dengan urutan pembelajaran model.

2. **Pemeriksaan `preprocessor.pkl` (`OneHotEncoder`):**
   - Atribut `encoder.categories_`: Menampilkan seluruh nilai kategori sah yang dipelajari pada data latih (4 kategori usia suami, 4 usia istri, 4 pekerjaan suami, 4 pekerjaan istri, 8 pendidikan suami, 8 pendidikan istri).
   - Atribut `encoder.get_feature_names_out()`: Memetakan 6 fitur kategorikal tersebut menjadi tepat **32 kolom biner (0 dan 1)**.

3. **Pemeriksaan `naive_bayes.pkl` (`MultinomialNB`):**
   - **Kelas Target:** `['Layak', 'Tidak Layak']`.
   - **Sampel Latih:** 300 sampel Layak (21.31%) dan 1.108 sampel Tidak Layak (78.69%) dari total 1.408 data latih.
   - **Prior Probability P(Kelas):**
     - $P(\text{Layak}) = 0.2131$ (21.31%)
     - $P(\text{Tidak Layak}) = 0.7869$ (78.69%)
   - **Likelihood Matrix $P(x_i \mid C_k)$:**  
     Matriks berukuran $2 \times 32$ yang menyimpan bobot kondisional setiap fitur. Skrip membuktikan secara matematis mengapa status pekerjaan suami sangat menentukan hasil:
     - $P(\text{Suami Tetap/Mapan} \mid \text{Layak}) = \mathbf{0.0786}$ vs $P(\text{Suami Tetap/Mapan} \mid \text{Tidak Layak}) = \mathbf{0.0013}$ (**~60x lebih tinggi** pada kelas Layak).
     - $P(\text{Suami Mandiri/Wiraswasta} \mid \text{Layak}) = \mathbf{0.0835}$ vs $P(\text{Suami Mandiri/Wiraswasta} \mid \text{Tidak Layak}) = \mathbf{0.0008}$ (**~100x lebih tinggi** pada kelas Layak).

---

## 8. Panduan Pertanyaan Sidang Skripsi (FAQ Dosen Penguji)

### Q1: Mengapa menggunakan istilah "Klasifikasi", bukan "Prediksi"?
> **Jawaban:** Tugas machine learning pada penelitian ini bertujuan memetakan data pasangan ke dalam kelas diskrit kategorikal saling lepas (*binary categorical classes*), yaitu **Layak** atau **Tidak Layak**, bukan meramalkan angka kontinu (*forecasting/regression*). Dalam taksonomi supervised learning, tugas pemetaan label diskrit disebut secara baku sebagai **Klasifikasi (*Classification*)**.

### Q2: Mengapa memilih algoritma Multinomial Naive Bayes, bukan Gaussian Naive Bayes?
> **Jawaban:** Data pernikahan KUA mayoritas bertipe data kategorikal (pekerjaan dan pendidikan). Setelah variabel usia dilakukan *binning* dan seluruh fitur ditransformasikan menggunakan *One-Hot Encoding*, ruang fitur berbentuk representasi diskrit/biner (0 dan 1). Gaussian Naive Bayes mengasumsikan distribusi normal kontinu yang rentan memunculkan probabilitas keliru (*false positive*) pada rentang usia muda akibat variansi kurva lonceng yang lebar. Multinomial Naive Bayes secara matematis menghitung proporsi frekuensi diskrit biner murni sehingga jauh lebih akurat dan menghasilkan akurasi 96.32% serta stabilitas CV 97.67%.

### Q3: Bagaimana Anda menangani risiko kebocoran data (*Data Leakage*) pada tahap One-Hot Encoding?
> **Jawaban:** Pemisahan dataset menjadi data latih (80%) dan data uji (20%) dilakukan **terlebih dahulu** menggunakan `train_test_split`. Objek `OneHotEncoder` hanya di-*fit* pada data latih (`X_train_raw`). Data uji (`X_test_raw`) hanya dikenakan fungsi `transform` tanpa proses pembelajaran baru. Dengan demikian, model sama sekali tidak melihat distribusi maupun variasi data uji sebelum proses evaluasi.

### Q4: Mengapa data pendidikan yang kosong diisi menggunakan imputasi korelasi usia-pekerjaan, bukan nilai modus atau drop baris?
> **Jawaban:** Kolom pendidikan memiliki 76 baris kosong. Menghapusnya akan membuang informasi berharga dari pasangan yang bersangkutan. Mengisinya dengan modus semata akan mengabaikan logika sosial-profesi. Oleh karena itu, diterapkan teknik *Domain-Knowledge Heuristic Imputation*: profesi formal (PNS/Guru/Medis) secara regulasi mempersyaratkan ijazah D4/S1, sementara usia muda di bawah 19 tahun secara riil berada di jenjang pendidikan menengah. Teknik ini menjaga koherensi data tanpa merusak sebaran fitur.

### Q5: Bagaimana pembobotan pekerjaan suami dan istri ditentukan?
> **Jawaban:** Berdasarkan hukum perkawinan dan sosiologi keluarga di Indonesia, suami berkedudukan sebagai kepala keluarga dan penanggung nafkah utama. Oleh karena itu, pekerjaan suami memiliki bobot penentu primer: jika suami sudah mapan (PNS, BUMN, Wiraswasta), keluarga dinyatakan layak baik istri berstatus IRT maupun bekerja. Sebaliknya, peran istri dihitung sebagai penopang ekonomi sekunder: jika suami bekerja sebagai buruh harian lepas, kehadiran istri yang berpenghasilan (*dual-income*) dapat menaikkan status menjadi layak karena mampu menutupi kerentanan nafkah suami.
