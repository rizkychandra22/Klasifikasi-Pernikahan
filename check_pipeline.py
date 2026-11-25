"""
Skrip Pemeriksaan Isi File Artifacts Model (.pkl)
- features.pkl     : Menyimpan struktur daftar fitur masukan
- preprocessor.pkl : Menyimpan objek OneHotEncoder beserta kamus kategori biner
- naive_bayes.pkl  : Menyimpan model Multinomial Naive Bayes beserta bobot probabilitasnya
"""

import os
import joblib
import numpy as np
import pandas as pd

models_dir = 'models' if os.path.exists('models') else os.path.join('Source-Code', 'models')

print("=" * 70)
print("1. MEMERIKSA ISI FILE: features.pkl")
print("=" * 70)
features = joblib.load(os.path.join(models_dir, 'features.pkl'))
print(f"Tipe Objek            : {type(features)}")
print(f"Jumlah Fitur          : {len(features)}")
print("Daftar Fitur Masukan  :")
for i, feat in enumerate(features, 1):
    print(f"  {i}. {feat}")

print("\n" + "=" * 70)
print("2. MEMERIKSA ISI FILE: preprocessor.pkl (OneHotEncoder)")
print("=" * 70)
encoder = joblib.load(os.path.join(models_dir, 'preprocessor.pkl'))
print(f"Tipe Objek            : {type(encoder)}")
print(f"Jumlah Fitur Input    : {encoder.n_features_in_} kolom")
print(f"Fitur Input Asal      : {list(encoder.feature_names_in_)}")

print("\nKamus Kategori yang Dipelajari Encoder (encoder.categories_):")
for col, cats in zip(encoder.feature_names_in_, encoder.categories_):
    print(f" - {col} ({len(cats)} kategori):")
    for cat in cats:
        print(f"     • {cat}")

binary_cols = encoder.get_feature_names_out(features)
print(f"\nTotal Dimensi Kolom Biner Output ({len(binary_cols)} fitur biner):")
for i, col in enumerate(binary_cols, 1):
    print(f"   [{i:2d}] {col}")

print("\n" + "=" * 70)
print("3. MEMERIKSA ISI FILE: naive_bayes.pkl (MultinomialNB)")
print("=" * 70)
model = joblib.load(os.path.join(models_dir, 'naive_bayes.pkl'))
print(f"Tipe Objek            : {type(model)}")
print(f"Parameter Smoothing   : alpha = {model.alpha}")
print(f"Kelas Target          : {model.classes_}")
print(f"Jumlah Data Latih     : {int(sum(model.class_count_))} sampel")
print("Sebaran Sampel Latih per Kelas (class_count_):")
for kelas, count in zip(model.classes_, model.class_count_):
    print(f" - Kelas '{kelas}': {int(count)} data ({count/sum(model.class_count_)*100:.2f}%)")

priors = np.exp(model.class_log_prior_)
print("\nProbabilitas Awal / Prior P(Kelas) (model.class_log_prior_):")
for kelas, p in zip(model.classes_, priors):
    print(f" - P('{kelas}') = {p:.4f} ({p*100:.2f}%)")

print("\nTabel Bobot Likelihood Fitur P(Fitur | Kelas) (Sampel 10 Fitur):")
df_prob = pd.DataFrame(
    np.exp(model.feature_log_prob_),
    index=model.classes_,
    columns=binary_cols
).T

print(df_prob.head(10).to_string())
print("\n" + "=" * 70)
print("SELESAI: Ketiga file .pkl berhasil dimuat dan diperiksa dengan sempurna.")
print("=" * 70)
