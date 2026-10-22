# Panduan Langkah Demi Langkah Eksekusi Sistem SkinDisNet

Panduan ini berisi urutan perintah yang dapat Anda jalankan satu per satu melalui terminal (PowerShell atau Command Prompt) dari folder workspace:
```text
c:\SkinDisNet A Multi-Class Clinical Images and Metad
```

---

### Ringkasan Alur Kerja & Roadmap Peningkatan

```
[1. inspect_dataset.py]         --> Audit file gambar, metadata, duplikasi & pasien
         │
[2. prepare_data.py]            --> Split 70% Train, 15% Val, 15% Test berbasis pasien (Zero Leakage)
         │
[3. train_models.py]            --> Baseline CNN (16.09%) & Initial EfficientNetB0 (36.02%)
         │
[4. train_fase1_augmented.py]   --> Fase 1: Ekspansi 9.616 data train Augmented + TTA (40.23%)
         │
[5. train_fase2_multimodal.py]  --> Fase 2: Multimodal Dual-Branch Network (+Lokasi, Usia, Sex: 43.30%)
         │
[6. train_fase3_advanced.py]    --> Fase 3, 4, 5: DenseNet121 + Label Smoothing + Ensemble Stacking (44.06%)
         │
[7. train_fase6_albumentations.py] --> Fase 6: Advanced Augmentation + TTA (target +3-5%)
         │
[8. train_fase7_focal_loss.py]  --> Fase 7: Focal Loss + Oversampling + Mixup (target +4-6%)
         │
[9. train_fase8_efficientnetv2.py] --> Fase 8: EfficientNetV2-S 300×300 + Focal Loss (target +5-8%)
         │
[10. train_fase9_hierarchical.py] --> Fase 9: 2-Level Hierarchical Classifier (target +4-7%)
         │
[11. verify_model.py]           --> Uji inferensi mandiri (Multimodal / Image-Only)
         │
[12. webapp/app.py]             --> Jalankan Web App Interaktif di browser (localhost:5000)
```

**Progression Akurasi:**
| Fase | Script | Akurasi |
|---|---|---|
| Baseline CNN | train_models.py | 16.09% |
| EfficientNetB0 awal | train_models.py | 36.02% |
| Fase 1 (Augmented) | train_fase1_augmented.py | 40.23% |
| Fase 2 (Multimodal) | train_fase2_multimodal.py | 43.30% |
| Fase 3-5 (DenseNet121) | train_fase3_advanced.py | **44.06%** ← baseline saat ini |
| Fase 6 (Albumentations+TTA) | train_fase6_albumentations.py | Target ~47-50% |
| Fase 7 (Focal Loss) | train_fase7_focal_loss.py | Target ~51-56% |
| Fase 8 (EfficientNetV2-S) | train_fase8_efficientnetv2.py | Target ~57-64% |
| Fase 9 (Hierarchical) | train_fase9_hierarchical.py | Target ~61-70% |

---

## Langkah 1: Inspeksi dan Audit Dataset

Skrip ini memeriksa seluruh direktori gambar (`Preprocessed/` dan `Augmented/`), membaca file `SkinDisNet_Metadata.csv`, memvalidasi keterbacaan gambar, mendeteksi duplikat, serta menghitung jumlah sampel per penyakit dan per pasien.

### Perintah:
```powershell
python inspect_dataset.py
```

### Apa yang Anda Lihat:
* **Total Metadata:** 1.710 baris data dengan 7 kolom tanpa ada nilai yang hilang (*zero missing values*).
* **Preprocessed:** 1.710 gambar terverifikasi cocok 100% dengan nama file di CSV.
* **Augmented:** 11.970 gambar (tepat 7 variasi turunan untuk tiap gambar asli).
* **Pasien:** 416 pasien unik, di mana tiap pasien hanya memiliki satu jenis diagnosis.
* **File Hasil Audit:** Otomatis tersimpan dalam format JSON di [`dataset_inspection_report.json`](file:///c:/SkinDisNet%20A%20Multi-Class%20Clinical%20Images%20and%20Metad/dataset_inspection_report.json) dan laporan terperinci di [`dataset_inspection_report.md`](file:///c:/SkinDisNet%20A%20Multi-Class%20Clinical%20Images%20and%20Metad/dataset_inspection_report.md).

---

## Langkah 2: Pemisahan Data Berbasis Pasien (Data Split)

Skrip ini membagi data secara terstratifikasi ke dalam **70% Train**, **15% Validation**, dan **15% Test** dengan unit pemisahan adalah **`Patient_id`** (bukan per gambar). Ini menjamin tidak ada gambar dari pasien yang sama muncul di lebih dari satu subset (*zero intra-patient data leakage*).

### Perintah:
```powershell
python prepare_data.py
```

### Apa yang Anda Lihat:
* **Distribusi Pasien:**
  * Train: 292 pasien (~70.2%)
  * Validation: 62 pasien (~14.9%)
  * Test: 62 pasien (~14.9%)
* **Distribusi Gambar Asli:**
  * Train: 1.202 gambar
  * Validation: 247 gambar
  * Test: 261 gambar
* **File Output yang Dihasilkan:**
  * [`artifacts/class_mapping.json`](file:///c:/SkinDisNet%20A%20Multi-Class%20Clinical%20Images%20and%20Metad/artifacts/class_mapping.json) (Pemetaan index 0–5 ke nama penyakit).
  * [`artifacts/dataset_split_manifest.csv`](file:///c:/SkinDisNet%20A%20Multi-Class%20Clinical%20Images%20and%20Metad/artifacts/dataset_split_manifest.csv) (Daftar manifest lengkap pembagian tiap gambar dan pasien).

---

## Langkah 3: Pelatihan Model Bertahap (Fase Eksperimen)

### A. Pelatihan Awal (1.202 Gambar Asli):
Melatih Baseline Custom CNN dan Initial EfficientNetB0 Image-Only:
```powershell
python train_models.py
```
* **Baseline Custom CNN:** Akurasi Test: **16.09%**, Macro F1: **0.0814**.
* **Initial EfficientNetB0:** Akurasi Test: **36.02%**, Macro F1: **0.3216**.

### B. Pelatihan Fase 1 (Ekspansi 9.616 Gambar Augmented + TTA):
Memasukkan 7 variasi data augmented khusus untuk 292 pasien di split Train (Val/Test tetap murni):
```powershell
python train_fase1_augmented.py
```
* **Hasil Fase 1:** Akurasi Test meningkat ke **40.23%**, Val Acc: **49.39%**.

### C. Pelatihan Fase 2 (Multimodal Dual-Branch Network: Gambar + Metadata Klinis):
Menggabungkan representasi fitur visual lesi dengan embedding lokasi anatomis lesi (`Leision_location`), usia (`Age`), dan jenis kelamin (`Sex`):
```powershell
python train_fase2_multimodal.py
```
* **Hasil Fase 2:** Akurasi Test meningkat ke **43.30%**, Weighted F1: **0.3985**.
* Model tersimpan di: [`artifacts/multimodal_skin_classifier.keras`](file:///c:/SkinDisNet%20A%20Multi-Class%20Clinical%20Images%20and%20Metad/artifacts/multimodal_skin_classifier.keras).

### D. Pelatihan Fase 3, 4, & 5 (DenseNet121 + Label Smoothing + Ensemble Stacking):
Mengeksplorasi arsitektur DenseNet121 dengan koneksi skip rapat multi-skala, regulasi Label Smoothing Cross-Entropy ($\alpha = 0.1$), dan penggabungan prediksi ensemble soft-voting:
```powershell
python train_fase3_advanced.py
```
* Model tersimpan di: [`artifacts/densenet_multimodal_skin_classifier.keras`](file:///c:/SkinDisNet%20A%20Multi-Class%20Clinical%20Images%20and%20Metad/artifacts/densenet_multimodal_skin_classifier.keras).
* Metrik evaluasi tersimpan di: [`artifacts/ensemble_evaluation_metrics.json`](file:///c:/SkinDisNet%20A%20Multi-Class%20Clinical%20Images%20and%20Metad/artifacts/ensemble_evaluation_metrics.json).

---

## Langkah 4: Verifikasi Model Mandiri (CLI Inference Test)

Skrip `verify_model.py` mendukung pengujian inferensi multimodal maupun image-only lengkap dengan Test-Time Augmentation (TTA).

### Perintah Otomatis (Menggunakan Sampel Test Set):
```powershell
python verify_model.py
```

### Perintah Kustom dengan Lokasi & Informasi Klinis:
```powershell
python verify_model.py --image "SkinDisNet_2/Preprocessed/Scabies (SC)/SC (321).jpg" --location "Finger" --age 24 --sex "Male"
```

### Perintah Menguji Model Citra Saja (Image-Only):
```powershell
python verify_model.py --image-only
```

---

## Langkah 5: Menjalankan Web Application Interaktif

Skrip ini menyalakan server web lokal berbasis Flask yang otomatis mendeteksi model multimodal dan menyajikan dashboard klinis interaktif.

### Perintah:
```powershell
python webapp/app.py
```

### Membuka Aplikasi di Browser:
Setelah server menyala, buka browser dan kunjungi:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

### Fitur Interaktif:
1. **Drag-and-Drop / Upload:** Unggah foto lesi kulit apa saja dari komputer Anda.
2. **Metadata Klinis Pasien (Opsional):** Pilih lokasi anatomis lesi (wajah, tangan, sela jari, kulit kepala, dll.), masukkan usia pasien, dan jenis kelamin untuk hasil klasifikasi multimodal yang jauh lebih tajam.
3. **Pilihan Sampel Cepat (1-Click):** Klik salah satu kartu sampel terverifikasi dari test set di bawah area upload untuk langsung mengisi foto beserta metadata klinisnya secara otomatis.
4. **Analisis Lesi:** Klik tombol **"Analyze & Classify Lesion"** untuk melihat hasil diagnosis utama, circular gauge probabilitas, breakdown 6 kelas penyakit, dan profil patologi klinis.
5. **Modal Performa Model:** Klik tombol **"Model Performance"** di bagian atas untuk melihat metrik akurasi test set, Confusion Matrix, dan grafik konvergensi pelatihan seluruh fase.

---

## Langkah 6: Pelatihan Fase 6 — Albumentations Advanced Augmentation + TTA

Fine-tune DenseNet121 Multimodal dari Fase 3/4 menggunakan augmentasi lebih kaya (CLAHE, ElasticTransform, CoarseDropout, dsb.) dan Test-Time Augmentation (8 variasi).

### Perintah:
```powershell
python train_fase6_albumentations.py
```

### Output:
- **Model tersimpan:** `artifacts/fase6_albumentations_model.keras`
- **Metrik tersimpan:** `artifacts/fase6_evaluation_metrics.json`
- **Confusion matrix:** `artifacts/fase6_confusion_matrix.png`

---

## Langkah 7: Pelatihan Fase 7 — Focal Loss + Class-Weighted Oversampling + Mixup

Mengganti Cross-Entropy dengan **Focal Loss** (γ=2) untuk fokus pada sampel sulit dan kelas minoritas. Ditambah oversampling kelas kecil dan Mixup augmentation.

### Perintah:
```powershell
python train_fase7_focal_loss.py
```

### Output:
- **Model tersimpan:** `artifacts/fase7_focal_loss_model.keras`
- **Metrik tersimpan:** `artifacts/fase7_evaluation_metrics.json`
- **Confusion matrix:** `artifacts/fase7_confusion_matrix.png`

---

## Langkah 8: Pelatihan Fase 8 — EfficientNetV2-S Backbone (300×300 Resolution)

Upgrade backbone dari DenseNet121 (74.98% ImageNet) ke **EfficientNetV2-S** (83.9% ImageNet) dengan input resolusi lebih tinggi (300×300) dan progressive fine-tuning bertahap.

### Perintah:
```powershell
python train_fase8_efficientnetv2.py
```

### Catatan:
- Training lebih lama karena model lebih besar (21.5M params) dan resolusi lebih tinggi
- 4-stage progressive unfreeze: head → top 30 → top 80 → full fine-tune

### Output:
- **Model tersimpan:** `artifacts/fase8_efficientnetv2s_model.keras`
- **Metrik tersimpan:** `artifacts/fase8_evaluation_metrics.json`
- **Confusion matrix:** `artifacts/fase8_confusion_matrix.png`

---

## Langkah 9: Pelatihan Fase 9 — Hierarchical 2-Level Classifier

Mengatasi overlap visual kelas eczematous dengan arsitektur hierarki:
- **Level 1:** 4-class kasar (Eczematous Group / Scabies / Seborrheic Dermatitis / Tinea Corporis)
- **Level 2:** 3-class halus (Atopic Dermatitis / Contact Dermatitis / Eczema) — hanya untuk eczematous

### Perintah:
```powershell
python train_fase9_hierarchical.py
```

### Output:
- **L1 Model:** `artifacts/fase9_hierarchical_level1.keras`
- **L2 Model:** `artifacts/fase9_hierarchical_level2.keras`
- **Metrik tersimpan:** `artifacts/fase9_evaluation_metrics.json`
- **Confusion matrix:** `artifacts/fase9_confusion_matrix.png`

---

## Urutan Eksekusi Optimal (Direkomendasikan)

Untuk mencapai akurasi tertinggi, jalankan dalam urutan berikut:

```powershell
# 1. Fase 7 (Focal Loss) — gain terbesar untuk kelas minoritas
python train_fase7_focal_loss.py

# 2. Fase 8 (EfficientNetV2-S) — upgrade backbone paling powerful
python train_fase8_efficientnetv2.py

# 3. Fase 6 (Albumentations TTA) — fine-tune dengan augmentasi lebih kaya
python train_fase6_albumentations.py

# 4. Fase 9 (Hierarchical) — spesialisasi untuk kelas eczematous
python train_fase9_hierarchical.py
```

> **Catatan:** Sistem ini dirancang hanya untuk penelitian akademik dan **bukan** pengganti diagnosis dokter profesional.
