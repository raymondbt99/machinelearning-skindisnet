"""
=============================================================================
FASE 6: Advanced Augmentation (Albumentations) + Test-Time Augmentation (TTA)
=============================================================================
Menggunakan DenseNet121 multimodal baseline dari Fase 3/4 sebagai starting 
point, kemudian:
  1. Augmentasi lebih kaya via Albumentations (CLAHE, ElasticTransform, dsb.)
  2. Test-Time Augmentation dengan 8 variasi (vs. 4 di Fase sebelumnya)
  3. Cosine Annealing learning rate schedule
Target: Naik ~3-5% dari 44.06%
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import tensorflow as tf
from tensorflow import keras
from keras import layers

try:
    import albumentations as A
    from albumentations.core.composition import Compose
    HAS_ALBUMENTATIONS = True
    print("[OK] albumentations tersedia, menggunakan advanced augmentation")
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("[WARN] albumentations tidak terinstall. Fallback ke augmentasi Keras. "
          "Jalankan: pip install albumentations")

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

WORKSPACE = r"c:\SkinDisNet A Multi-Class Clinical Images and Metad"
DATASET_DIR = os.path.join(WORKSPACE, "SkinDisNet_2")
ARTIFACTS_DIR = os.path.join(WORKSPACE, "artifacts")
MANIFEST_PATH = os.path.join(ARTIFACTS_DIR, "dataset_split_manifest.csv")
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "multimodal_metadata_encoders.json")
CLASS_MAPPING_PATH = os.path.join(ARTIFACTS_DIR, "class_mapping.json")
DENSENET_BASELINE_PATH = os.path.join(ARTIFACTS_DIR, "densenet_multimodal_skin_classifier.keras")
OUTPUT_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "fase6_albumentations_model.keras")
METRICS_OUT_PATH = os.path.join(ARTIFACTS_DIR, "fase6_evaluation_metrics.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32          # Lebih kecil agar fine-tuning lebih stabil
NUM_CLASSES = 6
LABEL_SMOOTHING = 0.1

# --─ Albumentations augmentation pipeline ----------------------------------─

def build_albumentations_pipeline():
    """Pipeline augmentasi kaya untuk training."""
    return A.Compose([
        A.RandomResizedCrop(height=224, width=224, scale=(0.75, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        ], p=0.6),
        A.OneOf([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),    # Contrast Limited AHE - bagus untuk kulit
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0)),
            A.GaussianBlur(blur_limit=(3, 7)),
        ], p=0.4),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(num_steps=5, distort_limit=0.3),
        ], p=0.3),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1,
                        min_height=8, min_width=8, fill_value=0, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def build_albumentations_tta_transforms():
    """8 variasi TTA yang deterministik."""
    base = [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return [
        A.Compose([A.Resize(224, 224)] + base),                          # 0: Original
        A.Compose([A.HorizontalFlip(p=1.0), A.Resize(224, 224)] + base), # 1: H-flip
        A.Compose([A.VerticalFlip(p=1.0), A.Resize(224, 224)] + base),   # 2: V-flip
        A.Compose([A.RandomRotate90(p=1.0), A.Resize(224, 224)] + base), # 3: Rotate 90
        A.Compose([A.Transpose(p=1.0), A.Resize(224, 224)] + base),      # 4: Transpose
        A.Compose([A.CLAHE(clip_limit=3.0, p=1.0), A.Resize(224, 224)] + base),   # 5: CLAHE
        A.Compose([A.Sharpen(alpha=(0.3, 0.3), p=1.0), A.Resize(224, 224)] + base), # 6: Sharpened
        A.Compose([A.RandomBrightnessContrast(brightness_limit=0.1,
                    contrast_limit=0.1, p=1.0), A.Resize(224, 224)] + base),         # 7: Bright
    ]

# --─ Metadata encoding ------------------------------------------------------─

def load_encoder():
    with open(ENCODER_PATH, 'r') as f:
        return json.load(f)

def encode_metadata_vector(age_val, sex_val, loc_val, encoder_info):
    num_locs = encoder_info["num_locations"]
    vec = np.zeros(num_locs + 1 + 2, dtype=np.float32)
    loc_clean = str(loc_val).strip().title()
    loc_idx = encoder_info["location_to_idx"].get(loc_clean, encoder_info["location_to_idx"]["Unknown"])
    vec[loc_idx] = 1.0
    try:
        age_float = float(age_val)
    except Exception:
        age_float = 30.0
    vec[num_locs] = np.clip(age_float / encoder_info["max_age"], 0.0, 1.2)
    sex_str = str(sex_val).strip().title()
    if sex_str == 'Female':
        vec[num_locs + 1] = 1.0
    elif sex_str == 'Male':
        vec[num_locs + 2] = 1.0
    else:
        vec[num_locs + 1] = 0.5
        vec[num_locs + 2] = 0.5
    return vec

# --─ Data loading ------------------------------------------------------------─

def load_dataset_arrays(encoder_info):
    """Load semua data sebagai numpy arrays (untuk albumentations)."""
    manifest_df = pd.read_csv(MANIFEST_PATH)
    with open(CLASS_MAPPING_PATH, 'r') as f:
        classes = json.load(f)['classes']

    splits = {
        'train': {'paths': [], 'meta': [], 'labels': []},
        'val':   {'paths': [], 'meta': [], 'labels': []},
        'test':  {'paths': [], 'meta': [], 'labels': []},
    }

    for _, row in manifest_df.iterrows():
        split = row['Split']
        folder = row['Folder_name']
        img_id = row['Image_id']
        class_idx = int(row['Class_index'])
        meta_vec = encode_metadata_vector(row['Age'], row['Sex'], row['Leision_location'], encoder_info)

        orig_full = os.path.join(DATASET_DIR, "Preprocessed", folder, f"{img_id}.jpg")
        if not os.path.exists(orig_full):
            continue

        if split == 'train':
            splits['train']['paths'].append(orig_full)
            splits['train']['meta'].append(meta_vec)
            splits['train']['labels'].append(class_idx)
            # Include existing augmented images
            for aug_i in range(1, 8):
                aug_full = os.path.join(DATASET_DIR, "Augmented", folder, f"{img_id}_{aug_i:02d}.jpg")
                if os.path.exists(aug_full):
                    splits['train']['paths'].append(aug_full)
                    splits['train']['meta'].append(meta_vec)
                    splits['train']['labels'].append(class_idx)
        else:
            splits[split]['paths'].append(orig_full)
            splits[split]['meta'].append(meta_vec)
            splits[split]['labels'].append(class_idx)

    print(f"Train: {len(splits['train']['paths'])} | Val: {len(splits['val']['paths'])} | Test: {len(splits['test']['paths'])}")

    counts = pd.Series(splits['train']['labels']).value_counts().sort_index().values
    total = len(splits['train']['labels'])
    smoothed_w = np.sqrt(total / (NUM_CLASSES * counts))
    smoothed_w /= np.mean(smoothed_w)
    class_weights = {i: float(round(w, 4)) for i, w in enumerate(smoothed_w)}
    print(f"Class weights: {class_weights}")

    return splits, class_weights, classes

# --─ tf.data pipeline with Albumentations ------------------------------------

def build_tf_dataset_albumentations(paths, meta_arr, labels_oh, augment_fn=None, shuffle=False):
    """
    tf.data pipeline. Augmentation dilakukan via tf.numpy_function agar
    albumentations bisa dipanggil di dalam graph mode.
    """
    IMG_H, IMG_W = IMG_SIZE

    def load_and_aug(path_bytes, meta, label):
        path_str = path_bytes.decode('utf-8')
        with Image.open(path_str) as img:
            rgb = img.convert('RGB').resize(IMG_SIZE, Image.Resampling.BILINEAR)
        arr = np.array(rgb, dtype=np.uint8)

        if augment_fn is not None and HAS_ALBUMENTATIONS:
            result = augment_fn(image=arr)
            arr = result['image'].astype(np.float32)
        else:
            # Fallback: simple normalization
            arr = arr.astype(np.float32) / 255.0

        return arr, meta, label

    def wrapper(path, meta, label):
        img, meta_out, label_out = tf.numpy_function(
            func=load_and_aug,
            inp=[path, meta, label],
            Tout=[tf.float32, tf.float32, tf.float32]
        )
        img.set_shape([IMG_H, IMG_W, 3])
        meta_out.set_shape([meta_arr.shape[1]])
        label_out.set_shape([NUM_CLASSES])
        return {"input_image": img, "input_metadata": meta_out}, label_out

    paths_t = tf.constant(paths)
    meta_t = tf.constant(meta_arr, dtype=tf.float32)
    labels_t = tf.constant(labels_oh, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths_t, meta_t, labels_t))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    ds = ds.map(wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

def build_plain_tf_dataset(paths, meta_arr, labels_oh):
    """Plain dataset tanpa albumentations (untuk val/test)."""
    def load_plain(path_bytes, meta, label):
        path_str = path_bytes.decode('utf-8')
        with Image.open(path_str) as img:
            rgb = img.convert('RGB').resize(IMG_SIZE, Image.Resampling.BILINEAR)
        arr = np.array(rgb, dtype=np.float32)
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
        arr = (arr - mean) / std
        return arr, meta, label

    def wrapper(path, meta, label):
        img, meta_out, label_out = tf.numpy_function(
            func=load_plain,
            inp=[path, meta, label],
            Tout=[tf.float32, tf.float32, tf.float32]
        )
        img.set_shape([224, 224, 3])
        meta_out.set_shape([meta_arr.shape[1]])
        label_out.set_shape([NUM_CLASSES])
        return {"input_image": img, "input_metadata": meta_out}, label_out

    paths_t = tf.constant(paths)
    meta_t = tf.constant(meta_arr, dtype=tf.float32)
    labels_t = tf.constant(labels_oh, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths_t, meta_t, labels_t))
    ds = ds.map(wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# --─ TTA Inference --------------------------------------------------------─

def predict_with_tta(model, test_paths, test_meta, tta_transforms):
    """Prediksi dengan TTA: rata-rata 8 augmentasi."""
    all_probs = []
    n = len(test_paths)
    print(f"Running TTA inference on {n} test samples with {len(tta_transforms)} augmentations...")

    for i, (p, meta_vec) in enumerate(zip(test_paths, test_meta)):
        if i % 50 == 0:
            print(f"  TTA {i}/{n}...")
        with Image.open(p) as img:
            rgb = img.convert('RGB')
        arr = np.array(rgb, dtype=np.uint8)

        aug_imgs = []
        for tfm in tta_transforms:
            if HAS_ALBUMENTATIONS:
                result = tfm(image=arr)
                aug_imgs.append(result['image'].astype(np.float32))
            else:
                aug_imgs.append(arr.astype(np.float32) / 255.0)

        img_batch = np.stack(aug_imgs, axis=0)   # [n_tta, H, W, 3]
        meta_batch = np.tile(meta_vec, (len(tta_transforms), 1))

        preds = model.predict({"input_image": img_batch, "input_metadata": meta_batch}, verbose=0)
        all_probs.append(np.mean(preds, axis=0))

    return np.array(all_probs)

# --─ Main training --------------------------------------------------------─

def train_fase6():
    print("=" * 72)
    print("FASE 6: Advanced Albumentations Augmentation + TTA")
    print("=" * 72)

    encoder_info = load_encoder()
    splits, class_weights, classes = load_dataset_arrays(encoder_info)

    meta_dim = len(splits['train']['meta'][0])
    train_meta = np.array(splits['train']['meta'], dtype=np.float32)
    val_meta = np.array(splits['val']['meta'], dtype=np.float32)
    test_meta = np.array(splits['test']['meta'], dtype=np.float32)

    train_labels_oh = tf.keras.utils.to_categorical(splits['train']['labels'], NUM_CLASSES)
    val_labels_oh = tf.keras.utils.to_categorical(splits['val']['labels'], NUM_CLASSES)
    test_labels_oh = tf.keras.utils.to_categorical(splits['test']['labels'], NUM_CLASSES)

    # -- Build augmentation pipelines --------------------------------------
    if HAS_ALBUMENTATIONS:
        train_aug_fn = build_albumentations_pipeline()
        tta_transforms = build_albumentations_tta_transforms()
        print("[Albumentations] Training augmentation pipeline ready (8 TTA transforms)")
    else:
        train_aug_fn = None
        tta_transforms = None
        print("[Fallback] Simple augmentation will be used")

    # -- Build tf.data datasets --------------------------------------------
    train_ds = build_tf_dataset_albumentations(
        splits['train']['paths'], train_meta, train_labels_oh,
        augment_fn=train_aug_fn, shuffle=True
    )
    val_ds = build_plain_tf_dataset(splits['val']['paths'], val_meta, val_labels_oh)
    test_ds = build_plain_tf_dataset(splits['test']['paths'], test_meta, test_labels_oh)

    # -- Load baseline model (DenseNet121 Multimodal dari Fase 3/4) --------
    print(f"\nLoading baseline model from: {DENSENET_BASELINE_PATH}")
    model = keras.models.load_model(DENSENET_BASELINE_PATH)
    model.summary(print_fn=lambda x: None)  # suppress full summary
    print(f"Model loaded. Total params: {model.count_params():,}")

    # -- Freeze backbone, fine-tune only top layers with higher LR --------─
    # Re-compile with Cosine Annealing
    total_steps = (len(splits['train']['paths']) // BATCH_SIZE) * 15
    lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=3e-4,
        first_decay_steps=total_steps // 3,
        t_mul=2.0,
        m_mul=0.9
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy']
    )

    print("\n-- Stage 1: Fine-tune fusion head with Albumentations (backbone frozen, 10 epochs) --")

    callbacks_s1 = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5,
                                      restore_best_weights=True, mode='max'),
        keras.callbacks.ModelCheckpoint(OUTPUT_MODEL_PATH, monitor='val_accuracy',
                                        save_best_only=True, mode='max', verbose=1),
    ]

    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        class_weight=class_weights,
        callbacks=callbacks_s1,
        verbose=1
    )

    # -- Stage 2: Unfreeze top 30 layers of backbone ----------------------─
    print("\n-- Stage 2: Unfreeze top 30 DenseNet layers (lr=1e-5, 10 epochs) --")
    for layer in model.layers:
        if hasattr(layer, 'layers'):  # It's a nested model (DenseNet)
            for sub_layer in layer.layers[-30:]:
                sub_layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy']
    )

    callbacks_s2 = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=6,
                                      restore_best_weights=True, mode='max'),
        keras.callbacks.ModelCheckpoint(OUTPUT_MODEL_PATH, monitor='val_accuracy',
                                        save_best_only=True, mode='max', verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                                          min_lr=1e-7, verbose=1),
    ]

    hist2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        class_weight=class_weights,
        callbacks=callbacks_s2,
        verbose=1
    )

    # -- Reload best model ------------------------------------------------─
    print(f"\nLoading best model from {OUTPUT_MODEL_PATH}...")
    model = keras.models.load_model(OUTPUT_MODEL_PATH)

    # -- Evaluation: Standard ----------------------------------------------
    print("\n-- Standard Evaluation (no TTA) --")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Standard Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    y_true = splits['test']['labels']
    y_pred_std = []
    for X_batch, _ in test_ds:
        preds = model.predict(X_batch, verbose=0)
        y_pred_std.extend(np.argmax(preds, axis=1))
    y_pred_std = y_pred_std[:len(y_true)]

    # -- Evaluation: TTA --------------------------------------------------
    tta_acc = test_acc
    y_pred_tta = y_pred_std
    if HAS_ALBUMENTATIONS and tta_transforms is not None:
        print("\n-- TTA Evaluation (8 augmentations) --")
        tta_probs = predict_with_tta(model, splits['test']['paths'], test_meta, tta_transforms)
        y_pred_tta = np.argmax(tta_probs, axis=1)
        tta_acc = accuracy_score(y_true, y_pred_tta)
        print(f"TTA Test Accuracy: {tta_acc:.4f} ({tta_acc*100:.2f}%)")

    # -- Classification report --------------------------------------------─
    use_pred = y_pred_tta if HAS_ALBUMENTATIONS else y_pred_std
    use_acc = tta_acc if HAS_ALBUMENTATIONS else test_acc

    print("\n-- Per-Class Report (TTA) --")
    print(classification_report(y_true, use_pred, target_names=classes, digits=4))

    # -- Confusion Matrix --------------------------------------------------
    cm = confusion_matrix(y_true, use_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Fase 6 Confusion Matrix (TTA, acc={use_acc*100:.2f}%)")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(ARTIFACTS_DIR, "fase6_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {cm_path}")

    # -- Save metrics ------------------------------------------------------
    report_dict = classification_report(y_true, use_pred, target_names=classes,
                                        digits=4, output_dict=True)
    macro_f1 = f1_score(y_true, use_pred, average='macro')
    metrics = {
        "model_name": "Fase6_Albumentations_TTA_DenseNet121_Multimodal",
        "standard_test_accuracy": float(test_acc),
        "tta_test_accuracy": float(tta_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(f1_score(y_true, use_pred, average='weighted')),
        "test_sample_count": len(y_true),
        "classes": classes,
        "per_class_metrics": {
            c: {k: v for k, v in report_dict[c].items()}
            for c in classes if c in report_dict
        },
        "confusion_matrix": cm.tolist(),
    }
    with open(METRICS_OUT_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved: {METRICS_OUT_PATH}")

    print("\n" + "=" * 72)
    print(f"FASE 6 SELESAI!")
    print(f"  Standard Accuracy : {test_acc*100:.2f}%")
    print(f"  TTA Accuracy      : {tta_acc*100:.2f}%")
    print(f"  Macro F1          : {macro_f1*100:.2f}%")
    print(f"  Model tersimpan   : {OUTPUT_MODEL_PATH}")
    print("=" * 72)

    return metrics

if __name__ == "__main__":
    train_fase6()
