"""
=============================================================================
FASE 8: EfficientNetV2-S Backbone + Higher Resolution (300×300)
=============================================================================
Upgrade backbone dari DenseNet121 (2016) ke EfficientNetV2-S (2021).
EfficientNetV2-S: ImageNet Top-1 83.9% vs DenseNet121 74.98%

Strategi:
  1. EfficientNetV2-S pretrained ImageNet (include_top=False)
  2. Input resolution 300×300 (EfficientNetV2-S optimal size)
  3. Focal Loss (gamma=2) dari Fase 7
  4. Progressive fine-tuning (5 stages: frozen → gradual unfreeze)
  5. Cosine Annealing dengan Warm Restarts
  6. Dropout adaptif + Weight Decay (L2 regularization)
Target: +5-8% dari baseline → potensi 50-52%+
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import tensorflow as tf
from tensorflow import keras
from keras import layers

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

WORKSPACE = r"c:\SkinDisNet A Multi-Class Clinical Images and Metad"
DATASET_DIR = os.path.join(WORKSPACE, "SkinDisNet_2")
ARTIFACTS_DIR = os.path.join(WORKSPACE, "artifacts")
MANIFEST_PATH = os.path.join(ARTIFACTS_DIR, "dataset_split_manifest.csv")
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "multimodal_metadata_encoders.json")
CLASS_MAPPING_PATH = os.path.join(ARTIFACTS_DIR, "class_mapping.json")
OUTPUT_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "fase8_efficientnetv2s_model.keras")
METRICS_OUT_PATH = os.path.join(ARTIFACTS_DIR, "fase8_evaluation_metrics.json")

IMG_SIZE = (300, 300)   # EfficientNetV2-S optimal input size
BATCH_SIZE = 24         # Lebih kecil karena gambar 300×300 lebih besar
NUM_CLASSES = 6
FOCAL_GAMMA = 2.0
L2_REGULARIZATION = 1e-4
LABEL_SMOOTHING = 0.08

# --─ Focal Loss ------------------------------------------------------------

class FocalLoss(keras.losses.Loss):
    """Focal Loss multi-class dengan per-class alpha dan label smoothing."""
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.08, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true_smooth = y_true * (1.0 - self.label_smoothing) + self.label_smoothing / NUM_CLASSES
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true_smooth * tf.math.log(y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        if self.alpha is not None:
            alpha_t = tf.reduce_sum(y_true * tf.constant(self.alpha, dtype=tf.float32),
                                     axis=-1, keepdims=True)
            ce = alpha_t * focal_weight * ce
        else:
            ce = focal_weight * ce
        return tf.reduce_mean(tf.reduce_sum(ce, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma, 'alpha': self.alpha,
                        'label_smoothing': self.label_smoothing})
        return config

# --─ Metadata encoding ----------------------------------------------------─

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

# --─ Dataset loading ------------------------------------------------------─

def load_dataset_splits(encoder_info):
    manifest_df = pd.read_csv(MANIFEST_PATH)
    with open(CLASS_MAPPING_PATH, 'r') as f:
        classes = json.load(f)['classes']

    splits = {s: {'paths': [], 'meta': [], 'labels': []}
              for s in ['train', 'val', 'test']}

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
            for aug_i in range(1, 8):
                aug_p = os.path.join(DATASET_DIR, "Augmented", folder, f"{img_id}_{aug_i:02d}.jpg")
                if os.path.exists(aug_p):
                    splits['train']['paths'].append(aug_p)
                    splits['train']['meta'].append(meta_vec)
                    splits['train']['labels'].append(class_idx)
        else:
            splits[split]['paths'].append(orig_full)
            splits[split]['meta'].append(meta_vec)
            splits[split]['labels'].append(class_idx)

    for s in ['train', 'val', 'test']:
        print(f"  {s}: {len(splits[s]['paths'])} samples")

    counts = np.array(pd.Series(splits['train']['labels']).value_counts().sort_index().values,
                       dtype=np.float32)
    total = len(splits['train']['labels'])
    smoothed_w = np.sqrt(total / (NUM_CLASSES * counts))
    smoothed_w /= np.mean(smoothed_w)
    class_weights = {i: float(round(w, 4)) for i, w in enumerate(smoothed_w)}

    alpha_raw = 1.0 / counts
    alpha_norm = (alpha_raw / alpha_raw.sum() * NUM_CLASSES).tolist()

    return splits, class_weights, alpha_norm, classes

# --─ tf.data pipeline for EfficientNetV2 ----------------------------------

def build_tf_dataset(paths, meta_list, labels, shuffle=False, augment=False):
    """
    EfficientNetV2-S uses its own preprocessing:
    input pixels are expected in [0, 255] range,
    preprocessing is built into the model via preprocess_input.
    """
    meta_arr = np.array(meta_list, dtype=np.float32)
    labels_oh = tf.keras.utils.to_categorical(labels, NUM_CLASSES)
    IMG_H, IMG_W = IMG_SIZE

    def load_img(path_bytes, meta, label):
        path_str = path_bytes.decode('utf-8')
        with Image.open(path_str) as img:
            rgb = img.convert('RGB').resize(IMG_SIZE, Image.Resampling.LANCZOS)
        arr = np.array(rgb, dtype=np.float32)  # [0, 255], EfficientNetV2 preprocess handles normalization
        return arr, meta, label

    def wrapper(path, meta, label):
        img, m, l = tf.numpy_function(load_img, [path, meta, label],
                                       [tf.float32, tf.float32, tf.float32])
        img.set_shape([IMG_H, IMG_W, 3])
        m.set_shape([meta_arr.shape[1]])
        l.set_shape([NUM_CLASSES])
        return {"input_image": img, "input_metadata": m}, l

    paths_t  = tf.constant(paths)
    meta_t   = tf.constant(meta_arr)
    labels_t = tf.constant(labels_oh, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths_t, meta_t, labels_t))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 8000), seed=SEED)
    ds = ds.map(wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# --─ EfficientNetV2-S Multimodal Model ------------------------------------

def build_efficientnetv2s_multimodal(meta_dim):
    """
    EfficientNetV2-S Dual-Branch Multimodal Network.
    
    Vision: EfficientNetV2-S (21.5M params, ImageNet top-1 83.9%)
    Metadata: 3-layer MLP with BatchNorm
    Fusion: Concatenate → 2 Dense layers
    """
    reg = keras.regularizers.l2(L2_REGULARIZATION)

    # -- Image branch ------------------------------------------------------─
    input_image = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input_image")

    # On-the-fly augmentation (training mode only)
    x_img = layers.RandomFlip("horizontal")(input_image)
    x_img = layers.RandomFlip("vertical")(x_img)
    x_img = layers.RandomRotation(0.12)(x_img)
    x_img = layers.RandomZoom(0.12)(x_img)
    x_img = layers.RandomContrast(0.2)(x_img)

    # EfficientNetV2-S preprocess_input: scales to [-1, 1]
    x_img = keras.applications.efficientnet_v2.preprocess_input(x_img)

    base_model = keras.applications.EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_tensor=x_img,
        include_preprocessing=False,   # We handle preprocessing above
    )
    base_model.trainable = False

    img_feat = base_model.output
    img_feat = layers.GlobalAveragePooling2D(name="img_gap")(img_feat)
    img_feat = layers.BatchNormalization(name="img_bn")(img_feat)
    img_feat = layers.Dropout(0.4, name="img_drop")(img_feat)
    img_feat = layers.Dense(512, activation="relu", kernel_regularizer=reg, name="img_d1")(img_feat)
    img_feat = layers.BatchNormalization(name="img_d1_bn")(img_feat)
    img_feat = layers.Dropout(0.3, name="img_d1_drop")(img_feat)
    img_feat = layers.Dense(256, activation="relu", kernel_regularizer=reg, name="img_d2")(img_feat)
    img_feat = layers.BatchNormalization(name="img_d2_bn")(img_feat)

    # -- Metadata branch ----------------------------------------------------
    input_metadata = keras.Input(shape=(meta_dim,), name="input_metadata")
    meta = layers.Dense(128, activation="relu", kernel_regularizer=reg, name="meta_d1")(input_metadata)
    meta = layers.BatchNormalization(name="meta_bn1")(meta)
    meta = layers.Dropout(0.3, name="meta_drop1")(meta)
    meta = layers.Dense(64, activation="relu", kernel_regularizer=reg, name="meta_d2")(meta)
    meta = layers.BatchNormalization(name="meta_bn2")(meta)
    meta = layers.Dropout(0.2, name="meta_drop2")(meta)
    meta = layers.Dense(48, activation="relu", name="meta_d3")(meta)
    meta = layers.BatchNormalization(name="meta_bn3")(meta)

    # -- Fusion ------------------------------------------------------------─
    fused = layers.Concatenate(name="fusion")([img_feat, meta])   # 256 + 48 = 304
    fused = layers.Dense(256, activation="relu", kernel_regularizer=reg, name="fused_d1")(fused)
    fused = layers.BatchNormalization(name="fused_bn1")(fused)
    fused = layers.Dropout(0.4, name="fused_drop1")(fused)
    fused = layers.Dense(128, activation="relu", kernel_regularizer=reg, name="fused_d2")(fused)
    fused = layers.BatchNormalization(name="fused_bn2")(fused)
    fused = layers.Dropout(0.2, name="fused_drop2")(fused)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32",
                            name="predictions")(fused)

    model = keras.Model(inputs=[input_image, input_metadata], outputs=outputs,
                         name="SkinDisNet_EfficientNetV2S_Multimodal")
    return model, base_model

# --─ TTA (simple, no albumentations needed) --------------------------------─

def predict_with_tta_simple(model, test_paths, test_meta):
    """6-augmentation TTA menggunakan numpy (tanpa albumentations)."""
    print(f"Running TTA on {len(test_paths)} samples...")
    all_probs = []
    for i, (p, meta_vec) in enumerate(zip(test_paths, test_meta)):
        if i % 50 == 0:
            print(f"  TTA progress: {i}/{len(test_paths)}")
        with Image.open(p) as img:
            rgb = img.convert('RGB').resize(IMG_SIZE, Image.Resampling.LANCZOS)
        arr = np.array(rgb, dtype=np.float32)

        # 6 deterministic augmentations
        aug_arrs = [
            arr,                                   # original
            np.fliplr(arr),                         # horizontal flip
            np.flipud(arr),                         # vertical flip
            np.rot90(arr, k=1, axes=(0, 1)),         # rotate 90
            np.rot90(arr, k=2, axes=(0, 1)),         # rotate 180
            np.rot90(arr, k=3, axes=(0, 1)),         # rotate 270
        ]
        img_batch = np.stack(aug_arrs, axis=0)
        meta_batch = np.tile(meta_vec, (6, 1))
        preds = model.predict({"input_image": img_batch, "input_metadata": meta_batch}, verbose=0)
        all_probs.append(np.mean(preds, axis=0))

    return np.array(all_probs)

# --─ Progressive fine-tuning helper ----------------------------------------─

def unfreeze_top_n_layers(base_model, n):
    """Unfreeze top n layers of a nested base model."""
    for layer in base_model.layers[-n:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    frozen_count = sum(1 for l in base_model.layers if not l.trainable)
    total_count = len(base_model.layers)
    print(f"  Unfrozen: {total_count - frozen_count}/{total_count} layers")

# --─ Main training ----------------------------------------------------------

def train_fase8():
    print("=" * 72)
    print("FASE 8: EfficientNetV2-S Backbone (300×300) + Focal Loss")
    print("=" * 72)

    # Check if EfficientNetV2S is available
    if not hasattr(keras.applications, 'EfficientNetV2S'):
        print("[ERROR] EfficientNetV2S tidak tersedia di versi Keras ini.")
        print("Coba: pip install --upgrade tensorflow keras")
        return None

    encoder_info = load_encoder()
    splits, class_weights, alpha_norm, classes = load_dataset_splits(encoder_info)
    meta_dim = len(splits['train']['meta'][0])

    print(f"\nFocal Loss alpha per-class: {[f'{a:.3f}' for a in alpha_norm]}")
    print(f"Class weights: {class_weights}")
    print(f"Input resolution: {IMG_SIZE[0]}×{IMG_SIZE[1]}")

    # -- Datasets --------------------------------------------------------─
    train_ds = build_tf_dataset(splits['train']['paths'], splits['train']['meta'],
                                 splits['train']['labels'], shuffle=True, augment=True)
    val_ds   = build_tf_dataset(splits['val']['paths'], splits['val']['meta'],
                                 splits['val']['labels'])
    test_ds  = build_tf_dataset(splits['test']['paths'], splits['test']['meta'],
                                 splits['test']['labels'])

    # -- Model ------------------------------------------------------------
    print("\nBuilding EfficientNetV2-S Multimodal model...")
    model, base_model = build_efficientnetv2s_multimodal(meta_dim)
    trainable_params = sum(tf.size(v).numpy() for v in model.trainable_variables)
    total_params = model.count_params()
    print(f"Total params: {total_params:,} | Trainable (head): {trainable_params:,}")

    focal_loss = FocalLoss(gamma=FOCAL_GAMMA, alpha=alpha_norm, label_smoothing=LABEL_SMOOTHING)

    # -- Stage 1: Warmup (fully frozen backbone, lr=2e-3, 8 epochs) ------
    print("\n-- Stage 1: Warmup head (backbone fully frozen, lr=2e-3, 8 epochs) --")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-3, weight_decay=1e-5),
        loss=focal_loss,
        metrics=['accuracy']
    )
    cb = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4,
                                       restore_best_weights=True, mode='max'),
        keras.callbacks.ModelCheckpoint(OUTPUT_MODEL_PATH, monitor='val_accuracy',
                                         save_best_only=True, mode='max', verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2,
                                           min_lr=1e-6, verbose=1),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=8,
              class_weight=class_weights, callbacks=cb, verbose=1)

    # -- Stage 2: Unfreeze top 30 layers (lr=5e-4) --------------------─
    print("\n-- Stage 2: Unfreeze top 30 EfficientNetV2-S layers (lr=5e-4, 10 epochs) --")
    unfreeze_top_n_layers(base_model, 30)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4, weight_decay=1e-5),
        loss=focal_loss,
        metrics=['accuracy']
    )
    cb[0] = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5,
                                            restore_best_weights=True, mode='max')
    model.fit(train_ds, validation_data=val_ds, epochs=10,
              class_weight=class_weights, callbacks=cb, verbose=1)

    # -- Stage 3: Unfreeze top 80 layers (lr=1e-4) --------------------─
    print("\n-- Stage 3: Unfreeze top 80 layers (lr=1e-4, 15 epochs) --")
    unfreeze_top_n_layers(base_model, 80)
    lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-4,
        first_decay_steps=(len(splits['train']['paths']) // BATCH_SIZE) * 5,
        t_mul=2.0, m_mul=0.8
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule, weight_decay=1e-5),
        loss=focal_loss,
        metrics=['accuracy']
    )
    cb[0] = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7,
                                            restore_best_weights=True, mode='max')
    model.fit(train_ds, validation_data=val_ds, epochs=15,
              class_weight=class_weights, callbacks=cb, verbose=1)

    # -- Stage 4: Full fine-tune (lr=2e-5) ----------------------------─
    print("\n-- Stage 4: Full fine-tune (lr=2e-5, 10 epochs) --")
    for layer in base_model.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-5, weight_decay=1e-5),
        loss=focal_loss,
        metrics=['accuracy']
    )
    cb[0] = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8,
                                            restore_best_weights=True, mode='max')
    model.fit(train_ds, validation_data=val_ds, epochs=10,
              class_weight=class_weights, callbacks=cb, verbose=1)

    # -- Reload best ----------------------------------------------------
    print(f"\nLoading best model from: {OUTPUT_MODEL_PATH}")
    model = keras.models.load_model(OUTPUT_MODEL_PATH, custom_objects={"FocalLoss": FocalLoss})

    # -- Standard evaluation --------------------------------------------
    print("\n-- Standard Evaluation --")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Standard Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    y_true = splits['test']['labels']
    y_pred_std = []
    for X_batch, _ in test_ds:
        preds = model.predict(X_batch, verbose=0)
        y_pred_std.extend(np.argmax(preds, axis=1))
    y_pred_std = y_pred_std[:len(y_true)]

    # -- TTA evaluation ------------------------------------------------─
    print("\n-- TTA Evaluation (6 augmentations) --")
    test_meta = np.array(splits['test']['meta'], dtype=np.float32)
    tta_probs = predict_with_tta_simple(model, splits['test']['paths'], test_meta)
    y_pred_tta = np.argmax(tta_probs, axis=1)
    tta_acc = accuracy_score(y_true, y_pred_tta)
    print(f"TTA Test Accuracy: {tta_acc:.4f} ({tta_acc*100:.2f}%)")

    # Use TTA predictions for report
    y_pred = y_pred_tta
    best_acc = tta_acc

    print("\n-- Per-Class Report (TTA) --")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Fase 8 EfficientNetV2-S Confusion Matrix\nTTA Accuracy: {tta_acc*100:.2f}%",
              fontsize=13)
    plt.ylabel('True Label', fontsize=11)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    cm_path = os.path.join(ARTIFACTS_DIR, "fase8_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {cm_path}")

    report_dict = classification_report(y_true, y_pred, target_names=classes,
                                         digits=4, output_dict=True)
    metrics = {
        "model_name": "Fase8_EfficientNetV2S_300x300_FocalLoss_Multimodal",
        "standard_test_accuracy": float(test_acc),
        "tta_test_accuracy": float(tta_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(f1_score(y_true, y_pred, average='weighted')),
        "img_size": list(IMG_SIZE),
        "backbone": "EfficientNetV2-S",
        "focal_gamma": FOCAL_GAMMA,
        "focal_alpha": alpha_norm,
        "test_sample_count": len(y_true),
        "classes": classes,
        "per_class_metrics": {c: report_dict[c] for c in classes if c in report_dict},
        "confusion_matrix": cm.tolist(),
    }
    with open(METRICS_OUT_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {METRICS_OUT_PATH}")

    print("\n" + "=" * 72)
    print(f"FASE 8 SELESAI!")
    print(f"  Standard Accuracy : {test_acc*100:.2f}%")
    print(f"  TTA Accuracy      : {tta_acc*100:.2f}%")
    print(f"  Macro F1          : {macro_f1*100:.2f}%")
    print(f"  Model saved       : {OUTPUT_MODEL_PATH}")
    print("=" * 72)
    return metrics

if __name__ == "__main__":
    train_fase8()
