"""
=============================================================================
FASE 7 FAST: Focal Loss + Smart Oversampling (CPU-Optimized)
=============================================================================
Versi dioptimalkan untuk CPU training:
  - Dataset lebih kecil (1.5x max class, bukan 1.5x * 6 class)  
  - BATCH_SIZE lebih besar (64) -> lebih sedikit steps per epoch
  - Epoch lebih sedikit tapi dengan EarlyStopping agresif
  - Mixup ringan (alpha=0.2)
  - Focal Loss tetap dipakai (fitur utama Fase 7)
  
Estimasi waktu: ~45-90 menit (vs 14 jam versi full)
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
DENSENET_BASELINE_PATH = os.path.join(ARTIFACTS_DIR, "densenet_multimodal_skin_classifier.keras")
OUTPUT_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "fase7_focal_loss_model.keras")
METRICS_OUT_PATH = os.path.join(ARTIFACTS_DIR, "fase7_evaluation_metrics.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 64          # Larger batch = fewer steps per epoch = faster
NUM_CLASSES = 6
FOCAL_GAMMA = 2.0
MIXUP_ALPHA = 0.2
OVERSAMPLE_RATIO = 0.5   # Oversample minority to 50% more than their natural count (conservative)

# ─── Focal Loss ─────────────────────────────────────────────────────────────

class FocalLoss(keras.losses.Loss):
    """Focal Loss: FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)"""
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.05, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true_s = y_true * (1.0 - self.label_smoothing) + self.label_smoothing / NUM_CLASSES
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true_s * tf.math.log(y_pred)
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
        cfg = super().get_config()
        cfg.update({'gamma': self.gamma, 'alpha': self.alpha,
                    'label_smoothing': self.label_smoothing})
        return cfg

# ─── Mixup ─────────────────────────────────────────────────────────────────

def mixup_batch(images, labels, alpha=0.2):
    batch_size = tf.shape(images)[0]
    lam = tf.random.uniform([batch_size], minval=alpha, maxval=1.0)
    lam = tf.reshape(tf.maximum(lam, 1.0 - lam), [-1, 1, 1, 1])
    idx = tf.random.shuffle(tf.range(batch_size))
    mixed_img = lam * images + (1.0 - lam) * tf.gather(images, idx)
    lam2 = tf.reshape(lam, [-1, 1])
    mixed_lbl = lam2 * labels + (1.0 - lam2) * tf.gather(labels, idx)
    return mixed_img, mixed_lbl

# ─── Metadata encoding ──────────────────────────────────────────────────────

def load_encoder():
    with open(ENCODER_PATH) as f:
        return json.load(f)

def encode_metadata_vector(age_val, sex_val, loc_val, enc):
    n = enc["num_locations"]
    vec = np.zeros(n + 3, dtype=np.float32)
    loc = str(loc_val).strip().title()
    vec[enc["location_to_idx"].get(loc, enc["location_to_idx"]["Unknown"])] = 1.0
    try:
        vec[n] = np.clip(float(age_val) / enc["max_age"], 0.0, 1.2)
    except:
        vec[n] = 0.5
    s = str(sex_val).strip().title()
    if s == 'Female':   vec[n+1] = 1.0
    elif s == 'Male':   vec[n+2] = 1.0
    else:               vec[n+1] = vec[n+2] = 0.5
    return vec

# ─── Smart oversampling (moderate) ──────────────────────────────────────────

def build_oversampled_splits(splits):
    """
    Smart oversampling: hanya upsample kelas yang count-nya < median.
    Target per kelas = median class count (bukan max), lebih balanced tapi
    tidak memperbesar dataset terlalu besar.
    """
    by_cls = {i: {'paths': [], 'meta': []} for i in range(NUM_CLASSES)}
    for p, m, l in zip(splits['train']['paths'], splits['train']['meta'], splits['train']['labels']):
        by_cls[l]['paths'].append(p)
        by_cls[l]['meta'].append(m)

    counts = [len(by_cls[i]['paths']) for i in range(NUM_CLASSES)]
    target = int(np.median(counts) * 1.5)
    print(f"Class counts before oversampling: {counts}")
    print(f"Oversample target per class     : {target}")

    all_p, all_m, all_l = [], [], []
    for cls in range(NUM_CLASSES):
        ps = by_cls[cls]['paths']
        ms = by_cls[cls]['meta']
        n = len(ps)
        t = max(n, target)  # Only upsample, never downsample
        reps = (t // n) + 1
        ps_ext = (ps * reps)[:t]
        ms_ext = (ms * reps)[:t]
        all_p.extend(ps_ext)
        all_m.extend(ms_ext)
        all_l.extend([cls] * t)

    idx = np.random.permutation(len(all_p))
    all_p = [all_p[i] for i in idx]
    all_m = [all_m[i] for i in idx]
    all_l = [all_l[i] for i in idx]

    new_counts = pd.Series(all_l).value_counts().sort_index().tolist()
    print(f"Class counts after  oversampling: {new_counts}")
    print(f"Total training samples          : {len(all_p)}")
    return all_p, all_m, all_l

# ─── tf.data pipeline ───────────────────────────────────────────────────────

def build_tf_dataset(paths, meta_list, labels, shuffle=False, use_mixup=False):
    meta_arr = np.array(meta_list, dtype=np.float32)
    labels_oh = tf.keras.utils.to_categorical(labels, NUM_CLASSES).astype(np.float32)

    def load_img(p_bytes, meta, lbl):
        with Image.open(p_bytes.decode()) as img:
            arr = np.array(img.convert('RGB').resize(IMG_SIZE, Image.Resampling.BILINEAR),
                           dtype=np.float32)
        # DenseNet preprocess: channel-wise subtract ImageNet mean (in [0,255] space)
        arr[..., 0] -= 103.939
        arr[..., 1] -= 116.779
        arr[..., 2] -= 123.68
        return arr, meta, lbl

    def wrapper(p, m, l):
        img, mo, lo = tf.numpy_function(load_img, [p, m, l],
                                         [tf.float32, tf.float32, tf.float32])
        img.set_shape([224, 224, 3])
        mo.set_shape([meta_arr.shape[1]])
        lo.set_shape([NUM_CLASSES])
        return {"input_image": img, "input_metadata": mo}, lo

    ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(paths), tf.constant(meta_arr), tf.constant(labels_oh))
    )
    if shuffle:
        ds = ds.shuffle(min(len(paths), 4000), seed=SEED)
    ds = ds.map(wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    if use_mixup:
        def apply_mixup(X, y):
            imgs, lbl = mixup_batch(X["input_image"], y, MIXUP_ALPHA)
            return {"input_image": imgs, "input_metadata": X["input_metadata"]}, lbl
        ds = ds.map(apply_mixup)
    return ds

# ─── Build model (load baseline + richer head) ──────────────────────────────

def build_model_from_baseline(meta_dim, focal_loss):
    """
    Load DenseNet121 baseline dan rebuild classifier head yang lebih kuat.
    Tidak melatih dari scratch - transfer learning dari Fase 3/4.
    """
    print(f"Loading baseline: {DENSENET_BASELINE_PATH}")
    base_model = keras.models.load_model(DENSENET_BASELINE_PATH)

    # Freeze all layers
    for layer in base_model.layers:
        layer.trainable = False

    # Extract feature layers (sebelum predictions layer)
    # Input: input_image (224,224,3) + input_metadata (44,)
    # kita rebuild head baru di atas feature extractor yang sudah ada

    # Get the fused features (before final Dense)
    feat_layer = base_model.get_layer("fused_dense")
    feat_out = feat_layer.output

    # New classification head dengan Focal Loss in mind
    x = layers.Dropout(0.4, name="new_drop1")(feat_out)
    x = layers.Dense(64, activation="relu", name="new_dense1")(x)
    x = layers.BatchNormalization(name="new_bn1")(x)
    x = layers.Dropout(0.3, name="new_drop2")(x)
    new_out = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32",
                            name="new_predictions")(x)

    new_model = keras.Model(
        inputs=base_model.inputs,
        outputs=new_out,
        name="SkinDisNet_FocalLoss_Fast"
    )
    new_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=focal_loss,
        metrics=['accuracy']
    )
    trainable = sum(tf.size(v).numpy() for v in new_model.trainable_variables)
    print(f"Model built. Trainable params (head only): {trainable:,}")
    return new_model, base_model

# ─── Main ───────────────────────────────────────────────────────────────────

def train_fase7_fast():
    print("=" * 70)
    print("FASE 7 FAST: Focal Loss + Smart Oversampling + Mixup (CPU-Optimized)")
    print("=" * 70)

    enc = load_encoder()
    manifest_df = pd.read_csv(MANIFEST_PATH)
    with open(CLASS_MAPPING_PATH) as f:
        classes = json.load(f)['classes']

    splits = {s: {'paths': [], 'meta': [], 'labels': []} for s in ['train','val','test']}
    for _, row in manifest_df.iterrows():
        split = row['Split']
        folder = row['Folder_name']
        img_id = row['Image_id']
        cls = int(row['Class_index'])
        mv = encode_metadata_vector(row['Age'], row['Sex'], row['Leision_location'], enc)
        orig = os.path.join(DATASET_DIR, "Preprocessed", folder, f"{img_id}.jpg")
        if not os.path.exists(orig):
            continue
        if split == 'train':
            splits['train']['paths'].append(orig)
            splits['train']['meta'].append(mv)
            splits['train']['labels'].append(cls)
            for i in range(1, 8):
                ap = os.path.join(DATASET_DIR, "Augmented", folder, f"{img_id}_{i:02d}.jpg")
                if os.path.exists(ap):
                    splits['train']['paths'].append(ap)
                    splits['train']['meta'].append(mv)
                    splits['train']['labels'].append(cls)
        else:
            splits[split]['paths'].append(orig)
            splits[split]['meta'].append(mv)
            splits[split]['labels'].append(cls)

    meta_dim = len(splits['train']['meta'][0])
    print(f"Raw Train: {len(splits['train']['paths'])} | Val: {len(splits['val']['paths'])} | Test: {len(splits['test']['paths'])}")

    # -- Compute focal alpha & class weights --------------------------------
    counts = np.array(pd.Series(splits['train']['labels']).value_counts().sort_index().values, dtype=np.float32)
    alpha_raw = 1.0 / counts
    alpha_norm = (alpha_raw / alpha_raw.sum() * NUM_CLASSES).tolist()
    total = len(splits['train']['labels'])
    sw = np.sqrt(total / (NUM_CLASSES * counts))
    sw /= np.mean(sw)
    class_weights = {i: float(round(w, 4)) for i, w in enumerate(sw)}
    print(f"Focal alpha per-class : {[f'{a:.3f}' for a in alpha_norm]}")
    print(f"Class weights         : {class_weights}")

    # -- Build focal loss ---------------------------------------------------
    focal_loss = FocalLoss(gamma=FOCAL_GAMMA, alpha=alpha_norm, label_smoothing=0.05)

    # -- Smart oversampled dataset ------------------------------------------
    ov_paths, ov_meta, ov_labels = build_oversampled_splits(splits)
    train_ds = build_tf_dataset(ov_paths, ov_meta, ov_labels, shuffle=True, use_mixup=True)
    val_ds   = build_tf_dataset(splits['val']['paths'],  splits['val']['meta'],  splits['val']['labels'])
    test_ds  = build_tf_dataset(splits['test']['paths'], splits['test']['meta'], splits['test']['labels'])

    steps_per_epoch = len(ov_paths) // BATCH_SIZE
    print(f"\nSteps per epoch: {steps_per_epoch} (batch={BATCH_SIZE})")

    # -- Stage 1: Train new head only (backbone fully frozen) ---------------
    print("\n-- Stage 1: New head only (backbone frozen, lr=1e-3, up to 8 epochs) --")
    model, base_model = build_model_from_baseline(meta_dim, focal_loss)

    cb1 = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4,
                                       restore_best_weights=True, mode='max', verbose=1),
        keras.callbacks.ModelCheckpoint(OUTPUT_MODEL_PATH, monitor='val_accuracy',
                                         save_best_only=True, mode='max', verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                           patience=2, min_lr=1e-6, verbose=1),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=8,
              class_weight=class_weights, callbacks=cb1, verbose=1)

    # -- Stage 2: Unfreeze top 30 DenseNet layers ---------------------------
    print("\n-- Stage 2: Unfreeze top 30 DenseNet layers (lr=5e-5, up to 10 epochs) --")
    for layer in model.layers:
        if hasattr(layer, 'layers'):  # nested DenseNet
            for sub in layer.layers[-30:]:
                if not isinstance(sub, layers.BatchNormalization):
                    sub.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-5),
        loss=focal_loss,
        metrics=['accuracy']
    )
    cb2 = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5,
                                       restore_best_weights=True, mode='max', verbose=1),
        keras.callbacks.ModelCheckpoint(OUTPUT_MODEL_PATH, monitor='val_accuracy',
                                         save_best_only=True, mode='max', verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.4,
                                           patience=3, min_lr=1e-7, verbose=1),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=10,
              class_weight=class_weights, callbacks=cb2, verbose=1)

    # -- Load best & evaluate -----------------------------------------------
    print(f"\nLoading best model from {OUTPUT_MODEL_PATH}...")
    model = keras.models.load_model(OUTPUT_MODEL_PATH, custom_objects={"FocalLoss": FocalLoss})

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\nTest Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")

    y_true = splits['test']['labels']
    y_pred = []
    for Xb, _ in test_ds:
        y_pred.extend(np.argmax(model.predict(Xb, verbose=0), axis=1))
    y_pred = y_pred[:len(y_true)]

    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Fase 7 Confusion Matrix (Focal Loss+Oversampling)\nTest Acc: {test_acc*100:.2f}%")
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.tight_layout()
    cm_path = os.path.join(ARTIFACTS_DIR, "fase7_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150); plt.close()
    print(f"Confusion matrix saved: {cm_path}")

    rd = classification_report(y_true, y_pred, target_names=classes, digits=4, output_dict=True)
    metrics = {
        "model_name": "Fase7_Fast_FocalLoss_SmartOversampling_Mixup_DenseNet121",
        "test_accuracy": float(test_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(f1_score(y_true, y_pred, average='weighted')),
        "focal_gamma": FOCAL_GAMMA,
        "focal_alpha_per_class": alpha_norm,
        "mixup_alpha": MIXUP_ALPHA,
        "test_sample_count": len(y_true),
        "classes": classes,
        "per_class_metrics": {c: rd[c] for c in classes if c in rd},
        "confusion_matrix": cm.tolist(),
        "vs_baseline": f"Baseline DenseNet121: 44.06% -> Fase7: {test_acc*100:.2f}%"
    }
    with open(METRICS_OUT_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 70)
    print(f"FASE 7 FAST SELESAI!")
    print(f"  Baseline (Fase 3/4) : 44.06%")
    print(f"  Fase 7 Test Accuracy: {test_acc*100:.2f}%")
    print(f"  Delta               : {(test_acc - 0.4406)*100:+.2f}%")
    print(f"  Macro F1            : {macro_f1*100:.2f}%")
    print(f"  Model saved         : {OUTPUT_MODEL_PATH}")
    print("=" * 70)
    return metrics

if __name__ == "__main__":
    train_fase7_fast()
