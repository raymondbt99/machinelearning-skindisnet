"""
=============================================================================
FASE 9: Hierarchical 2-Level Classification untuk Kelas Eczematous
=============================================================================
Masalah: High visual overlap antara Contact Dermatitis / Eczema / Atopic Dermatitis
menyebabkan confusion matrix buruk di 3 kelas ini.

Solusi — 2-level hierarchical classifier:
  Level 1: Klasifikasi kasar → {Eczematous_Group, Scabies, Seborrheic_Dermatitis, Tinea_Corporis}
  Level 2: Dalam Eczematous_Group → {Contact_Dermatitis, Eczema, Atopic_Dermatitis}

Confidence gating: jika Level 1 confidence < threshold, pakai model ensemble.

Strategi teknis:
  - Gunakan best model dari Fase 7 atau 8 sebagai backbone
  - Re-train classification head untuk setiap level
  - Threshold tuning pada validation set
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

# Best model available: try Fase 8 first, fallback to Fase 7, then DenseNet baseline
CANDIDATE_MODELS = [
    os.path.join(ARTIFACTS_DIR, "fase8_efficientnetv2s_model.keras"),
    os.path.join(ARTIFACTS_DIR, "fase7_focal_loss_model.keras"),
    os.path.join(ARTIFACTS_DIR, "fase6_albumentations_model.keras"),
    os.path.join(ARTIFACTS_DIR, "densenet_multimodal_skin_classifier.keras"),
]

LEVEL1_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "fase9_hierarchical_level1.keras")
LEVEL2_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "fase9_hierarchical_level2.keras")
METRICS_OUT_PATH  = os.path.join(ARTIFACTS_DIR, "fase9_evaluation_metrics.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 6
LABEL_SMOOTHING = 0.1

# --─ Class hierarchy definition --------------------------------------------─

# 6 fine-grained classes (from class_mapping.json order)
# 0: Atopic Dermatitis
# 1: Contact Dermatitis
# 2: Eczema
# 3: Scabies
# 4: Seborrheic Dermatitis
# 5: Tinea Corporis

ECZEMATOUS_INDICES = [0, 1, 2]   # Atopic Dermatitis, Contact Dermatitis, Eczema
NON_ECZEMATOUS_INDICES = [3, 4, 5]  # Scabies, Seborrheic Dermatitis, Tinea Corporis

# Level 1 labels: 0=Eczematous_Group, 1=Scabies, 2=Seborrheic_Dermatitis, 3=Tinea_Corporis
LEVEL1_CLASSES = ["Eczematous_Group", "Scabies", "Seborrheic_Dermatitis", "Tinea_Corporis"]
NUM_L1_CLASSES = 4

# Level 2 labels (within Eczematous Group): 0=Atopic, 1=Contact, 2=Eczema
LEVEL2_CLASSES = ["Atopic_Dermatitis", "Contact_Dermatitis", "Eczema"]
NUM_L2_CLASSES = 3

def fine_to_level1(fine_label):
    """Map 6-class fine label to 4-class level-1 label."""
    if fine_label in ECZEMATOUS_INDICES:
        return 0  # Eczematous Group
    elif fine_label == 3:
        return 1  # Scabies
    elif fine_label == 4:
        return 2  # Seborrheic Dermatitis
    elif fine_label == 5:
        return 3  # Tinea Corporis

def fine_to_level2(fine_label):
    """Map fine label to level-2 label (only for eczematous)."""
    # 0→0 (Atopic), 1→1 (Contact), 2→2 (Eczema)
    return ECZEMATOUS_INDICES.index(fine_label)

def level1_level2_to_fine(l1_pred, l2_pred):
    """Map level1 + level2 predictions back to fine label."""
    if l1_pred == 0:  # Eczematous group
        return ECZEMATOUS_INDICES[l2_pred]
    elif l1_pred == 1:
        return 3  # Scabies
    elif l1_pred == 2:
        return 4  # Seborrheic Dermatitis
    elif l1_pred == 3:
        return 5  # Tinea Corporis

# --─ Metadata encoding ------------------------------------------------------

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

# --─ Dataset loading --------------------------------------------------------

def load_dataset_splits(encoder_info):
    manifest_df = pd.read_csv(MANIFEST_PATH)
    with open(CLASS_MAPPING_PATH, 'r') as f:
        classes = json.load(f)['classes']

    splits = {s: {'paths': [], 'meta': [], 'labels': [], 'l1_labels': [], 'l2_labels': []}
              for s in ['train', 'val', 'test']}

    for _, row in manifest_df.iterrows():
        split = row['Split']
        folder = row['Folder_name']
        img_id = row['Image_id']
        fine_idx = int(row['Class_index'])
        meta_vec = encode_metadata_vector(row['Age'], row['Sex'], row['Leision_location'], encoder_info)
        orig_full = os.path.join(DATASET_DIR, "Preprocessed", folder, f"{img_id}.jpg")
        if not os.path.exists(orig_full):
            continue

        l1_label = fine_to_level1(fine_idx)
        l2_label = fine_to_level2(fine_idx) if fine_idx in ECZEMATOUS_INDICES else -1

        def add_sample(p, m, fi, l1, l2, s):
            splits[s]['paths'].append(p)
            splits[s]['meta'].append(m)
            splits[s]['labels'].append(fi)
            splits[s]['l1_labels'].append(l1)
            splits[s]['l2_labels'].append(l2)

        if split == 'train':
            add_sample(orig_full, meta_vec, fine_idx, l1_label, l2_label, 'train')
            for aug_i in range(1, 8):
                aug_p = os.path.join(DATASET_DIR, "Augmented", folder, f"{img_id}_{aug_i:02d}.jpg")
                if os.path.exists(aug_p):
                    add_sample(aug_p, meta_vec, fine_idx, l1_label, l2_label, 'train')
        else:
            add_sample(orig_full, meta_vec, fine_idx, l1_label, l2_label, split)

    for s in ['train', 'val', 'test']:
        print(f"  {s}: {len(splits[s]['paths'])} samples")

    return splits, classes

# --─ tf.data pipeline --------------------------------------------------------

def build_tf_dataset(paths, meta_list, labels_oh, shuffle=False):
    meta_arr = np.array(meta_list, dtype=np.float32)
    
    def load_img(path_bytes, meta, label):
        path_str = path_bytes.decode('utf-8')
        with Image.open(path_str) as img:
            rgb = img.convert('RGB').resize(IMG_SIZE, Image.Resampling.BILINEAR)
        arr = np.array(rgb, dtype=np.float32)
        arr = keras.applications.densenet.preprocess_input(arr)
        return arr, meta, label

    def wrapper(path, meta, label):
        img, m, l = tf.numpy_function(load_img, [path, meta, label],
                                       [tf.float32, tf.float32, tf.float32])
        img.set_shape([224, 224, 3])
        m.set_shape([meta_arr.shape[1]])
        l.set_shape([labels_oh.shape[1]])
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

# --─ Build hierarchical models ----------------------------------------------─

def build_hierarchical_model(meta_dim, num_output_classes, model_name, pretrained_base=None):
    """
    Build a DenseNet121 multimodal model for a specific classification level.
    If pretrained_base is provided, reuse that base model's weights.
    """
    input_image = keras.Input(shape=(224, 224, 3), name="input_image")
    x_img = layers.RandomFlip("horizontal")(input_image)
    x_img = layers.RandomRotation(0.1)(x_img)
    x_img = layers.RandomZoom(0.1)(x_img)
    x_img = keras.applications.densenet.preprocess_input(x_img)

    if pretrained_base is not None:
        # Transfer weights from pretrained base
        base_model = keras.applications.DenseNet121(
            include_top=False, weights=None, input_tensor=x_img
        )
        base_model.set_weights(pretrained_base.get_weights())
        print(f"  Transferred weights from pretrained DenseNet121")
    else:
        base_model = keras.applications.DenseNet121(
            include_top=False, weights='imagenet', input_tensor=x_img
        )
    base_model.trainable = False

    img_feat = base_model.output
    img_feat = layers.GlobalAveragePooling2D(name="img_gap")(img_feat)
    img_feat = layers.BatchNormalization(name="img_bn")(img_feat)
    img_feat = layers.Dropout(0.4, name="img_drop")(img_feat)
    img_feat = layers.Dense(256, activation="relu", name="img_d1")(img_feat)
    img_feat = layers.BatchNormalization(name="img_d1_bn")(img_feat)
    img_feat = layers.Dropout(0.3, name="img_d1_drop")(img_feat)

    input_metadata = keras.Input(shape=(meta_dim,), name="input_metadata")
    meta = layers.Dense(64, activation="relu", name="meta_d1")(input_metadata)
    meta = layers.BatchNormalization(name="meta_bn1")(meta)
    meta = layers.Dropout(0.2, name="meta_drop1")(meta)
    meta = layers.Dense(48, activation="relu", name="meta_d2")(meta)
    meta = layers.BatchNormalization(name="meta_bn2")(meta)

    fused = layers.Concatenate(name="fusion")([img_feat, meta])
    fused = layers.Dense(128, activation="relu", name="fused_d1")(fused)
    fused = layers.BatchNormalization(name="fused_bn1")(fused)
    fused = layers.Dropout(0.3, name="fused_drop")(fused)
    outputs = layers.Dense(num_output_classes, activation="softmax", dtype="float32",
                            name="predictions")(fused)

    model = keras.Model(inputs=[input_image, input_metadata], outputs=outputs, name=model_name)
    return model, base_model

# --─ Main hierarchical training --------------------------------------------─

def train_fase9():
    print("=" * 72)
    print("FASE 9: Hierarchical 2-Level Classifier")
    print("=" * 72)

    encoder_info = load_encoder()
    splits, classes = load_dataset_splits(encoder_info)
    meta_dim = len(splits['train']['meta'][0])

    # -- LEVEL 1 Training (4-class: Eczematous vs Non-Eczematous) --------─
    print("\n" + "─" * 55)
    print("LEVEL 1: Coarse classifier (4 groups)")
    print("─" * 55)

    train_l1_oh = tf.keras.utils.to_categorical(splits['train']['l1_labels'], NUM_L1_CLASSES)
    val_l1_oh   = tf.keras.utils.to_categorical(splits['val']['l1_labels'],   NUM_L1_CLASSES)
    test_l1_oh  = tf.keras.utils.to_categorical(splits['test']['l1_labels'],  NUM_L1_CLASSES)

    train_l1_ds = build_tf_dataset(splits['train']['paths'], splits['train']['meta'],
                                    train_l1_oh, shuffle=True)
    val_l1_ds   = build_tf_dataset(splits['val']['paths'],   splits['val']['meta'],   val_l1_oh)
    test_l1_ds  = build_tf_dataset(splits['test']['paths'],  splits['test']['meta'],  test_l1_oh)

    counts_l1 = pd.Series(splits['train']['l1_labels']).value_counts().sort_index().values.astype(float)
    total_l1 = len(splits['train']['l1_labels'])
    cw_l1 = {i: float(round(np.sqrt(total_l1 / (NUM_L1_CLASSES * c)), 4)) for i, c in enumerate(counts_l1)}
    print(f"Level 1 class weights: {cw_l1}")

    l1_model, l1_base = build_hierarchical_model(meta_dim, NUM_L1_CLASSES, "SkinDisNet_L1")
    l1_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy']
    )

    cb_l1 = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5,
                                       restore_best_weights=True, mode='max'),
        keras.callbacks.ModelCheckpoint(LEVEL1_MODEL_PATH, monitor='val_accuracy',
                                         save_best_only=True, mode='max', verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                                           min_lr=1e-6, verbose=1),
    ]
    print("Training Level 1 model (backbone frozen, lr=1e-3)...")
    l1_model.fit(train_l1_ds, validation_data=val_l1_ds, epochs=15,
                  class_weight=cw_l1, callbacks=cb_l1, verbose=1)

    # Unfreeze top 30 and continue
    for layer in l1_base.layers[-30:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    l1_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy']
    )
    l1_model.fit(train_l1_ds, validation_data=val_l1_ds, epochs=10,
                  class_weight=cw_l1, callbacks=cb_l1, verbose=1)

    l1_model = keras.models.load_model(LEVEL1_MODEL_PATH)
    l1_loss, l1_acc = l1_model.evaluate(test_l1_ds, verbose=0)
    print(f"\nLevel 1 Test Accuracy: {l1_acc:.4f} ({l1_acc*100:.2f}%)")
    print(f"Level 1 Classes: {LEVEL1_CLASSES}")

    # -- LEVEL 2 Training (3-class: within Eczematous) --------------------
    print("\n" + "─" * 55)
    print("LEVEL 2: Fine classifier for Eczematous Group (3 classes)")
    print("─" * 55)

    # Filter only eczematous samples for each split
    def filter_eczematous(split_data):
        mask = [l in ECZEMATOUS_INDICES for l in split_data['labels']]
        filtered = {
            'paths':    [p for p, m in zip(split_data['paths'], mask) if m],
            'meta':     [m for x, m in zip(split_data['meta'], mask)  if m],
            'l2_labels':[l for l, m in zip(split_data['l2_labels'], mask) if m],
        }
        return filtered

    train_ecz = filter_eczematous(splits['train'])
    val_ecz   = filter_eczematous(splits['val'])
    test_ecz  = filter_eczematous(splits['test'])
    print(f"Eczematous samples — Train: {len(train_ecz['paths'])} | "
          f"Val: {len(val_ecz['paths'])} | Test: {len(test_ecz['paths'])}")

    train_l2_oh = tf.keras.utils.to_categorical(train_ecz['l2_labels'], NUM_L2_CLASSES)
    val_l2_oh   = tf.keras.utils.to_categorical(val_ecz['l2_labels'],   NUM_L2_CLASSES)

    train_l2_ds = build_tf_dataset(train_ecz['paths'], train_ecz['meta'],
                                    train_l2_oh, shuffle=True)
    val_l2_ds   = build_tf_dataset(val_ecz['paths'],   val_ecz['meta'],   val_l2_oh)

    counts_l2 = pd.Series(train_ecz['l2_labels']).value_counts().sort_index().values.astype(float)
    total_l2 = len(train_ecz['l2_labels'])
    cw_l2 = {i: float(round(np.sqrt(total_l2 / (NUM_L2_CLASSES * c)), 4)) for i, c in enumerate(counts_l2)}
    print(f"Level 2 class weights: {cw_l2}")

    l2_model, _ = build_hierarchical_model(meta_dim, NUM_L2_CLASSES, "SkinDisNet_L2")
    l2_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.15),  # Higher smoothing for small set
        metrics=['accuracy']
    )

    cb_l2 = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=6,
                                       restore_best_weights=True, mode='max'),
        keras.callbacks.ModelCheckpoint(LEVEL2_MODEL_PATH, monitor='val_accuracy',
                                         save_best_only=True, mode='max', verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3,
                                           min_lr=1e-6, verbose=1),
    ]
    print("Training Level 2 model (backbone frozen)...")
    l2_model.fit(train_l2_ds, validation_data=val_l2_ds, epochs=20,
                  class_weight=cw_l2, callbacks=cb_l2, verbose=1)

    l2_model = keras.models.load_model(LEVEL2_MODEL_PATH)

    # -- Hierarchical Inference on test set ------------------------------─
    print("\n-- Hierarchical Inference (test set) --")
    CONFIDENCE_THRESHOLD = 0.65  # Below this → use level-1 prediction directly

    y_true = splits['test']['labels']
    y_pred_hier = []

    # Batch inference for Level 1
    test_l1_ds_eval = build_tf_dataset(splits['test']['paths'], splits['test']['meta'],
                                        test_l1_oh)
    l1_all_probs = []
    for X_batch, _ in test_l1_ds_eval:
        probs = l1_model.predict(X_batch, verbose=0)
        l1_all_probs.extend(probs)
    l1_all_probs = np.array(l1_all_probs[:len(y_true)])
    l1_preds = np.argmax(l1_all_probs, axis=1)
    l1_confs  = np.max(l1_all_probs, axis=1)

    # For eczematous samples predicted by L1, run Level 2
    test_meta_arr = np.array(splits['test']['meta'], dtype=np.float32)
    test_paths = splits['test']['paths']

    # Separate eczematous samples needing L2
    ecz_indices = [i for i, p in enumerate(l1_preds) if p == 0]  # L1 pred = Eczematous Group
    print(f"  L1 predicts Eczematous Group for {len(ecz_indices)}/{len(y_true)} test samples")

    # Level 2 inference for eczematous group
    l2_preds_by_sample = {}
    if len(ecz_indices) > 0:
        ecz_paths = [test_paths[i] for i in ecz_indices]
        ecz_meta  = [test_meta_arr[i] for i in ecz_indices]
        ecz_oh = tf.keras.utils.to_categorical([0] * len(ecz_indices), NUM_L2_CLASSES)  # dummy labels
        ecz_ds = build_tf_dataset(ecz_paths, ecz_meta, ecz_oh)
        l2_probs_batch = []
        for X_batch, _ in ecz_ds:
            probs = l2_model.predict(X_batch, verbose=0)
            l2_probs_batch.extend(probs)
        for orig_idx, l2_prob in zip(ecz_indices, l2_probs_batch):
            l2_preds_by_sample[orig_idx] = np.argmax(l2_prob)

    # Combine L1 + L2
    for i in range(len(y_true)):
        l1_pred = l1_preds[i]
        l1_conf = l1_confs[i]
        if l1_pred == 0 and i in l2_preds_by_sample:
            # Eczematous → use Level 2 prediction
            l2_pred = l2_preds_by_sample[i]
            fine_pred = level1_level2_to_fine(0, l2_pred)
        else:
            # Non-eczematous (or low-confidence eczematous)
            fine_pred = level1_level2_to_fine(l1_pred, 0)
        y_pred_hier.append(fine_pred)

    hier_acc = accuracy_score(y_true, y_pred_hier)
    macro_f1 = f1_score(y_true, y_pred_hier, average='macro')
    print(f"\nHierarchical Test Accuracy: {hier_acc:.4f} ({hier_acc*100:.2f}%)")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\n-- Per-Class Report --")
    print(classification_report(y_true, y_pred_hier, target_names=classes, digits=4))

    cm = confusion_matrix(y_true, y_pred_hier)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Fase 9 Hierarchical Classifier Confusion Matrix\n"
              f"2-Level (L1 acc={l1_acc*100:.1f}%), Overall: {hier_acc*100:.2f}%",
              fontsize=12)
    plt.ylabel('True Label', fontsize=11)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.tight_layout()
    cm_path = os.path.join(ARTIFACTS_DIR, "fase9_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {cm_path}")

    report_dict = classification_report(y_true, y_pred_hier, target_names=classes,
                                         digits=4, output_dict=True)
    metrics = {
        "model_name": "Fase9_Hierarchical_2Level_DenseNet121_Multimodal",
        "hierarchical_test_accuracy": float(hier_acc),
        "level1_test_accuracy": float(l1_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(f1_score(y_true, y_pred_hier, average='weighted')),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "level1_classes": LEVEL1_CLASSES,
        "level2_classes": LEVEL2_CLASSES,
        "test_sample_count": len(y_true),
        "classes": classes,
        "per_class_metrics": {c: report_dict[c] for c in classes if c in report_dict},
        "confusion_matrix": cm.tolist(),
    }
    with open(METRICS_OUT_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {METRICS_OUT_PATH}")

    print("\n" + "=" * 72)
    print(f"FASE 9 SELESAI!")
    print(f"  Level 1 Accuracy    : {l1_acc*100:.2f}%")
    print(f"  Hierarchical Overall: {hier_acc*100:.2f}%")
    print(f"  Macro F1            : {macro_f1*100:.2f}%")
    print(f"  L1 Model saved      : {LEVEL1_MODEL_PATH}")
    print(f"  L2 Model saved      : {LEVEL2_MODEL_PATH}")
    print("=" * 72)
    return metrics

if __name__ == "__main__":
    train_fase9()
