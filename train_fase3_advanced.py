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

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

WORKSPACE = r"c:\SkinDisNet A Multi-Class Clinical Images and Metad"
DATASET_DIR = os.path.join(WORKSPACE, "SkinDisNet_2")
ARTIFACTS_DIR = os.path.join(WORKSPACE, "artifacts")
MANIFEST_PATH = os.path.join(ARTIFACTS_DIR, "dataset_split_manifest.csv")
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "multimodal_metadata_encoders.json")
CLASS_MAPPING_PATH = os.path.join(ARTIFACTS_DIR, "class_mapping.json")

# Model paths
EFFNET_MULTIMODAL_PATH = os.path.join(ARTIFACTS_DIR, "multimodal_skin_classifier.keras")
DENSENET_MULTIMODAL_PATH = os.path.join(ARTIFACTS_DIR, "densenet_multimodal_skin_classifier.keras")

IMG_SIZE = (224, 224)
BATCH_SIZE = 64
NUM_CLASSES = 6
LABEL_SMOOTHING = 0.1

def load_encoder():
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Metadata encoder not found at {ENCODER_PATH}")
    with open(ENCODER_PATH, 'r') as f:
        return json.load(f)

def encode_metadata_vector(age_val, sex_val, loc_val, encoder_info):
    """Encodes tabular clinical features into a 44-dimensional float32 vector."""
    num_locs = encoder_info["num_locations"]
    vec = np.zeros(num_locs + 1 + 2, dtype=np.float32)
    
    # 1. Location one-hot
    loc_clean = str(loc_val).strip().title()
    loc_idx = encoder_info["location_to_idx"].get(loc_clean, encoder_info["location_to_idx"]["Unknown"])
    vec[loc_idx] = 1.0
    
    # 2. Normalized Age
    try:
        age_float = float(age_val)
    except Exception:
        age_float = 30.0
    vec[num_locs] = np.clip(age_float / encoder_info["max_age"], 0.0, 1.2)
    
    # 3. Sex one-hot
    sex_str = str(sex_val).strip().title()
    if sex_str == 'Female':
        vec[num_locs + 1] = 1.0
    elif sex_str == 'Male':
        vec[num_locs + 2] = 1.0
    else:
        vec[num_locs + 1] = 0.5
        vec[num_locs + 2] = 0.5
        
    return vec

def load_dataset(encoder_info):
    print(f"Loading split manifest from {MANIFEST_PATH}...")
    manifest_df = pd.read_csv(MANIFEST_PATH)
    
    with open(CLASS_MAPPING_PATH, 'r') as f:
        mapping_data = json.load(f)
    classes = mapping_data['classes']
    
    splits = {
        'train': {'paths': [], 'meta': [], 'labels': []},
        'val': {'paths': [], 'meta': [], 'labels': []},
        'test': {'paths': [], 'meta': [], 'labels': []}
    }
    
    for _, row in manifest_df.iterrows():
        split = row['Split']
        folder = row['Folder_name']
        img_id = row['Image_id']
        class_idx = int(row['Class_index'])
        age = row['Age']
        sex = row['Sex']
        location = row['Leision_location']
        
        meta_vec = encode_metadata_vector(age, sex, location, encoder_info)
        orig_rel = os.path.join("Preprocessed", folder, f"{img_id}.jpg")
        orig_full = os.path.join(DATASET_DIR, orig_rel)
        assert os.path.exists(orig_full), f"File not found: {orig_full}"
        
        if split == 'train':
            # Original image
            splits['train']['paths'].append(orig_full)
            splits['train']['meta'].append(meta_vec)
            splits['train']['labels'].append(class_idx)
            
            # Augmented images for train patient
            for aug_i in range(1, 8):
                aug_fname = f"{img_id}_{aug_i:02d}.jpg"
                aug_full = os.path.join(DATASET_DIR, "Augmented", folder, aug_fname)
                if os.path.exists(aug_full):
                    splits['train']['paths'].append(aug_full)
                    splits['train']['meta'].append(meta_vec)
                    splits['train']['labels'].append(class_idx)
        else:
            splits[split]['paths'].append(orig_full)
            splits[split]['meta'].append(meta_vec)
            splits[split]['labels'].append(class_idx)
            
    print(f"Dataset Summary:")
    print(f"  Train: {len(splits['train']['paths'])} samples")
    print(f"  Val:   {len(splits['val']['paths'])} samples")
    print(f"  Test:  {len(splits['test']['paths'])} samples")
    
    # Calculate smoothed class weights
    counts = pd.Series(splits['train']['labels']).value_counts().sort_index().values
    total = len(splits['train']['labels'])
    smoothed_weights = np.sqrt(total / (NUM_CLASSES * counts))
    smoothed_weights = smoothed_weights / np.mean(smoothed_weights)
    class_weights = {i: float(round(w, 4)) for i, w in enumerate(smoothed_weights)}
    print(f"Smoothed Class Weights: {class_weights}")
    
    # High-performance tf.data pipeline with One-Hot Labels (for Label Smoothing)
    def build_tf_dataset(paths, meta, labels, shuffle=False):
        meta_arr = np.array(meta, dtype=np.float32)
        labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)
        
        def load_img_and_pair(p, m, lbl_one_hot):
            raw = tf.io.read_file(p)
            img = tf.image.decode_jpeg(raw, channels=3)
            img = tf.image.resize(img, IMG_SIZE, method='bilinear')
            return {"input_image": img, "input_metadata": m}, lbl_one_hot
            
        ds = tf.data.Dataset.from_tensor_slices((paths, meta_arr, labels_one_hot))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
        ds = ds.map(load_img_and_pair, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = build_tf_dataset(splits['train']['paths'], splits['train']['meta'], splits['train']['labels'], shuffle=True)
    val_ds = build_tf_dataset(splits['val']['paths'], splits['val']['meta'], splits['val']['labels'], shuffle=False)
    test_ds = build_tf_dataset(splits['test']['paths'], splits['test']['meta'], splits['test']['labels'], shuffle=False)
    
    return train_ds, val_ds, test_ds, splits, class_weights, classes

def build_densenet_multimodal_model(meta_dim):
    """
    Builds DenseNet121 Multimodal Dual-Branch Network with:
    - Pretrained DenseNet121 feature extractor
    - Dense metadata branch (44-dim -> 48-dim)
    - Fusion layer + Dropout + Label-smoothed output
    """
    # 1. Vision Branch
    input_image = keras.Input(shape=(224, 224, 3), name="input_image")
    # Data augmentation on the fly
    x_img = layers.RandomFlip("horizontal")(input_image)
    x_img = layers.RandomRotation(0.05)(x_img)
    x_img = layers.RandomZoom(0.05)(x_img)
    # DenseNet requires preprocessing: scale to [0, 1] then normalize using ImageNet mean/std
    x_img = keras.applications.densenet.preprocess_input(x_img)
    
    base_model = keras.applications.DenseNet121(
        include_top=False,
        weights='imagenet',
        input_tensor=x_img
    )
    base_model.trainable = False
    
    img_feat = base_model.output
    img_feat = layers.GlobalAveragePooling2D(name="img_pool")(img_feat)
    img_feat = layers.BatchNormalization(name="img_bn")(img_feat)
    img_feat = layers.Dropout(0.35, name="img_dropout")(img_feat)
    img_feat = layers.Dense(256, activation="relu", name="img_dense")(img_feat)
    img_feat = layers.BatchNormalization(name="img_dense_bn")(img_feat)
    
    # 2. Tabular Metadata Branch
    input_metadata = keras.Input(shape=(meta_dim,), name="input_metadata")
    meta_feat = layers.Dense(64, activation="relu", name="meta_dense1")(input_metadata)
    meta_feat = layers.BatchNormalization(name="meta_bn1")(meta_feat)
    meta_feat = layers.Dropout(0.2, name="meta_dropout1")(meta_feat)
    meta_feat = layers.Dense(48, activation="relu", name="meta_dense2")(meta_feat)
    meta_feat = layers.BatchNormalization(name="meta_bn2")(meta_feat)
    
    # 3. Multimodal Fusion
    fused = layers.Concatenate(name="multimodal_concat")([img_feat, meta_feat]) # 256 + 48 = 304-dim
    fused = layers.Dense(128, activation="relu", name="fused_dense")(fused)
    fused = layers.Dropout(0.3, name="fused_dropout")(fused)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32", name="predictions")(fused)
    
    model = keras.Model(inputs=[input_image, input_metadata], outputs=outputs, name="SkinDisNet_DenseNet121_Multimodal")
    return model, base_model

def run_multimodal_tta(model, test_paths, test_meta):
    """Test-Time Augmentation (TTA) with 4 geometric transformations."""
    all_tta_probs = []
    for p, meta_vec in zip(test_paths, test_meta):
        with Image.open(p) as img:
            rgb = img.convert('RGB').resize(IMG_SIZE, Image.Resampling.BILINEAR)
            arr = np.array(rgb, dtype=np.float32)
            
        var1 = arr
        var2 = np.fliplr(arr)
        var3 = np.flipud(arr)
        var4 = np.rot90(arr, k=1, axes=(0, 1))
        img_batch = np.stack([var1, var2, var3, var4], axis=0)
        meta_batch = np.tile(meta_vec, (4, 1))
        
        preds = model.predict({"input_image": img_batch, "input_metadata": meta_batch}, verbose=0)
        avg_prob = np.mean(preds, axis=0)
        all_tta_probs.append(avg_prob)
        
    return np.array(all_tta_probs)

def train_fase3():
    print("=" * 72)
    print("FASE 3 & 4: DenseNet121 Multimodal + Label Smoothing (alpha=0.1)")
    print("=" * 72)
    
    encoder_info = load_encoder()
    train_ds, val_ds, test_ds, splits, class_weights, classes = load_dataset(encoder_info)
    
    meta_dim = len(splits['train']['meta'][0])
    model, base_model = build_densenet_multimodal_model(meta_dim)
    
    # -------------------------------------------------------------
    # Stage 1: Warmup DenseNet Head & Metadata Branch (lr=1e-3, 4 Epochs)
    # -------------------------------------------------------------
    print("\n" + "-" * 55)
    print("STAGE 1: Warmup Multimodal Head (Backbone Frozen, lr=1e-3, 4 Epochs)")
    print("-" * 55)
    
    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    stage1_callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)
    ]
    
    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=4,
        class_weight=class_weights,
        callbacks=stage1_callbacks,
        verbose=1
    )
    
    # -------------------------------------------------------------
    # Stage 2: Fine-Tuning Top Dense Blocks of DenseNet121 (lr=1e-5, 4 Epochs)
    # -------------------------------------------------------------
    print("\n" + "-" * 55)
    print("STAGE 2: End-to-End Fine-Tuning (Top 30 Layers, lr=1e-5, 4 Epochs)")
    print("-" * 55)
    
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    stage2_callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(DENSENET_MULTIMODAL_PATH, monitor='val_loss', save_best_only=True)
    ]
    
    hist2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=4,
        class_weight=class_weights,
        callbacks=stage2_callbacks,
        verbose=1
    )
    
    model.save(DENSENET_MULTIMODAL_PATH)
    print(f"\nSaved DenseNet121 multimodal model to: {DENSENET_MULTIMODAL_PATH}")
    
    # Save training curves plot
    try:
        acc = hist1.history['accuracy'] + hist2.history['accuracy']
        val_acc = hist1.history['val_accuracy'] + hist2.history['val_accuracy']
        loss = hist1.history['loss'] + hist2.history['loss']
        val_loss = hist1.history['val_loss'] + hist2.history['val_loss']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(acc, label='Train Accuracy', marker='o')
        axes[0].plot(val_acc, label='Val Accuracy', marker='s')
        axes[0].axvline(x=len(hist1.history['accuracy']) - 0.5, color='r', linestyle='--', label='Fine-Tuning Start')
        axes[0].set_title('DenseNet121 Multimodal Accuracy Curve', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(loss, label='Train Loss', marker='o')
        axes[1].plot(val_loss, label='Val Loss', marker='s')
        axes[1].axvline(x=len(hist1.history['loss']) - 0.5, color='r', linestyle='--', label='Fine-Tuning Start')
        axes[1].set_title('DenseNet121 Multimodal Loss Curve (Label Smoothing=0.1)', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        hist_path = os.path.join(ARTIFACTS_DIR, "densenet_training_history.png")
        fig.savefig(hist_path, dpi=300)
        plt.close()
        print(f"Saved training history curves to: {hist_path}")
    except Exception as e:
        print(f"Warning: Failed to save training curves: {e}")
    
    # -------------------------------------------------------------
    # Individual DenseNet121 Evaluation
    # -------------------------------------------------------------
    print("\n" + "=" * 72)
    print("EVALUASI TEST SET INDEPENDEN (261 Gambar Asli + Metadata Klinis Pasien)")
    print("=" * 72)
    
    test_paths = splits['test']['paths']
    test_meta = splits['test']['meta']
    test_labels = np.array(splits['test']['labels'])
    
    print("Running DenseNet121 TTA inference...")
    densenet_tta_probs = run_multimodal_tta(model, test_paths, test_meta)
    densenet_preds = np.argmax(densenet_tta_probs, axis=1)
    
    densenet_acc = accuracy_score(test_labels, densenet_preds)
    densenet_macro_f1 = f1_score(test_labels, densenet_preds, average='macro')
    densenet_weighted_f1 = f1_score(test_labels, densenet_preds, average='weighted')
    
    print(f"DenseNet121 Multimodal Test Accuracy (TTA): {densenet_acc*100:.2f}% | Macro F1: {densenet_macro_f1:.4f}")
    
    # -------------------------------------------------------------
    # FASE 5: ENSEMBLE STACKING (EfficientNetB0 + DenseNet121)
    # -------------------------------------------------------------
    print("\n" + "=" * 72)
    print("FASE 5: ENSEMBLE EVALUATION (EfficientNetB0 + DenseNet121 Multimodal Soft-Voting)")
    print("=" * 72)
    
    ensemble_active = False
    ensemble_acc = densenet_acc
    ensemble_macro_f1 = densenet_macro_f1
    ensemble_weighted_f1 = densenet_weighted_f1
    ensemble_probs = densenet_tta_probs
    
    if os.path.exists(EFFNET_MULTIMODAL_PATH):
        print(f"Loading trained EfficientNetB0 multimodal model from: {EFFNET_MULTIMODAL_PATH}...")
        effnet_model = tf.keras.models.load_model(EFFNET_MULTIMODAL_PATH)
        print("Running EfficientNetB0 TTA inference on test set...")
        effnet_tta_probs = run_multimodal_tta(effnet_model, test_paths, test_meta)
        
        # Soft voting: 50% DenseNet121 + 50% EfficientNetB0
        ensemble_probs = 0.5 * densenet_tta_probs + 0.5 * effnet_tta_probs
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        ensemble_acc = accuracy_score(test_labels, ensemble_preds)
        ensemble_macro_f1 = f1_score(test_labels, ensemble_preds, average='macro')
        ensemble_weighted_f1 = f1_score(test_labels, ensemble_preds, average='weighted')
        ensemble_active = True
        
        print("\n" + "-" * 50)
        print("HASIL EVALUASI ENSEMBLE TEST SET:")
        print(f"Ensemble Accuracy:     {ensemble_acc*100:.2f}%")
        print(f"Ensemble Macro F1:    {ensemble_macro_f1:.4f}")
        print(f"Ensemble Weighted F1: {ensemble_weighted_f1:.4f}")
        print("-" * 50)
    else:
        print("EfficientNetB0 multimodal model not found, skipping ensemble.")
        ensemble_preds = densenet_preds

    # Classification report
    rep_dict = classification_report(test_labels, ensemble_preds, target_names=classes, output_dict=True)
    rep_str = classification_report(test_labels, ensemble_preds, target_names=classes)
    print("\nEnsemble Multimodal Classification Report:")
    print(rep_str)
    
    rep_df = pd.DataFrame(rep_dict).transpose()
    rep_csv_path = os.path.join(ARTIFACTS_DIR, "ensemble_classification_report.csv")
    rep_df.to_csv(rep_csv_path)
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, ensemble_preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(f"Ensemble Multimodal Confusion Matrix (Accuracy: {ensemble_acc*100:.2f}%)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Predicted Diagnosis", fontsize=11)
    ax.set_ylabel("True Diagnosis", fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(ARTIFACTS_DIR, "ensemble_confusion_matrix.png")
    fig.savefig(cm_path, dpi=300)
    plt.close()
    
    # Save ensemble evaluation metrics
    metrics_data = {
        "model_name": "SkinDisNet Multimodal Dual-Backbone Ensemble (DenseNet121 + EfficientNetB0 + Label Smoothing)",
        "test_accuracy": float(ensemble_acc),
        "test_macro_f1": float(ensemble_macro_f1),
        "test_weighted_f1": float(ensemble_weighted_f1),
        "densenet_standalone_accuracy": float(densenet_acc),
        "densenet_standalone_macro_f1": float(densenet_macro_f1),
        "fase2_multimodal_effnet_accuracy": 0.4329501915708812,
        "fase1_image_only_accuracy": 0.40229885057471265,
        "baseline_custom_cnn_accuracy": 0.16091954022988506,
        "test_sample_count": len(test_labels),
        "classes": classes,
        "per_class_metrics": {
            cls: {
                "precision": float(rep_dict[cls]["precision"]),
                "recall": float(rep_dict[cls]["recall"]),
                "f1_score": float(rep_dict[cls]["f1-score"]),
                "support": int(rep_dict[cls]["support"])
            }
            for cls in classes
        },
        "confusion_matrix": cm.tolist()
    }
    
    metrics_path = os.path.join(ARTIFACTS_DIR, "ensemble_evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved ensemble metrics to: {metrics_path}")
    print(f"Saved ensemble confusion matrix to: {cm_path}")
    
    print("\n" + "=" * 72)
    print("RINGKASAN PROGRES PERJALANAN AKURASI SKINDISNET:")
    print(f"1. Baseline Custom CNN:            16.09%")
    print(f"2. Initial EfficientNetB0:         36.02%")
    print(f"3. Fase 1 (+Data Augmented):       40.23%")
    print(f"4. Fase 2 (+Multimodal Metadata):  43.30%")
    print(f"5. Fase 3 (+DenseNet121+LabelSm):  {densenet_acc*100:.2f}%")
    if ensemble_active:
        print(f"6. Fase 5 (Ensemble Stacking):     {ensemble_acc*100:.2f}%")
    print("=" * 72)

if __name__ == "__main__":
    train_fase3()
