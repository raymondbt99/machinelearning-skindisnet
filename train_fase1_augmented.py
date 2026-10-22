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
CLASS_MAPPING_PATH = os.path.join(ARTIFACTS_DIR, "class_mapping.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 64
NUM_CLASSES = 6

def build_datasets_with_offline_augmented():
    print(f"Loading patient-split manifest from {MANIFEST_PATH}...")
    manifest_df = pd.read_csv(MANIFEST_PATH)
    
    with open(CLASS_MAPPING_PATH, 'r') as f:
        mapping_data = json.load(f)
    classes = mapping_data['classes']
    
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []
    
    print("Constructing file lists (integrating Augmented/ for Train patients only)...")
    for _, row in manifest_df.iterrows():
        split = row['Split']
        folder = row['Folder_name']
        img_id = row['Image_id']
        class_idx = int(row['Class_index'])
        
        # Original preprocessed image
        orig_rel = os.path.join("Preprocessed", folder, f"{img_id}.jpg")
        orig_full = os.path.join(DATASET_DIR, orig_rel)
        assert os.path.exists(orig_full), f"File not found: {orig_full}"
        
        if split == 'train':
            train_paths.append(orig_full)
            train_labels.append(class_idx)
            
            # Add all 7 augmented versions for this training image
            for aug_i in range(1, 8):
                aug_fname = f"{img_id}_{aug_i:02d}.jpg"
                aug_full = os.path.join(DATASET_DIR, "Augmented", folder, aug_fname)
                if os.path.exists(aug_full):
                    train_paths.append(aug_full)
                    train_labels.append(class_idx)
        elif split == 'val':
            val_paths.append(orig_full)
            val_labels.append(class_idx)
        elif split == 'test':
            test_paths.append(orig_full)
            test_labels.append(class_idx)
            
    print(f"Dataset Size Summary:")
    print(f"  Train: {len(train_paths)} images (Original 1,202 + Augmented 8,414)")
    print(f"  Val:   {len(val_paths)} images (Untouched Original Preprocessed)")
    print(f"  Test:  {len(test_paths)} images (Untouched Original Preprocessed)")
    
    # Calculate smoothed / square-root class weights to prevent overshooting
    counts = pd.Series(train_labels).value_counts().sort_index().values
    total = len(train_labels)
    # Smoothed weight: sqrt(total / (num_classes * count))
    smoothed_weights = np.sqrt(total / (NUM_CLASSES * counts))
    # Normalize so mean weight is 1.0
    smoothed_weights = smoothed_weights / np.mean(smoothed_weights)
    class_weights = {i: float(round(w, 4)) for i, w in enumerate(smoothed_weights)}
    print(f"Smoothed Class Weights (Dampened to avoid false positives): {class_weights}")
    
    # High-performance tf.data image decoding pipeline
    def load_and_preprocess(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, IMG_SIZE, method='bilinear')
        return img, label

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.shuffle(buffer_size=len(train_paths), seed=SEED)
    train_ds = train_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    test_ds = test_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds, test_paths, test_labels, class_weights, classes

def build_model():
    inputs = keras.Input(shape=(224, 224, 3), name="input_image")
    
    # Online light random horizontal flip during training
    x = layers.RandomFlip("horizontal")(inputs)
    
    # Backbone
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=x
    )
    base_model.trainable = False
    
    feat = base_model.output
    feat = layers.GlobalAveragePooling2D(name="avg_pool")(feat)
    feat = layers.BatchNormalization(name="head_bn")(feat)
    feat = layers.Dropout(0.35, name="head_dropout")(feat)
    feat = layers.Dense(256, activation="relu", name="dense_features")(feat)
    feat = layers.BatchNormalization(name="dense_bn")(feat)
    feat = layers.Dropout(0.25, name="sub_dropout")(feat)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32", name="predictions")(feat)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_Augmented")
    return model, base_model

def run_tta_prediction(model, image_paths):
    """Test-Time Augmentation (TTA) combining 4 variations per image."""
    print("Running Test-Time Augmentation (TTA) on test images...")
    all_tta_probs = []
    
    for p in image_paths:
        with Image.open(p) as img:
            rgb = img.convert('RGB').resize(IMG_SIZE, Image.Resampling.BILINEAR)
            arr = np.array(rgb, dtype=np.float32)
            
        # 4 variations: original, h-flip, v-flip, 90-deg rotation
        var1 = arr
        var2 = np.fliplr(arr)
        var3 = np.flipud(arr)
        var4 = np.rot90(arr, k=1, axes=(0, 1))
        
        batch = np.stack([var1, var2, var3, var4], axis=0)
        preds = model.predict(batch, verbose=0)
        avg_prob = np.mean(preds, axis=0)
        all_tta_probs.append(avg_prob)
        
    return np.array(all_tta_probs)

def train_phase1():
    print("=" * 65)
    print("FASE 1: Training EfficientNetB0 dengan 9.616 Gambar Augmented")
    print("=" * 65)
    
    train_ds, val_ds, test_ds, test_paths, test_labels, class_weights, classes = build_datasets_with_offline_augmented()
    model, base_model = build_model()
    
    # -------------------------------------------------------------
    # Stage 1: Feature Extraction (Backbone Frozen)
    # -------------------------------------------------------------
    print("\n" + "-"*50)
    print("STAGE 1: Feature Extraction (5 Epochs, Frozen Backbone, lr=1e-3)")
    print("-" * 50)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    stage1_callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)
    ]
    
    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        class_weight=class_weights,
        callbacks=stage1_callbacks,
        verbose=1
    )
    
    # -------------------------------------------------------------
    # Stage 2: Fine-Tuning (Unfreezing Upper Layers)
    # -------------------------------------------------------------
    print("\n" + "-"*50)
    print("STAGE 2: Fine-Tuning (Top 35 Backbone Layers, lr=1e-5)")
    print("-" * 50)
    
    base_model.trainable = True
    for layer in base_model.layers[:-35]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            
    checkpoint_path = os.path.join(ARTIFACTS_DIR, "efficientnetb0_skin_classifier.keras")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    stage2_callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
    ]
    
    hist2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=4,
        class_weight=class_weights,
        callbacks=stage2_callbacks,
        verbose=1
    )
    
    # Save best model
    model.save(checkpoint_path)
    print(f"Saved optimized model to {checkpoint_path}")
    
    # -------------------------------------------------------------
    # Evaluation: Standard vs. TTA on Independent Test Set
    # -------------------------------------------------------------
    print("\n" + "="*65)
    print("EVALUASI TEST SET INDEPENDEN (261 Gambar Asli)")
    print("="*65)
    
    # Standard Inference
    std_probs = model.predict(test_ds)
    y_pred_std = np.argmax(std_probs, axis=1)
    acc_std = accuracy_score(test_labels, y_pred_std)
    macro_f1_std = f1_score(test_labels, y_pred_std, average='macro')
    
    # TTA Inference
    tta_probs = run_tta_prediction(model, test_paths)
    y_pred_tta = np.argmax(tta_probs, axis=1)
    acc_tta = accuracy_score(test_labels, y_pred_tta)
    macro_f1_tta = f1_score(test_labels, y_pred_tta, average='macro')
    weighted_f1_tta = f1_score(test_labels, y_pred_tta, average='weighted')
    
    print("\n" + "-"*40)
    print("PERBANDINGAN METRIK INFERENSI:")
    print(f"Standard Test Accuracy: {acc_std*100:.2f}% | Macro F1: {macro_f1_std:.4f}")
    print(f"TTA Test Accuracy:      {acc_tta*100:.2f}% | Macro F1: {macro_f1_tta:.4f}")
    print("-" * 40)
    
    # Use TTA results for final evaluation reports
    final_pred = y_pred_tta
    final_acc = float(acc_tta)
    final_macro_f1 = float(macro_f1_tta)
    final_weighted_f1 = float(weighted_f1_tta)
    
    rep_dict = classification_report(test_labels, final_pred, target_names=classes, output_dict=True)
    rep_df = pd.DataFrame(rep_dict).transpose()
    rep_csv_path = os.path.join(ARTIFACTS_DIR, "classification_report.csv")
    rep_df.to_csv(rep_csv_path)
    
    print("\nFinal Classification Report (with TTA):")
    print(classification_report(test_labels, final_pred, target_names=classes))
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, final_pred)
    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix (Fase 1 Augmented + TTA) - Akurasi: {final_acc*100:.2f}%", fontsize=13, pad=15)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    # Training History Plot
    loss_all = hist1.history['loss'] + hist2.history['loss']
    val_loss_all = hist1.history['val_loss'] + hist2.history['val_loss']
    acc_all = hist1.history['accuracy'] + hist2.history['accuracy']
    val_acc_all = hist1.history['val_accuracy'] + hist2.history['val_accuracy']
    
    hist_path = os.path.join(ARTIFACTS_DIR, "training_history.png")
    epochs = range(1, len(loss_all) + 1)
    split_epoch = len(hist1.history['loss'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, loss_all, 'o-', label='Train Loss', color='#2563eb')
    ax1.plot(epochs, val_loss_all, 's--', label='Val Loss', color='#dc2626')
    ax1.axvline(x=split_epoch + 0.5, color='gray', linestyle=':', label='Fine-Tuning Start')
    ax1.set_title('Loss Curve (Fase 1 9.6k Augmented)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2.plot(epochs, acc_all, 'o-', label='Train Acc', color='#2563eb')
    ax2.plot(epochs, val_acc_all, 's--', label='Val Acc', color='#16a34a')
    ax2.axvline(x=split_epoch + 0.5, color='gray', linestyle=':', label='Fine-Tuning Start')
    ax2.set_title('Accuracy Curve (Fase 1 9.6k Augmented)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(hist_path, dpi=300)
    plt.close()
    
    # Update evaluation_metrics.json
    metrics_summary = {
        "model_name": "EfficientNetB0 (Fase 1: 9.6k Offline Augmented + TTA)",
        "baseline_model_name": "Custom CNN Baseline",
        "test_accuracy": final_acc,
        "test_macro_f1": final_macro_f1,
        "test_weighted_f1": final_weighted_f1,
        "standard_accuracy_without_tta": float(acc_std),
        "baseline_test_accuracy": 0.16091954022988506,
        "baseline_test_macro_f1": 0.08136692176870748,
        "test_sample_count": int(len(test_labels)),
        "classes": classes,
        "per_class_metrics": {
            cls: {
                "precision": rep_dict[cls]["precision"],
                "recall": rep_dict[cls]["recall"],
                "f1_score": rep_dict[cls]["f1-score"],
                "support": int(rep_dict[cls]["support"])
            } for cls in classes
        },
        "confusion_matrix": cm.tolist()
    }
    
    with open(os.path.join(ARTIFACTS_DIR, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)
        
    print("\nFase 1 Pelatihan Selesai & Seluruh Artefak Berhasil Diperbarui!")

if __name__ == "__main__":
    train_phase1()
