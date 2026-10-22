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
METADATA_CSV_PATH = os.path.join(DATASET_DIR, "SkinDisNet_Metadata.csv")
CLASS_MAPPING_PATH = os.path.join(ARTIFACTS_DIR, "class_mapping.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 64
NUM_CLASSES = 6

def build_metadata_encoder():
    df = pd.read_csv(METADATA_CSV_PATH)
    clean_locations = sorted(df['Leision_location'].astype(str).str.strip().str.title().unique())
    location_to_idx = {loc: i for i, loc in enumerate(clean_locations)}
    # Add Unknown / Other
    unknown_idx = len(clean_locations)
    location_to_idx["Unknown"] = unknown_idx
    location_to_idx["Other"] = unknown_idx
    
    encoder_info = {
        "locations": clean_locations,
        "location_to_idx": location_to_idx,
        "num_locations": len(clean_locations) + 1,
        "max_age": 80.0
    }
    
    with open(os.path.join(ARTIFACTS_DIR, "multimodal_metadata_encoders.json"), "w") as f:
        json.dump(encoder_info, f, indent=2)
    print(f"Metadata encoder saved with {encoder_info['num_locations']} location categories.")
    return encoder_info

def encode_metadata_vector(age_val, sex_val, loc_val, encoder_info):
    """Encodes tabular clinical features into a 44-dimensional float32 vector."""
    num_locs = encoder_info["num_locations"]
    vec = np.zeros(num_locs + 1 + 2, dtype=np.float32)
    
    # 1. Location one-hot (indices 0 to num_locs-1)
    loc_clean = str(loc_val).strip().title()
    loc_idx = encoder_info["location_to_idx"].get(loc_clean, encoder_info["location_to_idx"]["Unknown"])
    vec[loc_idx] = 1.0
    
    # 2. Normalized Age (index num_locs)
    try:
        age_float = float(age_val)
    except Exception:
        age_float = 30.0
    vec[num_locs] = np.clip(age_float / encoder_info["max_age"], 0.0, 1.2)
    
    # 3. Sex one-hot (indices num_locs+1, num_locs+2)
    sex_str = str(sex_val).strip().title()
    if sex_str == 'Female':
        vec[num_locs + 1] = 1.0
    elif sex_str == 'Male':
        vec[num_locs + 2] = 1.0
    else:
        vec[num_locs + 1] = 0.5
        vec[num_locs + 2] = 0.5
        
    return vec

def load_multimodal_data(encoder_info):
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
            # Add original image
            splits['train']['paths'].append(orig_full)
            splits['train']['meta'].append(meta_vec)
            splits['train']['labels'].append(class_idx)
            
            # Add all 7 augmented versions (they share the same patient clinical metadata)
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
            
    print(f"Multimodal Data Summary:")
    print(f"  Train: {len(splits['train']['paths'])} samples")
    print(f"  Val:   {len(splits['val']['paths'])} samples")
    print(f"  Test:  {len(splits['test']['paths'])} samples")
    print(f"  Metadata vector dimension: {len(splits['train']['meta'][0])}")
    
    # Calculate smoothed class weights
    counts = pd.Series(splits['train']['labels']).value_counts().sort_index().values
    total = len(splits['train']['labels'])
    smoothed_weights = np.sqrt(total / (NUM_CLASSES * counts))
    smoothed_weights = smoothed_weights / np.mean(smoothed_weights)
    class_weights = {i: float(round(w, 4)) for i, w in enumerate(smoothed_weights)}
    print(f"Smoothed Class Weights: {class_weights}")
    
    # High performance tf.data generator
    def build_tf_dataset(paths, meta, labels, shuffle=False):
        meta_arr = np.array(meta, dtype=np.float32)
        labels_arr = np.array(labels, dtype=np.int32)
        
        def load_img_and_pair(p, m, lbl):
            raw = tf.io.read_file(p)
            img = tf.image.decode_jpeg(raw, channels=3)
            img = tf.image.resize(img, IMG_SIZE, method='bilinear')
            return {"input_image": img, "input_metadata": m}, lbl
            
        ds = tf.data.Dataset.from_tensor_slices((paths, meta_arr, labels_arr))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
        ds = ds.map(load_img_and_pair, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = build_tf_dataset(splits['train']['paths'], splits['train']['meta'], splits['train']['labels'], shuffle=True)
    val_ds = build_tf_dataset(splits['val']['paths'], splits['val']['meta'], splits['val']['labels'], shuffle=False)
    test_ds = build_tf_dataset(splits['test']['paths'], splits['test']['meta'], splits['test']['labels'], shuffle=False)
    
    return train_ds, val_ds, test_ds, splits, class_weights, classes

def build_multimodal_model(meta_dim):
    """Builds dual-branch neural network combining EfficientNetB0 image features with Tabular clinical metadata MLP."""
    # 1. Image Branch
    input_image = keras.Input(shape=(224, 224, 3), name="input_image")
    aug = layers.RandomFlip("horizontal")(input_image)
    
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=aug
    )
    base_model.trainable = False
    
    img_feat = base_model.output
    img_feat = layers.GlobalAveragePooling2D(name="img_pool")(img_feat)
    img_feat = layers.BatchNormalization(name="img_bn")(img_feat)
    img_feat = layers.Dropout(0.35, name="img_dropout")(img_feat)
    img_feat = layers.Dense(256, activation="relu", name="img_dense")(img_feat)
    img_feat = layers.BatchNormalization(name="img_dense_bn")(img_feat)
    
    # 2. Metadata Branch (Location, Age, Sex)
    input_metadata = keras.Input(shape=(meta_dim,), name="input_metadata")
    meta_feat = layers.Dense(64, activation="relu", name="meta_dense1")(input_metadata)
    meta_feat = layers.BatchNormalization(name="meta_bn1")(meta_feat)
    meta_feat = layers.Dropout(0.2, name="meta_dropout1")(meta_feat)
    meta_feat = layers.Dense(32, activation="relu", name="meta_dense2")(meta_feat)
    meta_feat = layers.BatchNormalization(name="meta_bn2")(meta_feat)
    
    # 3. Multimodal Fusion
    fused = layers.Concatenate(name="multimodal_concat")([img_feat, meta_feat]) # 256 + 32 = 288-dim
    fused = layers.Dense(128, activation="relu", name="fused_dense")(fused)
    fused = layers.Dropout(0.3, name="fused_dropout")(fused)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32", name="predictions")(fused)
    
    model = keras.Model(inputs=[input_image, input_metadata], outputs=outputs, name="SkinDisNet_Multimodal")
    return model, base_model

def run_multimodal_tta(model, test_paths, test_meta):
    """Test-Time Augmentation (TTA) on multimodal inputs."""
    print("Running Multimodal Test-Time Augmentation (TTA) on test images...")
    all_tta_probs = []
    
    for p, meta_vec in zip(test_paths, test_meta):
        with Image.open(p) as img:
            rgb = img.convert('RGB').resize(IMG_SIZE, Image.Resampling.BILINEAR)
            arr = np.array(rgb, dtype=np.float32)
            
        # 4 image variations
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

def train_fase2():
    print("=" * 68)
    print("FASE 2: Training Multimodal Dual-Branch Network (Image + Clinical Metadata)")
    print("=" * 68)
    
    encoder_info = build_metadata_encoder()
    train_ds, val_ds, test_ds, splits, class_weights, classes = load_multimodal_data(encoder_info)
    
    meta_dim = len(splits['train']['meta'][0])
    model, base_model = build_multimodal_model(meta_dim)
    
    checkpoint_path = os.path.join(ARTIFACTS_DIR, "multimodal_skin_classifier.keras")
    
    # -------------------------------------------------------------
    # Stage 1: Feature Extraction (Backbone Frozen)
    # -------------------------------------------------------------
    print("\n" + "-"*50)
    print("STAGE 1: Training Multimodal Head & Metadata Branch (lr=1e-3, 5 Epochs)")
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
    # Stage 2: Fine-Tuning (Unfreezing Upper Backbone Layers)
    # -------------------------------------------------------------
    print("\n" + "-"*50)
    print("STAGE 2: End-to-End Multimodal Fine-Tuning (Top 35 Layers, lr=1e-5, 4 Epochs)")
    print("-" * 50)
    
    base_model.trainable = True
    for layer in base_model.layers[:-35]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            
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
    
    model.save(checkpoint_path)
    print(f"Saved multimodal model to {checkpoint_path}")
    
    # -------------------------------------------------------------
    # Evaluation on Independent Test Set
    # -------------------------------------------------------------
    print("\n" + "="*68)
    print("EVALUASI TEST SET INDEPENDEN (261 Gambar Asli + Metadata Klinis Pasien)")
    print("="*68)
    
    test_paths = splits['test']['paths']
    test_meta = splits['test']['meta']
    test_labels = splits['test']['labels']
    
    # 1. Standard Prediction
    std_probs = model.predict(test_ds)
    y_pred_std = np.argmax(std_probs, axis=1)
    acc_std = accuracy_score(test_labels, y_pred_std)
    macro_f1_std = f1_score(test_labels, y_pred_std, average='macro')
    
    # 2. TTA Prediction
    tta_probs = run_multimodal_tta(model, test_paths, test_meta)
    y_pred_tta = np.argmax(tta_probs, axis=1)
    acc_tta = accuracy_score(test_labels, y_pred_tta)
    macro_f1_tta = f1_score(test_labels, y_pred_tta, average='macro')
    weighted_f1_tta = f1_score(test_labels, y_pred_tta, average='weighted')
    
    print("\n" + "-"*40)
    print("HASIL EVALUASI MULTIMODAL TEST SET:")
    print(f"Standard Accuracy:  {acc_std*100:.2f}% | Macro F1: {macro_f1_std:.4f}")
    print(f"TTA Accuracy:       {acc_tta*100:.2f}% | Macro F1: {macro_f1_tta:.4f}")
    print(f"Weighted F1-Score:  {weighted_f1_tta:.4f}")
    print("-" * 40)
    
    final_pred = y_pred_tta
    final_acc = float(acc_tta)
    final_macro_f1 = float(macro_f1_tta)
    final_weighted_f1 = float(weighted_f1_tta)
    
    # Classification Report
    rep_dict = classification_report(test_labels, final_pred, target_names=classes, output_dict=True)
    rep_df = pd.DataFrame(rep_dict).transpose()
    rep_csv_path = os.path.join(ARTIFACTS_DIR, "multimodal_classification_report.csv")
    rep_df.to_csv(rep_csv_path)
    print("\nMultimodal Classification Report (with TTA):")
    print(classification_report(test_labels, final_pred, target_names=classes))
    
    # Confusion Matrix Plot
    cm = confusion_matrix(test_labels, final_pred)
    cm_path = os.path.join(ARTIFACTS_DIR, "multimodal_confusion_matrix.png")
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix Multimodal (Image + Location/Age/Sex) - Akurasi: {final_acc*100:.2f}%", fontsize=12, pad=15)
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
    
    hist_path = os.path.join(ARTIFACTS_DIR, "multimodal_training_history.png")
    epochs = range(1, len(loss_all) + 1)
    split_epoch = len(hist1.history['loss'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, loss_all, 'o-', label='Train Loss', color='#2563eb')
    ax1.plot(epochs, val_loss_all, 's--', label='Val Loss', color='#dc2626')
    ax1.axvline(x=split_epoch + 0.5, color='gray', linestyle=':', label='Fine-Tuning Start')
    ax1.set_title('Multimodal Loss Curve', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2.plot(epochs, acc_all, 'o-', label='Train Acc', color='#2563eb')
    ax2.plot(epochs, val_acc_all, 's--', label='Val Acc', color='#16a34a')
    ax2.axvline(x=split_epoch + 0.5, color='gray', linestyle=':', label='Fine-Tuning Start')
    ax2.set_title('Multimodal Accuracy Curve', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(hist_path, dpi=300)
    plt.close()
    
    # Save Multimodal Metrics JSON
    metrics_summary = {
        "model_name": "SkinDisNet Multimodal Dual-Branch Network (EfficientNetB0 + Clinical Metadata)",
        "test_accuracy": final_acc,
        "test_macro_f1": final_macro_f1,
        "test_weighted_f1": final_weighted_f1,
        "standard_accuracy_without_tta": float(acc_std),
        "fase1_image_only_accuracy": 0.40229885057471265,
        "baseline_custom_cnn_accuracy": 0.16091954022988506,
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
    
    with open(os.path.join(ARTIFACTS_DIR, "multimodal_evaluation_metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)
        
    print("\nFase 2 Multimodal Selesai & Seluruh Artefak Disimpan!")

if __name__ == "__main__":
    train_fase2()
