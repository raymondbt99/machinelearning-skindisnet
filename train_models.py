import os
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import tensorflow as tf
from tensorflow import keras
from keras import layers

# Set random seeds for reproducibility
SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

WORKSPACE = r"c:\SkinDisNet A Multi-Class Clinical Images and Metad"
DATASET_DIR = os.path.join(WORKSPACE, "SkinDisNet_2")
ARTIFACTS_DIR = os.path.join(WORKSPACE, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

MANIFEST_PATH = os.path.join(ARTIFACTS_DIR, "dataset_split_manifest.csv")
CLASS_MAPPING_PATH = os.path.join(ARTIFACTS_DIR, "class_mapping.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 6

def load_data_from_manifest():
    print(f"Loading split manifest from {MANIFEST_PATH}...")
    manifest_df = pd.read_csv(MANIFEST_PATH)
    
    with open(CLASS_MAPPING_PATH, 'r') as f:
        mapping_data = json.load(f)
    classes = mapping_data['classes']
    
    splits = {'train': ([], []), 'val': ([], []), 'test': ([], [])}
    
    print("Loading and resizing images to 224x224...")
    start_t = time.time()
    for _, row in manifest_df.iterrows():
        split = row['Split']
        rel_path = row['Relative_path']
        full_path = os.path.join(DATASET_DIR, rel_path)
        class_idx = int(row['Class_index'])
        
        # Load and resize via PIL
        with Image.open(full_path) as img:
            img = img.convert('RGB').resize(IMG_SIZE, Image.Resampling.BILINEAR)
            img_arr = np.array(img, dtype=np.uint8)
            
        splits[split][0].append(img_arr)
        splits[split][1].append(class_idx)
        
    X_train = np.array(splits['train'][0], dtype=np.float32)
    y_train = np.array(splits['train'][1], dtype=np.int32)
    
    X_val = np.array(splits['val'][0], dtype=np.float32)
    y_val = np.array(splits['val'][1], dtype=np.int32)
    
    X_test = np.array(splits['test'][0], dtype=np.float32)
    y_test = np.array(splits['test'][1], dtype=np.int32)
    
    print(f"Loaded in {time.time() - start_t:.2f}s:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}")
    
    # Calculate class weights for training set
    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {int(cls): float(w) for cls, w in zip(np.unique(y_train), cw)}
    print(f"Computed Balanced Class Weights: {class_weights}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_weights, classes

def build_baseline_cnn():
    """Custom small CNN baseline model with data augmentation."""
    inputs = keras.Input(shape=(224, 224, 3), name="input_image")
    
    # Online augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    
    x = layers.Rescaling(1.0 / 255.0)(x)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="Baseline_CNN")
    return model

def build_efficientnetb0():
    """EfficientNetB0 transfer learning model with augmentation and custom head."""
    inputs = keras.Input(shape=(224, 224, 3), name="input_image")
    
    # Online data augmentation active during training
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomContrast(0.1)(x)
    
    # Pretrained backbone
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=x
    )
    base_model.trainable = False
    
    feat = base_model.output
    feat = layers.GlobalAveragePooling2D(name="avg_pool")(feat)
    feat = layers.BatchNormalization(name="head_bn")(feat)
    feat = layers.Dropout(0.4, name="head_dropout")(feat)
    feat = layers.Dense(256, activation="relu", name="dense_features")(feat)
    feat = layers.BatchNormalization(name="dense_bn")(feat)
    feat = layers.Dropout(0.3, name="sub_dropout")(feat)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32", name="predictions")(feat)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_SkinDisNet")
    return model, base_model

def train_and_evaluate():
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_weights, classes = load_data_from_manifest()
    
    def prepare_ds(X, y, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(X), seed=SEED)
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = prepare_ds(X_train, y_train, shuffle=True)
    val_ds = prepare_ds(X_val, y_val, shuffle=False)
    test_ds = prepare_ds(X_test, y_test, shuffle=False)
    
    training_history_record = {}
    
    # -------------------------------------------------------------
    # 1. Baseline Model Training
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 1: Training Baseline Custom CNN")
    print("="*60)
    baseline_model = build_baseline_cnn()
    baseline_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    baseline_callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)
    ]
    
    baseline_hist = baseline_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        class_weight=class_weights,
        callbacks=baseline_callbacks,
        verbose=1
    )
    
    # Baseline test evaluation
    baseline_preds = baseline_model.predict(test_ds)
    baseline_pred_cls = np.argmax(baseline_preds, axis=1)
    baseline_acc = float(accuracy_score(y_test, baseline_pred_cls))
    baseline_f1_macro = float(f1_score(y_test, baseline_pred_cls, average='macro'))
    print(f"\nBaseline CNN Test Accuracy: {baseline_acc:.4f}, Macro F1: {baseline_f1_macro:.4f}")
    
    # -------------------------------------------------------------
    # 2. EfficientNetB0 Training: Stage 1 Feature Extraction
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 2: EfficientNetB0 Stage 1 - Feature Extraction (Backbone Frozen)")
    print("="*60)
    eff_model, eff_base = build_efficientnetb0()
    eff_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    stage1_callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)
    ]
    
    hist_stage1 = eff_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        class_weight=class_weights,
        callbacks=stage1_callbacks,
        verbose=1
    )
    
    # -------------------------------------------------------------
    # 3. EfficientNetB0 Training: Stage 2 Fine-Tuning
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 3: EfficientNetB0 Stage 2 - Fine-Tuning (Unfreezing Top Backbone Layers)")
    print("="*60)
    
    # Unfreeze top layers of the backbone (e.g. last 30 layers) while keeping BatchNormalization frozen
    eff_base.trainable = True
    # Freeze all layers except the last 30 layers
    for layer in eff_base.layers[:-30]:
        layer.trainable = False
    for layer in eff_base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            
    print(f"Total layers in base: {len(eff_base.layers)}, Trainable layers in base: {sum(1 for l in eff_base.layers if l.trainable)}")
    
    checkpoint_path = os.path.join(ARTIFACTS_DIR, "efficientnetb0_skin_classifier.keras")
    
    eff_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    stage2_callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
    ]
    
    hist_stage2 = eff_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        class_weight=class_weights,
        callbacks=stage2_callbacks,
        verbose=1
    )
    
    # Make sure best model is saved
    eff_model.save(checkpoint_path)
    print(f"Saved primary model to {checkpoint_path}")
    
    # Combine training histories for plotting
    combined_loss = hist_stage1.history['loss'] + hist_stage2.history['loss']
    combined_val_loss = hist_stage1.history['val_loss'] + hist_stage2.history['val_loss']
    combined_acc = hist_stage1.history['accuracy'] + hist_stage2.history['accuracy']
    combined_val_acc = hist_stage1.history['val_accuracy'] + hist_stage2.history['val_accuracy']
    
    # -------------------------------------------------------------
    # 4. Evaluation on Untouched Test Set
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 4: Evaluating EfficientNetB0 on Independent Test Set")
    print("="*60)
    test_probs = eff_model.predict(test_ds)
    y_pred = np.argmax(test_probs, axis=1)
    
    test_acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average='macro'))
    weighted_f1 = float(f1_score(y_test, y_pred, average='weighted'))
    
    report_dict = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv_path = os.path.join(ARTIFACTS_DIR, "classification_report.csv")
    report_df.to_csv(report_csv_path)
    print(f"Saved classification report to {report_csv_path}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix - EfficientNetB0 on SkinDisNet Test Set", fontsize=14, pad=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")
    
    # Plot Training & Validation Curves
    history_plot_path = os.path.join(ARTIFACTS_DIR, "training_history.png")
    epochs_range = range(1, len(combined_loss) + 1)
    stage1_len = len(hist_stage1.history['loss'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs_range, combined_loss, 'o-', label='Training Loss', color='#2563eb')
    ax1.plot(epochs_range, combined_val_loss, 's--', label='Validation Loss', color='#dc2626')
    ax1.axvline(x=stage1_len + 0.5, color='gray', linestyle=':', label='Fine-Tuning Start')
    ax1.set_title('EfficientNetB0 Loss Across Stages', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Sparse Categorical Cross-Entropy')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Accuracy plot
    ax2.plot(epochs_range, combined_acc, 'o-', label='Training Accuracy', color='#2563eb')
    ax2.plot(epochs_range, combined_val_acc, 's--', label='Validation Accuracy', color='#16a34a')
    ax2.axvline(x=stage1_len + 0.5, color='gray', linestyle=':', label='Fine-Tuning Start')
    ax2.set_title('EfficientNetB0 Accuracy Across Stages', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(history_plot_path, dpi=300)
    plt.close()
    print(f"Saved training history curves to {history_plot_path}")
    
    # Save Evaluation Metrics JSON
    metrics_summary = {
        "model_name": "EfficientNetB0",
        "baseline_model_name": "Custom CNN Baseline",
        "test_accuracy": test_acc,
        "test_macro_f1": macro_f1,
        "test_weighted_f1": weighted_f1,
        "baseline_test_accuracy": baseline_acc,
        "baseline_test_macro_f1": baseline_f1_macro,
        "test_sample_count": int(len(y_test)),
        "classes": classes,
        "per_class_metrics": {
            cls: {
                "precision": report_dict[cls]["precision"],
                "recall": report_dict[cls]["recall"],
                "f1_score": report_dict[cls]["f1-score"],
                "support": int(report_dict[cls]["support"])
            } for cls in classes
        },
        "confusion_matrix": cm.tolist()
    }
    
    metrics_json_path = os.path.join(ARTIFACTS_DIR, "evaluation_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved evaluation metrics to {metrics_json_path}")
    
    # Save Training Config JSON
    training_config = {
        "architecture": "EfficientNetB0",
        "input_shape": [224, 224, 3],
        "pretrained_weights": "imagenet",
        "num_classes": NUM_CLASSES,
        "classes": classes,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "loss": "SparseCategoricalCrossentropy",
        "stage1": {
            "description": "Feature extraction with frozen backbone",
            "learning_rate": 1e-3,
            "epochs_trained": len(hist_stage1.history['loss'])
        },
        "stage2": {
            "description": "Fine-tuning top 30 layers with low learning rate",
            "learning_rate": 1e-5,
            "epochs_trained": len(hist_stage2.history['loss'])
        },
        "regularization": {
            "head_dropout": 0.4,
            "dense_dropout": 0.3,
            "dense_units": 256,
            "batch_normalization": True
        },
        "augmentation": ["RandomFlip(horizontal)", "RandomRotation(0.1)", "RandomZoom(0.1)", "RandomContrast(0.1)"],
        "class_weights": class_weights,
        "random_seed": SEED
    }
    
    config_json_path = os.path.join(ARTIFACTS_DIR, "training_config.json")
    with open(config_json_path, "w") as f:
        json.dump(training_config, f, indent=2)
    print(f"Saved training configuration to {config_json_path}")
    
    print("\nTraining and Evaluation Pipeline Complete!")

if __name__ == "__main__":
    train_and_evaluate()
