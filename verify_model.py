import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
import pandas as pd
import tensorflow as tf

WORKSPACE = r"c:\SkinDisNet A Multi-Class Clinical Images and Metad"
ARTIFACTS_DIR = os.path.join(WORKSPACE, "artifacts")
IMAGE_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "efficientnetb0_skin_classifier.keras")
EFFNET_MULTIMODAL_PATH = os.path.join(ARTIFACTS_DIR, "multimodal_skin_classifier.keras")
DENSENET_MULTIMODAL_PATH = os.path.join(ARTIFACTS_DIR, "densenet_multimodal_skin_classifier.keras")
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "multimodal_metadata_encoders.json")
CLASS_MAPPING_PATH = os.path.join(ARTIFACTS_DIR, "class_mapping.json")
MANIFEST_PATH = os.path.join(ARTIFACTS_DIR, "dataset_split_manifest.csv")

def encode_metadata(age_val, sex_val, loc_val, encoder):
    num_locs = encoder["num_locations"]
    vec = np.zeros(num_locs + 1 + 2, dtype=np.float32)
    loc_clean = str(loc_val).strip().title()
    loc_idx = encoder["location_to_idx"].get(loc_clean, encoder["location_to_idx"].get("Unknown", num_locs - 1))
    vec[loc_idx] = 1.0
    try:
        age_float = float(age_val)
    except Exception:
        age_float = 30.0
    vec[num_locs] = np.clip(age_float / encoder["max_age"], 0.0, 1.2)
    sex_str = str(sex_val).strip().title()
    if sex_str == 'Female':
        vec[num_locs + 1] = 1.0
    elif sex_str == 'Male':
        vec[num_locs + 2] = 1.0
    else:
        vec[num_locs + 1] = 0.5
        vec[num_locs + 2] = 0.5
    return vec

def verify(image_path=None, model_choice='auto', location=None, age=None, sex=None):
    print("=" * 60)
    print("SKINDISNET MODEL VERIFICATION UTILITY")
    print("=" * 60)

    # Resolve model mode
    if model_choice == 'auto':
        if os.path.exists(DENSENET_MULTIMODAL_PATH) and os.path.exists(EFFNET_MULTIMODAL_PATH):
            mode = 'ensemble'
        elif os.path.exists(DENSENET_MULTIMODAL_PATH):
            mode = 'densenet'
        elif os.path.exists(EFFNET_MULTIMODAL_PATH):
            mode = 'effnet'
        else:
            mode = 'image-only'
    else:
        mode = model_choice

    print(f"Active Verification Mode: {mode.upper()}")

    # 1. Verify class mapping
    if not os.path.exists(CLASS_MAPPING_PATH):
        print(f"Error: Class mapping not found at {CLASS_MAPPING_PATH}")
        sys.exit(1)
    with open(CLASS_MAPPING_PATH, 'r') as f:
        mapping = json.load(f)
    classes = mapping['classes']
    print(f"Loaded {len(classes)} target classes: {classes}")

    # 2. Select sample image if not provided
    sample_loc = location
    sample_age = age
    sample_sex = sex
    if not image_path:
        print("No image provided. Selecting sample from test split manifest...")
        manifest = pd.read_csv(MANIFEST_PATH)
        test_samples = manifest[manifest['Split'] == 'test']
        sample_row = test_samples.iloc[0]
        image_path = os.path.join(WORKSPACE, "SkinDisNet_2", sample_row['Relative_path'])
        expected_label = sample_row['Diagnosis']
        sample_loc = sample_row['Leision_location'] if not sample_loc else sample_loc
        sample_age = sample_row['Age'] if not sample_age else sample_age
        sample_sex = sample_row['Sex'] if not sample_sex else sample_sex
        print(f"Selected test sample: {image_path}")
        print(f"Ground truth label: {expected_label} (Patient: {sample_row['Patient_id']})")
    else:
        expected_label = "Unknown (User Provided)"
        print(f"Testing provided image: {image_path}")

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    print(f"Clinical Metadata: Location='{sample_loc or 'Unknown'}', Age={sample_age or '30'}, Sex='{sample_sex or 'Unknown'}'")

    # 3. Load Model(s)
    print("\nLoading trained Keras model(s)...")
    if mode == 'ensemble':
        effnet_m = tf.keras.models.load_model(EFFNET_MULTIMODAL_PATH)
        densenet_m = tf.keras.models.load_model(DENSENET_MULTIMODAL_PATH)
        print("Loaded both EfficientNetB0 and DenseNet121 for ensemble.")
    elif mode == 'densenet':
        densenet_m = tf.keras.models.load_model(DENSENET_MULTIMODAL_PATH)
        print("Loaded DenseNet121 Multimodal model.")
    elif mode == 'effnet':
        effnet_m = tf.keras.models.load_model(EFFNET_MULTIMODAL_PATH)
        print("Loaded EfficientNetB0 Multimodal model.")
    else:
        single_m = tf.keras.models.load_model(IMAGE_MODEL_PATH)
        print("Loaded Image-Only EfficientNetB0 model.")

    # 4. Preprocess Image
    print("\nPreprocessing input image...")
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        img_resized = img_rgb.resize((224, 224), Image.Resampling.BILINEAR)
        img_array = np.array(img_resized, dtype=np.float32)
        # Test-Time Augmentation (TTA)
        var1 = img_array
        var2 = np.fliplr(img_array)
        var3 = np.flipud(img_array)
        var4 = np.rot90(img_array, k=1, axes=(0, 1))
        batch_img = np.stack([var1, var2, var3, var4], axis=0)

    # 5. Execute Prediction with TTA
    print("Running inference with Test-Time Augmentation (TTA)...")
    if mode in ['ensemble', 'densenet', 'effnet']:
        with open(ENCODER_PATH, 'r') as f:
            encoder = json.load(f)
        meta_vec = encode_metadata(sample_age or 30, sample_sex or 'Unknown', sample_loc or 'Unknown', encoder)
        batch_meta = np.tile(meta_vec, (4, 1))
        inp = {"input_image": batch_img, "input_metadata": batch_meta}
        
        if mode == 'ensemble':
            p1 = effnet_m.predict(inp, verbose=0)
            p2 = densenet_m.predict(inp, verbose=0)
            batch_preds = 0.5 * p1 + 0.5 * p2
        elif mode == 'densenet':
            batch_preds = densenet_m.predict(inp, verbose=0)
        else:
            batch_preds = effnet_m.predict(inp, verbose=0)
    else:
        batch_preds = single_m.predict(batch_img, verbose=0)

    probs = np.mean(batch_preds, axis=0)
    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]
    pred_conf = float(probs[pred_idx])

    # 6. Verification Assertions
    assert len(probs) == len(classes), f"Output dimension {len(probs)} does not match {len(classes)} classes"
    assert np.isclose(np.sum(probs), 1.0, atol=1e-3), f"Probabilities do not sum to 1.0: {np.sum(probs)}"
    assert 0.0 <= pred_conf <= 1.0, f"Confidence {pred_conf} out of bounds"

    print("\n" + "-" * 40)
    print("INFERENCE RESULTS")
    print("-" * 40)
    print(f"Predicted Disease:  {pred_class}")
    print(f"Confidence Score:   {pred_conf * 100:.2f}%")
    print(f"Ground Truth:       {expected_label}")
    print("\nClass-by-Class Probability Breakdown:")
    for i, cls in enumerate(classes):
        bar = "#" * int(probs[i] * 30)
        print(f"  [{i}] {cls:<25} : {probs[i]*100:6.2f}% {bar}")
    print("-" * 40)
    print("Verification Passed: Model successfully loaded, executed inference, and returned valid calibrated probabilities.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify saved SkinDisNet model on sample image")
    parser.add_argument("--image", type=str, default=None, help="Path to image file to test")
    parser.add_argument("--model", type=str, choices=['auto', 'ensemble', 'densenet', 'effnet', 'image-only'], default='auto', help="Model architecture")
    parser.add_argument("--location", type=str, default=None, help="Lesion location")
    parser.add_argument("--age", type=float, default=None, help="Patient age")
    parser.add_argument("--sex", type=str, default=None, help="Patient sex (Male/Female)")
    args = parser.parse_args()
    verify(args.image, model_choice=args.model, location=args.location, age=args.age, sex=args.sex)
