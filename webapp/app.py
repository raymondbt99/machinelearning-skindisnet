import os
import io
import json
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
import pandas as pd

app = Flask(__name__, template_folder='templates', static_folder='static')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
DATASET_DIR = os.path.join(BASE_DIR, "SkinDisNet_2")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "efficientnetb0_skin_classifier.keras")
MULTIMODAL_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "multimodal_skin_classifier.keras")
DENSENET_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "densenet_multimodal_skin_classifier.keras")
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "multimodal_metadata_encoders.json")
CLASS_MAPPING_PATH = os.path.join(ARTIFACTS_DIR, "class_mapping.json")
MANIFEST_PATH = os.path.join(ARTIFACTS_DIR, "dataset_split_manifest.csv")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "evaluation_metrics.json")
MULTIMODAL_METRICS_PATH = os.path.join(ARTIFACTS_DIR, "multimodal_evaluation_metrics.json")
ENSEMBLE_METRICS_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_evaluation_metrics.json")

# Global model cache
MODEL = None
DENSENET_MODEL = None
IS_MULTIMODAL = False
IS_ENSEMBLE = False
METADATA_ENCODER = None
CLASS_NAMES = [
    "Atopic Dermatitis",
    "Contact Dermatitis",
    "Eczema",
    "Scabies",
    "Seborrheic Dermatitis",
    "Tinea Corporis"
]

CLINICAL_KNOWLEDGE = {
    "Atopic Dermatitis": {
        "short_name": "AD",
        "description": "A chronic, relapsing inflammatory skin disease characterized by pruritus, xerosis, and eczematous lesions.",
        "common_sites": "Flexural folds (antecubital and popliteal fossae), neck, wrists, face, and hands.",
        "visual_hallmarks": "Erythematous scaly patches, excoriations from scratching, skin lichenification, and dry cracked areas.",
        "differential_diagnoses": "Contact dermatitis, seborrheic dermatitis, psoriasis, scabies.",
        "clinical_category": "Allergic / Atopic Dermatoses"
    },
    "Contact Dermatitis": {
        "short_name": "CD",
        "description": "Localized inflammatory skin reaction triggered by direct contact with chemical irritants or allergen sensitization.",
        "common_sites": "Hands, wrists, forearms, face, neck, or areas exposed to jewellery, cosmetics, occupational chemicals, or poison ivy.",
        "visual_hallmarks": "Well-demarcated geometric erythema, papulovesicles, oozing or crusting matching the shape of the contact agent.",
        "differential_diagnoses": "Atopic dermatitis, cellulitis, dishydrotic eczema.",
        "clinical_category": "Inflammatory Dermatoses"
    },
    "Eczema": {
        "short_name": "EC",
        "description": "General term for a group of non-contagious skin inflammation conditions causing dry, sensitive skin with severe pruritus.",
        "common_sites": "Extremities, hands, feet, ankles, and flexor creases.",
        "visual_hallmarks": "Vesicles, erythematous papules, scaling, weeping lesions, and hyperpigmented thickened skin in chronic stages.",
        "differential_diagnoses": "Atopic dermatitis, fungal infections, lichen simplex chronicus.",
        "clinical_category": "Eczematous Dermatoses"
    },
    "Scabies": {
        "short_name": "SC",
        "description": "A contagious parasitic skin infestation caused by the microscopic burrowing mite Sarcoptes scabiei var. hominis.",
        "common_sites": "Interdigital web spaces of fingers, wrists, elbows, axillae, waistline, buttocks, and genital regions.",
        "visual_hallmarks": "Serpiginous or linear burrow tracks, pinpoint erythematous papules, excoriated nodules, and nocturnal pruritus.",
        "differential_diagnoses": "Atopic dermatitis, insect bite hypersensitivity, folliculitis.",
        "clinical_category": "Parasitic / Infectious Infestations"
    },
    "Seborrheic Dermatitis": {
        "short_name": "SD",
        "description": "A common chronic-recurrent inflammatory dermatosis associated with Malassezia yeast and sebaceous gland density.",
        "common_sites": "Scalp ('cradle cap' or dandruff), nasolabial folds, glabella, retroauricular regions, upper chest, and back.",
        "visual_hallmarks": "Greasy, yellowish scales overlying poorly defined erythematous macules or plaques.",
        "differential_diagnoses": "Psoriasis, atopic dermatitis, rosacea, lupus erythematosus.",
        "clinical_category": "Papulosquamous Dermatoses"
    },
    "Tinea Corporis": {
        "short_name": "TC",
        "description": "A superficial fungal infection (dermatophytosis) affecting keratinized layers of glabrous skin (body, trunk, limbs).",
        "common_sites": "Trunk, legs, arms, and neck.",
        "visual_hallmarks": "Annular, sharply demarcated erythematous plaques with raised advancing scaly/vesicular borders and central clearing ('ringworm').",
        "differential_diagnoses": "Pityriasis rosea, annular psoriasis, granuloma annulare, eczema.",
        "clinical_category": "Superficial Fungal Infections"
    }
}

def load_encoder():
    global METADATA_ENCODER
    if os.path.exists(ENCODER_PATH):
        with open(ENCODER_PATH, 'r') as f:
            METADATA_ENCODER = json.load(f)
    return METADATA_ENCODER

def encode_metadata_vec(age_val, sex_val, loc_val):
    enc = load_encoder()
    if enc is None:
        return np.zeros(44, dtype=np.float32)
    num_locs = enc["num_locations"]
    vec = np.zeros(num_locs + 1 + 2, dtype=np.float32)
    loc_clean = str(loc_val).strip().title()
    loc_idx = enc["location_to_idx"].get(loc_clean, enc["location_to_idx"].get("Unknown", num_locs - 1))
    vec[loc_idx] = 1.0
    try:
        age_float = float(age_val)
    except Exception:
        age_float = 30.0
    vec[num_locs] = np.clip(age_float / enc["max_age"], 0.0, 1.2)
    sex_str = str(sex_val).strip().title()
    if sex_str == 'Female':
        vec[num_locs + 1] = 1.0
    elif sex_str == 'Male':
        vec[num_locs + 2] = 1.0
    else:
        vec[num_locs + 1] = 0.5
        vec[num_locs + 2] = 0.5
    return vec

def get_model():
    global MODEL, DENSENET_MODEL, IS_MULTIMODAL, IS_ENSEMBLE, CLASS_NAMES
    import tensorflow as tf
    
    # Check if both multimodal models exist for ensemble
    if os.path.exists(DENSENET_MODEL_PATH) and os.path.exists(MULTIMODAL_MODEL_PATH) and os.path.exists(ENCODER_PATH):
        if DENSENET_MODEL is None:
            print(f"Loading DenseNet121 multimodal model from {DENSENET_MODEL_PATH}...")
            DENSENET_MODEL = tf.keras.models.load_model(DENSENET_MODEL_PATH)
        if MODEL is None:
            print(f"Loading EfficientNetB0 multimodal model from {MULTIMODAL_MODEL_PATH}...")
            MODEL = tf.keras.models.load_model(MULTIMODAL_MODEL_PATH)
        IS_ENSEMBLE = True
        IS_MULTIMODAL = True
        load_encoder()
        print("Ensemble multimodal models loaded successfully.")
    elif os.path.exists(MULTIMODAL_MODEL_PATH) and os.path.exists(ENCODER_PATH):
        if MODEL is None:
            print(f"Loading multimodal model from {MULTIMODAL_MODEL_PATH}...")
            MODEL = tf.keras.models.load_model(MULTIMODAL_MODEL_PATH)
            IS_MULTIMODAL = True
            load_encoder()
            print("Multimodal model loaded successfully.")
    elif os.path.exists(MODEL_PATH):
        if MODEL is None:
            print(f"Loading image-only model from {MODEL_PATH}...")
            MODEL = tf.keras.models.load_model(MODEL_PATH)
            IS_MULTIMODAL = False
            print("Image-only model loaded successfully.")
    else:
        print("No saved model found yet.")

    if os.path.exists(CLASS_MAPPING_PATH):
        with open(CLASS_MAPPING_PATH, 'r') as f:
            mapping = json.load(f)
            CLASS_NAMES = mapping['classes']
    return MODEL

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/info')
def api_info():
    model = get_model()
    model_ready = model is not None
    metrics = None
    
    # Priority: ensemble metrics > multimodal metrics > image-only metrics
    if IS_ENSEMBLE and os.path.exists(ENSEMBLE_METRICS_PATH):
        try:
            with open(ENSEMBLE_METRICS_PATH, 'r') as f:
                metrics = json.load(f)
        except Exception:
            pass
    elif IS_MULTIMODAL and os.path.exists(MULTIMODAL_METRICS_PATH):
        try:
            with open(MULTIMODAL_METRICS_PATH, 'r') as f:
                metrics = json.load(f)
        except Exception:
            pass
    elif os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, 'r') as f:
                metrics = json.load(f)
        except Exception:
            pass
            
    if IS_ENSEMBLE:
        arch_name = "SkinDisNet Multimodal Dual-Backbone Ensemble (DenseNet121 + EfficientNetB0)"
    elif IS_MULTIMODAL:
        arch_name = "SkinDisNet Multimodal Dual-Branch (Image + Metadata)"
    else:
        arch_name = "EfficientNetB0 (Transfer Learning)"
        
    return jsonify({
        "status": "ready" if model_ready else "training",
        "model_architecture": arch_name,
        "is_multimodal": IS_MULTIMODAL,
        "is_ensemble": IS_ENSEMBLE,
        "input_resolution": "224x224 RGB",
        "classes": CLASS_NAMES,
        "metrics": metrics,
        "disclaimer": "Academic research and preliminary screening tool. Not intended for direct medical diagnosis."
    })

@app.route('/api/samples')
def api_samples():
    """Return 1 sample image per class from the test manifest with full metadata."""
    samples = []
    if os.path.exists(MANIFEST_PATH):
        try:
            df = pd.read_csv(MANIFEST_PATH)
            test_df = df[df['Split'] == 'test']
            for cls in CLASS_NAMES:
                cls_samples = test_df[test_df['Diagnosis'] == cls]
                if not cls_samples.empty:
                    row = cls_samples.iloc[0]
                    samples.append({
                        "class_name": cls,
                        "image_id": row['Image_id'],
                        "patient_id": row['Patient_id'],
                        "age": str(row['Age']),
                        "sex": str(row['Sex']),
                        "location": str(row['Leision_location']),
                        "rel_path": row['Relative_path']
                    })
        except Exception as e:
            print("Error loading samples:", e)
    return jsonify({"samples": samples})

@app.route('/api/image/<path:subpath>')
def serve_image(subpath):
    safe_path = os.path.join(DATASET_DIR, subpath)
    if os.path.exists(safe_path):
        return send_file(safe_path, mimetype='image/jpeg')
    return "Image not found", 404

@app.route('/artifacts/<path:filename>')
def serve_artifact(filename):
    return send_from_directory(ARTIFACTS_DIR, filename)

@app.route('/api/predict', methods=['POST'])
def predict():
    model = get_model()
    if model is None:
        return jsonify({
            "error": "Model is still training or artifact not found. Please wait for training to complete."
        }), 503

    image_file = None
    loc_val = 'Unknown'
    age_val = '30'
    sex_val = 'Unknown'

    if 'image' in request.files:
        image_file = request.files['image']
        loc_val = request.form.get('location', 'Unknown')
        age_val = request.form.get('age', '30')
        sex_val = request.form.get('sex', 'Unknown')
    elif request.is_json and 'image_base64' in request.json:
        b64_data = request.json['image_base64']
        if ',' in b64_data:
            b64_data = b64_data.split(',', 1)[1]
        image_file = io.BytesIO(base64.b64decode(b64_data))
        loc_val = request.json.get('location', 'Unknown')
        age_val = request.json.get('age', '30')
        sex_val = request.json.get('sex', 'Unknown')
    elif request.is_json and 'image_path' in request.json:
        full_p = os.path.normpath(os.path.join(DATASET_DIR, request.json['image_path']))
        if os.path.exists(full_p):
            image_file = open(full_p, 'rb')
            loc_val = request.json.get('location', 'Unknown')
            age_val = request.json.get('age', '30')
            sex_val = request.json.get('sex', 'Unknown')
            
    if image_file is None:
        return jsonify({"error": "No image supplied"}), 400

    try:
        with Image.open(image_file) as img:
            rgb_img = img.convert('RGB')
            orig_width, orig_height = rgb_img.size
            resized_img = rgb_img.resize((224, 224), Image.Resampling.BILINEAR)
            img_arr = np.array(resized_img, dtype=np.float32)
            
            # Test-Time Augmentation (TTA) batch of 4 crops
            var1 = img_arr
            var2 = np.fliplr(img_arr)
            var3 = np.flipud(img_arr)
            var4 = np.rot90(img_arr, k=1, axes=(0, 1))
            img_batch = np.stack([var1, var2, var3, var4], axis=0)

        # Run inference (Dual-Backbone Ensemble vs Single Multimodal vs Single-branch Image)
        if IS_ENSEMBLE and DENSENET_MODEL is not None:
            meta_vec = encode_metadata_vec(age_val, sex_val, loc_val)
            meta_batch = np.tile(meta_vec, (4, 1))
            p_effnet = model.predict({"input_image": img_batch, "input_metadata": meta_batch}, verbose=0)
            p_densenet = DENSENET_MODEL.predict({"input_image": img_batch, "input_metadata": meta_batch}, verbose=0)
            batch_preds = 0.5 * p_effnet + 0.5 * p_densenet
        elif IS_MULTIMODAL:
            meta_vec = encode_metadata_vec(age_val, sex_val, loc_val)
            meta_batch = np.tile(meta_vec, (4, 1))
            batch_preds = model.predict({"input_image": img_batch, "input_metadata": meta_batch}, verbose=0)
        else:
            batch_preds = model.predict(img_batch, verbose=0)

        probs = np.mean(batch_preds, axis=0)
        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        pred_conf = float(probs[pred_idx])

        # Prepare class probabilities list sorted descending
        breakdown = []
        for i, cls in enumerate(CLASS_NAMES):
            breakdown.append({
                "class_name": cls,
                "probability": float(probs[i]),
                "percent": round(float(probs[i]) * 100, 2),
                "is_top": (i == pred_idx)
            })
        breakdown.sort(key=lambda x: x["probability"], reverse=True)

        info = CLINICAL_KNOWLEDGE.get(pred_class, {})

        return jsonify({
            "success": True,
            "predicted_class": pred_class,
            "confidence": pred_conf,
            "confidence_percent": round(pred_conf * 100, 2),
            "probabilities": breakdown,
            "clinical_info": info,
            "image_dimensions": f"{orig_width}x{orig_height}",
            "applied_metadata": {
                "location": loc_val,
                "age": age_val,
                "sex": sex_val,
                "is_multimodal_active": IS_MULTIMODAL
            },
            "disclaimer": "Academic research and preliminary screening tool only. Not a medical diagnosis. Consult a certified dermatologist."
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
