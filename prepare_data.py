import os
import csv
import json
import random
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

WORKSPACE = r"c:\SkinDisNet A Multi-Class Clinical Images and Metad"
DATASET_DIR = os.path.join(WORKSPACE, "SkinDisNet_2")
METADATA_PATH = os.path.join(DATASET_DIR, "SkinDisNet_Metadata.csv")
PREPROCESSED_DIR = os.path.join(DATASET_DIR, "Preprocessed")
ARTIFACTS_DIR = os.path.join(WORKSPACE, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

CLASS_NAMES = [
    "Atopic Dermatitis",
    "Contact Dermatitis",
    "Eczema",
    "Scabies",
    "Seborrheic Dermatitis",
    "Tinea Corporis"
]

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {str(i): name for i, name in enumerate(CLASS_NAMES)}

def prepare_splits(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    df = pd.read_csv(METADATA_PATH)
    
    # Verify columns and classes
    assert set(df['Diagnosis'].unique()) == set(CLASS_NAMES), "Classes do not match expected classes"
    
    # Save class mapping
    class_mapping_path = os.path.join(ARTIFACTS_DIR, "class_mapping.json")
    with open(class_mapping_path, "w") as f:
        json.dump({
            "class_to_idx": CLASS_TO_IDX,
            "idx_to_class": IDX_TO_CLASS,
            "classes": CLASS_NAMES
        }, f, indent=2)
    print(f"Saved class mapping to {class_mapping_path}")

    # Build patient to diagnosis and patient to images mapping
    patient_diagnosis = {}
    patient_images = defaultdict(list)
    
    for idx, row in df.iterrows():
        pid = row['Patient_id']
        diag = row['Diagnosis']
        folder = row['Folder_name']
        img_id = row['Image_id']
        fname = f"{img_id}.jpg"
        rel_path = os.path.join("Preprocessed", folder, fname)
        full_path = os.path.join(DATASET_DIR, rel_path)
        
        assert os.path.exists(full_path), f"File {full_path} not found"
        
        patient_diagnosis[pid] = diag
        patient_images[pid].append({
            "Patient_id": pid,
            "Image_id": img_id,
            "Folder_name": folder,
            "Diagnosis": diag,
            "Class_index": CLASS_TO_IDX[diag],
            "Relative_path": rel_path.replace("\\", "/"),
            "Age": row['Age'],
            "Sex": row['Sex'],
            "Leision_location": row['Leision_location']
        })

    # Group patients by diagnosis for stratified splitting
    diag_to_patients = defaultdict(list)
    for pid, diag in patient_diagnosis.items():
        diag_to_patients[diag].append(pid)

    train_patients = []
    val_patients = []
    test_patients = []

    # Target: 70% Train, 15% Val, 15% Test
    for diag, pids in diag_to_patients.items():
        pids_shuffled = sorted(pids) # deterministic initial sort
        random.shuffle(pids_shuffled)
        
        n = len(pids_shuffled)
        n_test = max(1, int(round(n * 0.15)))
        n_val = max(1, int(round(n * 0.15)))
        n_train = n - n_val - n_test
        
        # In small classes, make sure at least 1 patient in val and test
        if n_train < 1:
            n_train = max(1, n - 2)
            n_val = 1
            n_test = n - n_train - n_val

        test_p = pids_shuffled[:n_test]
        val_p = pids_shuffled[n_test:n_test + n_val]
        train_p = pids_shuffled[n_test + n_val:]

        train_patients.extend(train_p)
        val_patients.extend(val_p)
        test_patients.extend(test_p)

    print(f"Patient counts -> Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)} | Total: {len(patient_diagnosis)}")

    # Verify no overlap in patients
    assert len(set(train_patients).intersection(set(val_patients))) == 0
    assert len(set(train_patients).intersection(set(test_patients))) == 0
    assert len(set(val_patients).intersection(set(test_patients))) == 0

    manifest_records = []
    
    for split_name, plist in [("train", train_patients), ("val", val_patients), ("test", test_patients)]:
        for pid in plist:
            for item in patient_images[pid]:
                record = dict(item)
                record["Split"] = split_name
                manifest_records.append(record)

    manifest_df = pd.DataFrame(manifest_records)
    manifest_path = os.path.join(ARTIFACTS_DIR, "dataset_split_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Saved dataset split manifest to {manifest_path} with {len(manifest_df)} total images.")

    # Print summary statistics per split
    print("\nImage distribution across splits:")
    pivot = pd.crosstab(manifest_df['Diagnosis'], manifest_df['Split'], margins=True)
    print(pivot)

    print("\nPatient distribution across splits:")
    p_df = manifest_df[['Patient_id', 'Diagnosis', 'Split']].drop_duplicates()
    p_pivot = pd.crosstab(p_df['Diagnosis'], p_df['Split'], margins=True)
    print(p_pivot)

if __name__ == "__main__":
    prepare_splits(seed=42)
