import os
import csv
import hashlib
from collections import defaultdict, Counter
from PIL import Image
import json

WORKSPACE = r"c:\SkinDisNet A Multi-Class Clinical Images and Metad"
DATASET_DIR = os.path.join(WORKSPACE, "SkinDisNet_2")
METADATA_PATH = os.path.join(DATASET_DIR, "SkinDisNet_Metadata.csv")
PREPROCESSED_DIR = os.path.join(DATASET_DIR, "Preprocessed")
AUGMENTED_DIR = os.path.join(DATASET_DIR, "Augmented")

def inspect():
    results = {}
    
    # 1. Inspect metadata
    if not os.path.exists(METADATA_PATH):
        print(f"Metadata file not found at {METADATA_PATH}")
        return

    with open(METADATA_PATH, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    results['metadata_headers'] = headers
    results['total_rows'] = len(rows)

    # Check missing values per column
    missing_counts = defaultdict(int)
    for r in rows:
        for k, v in r.items():
            if v is None or v.strip() == "":
                missing_counts[k] += 1
    results['missing_values'] = dict(missing_counts)
    results['sample_rows'] = rows[:5]

    # Analyze metadata columns
    folder_counts = Counter(r['Folder_name'] for r in rows)
    diag_counts = Counter(r['Diagnosis'] for r in rows)
    patient_ids = set(r['Patient_id'] for r in rows if r.get('Patient_id'))
    images_in_csv = [r['Image_id'] for r in rows]
    
    results['folder_counts'] = dict(folder_counts)
    results['diagnosis_counts'] = dict(diag_counts)
    results['unique_patients_count'] = len(patient_ids)
    results['total_images_in_csv'] = len(images_in_csv)

    # Patient to diagnosis mapping (check if patients have single diagnosis)
    patient_to_diag = defaultdict(set)
    for r in rows:
        patient_to_diag[r['Patient_id']].add(r['Diagnosis'])
    multi_diag_patients = {p: list(diags) for p, diags in patient_to_diag.items() if len(diags) > 1}
    results['multi_diag_patients'] = multi_diag_patients

    # 2. Inspect Preprocessed Images directory
    preproc_files = {} # relative_path -> full_path
    preproc_by_class = defaultdict(list)
    preproc_unreadable = []
    preproc_shapes = Counter()
    preproc_formats = Counter()
    preproc_hashes = defaultdict(list)

    for root, dirs, files in os.walk(PREPROCESSED_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, PREPROCESSED_DIR)
            folder = os.path.dirname(rel_path)
            preproc_files[file] = full_path
            preproc_by_class[folder].append(file)

            # verify readability and format
            try:
                with Image.open(full_path) as img:
                    preproc_formats[img.format] += 1
                    preproc_shapes[img.size] += 1
            except Exception as e:
                preproc_unreadable.append((rel_path, str(e)))

            # calculate md5 hash for exact duplicate detection
            with open(full_path, 'rb') as fp:
                h = hashlib.md5(fp.read()).hexdigest()
                preproc_hashes[h].append(rel_path)

    results['preprocessed_total_files'] = len(preproc_files)
    results['preprocessed_by_folder'] = {k: len(v) for k, v in preproc_by_class.items()}
    results['preprocessed_unreadable'] = preproc_unreadable
    results['preprocessed_formats'] = dict(preproc_formats)
    results['preprocessed_shapes'] = {str(k): v for k, v in preproc_shapes.items()}
    
    preproc_duplicates = {h: paths for h, paths in preproc_hashes.items() if len(paths) > 1}
    results['preprocessed_exact_duplicates'] = preproc_duplicates

    # 3. Match CSV Image_id with Preprocessed Images
    # In CSV: 'Image_id' is e.g. 'AD (1)'. Files are 'AD (1).jpg'
    csv_missing_in_preproc = []
    preproc_missing_in_csv = []
    
    csv_filename_map = {}
    for r in rows:
        img_id = r['Image_id'].strip()
        expected_name = f"{img_id}.jpg"
        if expected_name in preproc_files:
            csv_filename_map[img_id] = expected_name
        else:
            # check without .jpg or case insensitivity
            csv_missing_in_preproc.append(img_id)

    for fname in preproc_files:
        base_name = os.path.splitext(fname)[0]
        if base_name not in [r['Image_id'].strip() for r in rows]:
            preproc_missing_in_csv.append(fname)

    results['csv_missing_in_preproc'] = csv_missing_in_preproc
    results['preproc_missing_in_csv'] = preproc_missing_in_csv
    results['csv_to_preproc_matched_count'] = len(csv_filename_map)

    # 4. Inspect Augmented Images directory
    aug_files = {}
    aug_by_class = defaultdict(list)
    aug_formats = Counter()
    aug_shapes = Counter()
    aug_unreadable = []
    
    for root, dirs, files in os.walk(AUGMENTED_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, AUGMENTED_DIR)
            folder = os.path.dirname(rel_path)
            aug_files[file] = full_path
            aug_by_class[folder].append(file)
            
            try:
                with Image.open(full_path) as img:
                    aug_formats[img.format] += 1
                    aug_shapes[img.size] += 1
            except Exception as e:
                aug_unreadable.append((rel_path, str(e)))

    results['augmented_total_files'] = len(aug_files)
    results['augmented_by_folder'] = {k: len(v) for k, v in aug_by_class.items()}
    results['augmented_unreadable'] = aug_unreadable
    results['augmented_formats'] = dict(aug_formats)
    results['augmented_shapes'] = {str(k): v for k, v in aug_shapes.items()}

    # Check relation between Preprocessed and Augmented
    # Pattern: AD (1).jpg -> AD (1)_01.jpg .. AD (1)_07.jpg
    derivatives_per_original = defaultdict(list)
    orphan_augmented = []
    for fname in aug_files:
        # split by '_0' or pattern
        base = os.path.splitext(fname)[0]
        # if ends with _01, _02, etc.
        parts = base.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            orig_base = parts[0]
            orig_fname = f"{orig_base}.jpg"
            if orig_fname in preproc_files:
                derivatives_per_original[orig_fname].append(fname)
            else:
                orphan_augmented.append(fname)
        else:
            orphan_augmented.append(fname)

    results['derivatives_count'] = len(derivatives_per_original)
    results['derivatives_per_orig_distribution'] = Counter(len(v) for v in derivatives_per_original.values())
    results['orphan_augmented_count'] = len(orphan_augmented)
    if orphan_augmented:
        results['sample_orphan_augmented'] = orphan_augmented[:10]

    # Save report JSON
    with open(os.path.join(WORKSPACE, "dataset_inspection_report.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Inspection complete! Summary:")
    print(f"Metadata rows: {results['total_rows']}")
    print(f"Preprocessed total images: {results['preprocessed_total_files']}")
    print(f"Preprocessed exact duplicates: {len(results['preprocessed_exact_duplicates'])}")
    print(f"CSV matched to Preprocessed: {results['csv_to_preproc_matched_count']}")
    print(f"CSV missing in Preprocessed: {len(results['csv_missing_in_preproc'])}")
    print(f"Preprocessed missing in CSV: {len(results['preproc_missing_in_csv'])}")
    print(f"Augmented total images: {results['augmented_total_files']}")
    print(f"Augmented derivatives matched to original: {sum(len(v) for v in derivatives_per_original.values())}")
    print(f"Unique patients: {results['unique_patients_count']}")
    print(f"Preprocessed folders & counts: {results['preprocessed_by_folder']}")
    print(f"Metadata diagnosis counts: {results['diagnosis_counts']}")

if __name__ == "__main__":
    inspect()
