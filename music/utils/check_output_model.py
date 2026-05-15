import os
import pandas as pd

# --- STATIC CONFIGURATION ---
GENRE_MAPPING = {
    "Ambient": "Ambiente",
    "Electronic": "Electrónico",
    "Hip-Hop R B": "Hip Hop Convencional",
    "Rock": "Rock",
    "Pop": "Pop",
    "Jazz": "Jazz Clásico",
    "Techno": "Techno Y Trance",
}

def get_ground_truth(directory, mapping):
    """Creates dictionary of {song_name: true_label}"""
    truth_dict = {}
    if not os.path.exists(directory):
        return {}

    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        if os.path.isdir(folder_path) and folder_name in mapping:
            true_label = mapping[folder_name]
            for filename in os.listdir(folder_path):
                if filename.startswith('.'): continue
                song_name_clean = os.path.splitext(filename)[0]
                truth_dict[song_name_clean] = true_label
    return truth_dict

def check_accuracy(csv_file_path, ground_truth_dir):
    # --- SETUP LOGGING ---
    report_lines = []
    
    def log(message=""):
        print(message)
        report_lines.append(str(message))

    # --- EXECUTION ---
    log("Scanning directories for ground truth...")
    truth_db = get_ground_truth(ground_truth_dir, GENRE_MAPPING)
    
    if not truth_db:
        log(f"Error: No songs found in {ground_truth_dir}")
        return

    log(f"Loading results from {csv_file_path}...")
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        log(f"Error: CSV file not found at {csv_file_path}")
        return

    correct_count = 0
    missing_count = 0
    total_checked = 0
    results_data = []

    for index, row in df.iterrows():
        song = row['song_name']
        predicted = row['predicted_label']
        
        if song in truth_db:
            actual = truth_db[song]
            is_correct = (predicted == actual)
            if is_correct: correct_count += 1
            
            results_data.append({'song_name': song, 'predicted_label': predicted, 'act': actual, 'stat': 'Correct' if is_correct else 'Incorrect'})
            total_checked += 1
        else:
            results_data.append({'song_name': song, 'predicted_label': predicted, 'act': 'NOT_FOUND', 'stat': 'Missing'})
            missing_count += 1

    results_df = pd.DataFrame(results_data)

    # --- GENERATE REPORT CONTENT ---
    log("\n" + "="*30)
    log("       VALIDATION REPORT       ")
    log("="*30)
    log(f"Total Songs in CSV: {len(df)}")
    log(f"Songs matched in folders: {total_checked}")
    log(f"Songs missing from folders: {missing_count}")
    log("-" * 30)
    
    if total_checked > 0:
        accuracy = (correct_count / total_checked) * 100
        log(f"Correct Predictions: {correct_count}")
        log(f"Incorrect Predictions: {total_checked - correct_count}")
        log(f"FINAL ACCURACY: {accuracy:.2f}%")
        
        incorrect = results_df[results_df['stat'] == 'Incorrect']
        if not incorrect.empty:
            log("\n--- Incorrect Predictions ---")
            log(incorrect[['song_name', 'predicted_label', 'act']].to_string(index=False))

    # --- SAVE REPORT TO FILE ---
    csv_dir = os.path.dirname(os.path.abspath(csv_file_path))
    report_filename = "validation_report.txt"
    report_path = os.path.join(csv_dir, report_filename)
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"\n[INFO] Report text saved to: {report_path}")
    except Exception as e:
        print(f"\n[ERROR] Could not save text report: {e}")