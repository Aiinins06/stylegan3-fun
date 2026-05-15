import os
import torch

# Ajusta esta ruta a tu carpeta real
# FEATURE_DIR = '/datafast/105-1/Datasets/INTERNS/anavarror/feature_spaces_1k_20s_contain_fragments/facebook_wav2vec2-base_attentionTrue_'
FEATURE_DIR = '/datafast/105-1/Datasets/INTERNS/anavarror/feature_spaces_1k_20s_contain_fragments/m-a-p_music2vec-v1_attentionTrue_'
# FEATURE_DIR = '/datafast/105-1/Datasets/INTERNS/anavarror/feature_spaces_1k_20s_contain_fragments/MERT-v1-95M_attentionTrue_'

print(f"Checking files in: {FEATURE_DIR}")

for f in os.listdir(FEATURE_DIR):
    if f.endswith(".pt"):
        full_path = os.path.join(FEATURE_DIR, f)
        file_size = os.path.getsize(full_path)
        
        print(f"File: {f} | Size: {file_size/1024:.2f} KB", end=" ")
        
        if file_size == 0:
            print("-> CORRUPT (Empty file)")
            continue
            
        try:
            # Intenta cargar solo la cabecera para ver si es un zip válido
            torch.load(full_path, map_location='cpu')
            print("-> OK")
        except Exception as e:
            print(f"-> CORRUPT ({e})")