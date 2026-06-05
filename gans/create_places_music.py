import os
import zipfile
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import io

def build_custom_stylegan_zip():
    kaggle_extracted_dir = "/dataslow/storage/Experiments/INTERNS/anavarror/gans/places_subset_raw_images" 
    centroids_path = "/dataslow/storage/Experiments/INTERNS/anavarror/gans/musical_centroids.npy"
    out_zip_path = "/dataslow/storage/Experiments/INTERNS/anavarror/gans/places_music_conditioned.zip"

    print("Loading musical centroids...")
    centroids = np.load(centroids_path, allow_pickle=True).item()

    # --- MAP FOLDERS TO MUSIC GENRES ---
    class_to_genre = {
        "a/abbey": "Pop indie",
        "b/boardwalk": "Mundo",
        "c/chalet": "Ambiente",
        "d/desert/sand": "Rnb",
        "f/forest_road": "Clásica",
        "h/home_office": "Hip Hop Convencional", 
        "i/iceberg": "Techno Y Trance",      
        "k/kasbah": "Piano clasico",
        "l/lighthouse": "Rock",
        "m/mountain": "Metal"
    }

    labels_list = []
    
    print(f"Creating StyleGAN ZIP directly at: {out_zip_path}")
    os.makedirs(os.path.dirname(out_zip_path), exist_ok=True)

    with zipfile.ZipFile(out_zip_path, 'w', compression=zipfile.ZIP_STORED) as zf:
        
        img_counter = 0
        
        for class_folder, genre in class_to_genre.items():
            folder_path = os.path.join(kaggle_extracted_dir, class_folder)
            
            if not os.path.exists(folder_path):
                print(f"WARNING: Folder '{class_folder}' not found. Skipping...")
                continue
                
            if genre not in centroids:
                raise KeyError(f"Error: Genre '{genre}' not found in centroids.")

            vector_condicion = [float(val) for val in centroids[genre]]
            
            # Process all images in this class folder
            images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_name in tqdm(images, desc=f"Processing {class_folder} -> {genre}"):
                img_path = os.path.join(folder_path, img_name)
                
                try:
                    # Open and enforce 256x256 RGB
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((256, 256), Image.Resampling.LANCZOS)
                    
                    new_img_name = f"img_{img_counter:07d}.png"
                    
                    # Save to ZIP
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    zf.writestr(new_img_name, img_byte_arr.getvalue())

                    # Append label
                    labels_list.append([new_img_name, vector_condicion])
                    img_counter += 1
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

        # Write dataset.json inside the ZIP
        print("Writing dataset.json into ZIP...")
        json_data = json.dumps({"labels": labels_list})
        zf.writestr("dataset.json", json_data)

    print(f"\nDone! Processed {img_counter} images. You can now train.")

if __name__ == "__main__":
    build_custom_stylegan_zip()


""" 
    FOR TRAINING AFTER THE ZIP FILE CONDITIONED HAS BEEN CREATED

    CUDA_VISIBLE_DEVICES=4 python train.py \
    --outdir=/dataslow/storage/Experiments/INTERNS/anavarror/gans/training_256 \
    --cfg=stylegan2 \
    --data=/dataslow/storage/Experiments/INTERNS/anavarror/gans/custom256_music_conditioned.zip \
    --gpus=1 \
    --batch=8 \
    --gamma=1 \
    --cond=True \
    --mirror=1
"""