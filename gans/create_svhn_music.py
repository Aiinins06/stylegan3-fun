import os
import zipfile
import json
import numpy as np
from torchvision.datasets import SVHN
from tqdm import tqdm
from PIL import Image
import io

def build_stylegan_zip():
    zip_path = "/dataslow/storage/Experiments/INTERNS/anavarror/gans/svhn_music_conditioned.zip"
    centroids_path = "/dataslow/storage/Experiments/INTERNS/anavarror/gans/musical_centroids.npy"

    print("Loading musical centroids...")
    try:
        centroides = np.load(centroids_path, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading centroids\n{e}")
        return

    svhn_to_genre = {
        0: "Pop indie",
        1: "Mundo",
        2: "Ambiente",
        3: "Rnb",
        4: "Clásica",
        5: "Hip Hop Convencional", 
        6: "Techno Y Trance",      
        7: "Piano clasico",
        8: "Rock",
        9: "Metal"
    }

    print("Downloading SVHN...")
    dataset = SVHN(root="./data", split='train', download=True)

    labels_list = []

    print(f"Creating file ZIP in: {zip_path}")
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)

    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_STORED) as zf:
        
        for i, (img, label_idx) in enumerate(tqdm(dataset, desc="Writing images to ZIP")):
            genero = svhn_to_genre[label_idx]
            
            if genero not in centroides:
                raise KeyError(f"¡Error! Genre '{genero}' not found in .npy")

            vector_condicion = [float(val) for val in centroides[genero]]
            
            img_name = f"img_{i:06d}.png"
            
            # Save image to memory and then to ZIP
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            zf.writestr(img_name, img_byte_arr.getvalue())


            labels_list.append([img_name, vector_condicion])

        print("Writing dataset.json inside ZIP...")
        json_data = json.dumps({"labels": labels_list})
        zf.writestr("dataset.json", json_data)

if __name__ == "__main__":
    build_stylegan_zip()


"""
CUDA_VISIBLE_DEVICES=0 python train.py --outdir=/dataslow/storage/Experiments/INTERNS/anavarror/gans/training_svhn --cfg=stylegan2
--data=/dataslow/storage/Experiments/INTERNS/anavarror/gans/svhn_music_conditioned.zip \
--gpus=1 --batch=32 --gamma=1 --cond=True --mirror=0 \
"""