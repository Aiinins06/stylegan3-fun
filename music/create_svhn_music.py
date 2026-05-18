import os
import json
import numpy as np
from torchvision.datasets import SVHN
from tqdm import tqdm

def prepare_stylegan_svhn_dataset():
    print("Loading musical centroids...")
    centroides = np.load("/dataslow/storage/Experiments/INTERNS/anavarror/gans/musical_centroids.npy", allow_pickle=True).item()

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

    out_dir = "/dataslow/storage/Experiments/INTERNS/anavarror/gans/svhn_music_raw"
    os.makedirs(out_dir, exist_ok=True)

    print("Downloading SVHN...")
    # SVHN usa 'split="train"' en lugar de 'train=True'
    dataset = SVHN(root="./data", split='train', download=True)

    labels_dict = {}

    print("Mapping images with musical vectors...")
    for i, (img, label_idx) in enumerate(tqdm(dataset, desc="Processing images")):
        # SVHN returns the labels as intenger from 0 to 9
        genero = svhn_to_genre[label_idx]
        
        if genero not in centroides:
            raise KeyError(f"¡Error! El género '{genero}' no se encontró en el .npy")

        vector_condicion = centroides[genero].tolist() 

        img_name = f"img_{i:06d}.png"
        img.save(os.path.join(out_dir, img_name))

        # Asigning conditional vector to image
        labels_dict[img_name] = vector_condicion

    print("Writing dataset.json...")
    with open(os.path.join(out_dir, "dataset.json"), "w") as f:
        json.dump({"labels": labels_dict}, f)

    print(f"\n¡Done! Folder '{out_dir}' ready for StyleGAN.")

if __name__ == "__main__":
    prepare_stylegan_svhn_dataset()

    # python dataset_tool.py --source=svhn_music_raw --dest=datasets/svhn_music_condicionado.zip

    # python train.py --outdir=training-runs --cfg=stylegan2 --data=datasets/svhn_music_condicionado.zip --gpus=1 --batch=32 --gamma=1 --cond=True --mirror=0 