import os
import sys
import torch
import numpy as np
from PIL import Image

stylegan_root = '/home/anavarror/stylegan3-fun'
if stylegan_root not in sys.path:
    sys.path.append(stylegan_root)

from dnnlib.util import open_url
from legacy import load_network_pkl

def generate_music_conditioned_images():
    device = torch.device('cuda:0')
    
    # "/dataslow/storage/Experiments/INTERNS/anavarror/gans/training_svhn/00003-stylegan2-svhn_music_conditioned-gpus2-batch64-gamma1-no_resume/network-snapshot-018950.pkl" 
    # '/dataslow/storage/Experiments/INTERNS/anavarror/gans/training_places/00001-stylegan2-places_music_conditioned-gpus4-batch32-gamma1-no_resume/network-snapshot-021400.pkl' 
    network_pkl = '/dataslow/storage/Experiments/INTERNS/anavarror/gans/training_places/00001-stylegan2-places_music_conditioned-gpus4-batch32-gamma1-no_resume/network-snapshot-021400.pkl' 
    centroids_path = '/dataslow/storage/Experiments/INTERNS/anavarror/gans/musical_centroids.npy'
    out_dir = '/dataslow/storage/Experiments/INTERNS/anavarror/gans/gan_places_music_results'
    os.makedirs(out_dir, exist_ok=True)
    
    print(f'Loading network from "{network_pkl}"...')
    with open_url(network_pkl) as f:
        G = load_network_pkl(f)['G_ema'].to(device) # G_ema is the stable version of Generator

    print('Loading musical centroids...')
    centroids = np.load(centroids_path, allow_pickle=True).item()
    
    # Creating one noisy vector
    torch.manual_seed(42) 
    z = torch.randn([1, G.z_dim]).to(device)

    print('Generating images...')
    for genre, vector in centroids.items():

        # ---------- FIX NOISE ---------- 
        # Preparing condition (c)
        c = torch.tensor(vector, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Fowrard through the Generator
        img_tensor = G(z, c, truncation_psi=0.7, noise_mode='const')
        
        # Post-process the tensor into an RGB image (StyleGAN output values are from -1 to 1)
        img_tensor = (img_tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_np = img_tensor[0].cpu().numpy()
        
        img = Image.fromarray(img_np, 'RGB')
        img.save(f'{out_dir}/generated_{genre.replace(" ", "_")}.png')
        # print(f'Saved image for: {genre}')

        # ---------- FIX GERNE ----------
        vector_fijo = centroids[genre]
        c_fijo = torch.tensor(vector_fijo, dtype=torch.float32).unsqueeze(0).to(device)

        styles_seeds = [10, 50, 100, 500, 1000, 9999]
        
        for semilla in styles_seeds:
            torch.manual_seed(semilla) 
            z_variable = torch.randn([1, G.z_dim]).to(device)
            
            img_tensor = G(z_variable, c_fijo, truncation_psi=0.7, noise_mode='const')
            img_tensor = (img_tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img_np = img_tensor[0].cpu().numpy()
            
            img = Image.fromarray(img_np, 'RGB')
            img.save(f'{out_dir}/styles_{genre}_{semilla}.png')
    

    print(f'\nResults saved to {out_dir}')

if __name__ == "__main__":
    generate_music_conditioned_images()