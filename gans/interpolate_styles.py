import os
import sys
import torch
import numpy as np
from PIL import Image

stylegan_root = '/home/anavarror/stylegan3-fun'
if stylegan_root not in sys.path:
    sys.path.append(stylegan_root)

from dnnlib.util import open_url
import legacy

def interpolate_styles():
    device = torch.device('cuda:0')
    
    # --- ROUTES ---
    network_pkl = '/dataslow/storage/Experiments/INTERNS/anavarror/gans/training_svhn/00003-stylegan2-svhn_music_conditioned-gpus2-batch64-gamma1-no_resume/network-snapshot-018950.pkl' 
    centroids_path = '/dataslow/storage/Experiments/INTERNS/anavarror/gans/musical_centroids.npy'
    base_out_dir = '/dataslow/storage/Experiments/INTERNS/anavarror/gans/gan_style_interpolations'
    os.makedirs(base_out_dir, exist_ok=True)
    
    print(f'Loading network from "{network_pkl}"...')
    with open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    print('Loading musical centroids...')
    centroides = np.load(centroids_path, allow_pickle=True).item()
    
    genero_fijo = "Ambiente"           
    c_fijo = torch.tensor(centroides[genero_fijo], dtype=torch.float32).unsqueeze(0).to(device)
    
    seed_A = 50      # Initial style
    seed_B = 500    # Final style
    pasos = 9        # Number of photos to transition
    
    print(f'Interpoling STILE from Seed {seed_A} to Seed {seed_B} for genre {genero_fijo}...')
    
    torch.manual_seed(seed_A)
    z_A = torch.randn([1, G.z_dim]).to(device)
    
    torch.manual_seed(seed_B)
    z_B = torch.randn([1, G.z_dim]).to(device)

    run_dir = os.path.join(base_out_dir, f"{genero_fijo}_Style_{seed_A}_to_{seed_B}")
    os.makedirs(run_dir, exist_ok=True)

    imagenes_generadas = []

    for i in range(pasos):
        alpha = i / (pasos - 1)
        
        # Linear Interpolation (LERP)
        z_mix = (1.0 - alpha) * z_A + alpha * z_B
        
        img_tensor = G(z_mix, c_fijo, truncation_psi=0.7, noise_mode='const')
        img_tensor = (img_tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_np = img_tensor[0].cpu().numpy()
        
        img = Image.fromarray(img_np, 'RGB')
        imagenes_generadas.append(img)
        
        filename = f'style_interp_{i:02d}_alpha{alpha:.2f}.png'
        img.save(os.path.join(run_dir, filename))
        print(f'Step {i+1}/{pasos} generated (Alpha={alpha:.2f})')

    total_width = sum(img.width for img in imagenes_generadas)
    max_height = max(img.height for img in imagenes_generadas)
    final_image = Image.new('RGB', (total_width, max_height))
    
    x_offset = 0
    for img in imagenes_generadas:
        final_image.paste(img, (x_offset, 0))
        x_offset += img.width
        
    grid_path = os.path.join(run_dir, f'Style_Transition_Grid.png')
    final_image.save(grid_path)
    
    print(f'\nDone: {grid_path}')

if __name__ == "__main__":
    interpolate_styles()