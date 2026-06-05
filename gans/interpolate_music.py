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

def interpolate_genres():
    device = torch.device('cuda:0')
    
    # --- ROUTES ---
    network_pkl = '/dataslow/storage/Experiments/INTERNS/anavarror/gans/training_places/00001-stylegan2-places_music_conditioned-gpus4-batch32-gamma1-no_resume/network-snapshot-021400.pkl' 
    centroids_path = '/dataslow/storage/Experiments/INTERNS/anavarror/gans/musical_centroids.npy'
    base_out_dir = '/dataslow/storage/Experiments/INTERNS/anavarror/gans/gan_places_interpolations'
    
    print(f'Loading network SVHN from "{network_pkl}"...')
    with open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    print('Loading musical centroids...')
    centroides = np.load(centroids_path, allow_pickle=True).item()
    
    # --- INTERPOLATION CONFIGURATION ---
    genre_A = "Clásica"           
    genre_B = "Techno Y Trance"  
    steps = 9                   # Total number of images (0% to 100%)
    
    seeds_to_test = [42, 50, 100, 500, 9999] 
    
    print(f'Interpolation from {genre_A} to {genre_B} in {steps} steps...')
    
    vector_A = torch.tensor(centroides[genre_A], dtype=torch.float32).to(device)
    vector_B = torch.tensor(centroides[genre_B], dtype=torch.float32).to(device)

    # --- FOLDER STRUCTURE ---
    safe_genre_A = genre_A.replace(" ", "_")
    safe_genre_B = genre_B.replace(" ", "_")
    condition_folder_name = f"{safe_genre_A}_to_{safe_genre_B}_truncation"
    condition_dir = os.path.join(base_out_dir, condition_folder_name)
    os.makedirs(condition_dir, exist_ok=True)

    for seed in seeds_to_test:
        print(f'\n--- Starting generation for Seed: {seed} ---')
        
        # Create the subfolder for this specific seed
        seed_dir = os.path.join(condition_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        
        # Set the seed and generate the fixed noise for this run
        torch.manual_seed(seed) 
        z_fix = torch.randn([1, G.z_dim]).to(device)

        generated_images = []

        # Iterate through the interpolation steps
        for i in range(steps):
            # Computing Alpha (0.0 first step, 1.0 last one)
            alpha = i / (steps - 1)
            
            # Linear Interpolation (LERP)
            vector_mezclado = (1.0 - alpha) * vector_A + alpha * vector_B
            c_mix = vector_mezclado.unsqueeze(0) # Add batch dimension
            
            # Generating image
            img_tensor = G(z_fix, c_mix, truncation_psi=0.7, noise_mode='const')
            img_tensor = (img_tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img_np = img_tensor[0].cpu().numpy()
            
            img = Image.fromarray(img_np, 'RGB')
            generated_images.append(img)
            
            # Saving individual image inside the seed folder
            filename = f'interp_{i:02d}_alpha{alpha:.2f}.png'
            img.save(os.path.join(seed_dir, filename))

        # Grouping all images into a horizontal grid
        total_width = sum(img.width for img in generated_images)
        max_height = max(img.height for img in generated_images)
        final_image = Image.new('RGB', (total_width, max_height))
        
        x_offset = 0
        for img in generated_images:
            final_image.paste(img, (x_offset, 0))
            x_offset += img.width
            
        # Save the final strip inside the seed folder
        grid_path = os.path.join(seed_dir, f'Interpolation_grid.png')
        final_image.save(grid_path)
        print(f'Saved grid to: {grid_path}')

    print(f'\nAll seeds completed! Check the directory: {condition_dir}')

if __name__ == "__main__":
    interpolate_genres()