import librosa
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
from transformers import AutoFeatureExtractor, AutoModel
import numpy as np
import os
from typing import List, Tuple, Optional
from matplotlib.lines import Line2D
from collections import defaultdict
import re
from matplotlib.animation import FuncAnimation


class MusicEmbedder:
    def __init__(self, model_name: str):
        print(f"Loading model: {model_name}")
        self.model_name = model_name
        self.only_name_model = model_name.split("/")[-1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model and feature extractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True
        ).to(self.device)
        
        self.target_sample_rate = self.feature_extractor.sampling_rate
        print(f"Model loaded. Target sample rate: {self.target_sample_rate} Hz")


    def get_song_embedding(self, mp3_path: str) -> Optional[torch.Tensor]:
        try:
            if os.path.getsize(mp3_path) < 5000:  # <5 KB = clearly broken
                print(f"⚠️ File too small, skipping (likely corrupted): {mp3_path}")
                return None
        except Exception:
            print(f"⚠️ Could not stat file, skipping: {mp3_path}")
            return None
        
        try:
            # Load and resample the audio
            audio_array, _ = librosa.load(
                mp3_path, sr=self.target_sample_rate, mono=True
            )
            
            # Prepare audio for the model
            inputs = self.feature_extractor(
                audio_array, 
                sampling_rate=self.target_sample_rate, 
                return_tensors="pt"
            )
            
            # Move inputs to GPU if available
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get embeddings safely
            with torch.no_grad():  # no gradient tracking
                outputs = self.model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                # Mean pooling over time dimension
                song_embedding = torch.mean(last_hidden_state, dim=1).squeeze()

            # Move to CPU immediately
            song_embedding_cpu = song_embedding.detach().cpu()

            # Cleanup GPU memory
            del outputs, inputs, song_embedding
            torch.cuda.empty_cache()

            return song_embedding_cpu

        except Exception as e:
            if "out of memory" in str(e):
                print(f"CUDA out of memory while processing {mp3_path}")
                torch.cuda.empty_cache()
                return None
            else:
                import traceback
                print(f"\n--- DETAILED ERROR processing {mp3_path} ---")
                print(f"Error Type: {type(e).__name__}")
                print(f"Error Message: {e}")
                traceback.print_exc()
                print("--------------------------------------------------\n")
                return None


    def process_genres(self, base_directory: str, genre_list: List[str], max_songs: int = None) -> Tuple[np.ndarray, List[str]]:
        def clean_song_name(filename: str):
            name = os.path.splitext(filename)[0]
            name = re.sub(r"_part\d+$", "", name)
            return name

        all_embeddings = []
        all_genre_labels = []
        name_genres = ""

        for genre in genre_list:
            genre_dir = os.path.join(base_directory, genre)
            if not os.path.isdir(genre_dir):
                print(f"Warning: Directory not found, skipping: {genre_dir}")
                continue

            print(f"\nProcessing genre: {genre}")
            mp3_files = sorted(f for f in os.listdir(genre_dir) if f.endswith(".mp3"))

            song_groups = defaultdict(list)
            genre_lookup = {}

            for counter, song_file in enumerate(mp3_files, start=1):
                file_path = os.path.join(genre_dir, song_file)
                print(f"Processing song {counter} of genre {genre}: {file_path}")

                embedding = self.get_song_embedding(file_path)

                if embedding is not None:
                    base_name = clean_song_name(song_file)
                    song_groups[base_name].append(embedding)
                    genre_lookup[base_name] = genre
                else:
                    print(f"Skipping unreadable file: {file_path}")
                    continue

            # Merge fragments into full-song embeddings
            song_counter = 0
            for base_name, fragment_embeddings in song_groups.items():

                if max_songs is not None and song_counter >= max_songs:
                    print(f"Reached max_songs={max_songs} for genre {genre}.")
                    break

                if len(fragment_embeddings) > 1:
                    print(f"   Averaging {len(fragment_embeddings)} fragments of song: {base_name}")
                    merged_embedding = torch.stack(fragment_embeddings).mean(dim=0)
                else:
                    merged_embedding = fragment_embeddings[0]

                all_embeddings.append(merged_embedding)
                all_genre_labels.append(genre_lookup[base_name])
                song_counter += 1

            name_genres += genre + "_"

        # Convert embedding list to matrix
        if all_embeddings:
            embedding_matrix = torch.stack(all_embeddings).to(self.device)
        else:
            embedding_matrix = torch.empty((0, 0), device=self.device)


        # Save the feature space
        os.makedirs('/datafast/105-1/Datasets/INTERNS/anavarror/feature_spaces_1k_under90_meanfrag', exist_ok=True) # ================== MODIFY IT ACCORDING TO THE RUN =================================
        save_path = f'/datafast/105-1/Datasets/INTERNS/anavarror/feature_spaces_1k_under90_manfrag/{self.only_name_model}_meanfrag_{name_genres}.pt' # ======= MODIFY IT ACCORDING TO THE RUN =======
        final_path = os.path.join(base_directory, save_path)
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        torch.save({'embeddings': embedding_matrix.cpu(), 'labels': all_genre_labels}, final_path)

        print(f"\n Feature space saved to: {final_path}")
        print(f"Total processed songs (after merging fragments): {len(all_embeddings)}")

        return embedding_matrix, all_genre_labels










def plot_embeddings(embedding_matrix: np.ndarray, genre_labels: List[str],
    method: str = "tsne",  # options: "tsne", "umap", "pca"
    title: Optional[str] = None, save_path: Optional[str] = None):

    if embedding_matrix.shape[0] <= 1:
        print("Not enough embeddings to plot (need > 1).")
        return

    method = method.lower()
    if title is None:
        title = f"{method.upper()} of Song Embeddings"

    print(f"Running {method.upper()}...")

    # --- Dimensionality reduction ---
    if method == "tsne":
        reducer = TSNE(
            n_components=2,
            perplexity=min(30, embedding_matrix.shape[0] - 1),
            init='pca',
            n_iter=2500,
            random_state=42,
        )
    elif method == "umap":
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
    elif method == "pca":
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'tsne', 'umap', or 'pca'.")

    embeddings_2d = reducer.fit_transform(embedding_matrix)

    # --- Color mapping per genre ---
    unique_genres = sorted(list(set(genre_labels)))
    cmap = plt.get_cmap('tab20' if len(unique_genres) <= 20 else 'hsv')
    genre_to_color = {
        genre: cmap(i / len(unique_genres))
        for i, genre in enumerate(unique_genres)
    }
    point_colors = [genre_to_color[g] for g in genre_labels]

    # --- Plot ---
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=point_colors,
        alpha=0.8,
        s=60
    )

    # Legend
    handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=genre_to_color[genre],
               markersize=10)
        for genre in unique_genres
    ]
    plt.legend(handles, unique_genres, title="Genres", loc='best', fontsize=10)

    plt.title(title, fontsize=16)
    plt.xlabel(f"{method.upper()} Component 1", fontsize=12)
    plt.ylabel(f"{method.upper()} Component 2", fontsize=12)
    plt.grid(True)

    if save_path:
        print(f"Saving plot to {save_path}...")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()


def plot_embeddings_3d(
    embedding_matrix: np.ndarray,
    genre_labels: List[str],
    method: str = "tsne",  # options: "tsne", "umap", "pca"
    title: Optional[str] = None,
    save_dir: str = "plots_3d",
    rotation_speed: int = 2,     # degrees per frame for animation
    n_frames: int = 180,         # total frames in animation
    dpi: int = 150,              # resolution for saved files
):
    """
    Plot 3D embeddings of songs with color-coded genres, save both static image and rotating animation.
    """
    if embedding_matrix.shape[0] <= 1:
        print("Not enough embeddings to plot (need > 1).")
        return

    os.makedirs(save_dir, exist_ok=True)
    method = method.lower()
    if title is None:
        title = f"{method.upper()} 3D of Song Embeddings"

    print(f"Running {method.upper()} in 3D...")

    # --- Dimensionality reduction ---
    if method == "tsne":
        reducer = TSNE(
            n_components=3,
            perplexity=min(30, embedding_matrix.shape[0] - 1),
            init='pca',
            n_iter=2500,
            random_state=42,
        )
    elif method == "umap":
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
    elif method == "pca":
        reducer = PCA(n_components=3, random_state=42)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'tsne', 'umap', or 'pca'.")

    embeddings_3d = reducer.fit_transform(embedding_matrix)

    # --- Color mapping per genre ---
    unique_genres = sorted(list(set(genre_labels)))
    cmap = plt.get_cmap('tab20' if len(unique_genres) <= 20 else 'hsv')
    genre_to_color = {
        genre: cmap(i / len(unique_genres))
        for i, genre in enumerate(unique_genres)
    }
    point_colors = [genre_to_color[g] for g in genre_labels]

    # --- Create figure ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        embeddings_3d[:, 0],
        embeddings_3d[:, 1],
        embeddings_3d[:, 2],
        c=point_colors,
        s=60,
        alpha=0.85
    )

    ax.set_title(title, fontsize=15)
    ax.set_xlabel(f"{method.upper()} Component 1", fontsize=11)
    ax.set_ylabel(f"{method.upper()} Component 2", fontsize=11)
    ax.set_zlabel(f"{method.upper()} Component 3", fontsize=11)
    ax.grid(True)

    # --- Legend ---
    handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=genre_to_color[genre],
               markersize=9)
        for genre in unique_genres
    ]
    ax.legend(handles, unique_genres, title="Genres", loc='best', fontsize=9)

    # --- Save static image ---
    image_path = os.path.join(save_dir, f"{method}_3d_plot.png")
    plt.savefig(image_path, bbox_inches='tight', dpi=dpi)
    print(f"Saved static 3D plot to {image_path}")

    # --- Animation (rotation) ---
    def rotate(angle):
        ax.view_init(elev=30, azim=angle)
        return fig,

    anim = FuncAnimation(
        fig, rotate,
        frames=np.arange(0, 360, rotation_speed),
        interval=50,
        blit=False
    )

    video_path = os.path.join(save_dir, f"{method}_3d_rotation.mp4")
    anim.save(video_path, fps=30, dpi=dpi, extra_args=['-vcodec', 'libx264'])
    print(f"Saved rotating 3D animation to {video_path}")

    plt.show()