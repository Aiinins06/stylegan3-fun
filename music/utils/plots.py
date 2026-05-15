import os
import torch
from music.utils.notused.embedding_meanfrag import plot_embeddings, plot_embeddings_3d


FEATURE_SPACE_PATH = "/home/anavarror/feature_spaces_1k_under90/facebook_wav2vec2-base_meanfrag_Hip Hop Convencional_Ambiente_Rock_Jazz Clásico_.pt"

# For naming & saving plots
only_name_model = FEATURE_SPACE_PATH.split("/")[-1].split("_meanfrag")[0]


GENRES_LIST_PIXABAY = ["Hip Hop Convencional", "Ambiente", "Rock", "Jazz Clásico"]
NAME_GENRES = "".join([g + "_" for g in GENRES_LIST_PIXABAY])

# Create plots folder
PLOTS_DIRECTORY = f"/datafast/105-1/Datasets/INTERNS/anavarror/tracks_under90s_pixabay/plots_{NAME_GENRES}"
os.makedirs(PLOTS_DIRECTORY, exist_ok=True)


if __name__ == "__main__":

    print(f"Loading feature space from:\n  {FEATURE_SPACE_PATH}\n")
    data = torch.load(FEATURE_SPACE_PATH, map_location="cpu")

    embeddings = data["embeddings"]
    labels = data["labels"]

    print(f"Loaded embeddings: {embeddings.shape}")
    print(f"Loaded labels: {len(labels)} songs\n")

    # Convert to numpy for plotting
    embedding_matrix_cpu = embeddings.numpy()

    # Plot using different projections
    methods = ["tsne", "pca", "umap"]

    for meth in methods:
        plot_filename = f"{only_name_model}_{meth}_meanfrag_{NAME_GENRES}"
        plot_save = os.path.join(PLOTS_DIRECTORY, plot_filename)

        plot_title = f"{only_name_model} {meth.upper()} Visualization of Song Embeddings"

        # ---- 2D plot ----
        plot_embeddings(
            embedding_matrix=embedding_matrix_cpu,
            genre_labels=labels,
            method=meth,
            title=plot_title,
            save_path=plot_save + ".png"
        )

        # ---- 3D plot ----
        plot_embeddings_3d(
            embedding_matrix_cpu,
            genre_labels=labels,
            method=meth,
            title=plot_title,
            save_dir=plot_save + "_3d",
            rotation_speed=2,
        )
