import warnings
from transformers.utils import logging as hf_logging
import os
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
import torch

# from music.utils.notused.embedding_meanfrag import plot_embeddings, plot_embeddings_3d
from utils.embedding_attentionsolved import MusicEmbedder

GENRES_LIST_PIXABAY = ["Piano clasico"]
# ["Electrónico", "Hip Hop Convencional", "Ambiente", "Rock", "Pop", "Jazz Clásico", "Techno Y Trance", "Clásica", 
# "Salsa", "Rnb", "Mundo", "Metal", "Alternativa", "Pop indie", "Piano clasico"]
GENRES_TO_PROCESS = GENRES_LIST_PIXABAY

# WHERE ARE THE SONG FILES STORED
BASE_DIRECTORY = '/datafast/105-1/Datasets/INTERNS/anavarror/songs' 

NAME_GENRES = ""
for genre in GENRES_TO_PROCESS:
    NAME_GENRES += genre + "_"


MODEL_NAME = "/dataslow/storage/Experiments/INTERNS/anavarror/models/MERT-v1-95M"
# MODEL_NAME = "/dataslow/storage/Experiments/INTERNS/anavarror/models/m-a-p_music2vec-v1"
# MODEL_NAME = "/dataslow/storage/Experiments/INTERNS/anavarror/models/facebook_wav2vec2-base"

ONLY_NAME_MODEL = MODEL_NAME.split("/")[-1]


ATTENTION = False

# Directory where the songs per genre are stored
SAVE_DIR = f'/dataslow/storage/Experiments/INTERNS/anavarror/feature_spaces_1k_20s_contain_fragments/{ONLY_NAME_MODEL}_attention{ATTENTION}'
os.makedirs(SAVE_DIR, exist_ok=True)



if __name__ == "__main__":
    name_genres = ""
    for genre in GENRES_TO_PROCESS:
        name_genres += genre + "_"

    embedder = MusicEmbedder(model_name=MODEL_NAME, attention=ATTENTION)
    
    embeddings, labels = embedder.process_genres(
        base_directory=BASE_DIRECTORY,
        genre_list=GENRES_TO_PROCESS,
        max_songs = None,
        save_dir=SAVE_DIR
    )