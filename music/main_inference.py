import warnings
from transformers.utils import logging as hf_logging
import os

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

from utils.model_training import MusicGenreClassifier
# from utils.check_output_model import check_accuracy, get_ground_truth

GENRES_TO_PROCESS = ["Hip Hop Convencional", "Ambiente", "Rock", "Techno Y Trance", "Clásica", "Rnb", "Mundo", "Metal", "Alternativa", "Piano clasico"]
# ["Hip Hop Convencional", "Ambiente", "Rock", "Techno Y Trance", "Clásica", "Rnb", "Mundo", "Metal"] 
# ["Electrónico", "Hip Hop Convencional", "Ambiente", "Rock", "Pop", "Jazz Clásico", "Techno Y Trance", 
# "Clásica", "Salsa", "Rnb", "Mundo", "Metal", "Pop indie", "Alternativa", "Piano clasico"]


MODEL_NAME = '/dataslow/storage/Experiments/INTERNS/anavarror/models/MERT-v1-95M'
# MODEL_NAME = '/dataslow/storage/Experiments/INTERNS/anavarror/models/m-a-p_music2vec-v1'
# MODEL_NAME = "/dataslow/storage/Experiments/INTERNS/anavarror/models/facebook_wav2vec2-base"

NAME_GENRES = ""
for genre in GENRES_TO_PROCESS:
        NAME_GENRES += genre + "_"

ONLY_NAME_MODEL = MODEL_NAME.split("/")[-1]

# ---------- PARAMETERS ---------
N_COMPONENTS = 15
ATTENTION = False
PCA = False
EPOCHS = 120
TYPE_MODEL = "mlp" # "svm" "logreg" "knn" "mlp"
BATCH_SIZE = 256
LR = 0.001
DROPOUT = 0.6

MODE = 'halo'

# DIRECTORY
FEATURE_SPACE = f'/dataslow/storage/Experiments/INTERNS/anavarror/feature_spaces_1k_20s_contain_fragments/{ONLY_NAME_MODEL}_attention{ATTENTION}'
# FILE
# FEATURE_SPACE = f'/datafast/105-1/Datasets/INTERNS/anavarror/feature_spaces_1k_20s_contain_fragments/{ONLY_NAME_MODEL}_attention{ATTENTION}_{NAME_GENRES}.pt'


# TEST_SONGS_FOLDER = "/datafast/105-1/Datasets/INTERNS/anavarror/test/hiphop_ambiente_rock_techno" 
TEST_SONGS_FOLDER = "/datafast/105-1/Datasets/INTERNS/anavarror/test" 

# GT_SONGS = "/datafast/105-1/Datasets/INTERNS/anavarror/test/songs_hiphop_ambiente_rock_techno"

BASE_DIR = f"/datafast/105-1/Datasets/INTERNS/anavarror/test/results_1k_20s/containfragments_{MODE}"
os.makedirs(BASE_DIR, exist_ok=True)
prefix = f"{ONLY_NAME_MODEL}_attention{ATTENTION}_pca{PCA}_dropout{DROPOUT}_epochs{EPOCHS}_{NAME_GENRES}" 
RESULTS_BASE_DIR = os.path.join(BASE_DIR, prefix)
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)



# NAME_CLASSIFIER = "/dataslow/storage/Experiments/INTERNS/anavarror/trainedmodels_containfrag_halo/" \
#     "MERT-v1-95M_attentionFalse_pcaFalse_batch512_lr0.001_dropout0.6_Hip Hop Convencional_Ambiente_Rock_Techno Y Trance_Clásica_Rnb_Mundo_Metal_/" \
#     "MERT-v1-95M_epochs120_lr0.001_batch512.joblib"

NAME_CLASSIFIER = "/dataslow/storage/Experiments/INTERNS/anavarror/trainedmodels_containfrag_halo/" \
    "MERT-v1-95M_attentionFalse_pcaFalse_batch1024_lr0.001_dropout0.4_" \
    "Hip Hop Convencional_Ambiente_Rock_Techno Y Trance_Clásica_Rnb_Mundo_Metal_Alternativa_Piano clasico_/" \
    "MERT-v1-95M_epochs120_lr0.001_batch1024.joblib"


if __name__ == "__main__":   
    classifier = MusicGenreClassifier(
        model_name=MODEL_NAME,
        feature_space_path = FEATURE_SPACE,
        classifier_path = NAME_CLASSIFIER,
        classifier_type = TYPE_MODEL,
        use_pca = PCA, 
        pca_components = N_COMPONENTS,
        attention = ATTENTION,
        genre_list = GENRES_TO_PROCESS,
    ) 
    
    mp3_files = [f for f in os.listdir(TEST_SONGS_FOLDER) if f.endswith(".mp3")]
    
    for counter, song_file in enumerate(mp3_files, start=1):
        path_song = os.path.join(TEST_SONGS_FOLDER, song_file)
        print("\n", "-" * 54)
        print(f"Processing song {counter} of test songs: {song_file}")
        
        result = classifier.predict_genre(mp3_path=path_song, mode=MODE, save_viz_dir=RESULTS_BASE_DIR)
        
        print(f"\nPrediction time: {result['inference_time']:.3f} seconds")
        print(result["label"])
        # print(result["probabilities"])
        # print(result["embedding"].shape)
        
        classifier.save_prediction_result(mp3_path=song_file, result=result, save_dir=RESULTS_BASE_DIR)
    
    full_csv_path = os.path.join(RESULTS_BASE_DIR, "results.csv")

    
    # print("\n--- STARTING VALIDATION ---")
    # check_accuracy(csv_file_path=full_csv_path, ground_truth_dir=GT_SONGS)
