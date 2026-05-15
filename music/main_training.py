import warnings
from transformers.utils import logging as hf_logging
import os
import torch.nn as nn
from pytorch_metric_learning import losses

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

from utils.model_training import MusicGenreClassifier

GENRES_TO_PROCESS = ["Hip Hop Convencional", "Ambiente", "Rock", "Techno Y Trance", "Clásica", "Rnb", "Mundo", "Metal", "Alternativa", "Piano clasico"] 
# ["Hip Hop Convencional", "Ambiente", "Rock", "Techno Y Trance", "Clásica", "Rnb", "Mundo", "Metal"] 
# ["Electrónico", "Hip Hop Convencional", "Ambiente", "Rock", "Pop", "Jazz Clásico", "Techno Y Trance", 
# "Clásica", "Salsa", "Rnb", "Mundo", "Metal", "Alternativa", "Pop indie", "Piano clasico"]


MODEL_NAME = '/dataslow/storage/Experiments/INTERNS/anavarror/models/MERT-v1-95M'
# MODEL_NAME = '/dataslow/storage/Experiments/INTERNS/anavarror/models/m-a-p_music2vec-v1'
# MODEL_NAME = "/dataslow/storage/Experiments/INTERNS/anavarror/models/facebook_wav2vec2-base"

NAME_GENRES = ""
for genre in GENRES_TO_PROCESS:
        NAME_GENRES += genre + "_"

ONLY_NAME_MODEL = MODEL_NAME.split("/")[-1]

# ---------- PARAMETERS ---------
TYPE_MODEL = "mlp" # "svm" "logreg" "knn" "mlp" "mlp_grid"
N_COMPONENTS = 15 # PCA reduction

# Parameters of the MLP model
ATTENTION = False
PCA = False
CRITERION_CE = nn.CrossEntropyLoss()
CRITERION_SUPCON = losses.SupConLoss()

BATCH_SIZE = 1024 # 1024 512 256 128 64
LAMBDA = 0.2 # weight to SupCon loss over CE
DROPOUT = 0.4 # 0.6 0.4 0.2
LR = 0.001 # 0.001
EPOCHS = 100 #200 150 120



FEATURE_SPACE = f'/dataslow/storage/Experiments/INTERNS/anavarror/feature_spaces_1k_20s_contain_fragments/{ONLY_NAME_MODEL}_attention{ATTENTION}'


if __name__ == "__main__":
    classifier = MusicGenreClassifier(
        model_name=MODEL_NAME,
        feature_space_path = FEATURE_SPACE,
        classifier_type = TYPE_MODEL,   
        use_pca=PCA,
        pca_components = N_COMPONENTS,
        attention = ATTENTION,
        genre_list = GENRES_TO_PROCESS,
    )

    #  ==================== CROSS-ENTROPY LOSS ======================
    # classifier.train_classifier_ce(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, criterion=CRITERION_CE)

    # ===================== CROSS-ENTROPY LOSS + SUPERVISED CONTRASTIVE LOSS ==================================
    # classifier.train_classifier_supcon(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, criterion_ce=CRITERION_CE, criterion_supcon=CRITERION_SUPCON, alpha=LAMBDA)

    # ===================== HALO LOSS ===================
    classifier.train_classifier_halo(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, dropout=DROPOUT)


    