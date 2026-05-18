import os
import numpy as np

from utils.model_training import MusicGenreClassifier



MODEL_NAME = '/dataslow/storage/Experiments/INTERNS/anavarror/models/MERT-v1-95M'
GENRES_TO_PROCESS = ["Hip Hop Convencional", "Ambiente", "Rock", "Techno Y Trance", "Clásica", "Rnb", "Mundo", "Metal", "Alternativa", "Piano clasico"] 
NAME_GENRES = ""
for genre in GENRES_TO_PROCESS:
        NAME_GENRES += genre + "_"

ONLY_NAME_MODEL = MODEL_NAME.split("/")[-1]

ATTENTION = False

FEATURE_SPACE = f'/dataslow/storage/Experiments/INTERNS/anavarror/feature_spaces_1k_20s_contain_fragments/{ONLY_NAME_MODEL}_attention{ATTENTION}'
NAME_CLASSIFIER = "/dataslow/storage/Experiments/INTERNS/anavarror/trainedmodels_containfrag_halo/" \
    "MERT-v1-95M_attentionFalse_pcaFalse_batch1024_lr0.001_dropout0.4_Hip Hop Convencional_Ambiente_Rock_Techno Y Trance_Clásica_Rnb_Mundo_Metal_Pop indie_Piano clasico_/" \
    "MERT-v1-95M_epochs120_lr0.001_batch1024.joblib"


if __name__ == "__main__":
    classifier = MusicGenreClassifier(
        model_name=MODEL_NAME,
        feature_space_path = FEATURE_SPACE,
        classifier_path= NAME_CLASSIFIER,
        attention = ATTENTION,
        genre_list = GENRES_TO_PROCESS,
    )


    classifier.load_classifier()

    model = classifier.classifier.model
    centroids = model.centroids.detach().cpu().numpy()

    centroids_dict = {}
    label_classes = classifier.label_encoder.classes_

    for idx, genre_name in enumerate(label_classes):
          centroids_dict[genre_name] = centroids[idx]
          print(f"Mapping: {genre_name} to vector shape {centroids_dict[genre_name].shape}")


    os.makedirs("/dataslow/storage/Experiments/INTERNS/anavarror/gans", exist_ok=True)
    np.save("/dataslow/storage/Experiments/INTERNS/anavarror/gans/musical_centroids.npy", centroids_dict)



    