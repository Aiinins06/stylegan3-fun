import warnings
from transformers.utils import logging as hf_logging
import os
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

from music_classifier_meanfrag import MusicGenreClassifier



MODEL_NAME = "/datafast/105-1/Datasets/INTERNS/anavarror/models/facebook_wav2vec2-base"
FEATURE_SPACE = "/home/anavarror/feature_spaces_1k_under90/facebook_wav2vec2-base_meanfrag_Electrónico_Hip Hop Convencional_Ambiente_.pt"


TYPE_MODEL = "knn"

if __name__ == "__main__":
    only_name_model = MODEL_NAME.split("/")[-1]

    
    classifier_dir = "trained_models_under90_1k"
    os.makedirs(classifier_dir, exist_ok=True)  
    CLASSIFIER = f"trained_models_under90_1k/studies/pca_{only_name_model}_{TYPE_MODEL}_.joblib"

    classifier = MusicGenreClassifier(
        model_name=MODEL_NAME,
        feature_space_path = FEATURE_SPACE,
        classifier_path = CLASSIFIER,
        classifier_type = TYPE_MODEL
    )

    X, _ = classifier.load_feature_space()

    classifier.plot_pca_variance(X=X, save_dir="trained_models_under90_1k/studies/", max_components=50)

    # K-Means elbow (true elbow)
    # res_kmeans = classifier.elbow_method_kmeans(max_k=50, save_path="trained_models_under90_1k/elbow/kmeans_elbow.png", compute_silhouette=True)

    # KNN elbow (accuracy vs neighbors)
    # res_knn = classifier.elbow_method_knn(max_k=75, save_path="trained_models_under90_1k/elbow/knn_elbow.png")
