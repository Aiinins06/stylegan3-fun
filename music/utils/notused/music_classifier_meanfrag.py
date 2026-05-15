import os
import torch
import librosa
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import re
import csv
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, Normalizer, StandardScaler
from sklearn.metrics import classification_report
from transformers import AutoFeatureExtractor, AutoModel
from scipy.special import softmax
from typing import Optional
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV


GENRES_LIST_PIXABAY = ["Hip Hop Convencional", "Ambiente", "Rock", "Jazz Clásico"] # ==================== CAMBIAR DEPENDIENDO DE RUN ===========================

NAME_GENRES = ""
for genre in GENRES_LIST_PIXABAY:
        NAME_GENRES += genre + "_"


import torch
import torch.nn as nn
import torch.optim as optim

class GPU_MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(GPU_MLP, self).__init__()
        # Matches your sklearn: hidden_layer_sizes=(512, 256, 128)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2), # Optional: helps prevent overfitting
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)


class MusicGenreClassifier:
    def __init__(
        self,
        model_name: str,
        feature_space_path: str,
        classifier_path: str,
        classifier_type: str,
        max_segment_duration: int = 90,
        use_pca: bool = False,
        pca_components: int = 15,
    ):
        self.model_name = model_name
        self.feature_space_path = feature_space_path
        self.classifier_path = classifier_path
        self.classifier_type = classifier_type.lower()
        self.max_segment_duration = max_segment_duration
        self.only_name_model =  model_name.split("/")[-1]
        self.use_pca = use_pca
        self.pca_components = pca_components

        self.pca = None
        self.scaler = None # Normalizador L2
        self.std_scaler = None # StandardScaler (opcional, bueno para MLP)
        self.classifier = None
        self.label_encoder = None


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Core audio model
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True
        ).to(self.device)
        self.target_sample_rate = self.feature_extractor.sampling_rate

        # Load classifier if available
        if os.path.exists(self.classifier_path):
            self.load_classifier()
        else:
            self.classifier = None
            self.label_encoder = None

    def plot_trained_decision_space(self, embedding_matrix, true_labels, method, save_path):
        if self.classifier is None or self.label_encoder is None:
            self.load_classifier()

        # Predict labels
        y_pred_idx = self.classifier.predict(embedding_matrix)
        pred_labels = self.label_encoder.inverse_transform(y_pred_idx)

        # Dimensionality reduction
        method = method.lower()
        if method == "tsne":
            reducer = TSNE(n_components=2, perplexity=min(30, embedding_matrix.shape[0]-1), init="pca", random_state=42)
            points = reducer.fit_transform(embedding_matrix)
        elif method == "umap":
            import umap
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
            points = reducer.fit_transform(embedding_matrix)
        elif method == "pca":
            reducer = PCA(n_components=2, random_state=42)
            points = reducer.fit_transform(embedding_matrix)
        else:
            raise ValueError("Unknown projection method.")

        # Color by predicted label
        unique_preds = sorted(list(set(pred_labels)))
        cmap = plt.get_cmap('tab20' if len(unique_preds) <= 20 else 'hsv')
        color_map = {label: cmap(i / len(unique_preds)) for i, label in enumerate(unique_preds)}
        point_colors = [color_map[label] for label in pred_labels]

        plt.figure(figsize=(12, 8))
        plt.scatter(points[:, 0], points[:, 1], c=point_colors, s=60, alpha=0.85)

        handles = [
            Line2D([0], [0], marker='o', color='w',
                markerfacecolor=color_map[label],
                markersize=10)
            for label in unique_preds
        ]
        plt.legend(handles, unique_preds, title="Predicted Genres", loc='best')
        plt.title(f"{method.upper()} — Classifier Decision Space")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True)

        if save_path:
            # os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved trained decision plot at: {save_path}")

        plt.show()


    def _get_base_name(self, filename: str):
        """Remove extension + _partN suffix to identify grouped fragments."""
        name = os.path.splitext(filename)[0]
        name = re.sub(r"_part\d+$", "", name)
        return name

    def _build_classifier(self):
        if self.classifier_type == "logreg":
            return LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="auto")
        elif self.classifier_type == "svm":
            return SVC(kernel="rbf", probability=True)
        elif self.classifier_type == "knn":
            return KNeighborsClassifier(n_neighbors=29) # selected 29 after seeing the plot of knn_elbow.png
        elif self.classifier_type == "mlp":
            # return MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu', solver='adam', learning_rate_init=0.001, max_iter=500)
            # return MLPClassifier(hidden_layer_sizes=(256,), activation='relu', solver='lbfgs', learning_rate_init=0.0001, max_iter=1000)
            return MLPClassifier(
                        hidden_layer_sizes=(512, 256), activation='relu', solver='adam', 
                        batch_size=32, learning_rate_init=0.0005, max_iter=1000, early_stopping=True            
                    ) # TRYING FOR IMPROVEMENT -- another configuration for MLP
        
        elif self.classifier_type == "mlp_grid":
            return None
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")


    def get_song_embedding(self, mp3_path: str):
        try:
            audio, _ = librosa.load(mp3_path, sr=self.target_sample_rate, mono=True)
            segment_len = int(self.max_segment_duration * self.target_sample_rate)
            segments = [audio[i:i + segment_len] for i in range(0, len(audio), segment_len)]

            embeddings = []
            for seg in segments:
                #  Normalization of audio -- TRYING FOR IMPROVEMENT
                seg = (seg - seg.mean()) / (seg.std() + 1e-7)

                inputs = self.feature_extractor(seg, sampling_rate=self.target_sample_rate, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)

                    if "MERT" in self.model_name or "music2vec" in self.model_name: # TRYING FOR IMPROVEMENT -- taking the last 4 layers
                        # Stack of all the hidden states: (n_layers, batch, time, dim)
                        hidden_states = outputs.hidden_states
                        
                        last_n_layers = torch.stack(hidden_states[-4:]) # Last 4 layers
                        cat_hidden = torch.mean(last_n_layers, dim=0)   # Mean layers -> (batch, time, dim)
                        
                        emb = cat_hidden.mean(dim=1).squeeze().cpu() # temporal mean
                        
                    else: 
                        # for wav2vec2 base
                        emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu()

                embeddings.append(emb)

            if not embeddings:
                return None

            return torch.stack(embeddings).mean(dim=0).numpy()

        except Exception as e:
            print(f"   Error processing {mp3_path}: {e}")
            return None


    def load_feature_space(self):
        data = torch.load(self.feature_space_path, map_location="cpu")
        return data["embeddings"].cpu().numpy(), data["labels"]


    def train_classifier(self):
        if self.classifier_type == "mlp_grid":
            return self.train_classifier_gridsearch() 

        else:
            # Loading data
            X, labels = self.load_feature_space()
            le = LabelEncoder()
            y = le.fit_transform(labels)
            print(f"Original shape: {X.shape}")

            # Applying L2 Normalization -- TRYING FOR IMPROVEMENT
            scaler = Normalizer(norm='l2') 
            X = scaler.fit_transform(X)

            # Applying PCA reduction before the split to have only useful information data -- TRYING FOR IMPROVEMENT
            if self.use_pca:
                print(f"Applying PCA reduction to {self.pca_components} components...")
                self.pca = PCA(n_components=self.pca_components, random_state=42)
                X = self.pca.fit_transform(X)
                print(f"Shape after PCA: {X.shape}")
            else:
                print("Skipping PCA. Using full raw embeddings.")
                self.std_scaler = StandardScaler()
                X = self.std_scaler.fit_transform(X)

            # Split the data into training and validation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

            # Build classifier model
            model = self._build_classifier()
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            print(classification_report(y_test, y_pred, target_names=le.classes_))

            # Save classifier model in a joblib file for later inference
            save_obj = {"model": model, "label_encoder": le, "scaler": self.scaler, "std_scaler": self.std_scaler, "pca": self.pca, "pca_components": self.pca_components}
            joblib.dump(save_obj, self.classifier_path)
            print(f"  Saved classifier ({self.classifier_type}) to {self.classifier_path}")

            # Saving a plot with the different songs classified in the feature space
            methods = ["umap", "pca"]
            directory_plot = f'/datafast/105-1/Datasets/INTERNS/anavarror/trained_models_under90_1k'
            os.makedirs(directory_plot, exist_ok=True)

            for meth in methods:
                name_file = f'{self.only_name_model}_{self.classifier_type}_{meth}_pca15_{NAME_GENRES}.png' #  ================== CONTINUE CHANGING IT ACCORDING TO THE RUN ======================================
                complete = os.path.join(directory_plot, name_file)
                self.plot_trained_decision_space(X, labels, method="umap", save_path=complete) 
                self.plot_trained_decision_space(X, labels, method="pca", save_path=complete) 

            self.classifier, self.label_encoder = model, le


    def load_classifier(self):
        data = joblib.load(self.classifier_path)
        self.classifier = data["model"]
        self.label_encoder = data["label_encoder"]
        self.scaler = data.get("scaler", None)
        self.std_scaler = data.get("std_scaler", None)
        self.pca = data.get("pca", None)
        self.use_pca = data.get("use_pca", False)
        print(f"Loaded classifier from {self.classifier_path}")


    # ==== INFERENCE TIME ====
    def predict_genre(self, mp3_path: str):
        if self.classifier is None:
            raise RuntimeError("Classifier not trained or loaded.")

        start_time = time.time()

        directory = os.path.dirname(mp3_path)
        filename = os.path.basename(mp3_path)

        base_name = self._get_base_name(filename)

        # Extract embedding
        final_embedding = self.get_song_embedding(mp3_path)
        if final_embedding is None:
            return None

        # Reshape to 2D (1 sample, N features)
        # This converts shape (768,) -> (1, 768)
        final_embedding = final_embedding.reshape(1, -1)

        # Apply Transformations (Scaler -> PCA)
        if self.scaler:
            final_embedding = self.scaler.transform(final_embedding)
        
        if self.use_pca:
            if self.pca is None:
                raise ValueError("   Critical Error: use_pca=True but self.pca is None.")
            final_embedding = self.pca.transform(final_embedding) # (1, 768) -> (1, 15)
        elif self.std_scaler:
            final_embedding = self.std_scaler.transform(final_embedding)

        # Predict
        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(final_embedding)[0]
        else:
            logits = self.classifier.decision_function(final_embedding)[0]
            probs = softmax(logits)

        pred_idx = np.argmax(probs)
        pred_label = self.label_encoder.inverse_transform([pred_idx])[0]

        return {
            "label": pred_label,
            "probabilities": dict(zip(self.label_encoder.classes_, probs)),
            "embedding": final_embedding,
            "inference_time": time.time() - start_time
        }

    def save_prediction_result(self, mp3_path: str, result: dict, save_dir: str):
        """
        Saves:
          - CSV with song name, predicted label and probability distribution
          - NPY file with embedding vector
        """
        os.makedirs(save_dir, exist_ok=True)

        song_name = os.path.splitext(os.path.basename(mp3_path))[0]

        # --- Save probabilities + label ---
        csv_filename = f'results.csv'
        # csv_path = os.path.join(final_folder, csv_filename)
        csv_path = os.path.join(save_dir, csv_filename)

        if not hasattr(self, "_results_cleared"):
            if os.path.exists(csv_path):
                # print(f"[INFO] Old results.csv found — deleting it.")
                os.remove(csv_path)
            self._results_cleared = True

        file_exists = os.path.isfile(csv_path)

        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)

            if not file_exists:
                # Write header only once
                header = ["song_name", "predicted_label"] + list(result["probabilities"].keys())
                writer.writerow(header)

            probabilities = [f"{value:.4f}" for value in result["probabilities"].values()]        
            row = [song_name, result["label"]] + probabilities
            writer.writerow(row)

        # --- Save embedding vector ---
        npy_filename = f'{self.classifier_type}_embedding_{mp3_path}.npy'
        emb_path = os.path.join(save_dir, npy_filename)
        np.save(emb_path, result["embedding"])

        print(f"  Saved prediction: {song_name}")
        print(f"   → Summary updated in: {csv_path}")
        print(f"   → Embedding saved at: {emb_path}")


    def _save_gridsearch_results(self, grid):
        results = pd.DataFrame(grid.cv_results_)

        # --- File paths ---
        base_dir = f"/datafast/105-1/Datasets/INTERNS/anavarror/trained_models_under90_1k"
        os.makedirs(base_dir, exist_ok=True)

        all_results_path = os.path.join(
            base_dir,
            f"{self.only_name_model}_mlp_allresults_{NAME_GENRES}.csv"
        )

        top_results_path = os.path.join(
            base_dir,
            f"{self.only_name_model}_mlp_topconfigs_{NAME_GENRES}.csv"
        )

        # --- Save all results ---
        results.to_csv(all_results_path, index=False)
        print(f"  Saved ALL GridSearch results to:  {all_results_path}")

        # --- Sort by accuracy and extract top configurations ---
        results_sorted = results.sort_values("mean_test_score", ascending=False)

        # top-3 configs (or fewer if grid < 3)
        top_k = results_sorted.head(3)
        top_k.to_csv(top_results_path, index=False)

        print(f"  Saved TOP configurations to: {top_results_path}")

        return all_results_path, top_results_path

    def train_classifier_gridsearch(self):
        print("  Starting Grid Search for MLPClassifier...")

        # Load embeddings
        X, labels = self.load_feature_space()
        le = LabelEncoder()
        y = le.fit_transform(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Base model
        mlp = MLPClassifier(max_iter=1000)

        # --- Hyperparameter grid ---
        param_grid = {
            "hidden_layer_sizes": [
                (256,), (384,), (512,), (768,),
                (512, 256), (384, 192), (256, 128),
                (512, 256, 128), (384, 192, 96)
            ],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001, 0.01, 0.1],
            "solver": ["adam", "lbfgs"],
            "learning_rate_init": [0.001, 0.0001],
            "batch_size": [64, 128],
            "early_stopping": [True],
            "validation_fraction": [0.1, 0.2],
        }

        grid = GridSearchCV(
            estimator=mlp,
            param_grid=param_grid,
            scoring="accuracy",
            n_jobs=-1,
            cv=3,
            verbose=2,
        )

        grid.fit(X_train, y_train)

        self._save_gridsearch_results(grid)

        # print("\n==============================")
        print("  BEST MLP PARAMETERS FOUND:")
        print(grid.best_params_)
        # print("==============================\n")


        self.classifier = grid.best_estimator_
        self.label_encoder = le

    def plot_pca_variance(self, X, save_dir, max_components=50):
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)

        max_components = min(max_components, X.shape[1])  

        pca = PCA(n_components=max_components)
        pca.fit(X)

        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        # === Plot ===
        plt.figure(figsize=(10, 6))

        # Varianza por componente
        plt.bar(
            range(1, max_components + 1),
            explained_variance,
            alpha=0.6,
            label="Explainable variance per component"
        )

        # File path for saving
        output_path = os.path.join(save_dir, "pca_variance_plot.png")


        # Varianza acumulada
        plt.plot(
            range(1, max_components + 1),
            cumulative_variance,
            marker='o',
            linewidth=2,
            label="Acumulated variance"
        )

        plt.xlabel("Number of components PCA")
        plt.ylabel("Explainable variance")
        plt.title("PCA study:Explainable variance vs number of components")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        
        print(f"  PCA plot saved at: {output_path}")

        return explained_variance, cumulative_variance

    def _ensure_embeddings_loaded(self):
        """
        Load embeddings and labels once and cache them (used by elbow methods).
        Returns X, labels
        """
        if self._cached_embeddings is None or self._cached_labels is None:
            X, labels = self.load_feature_space()
            self._cached_embeddings = X
            self._cached_labels = labels
        return self._cached_embeddings, self._cached_labels

    def elbow_method_kmeans(self, max_k: int = 15, save_path: Optional[str] = None, compute_silhouette: bool = False):
        """
        TRUE elbow method for K-Means: compute WCSS (inertia) for k=1..max_k.
        Optionally compute silhouette score for k>=2 to help with selection.
        Returns a dict with 'ks', 'wcss', and optionally 'silhouette'.
        """
        print("Running K-Means elbow on the embedding space...")
        X, _ = self._ensure_embeddings_loaded()

        wcss = []
        silhouettes = []

        # Ensure max_k is not larger than number of samples
        max_k = min(max_k, max(1, X.shape[0] - 1))

        for k in range(1, max_k + 1):
            # n_init set to 10 for wide compatibility
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
            if compute_silhouette and k >= 2:
                try:
                    labels_k = kmeans.labels_
                    sil = silhouette_score(X, labels_k)
                    silhouettes.append(sil)
                except Exception as e:
                    silhouettes.append(np.nan)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_k + 1), wcss, marker='o')
        plt.title("K-Means Elbow Method (WCSS vs k)")
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved K-Means elbow plot at: {save_path}")

        plt.show()

        if compute_silhouette:
            # Plot silhouette if requested
            ks_for_sil = list(range(2, max_k + 1))
            plt.figure(figsize=(10, 6))
            plt.plot(ks_for_sil, silhouettes, marker='o')
            plt.title("Silhouette Score vs k (K-Means)")
            plt.xlabel("Number of clusters (k)")
            plt.ylabel("Silhouette Score")
            plt.grid(True)
            if save_path:
                base, ext = os.path.splitext(save_path) if save_path else ("kmeans_elbow", ".png")
                sil_path = f"{base}_silhouette{ext}"
                plt.savefig(sil_path, dpi=300, bbox_inches="tight")
                print(f"Saved silhouette plot at: {sil_path}")
            plt.show()

        result = {"ks": list(range(1, max_k + 1)), "wcss": wcss}
        if compute_silhouette:
            result["silhouette"] = [None] + silhouettes  # pad to align indices (k=1 has None)
        return result

    def elbow_method_knn(self, max_k: int = 40, save_path: Optional[str] = None):
        """
        KNN 'elbow' style curve: compute validation accuracy for k = 1..max_k neighbors.
        Returns a dict with 'ks' and 'accuracies'.
        """
        print("Running KNN elbow (accuracy vs n_neighbors)...")
        X, labels = self._ensure_embeddings_loaded()
        le = LabelEncoder()
        y = le.fit_transform(labels)

        # Ensure max_k isn't larger than number of training samples
        max_possible_k = max(1, X.shape[0] - 1)
        max_k = min(max_k, max_possible_k)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        accuracies = []
        ks = list(range(1, max_k + 1))

        for k in ks:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            accuracies.append(acc)

        plt.figure(figsize=(10, 6))
        plt.plot(ks, accuracies, marker='o')
        plt.title("KNN Hyperparameter Elbow (Validation Accuracy vs n_neighbors)")
        plt.xlabel("n_neighbors (k)")
        plt.ylabel("Validation Accuracy")
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved KNN elbow plot at: {save_path}")

        plt.show()

        return {"ks": ks, "accuracies": accuracies}
