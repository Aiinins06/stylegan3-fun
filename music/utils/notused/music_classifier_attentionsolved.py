import os
import torch
import librosa
import numpy as np
import pandas as pd
import joblib
import csv
import time
import gc
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoFeatureExtractor, AutoModel
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
import torch.nn as nn
import torch.optim as optim
import unicodedata
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis   
import torch.nn.functional as F
from pytorch_metric_learning import losses

from utils.attention import SelfAttentionPooling 


def calculate_linear_separability(embeddings, labels, max_samples=50000):
    """
    Adaptation of Linear Separability metric of StyleGAN to music genres
    Compute Conditional Entropy H(Y|X) using linear SVMs One-vs-Rest
    """
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()
    
    # If 3D (N, T, D) reduce to 2D by averaging over time
    if embeddings.ndim == 3:
        embeddings = embeddings.mean(axis=1)

    
    if len(embeddings) > max_samples:
        embeddings, _, labels, _ = train_test_split(
            embeddings, labels, train_size=max_samples, stratify=labels, random_state=42
        )

    labels = np.array(labels)
    unique_genres = np.unique(labels)
    total_samples = len(labels)
    
    entropy_scores = {}
    total_entropy = 0.0

    print(f"\n[Separability Metric] Calculating for {len(unique_genres)} genres using Linear SVMs...")

    # Loop One-vs-Rest
    for genre in unique_genres:
        # 1 if actual genre, 0 if not
        binary_labels = (labels == genre).astype(int)
        
        # Train SVM
        svm = LinearSVC(random_state=42, max_iter=2000, dual=False)
        svm.fit(embeddings, binary_labels)
        predictions = svm.predict(embeddings)
        
        #  Computing Conditional Entropy H(Y|X)
        h_y_given_x = 0.0
        
        for x_val in [0, 1]:
            # Index where SVM predicted x_val
            idx_x = (predictions == x_val)
            count_x = np.sum(idx_x)
            
            if count_x == 0:
                continue
                
            p_x = count_x / total_samples
            
            # How many x_val were correct
            true_y_given_x = binary_labels[idx_x]
            counts_y = Counter(true_y_given_x)
            
            entropy_x = 0.0
            for y_val in [0, 1]:
                p_y_given_x = counts_y[y_val] / count_x
                if p_y_given_x > 0:
                    entropy_x -= p_y_given_x * np.log2(p_y_given_x)
                    
            h_y_given_x += p_x * entropy_x
            
        entropy_scores[genre] = h_y_given_x
        total_entropy += h_y_given_x
        print(f"  - {genre} vs Rest | H(Y|X) = {h_y_given_x:.4f}")

    # Like StyleGAN, applying exponent to pass from logarithmic to linear
    separability_score = np.exp(total_entropy)
    
    print("-" * 40)
    print(f"Total Conditional Entropy (logarithmic): {total_entropy:.4f}")
    print(f"FINAL SEPARABILITY SCORE (linear): {separability_score:.4f} (Lower is better)")
    
    return separability_score, entropy_scores

# -----------------------------------------------------------------------------
# MODEL DEFINITIONS
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)
    
class MLP_Attention(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP_Attention, self).__init__()
        
        self.attention_pool = SelfAttentionPooling(input_dim)
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x input shape: (batch, time_steps, input_dim)
        pooled_x = self.attention_pool(x) # -> (batch, input_dim)
        return self.network(pooled_x)
    
class MLP_SupCon(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP_SupCon, self).__init__()
        
        # 1. The Encoder: Everything up to the 128-dim feature space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # 2. The Projection Head
        self.projection_head = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # 3. The Classification Head: Just the final linear layer
        self.classifier_head = nn.Linear(128, num_classes)

    def forward(self, x):
        # Pass input through the encoder
        features = self.encoder(x)
        
        # Projection and L2 Normalization for SupCon
        projected = self.projection_head(features)
        normalized_features = F.normalize(features, p=2, dim=1)
        
        # Final prediction for CrossEntropy
        logits = self.classifier_head(features) 
        
        return logits, normalized_features

class MLP_Attention_SupCon(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.attention_pool = SelfAttentionPooling(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.head = nn.Linear(128, num_classes)
 
    def forward(self, x):
        # x: (batch, time_steps, input_dim)
        pooled = self.attention_pool(x)                          # (batch, input_dim)
        features = self.encoder(pooled)                          # (batch, 128)
        normalized_features = F.normalize(features, p=2, dim=1) # (batch, 128)
        logits = self.head(features)                             # (batch, num_classes)
        return logits, normalized_features

# -----------------------------------------------------------------------------
# SKLEARN-COMPATIBLE WRAPPERS
# -----------------------------------------------------------------------------
# This wrapper makes the PyTorch model look like a Scikit-Learn model for the plotting functions done
class PyTorchMLPWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, device, classes):
        self.model = model
        self.device = device
        self.classes_ = classes 

    def predict(self, X, batch_size=32):
        self.model.eval()
        preds_list = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                # Crop batch un CPU
                batch = X[i:i+batch_size]
                
                # Convert to tensor if it is numpy
                if isinstance(batch, np.ndarray):
                    batch = torch.tensor(batch, dtype=torch.float32)
                
                # Move this batch to GPU
                batch = batch.to(self.device)
                
                # Inference
                logits = self.model(batch)
                batch_preds = torch.argmax(logits, dim=1)
                
                # Move result to CPU to free GPU (VRAM)
                preds_list.append(batch_preds.cpu())

        #Concat everything   
        return torch.cat(preds_list).numpy()

    def predict_proba(self, X, batch_size=32):
        self.model.eval()
        probs_list = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                
                if isinstance(batch, np.ndarray):
                    batch = torch.tensor(batch, dtype=torch.float32)
                    
                batch = batch.to(self.device)
                logits = self.model(batch)
                probs = torch.softmax(logits, dim=1)
                
                probs_list.append(probs.cpu())
                
        return torch.cat(probs_list).numpy()

class PyTorchMLPWrapper_Supcon(BaseEstimator, ClassifierMixin):
    def __init__(self, model, device, classes):
        self.model = model
        self.device = device
        self.classes_ = classes 

    def predict(self, X, batch_size=32):
        self.model.eval()
        preds_list = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                # Crop batch un CPU
                batch = X[i:i+batch_size]
                
                # Convert to tensor if it is numpy
                if isinstance(batch, np.ndarray):
                    batch = torch.tensor(batch, dtype=torch.float32)
                
                # Move this batch to GPU
                batch = batch.to(self.device)
                
                # Inference
                logits, normalized_features = self.model(batch)
                batch_preds = torch.argmax(logits, dim=1)
                
                # Move result to CPU to free GPU (VRAM)
                preds_list.append(batch_preds.cpu())

        #Concat everything   
        return torch.cat(preds_list).numpy()

    def predict_proba(self, X, batch_size=32):
        self.model.eval()
        probs_list = []
        N = len(X)

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                
                if isinstance(batch, np.ndarray):
                    batch = torch.tensor(batch, dtype=torch.float32)
                    
                batch = batch.to(self.device)
                logits, normalized_features = self.model(batch)
                probs = torch.softmax(logits, dim=1)
                
                probs_list.append(probs.cpu())
                
        return torch.cat(probs_list).numpy()

# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def _build_mlp_model(input_dim, num_classes, attention: bool, supcon: bool):
    """
    Returns the correct model architecture based on flags.
 
    attention=False, supcon=False -> MLP
    attention=True,  supcon=False -> MLP_Attention
    attention=False, supcon=True  -> MLP_SupCon
    attention=True,  supcon=True  -> MLP_Attention_SupCon
    """
    if not attention and not supcon:
        return MLP(input_dim, num_classes)
    elif attention and not supcon:
        return MLP_Attention(input_dim, num_classes)
    elif not attention and supcon:
        return MLP_SupCon(input_dim, num_classes)
    else:  # attention and supcon
        return MLP_Attention_SupCon(input_dim, num_classes)
 
def _wrap_model(model, device, classes, supcon: bool):
    """Returns the correct sklearn-compatible wrapper."""
    if supcon:
        return PyTorchMLPWrapper_Supcon(model, device, classes)
    return PyTorchMLPWrapper(model, device, classes)

# -------------------------------------------------------------------------------
# MAIN CLASSIFIER CLASS
# -------------------------------------------------------------------------------
class MusicGenreClassifier:
    def __init__(
        self,
        model_name: str,
        feature_space_path: str = None,
        classifier_path: str = None,
        classifier_type: str = "mlp",
        use_pca: bool = False, 
        pca_components: int = 15,
        device: str = None,
        attention : bool = False,
        genre_list : list = None,
    ):
        self.model_name = model_name
        self.only_name_model =  model_name.split("/")[-1]
        self.feature_space_path = feature_space_path
        self.classifier_path = classifier_path
        self.classifier_type = classifier_type
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.attention = attention
        self.genre_list = genre_list

        self.name_genres = ""
        for genre in self.genre_list:
            self.name_genres += genre + "_"
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        if model_name:
            print(f"Loading base model: {model_name}")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
            self.target_sample_rate = self.feature_extractor.sampling_rate

        self.classifier = None
        self.label_encoder = None
        self.pca = None
        self.scaler = None
        
        if self.classifier_path and os.path.exists(self.classifier_path):
            self.load_classifier()

    # ------------------------------------------
    # Utilities
    # -------------------------------------------
    def _sanitize_filename(self, text):
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        return text.replace(" ", "")

    def _normalize_X(self, X, fit=True):
        """Apply L2 normalisation. Handles both 2D and 3D arrays."""
        if X.ndim == 3:
            N, T, D = X.shape
            X_flat = X.reshape(N * T, D)
            if fit:
                self.scaler = Normalizer(norm='l2')
                X_flat = self.scaler.fit_transform(X_flat)
            else:
                X_flat = self.scaler.transform(X_flat)
            return X_flat.reshape(N, T, D)
        else:
            if fit:
                self.scaler = Normalizer(norm='l2')
                return self.scaler.fit_transform(X)
            return self.scaler.transform(X)

    def _apply_pca(self, X, fit=True):
        """Apply PCA. Handles both 2D and 3D arrays."""
        if not self.use_pca:
            return X
        
        feat_dim = X.shape[2] if X.ndim == 3 else X.shape[1]

        print(f"  Applying PCA: {feat_dim} -> {self.pca_components} components...")

        if X.ndim == 3:
            N, T, D = X.shape
            X_flat = X.reshape(N * T, D)
            if fit:
                self.pca = PCA(n_components=self.pca_components, random_state=42)
                X_flat = self.pca.fit_transform(X_flat)
            else:
                X_flat = self.pca.transform(X_flat)
            return X_flat.reshape(N, T, self.pca_components)
        else:
            if fit:
                self.pca = PCA(n_components=self.pca_components, random_state=42)
                return self.pca.fit_transform(X)
            return self.pca.transform(X)

    def _get_input_dim(self, X):
        """Returns the feature dimension regardless of whether X is 2D or 3D."""
        return X.shape[2] if X.ndim == 3 else X.shape[1] 

    def _save_separability_metrics(self, X, labels, save_path):
        score, entropy_scores = calculate_linear_separability(X, labels)

        with open(save_path, "w") as f:
            f.write(f"Model: {self.model_name}\n")
            f.write("-" * 40 + "\n")
            f.write(f"FINAL SEPARABILITY SCORE (linear): {score:.4f} (Lower is better)\n")
            f.write(f"Total Conditional Entropy (logarithmic): {sum(entropy_scores.values()):.4f}\n")
            f.write("-" * 40 + "\n")
            f.write("Conditional Entropy H(Y|X) per Genre (One-vs-Rest):\n")
            for genre, s in entropy_scores.items():
                f.write(f"  - {genre}: {s:.4f}\n")
            
        print(f"Separability metrics saved to: {save_path}")
        return score, entropy_scores
 
    def _save_classifier_metrics(self, y_test, y_pred, le, epochs, save_dir):
        report_str = classification_report(y_test, y_pred, target_names=le.classes_)
        print(report_str)
 
        acc = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        print(f"Chunk-Level Accuracy: {acc:.4f}")
        print(f"F1 Score (Weighted):  {f1_weighted:.4f}")
 
        metrics_path = os.path.join(save_dir, f'{self.only_name_model}_{self.classifier_type}_metrics_{self.name_genres}.txt')
        cm_path = os.path.join(save_dir, f'{self.only_name_model}_attention{self.attention}_{self.classifier_type}_confusion_matrix_{self.name_genres}.png')

        with open(metrics_path, "w") as f:
            f.write(f"Model: {self.model_name}\nClassifier: {self.classifier_type}\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy: {acc:.4f}\nF1 Score (Weighted): {f1_weighted:.4f}\nF1 Score (Macro): {f1_macro:.4f}\n")
            f.write("-" * 30 + "\nClassification Report:\n" + report_str)
        print(f"Metrics saved to: {metrics_path}")
 
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

        plt.figure(figsize=(10, 8))
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
        plt.title(f"Confusion Matrix - {self.classifier_type}")
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to: {cm_path}")

    # ---------------------------------------------
    # Plotting helpers
    # ---------------------------------------------
    def plot2d_trained_decision_space(self, embedding_matrix, true_labels, method, save_path):
        if self.classifier is None or self.label_encoder is None:
            self.load_classifier()

        if embedding_matrix.ndim == 3:
            embedding_2d = embedding_matrix.mean(axis=1) # (N, Dim)
        else:
            embedding_2d = embedding_matrix

        print(f"\n  [Plotting 2D] Predicting labels for {embedding_matrix.shape[0]} samples in batches...")
        
        y_pred_idx = []
        batch_size = 32  
        n_samples = embedding_2d.shape[0]

        for i in range(0, n_samples, batch_size):
            batch = embedding_matrix[i : i + batch_size]
            
            preds = self.classifier.predict(batch)
            y_pred_idx.extend(preds)        
        
        pred_labels = self.label_encoder.inverse_transform(y_pred_idx)

        # 3. Dimensionality reduction
        method = method.lower()
        if method == "umap":
            import umap
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
            points = reducer.fit_transform(embedding_2d) 
        elif method == "pca":
            reducer = PCA(n_components=2, random_state=42)
            points = reducer.fit_transform(embedding_2d)
        else:
            raise ValueError("Unknown projection method.")

        # Color by predicted label
        unique_preds = sorted(set(pred_labels))
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
        plt.title(f"{method.upper()} — Classifier Decision Space (GPU Trained)")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved trained decision plot at: {save_path}")

        # plt.show()

    def plot3d_embedding_space(self, embedding_matrix, true_labels, method, save_path):
        if embedding_matrix.ndim == 3:
             embedding_matrix_2d = embedding_matrix.mean(axis=1) # (N, Dim)
        else:
             embedding_matrix_2d = embedding_matrix

        print(f"\n  [Plotting 3D] Embedding space generated...")

        method = method.lower()
        if method == "umap":
            import umap
            reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
            points = reducer.fit_transform(embedding_matrix_2d) 
        elif method == "pca":
            reducer = PCA(n_components=3, random_state=42)
            points = reducer.fit_transform(embedding_matrix_2d)
        elif method == "t-sne" or method == "tsne":
            reducer = TSNE(n_components=3, random_state=42)
            points = reducer.fit_transform(embedding_matrix_2d)
        elif method == "lda":
            reducer = LinearDiscriminantAnalysis(n_components=3)
            points = reducer.fit_transform(embedding_matrix_2d, true_labels) 
        else:
            raise ValueError("Unknown projection method.")
        
        # ============= IMAGE ====================
        # Color by predicted label
        unique_preds = sorted(list(set(true_labels)))
        cmap = plt.get_cmap('tab20' if len(unique_preds) <= 20 else 'hsv')
        color_map = {label: cmap(i / len(unique_preds)) for i, label in enumerate(unique_preds)}
        point_colors = [color_map[label] for label in true_labels]

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=point_colors, s=60, alpha=0.85)

        handles = [
            Line2D([0], [0], marker='o', color='w',
                markerfacecolor=color_map[label],
                markersize=10)
            for label in unique_preds
        ]
        ax.legend(handles, unique_preds, title="Predicted Genres", loc='best')
        ax.set_title(f"{method.upper()} — Embedding space")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")

        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved embedding {method} plot at: {save_path}")
        

        # ============ INTERACTIVE ================
        df = pd.DataFrame({
            'Component 1': points[:, 0],
            'Component 2': points[:, 1],
            'Component 3': points[:, 2],
            'Genre': true_labels
        })

        f = px.scatter_3d(
            df, 
            x='Component 1', 
            y='Component 2', 
            z='Component 3',
            color='Genre',
            title=f"{method.upper()} — Embedding Space (Interactive)",
            opacity=0.75
        )

        # Reducir el tamaño de los puntos para que la nube no se vea tan saturada
        f.update_traces(marker=dict(size=4))

        # 3. Guardar como HTML
        if save_path:
            # Nos aseguramos de que la extensión sea .html y no .png
            if save_path.endswith('.png'):
                save_path = save_path[:-4] + '.html'
            elif not save_path.endswith('.html'):
                save_path += '.html'
                
            f.write_html(save_path)
            print(f"Saved interactive 3D plot at: {save_path}")

    # --------------------------------------------
    # Data loading
    # --------------------------------------------
    def load_feature_space(self): 
        # If it is a .pt file
        if os.path.isfile(self.feature_space_path):
            print(f"\nLoading features from file: {self.feature_space_path}")
            data = torch.load(self.feature_space_path, map_location="cpu")
            print(f"[Data Load] Loaded {data['embeddings'].shape[0]} samples with dimension {data['embeddings'].shape[1]}")
            return (data["embeddings"], data["labels"], data.get("song_names"))
        
        # If it is a directory
        elif os.path.isdir(self.feature_space_path):
            print(f"\nLoading features from directory: {self.feature_space_path}")
            all_embeddings, all_labels, all_song_names = [], [], []

            # print(f"Filtering for specific genres in list: {self.genre_list}")

            for genre in self.genre_list:
                # Rebuilding the name of the file exactly as Embedder class saved it
                safe_genre = self._sanitize_filename(genre)
                filename = f"{self.only_name_model}_attention{self.attention}_{safe_genre}.pt"
                file_path = os.path.join(self.feature_space_path, filename)
                
                # Verifying the existance of the file
                if os.path.exists(file_path):
                    print(f"  Merging file: {filename}")
                    data = torch.load(file_path, map_location="cpu")
                    
                    all_embeddings.append(data["embeddings"])
                    all_labels.extend(data["labels"])
                    if "song_names" in data:
                        all_song_names.extend(data["song_names"])
                else:
                    print(f"  Warning: Feature space file not found for genre '{genre}' (Expected: {filename})")
            
            if not all_embeddings:
                raise FileNotFoundError(f"No matching .pt files found in {self.feature_space_path} for the provided genre list.")

            # Concat everything into a single tensor
            final_embeddings = torch.cat(all_embeddings, dim=0)
            print(f"Total loaded shape: {final_embeddings.shape}")
            
            return final_embeddings, all_labels, all_song_names

    def _build_classifier(self, input_dim):
        if self.classifier_type == "logreg":
            return LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="auto")
        elif self.classifier_type == "svm":
            return SVC(kernel="rbf", probability=True)
        elif self.classifier_type == "knn":
            return KNeighborsClassifier(n_neighbors=29)
        elif self.classifier_type == "mlp":
            # Just a placeholder, actual GPU construction happens in train loop
            return None 

    # --------------------------------------------
    # Training loop (CE)
    # --------------------------------------------
    def _run_training_loop_ce(self, model, X_train, y_train_t, optimizer, criterion, epochs, batch_size):
        model.train()
        n_samples = X_train.shape[0]
        is_supcon_model = isinstance(model, (MLP_SupCon, MLP_Attention_SupCon))

        for epoch in range(epochs):
            perm = torch.randperm(n_samples)

            for i in range(0, n_samples, batch_size):
                indices = perm[i:i + batch_size]
                batch_x = torch.tensor(X_train[indices.cpu().numpy()], dtype=torch.float32).to(self.device)
                batch_y = y_train_t[indices]
 
                if epoch == 0 and i == 0:
                    print(f"\n  [Training data] batch_x: {batch_x.shape}, batch_y: {batch_y.shape}")
 
                optimizer.zero_grad()
                out = model(batch_x)
                logits = out[0] if is_supcon_model else out
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
 
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")
            
    # -----------------------------------------
    # Training loop (CE + SupCon)
    # ------------------------------------------
    def _run_training_loop_supcon(self, model, X_train, y_train_t, optimizer,
                                   criterion_ce, criterion_supcon, alpha, epochs, batch_size):
        model.train()
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            perm = torch.randperm(n_samples)

            for i in range(0, n_samples, batch_size):
                indices = perm[i:i + batch_size]
                batch_x = torch.tensor(X_train[indices.cpu().numpy()], dtype=torch.float32).to(self.device)
                batch_y = y_train_t[indices]
 
                if epoch == 0 and i == 0:
                    print(f"\n  [Training data] batch_x: {batch_x.shape}, batch_y: {batch_y.shape}")
 
                optimizer.zero_grad()
 
                logits, normalized_features = model(batch_x)
 
                loss_ce = criterion_ce(logits, batch_y)
                loss_supcon = criterion_supcon(normalized_features, batch_y)
                loss = loss_ce + alpha * loss_supcon
 
                loss.backward()
                optimizer.step()
 
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f} "
                      f"(CE: {loss_ce.item():.4f}, SupCon: {loss_supcon.item():.4f})")
            
    # ----------------------------------------
    # Trainging (CE)
    # ----------------------------------------
    def train_classifier(self, epochs=100, batch_size=64, lr=0.001, criterion=nn.CrossEntropyLoss()):
        X, labels, song_names = self.load_feature_space()
 
        save_emb_dir = (f"/dataslow/storage/Experiments/INTERNS/anavarror/plots_feature_spaces/{self.only_name_model}_attention{self.attention}_pca{self.use_pca}_containfragments_{self.name_genres}")
        os.makedirs(save_emb_dir, exist_ok=True)
 
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
 
        le = LabelEncoder()
        y = le.fit_transform(labels)
        print(f"\n[Label Encoding] {len(le.classes_)} classes -> {le.classes_}")
        print(f"\nTraining on {X.shape[0]} chunks. Shape: {X.shape}")
 
        # Preprocessing
        print(f"  Before Normalizer: mean={np.mean(X):.4f}, std={np.std(X):.4f}")
        X = self._normalize_X(X, fit=True)
        print(f"  After Normalizer:  mean={np.mean(X):.4f}, std={np.std(X):.4f}")
        X = self._apply_pca(X, fit=True)
 
        # Separability on raw feature space
        self._save_separability_metrics(X, labels, os.path.join(save_emb_dir, f'{self.only_name_model}_separability_metrics_{self.name_genres}.txt') )
 
        # Train/test split (song-aware)
        X_train, X_test, y_train, y_test = self._song_aware_split(X, y, song_names)
        del X; gc.collect()
 
        save_model_dir = (f"/dataslow/storage/Experiments/INTERNS/anavarror/trained_models_1k_20s_contain_fragments/{self.only_name_model}_attention{self.attention}_pca{self.use_pca}_{self.classifier_type}_epochs{epochs}_containfragments_{self.name_genres}")
        os.makedirs(save_model_dir, exist_ok=True)
 
        if self.classifier_type == "mlp":
            input_dim = self._get_input_dim(X_train)
            model = _build_mlp_model(input_dim, len(le.classes_), self.attention, supcon=False).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            y_train_t = torch.tensor(y_train, dtype=torch.long).to(self.device)
            self._run_training_loop_ce(model, X_train, y_train_t, optimizer, criterion, epochs, batch_size)
            self.classifier = _wrap_model(model, self.device, le.classes_, supcon=False)
        else:
            print(f"Fitting {self.classifier_type.upper()}...")
            self.classifier = self._build_classifier(X_train.shape[1] if X_train.ndim == 2 else X_train.shape[2])
            self.classifier.fit(X_train, y_train)
 
        y_pred = self.classifier.predict(X_test)
        self.label_encoder = le
        self._save_classifier_metrics(y_test, y_pred, le, epochs, save_model_dir)
        self._save_plots_and_separability(X_train, y_train, le, save_model_dir, save_emb_dir)
        self._save_pipeline(le)

    # ----------------------------------------
    # Training (CE + Supcon)
    # ----------------------------------------   
    def train_classifier_supcon(self, epochs=100, batch_size=64, lr=0.001,
                                 criterion_ce=nn.CrossEntropyLoss(),
                                 criterion_supcon=losses.SupConLoss(),
                                 alpha=0.1):
        X, labels, song_names = self.load_feature_space()
 
        save_emb_dir = (f"/dataslow/storage/Experiments/INTERNS/anavarror/plots_feature_spaces/{self.only_name_model}_attention{self.attention}_pca{self.use_pca}_containfragments_SUPCON_{self.name_genres}")
        os.makedirs(save_emb_dir, exist_ok=True)
 
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
 
        le = LabelEncoder()
        y = le.fit_transform(labels)
        print(f"[Label Encoding] {len(le.classes_)} classes -> {le.classes_}")
        print(f"\nTraining on {X.shape[0]} chunks. Shape: {X.shape}")
 
        # Preprocessing
        print(f"  Before Normalizer: mean={np.mean(X):.4f}, std={np.std(X):.4f}")
        X = self._normalize_X(X, fit=True)
        print(f"  After Normalizer:  mean={np.mean(X):.4f}, std={np.std(X):.4f}")
        X = self._apply_pca(X, fit=True)
 
        self._save_separability_metrics(
            X, labels,
            os.path.join(save_emb_dir, f'{self.only_name_model}_separability_metrics_{self.name_genres}.txt')
        )
 
        X_train, X_test, y_train, y_test = self._song_aware_split(X, y, song_names)
        del X; gc.collect()
 
        save_model_dir = (f"/dataslow/storage/Experiments/INTERNS/anavarror/trained_models_1k_20s_contain_fragments/"
                          f"{self.only_name_model}_attention{self.attention}_pca{self.use_pca}_{self.classifier_type}_epochs{epochs}_containfragments_SUPCON_batch{batch_size}_alpha{alpha}_{self.name_genres}")
        os.makedirs(save_model_dir, exist_ok=True)
 
        if self.classifier_type == "mlp":
            input_dim = self._get_input_dim(X_train)
            # _build_mlp_model with supcon=True always returns a model with (logits, features) output
            model = _build_mlp_model(input_dim, len(le.classes_), self.attention, supcon=True).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            y_train_t = torch.tensor(y_train, dtype=torch.long).to(self.device)
            self._run_training_loop_supcon(model, X_train, y_train_t, optimizer,
                                            criterion_ce, criterion_supcon, alpha, epochs, batch_size)
            self.classifier = _wrap_model(model, self.device, le.classes_, supcon=True)
        else:
            print(f"Fitting {self.classifier_type.upper()} (SupCon not applicable for sklearn classifiers)...")
            self.classifier = self._build_classifier(X_train.shape[1] if X_train.ndim == 2 else X_train.shape[2])
            self.classifier.fit(X_train, y_train)
 
        y_pred = self.classifier.predict(X_test)
        self.label_encoder = le
        self._save_classifier_metrics(y_test, y_pred, le, epochs, save_model_dir)
        self._save_plots_and_separability(X_train, y_train, le, save_model_dir, save_emb_dir)
        self._save_pipeline(le)

    # ----------------------------------------
    # Private helpers used by multiple train_* methods
    # ----------------------------------------
    def _song_aware_split(self, X, y, song_names):
        if song_names is not None and len(song_names) == len(X):
            print("\n  Splitting by song ID (GroupShuffleSplit).")
            gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
            train_idx, test_idx = next(gss.split(X, y, groups=song_names))

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            print(f"  Chunks (Tensors) in Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

            train_songs = set(np.array(song_names)[train_idx])
            test_songs = set(np.array(song_names)[test_idx])
            print(f"  Songs in Train: {len(train_songs)}, Test: {len(test_songs)}, Leakage: {len(train_songs & test_songs)}")
        else:
            print("  WARNING: No song names — using random split (potential leakage).")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
        return X_train, X_test, y_train, y_test
 
    def _save_plots_and_separability(self, X_train, y_train, le, save_model_dir, save_emb_dir):
        y_train_names = le.inverse_transform(y_train)
        for meth in ["umap", "pca", "tsne", "lda"]:
            name_file = (f'{self.only_name_model}_attention{self.attention}_{self.classifier_type}_{meth}_contain_fragments_{self.name_genres}.png')
            self.plot3d_embedding_space(X_train, y_train_names, method=meth, save_path=os.path.join(save_model_dir, name_file))
        
        self._save_separability_metrics(X_train, y_train_names,
            os.path.join(save_emb_dir, f'{self.only_name_model}_separability_metrics_train_{self.name_genres}.txt')
        )
 
    def _save_pipeline(self, le):
        save_obj = {
            "model": self.classifier,
            "label_encoder": le,
            "scaler": self.scaler,
            "pca": self.pca,
            "use_pca": self.use_pca,
            "pca_components": self.pca_components,
        }
        joblib.dump(save_obj, self.classifier_path)
        print(f"Saved model pipeline to {self.classifier_path}")
    
    # -------------------------------------------
    # INFERENCE FUNCTIONS
    # --------------------------------------------
    def get_inference_chunks(self, mp3_path):
        """Extracts 20s chunks for inference (Overlapping for better voting coverage)."""
        try:
            audio, _ = librosa.load(mp3_path, sr=self.target_sample_rate, mono=True)
            print(f"\n  [Inference Audio] Loaded song, samples: {len(audio)} ({len(audio)/self.target_sample_rate:.1f}s)")

            chunk_seconds = 20.0
            hop_seconds = 10.0 
            
            chunk_size = int(chunk_seconds * self.target_sample_rate)
            hop_size = int(hop_seconds * self.target_sample_rate)
            
            segments = [audio[i:i + chunk_size] for i in range(0, len(audio) - chunk_size + 1, hop_size)]
            
            if not segments and len(audio) > 0:
                if len(audio) < chunk_size:
                    padding = chunk_size - len(audio)
                    segments = [np.pad(audio, (0, padding))]
                else:
                    segments = [audio[:chunk_size]]

            segments = segments[:12] 
            print(f"  [Inference Chunks] Processing {len(segments)} segments...")
            
            emb_list = []
            for seg in segments:
                seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-7) # Normalizing
                inputs = self.feature_extractor(seg, sampling_rate=self.target_sample_rate, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    # Stack last 4 layers -> (4, Batch, Time, Dim) y Mean over layers -> (Batch, Time, Dim)
                    fused = torch.stack(outputs.hidden_states[-4:]).mean(0)
                    
                    if self.attention:
                        # If attention we keep the time : (Batch, Time, Dim) -> (Time, Dim)
                        emb = fused.squeeze(0).cpu().numpy()
                    else:
                        # If NO attention we do mean of time: (Batch, Time, Dim) -> (Dim,)
                        emb = fused.mean(dim=1).squeeze(0).cpu().numpy()
                emb_list.append(emb)
            final_embs = np.array(emb_list)
            print(f"  [Inference Output] Final embedding batch shape: {final_embs.shape}")
            return final_embs if emb_list else None
            
        except Exception as e:
            print(f"Inference error: {e}")
            return None

    def predict_genre(self, mp3_path: str):
        if self.classifier is None:
            if not self.load_classifier(): return None

        start_time = time.time()
        
        chunks = self.get_inference_chunks(mp3_path)
        if chunks is None: return None
        
        if self.scaler:
            if chunks.ndim == 3:
                # 3D case (Batch, Time, Dim) -> Flatten to (Batch*Time, Dim)
                N, T, D = chunks.shape
                chunks_flat = chunks.reshape(N * T, D)
                chunks_flat = self.scaler.transform(chunks_flat)
                chunks = chunks_flat.reshape(N, T, D) # Regain 3D
            else:
                #  2D case
                chunks = self.scaler.transform(chunks)
        
        if self.use_pca and self.pca:
            if chunks.ndim == 3:
                N, T, D = chunks.shape
                chunks_flat = chunks.reshape(N * T, D)
                chunks_flat = self.pca.transform(chunks_flat)
                # dimension D changes to n_components
                chunks = chunks_flat.reshape(N, T, -1)
            else:
                chunks = self.pca.transform(chunks)

        # The Wrapper handles predict_proba just like sklearn
        if hasattr(self.classifier, "predict_proba"): # if the model is an MLP pytorch
            probs_matrix = self.classifier.predict_proba(chunks) # (N_chunks, N_genres)
            avg_probs = np.mean(probs_matrix, axis=0) # (N_genres,)
            pred_idx = np.argmax(avg_probs)
        else:
            preds = self.classifier.predict(chunks)
            pred_idx = Counter(preds).most_common(1)[0][0]
            avg_probs = np.zeros(len(self.label_encoder.classes_))
            avg_probs[pred_idx] = 1.0

        pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
        
        return {
            "label": pred_label,
            "probabilities": dict(zip(self.label_encoder.classes_, avg_probs)),
            "embedding": np.mean(chunks, axis=0), 
            "inference_time": time.time() - start_time
        }

    def load_classifier(self):
        if not os.path.exists(self.classifier_path): return False
        data = joblib.load(self.classifier_path)
        self.classifier = data["model"]
        self.label_encoder = data["label_encoder"]
        self.scaler = data.get("scaler")
        self.pca = data.get("pca")
        self.use_pca = data.get("use_pca", False)
        return True
    
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