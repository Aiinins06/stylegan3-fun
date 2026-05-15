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
from matplotlib.backends.backend_pdf import PdfPages

from utils.attention import SelfAttentionPooling 
from utils.halo import HALOLoss

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
    print(f"FINAL SEPARABILITY SCORE (linear): {separability_score:.4f}")
    
    return separability_score, entropy_scores

# -----------------------------------------------------------------------------
# MODEL DEFINITIONS
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.head(features)

        return logits, features
    
class MLP_Attention(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP_Attention, self).__init__()
        
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
        # x input shape: (batch, time_steps, input_dim)
        pooled_x = self.attention_pool(x) # -> (batch, input_dim)
        features = self.encoder(pooled_x)
        logits = self.head(features)
        return logits, features
    
class MLP_SupCon(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP_SupCon, self).__init__()
        
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

        self.projection_head = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.classifier_head = nn.Linear(128, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        
        # Projection and L2 Normalization for SupCon
        projected = self.projection_head(features)
        normalized_features = F.normalize(projected, p=2, dim=1)
        
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

class MLP_Halo(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP_Halo, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.centroids = nn.Parameter(torch.randn(num_classes, 128, dtype=torch.float32))
    
    def forward(self, x):
        embeddings = self.encoder(x)

        centroids = self.centroids - self.centroids.mean(dim=0, keepdim=True)
        return embeddings, centroids
    
class MLP_Attention_Halo(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP_Attention_Halo, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.attention_pool = SelfAttentionPooling(input_dim)
        self.centroids = nn.Parameter(torch.randn(num_classes, 128, dtype=torch.float32))

    def forward(self, x):
        pooled_x = self.attention_pool(x)
        embeddings = self.encoder(pooled_x)
        centroids = self.centroids - self.centroids.mean(dim=0, keepdim=True)
        return embeddings, centroids
    
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
                logits, features = self.model(batch)
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
                logits, features = self.model(batch)
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

class PyTorchMLPHaloWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, device, classes):
        self.model = model
        self.device = device
        self.classes_ = classes 

    def predict(self, X, batch_size=32):
        self.model.eval()
        preds_list = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                if isinstance(batch, np.ndarray): batch = torch.tensor(batch, dtype=torch.float32)
                batch = batch.to(self.device)
                
                embeddings, centroids = self.model(batch)
                # Compute Euclidean distance to each centroid. Argmin is the prediction
                distances = torch.cdist(embeddings, centroids) 
                batch_preds = torch.argmin(distances, dim=1)
                preds_list.append(batch_preds.cpu())
        return torch.cat(preds_list).numpy()

    def predict_proba(self, X, batch_size=32):
        self.model.eval()
        probs_list = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                if isinstance(batch, np.ndarray): batch = torch.tensor(batch, dtype=torch.float32)
                batch = batch.to(self.device)
                
                embeddings, centroids = self.model(batch)
                distances = torch.cdist(embeddings, centroids)
                # Convert distances to valid probabilities using softmax of negative distance
                probs = torch.softmax(-distances, dim=1)
                probs_list.append(probs.cpu())
        return torch.cat(probs_list).numpy()
    
# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def _build_mlp_model_nohalo(input_dim, num_classes, attention: bool, supcon: bool):
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
 
def _build_mlp_model(input_dim, num_classes, attention: bool, mode: str):
    """ mode in ['ce', 'halo'] """
    if mode == "ce":
        return MLP_Attention(input_dim, num_classes) if attention else MLP(input_dim, num_classes)
    elif mode == "supcon":
        return MLP_Attention_SupCon(input_dim, num_classes) if attention else MLP_SupCon(input_dim, num_classes)
    elif mode == "halo":
        return MLP_Attention_Halo(input_dim, num_classes) if attention else MLP_Halo(input_dim, num_classes)

def _wrap_model_nohalo(model, device, classes, supcon: bool):
    """Returns the correct sklearn-compatible wrapper."""
    if supcon:
        return PyTorchMLPWrapper_Supcon(model, device, classes)
    return PyTorchMLPWrapper(model, device, classes)

def _wrap_model(model, device, classes, mode: str):
    if mode == "halo":
        return PyTorchMLPHaloWrapper(model, device, classes)
    if mode == "supcon":
        return PyTorchMLPWrapper_Supcon(model, device, classes)
    return PyTorchMLPWrapper(model, device, classes)

# -------------------------------------------------------------------------------
# MAIN CLASSIFIER CLASS
# -------------------------------------------------------------------------------
class MusicGenreClassifier:
    def __init__(self, model_name: str, feature_space_path: str = None, classifier_path: str = None,
        classifier_type: str = "mlp", use_pca: bool = False, pca_components: int = 15, device: str = None,
        attention : bool = False, genre_list : list = None,
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

    def _save_run_report_nohalo(
        self,
        history,        # dict returned by a training loop
        y_test,         # ground-truth labels (int)
        y_pred,         # predicted labels (int)
        le,             # fitted LabelEncoder
        X_train,        # numpy array used for embedding plots
        y_train,        # int labels for X_train
        save_dir,       # directory where both files land
        prefix,         # filename prefix
        supcon=False,   # whether to show CE/SupCon sub-losses
    ):
        """
        Produces these files inside `save_dir`:
 
          {prefix}_training_history.csv          — per-epoch numbers (always)
          {prefix}_report.pdf                    — all visuals on one multi-page PDF
          {prefix}_embedding_{METHOD}.html       — interactive 3-D plotly per method
        """
        os.makedirs(save_dir, exist_ok=True)
 
        # ---- 1. CSV: per-epoch metrics ----------------------------------------
        csv_path = os.path.join(save_dir, f"{prefix}_training_history.csv")
        has_history = bool(history and history.get("epoch"))
        if has_history:
            pd.DataFrame(history).to_csv(csv_path, index=False)
            print(f"  Epoch history  → {csv_path}")
        else:
            print("  [Warning] No per-epoch history available — CSV skipped.")
 
        # ---- Shared metrics ---------------------------------------------------
        class_names = le.classes_
        acc        = accuracy_score(y_test, y_pred)
        f1_w       = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_m       = f1_score(y_test, y_pred, average="macro",    zero_division=0)
        report_str = classification_report(y_test, y_pred, target_names=class_names)
 
        # ---- Shared color setup for all embedding plots ----------------------
        y_train_names = le.inverse_transform(y_train)
        unique_labels = sorted(set(y_train_names))
        cmap = plt.get_cmap("tab20" if len(unique_labels) <= 20 else "hsv")
        color_map = {l: cmap(i / len(unique_labels)) for i, l in enumerate(unique_labels)}
        pt_colors = [color_map[l] for l in y_train_names]
        legend_handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color_map[l], markersize=8)
            for l in unique_labels
        ]
 
        # ---- Pre-compute all projections -------------------------------------
        print("\n  Extracting learned latent space features for plotting...")
        self.classifier.model.eval()
        learned_features = []
        with torch.no_grad():
            for i in range(0, len(X_train), 256):
                bx = torch.tensor(X_train[i:i+256], dtype=torch.float32).to(self.device)
                
                # Pasar por el modelo
                out = self.classifier.model(bx)
                
                # Como ambos (MLP y MLP_SupCon) devuelven (logits, features)
                if isinstance(out, tuple):
                    feats = out[1]
                else:
                    feats = out
                    
                learned_features.append(feats.cpu().numpy())
                
        # Matrix (N, 128) that represents the latent space of MLP model
        X_train_learned = np.concatenate(learned_features, axis=0)

        emb_flat = X_train_learned.mean(axis=1) if X_train_learned.ndim == 3 else X_train_learned


        projections = []
        print("\n  Computing embedding projections...")
 
        # PCA 
        pca3 = PCA(n_components=3, random_state=42).fit_transform(emb_flat)
        projections.append(("PCA", pca3))
        print("  [PCA] done")
 
        # t-SNE
        try:
            tsne3 = TSNE(
                n_components=3, random_state=42,
                init="pca", learning_rate="auto",
            ).fit_transform(emb_flat)
            projections.append(("t-SNE", tsne3))
            print("  [t-SNE] done")
        except Exception as e:
            print(f"  [Warning] t-SNE failed: {e}")
 
        # LDA — supervised; n_components ≤ n_classes − 1, capped at 3
        try:
            n_lda = min(3, len(unique_labels) - 1)
            lda_pts = LinearDiscriminantAnalysis(n_components=n_lda).fit_transform(emb_flat, y_train_names)
            if lda_pts.shape[1] < 3:
                lda_pts = np.hstack(
                    [lda_pts, np.zeros((lda_pts.shape[0], 3 - lda_pts.shape[1]))]
                )
            projections.append(("LDA", lda_pts))
            print("  [LDA] done")
        except Exception as e:
            print(f"  [Warning] LDA failed: {e}")
 
        # UMAP 
        try:
            import umap as umap_lib
            umap3 = umap_lib.UMAP(
                n_components=3, n_neighbors=15,
                min_dist=0.1, metric="cosine", random_state=42,
            ).fit_transform(emb_flat)
            projections.append(("UMAP", umap3))
            print("  [UMAP] done")
        except ImportError:
            print("  [Warning] umap-learn not installed — UMAP skipped.")
        except Exception as e:
            print(f"  [Warning] UMAP failed: {e}")
 
        # ---- Separability ----------------------------------------------------
        # sep_score, entropy_scores = calculate_linear_separability(X_train, y_train_names)
        sep_score, entropy_scores = calculate_linear_separability(emb_flat, y_train_names)
 
        # ---- 2. Interactive HTML: one file per projection method -------------
        html_paths = []
        for method_name, pts3 in projections:
            df_emb = pd.DataFrame({
                "C1":    pts3[:, 0],
                "C2":    pts3[:, 1],
                "C3":    pts3[:, 2],
                "Genre": y_train_names,
            })
            fig_html = px.scatter_3d(
                df_emb, x="C1", y="C2", z="C3", color="Genre",
                title=f"{method_name} — 3-D embedding space (train set)",
                opacity=0.75,
            )
            fig_html.update_traces(marker=dict(size=3))
            fig_html.update_layout(
                legend=dict(itemsizing="constant"),
                margin=dict(l=0, r=0, t=40, b=0),
            )
            safe_name = method_name.replace("-", "").replace(" ", "_")
            html_path = os.path.join(save_dir, f"{prefix}_embedding_{safe_name}.html")
            fig_html.write_html(html_path)
            html_paths.append(html_path)
            print(f"  [{method_name}] interactive HTML → {html_path}")
 
        # ---- 3. PDF: all static visuals --------------------------------------
        pdf_path = os.path.join(save_dir, f"{prefix}_report.pdf")
 
        with PdfPages(pdf_path) as pdf:
 
            # ------ Page 1: summary text ----------------------------------------
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.axis("off")
            summary = (
                f"Run summary\n"
                f"{'='*60}\n"
                f"Model      : {self.model_name}\n"
                f"Classifier : {self.classifier_type}"
                f"  |  Attention: {self.attention}"
                f"  |  PCA: {self.use_pca}\n"
                f"Date       : {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"Accuracy (chunk-level) : {acc:.4f}\n"
                f"F1 weighted            : {f1_w:.4f}\n"
                f"F1 macro               : {f1_m:.4f}\n\n"
                f"Separability score     : {sep_score:.4f} \n\n"
                f"{'='*60}\n"
                f"Classification report\n"
                f"{'='*60}\n"
                f"{report_str}\n\n"
                f"{'='*60}\n"
                f"Separability H(Y|X) per genre\n"
                f"{'='*60}\n"
            )
            for g, s in entropy_scores.items():
                summary += f"  {g}: {s:.4f}\n"
            ax.text(
                0.02, 0.98, summary,
                transform=ax.transAxes, fontsize=8.5,
                verticalalignment="top", family="monospace",
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
 
            # ------ Page 2: training curves ------------------------------------
            # Always rendered when history is present — NaN values are handled
            # gracefully by matplotlib (gaps in the line).
            if has_history:
                epochs_ax = history["epoch"]
                n_plots = 3 if (supcon and "train_loss_ce" in history) else 2
                fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
                if n_plots == 1:
                    axes = [axes]   # make always iterable
                fig.suptitle("Training curves", fontsize=12)
 
                # ---- subplot 0: total loss -----------------------------------
                axes[0].plot(epochs_ax, history["train_loss"],
                             label="train loss", linewidth=1.5)
                axes[0].plot(epochs_ax, history["val_loss"],
                             label="val loss", linewidth=1.5, linestyle="--")
                axes[0].set_title("Loss per epoch")
                axes[0].set_xlabel("Epoch")
                axes[0].set_ylabel("Loss")
                axes[0].legend()
                axes[0].grid(alpha=0.3)
 
                # ---- subplot 1: accuracy + val F1 ---------------------------
                axes[1].plot(epochs_ax, history["train_acc"],
                             label="train acc", linewidth=1.5)
                axes[1].plot(epochs_ax, history["val_acc"],
                             label="val acc", linewidth=1.5, linestyle="--")
                axes[1].plot(epochs_ax, history["val_f1"],
                             label="val F1 (weighted)", linewidth=1.2, linestyle=":")
                axes[1].set_title("Accuracy & F1 per epoch")
                axes[1].set_xlabel("Epoch")
                axes[1].set_ylabel("Score")
                axes[1].legend()
                axes[1].grid(alpha=0.3)
 
                # ---- subplot 2 (SupCon only): CE vs contrastive breakdown ---
                if n_plots == 3:
                    axes[2].plot(epochs_ax, history["train_loss_ce"],
                                 label="CE loss", linewidth=1.5)
                    axes[2].plot(epochs_ax, history["train_loss_supcon"],
                                 label="SupCon loss", linewidth=1.5, linestyle="--")
                    axes[2].set_title("CE vs SupCon loss per epoch")
                    axes[2].set_xlabel("Epoch")
                    axes[2].set_ylabel("Loss")
                    axes[2].legend()
                    axes[2].grid(alpha=0.3)
 
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                print("  Training curves page added to PDF")
            
                # ------ Page 3: epoch log table --------------------------------
                # Prints every epoch row exactly as it appeared in stdout,
                # split across multiple PDF pages if needed (50 rows per page).
                is_supcon_hist = "train_loss_ce" in history
                ROWS_PER_PAGE = 50

                if is_supcon_hist:
                    col_headers = ["Epoch", "CE loss", "SC loss", "Norm loss",
                                    "Train acc", "Val loss", "Val acc", "Val F1"]
                    def row_vals(i):
                        e   = history["epoch"][i]
                        ce  = history["train_loss_ce"][i]
                        sc  = history["train_loss_supcon"][i]
                        nl  = history["train_loss"][i]
                        ta  = history["train_acc"][i]
                        vl  = history["val_loss"][i]
                        va  = history["val_acc"][i]
                        vf  = history["val_f1"][i]
                        vl_s = f"{vl:.4f}" if not np.isnan(vl) else "—"
                        va_s = f"{va:.4f}" if not np.isnan(va) else "—"
                        vf_s = f"{vf:.4f}" if not np.isnan(vf) else "—"
                        return [str(e), f"{ce:.4f}", f"{sc:.4f}", f"{nl:.4f}",
                                f"{ta:.4f}", vl_s, va_s, vf_s]
                else:
                    col_headers = ["Epoch", "Train loss", "Train acc",
                                    "Val loss", "Val acc", "Val F1"]
                    def row_vals(i):
                        e  = history["epoch"][i]
                        tl = history["train_loss"][i]
                        ta = history["train_acc"][i]
                        vl = history["val_loss"][i]
                        va = history["val_acc"][i]
                        vf = history["val_f1"][i]
                        vl_s = f"{vl:.4f}" if not np.isnan(vl) else "—"
                        va_s = f"{va:.4f}" if not np.isnan(va) else "—"
                        vf_s = f"{vf:.4f}" if not np.isnan(vf) else "—"
                        return [str(e), f"{tl:.4f}", f"{ta:.4f}", vl_s, va_s, vf_s]

                n_epochs = len(history["epoch"])
                for page_start in range(0, n_epochs, ROWS_PER_PAGE):
                    page_rows = [
                        row_vals(i)
                        for i in range(page_start, min(page_start + ROWS_PER_PAGE, n_epochs))
                    ]
                    fig, ax = plt.subplots(figsize=(12, max(3, len(page_rows) * 0.28 + 1.5)))
                    ax.axis("off")
                    tbl = ax.table(
                        cellText=page_rows,
                        colLabels=col_headers,
                        cellLoc="center",
                        loc="center",
                    )
                    tbl.auto_set_font_size(False)
                    tbl.set_fontsize(8)
                    tbl.auto_set_column_width(col=list(range(len(col_headers))))
                    # Header row styling
                    for col in range(len(col_headers)):
                        tbl[(0, col)].set_facecolor("#2c3e50")
                        tbl[(0, col)].set_text_props(color="white", fontweight="bold")
                    # Alternating row shading
                    for row in range(1, len(page_rows) + 1):
                        for col in range(len(col_headers)):
                            tbl[(row, col)].set_facecolor("#f0f0f0" if row % 2 == 0 else "white")

                    page_label = (f"Epoch log  (rows {page_start + 1}–"
                                    f"{min(page_start + ROWS_PER_PAGE, n_epochs)} of {n_epochs})")
                    ax.set_title(page_label, fontsize=10, pad=6, loc="left")
                    fig.tight_layout()
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                print(f"  Epoch log table added to PDF ({n_epochs} rows)")
 
            else:
                print("  [Warning] Training curves and epoch log skipped (no history).")

            # ------ Page N: confusion matrix -----------------------------------
            cm = confusion_matrix(y_test, y_pred)
            sz = max(6, len(class_names))
            fig, ax = plt.subplots(figsize=(sz, sz - 1))
            ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
                cmap=plt.cm.Blues, xticks_rotation=45, ax=ax, colorbar=False,
            )
            ax.set_title("Confusion matrix")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
 
            # ------ Pages 4 … 4+N: static 2-D + 3-D side-by-side per method --
            for method_name, pts3 in projections:
                pts2 = pts3[:, :2]
 
                fig = plt.figure(figsize=(16, 6))
                fig.suptitle(
                    f"{method_name} — embedding space (train set)",
                    fontsize=12, y=1.01,
                )
 
                # 2-D panel
                ax2 = fig.add_subplot(1, 2, 1)
                ax2.scatter(pts2[:, 0], pts2[:, 1],
                            c=pt_colors, s=12, alpha=0.6)
                ax2.legend(legend_handles, unique_labels,
                           title="Genre", loc="best", fontsize=6, markerscale=1.2)
                ax2.set_title(f"{method_name} 2-D")
                ax2.set_xlabel("Component 1")
                ax2.set_ylabel("Component 2")
                ax2.grid(alpha=0.2)
 
                # 3-D panel
                ax3 = fig.add_subplot(1, 2, 2, projection="3d")
                ax3.scatter(pts3[:, 0], pts3[:, 1], pts3[:, 2],
                            c=pt_colors, s=8, alpha=0.5)
                ax3.legend(legend_handles, unique_labels,
                           title="Genre", loc="best", fontsize=5, markerscale=1.2)
                ax3.set_title(f"{method_name} 3-D")
                ax3.set_xlabel("C1")
                ax3.set_ylabel("C2")
                ax3.set_zlabel("C3")
 
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                print(f"  [{method_name}] static page added to PDF")
 
            # PDF metadata
            d = pdf.infodict()
            d["Title"]   = f"{self.only_name_model} — {self.classifier_type} run report"
            d["Author"]  = "MusicGenreClassifier"
            d["Subject"] = f"genres: {self.name_genres}"
 
        print(f"  Full report    → {pdf_path}")
        return pdf_path, csv_path, html_paths

    def _save_run_report(self, history, y_test, y_pred, le, X_train, y_train, save_dir, prefix, mode="ce"):
            os.makedirs(save_dir, exist_ok=True)
            csv_path = os.path.join(save_dir, f"{prefix}_training_history.csv")
            has_history = bool(history and history.get("epoch"))
            if has_history:
                pd.DataFrame(history).to_csv(csv_path, index=False)

            class_names = le.classes_
            acc        = accuracy_score(y_test, y_pred)
            f1_w       = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            f1_m       = f1_score(y_test, y_pred, average="macro",    zero_division=0)
            report_str = classification_report(y_test, y_pred, target_names=class_names)

            y_train_names = le.inverse_transform(y_train)
            unique_labels = sorted(set(y_train_names))
            cmap = plt.get_cmap("tab20" if len(unique_labels) <= 20 else "hsv")
            color_map = {l: cmap(i / len(unique_labels)) for i, l in enumerate(unique_labels)}
            pt_colors = [color_map[l] for l in y_train_names]
            legend_handles = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[l], markersize=8)
                for l in unique_labels
            ]

            print("\n  Extracting learned latent space features for plotting...")
            self.classifier.model.eval()
            learned_features = []
            with torch.no_grad():
                for i in range(0, len(X_train), 256):
                    bx = torch.tensor(X_train[i:i+256], dtype=torch.float32).to(self.device)
                    out = self.classifier.model(bx)
                    
                    # HALO returns (embeddings, centroids), CE returns (logits, features)
                    feats = out[0] if mode == "halo" else out[1]
                    learned_features.append(feats.cpu().numpy())
                    
            X_train_learned = np.concatenate(learned_features, axis=0)
            emb_flat = X_train_learned.mean(axis=1) if X_train_learned.ndim == 3 else X_train_learned

            projections = []
            
            # PCA
            pca3 = PCA(n_components=3, random_state=42).fit_transform(emb_flat)
            projections.append(("PCA", pca3))
            
            # t-SNE
            try:
                tsne3 = TSNE(n_components=3, random_state=42, init="pca", learning_rate="auto").fit_transform(emb_flat)
                projections.append(("t-SNE", tsne3))
            except Exception as e:
                print(f"  [Warning] t-SNE failed: {e}")
                
            # LDA
            try:
                n_lda = min(3, len(unique_labels) - 1)
                lda_pts = LinearDiscriminantAnalysis(n_components=n_lda).fit_transform(emb_flat, y_train_names)
                if lda_pts.shape[1] < 3:
                    lda_pts = np.hstack([lda_pts, np.zeros((lda_pts.shape[0], 3 - lda_pts.shape[1]))])
                projections.append(("LDA", lda_pts))
            except Exception as e:
                print(f"  [Warning] LDA failed: {e}")
                
            # UMAP
            try:
                import umap as umap_lib
                umap3 = umap_lib.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42).fit_transform(emb_flat)
                projections.append(("UMAP", umap3))
            except Exception as e:
                print(f"  [Warning] UMAP failed: {e}")

            sep_score, entropy_scores = calculate_linear_separability(emb_flat, y_train_names)

            html_paths = []
            for method_name, pts3 in projections:
                df_emb = pd.DataFrame({"C1": pts3[:, 0], "C2": pts3[:, 1], "C3": pts3[:, 2], "Genre": y_train_names})
                fig_html = px.scatter_3d(df_emb, x="C1", y="C2", z="C3", color="Genre", title=f"{method_name} — 3-D embedding space", opacity=0.75)
                fig_html.update_traces(marker=dict(size=3))
                fig_html.update_layout(legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=40, b=0))
                html_path = os.path.join(save_dir, f"{prefix}_embedding_{method_name.replace('-', '')}.html")
                fig_html.write_html(html_path)
                html_paths.append(html_path)

            pdf_path = os.path.join(save_dir, f"{prefix}_report.pdf")
            with PdfPages(pdf_path) as pdf:
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.axis("off")
                summary = (
                    f"Run summary\n{'='*60}\n"
                    f"Model      : {self.model_name}\n"
                    f"Classifier : {self.classifier_type}  |  Attention: {self.attention}  |  PCA: {self.use_pca}\n"
                    f"Loss Mode  : {mode.upper()}\n"
                    f"Date       : {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    f"Accuracy (chunk-level) : {acc:.4f}\n"
                    f"F1 weighted            : {f1_w:.4f}\n"
                    f"F1 macro               : {f1_m:.4f}\n\n"
                    f"Separability score     : {sep_score:.4f}\n\n"
                    f"{'='*60}\nClassification report\n{'='*60}\n{report_str}\n\n"
                    f"{'='*60}\nSeparability H(Y|X) per genre\n{'='*60}\n"
                )
                for g, s in entropy_scores.items(): summary += f"  {g}: {s:.4f}\n"
                ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=8.5, verticalalignment="top", family="monospace")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

                if has_history:
                    epochs_ax = history["epoch"]
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    fig.suptitle("Training curves", fontsize=12)
                    
                    axes[0].plot(epochs_ax, history["train_loss"], label="train loss", linewidth=1.5)
                    axes[0].plot(epochs_ax, history["val_loss"], label="val loss", linewidth=1.5, linestyle="--")
                    axes[0].set_title("Loss per epoch")
                    axes[0].legend()
                    axes[0].grid(alpha=0.3)

                    axes[1].plot(epochs_ax, history["train_acc"], label="train acc", linewidth=1.5)
                    axes[1].plot(epochs_ax, history["val_acc"], label="val acc", linewidth=1.5, linestyle="--")
                    axes[1].plot(epochs_ax, history["val_f1"], label="val F1", linewidth=1.2, linestyle=":")
                    axes[1].set_title("Accuracy & F1")
                    axes[1].legend()
                    axes[1].grid(alpha=0.3)

                    fig.tight_layout()
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

                cm = confusion_matrix(y_test, y_pred)
                sz = max(6, len(class_names))
                fig, ax = plt.subplots(figsize=(sz, sz - 1))
                ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax, colorbar=False)
                ax.set_title("Confusion matrix")
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

                for method_name, pts3 in projections:
                    pts2 = pts3[:, :2]
                    fig = plt.figure(figsize=(16, 6))
                    fig.suptitle(f"{method_name} — embedding space", fontsize=12, y=1.01)

                    ax2 = fig.add_subplot(1, 2, 1)
                    ax2.scatter(pts2[:, 0], pts2[:, 1], c=pt_colors, s=12, alpha=0.6)
                    ax2.set_title(f"{method_name} 2-D")

                    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
                    ax3.scatter(pts3[:, 0], pts3[:, 1], pts3[:, 2], c=pt_colors, s=8, alpha=0.5)
                    ax3.set_title(f"{method_name} 3-D")

                    fig.tight_layout()
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

            print(f"  Full report    → {pdf_path}")
            return pdf_path, csv_path, html_paths

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

    # -------------------------------------------------------------------------
    # Per-epoch validation
    # -------------------------------------------------------------------------
    def _eval_model_nohalo(self, model, X_val, y_val, criterion, is_supcon=False):
        """
        Runs a full validation pass on (X_val, y_val) without the sklearn wrapper.
        Returns (val_loss, val_acc, val_f1_weighted).
        Returns (nan, nan, nan) when no validation data is provided.
        """
        if X_val is None or y_val is None or len(X_val) == 0:
            return float("nan"), float("nan"), float("nan")
 
        model.eval()
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(self.device)
        total_loss, correct, total = 0.0, 0, 0
        all_preds = []
 
        with torch.no_grad():
            for i in range(0, len(X_val), 256):
                bx = torch.tensor(X_val[i:i + 256], dtype=torch.float32).to(self.device)
                by = y_val_t[i:i + 256]
                logits, features = model(bx)
                total_loss += criterion(logits, by).item() * len(by)
                preds = logits.argmax(dim=1)
                correct += (preds == by).sum().item()
                all_preds.append(preds.cpu().numpy())
                total += len(by)
 
        all_preds = np.concatenate(all_preds)
        return (
            total_loss / total,
            correct / total,
            f1_score(y_val, all_preds, average="weighted", zero_division=0),
        )

    def _eval_model(self, model, X_val, y_val, criterion, mode="ce", centroid_targets=None):
        if X_val is None or y_val is None or len(X_val) == 0:
            return float("nan"), float("nan"), float("nan")

        model.eval()
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(self.device)
        total_loss, correct, total = 0.0, 0, 0
        all_preds = []

        with torch.no_grad():
            for i in range(0, len(X_val), 256):
                bx = torch.tensor(X_val[i:i + 256], dtype=torch.float32).to(self.device)
                by = y_val_t[i:i + 256]
                
                if mode == "halo":
                    embeddings, centroids = model(bx)
                    loss, _ = criterion(embeddings, by, centroids, centroid_targets)
                    total_loss += loss.item() * len(by)
                    
                    distances = torch.cdist(embeddings, centroids)
                    preds = distances.argmin(dim=1)
                else: # CE
                    logits, features = model(bx)
                    total_loss += criterion(logits, by).item() * len(by)
                    preds = logits.argmax(dim=1)
                    
                correct += (preds == by).sum().item()
                all_preds.append(preds.cpu().numpy())
                total += len(by)

        all_preds = np.concatenate(all_preds)
        return (total_loss / total, correct / total, f1_score(y_val, all_preds, average="weighted", zero_division=0))

    # --------------------------------------------
    # Training loop (CE)
    # --------------------------------------------
    def _run_training_loop_ce(self, model, X_train, y_train_t, optimizer, criterion, epochs, batch_size, X_val=None, y_val=None):
        n_samples = X_train.shape[0]
        is_supcon_model = isinstance(model, (MLP_SupCon, MLP_Attention_SupCon))
        history = {k: [] for k in ("epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1")}

        for epoch in range(epochs):
            model.train()
            ep_loss, correct, total = 0.0, 0, 0
            perm = torch.randperm(n_samples)

            for i in range(0, n_samples, batch_size):
                indices = perm[i:i + batch_size]
                batch_x = torch.tensor(X_train[indices.cpu().numpy()], dtype=torch.float32).to(self.device)
                batch_y = y_train_t[indices]
 
                if epoch == 0 and i == 0:
                    print(f"\n  [Training data] batch_x: {batch_x.shape}, batch_y: {batch_y.shape}")
 
                # Forward
                optimizer.zero_grad()
                logits, features = model(batch_x)
                
                # Loss computation
                loss = criterion(logits, batch_y)
                
                # Backpropagation
                loss.backward()
                optimizer.step()
 
                # Mean of the batch times the examples of the batch --> total loss
                ep_loss += loss.item() * len(batch_y)

                # Accumulating examples seen across the epoch
                total += len(batch_y)

                # Computing correct predictions for accuracy computation
                correct += (logits.argmax(dim=1) == batch_y).sum().item()
 
            # Computing total metrics for epoch
            train_loss = ep_loss / total
            train_acc = correct / total

            # Evaluating the model
            val_loss, val_acc, val_f1 = self._eval_model(
                model, X_val, y_val, criterion, mode="ce"
            )
 
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)
 
            if (epoch + 1) % 10 == 0:
                val_str = f"val loss {val_loss:.4f}  acc {val_acc:.4f}  f1 {val_f1:.4f}" if not np.isnan(val_loss) else "no val"
                print(f"Epoch {epoch+1}/{epochs} | Loss {train_loss:.4f}  Acc {train_acc:.4f} | {val_str}")
 
        return history            
    
    # -----------------------------------------
    # Training loop (CE + SupCon)
    # ------------------------------------------
    def _run_training_loop_supcon(self, model, X_train, y_train_t, optimizer,
                                   criterion_ce, criterion_supcon, alpha, epochs, batch_size, X_val=None, y_val=None):
        n_samples = X_train.shape[0]
        history = {k: [] for k in (
            "epoch", "train_loss", "train_loss_ce", "train_loss_supcon",
            "train_acc", "val_loss", "val_acc", "val_f1",
        )}


        for epoch in range(epochs):
            model.train()
            # Reset per-epoch accumulators inside the loop
            ep_loss = ep_ce = ep_sc = correct = total = 0
 
            perm = torch.randperm(n_samples)
            for i in range(0, n_samples, batch_size):
                indices = perm[i:i + batch_size]
                batch_x = torch.tensor(X_train[indices.cpu().numpy()], dtype=torch.float32).to(self.device)
                batch_y = y_train_t[indices]
 
                if epoch == 0 and i == 0:
                    print(f"\n  [Training data] batch_x: {batch_x.shape}, batch_y: {batch_y.shape}")
 
                optimizer.zero_grad()
                logits, norm_feat = model(batch_x)
 
                loss_ce = criterion_ce(logits, batch_y)
                loss_sc = criterion_supcon(norm_feat, batch_y)

                loss = loss_ce + alpha * loss_sc
 
                loss.backward()
                optimizer.step()

                ce_val = loss_ce.item()
                sc_val = loss_sc.item()
 
                nb = len(batch_y)
                ep_loss += loss.item() * nb
                ep_ce   += ce_val * nb
                ep_sc   += sc_val * nb
                correct += (logits.argmax(dim=1) == batch_y).sum().item()
                total   += nb
 
            train_loss = ep_loss / total
            train_acc  = correct / total
            raw_ce = ep_ce / total
            raw_sc = ep_sc / total
            val_loss, val_acc, val_f1 = self._eval_model(model, X_val, y_val, criterion_ce, mode="supcon")
 
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(train_loss)
            history["train_loss_ce"].append(raw_ce)
            history["train_loss_supcon"].append(raw_sc)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)
 
            if (epoch + 1) % 10 == 0:
                val_str = f"val loss {val_loss:.4f}  acc {val_acc:.4f}  f1 {val_f1:.4f}" if not np.isnan(val_loss) else "no val"
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"CE {raw_ce:.4f}  SC {raw_sc:.4f}  "
                      f"Loss {train_loss:.4f}  Acc {train_acc:.4f} | {val_str}")
 
        return history

    # --------------------------------------------
    # Training loop (HALO)
    # --------------------------------------------
    def _run_training_loop_halo(self, model, X_train, y_train_t, optimizer, criterion, epochs, batch_size, X_val=None, y_val=None):
        n_samples = X_train.shape[0]
        history = {k: [] for k in ("epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1")}
        
        num_classes = len(self.label_encoder.classes_)
        centroids_targets = torch.arange(num_classes, device=self.device)

        for epoch in range(epochs):
            model.train()
            ep_loss, correct, total = 0.0, 0, 0
            perm = torch.randperm(n_samples)

            for i in range(0, n_samples, batch_size):
                indices = perm[i:i + batch_size]
                batch_x = torch.tensor(X_train[indices.cpu().numpy()], dtype=torch.float32).to(self.device)
                batch_y = y_train_t[indices]
 
                if epoch == 0 and i == 0:
                    print(f"\n  [Training data] batch_x: {batch_x.shape}, batch_y: {batch_y.shape}")
 
                # Forward
                optimizer.zero_grad()
                embeddings, centroids = model(batch_x)
                
                # Loss computation
                loss, logits_true = criterion(embeddings, batch_y, centroids, centroids_targets)
                
                # Backpropagation
                loss.backward()
                optimizer.step()
 
                # Mean of the batch times the examples of the batch --> total loss
                ep_loss += loss.item() * len(batch_y)

                # Accumulating examples seen across the epoch
                total += len(batch_y)

                distances = torch.cdist(embeddings, centroids)
                correct += (distances.argmin(dim=1) == batch_y).sum().item()


            # Computing total metrics for epoch
            train_loss = ep_loss / total
            train_acc = correct / total

            val_loss, val_acc, val_f1 = self._eval_model(
                model, X_val, y_val, criterion, mode="halo", centroid_targets=centroids_targets
            )

            history["epoch"].append(epoch + 1)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)
 
            if (epoch + 1) % 10 == 0:
                val_str = f"val loss {val_loss:.4f}  acc {val_acc:.4f}  f1 {val_f1:.4f}" if not np.isnan(val_loss) else "no val"
                print(f"Epoch {epoch+1}/{epochs} | Loss {train_loss:.4f}  Acc {train_acc:.4f} | {val_str}")
 
        return history            
            
    # ----------------------------------------
    # Trainging (CE)
    # ----------------------------------------
    def train_classifier(self, epochs=100, batch_size=64, lr=0.001, criterion=nn.CrossEntropyLoss()):
        X, labels, song_names = self.load_feature_space()

        # sep_score, entropy_scores = calculate_linear_separability(X, labels)
 
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
 
        le = LabelEncoder()
        y = le.fit_transform(labels)
        self.label_encoder = le
        print(f"\n[Label Encoding] {len(le.classes_)} classes -> {le.classes_}")
        print(f"\nTraining on {X.shape[0]} chunks. Shape: {X.shape}")
 
        # Preprocessing
        print(f"  Before Normalizer: mean={np.mean(X):.4f}, std={np.std(X):.4f}")
        X = self._normalize_X(X, fit=True)
        print(f"  After Normalizer:  mean={np.mean(X):.4f}, std={np.std(X):.4f}")
        X = self._apply_pca(X, fit=True)
 
        # Train/test split (song-aware)
        X_train, X_test, y_train, y_test = self._song_aware_split(X, y, song_names)
        del X; gc.collect()

        history = None
        if self.classifier_type == "mlp":
            input_dim = self._get_input_dim(X_train)
            model = _build_mlp_model(input_dim, len(le.classes_), self.attention, mode="ce").to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            y_train_t = torch.tensor(y_train, dtype=torch.long).to(self.device) # torch.long==int64
            history = self._run_training_loop_ce(model, X_train, y_train_t, optimizer, criterion, epochs, batch_size,
                X_val=X_test, y_val=y_test)
            self.classifier = _wrap_model(model, self.device, le.classes_, mode="ce")
        else:
            print(f"Fitting {self.classifier_type.upper()}...")
            self.classifier = self._build_classifier(X_train.shape[1] if X_train.ndim == 2 else X_train.shape[2])
            self.classifier.fit(X_train, y_train)
 
        y_pred = self.classifier.predict(X_test)
        
        
        save_dir = (f"/dataslow/storage/Experiments/INTERNS/anavarror/trainedmodels_containfrag/{self.only_name_model}"
            f"attention{self.attention}_pca{self.use_pca}_batch{batch_size}_lr{lr}_{self.name_genres}"
        )
        prefix = f"{self.only_name_model}_epochs{epochs}_lr{lr}_batch{batch_size}"
 
        self._save_run_report(
            history, y_test, y_pred, le, X_train, y_train,
            save_dir=save_dir, prefix=prefix, mode="ce",
        )

        if not self.classifier_path:
            name_classifier = prefix + ".joblib"
            self.classifier_path = os.path.join(save_dir, name_classifier)

        self._save_pipeline(le)

    # ----------------------------------------
    # Training (CE + Supcon)
    # ----------------------------------------   
    def train_classifier_supcon(self, epochs=100, batch_size=64, lr=0.001,
                                 criterion_ce=nn.CrossEntropyLoss(),
                                 criterion_supcon=losses.SupConLoss(),
                                 alpha=0.1):
        X, labels, song_names = self.load_feature_space()

        # sep_score, entropy_scores = calculate_linear_separability(X, labels)
 
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
 
        X_train, X_test, y_train, y_test = self._song_aware_split(X, y, song_names)
        del X; gc.collect()

        history = None
        if self.classifier_type == "mlp":
            input_dim = self._get_input_dim(X_train)
            # _build_mlp_model with supcon=True always returns a model with (logits, features) output
            model = _build_mlp_model(input_dim, len(le.classes_), self.attention, mode="supcon").to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            y_train_t = torch.tensor(y_train, dtype=torch.long).to(self.device)
            history = self._run_training_loop_supcon(model, X_train, y_train_t, optimizer,
                criterion_ce, criterion_supcon, alpha, epochs, batch_size,
                X_val=X_test, y_val=y_test)
            self.classifier = _wrap_model(model, self.device, le.classes_, mode="supcon")
        else:
            print(f"Fitting {self.classifier_type.upper()} (SupCon not applicable for sklearn classifiers)...")
            self.classifier = self._build_classifier(X_train.shape[1] if X_train.ndim == 2 else X_train.shape[2])
            self.classifier.fit(X_train, y_train)
 
        y_pred = self.classifier.predict(X_test)
        self.label_encoder = le
        save_dir = (f"/dataslow/storage/Experiments/INTERNS/anavarror/trainedmodels_containfrag_supcon/{self.only_name_model}"
            f"attention{self.attention}_pca{self.use_pca}_batch{batch_size}_lr{lr}_alpha{alpha}_{self.name_genres}"
        )
        prefix = f"{self.only_name_model}_epochs{epochs}_lr{lr}_alpha{alpha}_batch{batch_size}"
 
        self._save_run_report(
            history, y_test, y_pred, le, X_train, y_train,
            save_dir=save_dir, prefix=prefix, mode="supcon",
        )

        if not self.classifier_path:
            name_classifier = prefix + ".joblib"
            self.classifier_path = os.path.join(save_dir, name_classifier)
        self._save_pipeline(le)

    # ----------------------------------------
    # Trainging (HALO)
    # ----------------------------------------
    def train_classifier_halo(self, epochs=100, batch_size=64, lr=0.001):
        X, labels, song_names = self.load_feature_space()

        # sep_score, entropy_scores = calculate_linear_separability(X, labels)
 
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
 
        le = LabelEncoder()
        y = le.fit_transform(labels)
        self.label_encoder = le
        print(f"\n[Label Encoding] {len(le.classes_)} classes -> {le.classes_}")
        print(f"\nTraining on {X.shape[0]} chunks. Shape: {X.shape}")
 
        # Preprocessing
        print(f"  Before Normalizer: mean={np.mean(X):.4f}, std={np.std(X):.4f}")
        X = self._normalize_X(X, fit=True)
        print(f"  After Normalizer:  mean={np.mean(X):.4f}, std={np.std(X):.4f}")
        X = self._apply_pca(X, fit=True)
 
        # Train/test split (song-aware)
        X_train, X_test, y_train, y_test = self._song_aware_split(X, y, song_names)
        del X; gc.collect()

        history = None
        input_dim = self._get_input_dim(X_train)
        num_classes = len(le.classes_)
        model = _build_mlp_model(input_dim, num_classes, self.attention, mode="halo").to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = HALOLoss(emb_dims=128, num_classes=num_classes).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(self.device)
        history = self._run_training_loop_halo(model, X_train, y_train_t, optimizer, criterion, epochs, batch_size,
            X_val=X_test, y_val=y_test)
        
        self.classifier = _wrap_model(model, self.device, le.classes_, mode="halo")
        y_pred = self.classifier.predict(X_test)
        self.label_encoder = le
        
        save_dir = (f"/dataslow/storage/Experiments/INTERNS/anavarror/trainedmodels_containfrag_halo/{self.only_name_model}"
            f"attention{self.attention}_pca{self.use_pca}_batch{batch_size}_lr{lr}_{self.name_genres}"
        )
        prefix = f"{self.only_name_model}_epochs{epochs}_lr{lr}_batch{batch_size}"
 
        self._save_run_report(
            history, y_test, y_pred, le, X_train, y_train,
            save_dir=save_dir, prefix=prefix, mode="halo",
        )

        if not self.classifier_path:
            name_classifier = prefix + ".joblib"
            self.classifier_path = os.path.join(save_dir, name_classifier)
        
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
        print(f"\nSaved model pipeline to {self.classifier_path}")
    
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