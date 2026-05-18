import os
import torch
# import librosa
import torchaudio
import numpy as np
import pandas as pd
import joblib
import csv
import time
import gc   
# import umap as umap_lib
import plotly.express as px
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, Normalizer
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE 
from transformers import AutoFeatureExtractor, AutoModel
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
import torch.nn as nn
import torch.optim as optim
import unicodedata
from collections import Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis   
import torch.nn.functional as F
from pytorch_metric_learning import losses
from matplotlib.backends.backend_pdf import PdfPages
#NVIDIA PLOTTING
from cuml.decomposition import PCA
from cuml.manifold import TSNE
from cuml.manifold import UMAP
from cuml.svm import LinearSVC
import plotly.graph_objects as go

from utils.attention import SelfAttentionPooling 
from utils.halo import HALOLoss

def calculate_linear_separability(embeddings, labels, max_samples=50000):
    """
    Adaptation of Linear Separability metric of StyleGAN to music genres
    Compute Conditional Entropy H(Y|X) using linear SVMs One-vs-Rest
    """
    # if torch.is_tensor(embeddings):
    #     embeddings = embeddings.cpu().numpy()
    
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
        binary_labels = (labels == genre).astype(int)
        
        # Train SVM
        # svm = LinearSVC(random_state=42, max_iter=2000, dual=False)
        svm = LinearSVC(max_iter=2000)
        svm.fit(embeddings, binary_labels)
        predictions = svm.predict(embeddings)
        
        #  Computing Conditional Entropy H(Y|X)
        h_y_given_x = 0.0
        
        for x_val in [0, 1]:
            idx_x = (predictions == x_val)
            count_x = np.sum(idx_x)
            
            if count_x == 0:
                continue
                
            p_x = count_x / total_samples
            
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
        print(f"  - {genre} vs Rest | H(Y|X) = {h_y_given_x:.6f}")

    separability_score = np.exp(total_entropy)
    
    print("-" * 40)
    print(f"Total Conditional Entropy (logarithmic): {total_entropy:.4f}")
    print(f"FINAL SEPARABILITY SCORE (linear): {separability_score:.4f}")
    
    return total_entropy, separability_score, entropy_scores

# -----------------------------------------------------------------------------
# MODEL DEFINITIONS
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, dropout):
        super(MLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.head(features)
        return logits, features
    
class MLP_Attention(nn.Module):
    def __init__(self, input_dim, num_classes, dropout):
        super(MLP_Attention, self).__init__()
        self.attention_pool = SelfAttentionPooling(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        pooled_x = self.attention_pool(x) 
        features = self.encoder(pooled_x)
        logits = self.head(features)
        return logits, features
    
class MLP_SupCon(nn.Module):
    def __init__(self, input_dim, num_classes, dropout):
        super(MLP_SupCon, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
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
        projected = self.projection_head(features)
        normalized_features = F.normalize(projected, p=2, dim=1)
        logits = self.classifier_head(features) 
        return logits, normalized_features

class MLP_Attention_SupCon(nn.Module):
    def __init__(self, input_dim, num_classes, dropout):
        super().__init__()
        self.attention_pool = SelfAttentionPooling(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.head = nn.Linear(128, num_classes)
 
    def forward(self, x):
        pooled = self.attention_pool(x)                         
        features = self.encoder(pooled)                          
        normalized_features = F.normalize(features, p=2, dim=1) 
        logits = self.head(features)                             
        return logits, normalized_features

class MLP_Halo(nn.Module):
    def __init__(self, input_dim, num_classes, dropout):
        super(MLP_Halo, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.centroids = nn.Parameter(torch.randn(num_classes, 128, dtype=torch.float32))
    
    def forward(self, x):
        embeddings = self.encoder(x)
        centroids = self.centroids - self.centroids.mean(dim=0, keepdim=True)
        return embeddings, centroids
    
class MLP_Attention_Halo(nn.Module):
    def __init__(self, input_dim, num_classes, dropout):
        super(MLP_Attention_Halo, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
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
                batch = X[i:i+batch_size]
                if isinstance(batch, np.ndarray):
                    batch = torch.tensor(batch, dtype=torch.float32)
                batch = batch.to(self.device)
                logits, features = self.model(batch)
                batch_preds = torch.argmax(logits, dim=1)
                preds_list.append(batch_preds.cpu())
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
                batch = X[i:i+batch_size]
                if isinstance(batch, np.ndarray):
                    batch = torch.tensor(batch, dtype=torch.float32)
                batch = batch.to(self.device)
                logits, normalized_features = self.model(batch)
                batch_preds = torch.argmax(logits, dim=1)
                preds_list.append(batch_preds.cpu())
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
                probs = torch.softmax(-distances, dim=1)
                probs_list.append(probs.cpu())
        return torch.cat(probs_list).numpy()
    
# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def _build_mlp_model(input_dim, num_classes, attention: bool, mode: str, dropout : float):
    if mode == "ce":
        return MLP_Attention(input_dim, num_classes, dropout) if attention else MLP(input_dim, num_classes, dropout)
    elif mode == "supcon":
        return MLP_Attention_SupCon(input_dim, num_classes, dropout) if attention else MLP_SupCon(input_dim, num_classes, dropout)
    elif mode == "halo":
        return MLP_Attention_Halo(input_dim, num_classes, dropout) if attention else MLP_Halo(input_dim, num_classes, dropout)

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
                # self.pca = PCA(n_components=self.pca_components, random_state=42)
                self.pca = PCA(n_components=self.pca_components)
                X_flat = self.pca.fit_transform(X_flat)
            else:
                X_flat = self.pca.transform(X_flat)
            return X_flat.reshape(N, T, self.pca_components)
        else:
            if fit:
                # self.pca = PCA(n_components=self.pca_components, random_state=42)
                self.pca = PCA(n_components=self.pca_components)
                return self.pca.fit_transform(X)
            return self.pca.transform(X)

    def _get_input_dim(self, X):
        """Returns the feature dimension regardless of whether X is 2D or 3D."""
        return X.shape[2] if X.ndim == 3 else X.shape[1] 

    def _save_run_report(
        self,
        history,        # dict returned by a training loop
        y_test,         # ground-truth labels (int)
        y_pred,         # predicted labels (int)
        y_prob,         # predicted probabilities (float) - used for Song-Level
        le,             # fitted LabelEncoder
        X_train,        # numpy array used for embedding plots
        y_train,        # int labels for X_train
        sn_test,        # song names corresponding to y_test (for song aggregation)
        save_dir,       # directory where both files land
        prefix,         # filename prefix
        mode="ce",      # whether to show CE/SupCon sub-losses
    ):
            os.makedirs(save_dir, exist_ok=True)
            csv_path = os.path.join(save_dir, f"{prefix}_training_history.csv")
            has_history = bool(history and history.get("epoch"))
            if has_history:
                pd.DataFrame(history).to_csv(csv_path, index=False)

            class_names = le.classes_

            # CHUNK-LEVEL METRICS 
            acc_chunk        = accuracy_score(y_test, y_pred)
            f1_w_chunk       = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            f1_m_chunk       = f1_score(y_test, y_pred, average="macro",    zero_division=0)
            report_str_chunk = classification_report(y_test, y_pred, target_names=class_names)

            # SONG-LEVEL METRICS (Mean of Chunks)
            unique_songs = np.unique(sn_test)
            y_test_song = []
            y_pred_song = []
            
            for song in unique_songs:
                # Find all chunks belonging to this specific song
                idx = np.where(sn_test == song)[0]
                
                # The true label is the same for all chunks of the song
                y_test_song.append(y_test[idx[0]])
                
                if y_prob is not None:
                    # Average the probabilities across all chunks and pick the max
                    mean_probs = np.mean(y_prob[idx], axis=0)
                    y_pred_song.append(np.argmax(mean_probs))
                else:
                    # Fallback to majority vote if probabilities aren't passed
                    chunk_preds = y_pred[idx]
                    most_common = Counter(chunk_preds).most_common(1)[0][0]
                    y_pred_song.append(most_common)
                    
            y_test_song = np.array(y_test_song)
            y_pred_song = np.array(y_pred_song)

            acc_song        = accuracy_score(y_test_song, y_pred_song)
            f1_w_song       = f1_score(y_test_song, y_pred_song, average="weighted", zero_division=0)
            f1_m_song       = f1_score(y_test_song, y_pred_song, average="macro",    zero_division=0)
            report_str_song = classification_report(y_test_song, y_pred_song, target_names=class_names)

            y_train_names = le.inverse_transform(y_train)
            all_classes = sorted(le.classes_)
            cmap = plt.get_cmap("tab20" if len(all_classes) <= 20 else "hsv")
            color_map = {l: cmap(i / len(all_classes)) for i, l in enumerate(all_classes)}
            color_map_hex = {l: mcolors.to_hex(color_map[l]) for l in all_classes}
            
            pt_colors = [color_map[l] for l in y_train_names]
            legend_handles = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[l], markersize=8)
                for l in all_classes
            ]

            print("\n  Extracting learned latent space features for plotting...")
            self.classifier.model.eval()
            learned_features = []
            with torch.no_grad():
                for i in range(0, len(X_train), 256):
                    bx = torch.tensor(X_train[i:i+256], dtype=torch.float32).to(self.device)
                    out = self.classifier.model(bx)
                    
                    feats = out[0] if mode == "halo" else out[1]
                    learned_features.append(feats.cpu().numpy())
                    
            X_train_learned = np.concatenate(learned_features, axis=0)
            emb_flat = X_train_learned.mean(axis=1) if X_train_learned.ndim == 3 else X_train_learned

            projections = []
            
            # PCA
            # pca3 = PCA(n_components=3, random_state=42).fit_transform(emb_flat) # CPU
            pca3 = PCA(n_components=3).fit_transform(emb_flat) # CUML NVIDIA
            projections.append(("PCA", pca3))
            
            # t-SNE
            try:
                # tsne3 = TSNE(n_components=3, random_state=42, init="pca", learning_rate="auto").fit_transform(emb_flat) # CPU
                tsne2 = TSNE(n_components=2, random_state=42, init="pca", learning_rate=200.0).fit_transform(emb_flat) # CUML NVIDIA
                tsne3 = np.hstack([tsne2, np.zeros((tsne2.shape[0], 1))]) # CUML NVIDIA
                projections.append(("t-SNE", tsne3))
            except Exception as e:
                print(f"  [Warning] t-SNE failed: {e}")
                
            # LDA
            try:
                n_lda = min(3, len(all_classes) - 1)
                lda_pts = LinearDiscriminantAnalysis(n_components=n_lda).fit_transform(emb_flat, y_train_names)
                if lda_pts.shape[1] < 3:
                    lda_pts = np.hstack([lda_pts, np.zeros((lda_pts.shape[0], 3 - lda_pts.shape[1]))])
                projections.append(("LDA", lda_pts))
            except Exception as e:
                print(f"  [Warning] LDA failed: {e}")
                
            # UMAP
            # try:
            #     emb_flat = emb_flat + np.random.normal(0, 1e-6, emb_flat.shape)
            #     umap3 = UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42).fit_transform(emb_flat) # same line for CPU and CUML NVIDIA
            #     projections.append(("UMAP", umap3))
            except Exception as e:
                print(f"  [Warning] UMAP failed: {e}")

            entropy_score, sep_score, entropy_scores = calculate_linear_separability(emb_flat, y_train_names)

            html_paths = []
            for method_name, pts3 in projections:
                df_emb = pd.DataFrame({"C1": pts3[:, 0], "C2": pts3[:, 1], "C3": pts3[:, 2], "Genre": y_train_names})
                fig_html = px.scatter_3d(df_emb, x="C1", y="C2", z="C3", color="Genre", color_discrete_map=color_map_hex, 
                                         title=f"{method_name} — 3-D embedding space", opacity=0.75)
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
                    f"--- CHUNK-LEVEL METRICS ---\n"
                    f"Accuracy   : {acc_chunk:.4f}\n"
                    f"F1 weighted: {f1_w_chunk:.4f}\n"
                    f"F1 macro   : {f1_m_chunk:.4f}\n\n"
                    f"--- SONG-LEVEL METRICS ---\n"
                    f"Accuracy   : {acc_song:.4f}\n"
                    f"F1 weighted: {f1_w_song:.4f}\n"
                    f"F1 macro   : {f1_m_song:.4f}\n\n"
                    f"Entropy score (logarithmic)    : {entropy_score:.4f}\n"
                    f"Separability score (linear)    : {sep_score:.4f}\n\n"
                    f"{'='*60}\nClassification report (Chunk-Level)\n{'='*60}\n{report_str_chunk}\n\n"
                    f"{'='*60}\nClassification report (Song-Level)\n{'='*60}\n{report_str_song}\n\n"
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

                # PLOT DUAL CONFUSION MATRICES (CHUNK VS SONG LEVEL)
                cm_chunk = confusion_matrix(y_test, y_pred)
                cm_song = confusion_matrix(y_test_song, y_pred_song)
                
                sz = max(6, len(class_names))
                fig, axes = plt.subplots(1, 2, figsize=(sz * 2, sz - 1))
                
                ConfusionMatrixDisplay(cm_chunk, display_labels=class_names).plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=axes[0], colorbar=False)
                axes[0].set_title("Confusion Matrix (Chunk-Level)")
                
                ConfusionMatrixDisplay(cm_song, display_labels=class_names).plot(cmap=plt.cm.Greens, xticks_rotation=45, ax=axes[1], colorbar=False)
                axes[1].set_title("Confusion Matrix (Song-Level)")
                
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
    def load_feature_space(self, mode='train'): 
        # If it is a .pt file
        if os.path.isfile(self.feature_space_path):
            if mode == 'train': # Print only in training
                print(f"\nLoading features from file: {self.feature_space_path}")
            data = torch.load(self.feature_space_path, map_location="cpu")
            if mode == 'train': # Print only in training
                print(f"[Data Load] Loaded {data['embeddings'].shape[0]} samples with dimension {data['embeddings'].shape[1]}")
            return (data["embeddings"], data["labels"], data.get("song_names"))
        
        # If it is a directory
        elif os.path.isdir(self.feature_space_path):
            print(f"\nLoading features from directory: {self.feature_space_path}")
            all_embeddings, all_labels, all_song_names = [], [], []

            for genre in self.genre_list:
                safe_genre = self._sanitize_filename(genre)
                filename = f"{self.only_name_model}_attention{self.attention}_{safe_genre}.pt"
                file_path = os.path.join(self.feature_space_path, filename)
                
                if os.path.exists(file_path):
                    # print(f"  Merging file: {filename}")
                    data = torch.load(file_path, map_location="cpu")
                    if mode == 'train': # Print only in training
                        print(f"    Merging genre {data['labels'][0]} with {data['embeddings'].size(0)} chunks") 
                    
                    all_embeddings.append(data["embeddings"])
                    all_labels.extend(data["labels"])
                    if "song_names" in data:
                        all_song_names.extend(data["song_names"])
                else:
                    print(f"  Warning: Feature space file not found for genre '{genre}' (Expected: {filename})")
            
            if not all_embeddings:
                raise FileNotFoundError(f"No matching .pt files found in {self.feature_space_path} for the provided genre list.")

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
            return None 

    # -------------------------------------------------------------------------
    # Per-epoch validation
    # -------------------------------------------------------------------------
    def _eval_model(self, model, X_val, y_val, criterion, mode="ce", centroid_targets=None):
        if X_val is None or y_val is None or len(X_val) == 0:
            return float("nan"), float("nan"), float("nan")

        model.eval()
        
        if not torch.is_tensor(X_val):
            X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        else:
            X_val_t = X_val.to(dtype=torch.float32, device=self.device)
            
        if not torch.is_tensor(y_val):
            y_val_t = torch.tensor(y_val, dtype=torch.long, device=self.device)
        else:
            y_val_t = y_val.to(dtype=torch.long, device=self.device)

        total_loss, correct, total = 0.0, 0, 0
        all_preds = []

        with torch.no_grad():
            for i in range(0, len(X_val_t), 256):
                bx = X_val_t[i:i + 256] 
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
    def _run_training_loop_ce(self, model, X_train, y_train_t, optimizer, criterion, epochs, batch_size, X_val=None, y_val=None, lr=0.001, dropout=0.6):
        n_samples = X_train.shape[0]
        history = {k: [] for k in ("epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1")}

        if not torch.is_tensor(X_train):
            X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        else:
            X_train_t = X_train.to(dtype=torch.float32, device=self.device)
            
        y_train_t = y_train_t.to(dtype=torch.long, device=self.device)

        # ----- CLASS BALANCING -----
        class_counts = torch.bincount(y_train_t)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[y_train_t] # Asign to each sample their probability
        print(f"\nComputed weights:")
        for i, genre_name in enumerate(self.label_encoder.classes_):
            count = class_counts[i].item()
            weight = class_weights[i].item()
            print(f"  {genre_name} | Chunks: {count:<5} | Weight: {weight:.6f}")
        print("\n")

        # Report batch
        epoch1_batch_counts = []
        num_classes = len(self.label_encoder.classes_)

        for epoch in range(epochs):
            model.train()
            ep_loss, correct, total = 0.0, 0, 0
            
            # perm = torch.randperm(n_samples, device=self.device)
            perm = torch.multinomial(sample_weights, num_samples=n_samples, replacement=True) # Doing class balancing

            for i in range(0, n_samples, batch_size):
                indices = perm[i:i + batch_size]
                
                batch_x = X_train_t[indices]
                batch_y = y_train_t[indices]

                # Report batch
                if epoch == 0:
                    counts = torch.bincount(batch_y, minlength=num_classes).cpu().numpy()
                    epoch1_batch_counts.append(counts)

                optimizer.zero_grad()
                logits, features = model(batch_x)
                
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                ep_loss += loss.item() * len(batch_y)
                total += len(batch_y)
                correct += (logits.argmax(dim=1) == batch_y).sum().item()

            train_loss = ep_loss / total
            train_acc = correct / total

            # Report batch
            if epoch == 0:
                try:                    
                    pdf_folder = f"/dataslow/storage/Experiments/INTERNS/anavarror/trainedmodels_containfrag_ce/{self.only_name_model}_attention{self.attention}_pca{self.use_pca}_batch{batch_size}_lr{lr}_dropout{dropout}_{self.name_genres}"
                    pdf_path = os.path.join(pdf_folder, f"batch_balance_report_{self.classifier_type}.pdf")
                    os.makedirs(pdf_folder, exist_ok=True)
                    with PdfPages(pdf_path) as pdf:
                        # --- Line plot ---
                        counts_array = np.array(epoch1_batch_counts) # Shape: (num_batches, num_classes)
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        for c_idx, c_name in enumerate(self.label_encoder.classes_):
                            ax.plot(counts_array[:, c_idx], label=c_name, alpha=0.8, linewidth=1.5)
                            
                        ax.set_title(f"Batch distribution across all batches of epoch 1", fontsize=14, fontweight='bold')
                        ax.set_xlabel("Number of Batches", fontsize=12)
                        ax.set_ylabel("Amount of Chunks in Batch", fontsize=12)
                        
                        ideal_balance = batch_size / num_classes
                        ax.axhline(y=ideal_balance, color='black', linestyle='--', linewidth=2, label=f"Ideal Balance ({ideal_balance})")
                        
                        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        ax.grid(alpha=0.3)
                        plt.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)
                        
                        # --- Histogram of first 6 batches ---
                        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                        fig.suptitle("Histogram of first 6 Batches", fontsize=16, fontweight='bold')
                        axes = axes.flatten()
                        
                        for b_idx in range(min(6, len(epoch1_batch_counts))):
                            ax = axes[b_idx]
                            ax.bar(self.label_encoder.classes_, epoch1_batch_counts[b_idx], color='teal', edgecolor='black')
                            ax.set_title(f"Batch {b_idx+1}")
                            ax.set_xticklabels(self.label_encoder.classes_, rotation=45, ha='right', fontsize=8)
                            ax.axhline(y=ideal_balance, color='black', linestyle='--', alpha=0.5)
                            
                        plt.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)
                        
                    print(f"\n  [Batch Balancing] PDF report of balance batch saved to '{pdf_path}'\n")
                except Exception as e:
                    print(f"  [Warning] Failed doing PDF of Batch Balancing: {e}")

            val_loss, val_acc, val_f1 = self._eval_model(model, X_val, y_val, criterion, mode="ce")

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
                                   criterion_ce, criterion_supcon, alpha, epochs, batch_size, X_val=None, y_val=None, lr=0.001, dropout=0.6):
        n_samples = X_train.shape[0]
        history = {k: [] for k in (
            "epoch", "train_loss", "train_loss_ce", "train_loss_supcon",
            "train_acc", "val_loss", "val_acc", "val_f1",
        )}

        if not torch.is_tensor(X_train):
            X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        else:
            X_train_t = X_train.to(dtype=torch.float32, device=self.device)
            
        y_train_t = y_train_t.to(dtype=torch.long, device=self.device)

        # ----- CLASS BALANCING -----
        class_counts = torch.bincount(y_train_t)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[y_train_t] # Asign to each sample their probability
        print(f"\nComputed weights:")
        for i, genre_name in enumerate(self.label_encoder.classes_):
            count = class_counts[i].item()
            weight = class_weights[i].item()
            print(f"  {genre_name} | Chunks: {count:<5} | Weight: {weight:.6f}")
        print("\n")

        # Report batch
        epoch1_batch_counts = []
        num_classes = len(self.label_encoder.classes_)

        for epoch in range(epochs):
            model.train()
            ep_loss = ep_ce = ep_sc = correct = total = 0
            
            # perm = torch.randperm(n_samples, device=self.device) # No class balancing
            perm = torch.multinomial(sample_weights, num_samples=n_samples, replacement=True) # Doing class balancing
            
            for i in range(0, n_samples, batch_size):
                indices = perm[i:i + batch_size]
                
                batch_x = X_train_t[indices]
                batch_y = y_train_t[indices]

                # Report batch
                if epoch == 0:
                    counts = torch.bincount(batch_y, minlength=num_classes).cpu().numpy()
                    epoch1_batch_counts.append(counts)

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

            # Report batch
            if epoch == 0:
                try:                    
                    pdf_folder = f"/dataslow/storage/Experiments/INTERNS/anavarror/trainedmodels_containfrag_supcon/{self.only_name_model}_attention{self.attention}_pca{self.use_pca}_batch{batch_size}_lr{lr}_dropout{dropout}_alpha{alpha}_{self.name_genres}"
                    pdf_path = os.path.join(pdf_folder, f"batch_balance_report_{self.classifier_type}.pdf")
                    os.makedirs(pdf_folder, exist_ok=True)
                    with PdfPages(pdf_path) as pdf:
                        # --- Line plot ---
                        counts_array = np.array(epoch1_batch_counts) # Shape: (num_batches, num_classes)
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        for c_idx, c_name in enumerate(self.label_encoder.classes_):
                            ax.plot(counts_array[:, c_idx], label=c_name, alpha=0.8, linewidth=1.5)
                            
                        ax.set_title(f"Batch distribution across all batches of epoch 1", fontsize=14, fontweight='bold')
                        ax.set_xlabel("Number of Batches", fontsize=12)
                        ax.set_ylabel("Amount of Chunks in Batch", fontsize=12)
                        
                        ideal_balance = batch_size / num_classes
                        ax.axhline(y=ideal_balance, color='black', linestyle='--', linewidth=2, label=f"Ideal Balance ({ideal_balance})")
                        
                        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        ax.grid(alpha=0.3)
                        plt.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)
                        
                        # --- Histogram of first 6 batches ---
                        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                        fig.suptitle("Histogram of first 6 Batches", fontsize=16, fontweight='bold')
                        axes = axes.flatten()
                        
                        for b_idx in range(min(6, len(epoch1_batch_counts))):
                            ax = axes[b_idx]
                            ax.bar(self.label_encoder.classes_, epoch1_batch_counts[b_idx], color='teal', edgecolor='black')
                            ax.set_title(f"Batch {b_idx+1}")
                            ax.set_xticklabels(self.label_encoder.classes_, rotation=45, ha='right', fontsize=8)
                            ax.axhline(y=ideal_balance, color='black', linestyle='--', alpha=0.5)
                            
                        plt.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)
                        
                    print(f"\n  [Batch Balancing] PDF report of balance batch saved to '{pdf_path}'\n")
                except Exception as e:
                    print(f"  [Warning] Failed doing PDF of Batch Balancing: {e}")

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
    def _run_training_loop_halo(self, model, X_train, y_train_t, optimizer, criterion, epochs, batch_size, X_val=None, y_val=None, lr=0.001, dropout=0.6):
        n_samples = X_train.shape[0]
        history = {k: [] for k in ("epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1")}
        
        num_classes = len(self.label_encoder.classes_)
        centroids_targets = torch.arange(num_classes, device=self.device)

        if not torch.is_tensor(X_train):
            X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        else:
            X_train_t = X_train.to(dtype=torch.float32, device=self.device)
            
        y_train_t = y_train_t.to(dtype=torch.long, device=self.device)

        class_counts = torch.bincount(y_train_t)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[y_train_t] # Asign to each sample their probability
        print(f"\nComputed weights:")
        for i, genre_name in enumerate(self.label_encoder.classes_):
            count = class_counts[i].item()
            weight = class_weights[i].item()
            print(f"  {genre_name} | Chunks: {count:<5} | Weight: {weight:.6f}")
        print("\n")

        # REPORT BATCHES
        epoch1_batch_counts = []
        num_classes = len(self.label_encoder.classes_)

        for epoch in range(epochs):
            model.train()
            ep_loss, correct, total = 0.0, 0, 0
            
            # perm = torch.randperm(n_samples, device=self.device) # No class balancing
            perm = torch.multinomial(sample_weights, num_samples=n_samples, replacement=True) # Doing class balancing

            for i in range(0, n_samples, batch_size):
                indices = perm[i:i + batch_size]
                
                batch_x = X_train_t[indices]
                batch_y = y_train_t[indices]

                # Report batches
                if epoch == 0:
                    counts = torch.bincount(batch_y, minlength=num_classes).cpu().numpy()
                    epoch1_batch_counts.append(counts)

                optimizer.zero_grad()
                embeddings, centroids = model(batch_x)
                
                loss, logits_true = criterion(embeddings, batch_y, centroids, centroids_targets)
                
                loss.backward()
                optimizer.step()

                ep_loss += loss.item() * len(batch_y)
                total += len(batch_y)

                distances = torch.cdist(embeddings, centroids)
                correct += (distances.argmin(dim=1) == batch_y).sum().item()

            train_loss = ep_loss / total
            train_acc = correct / total

            # Report batch
            if epoch == 0:
                try:                 
                    pdf_folder = f"/dataslow/storage/Experiments/INTERNS/anavarror/trainedmodels_containfrag_halo/{self.only_name_model}_attention{self.attention}_pca{self.use_pca}_batch{batch_size}_lr{lr}_dropout{dropout}_{self.name_genres}"   
                    pdf_path = os.path.join(pdf_folder, f"batch_balance_report_{self.classifier_type}.pdf")
                    os.makedirs(pdf_folder, exist_ok=True)
                    with PdfPages(pdf_path) as pdf:
                        # --- Line plot ---
                        counts_array = np.array(epoch1_batch_counts) # Shape: (num_batches, num_classes)
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        for c_idx, c_name in enumerate(self.label_encoder.classes_):
                            ax.plot(counts_array[:, c_idx], label=c_name, alpha=0.8, linewidth=1.5)
                            
                        ax.set_title(f"Batch distribution across all batches of epoch 1", fontsize=14, fontweight='bold')
                        ax.set_xlabel("Number of Batches", fontsize=12)
                        ax.set_ylabel("Amount of Chunks in Batch", fontsize=12)
                        
                        ideal_balance = batch_size / num_classes
                        ax.axhline(y=ideal_balance, color='black', linestyle='--', linewidth=2, label=f"Ideal Balance ({ideal_balance})")
                        
                        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        ax.grid(alpha=0.3)
                        plt.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)
                        
                        # --- Histogram of first 6 batches ---
                        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                        fig.suptitle("Histogram of first 6 Batches", fontsize=16, fontweight='bold')
                        axes = axes.flatten()
                        
                        for b_idx in range(min(6, len(epoch1_batch_counts))):
                            ax = axes[b_idx]
                            ax.bar(self.label_encoder.classes_, epoch1_batch_counts[b_idx], color='teal', edgecolor='black')
                            ax.set_title(f"Batch {b_idx+1}")
                            ax.set_xticklabels(self.label_encoder.classes_, rotation=45, ha='right', fontsize=8)
                            ax.axhline(y=ideal_balance, color='black', linestyle='--', alpha=0.5)
                            
                        plt.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)
                        
                    print(f"\n  [Batch Balancing] PDF report of balance batch saved to '{pdf_path}'\n")
                except Exception as e:
                    print(f"  [Warning] Failed doing PDF of Batch Balancing: {e}")

            val_loss, val_acc, val_f1 = self._eval_model(model, X_val, y_val, criterion, mode="halo", centroid_targets=centroids_targets)

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
    def train_classifier_ce(self, epochs=100, batch_size=64, lr=0.001, criterion=nn.CrossEntropyLoss(), dropout=0.4):
        X, labels, song_names = self.load_feature_space()
 
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
        X_train, X_test, y_train, y_test, sn_train, sn_test = self._song_aware_split(X, y, song_names)
        del X; gc.collect()

        # #  ---- Optional Class Balancing for Cross-Entropy Loss ----
        # class_counts = np.bincount(y_train)
        # print(f"\n[Class Balancing] Chunks for each class: {class_counts}")
        # total_samples = len(y_train)
        # num_classes = len(le.classes_)
        
        # class_weights = total_samples / (num_classes * class_counts)
        # class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        # print(f"Computed weights: {class_weights}\n")
        # criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        history = None
        if self.classifier_type == "mlp":
            input_dim = self._get_input_dim(X_train)
            model = _build_mlp_model(input_dim, len(le.classes_), self.attention, mode="ce", dropout=dropout).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            y_train_t = torch.tensor(y_train, dtype=torch.long).to(self.device)
            history = self._run_training_loop_ce(model, X_train, y_train_t, optimizer, criterion, epochs, batch_size,
                X_val=X_test, y_val=y_test, lr=lr, dropout=dropout)
            self.classifier = _wrap_model(model, self.device, le.classes_, mode="ce")
        else:
            print(f"Fitting {self.classifier_type.upper()}...")
            self.classifier = self._build_classifier(X_train.shape[1] if X_train.ndim == 2 else X_train.shape[2])
            self.classifier.fit(X_train, y_train)
 
        y_pred = self.classifier.predict(X_test)
        y_prob = self.classifier.predict_proba(X_test) if hasattr(self.classifier, "predict_proba") else None
        
        save_dir = (f"/dataslow/storage/Experiments/INTERNS/anavarror/trainedmodels_containfrag_ce/{self.only_name_model}_"
            f"attention{self.attention}_pca{self.use_pca}_batch{batch_size}_lr{lr}_dropout{dropout}_{self.name_genres}"
        )
        prefix = f"{self.only_name_model}_epochs{epochs}_lr{lr}_batch{batch_size}"
 
        self._save_run_report(
            history, y_test, y_pred, y_prob, le, X_train, y_train, sn_test,
            save_dir=save_dir, prefix=prefix, mode="ce",
        )

        if not self.classifier_path:
            name_classifier = prefix + ".joblib"
            self.classifier_path = os.path.join(save_dir, name_classifier)

        self._save_pipeline(le)

    # ----------------------------------------
    # Training (CE + Supcon)
    # ----------------------------------------   
    def train_classifier_supcon(self, epochs=100, batch_size=64, lr=0.001, criterion_ce=nn.CrossEntropyLoss(),
        criterion_supcon=losses.SupConLoss(), alpha=0.1, dropout=0.4
    ):
        X, labels, song_names = self.load_feature_space()
 
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
 
        le = LabelEncoder()
        y = le.fit_transform(labels)
        self.label_encoder = le
        print(f"[Label Encoding] {len(le.classes_)} classes -> {le.classes_}")
        print(f"\nTraining on {X.shape[0]} chunks. Shape: {X.shape}")
 
        # Preprocessing
        print(f"  Before Normalizer: mean={np.mean(X):.4f}, std={np.std(X):.4f}")
        X = self._normalize_X(X, fit=True)
        print(f"  After Normalizer:  mean={np.mean(X):.4f}, std={np.std(X):.4f}")
        X = self._apply_pca(X, fit=True)
 
        X_train, X_test, y_train, y_test, sn_train, sn_test = self._song_aware_split(X, y, song_names)
        del X; gc.collect()

        history = None
        if self.classifier_type == "mlp":
            input_dim = self._get_input_dim(X_train)
            model = _build_mlp_model(input_dim, len(le.classes_), self.attention, mode="supcon", dropout=dropout).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            y_train_t = torch.tensor(y_train, dtype=torch.long).to(self.device)
            history = self._run_training_loop_supcon(model, X_train, y_train_t, optimizer,
                criterion_ce, criterion_supcon, alpha, epochs, batch_size,
                X_val=X_test, y_val=y_test, lr=lr, dropout=dropout)
            self.classifier = _wrap_model(model, self.device, le.classes_, mode="supcon")
        else:
            print(f"Fitting {self.classifier_type.upper()} (SupCon not applicable)...")
            self.classifier = self._build_classifier(X_train.shape[1] if X_train.ndim == 2 else X_train.shape[2])
            self.classifier.fit(X_train, y_train)
 
        y_pred = self.classifier.predict(X_test)
        y_prob = self.classifier.predict_proba(X_test) if hasattr(self.classifier, "predict_proba") else None
        
        self.label_encoder = le
        save_dir = (f"/dataslow/storage/Experiments/INTERNS/anavarror/trainedmodels_containfrag_supcon/{self.only_name_model}_"
            f"attention{self.attention}_pca{self.use_pca}_batch{batch_size}_lr{lr}_dropout{dropout}_alpha{alpha}_{self.name_genres}"
        )
        prefix = f"{self.only_name_model}_epochs{epochs}_lr{lr}_alpha{alpha}_batch{batch_size}"
 
        self._save_run_report(
            history, y_test, y_pred, y_prob, le, X_train, y_train, sn_test,
            save_dir=save_dir, prefix=prefix, mode="supcon",
        )

        if not self.classifier_path:
            name_classifier = prefix + ".joblib"
            self.classifier_path = os.path.join(save_dir, name_classifier)
        self._save_pipeline(le)

    # ----------------------------------------
    # Trainging (HALO)
    # ----------------------------------------
    def train_classifier_halo(self, epochs=100, batch_size=64, lr=0.001, dropout=0.2):
        X, labels, song_names = self.load_feature_space()
 
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
        X_train, X_test, y_train, y_test, sn_train, sn_test = self._song_aware_split(X, y, song_names)
        del X; gc.collect()

        history = None
        input_dim = self._get_input_dim(X_train)
        num_classes = len(le.classes_)
        model = _build_mlp_model(input_dim, num_classes, self.attention, mode="halo", dropout=dropout).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = HALOLoss(emb_dims=128, num_classes=num_classes).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(self.device)
        history = self._run_training_loop_halo(model, X_train, y_train_t, optimizer, criterion, epochs, batch_size,
            X_val=X_test, y_val=y_test, lr=lr, dropout=dropout)
        
        self.classifier = _wrap_model(model, self.device, le.classes_, mode="halo")
        y_pred = self.classifier.predict(X_test)
        y_prob = self.classifier.predict_proba(X_test) if hasattr(self.classifier, "predict_proba") else None
        self.label_encoder = le
        
        save_dir = (f"/dataslow/storage/Experiments/INTERNS/anavarror/trainedmodels_containfrag_halo/{self.only_name_model}_"
            f"attention{self.attention}_pca{self.use_pca}_batch{batch_size}_lr{lr}_dropout{dropout}_{self.name_genres}"
        )
        prefix = f"{self.only_name_model}_epochs{epochs}_lr{lr}_batch{batch_size}"
 
        self._save_run_report(
            history, y_test, y_pred, y_prob, le, X_train, y_train, sn_test,
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
            print("\n  Splitting by song ID")
            
            train_idx = []
            test_idx = []
            
            song_names_arr = np.array(song_names)
            unique_genres = np.unique(y)
            
            # Hacemos el split 90/10 independientemente para CADA género
            for genre in unique_genres:
                genre_indices = np.where(y == genre)[0]
                genre_songs = song_names_arr[genre_indices]
                unique_songs_in_genre = np.unique(genre_songs)
                
                np.random.seed(42) 
                np.random.shuffle(unique_songs_in_genre)
                
                split_point = int(len(unique_songs_in_genre) * 0.9) # 90/10 test split
                train_songs = unique_songs_in_genre[:split_point]
                test_songs = unique_songs_in_genre[split_point:]
                
                train_idx.extend(genre_indices[np.isin(genre_songs, train_songs)])
                test_idx.extend(genre_indices[np.isin(genre_songs, test_songs)])
            
            train_idx = np.array(train_idx)
            test_idx = np.array(test_idx)

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            sn_train, sn_test = song_names_arr[train_idx], song_names_arr[test_idx]
            
            print(f"  Chunks (Tensors) in Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

            train_songs_set = set(sn_train)
            test_songs_set = set(sn_test)
            print(f"  Songs in Train: {len(train_songs_set)}, Test: {len(test_songs_set)}, Leakage: {len(train_songs_set & test_songs_set)}")
        else:
            print("  WARNING: No song names — using random split (potential leakage).")
            X_train, X_test, y_train, y_test, sn_train, sn_test = train_test_split(X, y, song_names, test_size=0.1, stratify=y, random_state=42)
            
        return X_train, X_test, y_train, y_test, sn_train, sn_test

    def group_shuffle_split(self, X, y, song_names):
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
    def get_inference_chunks_old(self, mp3_path):
        try:
            waveform, sr = torchaudio.load(mp3_path)
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)
                
            if sr != self.target_sample_rate:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.target_sample_rate)
            
            audio = waveform # Ya es un tensor de PyTorch
            print(f"  [Inference Audio] Samples: {len(audio)} ({len(audio)/self.target_sample_rate:.1f}s)")

            chunk_seconds = 20.0
            hop_seconds = 10.0 
            
            chunk_size = int(chunk_seconds * self.target_sample_rate)
            hop_size = int(hop_seconds * self.target_sample_rate)
            
            segments = [audio[i:i + chunk_size] for i in range(0, len(audio) - chunk_size + 1, hop_size)]
            
            if not segments and len(audio) > 0:
                if len(audio) < chunk_size:
                    padding = chunk_size - len(audio)
                    segments = [F.pad(audio, (0, padding))]
                else:
                    segments = [audio[:chunk_size]]

            segments = segments[:12] 
            print(f"  [Inference Chunks] Processing {len(segments)} segments...")
            
            # Embedding process
            processed_segments = []
            for seg in segments:
                seg_norm = (seg - seg.mean()) / (seg.std() + 1e-7)
                processed_segments.append(seg_norm.numpy()) 
            inputs = self.feature_extractor(processed_segments, sampling_rate=self.target_sample_rate, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                fused = torch.stack(outputs.hidden_states[-4:]).mean(0)
                
                if self.attention:
                    final_embs = fused.squeeze(0).cpu().numpy()
                else:
                    final_embs = fused.mean(dim=1).squeeze(0).cpu().numpy()
            print(f"  [Inference Output] Final embedding batch shape: {final_embs.shape}")
            return final_embs
            
        except Exception as e:
            print(f"Inference error: {e}")
            return None

    def get_inference_chunks(self, mp3_path):
        try:
            waveform, sr = torchaudio.load(mp3_path)
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)
                
            if sr != self.target_sample_rate:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.target_sample_rate)
            
            audio = waveform 
            print(f"  [Inference Audio] Samples: {len(audio)} ({len(audio)/self.target_sample_rate:.1f}s)")

            chunk_seconds = 20.0
            hop_seconds = 10.0 
            
            chunk_size = int(chunk_seconds * self.target_sample_rate)
            hop_size = int(hop_seconds * self.target_sample_rate)
            
            segments = [audio[i:i + chunk_size] for i in range(0, len(audio) - chunk_size + 1, hop_size)]
            
            if not segments and len(audio) > 0:
                if len(audio) < chunk_size:
                    padding = chunk_size - len(audio)
                    segments = [F.pad(audio, (0, padding))]
                else:
                    segments = [audio[:chunk_size]]

            print(f"  [Inference Chunks] Processing {len(segments)} segments...")
            
            processed_segments = []
            for seg in segments:
                seg_norm = (seg - seg.mean()) / (seg.std() + 1e-7)
                processed_segments.append(seg_norm.numpy()) 
            

            batch_size = 12 
            all_embs = []
            
            for i in range(0, len(processed_segments), batch_size):
                batch_segs = processed_segments[i:i + batch_size]
                
                inputs = self.feature_extractor(batch_segs, sampling_rate=self.target_sample_rate, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=('cuda' in str(self.device))):
                        outputs = self.model(**inputs, output_hidden_states=True)
                        fused = torch.stack(outputs.hidden_states[-4:]).mean(0)
                        
                        if self.attention:
                            embs = fused.cpu().numpy()
                        else:
                            embs = fused.mean(dim=1).cpu().numpy()
                            
                        all_embs.append(embs)
            
            final_embs = np.concatenate(all_embs, axis=0)

            print(f"  [Inference Output] Final embedding batch shape: {final_embs.shape}")
            return final_embs
            
        except Exception as e:
            print(f"Inference error: {e}")
            return None

    def predict_genre(self, mp3_path: str, classifier : str = None, mode = 'ce', save_viz_dir = None):
        self.load_classifier()
        start_time = time.time()
        
        chunks = self.get_inference_chunks(mp3_path)
        if chunks is None: 
            print(f"No chunks generated")
            return None
        
        raw_chunks_for_viz = chunks.copy()

        if self.scaler:
            if chunks.ndim == 3:
                N, T, D = chunks.shape
                chunks_flat = chunks.reshape(N * T, D)
                chunks_flat = self.scaler.transform(chunks_flat)
                chunks = chunks_flat.reshape(N, T, D) 
            else:
                chunks = self.scaler.transform(chunks)
        
        if self.use_pca and self.pca:
            if chunks.ndim == 3:
                N, T, D = chunks.shape
                chunks_flat = chunks.reshape(N * T, D)
                chunks_flat = self.pca.transform(chunks_flat)
                chunks = chunks_flat.reshape(N, T, -1)
            else:
                chunks = self.pca.transform(chunks)
        

        if hasattr(self.classifier, "predict_proba"): 
            probs_matrix = self.classifier.predict_proba(chunks) 
            avg_probs = np.mean(probs_matrix, axis=0) 
            pred_idx = np.argmax(avg_probs)
        else:
            preds = self.classifier.predict(chunks)
            pred_idx = Counter(preds).most_common(1)[0][0]
            avg_probs = np.zeros(len(self.label_encoder.classes_))
            avg_probs[pred_idx] = 1.0

        pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
        
        self.visualize_latent_space_inference(mp3_path=mp3_path, chunks=raw_chunks_for_viz, pred_label=pred_label, save_dir=save_viz_dir, mode=mode, n_bg_samples=3000)

        return {
            "label": pred_label,
            "probabilities": dict(zip(self.label_encoder.classes_, avg_probs)),
            "embedding": np.mean(chunks, axis=0), 
            "inference_time": time.time() - start_time
        }

    def load_classifier(self):
        if not os.path.exists(self.classifier_path): 
            print(f"File not found")
            return False
        data = joblib.load(self.classifier_path)
        self.classifier = data["model"]
        self.label_encoder = data["label_encoder"]
        self.scaler = data.get("scaler")
        self.pca = data.get("pca")
        self.use_pca = data.get("use_pca", False)
        self.pca_components = data.get("pca_components")
        return True
    
    def save_prediction_result(self, mp3_path: str, result: dict, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        song_name = os.path.splitext(os.path.basename(mp3_path))[0]

        csv_filename = f'results.csv'
        csv_path = os.path.join(save_dir, csv_filename)

        if not hasattr(self, "_results_cleared"):
            if os.path.exists(csv_path):
                os.remove(csv_path)
            self._results_cleared = True

        file_exists = os.path.isfile(csv_path)

        with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            if not file_exists:
                header = ["song_name", "predicted_label"] + list(result["probabilities"].keys())
                writer.writerow(header)

            probabilities = [f"{value:.4f}" for value in result["probabilities"].values()]        
            row = [song_name, result["label"]] + probabilities
            writer.writerow(row)

    def visualize_latent_space_inference(self, mp3_path, chunks, pred_label, save_dir: str, mode="ce", n_bg_samples=3000):   
        song_name = os.path.splitext(os.path.basename(mp3_path))[0]
        subfolder = os.path.join(save_dir, song_name)
        os.makedirs(subfolder, exist_ok=True)

        # Saving an npy file with the embedding of the song
        npy_filename = f'{self.classifier_type}_embedding_{song_name}.npy'
        emb_path = os.path.join(subfolder, npy_filename)
        embedding_song = np.mean(chunks, axis=0)
        np.save(emb_path, embedding_song)

        # Getting embedding of the song
        self.classifier.model.eval()
        with torch.no_grad():
            chunks_t = torch.tensor(chunks, dtype=torch.float32, device=self.device)

            if self.scaler:
                chunks_scaled = self.scaler.transform(chunks)
            else:
                chunks_scaled = chunks
                
            if self.use_pca and self.pca:
                chunks_pca = self.pca.transform(chunks_scaled)
            else:
                chunks_pca = chunks_scaled
                
            chunks_t = torch.tensor(chunks_pca, dtype=torch.float32, device=self.device)
            out = self.classifier.model(chunks_t)
            
            song_features = out[0] if mode == "halo" else out[1]
            song_feats_np = song_features.cpu().numpy()
            
        # Calculating the centroid of the inference song (mean over all chunks)
        song_centroid = np.mean(song_feats_np, axis=0, keepdims=True)

        # Computing distance from each chunk to the centroid
        chunk_distances = np.linalg.norm(song_feats_np - song_centroid, axis=1)
        mean_distance = np.mean(chunk_distances)
        print(f"  [Metrics] Mean distance to centroid: {mean_distance:.4f}")

        # print("\nCreating embedding space...")
        X_bg_raw, y_bg_raw, _ = self.load_feature_space(mode='test')
        if isinstance(X_bg_raw, torch.Tensor):         
            if len(X_bg_raw) > n_bg_samples:
                # Random points maintains the variable in memory
                indices = torch.randperm(len(X_bg_raw))[:n_bg_samples]
                X_bg_raw = X_bg_raw[indices]
                y_bg_raw = np.array(y_bg_raw)[indices.numpy()]
            
            X_bg_np = X_bg_raw.numpy()
            if self.scaler:
                X_bg_np = self.scaler.transform(X_bg_np)
            if self.use_pca and self.pca:
                X_bg_np = self.pca.transform(X_bg_np)
                
            X_bg_t = torch.tensor(X_bg_np, dtype=torch.float32, device=self.device)
        else:
            pass

        bg_feats_list = []
        with torch.no_grad():
            for i in range(0, len(X_bg_raw), 256):
                bx = X_bg_t[i:i+256]
                out = self.classifier.model(bx)
                feats = out[0] if mode == "halo" else out[1]
                bg_feats_list.append(feats.cpu().numpy())

        bg_feats_np = np.concatenate(bg_feats_list, axis=0)


        reducer = PCA(n_components=3)

        bg_3d = reducer.fit_transform(bg_feats_np)
        song_chunks_3d = reducer.transform(song_feats_np)
        song_centroid_3d = reducer.transform(song_centroid)

        # Prep colors and legends
        y_bg_names = self.label_encoder.inverse_transform(self.label_encoder.transform(y_bg_raw))
        all_classes = sorted(self.label_encoder.classes_)
        cmap = plt.get_cmap("tab20" if len(all_classes) <= 20 else "hsv")
        color_map = {l: cmap(i / len(all_classes)) for i, l in enumerate(all_classes)}
        color_map_hex = {l: mcolors.to_hex(color_map[l]) for l in all_classes}
        
        # Idx for chunks of test song
        chunk_indices = list(range(1, len(song_chunks_3d) + 1))

        # ----- HTML ----- 
        df_bg = pd.DataFrame({"C1": bg_3d[:,0], "C2": bg_3d[:,1], "C3": bg_3d[:,2], "Genre": y_bg_names})
        
        # Training points with no opacity
        fig_html = px.scatter_3d(df_bg, x="C1", y="C2", z="C3", color="Genre", color_discrete_map=color_map_hex, opacity=0.55)
        fig_html.update_traces(marker=dict(size=4))
        
        # Trace the cronological order of the song
        fig_html.add_trace(go.Scatter3d(
            x=song_chunks_3d[:,0], y=song_chunks_3d[:,1], z=song_chunks_3d[:,2],
            mode='markers+text',
            text=[str(i) for i in chunk_indices], 
            textfont=dict(color='black', size=14, weight='bold'),
            textposition="top center",
            marker=dict(
                size=8, 
                color=chunk_indices, 
                colorscale='Viridis', # (purple to yellow)
                symbol='diamond', 
                line=dict(color='black', width=1)
            ),
            name=f"Secuencia Chunks ({song_name})"
        ))
        
        # Song centroid (Big red cross)
        fig_html.add_trace(go.Scatter3d(
            x=song_centroid_3d[:,0], y=song_centroid_3d[:,1], z=song_centroid_3d[:,2],
            mode='markers', marker=dict(size=14, color='red', symbol='cross', line=dict(color='black', width=2)),
            name=f"CENTROIDE (Pred: {pred_label})"
        ))
        
        fig_html.update_layout(title=f"Inference Projection: {song_name} | Predicted: {pred_label}", margin=dict(l=0, r=0, t=40, b=0))
        html_path = os.path.join(subfolder, f"inference_space_{song_name}.html")
        fig_html.write_html(html_path)
        
        # ----- PDF -----
        pdf_path = os.path.join(subfolder, f"inference_space_{song_name}.pdf")
        with PdfPages(pdf_path) as pdf:
            fig = plt.figure(figsize=(16, 8))
            fig.suptitle(f"Latent Space Inference Mapping: {song_name}\nPredicted Genre: {pred_label}", fontsize=14, fontweight='bold')
            
            # --- 2D ---
            ax2 = fig.add_subplot(1, 2, 1)
            for cls in all_classes:
                mask = np.array(y_bg_names) == cls
                if np.any(mask):
                    ax2.scatter(bg_3d[mask, 0], bg_3d[mask, 1], c=[color_map[cls]], alpha=0.4, s=20, label=cls)
            
            ax2.scatter(song_chunks_3d[:,0], song_chunks_3d[:,1], c=chunk_indices, cmap='viridis', marker='d', s=100, edgecolor='black', zorder=5)
            for i, (x, y) in enumerate(zip(song_chunks_3d[:,0], song_chunks_3d[:,1])):
                ax2.text(x, y + 0.15, str(i+1), color='black', fontsize=11, fontweight='bold', ha='center', va='bottom', zorder=6)
                
            ax2.scatter(song_centroid_3d[:,0], song_centroid_3d[:,1], c='red', marker='X', s=300, edgecolor='black', zorder=10)
            
            ax2.legend(loc="best", fontsize=8)
            ax2.set_title("2D Projection")
            ax2.grid(alpha=0.3)
            
            # 3D
            ax3 = fig.add_subplot(1, 2, 2, projection="3d")
            for cls in all_classes:
                mask = np.array(y_bg_names) == cls
                if np.any(mask):
                    ax3.scatter(bg_3d[mask, 0], bg_3d[mask, 1], bg_3d[mask, 2], c=[color_map[cls]], alpha=0.4, s=20)
            
            ax3.scatter(song_chunks_3d[:,0], song_chunks_3d[:,1], song_chunks_3d[:,2], c=chunk_indices, cmap='viridis', marker='d', s=100, edgecolor='black', zorder=5)
            for i, (x, y, z) in enumerate(zip(song_chunks_3d[:,0], song_chunks_3d[:,1], song_chunks_3d[:,2])):
                ax3.text(x, y, z + 0.15, str(i+1), color='black', fontsize=11, fontweight='bold', zorder=6)
                
            ax3.scatter(song_centroid_3d[:,0], song_centroid_3d[:,1], song_centroid_3d[:,2], c='red', marker='X', s=300, edgecolor='black', zorder=10)
            ax3.set_title("3D Projection")
            
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        
            # CHUNK DISTANCES & METRICS
            fig_metrics = plt.figure(figsize=(10, 6))
            ax_metrics = fig_metrics.add_subplot(1, 1, 1)
            
            # Bar graph
            ax_metrics.plot(chunk_indices, chunk_distances, marker='o', linestyle='-', color='teal', linewidth=2, markersize=8)
            ax_metrics.axhline(y=mean_distance, color='red', linestyle='--', linewidth=2, label=f'Mean Distance ({mean_distance:.4f})')
            
            # Anotate every point with its value
            for i, d in zip(chunk_indices, chunk_distances):
                ax_metrics.text(i, d + (max(chunk_distances)*0.02), f"{d:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

            ax_metrics.set_title(f"Euclidean Distance to Centroid\n{song_name}", fontsize=14, fontweight='bold')
            ax_metrics.set_xlabel("Cronological order of Chunks", fontsize=12)
            ax_metrics.set_ylabel("Feature Space Distance", fontsize=12)
            ax_metrics.set_xticks(chunk_indices)
            ax_metrics.grid(alpha=0.3)
            ax_metrics.legend(loc='best', fontsize=10)
            
            text_str = f"HOMOGENITY METRICS\n{'='*25}\n\n"
            text_str += f"Mean Distance : {mean_distance:.4f}\n"
            text_str += f"Max Distance : {np.max(chunk_distances):.4f} (Chunk {chunk_indices[np.argmax(chunk_distances)]})\n"
            text_str += f"Min Distance : {np.min(chunk_distances):.4f} (Chunk {chunk_indices[np.argmin(chunk_distances)]})\n\n"
            text_str += "Individual Distances:\n"
            for i, d in zip(chunk_indices, chunk_distances):
                text_str += f" • Chunk {i:02d}: {d:.4f}\n"
                
            fig_metrics.text(1.02, 0.5, text_str, fontsize=11, family='monospace', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            fig_metrics.tight_layout()
            pdf.savefig(fig_metrics, bbox_inches="tight")
            plt.close(fig_metrics)

        # print(f"  Everything saved to: {subfolder}")
        return html_path, pdf_path
