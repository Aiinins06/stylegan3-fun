import os
import torch
import librosa
import numpy as np
import joblib
import csv
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.decomposition import PCA
from transformers import AutoFeatureExtractor, AutoModel
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
import torch.nn as nn
import torch.optim as optim


from attention import SelfAttentionPooling 

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

# This wrapper makes the PyTorch model look like a Scikit-Learn model
# allowing it to work with joblib, plot functions, and your inference script.
class PyTorchMLPWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, device, classes):
        self.model = model
        self.device = device
        self.classes_ = classes 

    def predict(self, X):
        self.model.eval()
        # Handle inputs: if X is numpy, convert to tensor
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        
        with torch.no_grad():
            X = X.to(self.device)
            logits = self.model(X)
            preds = torch.argmax(logits, dim=1)
        return preds.cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
            
        with torch.no_grad():
            X = X.to(self.device)
            logits = self.model(X)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()



GENRES_LIST_PIXABAY = ["Hip Hop Convencional", "Ambiente", "Rock", "Jazz Clásico", "Techno Y Trance"] # ================== CAMBIAR DEPENDIENDO DE LA RUN ========================================

NAME_GENRES = ""
for genre in GENRES_LIST_PIXABAY:
        NAME_GENRES += genre + "_"




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
    ):
        self.model_name = model_name
        self.only_name_model =  model_name.split("/")[-1]
        self.feature_space_path = feature_space_path
        self.classifier_path = classifier_path
        self.classifier_type = classifier_type
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.attention = attention
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load Audio Model
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

    def plot_trained_decision_space(self, embedding_matrix, true_labels, method, save_path):
        if self.classifier is None or self.label_encoder is None:
            self.load_classifier()

        # Predict labels using the wrapper (works for GPU model transparently)
        y_pred_idx = self.classifier.predict(embedding_matrix)
        pred_labels = self.label_encoder.inverse_transform(y_pred_idx)

        # Dimensionality reduction
        method = method.lower()
        if method == "umap":
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
        plt.title(f"{method.upper()} — Classifier Decision Space (GPU Trained)")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved trained decision plot at: {save_path}")

        # plt.show()

    def load_feature_space(self):
        print(f"Loading features from: {self.feature_space_path}")
        data = torch.load(self.feature_space_path, map_location="cpu")
        embeddings = data["embeddings"]
        labels = data["labels"]
        print(f"[Data Load] Loaded {embeddings.shape[0]} samples with dimension {embeddings.shape[1]}")
        return (data["embeddings"].cpu().numpy(), data["labels"], data.get("song_names"))

    def _build_classifier(self, input_dim):
        # NOTE: This method is only used if you choose 'svm', 'logreg' or 'knn'.
        # For 'mlp', we now use the GPU class defined above inside train_classifier.
        if self.classifier_type == "logreg":
            return LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="auto")
        elif self.classifier_type == "svm":
            return SVC(kernel="rbf", probability=True)
        elif self.classifier_type == "knn":
            return KNeighborsClassifier(n_neighbors=29)
        elif self.classifier_type == "mlp":
            # Just a placeholder, actual GPU construction happens in train loop
            return None 

    def train_classifier(self, epochs=100, batch_size=64, lr=0.001, criterion=nn.CrossEntropyLoss()):
        # 1. Load Data
        X, labels, song_names = self.load_feature_space()
        le = LabelEncoder()
        y = le.fit_transform(labels)
        print(f"[Label Encoding] Classes found: {len(le.classes_)} -> {le.classes_}")
        
        if X.ndim == 3:
            print(f"Training on {X.shape[0]} chunks (20s each). Dimension: ({X.shape[1]},{X.shape[2]})")
        else:
            print(f"Training on {X.shape[0]} chunks (20s each). Dimension: ({X.shape[1]})")

        # 2. Normalize
        print(f"  Before Normalizer: X shape {X.shape}, Mean {np.mean(X):.4f}, Std {np.std(X):.4f}")
        self.scaler = Normalizer(norm='l2')
        # X = self.scaler.fit_transform(X)
        if X.ndim == 3:
            # 3D case: (Batch, Time, Dim). Flatten Batch and Time
            N, T, D = X.shape
            X_flat = X.reshape(N * T, D) 
            X_flat = self.scaler.fit_transform(X_flat)
            X = X_flat.reshape(N, T, D) # Regain 3D
        else:
            # 2D case
            X = self.scaler.fit_transform(X)
        print(f"  After Normalizer:  X shape {X.shape}, Mean {np.mean(X):.4f}, Std {np.std(X):.4f}")
        
        # 3. PCA (Optional)
        if self.use_pca:
            feat_dim = X.shape[2] if X.ndim == 3 else X.shape[1]
            print(f"  Applying PCA to reduce from {feat_dim} to {self.pca_components} components...")
            self.pca = PCA(n_components=self.pca_components, random_state=42)
            if X.ndim == 3:
                N, T, D = X.shape
                X_flat = X.reshape(N * T, D)
                X_flat = self.pca.fit_transform(X_flat)
                # New dimension is pca_components
                X = X_flat.reshape(N, T, self.pca_components)
            else:
                X = self.pca.fit_transform(X)
                
            print(f"  New X shape: {X.shape}")  

        # 4. Split and Train
        if song_names is not None and len(song_names) == len(X):
            print("  Splitting data by SONG ID (GroupShuffleSplit).")
            gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
            
            # gss.split returns indices
            train_idx, test_idx = next(gss.split(X, y, groups=song_names))
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Validation print
            train_groups = set(np.array(song_names)[train_idx])
            test_groups = set(np.array(song_names)[test_idx])
            print(f"  - Unique Songs in Train: {len(train_groups)}")
            print(f"  - Unique Songs in Test:  {len(test_groups)}")
            print(f"  - Intersection (Leakage): {len(train_groups.intersection(test_groups))}")
            
        else:
            print("   Song names missing or mismatched. Using standard random split (POTENTIAL LEAKAGE).")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

        # Check if we should use GPU MLP or Standard Sklearn
        if self.classifier_type == "mlp":
            print(f"  Training MLP on {self.device}...")
            
            # Convert to Tensors but keeping them in CPU because GPUs don't have enough space with attention
            X_train_t = torch.tensor(X_train, dtype=torch.float32) 
            y_train_t = torch.tensor(y_train, dtype=torch.long)
            
            # Initialize GPU Model
            input_dim = X_train.shape[1] if X_train.ndim == 2 else X_train.shape[2] 
            num_classes = len(le.classes_)

            if self.attention:
                model = MLP_Attention(input_dim, num_classes).to(self.device)
            else:
                model = MLP(input_dim, num_classes).to(self.device)
            
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Training Loop
            model.train()
            n_samples = X_train_t.shape[0]

            for epoch in range(epochs):
                permutation = torch.randperm(n_samples)
                for i in range(0, n_samples, batch_size):
                    indices = permutation[i:i+batch_size]
                    batch_x, batch_y = X_train_t[indices], y_train_t[indices]

                    # Move this batch to GPU
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    if epoch == 0 and i == 0:
                        print(f"  [Debug Batch] Input Batch Shape: {batch_x.shape}, Target Batch Shape: {batch_y.shape}")

                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                
                if (epoch+1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

            # WRAP THE MODEL so it behaves like sklearn
            self.classifier = PyTorchMLPWrapper(model, self.device, le.classes_)
            
        else:
            # Fallback for SVM, KNN, etc (CPU)
            print(f"Fitting {self.classifier_type.upper()} on {self.device}...")
            self.classifier = self._build_classifier(X_train.shape[1])
            self.classifier.fit(X_train, y_train)

        # -----  Validation -----
        y_pred = self.classifier.predict(X_test)
        print(f"  Prediction output shape: {y_pred.shape}")

        # ----- Metrics -----
        report_str = classification_report(y_test, y_pred, target_names=le.classes_)
        print(report_str)

        # Calculating metrics accuracy and f1 score
        acc = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        print(f"Chunk-Level Accuracy: {acc:.4f}")
        print(f"F1 Score (Weighted): {f1_weighted:.4f}")

        save_model_metrics_plots_path = f"/datafast/105-1/Datasets/INTERNS/anavarror/trained_models_20s_1k_contain_fragments/{self.only_name_model}_{self.classifier_type}_epochs{epochs}_contain_fragments_{NAME_GENRES}"
        os.makedirs(save_model_metrics_plots_path, exist_ok=True)  

        # Saving accuracy and f1-score metrics
        metrics_filename = f'{self.only_name_model}_{self.classifier_type}_metrics_{NAME_GENRES}.txt'
        metrics_path = os.path.join(save_model_metrics_plots_path, metrics_filename)

        with open(metrics_path, "w") as f:
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Classifier: {self.classifier_type}\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"F1 Score (Weighted): {f1_weighted:.4f}\n")
            f.write(f"F1 Score (Macro): {f1_macro:.4f}\n")
            f.write("-" * 30 + "\n")
            f.write("Classification Report:\n")
            f.write(report_str)
        
        print(f"Metrics saved to: {metrics_path}")

        # Computing the confusion matric
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
        
        # Plot setup
        plt.figure(figsize=(10, 8))
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
        plt.title(f"Confusion Matrix - {self.classifier_type}")
        
        cm_filename = f'{self.only_name_model}_{self.classifier_type}_confusion_matrix_{NAME_GENRES}.png'
        cm_path = os.path.join(save_model_metrics_plots_path, cm_filename)
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close() # Close figure
        print(f"Confusion matrix saved to: {cm_path}")

        save_obj = {
            "model": self.classifier, # Saves Wrapper+PyTorch Model OR Sklearn Model
            "label_encoder": le,
            "scaler": self.scaler,         
            "pca": self.pca,               
            "use_pca": self.use_pca,
            "pca_components": self.pca_components
        }
        joblib.dump(save_obj, self.classifier_path)
        print(f"Saved model pipeline to {self.classifier_path}")

        # Saving plots
        methods = ["umap", "pca"]

        for meth in methods:
            name_file = f'{self.only_name_model}_attention{self.attention}_{self.classifier_type}_{meth}_contain_fragments_{NAME_GENRES}.png'   # ===================================== CAMBIAR DEPENDIENDO DE LA RUN =================================
            complete = os.path.join(save_model_metrics_plots_path, name_file)
            self.plot_trained_decision_space(X, labels, method=meth, save_path=complete) 

    def get_inference_chunks(self, mp3_path):
        """Extracts 20s chunks for inference (Overlapping for better voting coverage)."""
        try:
            audio, _ = librosa.load(mp3_path, sr=self.target_sample_rate, mono=True)
            print(f"  [Inference Audio] Loaded song, samples: {len(audio)} ({len(audio)/self.target_sample_rate:.1f}s)")

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
            from collections import Counter
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