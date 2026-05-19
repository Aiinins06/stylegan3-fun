import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin

from utils.attention import SelfAttentionPooling 

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
    
