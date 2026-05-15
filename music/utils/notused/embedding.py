import librosa
import torch
import numpy as np
import os
from typing import List, Optional
from transformers import AutoFeatureExtractor, AutoModel
import re

class MusicEmbedder:
    def __init__(self, model_name: str, attention : bool = False):
        print(f"Loading model: {model_name}")
        self.model_name = model_name
        self.only_name_model = model_name.split("/")[-1]
        self.attention = attention
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model and feature extractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True
        ).to(self.device)
        
        self.target_sample_rate = self.feature_extractor.sampling_rate
        print(f"Model loaded. Target sample rate: {self.target_sample_rate} Hz")

    def _get_layer_fusion_embedding(self, inputs):
        """
        Extracts and averages the last 4 layers (Rich Representation).
        """
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # Check if model has hidden_states (MERT/Wav2Vec2 usually do)
            if hasattr(outputs, 'hidden_states'):
                # hidden_states is a tuple of tensors (one for each layer)
                # [DEBUG] Print number of layers
                # print(f"  [LayerFusion] Total hidden layers available: {len(outputs.hidden_states)}")

                # Stack last 4 layers
                last_n_layers = torch.stack(outputs.hidden_states[-4:]) 
                # [DEBUG] Expected shape: (4, batch_size, sequence_length, hidden_dim)
                print(f"  [LayerFusion] Stacked last 4 layers shape: {last_n_layers.shape}")

                # Mean over layers -> (batch, time, dim)
                fused_embedding = torch.mean(last_n_layers, dim=0) 
                # [DEBUG] Expected shape: (batch_size, sequence_length, hidden_dim)
                print(f"  [LayerFusion] Fused (mean over layers) shape: {fused_embedding.shape}")

                # Mean over time -> (dim,)
                final_vec = fused_embedding.mean(dim=1).squeeze().cpu()
                # [DEBUG] Final shape after time averaging and squeeze
                # print(f"  [LayerFusion] Final vector shape: {final_vec.shape}")

            else:
                # Fallback
                print("  [LayerFusion] 'hidden_states' not found. Using 'last_hidden_state'.")
                final_vec = outputs.last_hidden_state.mean(dim=1).squeeze().cpu()
                
        return final_vec
    
    def _get_layer_fusion_embedding_attention(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # Check if model has hidden_states
            if hasattr(outputs, 'hidden_states'):
                # hidden_states is a tuple of tensors (one for each layer)
                # [DEBUG] Print number of layers
                # print(f"  [LayerFusion] Total hidden layers available: {len(outputs.hidden_states)}")

                # Stack last 4 layers
                last_n_layers = torch.stack(outputs.hidden_states[-4:]) 
                # [DEBUG] Expected shape: (4, batch_size, sequence_length, hidden_dim)
                print(f"  [LayerFusion] Stacked last 4 layers shape: {last_n_layers.shape}")

                # Mean over layers -> (batch, time, dim)
                fused_embedding = torch.mean(last_n_layers, dim=0) 
                # [DEBUG] Expected shape: (batch_size, sequence_length, hidden_dim)
                print(f"  [LayerFusion] Fused (mean over layers) shape: {fused_embedding.shape}")

                # Eliminating dimension of batch if equals to 1 (sequence_length, hidden_dim)
                final_vec = fused_embedding.squeeze().cpu()
                # [DEBUG] Final shape after time averaging and squeeze
                # print(f"  [LayerFusion] Final vector shape: {final_vec.shape}")

            else:
                # Fallback
                print("  [LayerFusion] 'hidden_states' not found. Using 'last_hidden_state'.")
                final_vec = outputs.last_hidden_state.mean(dim=1).squeeze().cpu()
                
        return final_vec

    def get_song_chunks(self, mp3_path: str, batch_size: int = 8) -> Optional[List[torch.Tensor]]:
        """
        Splits song into 20-second chunks with 10s overlap.
        """
        try:
            # 1. Load Audio
            audio, _ = librosa.load(mp3_path, sr=self.target_sample_rate, mono=True)
            # [DEBUG] Show loaded audio length
            # print(f"\n[Processing] Song: {os.path.basename(mp3_path)}")
            print(f"  [Audio] Loaded raw audio shape: {audio.shape} ({len(audio)/self.target_sample_rate:.2f} sec)")
            
            # 2. Define Window (20 seconds) and Hop (10 seconds)
            chunk_seconds = 20.0
            hop_seconds = 10.0
            
            chunk_size = int(chunk_seconds * self.target_sample_rate)
            hop_size = int(hop_seconds * self.target_sample_rate)
            
            # Limit analysis to first 3 minutes to save space/time
            max_samples = self.target_sample_rate * 180 
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            # Create sliding windows
            segments = []
            for i in range(0, len(audio) - chunk_size, hop_size):
                seg = audio[i:i + chunk_size]
                # Normalization
                seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-7)
                segments.append(seg)
            
            if not segments:
                # If song is shorter than 20s, pad it once
                if len(audio) > self.target_sample_rate * 5: # Min 5 secs to process
                     raw_segments = [(audio - np.mean(audio)) / (np.std(audio) + 1e-7)]
                else:
                    return None

            chunk_embeddings = []

            for i, seg in enumerate(segments):
                # Padding if segment is slightly shorter than chunk_size (e.g. end of song)
                if len(seg) < chunk_size:
                    padding = chunk_size - len(seg)
                    seg = np.pad(seg, (0, padding))
                
                # Normalization
                seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-7)

                inputs = self.feature_extractor(
                    seg, 
                    sampling_rate=self.target_sample_rate, 
                    return_tensors="pt",
                    padding=True
                )

                # [DEBUG] Check input shape before model
                # input_values usually has shape (1, sequence_length)
                print(f"  [Chunk {i}] Input tensor shape: {inputs['input_values'].shape}")
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Extract using Layer Fusion
                if self.attention:
                    emb = self._get_layer_fusion_embedding_attention(inputs)
                else:
                    emb = self._get_layer_fusion_embedding(inputs)

                # [DEBUG] Check embedding shape immediately after generation
                print(f"  [Chunk {i}] Output embedding shape: {emb.shape}")

                if emb.dim() == 0: 
                    # Es un escalar, intentamos convertirlo o lo saltamos
                    print(f"   Warning: Scalar embedding produced for {mp3_path}. Skipping chunk.")
                    continue

                chunk_embeddings.append(emb)

            return chunk_embeddings

        except Exception as e:
            if "out of memory" in str(e):
                print(f"   OOM skipping {mp3_path}")
                torch.cuda.empty_cache()
            else:
                print(f"Error processing {mp3_path}: {e}")
            return None

    def process_genres(self, base_directory: str, genre_list: List[str], max_songs: int = None):
        all_embeddings = []
        all_genre_labels = []
        all_song_names = [] 
        
        name_genres = "".join([g + "_" for g in genre_list])

        for genre in genre_list:
            genre_dir = os.path.join(base_directory, genre)
            if not os.path.isdir(genre_dir):
                print(f"  Warning: Directory not found: {genre_dir}")
                continue

            print(f"\nProcessing genre: {genre}")
            mp3_files = sorted(f for f in os.listdir(genre_dir) if f.endswith(".mp3"))

            if max_songs:
                mp3_files = mp3_files[:max_songs]

            for counter, song_file in enumerate(mp3_files, start=1):
                file_path = os.path.join(genre_dir, song_file)
                print(f"Processing ({counter}/{len(mp3_files)}) of {genre}: {song_file}")

                # Turns "Song Name_part1.mp3" into "Song Name"
                if "_part" in song_file:
                    base_name = song_file.rsplit('_part', 1)[0]
                else:
                    base_name = os.path.splitext(song_file)[0]

                # GET LIST OF CHUNKS
                chunks = self.get_song_chunks(file_path)

                if chunks is not None:
                    for chunk_vec in chunks:
                        # [DEBUG] Final check before adding to the big list
                        # print(f"    Appending vector of shape {chunk_vec.shape}")
                        all_embeddings.append(chunk_vec)
                        all_genre_labels.append(genre)
                        all_song_names.append(base_name)

        # Convert to Tensor
        if all_embeddings:
            # [DEBUG] Check list consistency
            sizes = [e.shape for e in all_embeddings]
            print(f"Final List Sizes Summary: {sizes[:5]} ...") # Print first 5

            if torch.is_tensor(all_embeddings[0]):
                embedding_matrix = torch.stack(all_embeddings)
            else:
                embedding_matrix = torch.tensor(np.array(all_embeddings))
        else:
            embedding_matrix = torch.empty((0, 0))

        # Save Feature Space
        os.makedirs('/datafast/105-1/Datasets/INTERNS/anavarror/feature_spaces_1k_20s_contain_fragments', exist_ok=True) # ==================================== CAMBIAR DEPENDIENDO DE RUN =======================
        save_path = f'/datafast/105-1/Datasets/INTERNS/anavarror/feature_spaces_1k_20s_contain_fragments/{self.only_name_model}_attention{self.attention}_{name_genres}.pt'
        
        print(f"\nSaving to: {save_path}")
        torch.save({
            'embeddings': embedding_matrix.cpu(), 
            'labels': all_genre_labels,
            'song_names': all_song_names
        }, save_path)

        print(f"DONE. Total chunks processed: {len(all_embeddings)}")
        print(f"Shape: {embedding_matrix.shape}")

        return embedding_matrix, all_genre_labels