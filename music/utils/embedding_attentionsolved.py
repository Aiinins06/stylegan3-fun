import torch
import numpy as np
import os
from typing import List, Optional
from transformers import AutoFeatureExtractor, AutoModel
import unicodedata
import torchaudio

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

    def _sanitize_filename(self, text):
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        return text.replace(" ", "")

    def _get_layer_fusion_embedding(self, inputs, printing : bool = False):
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
                if printing:
                    print(f"  [LayerFusion] Stacked last 4 layers shape: {last_n_layers.shape}")

                # Mean over layers -> (batch, time, dim)
                fused_embedding = torch.mean(last_n_layers, dim=0) 
                # [DEBUG] Expected shape: (batch_size, sequence_length, hidden_dim)
                if printing:
                    print(f"  [LayerFusion] Fused (mean over layers) shape: {fused_embedding.shape}")

                # Mean over time -> (batch, dim)
                final_vec = fused_embedding.mean(dim=1).cpu()
                # [DEBUG] Final shape after time averaging and squeeze
                # print(f"  [LayerFusion] Final vector shape: {final_vec.shape}")

            else:
                # Fallback
                print("  [LayerFusion] 'hidden_states' not found. Using 'last_hidden_state'.")
                final_vec = outputs.last_hidden_state.mean(dim=1).cpu()
                
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
                final_vec = fused_embedding.cpu()
                # [DEBUG] Final shape after time averaging and squeeze
                # print(f"  [LayerFusion] Final vector shape: {final_vec.shape}")

            else:
                # Fallback
                print("  [LayerFusion] 'hidden_states' not found. Using 'last_hidden_state'.")
                final_vec = outputs.last_hidden_state.mean(dim=1).cpu()
                
        return final_vec

    def get_song_chunks(self, mp3_path: str, batch_size: int = 8) -> Optional[List[torch.Tensor]]:
        """
        Splits song into 20-second chunks with 10s overlap.
        """
        try:
            # Load Audio
            waveform, sr = torchaudio.load(mp3_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)
                
            if sr != self.target_sample_rate:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.target_sample_rate)
                
            audio = waveform.numpy() 

            # [DEBUG] Show loaded audio length
            # print(f"\n[Processing] Song: {os.path.basename(mp3_path)}")
            print(f"  [Audio] Loaded raw audio shape: {audio.shape} ({len(audio)/self.target_sample_rate:.2f} sec)")
            
            # Define Window (20 seconds) and Hop (10 seconds)
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
            audio_tensor = torch.tensor(audio, dtype=torch.float32) # Convert to tensor once
            
            for i in range(0, len(audio_tensor) - chunk_size + 1, hop_size):
                seg = audio_tensor[i:i + chunk_size]
                # Pytorch native normalization
                seg = (seg - seg.mean()) / (seg.std() + 1e-7) 
                segments.append(seg.numpy())
            
            # Dealing with last fragment if it is smaller than chunk_size
            if len(audio) > 0 and len(audio) < chunk_size:
                seg = np.pad(audio, (0, chunk_size - len(audio)))
                seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-7)
                segments.append(seg)

            print(f"  [Cunks] A total of {len(segments)} were created for this file")
            chunk_embeddings = []

            for i in range(0, len(segments), batch_size):
                batch_segs = segments[i:i + batch_size]
                
                inputs = self.feature_extractor(
                    batch_segs, 
                    sampling_rate=self.target_sample_rate, 
                    return_tensors="pt",
                    padding=True
                )

                # [DEBUG] Check input shape before model
                # input_values usually has shape (1, sequence_length)
                if i == 0:
                    print(f"  [Chunk {i}] Input tensor shape: {inputs['input_values'].shape}")
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Extract using Layer Fusion
                if self.attention:
                    emb = self._get_layer_fusion_embedding_attention(inputs)
                else:
                    if i == 0:                    
                        emb = self._get_layer_fusion_embedding(inputs, printing = True)
                    else:
                        emb = self._get_layer_fusion_embedding(inputs, printing = False)

                # [DEBUG] Check embedding shape immediately after generation
                if i == 0:
                    print(f"  [Chunk {i}] Output embedding shape: {emb.shape}")

                if emb.dim() == 0: 
                    # Es un escalar, intentamos convertirlo o lo saltamos
                    print(f"   Warning: Scalar embedding produced for {mp3_path}. Skipping chunk.")
                    continue

                for j in range(emb.size(0)):
                    chunk_embeddings.append(emb[j])

            return chunk_embeddings

        except Exception as e:
            if "out of memory" in str(e):
                print(f"   OOM skipping {mp3_path}")
                torch.cuda.empty_cache()
            else:
                print(f"Error processing {mp3_path}: {e}")
            return None

    def process_genres(self, base_directory: str, genre_list: List[str], max_songs: int = None, save_dir : str = "./"):        
        for genre in genre_list:
            print(f"\nProcessing genre: {genre}")
            genre_dir = os.path.join(base_directory, genre)
            if not os.path.isdir(genre_dir):
                print(f"Warning: Directory not found: {genre_dir}")
                continue

            mp3_files_raw = sorted(f for f in os.listdir(genre_dir) if f.endswith(".mp3"))
            
            if max_songs:
                # Diccionario para agrupar partes bajo el mismo Parent ID
                canciones_agrupadas = {}
                for f in mp3_files_raw:
                    # Extraemos el nombre base 
                    if "_part" in f:
                        base_name = f.rsplit('_part', 1)[0]
                    else:
                        base_name = os.path.splitext(f)[0]
                        
                    if base_name not in canciones_agrupadas:
                        canciones_agrupadas[base_name] = []
                    canciones_agrupadas[base_name].append(f)
                
                # Seleccionamos las primeras 'max_songs' canciones completadas
                mp3_files = []
                for base_name in list(canciones_agrupadas.keys())[:max_songs]:
                    mp3_files.extend(canciones_agrupadas[base_name])
                
                print(f"  [Info] Selected {len(list(canciones_agrupadas.keys())[:max_songs])} unique songs (Total {len(mp3_files)} chunks/files)")
            else:
                mp3_files = mp3_files_raw

            # Temporal lists only for the current genre
            genre_embeddings = []
            genre_labels = []
            genre_song_names = []

            for counter, song_file in enumerate(mp3_files, start=1):
                file_path = os.path.join(genre_dir, song_file)
                print(f"\nProcessing ({counter}/{len(mp3_files)}) {genre}: {song_file}")
                
                # Turns "Song Name_part1.mp3" into "Song Name"
                if "_part" in song_file:
                    base_name = song_file.rsplit('_part', 1)[0]
                else:
                    base_name = os.path.splitext(song_file)[0]

                base_name = f"{genre}_{base_name}"

                # Get list of chunks
                chunks = self.get_song_chunks(file_path, batch_size=16)

                if chunks is not None:
                    for chunk_vec in chunks:
                        # [DEBUG] Final check before adding to the big list
                        # print(f"    Appending vector of shape {chunk_vec.shape}")
                        # Move to CPU inmediately after for avoiding GPU saturation
                        genre_embeddings.append(chunk_vec.cpu())
                        genre_labels.append(genre)
                        genre_song_names.append(base_name)

            # Saving feature space for the current genre
            if genre_embeddings:
                print(f"Stacking tensors for {genre}...")
                embedding_matrix = torch.stack(genre_embeddings)
                print(f'Total chunks processed: {len(genre_embeddings)}')
                print(f"Shape: {embedding_matrix.shape}")
                
                safe_genre = self._sanitize_filename(genre)
                filename = f"{self.only_name_model}_attention{self.attention}_{safe_genre}.pt"
                save_path = os.path.join(save_dir, filename)
                
                print(f"Saving {genre} data to: {save_path}")
                torch.save({
                    'embeddings': embedding_matrix, 
                    'labels': genre_labels,
                    'song_names': genre_song_names
                }, save_path)
                
                del embedding_matrix
                del genre_embeddings
                del genre_labels
                del genre_song_names
                import gc
                gc.collect()

        print("\nDONE. All genres saved individually.")
        return torch.empty(0), [] 
    
