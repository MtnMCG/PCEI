"""
PCEI File Ingestors — Real Files → Model Tensors

Converts actual files from disk into the tensor formats expected by each Stack:
- Text (.txt, .md, .pdf) → token IDs for TextStack.embedding
- Images (.png, .jpg) → flattened 16x16x3 patches for ImageStack.patch_proj
- Audio (.mp3, .wav) → mel spectrograms for AudioStack.audio_proj
- Video (.mp4) → frames to ImageStack + audio track to AudioStack
"""
import torch
import os
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Image processing
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Audio processing
try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

# Video processing
import subprocess
import tempfile


# ═══════════════════════════════════════════
# TEXT INGESTOR
# ═══════════════════════════════════════════
class TextIngestor:
    """
    Character-level tokenizer for the prototype.
    Maps characters to integer IDs. Simple but functional —
    every char gets a unique ID, unknown chars map to <UNK>.
    
    For production: swap this with tiktoken or SentencePiece.
    """
    def __init__(self, vocab_size: int = 1000, max_seq_len: int = 16):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Build character vocab from printable ASCII + common unicode
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        
        # Special tokens
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        
        self.char_to_id['<PAD>'] = 0
        self.char_to_id['<UNK>'] = 1
        self.char_to_id['<BOS>'] = 2
        self.char_to_id['<EOS>'] = 3
        
        # Map printable chars
        idx = 4
        for i in range(32, 127):  # ASCII printable
            ch = chr(i)
            self.char_to_id[ch] = idx
            self.id_to_char[idx] = ch
            idx += 1
        
        # Common extended chars
        for ch in "àáâãäåèéêëìíîïòóôõöùúûüýÿñçßæœ":
            if ch not in self.char_to_id and idx < vocab_size:
                self.char_to_id[ch] = idx
                self.id_to_char[idx] = ch
                idx += 1
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to token IDs tensor of shape (seq_len,)."""
        ids = [self.bos_id]
        for ch in text:
            if ch in self.char_to_id:
                ids.append(self.char_to_id[ch])
            else:
                ids.append(self.unk_id)
        ids.append(self.eos_id)
        
        # Truncate or pad to max_seq_len
        if len(ids) > self.max_seq_len:
            ids = ids[:self.max_seq_len]
        else:
            ids = ids + [self.pad_id] * (self.max_seq_len - len(ids))
        
        return torch.tensor(ids, dtype=torch.long)
    
    def decode(self, ids: torch.Tensor) -> str:
        """Decode token IDs back to text."""
        chars = []
        for idx in ids.tolist():
            if idx in (self.pad_id, self.bos_id, self.eos_id):
                continue
            if idx in self.id_to_char:
                chars.append(self.id_to_char[idx])
            else:
                chars.append('?')
        return ''.join(chars)
    
    def encode_file(self, filepath: Path) -> torch.Tensor:
        """Reads a text file and returns batched token tensors.
        Splits the file into max_seq_len chunks, returns (num_chunks, seq_len)."""
        ext = filepath.suffix.lower()
        
        if ext == '.pdf':
            text = self._read_pdf(filepath)
        else:
            text = filepath.read_text(encoding='utf-8', errors='replace')
        
        if not text.strip():
            return self.encode("empty document")
        
        # Split into chunks
        chunks = []
        for i in range(0, len(text), self.max_seq_len - 2):  # -2 for BOS/EOS
            chunk = text[i:i + self.max_seq_len - 2]
            chunks.append(self.encode(chunk))
        
        if not chunks:
            chunks = [self.encode("")]
            
        return torch.stack(chunks)  # (num_chunks, seq_len)
    
    def _read_pdf(self, filepath: Path) -> str:
        try:
            import fitz
            doc = fitz.open(str(filepath))
            return "\n".join(page.get_text() for page in doc)
        except ImportError:
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(str(filepath))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                return filepath.read_text(encoding='utf-8', errors='replace')


# ═══════════════════════════════════════════
# IMAGE INGESTOR
# ═══════════════════════════════════════════
class ImageIngestor:
    """
    Converts images to the format expected by ImageStack:
    - Resize to target_size (32x32 for prototype)
    - Split into patch_size x patch_size patches
    - Flatten each patch to (patch_size^2 * 3) vector
    - Returns (num_patches, patch_dim) tensor
    """
    def __init__(self, target_size: int = 32, patch_size: int = 16):
        self.target_size = target_size
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size * 3
        self.num_patches = (target_size // patch_size) ** 2
    
    def load_image(self, filepath: Path) -> torch.Tensor:
        """
        Load image file → (1, num_patches, patch_dim) tensor.
        Ready to feed into ImageStack.patch_proj.
        """
        if not HAS_PIL:
            raise ImportError("Pillow required: pip install pillow")
        
        img = Image.open(filepath).convert('RGB')
        img = img.resize((self.target_size, self.target_size), Image.LANCZOS)
        
        # Convert to tensor (C, H, W) normalized to [0, 1]
        import numpy as np
        arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
        tensor = torch.from_numpy(arr)  # (H, W, 3)
        
        # Extract patches
        patches = []
        for i in range(0, self.target_size, self.patch_size):
            for j in range(0, self.target_size, self.patch_size):
                patch = tensor[i:i+self.patch_size, j:j+self.patch_size, :]  # (P, P, 3)
                patches.append(patch.reshape(-1))  # (P*P*3,)
        
        # Flatten to (num_patches, P*P*3)
        img_tensor = torch.stack(patches)
        
        # Zero-Mean Unit-Variance Normalization to prevent polynomial explosion
        # Mean across the feature dimension (P*P*3) per patch
        mean = img_tensor.mean(dim=-1, keepdim=True)
        std = img_tensor.std(dim=-1, keepdim=True) + 1e-5
        img_normalized = (img_tensor - mean) / std
        
        return img_normalized.unsqueeze(0)  # (1, num_patches, patch_dim)


# ═══════════════════════════════════════════
# AUDIO INGESTOR
# ═══════════════════════════════════════════
class AudioIngestor:
    """
    Converts audio files to mel spectrograms for AudioStack.
    Output shape: (1, num_frames, mel_bins=80)
    """
    def __init__(self, sample_rate: int = 16000, n_mels: int = 80, 
                 max_duration: float = 5.0):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_duration = max_duration
        self.max_samples = int(sample_rate * max_duration)
    
    def load_audio(self, filepath: Path) -> torch.Tensor:
        """
        Load audio file → (1, num_frames, 80) mel spectrogram tensor.
        Ready to feed into AudioStack.audio_proj.
        """
        import numpy as np
        import soundfile as sf
        
        try:
            # Robust loading with soundfile (bypasses broken torchaudio C++ bindings on Mac)
            waveform_np, sr = sf.read(str(filepath), dtype='float32')
        except Exception as e:
            raise RuntimeError(f"Failed to load audio {filepath} via soundfile: {e}")
            
        # Mono conversion
        if len(waveform_np.shape) > 1:
            waveform_np = waveform_np.mean(axis=1)
            
        waveform = torch.from_numpy(waveform_np).unsqueeze(0)  # (1, T)
        
        # Resample if needed using torchaudio
        if HAS_TORCHAUDIO and sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        
        # Truncate to max duration
        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        
        # Compute mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=1024,
            hop_length=512
        )
        mel = mel_transform(waveform)  # (1, n_mels, time)
        
        # Log scale (can produce large negative values like -20)
        mel = torch.log(mel + 1e-9)
        
        # Transpose to (1, time, n_mels) for AudioStack
        mel = mel.squeeze(0).transpose(0, 1)  # (time, n_mels)
        
        # Zero-Mean Unit-Variance Normalization to prevent polynomial explosion
        mean = mel.mean(dim=-1, keepdim=True)
        std = mel.std(dim=-1, keepdim=True) + 1e-5
        mel_normalized = (mel - mean) / std
        
        return mel_normalized.unsqueeze(0)  # (1, time, n_mels)


# ═══════════════════════════════════════════
# VIDEO INGESTOR
# ═══════════════════════════════════════════
class VideoIngestor:
    """
    Splits video into synchronized visual frames and audio track.
    - Frames → ImageIngestor → stacked patches for ImageStack
    - Audio → AudioIngestor → mel spectrogram for AudioStack
    """
    def __init__(self, max_frames: int = 8, fps: float = 2.0):
        self.max_frames = max_frames
        self.fps = fps
        self.image_ingestor = ImageIngestor()
        self.audio_ingestor = AudioIngestor()
    
    def load_video(self, filepath: Path) -> Dict[str, torch.Tensor]:
        """
        Returns {'image': (1, total_patches, patch_dim), 'audio': (1, time, 80), 'text': str}
        Uses fast-seeking to extract only a subset of frames/audio for speed.
        """
        results = {}
        transcript_text = None
        
        # Look for transcript
        transcript_path = filepath.parent / "transcript.json"
        if not transcript_path.exists():
             transcript_path = filepath.with_suffix(".json")
             
        transcript_data = []
        if transcript_path.exists():
            try:
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript_data = json.load(f)
                # Full text fallback
                transcript_text = " ".join([seg['text'] for seg in transcript_data])
            except (json.JSONDecodeError, KeyError, OSError) as e:
                print(f"[Warning] Failed to load transcript {transcript_path}: {e}")

        if filepath.is_dir():
            # Pre-processed logic remains same
            frame_files = sorted(filepath.glob("frame_*.png"))
            audio_path = filepath.parent / "audio" / f"{filepath.name}.wav"
            frame_tensors = []
            if frame_files:
                for ff in frame_files[:self.max_frames]:
                    patches = self.image_ingestor.load_image(ff)
                    frame_tensors.append(patches.squeeze(0))
            if frame_tensors:
                stacked = torch.stack(frame_tensors).view(-1, self.image_ingestor.patch_dim)
                # Pad/Crop to max_frames * patches_per_frame
                target_patches = self.max_frames * self.image_ingestor.num_patches
                if stacked.shape[0] > target_patches:
                    stacked = stacked[:target_patches]
                elif stacked.shape[0] < target_patches:
                    pad = torch.zeros(target_patches - stacked.shape[0], stacked.shape[1])
                    stacked = torch.cat([stacked, pad])
                results['image'] = stacked.unsqueeze(0)
            else:
                results['image'] = torch.randn(1, self.max_frames * self.image_ingestor.num_patches, self.image_ingestor.patch_dim)
            if audio_path.exists():
                results['audio'] = self.audio_ingestor.load_audio(audio_path)
            else:
                results['audio'] = torch.randn(1, 100, 80)
            if transcript_text: results['text_raw'] = transcript_text
            return results

        import tempfile
        import subprocess
        import random
        
        # 1. Get Duration for random seeking
        duration = 60.0 # fallback
        try:
            probe = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(filepath)
            ], capture_output=True, text=True)
            duration = float(probe.stdout.strip())
        except (subprocess.SubprocessError, ValueError, OSError):
            pass  # ffprobe may not be available; fallback duration used

        # 2. Pick a random window
        window_size = self.max_frames / self.fps
        start_time = random.uniform(0, max(0, duration - window_size - 1))
        
        # 3. Targeted Transcript Slice
        if transcript_data:
            window_text = [seg['text'] for seg in transcript_data 
                          if seg.get('start', 0) >= start_time and seg.get('start', 0) <= start_time + window_size]
            if window_text:
                transcript_text = " ".join(window_text)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Fast seek extraction: extract max_frames starting at start_time
            try:
                subprocess.run([
                    "ffmpeg", "-ss", f"{start_time:.2f}", "-i", str(filepath),
                    "-frames:v", str(self.max_frames),
                    "-vf", f"fps={self.fps}",
                    "-loglevel", "error",
                    str(tmpdir_path / "frame_%04d.png")
                ], check=True, capture_output=True)
            except (subprocess.SubprocessError, OSError) as e:
                print(f"[Warning] ffmpeg frame extraction failed: {e}")

            frame_files = sorted(tmpdir_path.glob("frame_*.png"))
            if frame_files:
                frame_tensors = []
                for ff in frame_files:
                    patches = self.image_ingestor.load_image(ff)
                    frame_tensors.append(patches.squeeze(0))
                all_patches = torch.cat(frame_tensors, dim=0) # (total_patches, patch_dim)
                
                # Pad/Crop to exact target
                target_patches = self.max_frames * self.image_ingestor.num_patches
                if all_patches.shape[0] > target_patches:
                    all_patches = all_patches[:target_patches]
                elif all_patches.shape[0] < target_patches:
                    pad = torch.zeros(target_patches - all_patches.shape[0], all_patches.shape[1])
                    all_patches = torch.cat([all_patches, pad])
                    
                results['image'] = all_patches.unsqueeze(0)
            else:
                results['image'] = torch.randn(1, self.max_frames * self.image_ingestor.num_patches, self.image_ingestor.patch_dim)
            
            # Targeted audio extraction (5 seconds)
            audio_path = tmpdir_path / "audio.wav"
            try:
                subprocess.run([
                    "ffmpeg", "-ss", f"{start_time:.2f}", "-i", str(filepath),
                    "-t", "5.0",
                    "-vn", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1",
                    "-loglevel", "error",
                    str(audio_path)
                ], check=True, capture_output=True)
                
                if audio_path.exists():
                    results['audio'] = self.audio_ingestor.load_audio(audio_path)
                else:
                    results['audio'] = torch.randn(1, 100, 80)
            except (subprocess.SubprocessError, OSError, RuntimeError) as e:
                print(f"[Warning] Audio extraction failed: {e}")
                results['audio'] = torch.randn(1, 100, 80)
                
        if transcript_text: results['text_raw'] = transcript_text
        return results


# ═══════════════════════════════════════════
# UNIFIED DATASET
# ═══════════════════════════════════════════
class PCEIDataset:
    """
    Unified dataset that reads the training_database.jsonl and loads
    actual files using the appropriate ingestor.
    
    Returns batches of real tensors ready for the model.
    """
    def __init__(self, db_path: Path, vocab_size: int = 1000, max_seq_len: int = 128, tokenizer=None):
        self.db_path = db_path
        self.records: List[Dict] = []
        self.custom_text_pairs: List[Tuple[str, str]] = []
        self.custom_records: List[Dict] = []
        self.tokenizer = tokenizer
        
        self.known_paths = set()
        
        if db_path.exists():
            with open(db_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            # Schema check
                            if not all(k in record for k in ('id', 'type', 'text', 'image_path', 'audio_path')):
                                print(f"[Warning] Skipping malformed record: {record.get('id', 'unknown')}")
                                continue
                            
                            r_img = record.get('image_path')
                            r_aud = record.get('audio_path')
                            
                            # Resolve paths relative to project root
                            root_path = db_path.parent.parent.parent
                            img_ok = r_img is None or (root_path / r_img).exists()
                            aud_ok = r_aud is None or (root_path / r_aud).exists()
                            
                            if img_ok and aud_ok:
                                self.records.append(record)
                                # Track known filenames for deduplication
                                if r_img: self.known_paths.add(Path(r_img).name)
                                if r_aud: self.known_paths.add(Path(r_aud).name)
                            else:
                                print(f"[PCEIDataset] Missing file for record {record['id']}")
                        except json.JSONDecodeError:
                            pass
        
        # Ingestors
        # text_ingestor is only used if no tokenizer is provided
        self.text_ingestor = TextIngestor(vocab_size=vocab_size, max_seq_len=max_seq_len) if tokenizer is None else tokenizer
        self.image_ingestor = ImageIngestor()
        self.audio_ingestor = AudioIngestor()
        self.video_ingestor = VideoIngestor()

        # Load custom/synthetic user logic
        root_path = db_path.parent.parent.parent if db_path else Path.cwd()
        self._load_custom_data(root_path)

        # Apply Priority Weighting
        # Evenly mix custom multimodal data without artificial duplication
        self.records = self.custom_records + self.records

        # Also ingest labels as text training data
        base_label_pairs = self._extract_label_pairs()
        
        # Evenly mix custom text logic without artificial duplication
        self.label_pairs = self.custom_text_pairs + base_label_pairs
        
        print(f"[PCEIDataset] Loaded {len(self.records)} records ({len(self.custom_records)} custom, even mix)")
        print(f"[PCEIDataset] {len(self.label_pairs)} text pairs ({len(self.custom_text_pairs)} custom, even mix)")
        
    def _load_custom_data(self, root_path: Path):
        """Load user's custom synthetic text logic."""
        # 1. Synthetic reasoning matching {"text": "..."}
        synth_dir = root_path / "data" / "synthetic"
        if synth_dir.exists():
            for fpath in synth_dir.glob("*.jsonl"):
                with open(fpath, "r") as f:
                    for line in f:
                        if line.strip():
                            try:
                                d = json.loads(line)
                                if "text" in d:
                                    self.custom_text_pairs.append((fpath.name, d["text"]))
                            except json.JSONDecodeError: pass
                            
        # 2. Raw custom text
        txt_dir = root_path / "data" / "raw" / "text"
        if txt_dir.exists():
            for fpath in txt_dir.glob("*.txt"):
                # Skip COCO text files as they are in the database
                if fpath.name.startswith("coco_"): continue
                try:
                    content = fpath.read_text(encoding='utf-8', errors='ignore').strip()
                    if content:
                        # Chunk books into paragraphs so they fit within context window
                        chunks = [c.strip() for c in content.split('\n\n') if len(c.strip()) > 20]
                        for chunk in chunks:
                            self.custom_text_pairs.append((fpath.name, chunk))
                except Exception: pass
                            
        # 3. Targeted Multimodal (Video + Transcript)
        target_dir = root_path / "data" / "raw" / "targeted_video"
        if target_dir.exists():
            for fpath in target_dir.glob("*.mp4"):
                # Add as a record
                self.custom_records.append({
                    "id": f"targeted_{fpath.stem}",
                    "type": "video_multimodal",
                    "text": None, # Will be filled from transcript by ingestor
                    "image_path": str(fpath.relative_to(root_path)),
                    "audio_path": None
                })
    
    def _extract_label_pairs(self) -> List[Tuple[str, str]]:
        """Extract (id, text) pairs as text training data."""
        pairs = []
        for r in self.records:
            text = r.get('text', '')
            r_id = r.get('id', 'unknown')
            if text:
                pairs.append((r_id, text))
        return pairs
    
    def __len__(self):
        return max(len(self.records), 1)
    
    def get_text_batch(self, batch_size: int = 2) -> torch.Tensor:
        """Returns tokenized text from labels as training data."""
        if not self.label_pairs:
            return self.text_ingestor.encode("no data available yet").unsqueeze(0).expand(batch_size, -1)
        
        import random
        batch = []
        for _ in range(batch_size):
            fname, label = random.choice(self.label_pairs)
            if hasattr(self.text_ingestor, 'encode'):
                tokens = self.text_ingestor.encode(f"{fname}: {label}")
            else:
                tokens = self.text_ingestor(f"{fname}: {label}")  # callable
            batch.append(tokens)
        
        return torch.stack(batch)  # (B, seq_len)
    
    def get_image_batch(self, batch_size: int = 2) -> Optional[torch.Tensor]:
        """Load batch of real images."""
        image_records = [r for r in self.records if r.get('image_path') is not None]
        if not image_records:
            return None
        
        import random
        batch = []
        root_path = self.db_path.parent.parent.parent
        for _ in range(batch_size):
            r = random.choice(image_records)
            try:
                tensor = self.image_ingestor.load_image(root_path / r['image_path'])
                batch.append(tensor.squeeze(0))  # (num_patches, patch_dim)
            except Exception as e:
                print(f"  [Warning] Failed to load image {r['image_path']}: {e}")
                continue
        
        if not batch:
            return None
        return torch.stack(batch)  # (B, num_patches, patch_dim)
    
    def get_audio_batch(self, batch_size: int = 2) -> Optional[torch.Tensor]:
        """Load batch of real audio mel spectrograms."""
        audio_records = [r for r in self.records if r.get('audio_path') is not None]
        if not audio_records:
            return None
        
        import random
        target_len = 100  # Pad/truncate mel frames
        batch = []
        root_path = self.db_path.parent.parent.parent
        for _ in range(batch_size):
            r = random.choice(audio_records)
            try:
                mel = self.audio_ingestor.load_audio(root_path / r['audio_path'])  # (1, T, 80)
                mel = mel.squeeze(0)  # (T, 80)
                # Pad or truncate
                if mel.shape[0] > target_len:
                    mel = mel[:target_len]
                elif mel.shape[0] < target_len:
                    pad = torch.zeros(target_len - mel.shape[0], mel.shape[1])
                    mel = torch.cat([mel, pad])
                batch.append(mel)
            except Exception as e:
                print(f"  [Warning] Failed to load audio {r['audio_path']}: {e}")
                continue
        
        if not batch:
            return None
        return torch.stack(batch)  # (B, target_len, 80)
    
    def get_multimodal_batch(self, batch_size: int = 2) -> Dict[str, Optional[torch.Tensor]]:
        """
        Returns a synchronized multimodal batch.
        Each sample in the batch contains a text label/explanation paired with its image or audio.
        """
        import random
        
        batch_text = []
        batch_image = []
        batch_audio = []
        
        # Target lengths for consistency
        audio_target_len = 100
        
        for _ in range(batch_size):
            # Pick a random record
            r = random.choice(self.records)
            label = r.get('text', 'unlabeled signal')
            img_path = r.get('image_path')
            aud_path = r.get('audio_path')
            root_path = self.db_path.parent.parent.parent
            
            # 1. Handle Text Component (Always exists for labeled data)
            safe_label = label if label is not None else "multimodal signal"
            if hasattr(self.text_ingestor, 'encode'):
                batch_text.append(self.text_ingestor.encode(safe_label))
            else:
                batch_text.append(self.text_ingestor(safe_label))
                
            # 2. Handle Multimodal Components
            target_image = None
            target_audio = None
            
            # Correct type matching
            if r.get('type') in ('video_multimodal', 'video', 'custom_video'):
                # Note: 'custom_video' uses audio_path for the raw file if frames not extracted
                video_res = self.video_ingestor.load_video(root_path / (img_path or aud_path or ""))
                target_image = video_res.get('image')
                target_audio = video_res.get('audio')
                # Optional: Override label with transcript text
                if video_res.get('text_raw'):
                    label = video_res['text_raw'][:500] 
                    # Re-tokenize
                    if hasattr(self.text_ingestor, 'encode'):
                        batch_text[-1] = self.text_ingestor.encode(label)
                    else:
                        batch_text[-1] = self.text_ingestor(label)
            else:
                if img_path:
                    try:
                        target_image = self.image_ingestor.load_image(root_path / img_path)
                    except (OSError, RuntimeError) as e:
                        print(f"[Warning] Failed to load image {img_path}: {e}")
                if aud_path:
                    try:
                        target_audio = self.audio_ingestor.load_audio(root_path / aud_path)
                    except (OSError, RuntimeError) as e:
                        print(f"[Warning] Failed to load audio {aud_path}: {e}")
            
            # --- PAD / FORMAT ---
            # Pad/Crop ti to match multimodal expectations (max_frames * patches_per_frame)
            img_target_len = self.video_ingestor.max_frames * self.image_ingestor.num_patches
            if target_image is not None:
                # Ensure we have the right shape: (seq_len, patch_dim)
                ti = target_image.squeeze(0) if target_image.dim() == 3 else target_image
                if ti.shape[0] > img_target_len:
                    ti = ti[:img_target_len]
                elif ti.shape[0] < img_target_len:
                    pad = torch.zeros(img_target_len - ti.shape[0], ti.shape[1])
                    ti = torch.cat([ti, pad])
                batch_image.append(ti)
            else:
                batch_image.append(torch.randn(img_target_len, self.image_ingestor.patch_dim))
            
            # Audio
            if target_audio is not None:
                mel = target_audio.squeeze(0) # (T, 80)
                if mel.shape[0] > audio_target_len:
                    mel = mel[:audio_target_len]
                elif mel.shape[0] < audio_target_len:
                    pad = torch.zeros(audio_target_len - mel.shape[0], mel.shape[1])
                    mel = torch.cat([mel, pad])
                batch_audio.append(mel)
            else:
                batch_audio.append(torch.randn(audio_target_len, 80))
        
        return {
            "text": torch.stack(batch_text) if batch_text else None,
            "image": torch.stack(batch_image) if batch_image else None,
            "audio": torch.stack(batch_audio) if batch_audio else None
        }
