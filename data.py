"""
Data processing utilities for speech-to-text tasks.
Includes feature extraction, text encoding, and data loading functions.
"""

import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchaudio.transforms import MFCC, MelSpectrogram


class TextEncoder:
    """
    Encodes text to integer sequences and decodes predictions back to text.
    """
    def __init__(self, symbols="_-!'(),.:;? abcdefghijklmnopqrstuvwxyz", blank_idx=0):
        """
        Initialize text encoder.
        
        Args:
            symbols (str): Character set to encode
            blank_idx (int): Index for CTC blank token
        """
        self.symbols = symbols
        self.blank_idx = blank_idx
        
        # Create mappings
        self.char2idx = {s: i for i, s in enumerate(symbols)}
        self.label_map = {s: i + 1 for s, i in self.char2idx.items()}
        self.idx2char = {i: s for s, i in self.label_map.items()}
        self.idx2char[blank_idx] = ''
        
        self.num_classes = len(self.label_map) + 1
    
    def encode(self, text):
        """
        Encode text to tensor.
        
        Args:
            text (str): Text to encode
            
        Returns:
            torch.Tensor: Encoded text
        """
        return torch.tensor(
            [self.label_map[c] for c in text.lower() if c in self.symbols],
            dtype=torch.long
        )
    
    def decode(self, indices):
        """
        Decode indices to text.
        
        Args:
            indices (list or torch.Tensor): Indices to decode
            
        Returns:
            str: Decoded text
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        
        text = []
        prev = None
        for idx in indices:
            idx_int = int(idx)
            if idx_int != prev and idx_int != self.blank_idx:
                text.append(self.idx2char.get(idx_int, ''))
            prev = idx_int
        return ''.join(text)


class FeatureExtractor:
    """
    Extracts acoustic features from waveforms (MFCC or MelSpectrogram).
    """
    def __init__(self, feature_type='mfcc', sample_rate=22050):
        """
        Initialize feature extractor.
        
        Args:
            feature_type (str): 'mfcc' or 'mel'
            sample_rate (int): Sample rate of audio
        """
        self.feature_type = feature_type
        self.sample_rate = sample_rate
        
        if feature_type == 'mfcc':
            self.transform = MFCC(sample_rate=sample_rate, n_mfcc=128)
        elif feature_type == 'mel':
            self.transform = MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=64,
                n_fft=400,
                hop_length=160
            )
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")
    
    def extract(self, waveform, sr):
        """
        Extract features from waveform.
        
        Args:
            waveform (torch.Tensor): Audio waveform
            sr (int): Sample rate of waveform
            
        Returns:
            torch.Tensor: Extracted features of shape (T, n_features)
        """
        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        # Extract features
        features = self.transform(waveform)
        
        # Process based on feature type
        if self.feature_type == 'mfcc':
            # MFCC: (1, n_mfcc, time) -> (time, n_mfcc)
            features = features.mean(dim=0).transpose(0, 1)
        else:
            # MelSpectrogram: (1, n_mels, time) -> (time, n_mels)
            features = features.squeeze(0)
            features = torch.log(features + 1e-9)
            features = features.transpose(0, 1)
        
        return features


class STTDataset:
    """
    Custom dataset wrapper for speech-to-text tasks.
    """
    def __init__(self, base_dataset, feature_extractor, text_encoder):
        """
        Initialize dataset.
        
        Args:
            base_dataset: PyTorch dataset with (waveform, sr, transcript) tuples
            feature_extractor (FeatureExtractor): Feature extraction object
            text_encoder (TextEncoder): Text encoding object
        """
        self.base_dataset = base_dataset
        self.feature_extractor = feature_extractor
        self.text_encoder = text_encoder
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        waveform, sr, transcript = self.base_dataset[idx][0], self.base_dataset[idx][1], self.base_dataset[idx][2]
        features = self.feature_extractor.extract(waveform, sr)
        encoded_text = self.text_encoder.encode(transcript)
        return features, encoded_text


def create_collate_fn(text_encoder):
    """
    Create a collate function for DataLoader.
    
    Args:
        text_encoder (TextEncoder): Text encoding object
        
    Returns:
        function: Collate function for DataLoader
    """
    def collate_fn(batch):
        """
        Collate batch samples.
        
        Args:
            batch (list): List of (features, encoded_text) tuples
            
        Returns:
            tuple: (feats_padded, feat_lens, targets_concat, target_lens)
        """
        feats = []
        feat_lens = []
        targets = []
        target_lens = []
        
        for features, encoded_text in batch:
            feats.append(features)
            feat_lens.append(features.size(0))
            targets.append(encoded_text)
            target_lens.append(len(encoded_text))
        
        feats_padded = pad_sequence(feats, batch_first=True)
        targets_concat = torch.cat(targets) if targets else torch.tensor([], dtype=torch.long)
        
        return (
            feats_padded,
            torch.tensor(feat_lens, dtype=torch.long),
            targets_concat,
            torch.tensor(target_lens, dtype=torch.long)
        )
    
    return collate_fn


def create_dataloaders(train_dataset, test_dataset, feature_extractor, 
                       text_encoder, batch_size=16, num_workers=0):
    """
    Create train and test dataloaders.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        feature_extractor (FeatureExtractor): Feature extraction object
        text_encoder (TextEncoder): Text encoding object
        batch_size (int): Batch size
        num_workers (int): Number of workers for DataLoader
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Wrap datasets
    train_wrapped = STTDataset(train_dataset, feature_extractor, text_encoder)
    test_wrapped = STTDataset(test_dataset, feature_extractor, text_encoder)
    
    # Create collate function
    collate_fn = create_collate_fn(text_encoder)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_wrapped,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_wrapped,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    return train_loader, test_loader


def greedy_decode(log_probs, text_encoder):
    """
    Decode predictions using greedy decoding.
    
    Args:
        log_probs (torch.Tensor): Model output of shape (T, N, C)
        text_encoder (TextEncoder): Text encoding object
        
    Returns:
        list: Decoded predictions
    """
    preds = log_probs.argmax(dim=-1)    # (T, N)
    preds = preds.transpose(0, 1)       # (N, T)
    decoded = []
    
    for seq in preds:
        text = text_encoder.decode(seq)
        decoded.append(text)
    
    return decoded
