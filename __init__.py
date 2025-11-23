"""
Speech-to-Text: Deep learning models for automatic speech recognition.

Modules:
    models: Neural network architectures (MLP, CNN, LSTM, GRU, BiLSTM)
    data: Data loading and feature extraction utilities
    trainer: Training loops and model management
    train: Main training script
"""

from .models import MLP, CNN, RNN_LSTM, RNN_GRU, RNN_BiLSTM
from .data import TextEncoder, FeatureExtractor, create_dataloaders, greedy_decode
from .trainer import Trainer, CNNTrainer, TrainingConfig, ModelComparator

__version__ = "0.1.0"
__all__ = [
    'MLP',
    'CNN',
    'RNN_LSTM',
    'RNN_GRU',
    'RNN_BiLSTM',
    'TextEncoder',
    'FeatureExtractor',
    'create_dataloaders',
    'greedy_decode',
    'Trainer',
    'CNNTrainer',
    'TrainingConfig',
    'ModelComparator',
]
