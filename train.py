"""
Main script for training and evaluating speech-to-text models.
Demonstrates usage of models, data processing, and training utilities.
"""

import torch
from torch.utils.data import random_split
from torchaudio.datasets import LJSPEECH

from models import MLP, CNN, RNN_LSTM, RNN_GRU, RNN_BiLSTM
from data import TextEncoder, FeatureExtractor, create_dataloaders, greedy_decode
from trainer import TrainingConfig, Trainer, CNNTrainer, ModelComparator


def setup_device():
    """Setup computation device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    return device


def load_dataset(root="./data"):
    """
    Load LJSPEECH dataset and split into train/test.
    
    Args:
        root (str): Path to dataset directory
        
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    print("Loading LJSPEECH dataset...")
    dataset = LJSPEECH(root=root, download=True)
    
    train_len = int(len(dataset) * 0.9)
    test_len = len(dataset) - train_len
    
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])
    print(f"Dataset loaded: {train_len} train, {test_len} test")
    
    return train_dataset, test_dataset


def create_decoders():
    """Create text encoder."""
    text_encoder = TextEncoder()
    print(f"Text encoder created with {text_encoder.num_classes} classes")
    return text_encoder


def train_mlp_model(train_loader, test_loader, text_encoder, device):
    """
    Train MLP + MFCC model.
    
    Args:
        train_loader: Training DataLoader
        test_loader: Test DataLoader
        text_encoder: Text encoder/decoder
        device: Computation device
        
    Returns:
        dict: Training metrics
    """
    # Feature extraction with MFCC
    feature_extractor = FeatureExtractor(feature_type='mfcc')
    train_loader, test_loader = create_dataloaders(
        train_dataset, test_dataset, feature_extractor, 
        text_encoder, batch_size=16
    )
    
    # Create model
    config = TrainingConfig(num_epochs=2, learning_rate=1e-3, device=device)
    model = MLP(in_dim=128, hidden=256, num_classes=text_encoder.num_classes)
    
    # Train
    trainer = Trainer(model, config, blank_idx=0)
    
    def decode_fn(log_probs):
        return greedy_decode(log_probs, text_encoder)
    
    metrics = trainer.train(train_loader, test_loader, decode_fn, "MLP + MFCC")
    trainer.save("mlp_ctc.pth")
    
    return metrics


def train_cnn_model(train_dataset, test_dataset, text_encoder, device):
    """
    Train CNN + MelSpectrogram model.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        text_encoder: Text encoder/decoder
        device: Computation device
        
    Returns:
        dict: Training metrics
    """
    # Feature extraction with MelSpectrogram
    feature_extractor = FeatureExtractor(feature_type='mel')
    train_loader, test_loader = create_dataloaders(
        train_dataset, test_dataset, feature_extractor, 
        text_encoder, batch_size=16
    )
    
    # Create model
    config = TrainingConfig(num_epochs=2, learning_rate=1e-3, device=device)
    model = CNN(n_mels=64, num_classes=text_encoder.num_classes)
    
    # Train with CNN-specific trainer
    trainer = CNNTrainer(model, config, blank_idx=0)
    
    def decode_fn(log_probs):
        return greedy_decode(log_probs, text_encoder)
    
    metrics = trainer.train(train_loader, test_loader, decode_fn, "CNN + MelSpectrogram")
    trainer.save("cnn_mel_ctc.pth")
    
    return metrics


def train_rnn_models(train_dataset, test_dataset, text_encoder, device):
    """
    Train all RNN variants (LSTM, GRU, BiLSTM).
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        text_encoder: Text encoder/decoder
        device: Computation device
        
    Returns:
        dict: Training metrics for all RNN models
    """
    # Feature extraction with MelSpectrogram
    feature_extractor = FeatureExtractor(feature_type='mel')
    train_loader, test_loader = create_dataloaders(
        train_dataset, test_dataset, feature_extractor, 
        text_encoder, batch_size=16
    )
    
    rnn_metrics = {}
    models_to_train = [
        (RNN_LSTM, "LSTM + MelSpectrogram"),
        (RNN_GRU, "GRU + MelSpectrogram"),
        (RNN_BiLSTM, "BiLSTM + MelSpectrogram"),
    ]
    
    for model_class, model_name in models_to_train:
        print("\n" + "=" * 70)
        
        # Create model
        config = TrainingConfig(num_epochs=2, learning_rate=1e-3, device=device)
        model = model_class(input_dim=64, hidden_dim=256, num_layers=2, 
                           num_classes=text_encoder.num_classes)
        
        # Train
        trainer = Trainer(model, config, blank_idx=0)
        
        def decode_fn(log_probs):
            return greedy_decode(log_probs, text_encoder)
        
        metrics = trainer.train(train_loader, test_loader, decode_fn, model_name)
        rnn_metrics[model_name] = metrics
        trainer.save(f"{model_class.__name__.lower()}_ctc.pth")
    
    return rnn_metrics


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("Speech-to-Text Model Training")
    print("=" * 70)
    
    # Setup
    device = setup_device()
    train_dataset, test_dataset = load_dataset()
    text_encoder = create_decoders()
    
    # Initialize comparator
    comparator = ModelComparator()
    
    # Train models
    print("\n" + "=" * 70)
    print("Training Models")
    print("=" * 70)
    
    # MLP + MFCC
    print("\n[1/3] Training MLP + MFCC...")
    feature_extractor_mfcc = FeatureExtractor(feature_type='mfcc')
    train_loader_mfcc, test_loader_mfcc = create_dataloaders(
        train_dataset, test_dataset, feature_extractor_mfcc, 
        text_encoder, batch_size=16
    )
    
    config = TrainingConfig(num_epochs=2, learning_rate=1e-3, device=device)
    model_mlp = MLP(in_dim=128, hidden=256, num_classes=text_encoder.num_classes)
    trainer_mlp = Trainer(model_mlp, config, blank_idx=0)
    
    def decode_fn(log_probs):
        return greedy_decode(log_probs, text_encoder)
    
    metrics_mlp = trainer_mlp.train(train_loader_mfcc, test_loader_mfcc, 
                                    decode_fn, "MLP + MFCC")
    trainer_mlp.save("mlp_ctc.pth")
    comparator.add_result("MLP + MFCC", metrics_mlp)
    
    # CNN + MelSpectrogram
    print("\n[2/3] Training CNN + MelSpectrogram...")
    feature_extractor_mel = FeatureExtractor(feature_type='mel')
    train_loader_mel, test_loader_mel = create_dataloaders(
        train_dataset, test_dataset, feature_extractor_mel, 
        text_encoder, batch_size=16
    )
    
    model_cnn = CNN(n_mels=64, num_classes=text_encoder.num_classes)
    trainer_cnn = CNNTrainer(model_cnn, config, blank_idx=0)
    metrics_cnn = trainer_cnn.train(train_loader_mel, test_loader_mel, 
                                    decode_fn, "CNN + MelSpectrogram")
    trainer_cnn.save("cnn_mel_ctc.pth")
    comparator.add_result("CNN + MelSpectrogram", metrics_cnn)
    
    # RNN models
    print("\n[3/3] Training RNN Variants...")
    rnn_models = [
        (RNN_LSTM, "LSTM + MelSpectrogram"),
        (RNN_GRU, "GRU + MelSpectrogram"),
        (RNN_BiLSTM, "BiLSTM + MelSpectrogram"),
    ]
    
    for i, (model_class, model_name) in enumerate(rnn_models, 1):
        print(f"\n[3.{i}/3] Training {model_name}...")
        model_rnn = model_class(input_dim=64, hidden_dim=256, num_layers=2,
                               num_classes=text_encoder.num_classes)
        trainer_rnn = Trainer(model_rnn, config, blank_idx=0)
        metrics_rnn = trainer_rnn.train(train_loader_mel, test_loader_mel, 
                                       decode_fn, model_name)
        trainer_rnn.save(f"{model_class.__name__.lower()}_ctc.pth")
        comparator.add_result(model_name, metrics_rnn)
    
    # Print comparison
    comparator.print_comparison()
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
