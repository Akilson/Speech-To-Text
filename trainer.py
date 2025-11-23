"""
Training utilities for speech-to-text models.
Includes training loops, evaluation, and model management.
"""

import time
import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class TrainingConfig:
    """Configuration for training."""
    def __init__(self, 
                 num_epochs=2,
                 learning_rate=1e-3,
                 batch_size=16,
                 grad_clip=1.0,
                 device=None):
        """
        Initialize training configuration.
        
        Args:
            num_epochs (int): Number of training epochs
            learning_rate (float): Learning rate for optimizer
            batch_size (int): Batch size
            grad_clip (float): Gradient clipping value
            device (torch.device): Device to train on
        """
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    """Unified trainer for all model types."""
    
    def __init__(self, model, config, blank_idx=0):
        """
        Initialize trainer.
        
        Args:
            model (nn.Module): Model to train
            config (TrainingConfig): Training configuration
            blank_idx (int): Index of CTC blank token
        """
        self.model = model.to(config.device)
        self.config = config
        self.blank_idx = blank_idx
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
        
        self.metrics = {
            'epochs': [],
            'train_losses': [],
            'train_times': [],
            'param_count': sum(p.numel() for p in model.parameters())
        }
    
    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
            epoch (int): Current epoch number
            
        Returns:
            dict: Epoch metrics (loss, time, batch_count)
        """
        self.model.train()
        t0 = time.time()
        running_loss = 0.0
        batch_count = 0
        
        for batch_idx, (feats_padded, feat_lens, targets_concat, target_lens) in enumerate(train_loader):
            feats_padded = feats_padded.to(self.config.device).float()
            targets_concat = targets_concat.to(self.config.device)
            
            # Adjust feature lengths for CNN models (if applicable)
            feat_lens_adjusted = self._adjust_feat_lens(feat_lens)
            
            self.optimizer.zero_grad()
            log_probs = self.model(feats_padded)
            loss = self.ctc_loss(log_probs, targets_concat, feat_lens_adjusted, target_lens)
            loss.backward()
            
            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            self.optimizer.step()
            running_loss += loss.item()
            batch_count += 1
            
            # Log progress
            if (batch_idx + 1) % 50 == 0:
                avg_loss = running_loss / 50
                print(f"  Epoch {epoch+1} Batch {batch_idx+1}: loss = {avg_loss:.4f}")
                running_loss = 0.0
        
        epoch_time = time.time() - t0
        
        return {
            'time': epoch_time,
            'avg_loss': running_loss / batch_count if batch_count > 0 else 0,
            'batch_count': batch_count
        }
    
    def evaluate(self, test_loader, decode_fn=None):
        """
        Evaluate model on test set.
        
        Args:
            test_loader (DataLoader): Test data loader
            decode_fn (callable): Decoding function for predictions
            
        Returns:
            list: Sample predictions (if decode_fn provided)
        """
        self.model.eval()
        samples = []
        
        with torch.no_grad():
            for batch_idx, (feats_padded, feat_lens, targets_concat, target_lens) in enumerate(test_loader):
                feats_padded = feats_padded.to(self.config.device).float()
                log_probs = self.model(feats_padded)
                
                if decode_fn:
                    preds = decode_fn(log_probs)
                    
                    # Rebuild references
                    refs = []
                    idx = 0
                    for L in target_lens:
                        seq = targets_concat[idx: idx + L]
                        refs.append(seq)
                        idx += L
                    
                    # Store samples
                    if batch_idx == 0:  # First batch only
                        for ref, pred in zip(refs[:3], preds[:3]):
                            samples.append({'reference': ref, 'prediction': pred})
                
                # Only first batch for evaluation
                break
        
        return samples
    
    def train(self, train_loader, test_loader=None, decode_fn=None, model_name="model"):
        """
        Full training loop.
        
        Args:
            train_loader (DataLoader): Training data loader
            test_loader (DataLoader): Test data loader (optional)
            decode_fn (callable): Decoding function (optional)
            model_name (str): Name for logging
            
        Returns:
            dict: Training metrics
        """
        print("=" * 70)
        print(f"Training: {model_name}")
        print("=" * 70)
        print(f"Parameters: {self.metrics['param_count']:,}")
        print(f"Device: {self.config.device}")
        
        for epoch in range(self.config.num_epochs):
            # Train
            epoch_metrics = self.train_epoch(train_loader, epoch)
            self.metrics['epochs'].append(epoch + 1)
            self.metrics['train_losses'].append(epoch_metrics['avg_loss'])
            self.metrics['train_times'].append(epoch_metrics['time'])
            
            print(f"Epoch {epoch+1} - Time: {epoch_metrics['time']:.1f}s, "
                  f"Avg Loss: {epoch_metrics['avg_loss']:.4f}")
            
            # Evaluate
            if test_loader is not None:
                print(f"Evaluating on test set...")
                samples = self.evaluate(test_loader, decode_fn)
                
                print(f"\n--- {model_name} Predictions (Epoch {epoch+1}) ---")
                for sample in samples:
                    print(f"REF:  {sample['reference']}")
                    print(f"PRED: {sample['prediction']}")
                    print()
        
        return self.metrics
    
    def save(self, path):
        """Save model state."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model state."""
        self.model.load_state_dict(torch.load(path, map_location=self.config.device))
        print(f"Model loaded from {path}")
    
    def _adjust_feat_lens(self, feat_lens):
        """
        Adjust feature lengths for models with pooling (e.g., CNN).
        Override in subclasses if needed.
        
        Args:
            feat_lens (torch.Tensor): Original feature lengths
            
        Returns:
            torch.Tensor: Adjusted feature lengths
        """
        return feat_lens


class CNNTrainer(Trainer):
    """Trainer specialized for CNN models with pooling."""
    
    def _adjust_feat_lens(self, feat_lens):
        """
        Adjust feature lengths for CNN pooling (2x2 pooling twice = /4).
        
        Args:
            feat_lens (torch.Tensor): Original feature lengths
            
        Returns:
            torch.Tensor: Adjusted feature lengths
        """
        return (feat_lens // 4).clamp(min=1)


class ModelComparator:
    """Compare multiple trained models."""
    
    def __init__(self):
        """Initialize comparator."""
        self.results = {}
    
    def add_result(self, model_name, metrics):
        """
        Add training results.
        
        Args:
            model_name (str): Name of model
            metrics (dict): Training metrics
        """
        self.results[model_name] = metrics
    
    def print_comparison(self):
        """Print comparison table."""
        if not self.results:
            print("No results to compare")
            return
        
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        
        # Training time comparison
        print("\nTRAINING TIME (seconds):")
        print("-" * 80)
        for model_name in sorted(self.results.keys()):
            metrics = self.results[model_name]
            total_time = sum(metrics['train_times'])
            avg_time = total_time / len(metrics['train_times']) if metrics['train_times'] else 0
            print(f"{model_name:20s}: Total={total_time:7.1f}s | Avg/epoch={avg_time:6.1f}s")
        
        # Parameter count comparison
        print("\nPARAMETER COUNT:")
        print("-" * 80)
        for model_name in sorted(self.results.keys(), 
                                 key=lambda x: self.results[x]['param_count']):
            metrics = self.results[model_name]
            print(f"{model_name:20s}: {metrics['param_count']:,} parameters")
        
        # Loss convergence
        print("\nLOSS CONVERGENCE:")
        print("-" * 80)
        for model_name in sorted(self.results.keys()):
            metrics = self.results[model_name]
            losses = metrics['train_losses']
            if len(losses) >= 2:
                improvement = ((losses[0] - losses[-1]) / losses[0]) * 100
                print(f"{model_name:20s}: Initial={losses[0]:.4f}, "
                      f"Final={losses[-1]:.4f}, Improvement={improvement:6.1f}%")
            else:
                print(f"{model_name:20s}: Only {len(losses)} epoch(s)")
        
        print("=" * 80)
