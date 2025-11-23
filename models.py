"""
Speech-to-Text Models
Implementations of MLP, CNN, LSTM, GRU, and BiLSTM architectures for CTC-based STT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multilayer Perceptron for speech recognition.
    Simple baseline model without temporal modeling.
    
    Args:
        in_dim (int): Input dimension (e.g., 128 for MFCC)
        hidden (int): Hidden layer dimension
        num_classes (int): Number of output classes
    """
    def __init__(self, in_dim=128, hidden=256, num_classes=44):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input of shape (N, T, D)
                N: batch size, T: time steps, D: input dimension
        
        Returns:
            torch.Tensor: Log probabilities of shape (T, N, num_classes)
        """
        N, T, D = x.size()
        x = x.view(N * T, D)
        h = F.relu(self.fc1(x))
        logits = self.fc2(h).view(T, N, -1)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs


class CNN(nn.Module):
    """
    Convolutional Neural Network for speech recognition.
    Uses 2D convolutions on MelSpectrogram features.
    
    Args:
        n_mels (int): Number of mel-frequency bins
        num_classes (int): Number of output classes
    """
    def __init__(self, n_mels=64, num_classes=44):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input of shape (N, T, n_mels)
                N: batch size, T: time steps, n_mels: mel-frequency bins
        
        Returns:
            torch.Tensor: Log probabilities of shape (T, N, num_classes)
        """
        N, T, n_mels = x.size()
        x = x.unsqueeze(1)  # (N, 1, T, n_mels) - add channel dimension
        
        # Conv block 1
        x = F.relu(self.conv1(x))  # (N, 32, T, n_mels)
        x = self.pool(x)           # (N, 32, T//2, n_mels//2)
        
        # Conv block 2
        x = F.relu(self.conv2(x))  # (N, 64, T//2, n_mels//2)
        x = self.pool(x)           # (N, 64, T//4, n_mels//4)
        
        # Reshape for fully connected layers
        N, C, T_out, F_out = x.size()
        x = x.permute(0, 2, 1, 3)  # (N, T//4, 64, n_mels//4)
        x = x.contiguous().view(N, T_out, C * F_out)  # (N, T//4, 1024)
        
        # FC layers
        x = x.view(N * T_out, -1)  # (N*T//4, 1024)
        h = F.relu(self.fc1(x))    # (N*T//4, 256)
        h = self.dropout(h)
        logits = self.fc2(h)       # (N*T//4, num_classes)
        
        # Reshape for CTC loss
        logits = logits.view(T_out, N, -1)  # (T//4, N, num_classes)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs


class RNN_LSTM(nn.Module):
    """
    LSTM-based model for speech recognition.
    Captures long-range dependencies in forward direction.
    
    Args:
        input_dim (int): Input dimension (e.g., 64 for MelSpectrogram)
        hidden_dim (int): Hidden state dimension
        num_layers (int): Number of LSTM layers
        num_classes (int): Number of output classes
    """
    def __init__(self, input_dim=64, hidden_dim=256, num_layers=2, num_classes=44):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input of shape (N, T, input_dim)
        
        Returns:
            torch.Tensor: Log probabilities of shape (T, N, num_classes)
        """
        lstm_out, _ = self.lstm(x)  # (N, T, hidden_dim)
        N, T, H = lstm_out.size()
        logits = self.fc(lstm_out.reshape(N * T, H)).reshape(T, N, -1)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs


class RNN_GRU(nn.Module):
    """
    GRU-based model for speech recognition.
    Lighter alternative to LSTM with fewer parameters.
    
    Args:
        input_dim (int): Input dimension (e.g., 64 for MelSpectrogram)
        hidden_dim (int): Hidden state dimension
        num_layers (int): Number of GRU layers
        num_classes (int): Number of output classes
    """
    def __init__(self, input_dim=64, hidden_dim=256, num_layers=2, num_classes=44):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input of shape (N, T, input_dim)
        
        Returns:
            torch.Tensor: Log probabilities of shape (T, N, num_classes)
        """
        gru_out, _ = self.gru(x)  # (N, T, hidden_dim)
        N, T, H = gru_out.size()
        logits = self.fc(gru_out.reshape(N * T, H)).reshape(T, N, -1)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs


class RNN_BiLSTM(nn.Module):
    """
    Bidirectional LSTM for speech recognition.
    Processes audio in both forward and backward directions.
    Best accuracy for offline STT tasks.
    
    Args:
        input_dim (int): Input dimension (e.g., 64 for MelSpectrogram)
        hidden_dim (int): Hidden state dimension
        num_layers (int): Number of BiLSTM layers
        num_classes (int): Number of output classes
    """
    def __init__(self, input_dim=64, hidden_dim=256, num_layers=2, num_classes=44):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=0.3, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input of shape (N, T, input_dim)
        
        Returns:
            torch.Tensor: Log probabilities of shape (T, N, num_classes)
        """
        bilstm_out, _ = self.bilstm(x)  # (N, T, 2*hidden_dim)
        N, T, H = bilstm_out.size()
        logits = self.fc(bilstm_out.reshape(N * T, H)).reshape(T, N, -1)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
