"""
Multi-Regional Model Module for Paragliding Flyability Prediction.

This module implements a neural network with regional adaptation using attention mechanisms.
The model is designed to handle geographical variations in flyability patterns across
different regions.

Key concepts:
- Regional Embeddings: Learn vector representations for each geographical region
- Regional Attention: Adapt shared features to region-specific patterns
- Geographical Clustering: Group nearby cells into regions using K-means
- Confidence Weighting: Weight samples by label confidence during training

Example usage in notebook:
    from paraglideml.multiregional import *

    config = MultiRegionalConfig(num_regions=3, regional_embedding_dim=32)
    train_df, test_df, feature_names, region_mapping = load_and_prepare_data(config)

    model = MultiRegionalModel(len(feature_names), config.num_regions, config.regional_embedding_dim)
    trainer = MultiRegionalTrainer(model, config, criterion, optimizer)

    trainer.fit(train_loader, val_loader)
    plot_training_history(trainer.history)
"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# =============================================================================
# Path Utilities
# =============================================================================


def get_project_root() -> Path:
    """
    Get the project root directory.

    Looks for the project root by finding pyproject.toml or setup.py,
    or by navigating from the package location.

    Returns:
        Path to project root directory
    """
    # Try to find project root from current working directory
    cwd = Path.cwd()

    # Check if we're in notebooks directory
    if cwd.name == "notebooks":
        return cwd.parent

    # Check if we're already at project root (has pyproject.toml or data/ directory)
    if (cwd / "pyproject.toml").exists() or (cwd / "data").exists():
        return cwd

    # Try from package location
    try:
        package_dir = Path(__file__).resolve().parent.parent.parent
        if (package_dir / "pyproject.toml").exists() or (package_dir / "data").exists():
            return package_dir
    except (OSError, RuntimeError):
        pass

    # Fallback: assume we're in project root or notebooks/
    if (cwd / "data").exists():
        return cwd
    return cwd.parent


def resolve_path(path: str) -> str:
    """
    Resolve a path relative to the project root.

    If the path is absolute, return it as-is.
    If the path is relative, make it relative to the project root.

    Args:
        path: File path (absolute or relative)

    Returns:
        Resolved absolute path as string
    """
    p = Path(path)
    if p.is_absolute():
        return path
    return str(get_project_root() / path)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class MultiRegionalConfig:
    """Configuration for multi-regional model training."""

    num_regions: int = 3
    regional_embedding_dim: int = 32
    learning_rate: float = 0.001
    dropout_rate: float = 0.3
    batch_size: int = 32
    epochs: int = 150
    patience: int = 25
    val_split: float = 0.2
    hysteresis_margin: float = 0.15
    use_confidence_weighting: bool = True
    random_seed: int = 42
    data_path: str = "data/processed/multicell_dataset.csv"
    experiments_dir: str = "models/experiments"

    def __post_init__(self):
        """Set random seeds and resolve paths after initialization."""
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        # Resolve relative paths
        self.data_path = resolve_path(self.data_path)
        self.experiments_dir = resolve_path(self.experiments_dir)


# =============================================================================
# Model Architecture
# =============================================================================


class RegionalEmbedding(nn.Module):
    """
    Converts region identifiers to vector representations.

    Each region gets a learnable embedding vector that captures
    region-specific characteristics (terrain, typical weather patterns, etc.).

    Args:
        num_regions: Number of geographical regions
        embedding_dim: Dimension of the embedding vector
    """

    def __init__(self, num_regions: int, embedding_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(num_regions, embedding_dim)

    def forward(self, region_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            region_ids: [batch_size] tensor of region indices

        Returns:
            [batch_size, embedding_dim] tensor of region embeddings
        """
        return self.embedding(region_ids)


class RegionalAttention(nn.Module):
    """
    Uses multi-head attention to adapt shared features for specific regions.

    The attention mechanism allows the model to learn which features
    should be emphasized or de-emphasized for each region.

    Args:
        hidden_dim: Dimension of the feature space
        num_regions: Number of geographical regions
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout rate for attention (default: 0.1)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_regions: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout)
        self.region_embedding = RegionalEmbedding(num_regions, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: torch.Tensor, region_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, hidden_dim] shared features
            region_ids: [batch_size] region indices

        Returns:
            [batch_size, hidden_dim] regionally-adapted features
        """
        # Get region embeddings
        region_emb = self.region_embedding(region_ids)  # [batch_size, hidden_dim]

        # Add dimension for multihead attention (seq_len=1)
        features = features.unsqueeze(0)  # [1, batch_size, hidden_dim]
        region_emb = region_emb.unsqueeze(0)  # [1, batch_size, hidden_dim]

        # Attention: features as query, region_emb as key and value
        attended_features, _ = self.attention(features, region_emb, region_emb)

        # Remove seq dimension
        attended_features = attended_features.squeeze(0)  # [batch_size, hidden_dim]

        # Residual connection + layer norm
        attended_features = self.layer_norm(features.squeeze(0) + attended_features)

        return attended_features


class MultiRegionalModel(nn.Module):
    """
    Neural network with regional adaptation for paragliding flyability prediction.

    Architecture:
        1. Backbone: Shared feature extractor (128 -> 64 -> 32)
        2. Regional Attention: Adapts features to specific regions
        3. Classifier: Binary classification (flyable / not flyable)

    Args:
        input_dim: Number of input features
        num_regions: Number of geographical regions
        embedding_dim: Dimension of regional embeddings
        dropout_rate: Dropout rate for backbone layers
    """

    def __init__(
        self,
        input_dim: int,
        num_regions: int,
        embedding_dim: int = 32,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        # Shared backbone (extracts common patterns)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Regional adaptation
        self.regional_attention = RegionalAttention(32, num_regions)

        # Binary classifier
        self.classifier = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor, region_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim] input features
            region_ids: [batch_size] region indices

        Returns:
            [batch_size, 1] logits (use sigmoid for probabilities)
        """
        # Extract shared features
        features = self.backbone(x)  # [batch_size, 32]

        # Region-specific adaptation
        attended_features = self.regional_attention(features, region_ids)  # [batch_size, 32]

        # Classification
        output = self.classifier(attended_features)  # [batch_size, 1]

        return output

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Dataset
# =============================================================================


class RegionalFlyableDataset(Dataset):
    """
    PyTorch Dataset for regional flyability prediction.

    Args:
        X: Feature array [n_samples, n_features]
        y: Binary labels [n_samples]
        region_ids: Region indices [n_samples]
        confidence_weights: Optional confidence weights for each sample [n_samples]
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        region_ids: np.ndarray,
        confidence_weights: Optional[np.ndarray] = None,
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self.region_ids = torch.tensor(region_ids, dtype=torch.long)

        if confidence_weights is not None:
            self.confidence_weights = torch.tensor(
                confidence_weights, dtype=torch.float32
            ).unsqueeze(1)
        else:
            self.confidence_weights = torch.ones_like(self.y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (features, label, region_id, confidence_weight)."""
        return (
            self.X[idx],
            self.y[idx],
            self.region_ids[idx],
            self.confidence_weights[idx],
        )


# =============================================================================
# Data Processing
# =============================================================================


def cluster_regions(
    df: pd.DataFrame, n_clusters: int = 5, random_state: int = 42
) -> Dict[str, int]:
    """
    Cluster geographical cells into regions using K-means.

    Args:
        df: DataFrame with 'cell_id' column (format: "lat_lon")
        n_clusters: Number of regions to create
        random_state: Random seed for reproducibility

    Returns:
        Dictionary mapping cell_id -> region_id
    """
    # Extract cell centers
    coords = []
    cell_ids = []

    for cell_id in df["cell_id"].unique():
        lat, lon = map(int, cell_id.split("_"))
        center_lat = lat + 0.5
        center_lon = lon + 0.5
        coords.append([center_lat, center_lon])
        cell_ids.append(cell_id)

    coords = np.array(coords)

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(coords)

    # Create mapping
    region_mapping = dict(zip(cell_ids, clusters))

    return region_mapping


def load_and_prepare_data(
    config: MultiRegionalConfig, data_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, int]]:
    """
    Load and prepare data for multi-regional training.

    Args:
        config: Configuration object
        data_path: Path to CSV file (overrides config.data_path if provided)

    Returns:
        train_df: Training DataFrame
        test_df: Test DataFrame
        feature_names: List of feature column names
        region_mapping: Dictionary mapping cell_id -> region_id
    """
    if data_path is None:
        data_path = config.data_path

    # Load data
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    # Train/test split by year
    train_df = df[df["year"] < 2025].copy()
    test_df = df[df["year"] == 2025].copy()

    # Geographical clustering
    print(f"Performing geographical clustering into {config.num_regions} regions...")
    region_mapping = cluster_regions(
        train_df, n_clusters=config.num_regions, random_state=config.random_seed
    )

    # Add region IDs
    train_df["region_id"] = train_df["cell_id"].map(region_mapping)
    test_df["region_id"] = test_df["cell_id"].map(region_mapping)

    # Validate mapping
    if train_df["region_id"].isna().any() or test_df["region_id"].isna().any():
        missing_cells = set(test_df["cell_id"].unique()) - set(region_mapping.keys())
        raise ValueError(f"Some cells don't have region mapping! Missing: {missing_cells}")

    # Print statistics
    print("Loaded multicell dataset:")
    print(
        f"  Train: {len(train_df)} samples ({train_df['cell_id'].nunique()} cells, {train_df['region_id'].nunique()} regions)"
    )
    print(
        f"  Test:  {len(test_df)} samples ({test_df['cell_id'].nunique()} cells, {test_df['region_id'].nunique()} regions)"
    )
    print(f"  Train balance: {train_df['is_flyable'].mean():.1%} flyable")
    print(f"  Test balance:  {test_df['is_flyable'].mean():.1%} flyable")

    # Extract feature names.
    # NOTE: is_weekend and day_of_year are dropped on purpose. They are calendar
    # artifacts of the labeling process (more pilots fly on weekends), not weather,
    # and leaked the "weekend -> flyable" shortcut into the model. is_weekend stays
    # available for the label-confidence logic only, never as a predictor.
    drop_cols = [
        "date",
        "year",
        "flight_count",
        "is_flyable",
        "cell_id",
        "cell_lat",
        "cell_lon",
        "label_confidence",
        "region_id",
        "is_weekend",
        "day_of_year",
    ]
    feature_names = [c for c in df.columns if c not in drop_cols]
    print(f"  Features: {len(feature_names)}")

    return train_df, test_df, feature_names, region_mapping


def create_data_loaders(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: List[str],
    config: MultiRegionalConfig,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    StandardScaler,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Create PyTorch DataLoaders with scaling and a TEMPORAL train/val split.

    The validation set is the most recent year present in ``train_df`` (e.g. 2024),
    held out chronologically instead of sampled at random. A random split leaks
    near-duplicate samples into validation (the same day spans ~45 cells and
    adjacent days are strongly autocorrelated), which makes early stopping and
    threshold selection over-optimistic. A temporal split mirrors the real task:
    choose hyperparameters/threshold on an unseen future year, then report on the
    held-out test year. The scaler is fit on the training years only.

    Args:
        train_df: Training DataFrame (must contain a 'year' column)
        test_df: Test DataFrame
        feature_names: List of feature column names
        config: Configuration object
        scaler: Optional fitted scaler (re-used instead of re-fitting if provided)

    Returns:
        train_loader, val_loader, test_loader, scaler,
        y_train (training labels, for loss weighting),
        X_val, y_val, region_val,
        X_test, y_test, region_test
    """
    # Temporal validation split: hold out the most recent training year.
    val_year = int(train_df["year"].max())
    is_val = train_df["year"] == val_year
    fit_df = train_df[~is_val]
    val_df = train_df[is_val]

    # Extract features
    X_fit_raw = fit_df[feature_names].values.astype(np.float32)
    y_train = fit_df["is_flyable"].values.astype(np.float32)
    X_val_raw = val_df[feature_names].values.astype(np.float32)
    y_val = val_df["is_flyable"].values.astype(np.float32)
    X_test_raw = test_df[feature_names].values.astype(np.float32)
    y_test = test_df["is_flyable"].values.astype(np.float32)

    # Extract region IDs and confidence weights
    region_train = fit_df["region_id"].values.astype(np.int64)
    region_val = val_df["region_id"].values.astype(np.int64)
    region_test = test_df["region_id"].values.astype(np.int64)
    conf_train = fit_df["label_confidence"].values.astype(np.float32)
    conf_val = val_df["label_confidence"].values.astype(np.float32)
    conf_test = test_df["label_confidence"].values.astype(np.float32)

    # Scale features (fit on training years only, never on val/test)
    if scaler is None:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_fit_raw)
    else:
        X_train = scaler.transform(X_fit_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # Create DataLoaders
    train_loader = DataLoader(
        RegionalFlyableDataset(X_train, y_train, region_train, conf_train),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        RegionalFlyableDataset(X_val, y_val, region_val, conf_val),
        batch_size=config.batch_size,
    )
    test_loader = DataLoader(
        RegionalFlyableDataset(X_test, y_test, region_test, conf_test),
        batch_size=config.batch_size,
    )

    print("Data splits (temporal):")
    print(f"  Train: {len(X_train)} samples (years < {val_year})")
    print(f"  Val:   {len(X_val)} samples (year == {val_year})")
    print(f"  Test:  {len(X_test)} samples")

    return (
        train_loader,
        val_loader,
        test_loader,
        scaler,
        y_train,
        X_val,
        y_val,
        region_val,
        X_test,
        y_test,
        region_test,
    )


def create_weighted_bce_loss(
    y_train: np.ndarray, pos_weight: Optional[float] = None, use_confidence: bool = True
) -> nn.BCEWithLogitsLoss:
    """
    Create weighted BCE loss for imbalanced binary classification.

    Args:
        y_train: Training labels
        pos_weight: Positive class weight (auto-calculated if None)
        use_confidence: If True, use reduction='none' for confidence weighting

    Returns:
        BCEWithLogitsLoss with configured weights
    """
    if pos_weight is None:
        num_pos = int(y_train.sum())
        num_neg = len(y_train) - num_pos
        pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

    reduction = "none" if use_confidence else "mean"

    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]), reduction=reduction)


# =============================================================================
# Trainer
# =============================================================================


class MultiRegionalTrainer:
    """
    Trainer class for multi-regional model with history tracking.

    Handles training loop, early stopping, and maintains training history
    for visualization.

    Args:
        model: MultiRegionalModel instance
        config: MultiRegionalConfig instance
        criterion: Loss function
        optimizer: Optimizer instance
    """

    def __init__(
        self,
        model: MultiRegionalModel,
        config: MultiRegionalConfig,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ):
        self.model = model
        self.config = config
        self.criterion = criterion
        self.optimizer = optimizer
        self.history = {"train_loss": [], "val_loss": []}
        self.best_model_state = None
        self.best_val_loss = float("inf")
        self.epochs_trained = 0

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch. Returns average training loss."""
        self.model.train()
        train_loss = 0

        for X_batch, y_batch, region_batch, conf_batch in train_loader:
            self.optimizer.zero_grad()

            # Forward pass
            y_pred = self.model(X_batch, region_batch)

            # Compute loss. Normalize the confidence-weighted loss by the sum of
            # weights (a proper weighted mean), not by sample count — otherwise
            # batches with many low-confidence samples are silently down-scaled.
            if self.config.use_confidence_weighting:
                loss_per_sample = self.criterion(y_pred, y_batch)
                loss = (loss_per_sample * conf_batch).sum() / conf_batch.sum().clamp(min=1e-8)
            else:
                loss = self.criterion(y_pred, y_batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        return train_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> float:
        """Validate on validation set. Returns average validation loss."""
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for X_batch, y_batch, region_batch, conf_batch in val_loader:
                y_pred = self.model(X_batch, region_batch)

                if self.config.use_confidence_weighting:
                    loss_per_sample = self.criterion(y_pred, y_batch)
                    loss = (loss_per_sample * conf_batch).sum() / conf_batch.sum().clamp(min=1e-8)
                else:
                    loss = self.criterion(y_pred, y_batch)

                val_loss += loss.item()

        return val_loss / len(val_loader)

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader, verbose: bool = True
    ) -> "MultiRegionalTrainer":
        """
        Train the model with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            verbose: Whether to print progress

        Returns:
            self (for method chaining)
        """
        epochs_no_improve = 0

        # Halve the LR when validation loss plateaus (well before early-stop fires).
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=8
        )

        if verbose:
            print(
                f"Training for max {self.config.epochs} epochs (patience={self.config.patience})..."
            )
            if self.config.use_confidence_weighting:
                print("  Using confidence-weighted loss")

        for epoch in range(self.config.epochs):
            # Train and validate
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            scheduler.step(val_loss)

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"  Epoch {epoch+1}/{self.config.epochs}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                )

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                epochs_no_improve = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.config.patience:
                if verbose:
                    print(
                        f"  Early stopping at epoch {epoch+1}. Best val_loss: {self.best_val_loss:.4f}"
                    )
                break

            self.epochs_trained = epoch + 1

        # Load best model
        self.model.load_state_dict(self.best_model_state)

        if verbose:
            print(
                f"  Training complete. Epochs: {self.epochs_trained}, Best val_loss: {self.best_val_loss:.4f}"
            )

        return self

    def predict(self, X: np.ndarray, region_ids: np.ndarray) -> np.ndarray:
        """
        Make predictions on data.

        Args:
            X: Feature array [n_samples, n_features]
            region_ids: Region indices [n_samples]

        Returns:
            Probabilities [n_samples]
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(region_ids, dtype=torch.long),
            )
            probs = torch.sigmoid(logits).numpy()
        return probs.flatten()


# =============================================================================
# Evaluation
# =============================================================================


def find_optimal_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    metric: str = "macro_f1",
    threshold_range: Tuple[float, float] = (0.1, 0.9),
    step: float = 0.01,
) -> Tuple[float, float, Dict[str, List[float]]]:
    """
    Find optimal threshold for binary classification.

    Args:
        y_true: True labels
        probs: Predicted probabilities
        metric: Metric to optimize ('macro_f1', 'f1', 'accuracy')
        threshold_range: (min, max) threshold range
        step: Threshold step size

    Returns:
        (best_threshold, best_score, history_dict)
        history_dict contains f1_not_flyable, f1_flyable, macro_f1, thresholds
    """
    thresholds = np.arange(threshold_range[0], threshold_range[1], step)
    f1_not_flyable = []
    f1_flyable = []
    macro_f1_scores = []

    for thresh in thresholds:
        y_pred = (probs > thresh).astype(float)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
        macro_f1 = (f1[0] + f1[1]) / 2

        f1_not_flyable.append(f1[0])
        f1_flyable.append(f1[1])
        macro_f1_scores.append(macro_f1)

    # Select best threshold based on metric
    if metric == "macro_f1":
        scores = macro_f1_scores
    elif metric == "f1":
        scores = f1_flyable
    else:  # accuracy
        from sklearn.metrics import accuracy_score

        scores = [accuracy_score(y_true, (probs > t).astype(float)) for t in thresholds]

    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]
    best_score = scores[best_idx]

    history = {
        "thresholds": thresholds,
        "f1_not_flyable": f1_not_flyable,
        "f1_flyable": f1_flyable,
        "macro_f1": macro_f1_scores,
    }

    return best_threshold, best_score, history


def evaluate_by_regions(
    model: nn.Module, test_loader: DataLoader, region_names: Dict[int, str]
) -> Dict[str, Dict[str, List]]:
    """
    Evaluate model performance by region.

    Args:
        model: Trained model
        test_loader: Test data loader
        region_names: Mapping {region_id: region_name}

    Returns:
        Dictionary with results per region
    """
    model.eval()
    region_results = {
        region: {"y_true": [], "y_pred": [], "probs": []} for region in region_names.values()
    }

    with torch.no_grad():
        for X_batch, y_batch, region_batch, _ in test_loader:
            probs = torch.sigmoid(model(X_batch, region_batch))
            preds = (probs > 0.5).float()

            for i, region_id in enumerate(region_batch):
                region_name = region_names[region_id.item()]
                region_results[region_name]["y_true"].append(y_batch[i].item())
                region_results[region_name]["y_pred"].append(preds[i].item())
                region_results[region_name]["probs"].append(probs[i].item())

    return region_results


def evaluate_per_cell(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate performance metrics per cell.

    Args:
        df_results: DataFrame with prediction results

    Returns:
        DataFrame with per-cell statistics
    """
    cell_stats = []

    for cell_id in sorted(df_results["cell_id"].unique()):
        cell_data = df_results[df_results["cell_id"] == cell_id]
        cell_y_true = cell_data["is_flyable"].values
        cell_y_pred = cell_data["pred"].values

        # Skip cells with no positive examples
        if cell_y_true.sum() == 0:
            continue

        acc = accuracy_score(cell_y_true, cell_y_pred)
        f1 = f1_score(cell_y_true, cell_y_pred, zero_division=0)

        cell_stats.append(
            {
                "cell_id": cell_id,
                "region_id": cell_data["region_id"].iloc[0],
                "samples": len(cell_data),
                "accuracy": acc,
                "f1": f1,
            }
        )

    return pd.DataFrame(cell_stats).sort_values("f1", ascending=False)


# =============================================================================
# Visualization Functions
# =============================================================================


def plot_training_history(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (10, 5),
    title: str = "Training History",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training and validation loss curves.

    Args:
        history: Dictionary with 'train_loss' and 'val_loss' lists
        figsize: Figure size
        title: Plot title
        save_path: If provided, save plot to this path instead of showing
    """
    plt.figure(figsize=figsize)
    plt.plot(history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(history["val_loss"], label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_threshold_analysis(
    y_true: np.ndarray,
    probs: np.ndarray,
    best_threshold: float,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot F1 scores vs threshold.

    Args:
        y_true: True labels
        probs: Predicted probabilities
        best_threshold: Optimal threshold to mark
        figsize: Figure size
        save_path: If provided, save plot to this path instead of showing
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_not_flyable = []
    f1_flyable = []
    macro_f1_scores = []

    for thresh in thresholds:
        y_pred = (probs > thresh).astype(float)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
        macro_f1 = (f1[0] + f1[1]) / 2
        f1_not_flyable.append(f1[0])
        f1_flyable.append(f1[1])
        macro_f1_scores.append(macro_f1)

    plt.figure(figsize=figsize)
    plt.plot(thresholds, f1_not_flyable, label="F1 (Not Flyable)", linewidth=2)
    plt.plot(thresholds, f1_flyable, label="F1 (Flyable)", linewidth=2)
    plt.plot(thresholds, macro_f1_scores, label="Macro F1 (avg)", linewidth=2, linestyle="--")
    plt.axvline(
        best_threshold,
        color="red",
        linestyle=":",
        label=f"Optimal threshold = {best_threshold:.2f}",
    )
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Threshold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    figsize: Tuple[int, int] = (6, 5),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot confusion matrix as heatmap.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        figsize: Figure size
        title: Plot title
        save_path: If provided, save plot to this path instead of showing
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    if title:
        plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_per_region_performance(
    region_results: Dict[str, Dict[str, List]],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot per-region performance comparison.

    Args:
        region_results: Dictionary from evaluate_by_regions
        figsize: Figure size
        save_path: If provided, save plot to this path instead of showing
    """
    regions = []
    f1_scores = []
    accuracies = []
    precisions = []
    recalls = []
    sample_counts = []

    for region_name, results in region_results.items():
        if len(results["y_true"]) == 0:
            continue

        y_true = results["y_true"]
        y_pred = results["y_pred"]

        regions.append(region_name)
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        accuracies.append(accuracy_score(y_true, y_pred))
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        sample_counts.append(len(y_true))

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Bar chart of F1 scores
    x_pos = np.arange(len(regions))
    axes[0].bar(x_pos, f1_scores, alpha=0.8)
    axes[0].set_xlabel("Region")
    axes[0].set_ylabel("F1 Score")
    axes[0].set_title("Per-Region F1 Scores")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(regions, rotation=45, ha="right")
    axes[0].grid(alpha=0.3, axis="y")

    # Add sample count labels
    for i, (f1, count) in enumerate(zip(f1_scores, sample_counts)):
        axes[0].text(i, f1 + 0.01, f"n={count}", ha="center", fontsize=8)

    # Metrics comparison
    x_pos = np.arange(len(regions))
    width = 0.2
    axes[1].bar(x_pos - width * 1.5, accuracies, width, label="Accuracy", alpha=0.8)
    axes[1].bar(x_pos - width * 0.5, precisions, width, label="Precision", alpha=0.8)
    axes[1].bar(x_pos + width * 0.5, recalls, width, label="Recall", alpha=0.8)
    axes[1].bar(x_pos + width * 1.5, f1_scores, width, label="F1", alpha=0.8)
    axes[1].set_xlabel("Region")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Per-Region Metrics Comparison")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(regions, rotation=45, ha="right")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# =============================================================================
# Experiment Saving
# =============================================================================


class ExperimentSaver:
    """
    Save experiment results and artifacts.

    Args:
        exp_dir: Directory to save experiment (created if doesn't exist)
    """

    def __init__(self, exp_dir: str):
        self.exp_dir = exp_dir
        os.makedirs(exp_dir, exist_ok=True)

    def save_model(self, model: nn.Module, filename: str = "model.pth") -> str:
        """Save model state dict."""
        path = os.path.join(self.exp_dir, filename)
        torch.save(model.state_dict(), path)
        return path

    def save_scaler(self, scaler: StandardScaler, filename: str = "scaler.pkl") -> str:
        """
        Persist the fitted feature scaler next to the model.

        Required for inference on new data (e.g. the FlyBeeper forecast pipeline):
        without the exact training-time StandardScaler, raw GFS features are not
        normalized the same way and predictions are meaningless.
        """
        import joblib

        path = os.path.join(self.exp_dir, filename)
        joblib.dump(scaler, path)
        return path

    def save_config(self, config: MultiRegionalConfig, results: Dict[str, Any]) -> str:
        """Save configuration and results as JSON."""
        data = asdict(config)
        data.update(results)
        path = os.path.join(self.exp_dir, "config.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def save_features(self, feature_names: List[str]) -> str:
        """Save feature list."""
        path = os.path.join(self.exp_dir, "features.txt")
        with open(path, "w") as f:
            f.write("\n".join(feature_names))
        return path

    def save_predictions(
        self,
        df_results: pd.DataFrame,
        threshold: float,
        hysteresis_margin: float = 0.15,
    ) -> Dict[str, str]:
        """
        Save predictions with classification breakdown.

        Returns dict of saved file paths.
        """
        saved_paths = {}

        # Add classification
        df_results["conf_matrix"] = "unknown"
        df_results.loc[
            (df_results["is_flyable"] == 1) & (df_results["pred"] == 1), "conf_matrix"
        ] = "TP"
        df_results.loc[
            (df_results["is_flyable"] == 0) & (df_results["pred"] == 0), "conf_matrix"
        ] = "TN"
        df_results.loc[
            (df_results["is_flyable"] == 0) & (df_results["pred"] == 1), "conf_matrix"
        ] = "FP"
        df_results.loc[
            (df_results["is_flyable"] == 1) & (df_results["pred"] == 0), "conf_matrix"
        ] = "FN"

        # Neutral zone detection
        df_results["in_neutral_zone"] = (
            np.abs(df_results["prob"] - threshold) < hysteresis_margin
        ).astype(int)

        # Save all predictions
        all_path = os.path.join(self.exp_dir, "all_predictions.csv")
        df_results.to_csv(all_path, index=False)
        saved_paths["all"] = all_path

        # Save separate files by classification
        for category in ["TP", "TN", "FP", "FN"]:
            subset = df_results[df_results["conf_matrix"] == category]
            if len(subset) > 0:
                path = os.path.join(self.exp_dir, f"{category.lower()}.csv")
                subset.to_csv(path, index=False)
                saved_paths[category.lower()] = path

        # Save neutral zone
        neutral = df_results[df_results["in_neutral_zone"] == 1]
        if len(neutral) > 0:
            path = os.path.join(self.exp_dir, "neutral_zone.csv")
            neutral.to_csv(path, index=False)
            saved_paths["neutral_zone"] = path

        return saved_paths

    def save_report(self, report_text: str, filename: str = "report.txt") -> str:
        """Save text report."""
        path = os.path.join(self.exp_dir, filename)
        with open(path, "w") as f:
            f.write(report_text)
        return path

    def save_region_mapping(self, region_mapping: Dict[str, int]) -> str:
        """Save region mapping."""
        # Convert numpy types to Python types for JSON serialization
        region_mapping_json = {k: int(v) for k, v in region_mapping.items()}
        path = os.path.join(self.exp_dir, "region_mapping.json")
        with open(path, "w") as f:
            json.dump(region_mapping_json, f, indent=2)
        return path

    def save_per_cell_stats(self, cell_stats: pd.DataFrame) -> str:
        """Save per-cell statistics."""
        path = os.path.join(self.exp_dir, "per_cell_stats.csv")
        cell_stats.to_csv(path, index=False)
        return path


def get_next_experiment_dir(experiments_dir: str = "models/experiments") -> str:
    """
    Get next experiment directory path.

    Args:
        experiments_dir: Base experiments directory (relative or absolute)

    Returns:
        Path to new experiment directory (e.g., "models/experiments/exp_001")
    """
    # Resolve path relative to project root
    experiments_dir = resolve_path(experiments_dir)
    os.makedirs(experiments_dir, exist_ok=True)
    existing_exps = [d for d in os.listdir(experiments_dir) if d.startswith("exp_")]
    next_id = len(existing_exps) + 1
    exp_dir = os.path.join(experiments_dir, f"exp_{next_id:03d}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir
