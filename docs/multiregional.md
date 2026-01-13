# Multi-Regional Model for Flyable Conditions Prediction

## Overview

Multi-Regional Model is an advanced neural network architecture designed to handle geographical variations in flyable conditions prediction across different regions (Alps, Apennines, etc.) with a single unified model.

## Architecture

### Core Components

1. **Shared Backbone**: Common feature extractor for all regions
2. **Regional Embedding**: Converts region IDs to vector representations
3. **Regional Attention**: Adapts shared features to specific regions using attention mechanism
4. **Classifier**: Final prediction layer

### Model Flow

```
Input Features → Shared Backbone → Regional Adapters → Output
                    ↓
               Region Embedding + Attention
```

## Key Features

### 🌍 Geographical Adaptation
- Automatic clustering of geographical regions
- Region-specific feature adaptation
- Attention mechanism for regional patterns

### 🎯 Multi-Region Support
- Single model for multiple regions
- Easy scaling to new geographical areas
- Transfer learning between regions

### 📊 Confidence Weighting
- Uses confidence weights from label quality assessment
- Handles uncertain labels effectively
- Better training with noisy data

## Usage

### Training

```bash
python run_multiregional_experiment.py
```

### Configuration

Key parameters in `train_multiregional.py`:

```python
NUM_REGIONS = 3  # Number of geographical regions
REGIONAL_EMBEDDING_DIM = 32  # Dimension for regional embeddings
USE_CONFIDENCE_WEIGHTING = True  # Use label confidence weights
```

## Results

### Expected Improvements

| Region Type | Current F1 | Expected F1 | Improvement |
|-------------|------------|-------------|-------------|
| Strong (Alps) | 0.90-0.96 | 0.92-0.97 | +2-3% |
| Weak (Apennines) | 0.46-0.62 | 0.60-0.75 | +15-20% |

### Evaluation Metrics

The model provides comprehensive evaluation:

1. **Overall Performance**: Macro F1, accuracy, confusion matrix
2. **Per-Region Performance**: F1, precision, recall for each region
3. **Per-Cell Performance**: Detailed metrics for each cell
4. **Confidence Analysis**: Neutral zone detection and analysis

## Files Structure

```
src/paraglideml/
├── train_multiregional.py    # Main training script
├── data/
│   └── dataset_builder_v2.py # Dataset preparation with confidence weights
└── ...

run_multiregional_experiment.py  # Training runner script
MULTIREGIONAL_MODEL_README.md    # This documentation
```

## Technical Details

### Regional Clustering

The model automatically clusters geographical regions using K-means:

```python
def cluster_regions(df, n_clusters=3):
    """Кластеризует ячейки по географическому положению"""
    # K-means кластеризация по координатам центров ячеек
```

### Attention Mechanism

Regional adaptation uses multi-head attention:

```python
class RegionalAttention(nn.Module):
    def forward(self, features, region_ids):
        # Attention: features как query, region_emb как key и value
        attended_features, _ = self.attention(features, region_emb, region_emb)
```

### Dataset Structure

```python
class RegionalFlyableDataset(Dataset):
    def __init__(self, X, y, region_ids, confidence_weights=None):
        # Includes region IDs for geographical adaptation
```

## Comparison with Previous Models

### vs. train_v4.py (Baseline)

| Feature | train_v4.py | Multi-Regional |
|---------|-------------|----------------|
| Geographical adaptation | ❌ | ✅ |
| Single model for all regions | ✅ | ✅ |
| Regional attention | ❌ | ✅ |
| Automatic region clustering | ❌ | ✅ |
| Expected improvement for weak regions | 0% | +15-20% |

### vs. Separate Models per Region

| Feature | Separate Models | Multi-Regional |
|---------|----------------|----------------|
| Number of models | N (per region) | 1 |
| Shared knowledge | ❌ | ✅ |
| Training efficiency | Low | High |
| Memory usage | High | Low |
| Adaptation to new regions | Slow | Fast |

## Future Enhancements

### Planned Features

1. **Meta-Learning**: Fast adaptation to new regions with few samples
2. **Hierarchical Model**: Global → Regional → Local hierarchy
3. **Dynamic Region Detection**: Automatic region discovery
4. **Multi-Modal Features**: Integration with additional data sources

### Research Directions

1. **Transfer Learning**: Between similar geographical regions
2. **Few-Shot Learning**: For regions with limited data
3. **Active Learning**: Smart data collection for new regions
4. **Uncertainty Quantification**: Better confidence estimation

## Troubleshooting

### Common Issues

1. **Region Clustering Fails**: Check if all cells have valid coordinates
2. **Poor Regional Performance**: Adjust number of regions or embedding dimension
3. **Training Instability**: Reduce learning rate or increase dropout

### Performance Optimization

1. **Memory Usage**: Use smaller embedding dimensions for large regions
2. **Training Speed**: Increase batch size or use mixed precision
3. **Model Size**: Consider model compression for deployment

## Contributing

To contribute to the Multi-Regional Model:

1. Test on new geographical regions
2. Experiment with different clustering algorithms
3. Optimize attention mechanism
4. Add new evaluation metrics

## License

This model is part of the ParaglideML project. See LICENSE for details.