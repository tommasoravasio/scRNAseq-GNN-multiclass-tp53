# Class Imbalance Solutions for GNN Multiclass Classification

## Problem Description

Your dataset has a severe class imbalance where:
- **Class 0 (Missense_Mutation)**: 21,560 samples (56.2% of data)
- **Class 1 (Nonsense_Mutation)**: 6,118 samples (15.9%)
- **Class 2 (Frame_Shift_Del)**: 5,012 samples (13.1%)
- **Class 3 (Splice_Site)**: 2,291 samples (6.0%)
- **Class 4 (Frame_Shift_Ins)**: 964 samples (2.5%)
- **Class 5 (Intron)**: 931 samples (2.4%)
- **Class 6 (In_Frame_Del)**: 574 samples (1.5%)
- **Class 7 (Silent)**: 384 samples (1.0%)
- **Class 8 (In_Frame_Ins)**: 300 samples (0.8%)
- **Class 9 (Splice_Region)**: 230 samples (0.6%)

This imbalance causes the model to predict only the majority class, resulting in macro F1 = 0.

## Solutions Implemented

### 1. **Focal Loss**
- **Purpose**: Reduces the relative loss for well-classified examples and focuses on hard examples
- **Parameters**: 
  - `alpha`: Controls class weighting (default: 1)
  - `gamma`: Controls focusing (default: 2)
- **Usage**: Set `use_focal_loss=True` in configuration

### 2. **Data Balancing with Oversampling/Undersampling**
- **Purpose**: Creates a balanced dataset by oversampling minority classes and undersampling majority classes
- **Parameters**:
  - `target_samples_per_class`: Target number of samples per class (default: median class size)
  - `max_oversampling_ratio`: Maximum oversampling ratio to prevent overfitting (default: 3.0)
- **Usage**: Set `use_data_balancing=True` in configuration

### 3. **Weighted Random Sampling**
- **Purpose**: Ensures each batch contains samples from all classes during training
- **Usage**: Set `use_balanced_sampling=True` in configuration

### 4. **Class-Weighted Loss Function**
- **Purpose**: Gives higher weight to minority classes in the loss function
- **Usage**: Set `loss_weight=True` in configuration

## Configuration Examples

### Basic Balanced Model
```json
{
    "epochs": 50,
    "batch_size": 32,
    "ID_model": "balanced_gat",
    "use_adamW": true,
    "model_type": "gat",
    "use_graphnorm": true,
    "feature_selection": "target",
    "hidden_channels": 128,
    "dropout_rate": 0.3,
    "lr": 0.001,
    "loss_weight": false,
    "weight_decay": 1e-4,
    "heads": 4,
    "use_third_layer": true,
    "early_stopping": true,
    "use_balanced_sampling": true,
    "use_focal_loss": true,
    "focal_alpha": 1,
    "focal_gamma": 2,
    "use_data_balancing": true,
    "target_samples_per_class": 2000,
    "max_oversampling_ratio": 2.5
}
```

### Focal Loss Only (No Data Balancing)
```json
{
    "use_focal_loss": true,
    "focal_alpha": 1,
    "focal_gamma": 2,
    "use_data_balancing": false,
    "use_balanced_sampling": true
}
```

### Data Balancing Only (No Focal Loss)
```json
{
    "use_focal_loss": false,
    "use_data_balancing": true,
    "target_samples_per_class": 1500,
    "max_oversampling_ratio": 2.0,
    "use_balanced_sampling": true
}
```

## Running the Enhanced Model

### Option 1: Use the new configuration
```bash
python src/model_constructor.py --mode train --config configs/balanced_gat.json
```

### Option 2: Test different approaches
```bash
python test_balanced_model.py
```

### Option 3: Use the enhanced function directly
```python
from src.model_constructor import train_model, load_graphs

# Load your data
train_graphs = load_graphs("output/graphs_target_None/train")
test_graphs = load_graphs("output/graphs_target_None/test")

# Train with enhanced balancing
model = train_model(
    train_PyG=train_graphs,
    test_PyG=test_graphs,
    epochs=50,
    batch_size=32,
    ID_model="my_balanced_model",
    use_focal_loss=True,
    focal_alpha=1,
    focal_gamma=2,
    use_data_balancing=True,
    target_samples_per_class=2000,
    max_oversampling_ratio=2.5,
    use_balanced_sampling=True,
    model_type="gat",
    use_graphnorm=True
)
```

## Expected Improvements

1. **Macro F1 Score**: Should improve from 0 to >0.3-0.5
2. **Per-class Performance**: All classes should show some predictive power
3. **Confusion Matrix**: Should show predictions across all classes, not just the majority class

## Monitoring Progress

The enhanced training function provides detailed logging:
- Original vs. balanced class distribution
- Per-class F1 scores during training
- Final per-class precision, recall, and F1 scores

## Recommendations

1. **Start with Focal Loss**: Try `use_focal_loss=True` first as it's less aggressive
2. **Add Data Balancing**: If Focal Loss alone isn't sufficient, add `use_data_balancing=True`
3. **Tune Parameters**: Adjust `target_samples_per_class` and `max_oversampling_ratio` based on your results
4. **Monitor Overfitting**: Watch for signs of overfitting when using aggressive oversampling

## Troubleshooting

- **If macro F1 is still 0**: Try increasing `target_samples_per_class` or `max_oversampling_ratio`
- **If overfitting occurs**: Reduce `max_oversampling_ratio` or increase `dropout_rate`
- **If training is slow**: Reduce `target_samples_per_class` or use focal loss only

The enhanced model should significantly improve your macro F1 score and provide better predictions across all classes. 