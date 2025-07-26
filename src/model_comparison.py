"""
This script orchestrates the comparison of different architecture on single-cell RNA-seq graph datasets. 
    python model_comparison.py --config configs/comparison_template.json
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import mygene
sys.path.append(os.path.abspath("../src"))
import importlib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.stats
import math
from torch_geometric.transforms import LargestConnectedComponents
import torch_geometric.utils as tg_utils
from torch_geometric.data import Data 
import networkx as nx
import torch
import seaborn as sns
import gc
from pathlib import Path
import scanpy.external as sce
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import argparse



import model_constructor


def plot_training_curves(csv_path, model_name="Model", num_classes=2):
    """Plot training/validation curves from CSV log for multiclass classification."""
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(csv_path)
    plt.figure(figsize=(18, 5))
    
    # Accuracy plot
    plt.subplot(1, 3, 1)
    sns.lineplot(data=df, x="Epoch", y="Train Accuracy", label="Train Accuracy")
    sns.lineplot(data=df, x="Epoch", y="Test Accuracy", label="Validation Accuracy")
    plt.title(f"{model_name} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)

    # Loss plot
    plt.subplot(1, 3, 2)
    sns.lineplot(data=df, x="Epoch", y="Loss", label="Train Loss")
    if "Test Loss" in df.columns:
        sns.lineplot(data=df, x="Epoch", y="Test Loss", label="Validation Loss")
    plt.title(f"{model_name} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    # F1 Score plot (macro average for multiclass)
    plt.subplot(1, 3, 3)
    f1_plotted = False
    if "Train F1" in df.columns:
        sns.lineplot(data=df, x="Epoch", y="Train F1", label="Train F1 (Macro)")
        f1_plotted = True
    if "Test F1" in df.columns:
        sns.lineplot(data=df, x="Epoch", y="Test F1", label="Validation F1 (Macro)")
        f1_plotted = True
    if f1_plotted:
        plt.title(f"{model_name} - F1 Score (Macro Average)")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.grid(True)
    else:
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def evaluate_multiclass_model(y_true, y_pred, y_proba=None, class_names=None):
    """
    Evaluate multiclass classification model with comprehensive metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for ROC AUC)
        class_names: List of class names (optional)
    
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1 (macro average for multiclass)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Weighted average (accounts for class imbalance)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Micro average
    metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    # ROC AUC (if probabilities are provided)
    if y_proba is not None:
        try:
            # For multiclass, use one-vs-rest approach
            metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_proba, multi_class='ovo', average='macro')
        except Exception as e:
            print(f"Warning: Could not calculate ROC AUC: {e}")
    
    # Per-class metrics
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(np.unique(y_true)))]
    
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['per_class'] = {
        'precision': dict(zip(class_names, precision_per_class)),
        'recall': dict(zip(class_names, recall_per_class)),
        'f1': dict(zip(class_names, f1_per_class))
    }
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # Classification report
    metrics['classification_report'] = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    return metrics


def plot_multiclass_results(metrics, model_name="Model", save_path=None):
    """
    Plot comprehensive results for multiclass classification.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        model_name: Name of the model for plot titles
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Confusion Matrix
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
    axes[0,0].set_title(f'{model_name} - Confusion Matrix')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Actual')
    
    # 2. Overall Metrics Comparison
    overall_metrics = ['precision_macro', 'recall_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    metric_values = [metrics[metric] for metric in overall_metrics]
    metric_names = ['P-Macro', 'R-Macro', 'F1-Macro', 'P-Weighted', 'R-Weighted', 'F1-Weighted']
    
    axes[0,1].bar(metric_names, metric_values, color=['blue', 'green', 'red', 'orange', 'purple', 'brown'])
    axes[0,1].set_title(f'{model_name} - Overall Metrics')
    axes[0,1].set_ylabel('Score')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].set_ylim(0, 1)
    
    # 3. Per-class F1 Scores
    class_names = list(metrics['per_class']['f1'].keys())
    f1_scores = list(metrics['per_class']['f1'].values())
    
    axes[1,0].bar(class_names, f1_scores, color='skyblue')
    axes[1,0].set_title(f'{model_name} - F1 Score per Class')
    axes[1,0].set_ylabel('F1 Score')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].set_ylim(0, 1)
    
    # 4. Per-class Precision vs Recall
    precision_scores = list(metrics['per_class']['precision'].values())
    recall_scores = list(metrics['per_class']['recall'].values())
    
    x = np.arange(len(class_names))
    width = 0.35
    
    axes[1,1].bar(x - width/2, precision_scores, width, label='Precision', color='lightcoral')
    axes[1,1].bar(x + width/2, recall_scores, width, label='Recall', color='lightgreen')
    axes[1,1].set_title(f'{model_name} - Precision vs Recall per Class')
    axes[1,1].set_ylabel('Score')
    axes[1,1].set_xlabel('Classes')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(class_names, rotation=45)
    axes[1,1].legend()
    axes[1,1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def main(config_path):
    """Run model comparisons from a config file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {config_path}. Details: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded configuration from: {config_path}")

    global_feature_selection = config.get("feature_selection", "HVG")
    base_data_path_prefix = config.get("base_data_path_prefix", "data/graphs_")
    dataset_variants_map = config.get("dataset_variants", {})
    default_train_params = config.get("default_train_model_params", {})
    comparison_runs = config.get("comparison_runs", [])
    
    # Add multiclass configuration
    num_classes = config.get("num_classes", 2)  # Default to binary if not specified
    class_names = config.get("class_names", None)  # Optional class names for better visualization

    if not comparison_runs:
        print("Warning: No comparison runs defined in config.", file=sys.stderr)
        return

    print(f"Global feature selection: {global_feature_selection}")
    print(f"Number of classes: {num_classes}")
    if class_names:
        print(f"Class names: {class_names}")

    for i, run_config in enumerate(comparison_runs):
        run_id = run_config.get("run_id")
        dataset_variant_key = run_config.get("dataset_variant_key")
        model_type = run_config.get("model_type")

        if not all([run_id, dataset_variant_key, model_type]):
            print(f"Run {i+1}: Skipping due to missing 'run_id', 'dataset_variant_key', or 'model_type'. Config: {run_config}", file=sys.stderr)
            continue

        print(f"\n--- Starting Run {i+1}/{len(comparison_runs)}: {run_id} ---")
        print(f"Model: {model_type}, Dataset Key: {dataset_variant_key}")

        variant_path_template = dataset_variants_map.get(dataset_variant_key)
        if not variant_path_template:
            print(f"Run '{run_id}': Dataset variant key '{dataset_variant_key}' not in 'dataset_variants'. Skipping.", file=sys.stderr)
            continue

        actual_variant_suffix = variant_path_template.replace("{feature_selection}", global_feature_selection)

        train_path = Path(f"{base_data_path_prefix}{actual_variant_suffix}/train")
        test_path = Path(f"{base_data_path_prefix}{actual_variant_suffix}/test")

        print(f"Loading train data from: {train_path}")
        train_pyg = model_constructor.load_graphs(str(train_path))
        if not train_pyg: 
             print(f"Run '{run_id}': Failed to load training data from {train_path}. Skipping.", file=sys.stderr)
             continue

        print(f"Loading test data from: {test_path}")
        test_pyg = model_constructor.load_graphs(str(test_path))
        if not test_pyg: 
             print(f"Run '{run_id}': Failed to load test data from {test_path}. Skipping.", file=sys.stderr)
             continue

        current_run_params = default_train_params.copy()
        current_run_params.update(run_config.get("train_model_params", {}))

        current_run_params["ID_model"] = run_id
        current_run_params["model_type"] = model_type
        current_run_params["feature_selection"] = global_feature_selection

        print(f"Run '{run_id}': Training with params: {current_run_params}")

        try:
            model_constructor.train_model(
                train_PyG=train_pyg,
                test_PyG=test_pyg,
                **current_run_params 
            )
            print(f"--- Finished Run: {run_id} ---")
        except Exception as e:
            print(f"Run '{run_id}': Error during training: {e}", file=sys.stderr)

    print("\nAll configured comparison runs attempted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file for model comparisons.")
    args = parser.parse_args()
    main(config_path=args.config)