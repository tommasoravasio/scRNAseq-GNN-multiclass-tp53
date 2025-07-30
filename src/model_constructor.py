"""
GNN model, training, and evaluation utilities for single-cell data.
Adapted for multiclass classification.
"""
import numpy as np
import pandas as pd
import torch
import os
import csv
import json
import sys
from datetime import datetime
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Module, CrossEntropyLoss
from torch_geometric.loader import DataLoader
from pathlib import Path
from torch_geometric.nn import GraphNorm
from torch_geometric.nn import GATConv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import optuna
import argparse

def get_num_classes(graphs):
    """Infer the number of classes from a list of PyG Data objects."""
    all_labels = torch.cat([data.y for data in graphs])
    return int(all_labels.max().item() + 1)

class GCN(Module):
    """Graph Convolutional Network (GCN) model for multiclass classification."""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate, use_graphnorm=False):
        super(GCN,self).__init__()
        self.use_graphnorm = use_graphnorm

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1=GraphNorm(hidden_channels) if use_graphnorm else torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2=GraphNorm(hidden_channels) if use_graphnorm else torch.nn.BatchNorm1d(hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout( p=dropout_rate )
        

    def forward(self, x, edge_index,batch ):
        """Forward pass for GCN."""
        x=self.conv1(x, edge_index)
        x=self.bn1(x,batch) if self.use_graphnorm else self.bn1(x)
        x=F.relu(x)
        x=self.dropout(x)

        x=self.conv2(x,edge_index)
        x=self.bn2(x,batch) if self.use_graphnorm else self.bn2(x)
        x=F.relu(x)
        x=self.dropout(x)

        x= global_mean_pool(x, batch)
        x= self.lin(x)
        return x

class GAT(Module):
    """Graph Attention Network (GAT) model for multiclass classification."""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate, use_graphnorm=False, heads=1,use_third_layer=False):
        super(GAT,self).__init__()
        self.use_graphnorm = use_graphnorm
        self.heads = heads
        self.use_third_layer = use_third_layer

        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.bn1 = GraphNorm(hidden_channels * heads) if use_graphnorm else torch.nn.BatchNorm1d(hidden_channels * heads)

        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
        self.bn2 = GraphNorm(hidden_channels * heads) if use_graphnorm else torch.nn.BatchNorm1d(hidden_channels * heads)

        if use_third_layer:
            self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
            self.bn3 = GraphNorm(hidden_channels * heads) if use_graphnorm else torch.nn.BatchNorm1d(hidden_channels * heads)

        self.lin = torch.nn.Linear(hidden_channels * heads, out_channels)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_index, batch):
        """Forward pass for GAT."""
        x = self.conv1(x, edge_index)
        x = self.bn1(x, batch) if self.use_graphnorm else self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x, batch) if self.use_graphnorm else self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        if self.use_third_layer:
            x = self.conv3(x, edge_index)
            x = self.bn3(x, batch) if self.use_graphnorm else self.bn3(x)
            x = F.relu(x)
            x = self.dropout(x)

        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

def train(model,train_loader,optimizer,criterion,device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def get_batch_class_distribution(batch):
    """Get class distribution in a batch for debugging."""
    unique, counts = torch.unique(batch.y, return_counts=True)
    distribution = {int(unique[i]): int(counts[i]) for i in range(len(unique))}
    return distribution
    

def evaluate(model, loader, device, criterion, compute_confusion_matrix=False, num_classes=None):
    """Evaluate model and optionally return confusion matrix."""
    model.eval()
    y_true = []
    y_pred = []
    loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            loss += criterion(out, batch.y).item()
    acc = accuracy_score(y_true, y_pred)
    avg_loss = loss / len(loader)
    mat = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))) if (compute_confusion_matrix and num_classes is not None) else None
    return acc, avg_loss, mat

def train_model(train_PyG, test_PyG, batch_size=32, hidden_channels=64, dropout_rate=0.2, lr=0.0001,
                epochs=30, ID_model="baseline", loss_weight=False, use_graphnorm=False, use_adamW=False, 
                weight_decay=1e-4, model_type="gcn", heads=1, use_third_layer=False, feature_selection="HVG", early_stopping=False ):
    """Train and evaluate a GCN or GAT model for multiclass classification."""
    
    # Create balanced sampler for training data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = get_num_classes(train_PyG)
    
    # Extract labels from training data
    train_labels = torch.cat([data.y for data in train_PyG])
    
    # Calculate class weights for balanced sampling
    class_counts = torch.bincount(train_labels, minlength=num_classes)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[train_labels]
    
    # Create weighted random sampler
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_PyG),
        replacement=True
    )
    
    # Create data loaders with balanced sampler for training
    train_loader = DataLoader(train_PyG, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_PyG, batch_size=batch_size)
    
    # Print class distribution information
    print(f"Class distribution in training data:")
    for i in range(num_classes):
        count = class_counts[i].item()
        percentage = (count / len(train_PyG)) * 100
        print(f"  Class {i}: {count} samples ({percentage:.1f}%)")
    
    print(f"Using balanced sampling with class weights: {class_weights.tolist()}")
    
    # Calculate class weights for loss function (inverse frequency weighting)
    labels = torch.cat([data.y for data in train_PyG])
    class_counts = torch.bincount(labels, minlength=num_classes)
    total = class_counts.sum()
    weights = (total / (num_classes * class_counts)).to(device)
    
    if loss_weight:
        print(f"Using weighted loss function with weights: {weights.tolist()}")
    else:
        print("Using standard loss function (no class weights)")

    if model_type == "gcn":
        model = GCN(
            in_channels=train_PyG[0].x.shape[1],
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            dropout_rate=dropout_rate,
            use_graphnorm=use_graphnorm
        ).to(device)
    elif model_type == "gat":
        model = GAT(
            in_channels=train_PyG[0].x.shape[1],
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            dropout_rate=dropout_rate,
            use_graphnorm=use_graphnorm,
            heads=heads,
            use_third_layer=use_third_layer
        ).to(device)
    else:
        raise KeyError("model_type does not exist, has to be either \"gcn\" (default value) or \"gat\" ")
    
    if use_adamW:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create loss function with or without class weights
    if loss_weight:
        criterion = CrossEntropyLoss(weight=weights).to(device)
    else:
        criterion = CrossEntropyLoss().to(device)
    results_dir = f"Results/{feature_selection}/{model_type}_results/{ID_model}"
    os.makedirs(results_dir, exist_ok=True)
    log_path = f"{results_dir}/training_log.csv"
    with open(log_path,mode="w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Loss", "Train Accuracy", "Train Macro F1", "Test Accuracy", "Test Macro F1", "Test Loss", "Min Class F1", "Max Class F1"])
        
        # Early stopping variables
        patience = 10  
        best_f1 = 0
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(1,epochs+1):
            # Debug batch composition for first few epochs
            if epoch <= 3:
                print(f"\nEpoch {epoch} - First batch class distribution:")
                for i, batch in enumerate(train_loader):
                    if i == 0:  # Only check first batch
                        batch_dist = get_batch_class_distribution(batch)
                        print(f"  Batch classes: {batch_dist}")
                        break
            
            loss = train(model, train_loader, optimizer, criterion, device)
            train_acc, train_loss, _ = evaluate(model, train_loader, device, criterion, compute_confusion_matrix=False, num_classes=num_classes)
            test_acc, test_loss, _ = evaluate(model, test_loader, device, criterion, compute_confusion_matrix=False, num_classes=num_classes)
            # Calculate F1 train/test
            model.eval()
            y_true_train, y_pred_train = [], []
            y_true_test, y_pred_test = [], []

            with torch.no_grad():
                for batch in train_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.batch)
                    preds = out.argmax(dim=1)
                    y_pred_train.extend(preds)
                    y_true_train.extend(batch.y.cpu().numpy())

                for batch in test_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.batch)
                    preds = out.argmax(dim=1)
                    y_pred_test.extend(preds)
                    y_true_test.extend(batch.y.cpu().numpy())

            # # DEBUG: Print F1 calculation inputs
            # print(f"DEBUG: y_true_train: {y_true_train}")
            # print(f"DEBUG: y_pred_train: {y_pred_train}")
            # print(f"DEBUG: y_true_test: {y_true_test}")
            # print(f"DEBUG: y_pred_test: {y_pred_test}")

            try:
                train_f1 = f1_score(y_true_train, y_pred_train, average="macro", zero_division="warn")
            except Exception as e:
                # print(f"DEBUG: Exception in train_f1 calculation: {e}")
                train_f1 = 0.0
            try:
                test_f1 = f1_score(y_true_test, y_pred_test, average="macro", zero_division="warn")
            except Exception as e:
                # print(f"DEBUG: Exception in test_f1 calculation: {e}")
                test_f1 = 0.0

            # Calculate per-class F1 scores for better monitoring
            try:
                per_class_f1 = f1_score(y_true_test, y_pred_test, average=None, zero_division=0)
                min_class_f1 = min(per_class_f1)
                max_class_f1 = max(per_class_f1)
            except:
                min_class_f1 = max_class_f1 = 0.0
            
            print(f"Epoch: {epoch} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Train Macro F1: {train_f1:.4f} | Test Acc: {test_acc:.4f} | Test Macro F1: {test_f1:.4f} | Test Loss: {test_loss:.4f}")
            print(f"  Per-class F1 range: [{min_class_f1:.3f}, {max_class_f1:.3f}]")
            writer.writerow([epoch, loss, train_acc, train_f1, test_acc, test_f1, test_loss, min_class_f1, max_class_f1])

            if test_f1 > best_f1:
                best_f1 = test_f1
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
                if early_stopping and epochs_no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model_path = f"{results_dir}/{model_type}_model.pt"
    torch.save(model.state_dict(), model_path)
    accuracy, avg_loss, mat = evaluate(model, test_loader, device, criterion, compute_confusion_matrix=True, num_classes=num_classes)
    if mat is not None:
        np.savetxt(f"{results_dir}/confusion_matrix.csv", mat, delimiter=",", fmt="%d")

    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index,batch.batch)
            probs = torch.softmax(out, dim=1).cpu().numpy()  # shape: (batch_size, num_classes)
            preds = out.argmax(dim=1).cpu().numpy()
            y_prob.extend(probs)
            y_pred.extend(preds)
            y_true.extend(batch.y.cpu().numpy())
    # # DEBUG: Print y_true, y_pred, y_prob for metrics
    # print(f"DEBUG: Final y_true: {y_true}")
    # print(f"DEBUG: Final y_pred: {y_pred}")
    # print(f"DEBUG: Final y_prob: {y_prob}")

    # Compute metrics for multiclass
    try:
        precision = precision_score(y_true, y_pred, average="macro", zero_division="warn")
    except Exception as e:
        # print(f"DEBUG: Exception in precision_score: {e}")
        precision = 0.0
    try:
        recall = recall_score(y_true, y_pred, average="macro", zero_division="warn")
    except Exception as e:
        # print(f"DEBUG: Exception in recall_score: {e}")
        recall = 0.0
    try:
        f1 = f1_score(y_true, y_pred, average="macro", zero_division="warn")
    except Exception as e:
        # print(f"DEBUG: Exception in f1_score: {e}")
        f1 = 0.0
    # Compute multiclass ROC AUC if possible
    try:
        # One-hot encode y_true for roc_auc_score
        y_true_oh = np.eye(num_classes)[np.array(y_true)]
        auc = roc_auc_score(y_true_oh, np.array(y_prob), average="macro", multi_class="ovr")
    except Exception as e:
        # print(f"DEBUG: Exception in roc_auc_score: {e}")
        auc = None

    # Calculate per-class metrics
    try:
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    except:
        per_class_precision = per_class_recall = per_class_f1 = [0.0] * num_classes
    
    summary_metrics = {
        "final_accuracy": accuracy,
        "final_loss": avg_loss,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
        "per_class_precision": per_class_precision.tolist() if hasattr(per_class_precision, 'tolist') else per_class_precision,
        "per_class_recall": per_class_recall.tolist() if hasattr(per_class_recall, 'tolist') else per_class_recall,
        "per_class_f1": per_class_f1.tolist() if hasattr(per_class_f1, 'tolist') else per_class_f1,
        "number_of_epochs": epochs,
        "hidden_channels":hidden_channels,
        "dropout_rate":dropout_rate,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "heads": heads,
        "use_graphnorm": use_graphnorm,
        "use_third_layer": use_third_layer,
        "best_epoch": epoch - epochs_no_improve,
        "best_f1": best_f1,
        "early_stopping": early_stopping,
        "ID_model": ID_model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_classes": num_classes
    }
    with open(f"{results_dir}/summary_metrics.json", "w") as f:
        json.dump(summary_metrics, f, indent=4)
    
    # Print final per-class performance
    print(f"\nFinal per-class performance:")
    for i in range(num_classes):
        print(f"  Class {i}: Precision={per_class_precision[i]:.3f}, Recall={per_class_recall[i]:.3f}, F1={per_class_f1[i]:.3f}")

    return model

def load_graphs(path):
    """Load graph data objects from .pt files in a directory."""
    graph_list = []
    path_obj = Path(path)
    if not path_obj.exists() or not path_obj.is_dir():
        print(f"Error: Graph directory not found or is not a directory: {path}", file=sys.stderr)
        sys.exit(1) # Exit if path is invalid

    pt_files = sorted(path_obj.glob("*.pt"))
    if not pt_files:
        print(f"Warning: No .pt files found in {path}. Exiting.", file=sys.stderr)
        sys.exit(1) # Exit if no .pt files are found

    for pt_file in pt_files:
        graph = torch.load(pt_file, weights_only=False)

        if isinstance(graph, list):
            graph_list.extend(graph)
        else:
            graph_list.append(graph)

    if not graph_list:
        print(f"Warning: No graphs were successfully loaded from {path}. Exiting.", file=sys.stderr)
        sys.exit(1) # Exit if no graphs loaded

    return graph_list

def main_optuna(optuna_config_path):
    """Run Optuna hyperparameter search from config file."""
    from optuna_utils import run_optuna_study 
    print(f"Starting Optuna hyperparameter tuning with config: {optuna_config_path}")
    run_optuna_study(optuna_config_path)

def main_baseline(config_path):
    """Train and evaluate model from config file."""
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from configuration file {config_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Configuration loaded: {config}")

    epochs = config["epochs"]
    batch_size = config["batch_size"]
    ID_model = config["ID_model"]
    use_adamW = config["use_adamW"]
    model_type = config["model_type"]
    use_graphnorm = config["use_graphnorm"]
    feature_selection = config["feature_selection"]
    graphs_path_suffix = config["graphs_path"] # Read from config

    hidden_channels = config["hidden_channels"]
    dropout_rate = config["dropout_rate"]
    lr = config["lr"]
    loss_weight = config["loss_weight"]
    weight_decay = config["weight_decay"]
    heads = config["heads"]
    use_third_layer = config["use_third_layer"]
    early_stopping = config["early_stopping"]

    train_data_path = f"{graphs_path_suffix}/train"
    test_data_path = f"{graphs_path_suffix}/test"
    print(f"Loading training data from: {train_data_path}")
    train_df_pyg = load_graphs(train_data_path)
    print(f"Loading test data from: {test_data_path}")
    test_df_pyg = load_graphs(test_data_path)

    # print(f"DEBUG: train_df_pyg length: {len(train_df_pyg)}")
    # print(f"DEBUG: test_df_pyg length: {len(test_df_pyg)}")
    # if len(train_df_pyg) > 0:
    #     print(f"DEBUG: train_df_pyg[0] x shape: {getattr(train_df_pyg[0].x, 'shape', None)}")
    #     print(f"DEBUG: train_df_pyg[0] y: {getattr(train_df_pyg[0], 'y', None)}")
    # if len(test_df_pyg) > 0:
    #     print(f"DEBUG: test_df_pyg[0] x shape: {getattr(test_df_pyg[0].x, 'shape', None)}")
    #     print(f"DEBUG: test_df_pyg[0] y: {getattr(test_df_pyg[0], 'y', None)}")

    model = train_model(
        train_PyG=train_df_pyg,
        test_PyG=test_df_pyg,
        epochs=epochs,
        batch_size=batch_size,
        ID_model=ID_model,
        use_adamW=use_adamW,
        model_type=model_type,
        use_graphnorm=use_graphnorm,
        feature_selection=feature_selection,
        hidden_channels=hidden_channels,
        dropout_rate=dropout_rate,
        lr=lr,
        loss_weight=loss_weight,
        weight_decay=weight_decay,
        heads=heads,
        use_third_layer=use_third_layer,
        early_stopping=early_stopping
    )

# LOCAL TESTING
def test_run_baseline():
    """Quick local test run of GAT model training."""
    train_df_pyg_big = load_graphs("output/graphs_HVG_None/train")
    test_df_pyg_big = load_graphs("output/graphs_target_None/test")
    train_df_pyg_small = train_df_pyg_big[:5]
    test_df_pyg_small = test_df_pyg_big[:5]
    
    # print(f"DEBUG: test_run_baseline train_df_pyg_small length: {len(train_df_pyg_small)}")
    # print(f"DEBUG: test_run_baseline test_df_pyg_small length: {len(test_df_pyg_small)}")
    # if len(train_df_pyg_small) > 0:
    #     print(f"DEBUG: train_df_pyg_small[0] x shape: {getattr(train_df_pyg_small[0].x, 'shape', None)}")
    #     print(f"DEBUG: train_df_pyg_small[0] y: {getattr(train_df_pyg_small[0], 'y', None)}")
    # if len(test_df_pyg_small) > 0:
    #     print(f"DEBUG: test_df_pyg_small[0] x shape: {getattr(test_df_pyg_small[0].x, 'shape', None)}")
    #     print(f"DEBUG: test_df_pyg_small[0] y: {getattr(test_df_pyg_small[0], 'y', None)}")
    
    model = train_model(
        train_PyG=train_df_pyg_small,
        test_PyG=test_df_pyg_small,
        hidden_channels=32,
        dropout_rate=0.3,
        lr=0.001,
        use_adamW=True,
        weight_decay=1e-4,
        loss_weight=False,
        epochs=2,
        batch_size=2,
        ID_model="test_run",
        model_type="gat",
        heads=2,
        use_graphnorm=True,
        use_third_layer=False,
        feature_selection="local_testing"
    )
    print("Test run completed successfully.")

# LOCAL TESTING
def main_optuna_test():
    """Quick local test run of Optuna integration."""
    train_df_pyg_big = load_graphs("output/graphs_HVG_None/train")
    test_df_pyg_big = load_graphs("output/graphs_target_None/test")
    train_df_pyg_small = train_df_pyg_big[:5]
    test_df_pyg_small = test_df_pyg_big[:5]

    # print(f"DEBUG: main_optuna_test train_df_pyg_small length: {len(train_df_pyg_small)}")
    # print(f"DEBUG: main_optuna_test test_df_pyg_small length: {len(test_df_pyg_small)}")
    # if len(train_df_pyg_small) > 0:
    #     print(f"DEBUG: train_df_pyg_small[0] x shape: {getattr(train_df_pyg_small[0].x, 'shape', None)}")
    #     print(f"DEBUG: train_df_pyg_small[0] y: {getattr(train_df_pyg_small[0], 'y', None)}")
    # if len(test_df_pyg_small) > 0:
    #     print(f"DEBUG: test_df_pyg_small[0] x shape: {getattr(test_df_pyg_small[0].x, 'shape', None)}")
    #     print(f"DEBUG: test_df_pyg_small[0] y: {getattr(test_df_pyg_small[0], 'y', None)}")

    def objective_test(trial):
        """Objective function for the `main_optuna_test` Optuna study.

        Called by Optuna during the test optimization. It samples
        hyperparameters from a reduced search space (suitable for testing)
        using the `trial` object, trains a GAT model on a small subset of
        data, and returns the F1 score.

        Args:
            trial (optuna.trial.Trial): An Optuna trial object, suggesting
                hyperparameters from a limited test range.

        Returns:
            float: The F1 score from training the model with test hyperparameters.
        """
        hidden_channels = trial.suggest_categorical("hidden_channels", [32])
        dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.3)
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        heads = trial.suggest_categorical("heads", [2])
        loss_weight = trial.suggest_categorical("loss_weight", [False])
        use_third_layer = trial.suggest_categorical("use_third_layer", [False])

        # print(f"DEBUG: Optuna trial {trial.number} - hidden_channels: {hidden_channels}, dropout_rate: {dropout_rate}, lr: {lr}, weight_decay: {weight_decay}, heads: {heads}, loss_weight: {loss_weight}, use_third_layer: {use_third_layer}")

        model = train_model(
            train_PyG=train_df_pyg_small,
            test_PyG=test_df_pyg_small,
            hidden_channels=hidden_channels,
            dropout_rate=dropout_rate,
            lr=lr,
            use_adamW=True,
            weight_decay=weight_decay,
            loss_weight=loss_weight,
            epochs=2,
            batch_size=2,
            ID_model=f"optuna_test_{trial.number}",
            model_type="gat",
            heads=heads,
            use_graphnorm=True,
            use_third_layer=use_third_layer,
            feature_selection="local_testing"
        )



    study = optuna.create_study(direction="maximize")
    study.optimize(objective_test, n_trials=1)

    print("Optuna test run completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model training or hyperparameter tuning.")
    parser.add_argument("--mode", choices=["train", "optuna"], required=True,
                        help="'train' to train a model with a specific config, 'optuna' to run hyperparameter tuning.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the JSON configuration file. For 'train' mode, this is the model config. For 'optuna' mode, this is the Optuna study config.")

    args = parser.parse_args()

    if args.mode == "optuna":
        
        main_optuna(optuna_config_path=args.config)

    elif args.mode == "train":
        
        main_baseline(config_path=args.config)
    else:
        
        print(f"Error: Invalid mode '{args.mode}'. Choose 'train' or 'optuna'.", file=sys.stderr)
        sys.exit(1)

    # #FOR TESTING
    # test_run_baseline()
    # main_optuna_test()