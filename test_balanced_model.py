#!/usr/bin/env python3
"""
Test script for the enhanced class balancing techniques.
This script demonstrates different approaches to handle class imbalance.
"""

import sys
import os
sys.path.append('src')

from model_constructor import train_model, load_graphs
import json

def test_baseline_vs_balanced():
    """Compare baseline model with balanced model."""
    
    print("=" * 60)
    print("TESTING CLASS BALANCING TECHNIQUES")
    print("=" * 60)
    
    # Load data
    print("Loading data...")
    train_data_path = "output/graphs_target_/train"
    test_data_path = "output/graphs_target_/test"
    
    train_graphs = load_graphs(train_data_path)
    test_graphs = load_graphs(test_data_path)
    
    print(f"Training samples: {len(train_graphs)}")
    print(f"Test samples: {len(test_graphs)}")
    
    # Test 1: Baseline model (current approach)
    print("\n" + "=" * 40)
    print("TEST 1: BASELINE MODEL")
    print("=" * 40)
    
    baseline_model = train_model(
        train_PyG=train_graphs,
        test_PyG=test_graphs,
        epochs=10,  # Reduced for testing
        batch_size=32,
        ID_model="baseline_test",
        use_adamW=True,
        model_type="gat",
        use_graphnorm=True,
        feature_selection="target",
        hidden_channels=64,
        dropout_rate=0.3,
        lr=0.001,
        loss_weight=True,  # Use weighted loss
        weight_decay=1e-4,
        heads=2,
        use_third_layer=False,
        early_stopping=False,
        use_balanced_sampling=True,
        use_focal_loss=False,
        use_data_balancing=False
    )
    
    # Test 2: Enhanced balanced model
    print("\n" + "=" * 40)
    print("TEST 2: ENHANCED BALANCED MODEL")
    print("=" * 40)
    
    balanced_model = train_model(
        train_PyG=train_graphs,
        test_PyG=test_graphs,
        epochs=10,  # Reduced for testing
        batch_size=32,
        ID_model="balanced_test",
        use_adamW=True,
        model_type="gat",
        use_graphnorm=True,
        feature_selection="target",
        hidden_channels=64,
        dropout_rate=0.3,
        lr=0.001,
        loss_weight=False,  # Use focal loss instead
        weight_decay=1e-4,
        heads=2,
        use_third_layer=False,
        early_stopping=False,
        use_balanced_sampling=True,
        use_focal_loss=True,
        focal_alpha=1,
        focal_gamma=2,
        use_data_balancing=True,
        target_samples_per_class=1000,  # Target 1000 samples per class
        max_oversampling_ratio=2.0
    )
    
    # Test 3: Focal Loss only (no data balancing)
    print("\n" + "=" * 40)
    print("TEST 3: FOCAL LOSS ONLY")
    print("=" * 40)
    
    focal_model = train_model(
        train_PyG=train_graphs,
        test_PyG=test_graphs,
        epochs=10,  # Reduced for testing
        batch_size=32,
        ID_model="focal_test",
        use_adamW=True,
        model_type="gat",
        use_graphnorm=True,
        feature_selection="target",
        hidden_channels=64,
        dropout_rate=0.3,
        lr=0.001,
        loss_weight=False,
        weight_decay=1e-4,
        heads=2,
        use_third_layer=False,
        early_stopping=False,
        use_balanced_sampling=True,
        use_focal_loss=True,
        focal_alpha=1,
        focal_gamma=2,
        use_data_balancing=False
    )
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETED")
    print("=" * 60)
    print("Check the Results/target/gat_results/ directory for detailed metrics.")
    print("Compare the macro F1 scores between baseline_test, balanced_test, and focal_test.")

if __name__ == "__main__":
    test_baseline_vs_balanced() 