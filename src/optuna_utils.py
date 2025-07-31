"""
Optuna utilities for GNN hyperparameter search for multiclass classification.
"""
import optuna
import json
import sys
import os
from functools import partial 
from model_constructor import train_model, load_graphs


def _suggest_hyperparameter(trial, param_config):
    """Suggest a hyperparameter value for Optuna."""
    param_type = param_config["type"]
    if param_type == "categorical":
        return trial.suggest_categorical(param_config["name"], param_config["choices"])
    elif param_type == "float":
        log = param_config.get("log", False)
        if "step" in param_config:
            return trial.suggest_float(param_config["name"], param_config["low"], param_config["high"], step=param_config["step"], log=log)
        else:
            return trial.suggest_float(param_config["name"], param_config["low"], param_config["high"], log=log)
    elif param_type == "int":
        log = param_config.get("log", False) 
        if "step" in param_config:
             return trial.suggest_int(param_config["name"], param_config["low"], param_config["high"], step=param_config["step"], log=log)
        else:
            return trial.suggest_int(param_config["name"], param_config["low"], param_config["high"], log=log)
    else:
        raise ValueError(f"Unsupported hyperparameter type: {param_type}")


def objective(trial, optuna_config, train_PyG, test_PyG):
    """Optuna objective: train model and return metric for multiclass classification."""
    fixed_params = optuna_config["fixed_params"]
    epochs = fixed_params["epochs"]
    batch_size = fixed_params["batch_size"]
    feature_selection = fixed_params["feature_selection"]
    model_type = fixed_params["model_type"]
    use_graphnorm = fixed_params["use_graphnorm"]
    use_adamW = fixed_params.get("use_adamW", True)
    early_stopping = fixed_params.get("early_stopping", False)
    
    # Multiclass-specific fixed parameters
    use_balanced_sampling = fixed_params.get("use_balanced_sampling", True)
    use_data_balancing = fixed_params.get("use_data_balancing", False)

    # Get hyperparameters to tune
    tuned_params = {}
    for param_name, param_details in optuna_config["hyperparameters"].items():
        tuned_params[param_name] = _suggest_hyperparameter(trial, {"name": param_name, **param_details})

    # Handle multiclass-specific hyperparameters that might be in tuned_params
    use_focal_loss = tuned_params.pop("use_focal_loss", False) if "use_focal_loss" in tuned_params else False
    focal_alpha = tuned_params.pop("focal_alpha", 1) if "focal_alpha" in tuned_params else 1
    focal_gamma = tuned_params.pop("focal_gamma", 2) if "focal_gamma" in tuned_params else 2
    
    # Handle data balancing parameters that might be in tuned_params
    target_samples_per_class = tuned_params.pop("target_samples_per_class", None) if "target_samples_per_class" in tuned_params else None
    max_oversampling_ratio = tuned_params.pop("max_oversampling_ratio", 3.0) if "max_oversampling_ratio" in tuned_params else 3.0

    model = train_model(
        train_PyG=train_PyG,
        test_PyG=test_PyG,
        epochs=epochs,
        batch_size=batch_size,
        ID_model=f"optuna_{optuna_config['study_name']}_{trial.number}",
        model_type=model_type,
        use_graphnorm=use_graphnorm,
        feature_selection=feature_selection,
        use_adamW=use_adamW,
        early_stopping=early_stopping,
        use_balanced_sampling=use_balanced_sampling,
        use_data_balancing=use_data_balancing,
        target_samples_per_class=target_samples_per_class,
        max_oversampling_ratio=max_oversampling_ratio,
        use_focal_loss=use_focal_loss,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        **tuned_params
    )

    results_dir = f"Results/{feature_selection}/{model_type}_results/optuna_{optuna_config['study_name']}_{trial.number}"
    metrics_path = f"{results_dir}/summary_metrics.json"

    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
    except FileNotFoundError:
        print(f"Error: summary_metrics.json not found at {metrics_path} for trial {trial.number}", file=sys.stderr)
        return 0.0

    metric_to_optimize = optuna_config.get("metric_to_optimize", "f1_score")
    
    # Handle multiclass metrics properly
    if metric_to_optimize in metrics:
        return metrics[metric_to_optimize]
    else:
        print(f"Warning: Metric '{metric_to_optimize}' not found in results. Available metrics: {list(metrics.keys())}", file=sys.stderr)
        # Fallback to f1_score if available, otherwise return 0
        return metrics.get("f1_score", 0.0)


def run_optuna_study(optuna_config_path):
    """Run an Optuna study from a config file for multiclass classification."""
    print(f"Loading Optuna configuration from: {optuna_config_path}")
    try:
        with open(optuna_config_path, 'r') as f:
            optuna_config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Optuna configuration file not found at {optuna_config_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from Optuna configuration file {optuna_config_path}", file=sys.stderr)
        sys.exit(1)

    study_name = optuna_config.get("study_name", "optuna_study")
    direction = optuna_config.get("direction", "maximize")
    n_trials = optuna_config.get("n_trials", 20)

    fixed_params = optuna_config["fixed_params"]
    graphs_path_suffix = fixed_params["graphs_path"]
    
    # Handle different data path formats
    if graphs_path_suffix.startswith("graphs_"):
        # If it's already a full path suffix
        train_data_path = f"output/{graphs_path_suffix}/train"
        test_data_path = f"output/{graphs_path_suffix}/test"
    else:
        # If it's just a suffix, construct the full path
        train_data_path = f"output/graphs_{graphs_path_suffix}/train"
        test_data_path = f"output/graphs_{graphs_path_suffix}/test"

    print(f"Loading training data from: {train_data_path}")
    try:
        train_PyG = load_graphs(train_data_path)
        print(f"Successfully loaded {len(train_PyG)} training graphs")
    except Exception as e:
        print(f"Error loading training data: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loading test data from: {test_data_path}")
    try:
        test_PyG = load_graphs(test_data_path)
        print(f"Successfully loaded {len(test_PyG)} test graphs")
    except Exception as e:
        print(f"Error loading test data: {e}", file=sys.stderr)
        sys.exit(1)

    # Print dataset information
    from model_constructor import get_num_classes
    num_classes = get_num_classes(train_PyG)
    print(f"Number of classes in dataset: {num_classes}")

    objective_with_args = partial(objective, optuna_config=optuna_config, train_PyG=train_PyG, test_PyG=test_PyG)
    study = optuna.create_study(study_name=study_name, direction=direction)
    study.optimize(objective_with_args, n_trials=n_trials)

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value ({optuna_config.get('metric_to_optimize', 'f1_score')}): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save study results
    study_results_dir = f"Results/{fixed_params['feature_selection']}/optuna_study_results"
    os.makedirs(study_results_dir, exist_ok=True)
    
    # Save best trial parameters
    best_params_path = f"{study_results_dir}/best_params_{study_name}.json"
    with open(best_params_path, 'w') as f:
        json.dump({
            "study_name": study_name,
            "best_value": trial.value,
            "best_params": trial.params,
            "n_trials": n_trials,
            "direction": direction
        }, f, indent=4)
    
    print(f"\nBest parameters saved to: {best_params_path}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        optuna_config_arg = sys.argv[1]
        print(f"Running Optuna study with configuration: {optuna_config_arg}")
        run_optuna_study(optuna_config_path=optuna_config_arg)
    else:
        print("Usage: python src/optuna_utils.py <path_to_optuna_config.json>")
    pass
