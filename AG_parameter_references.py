# Import libraries
from autogluon.tabular import TabularPredictor


###################################################
# TabularPredictor.__init__() — **CONSTRUCTOR**
###################################################

predictor = TabularPredictor(
    label="price",
    # Column you want to predict (target variable)
    problem_type="regression",
    # Type of ML problem
    # Options: "binary", "multiclass", "regression", "quantile"
    eval_metric="root_mean_squared_error",
    # Metric used to evaluate models
    # Examples:
    # classification → "accuracy", "f1"
    # regression → "root_mean_squared_error", "mean_absolute_error"
    path="models/trained_model",
    # Folder where trained models will be saved
    verbosity=2,  # Verbosity controls how much information or detail a program prints while running
    # Logging level
    # 0 = silent
    # 1 = warnings
    # 2 = normal
    # 3 = verbose
    # 4 = very detailed logs
    log_to_file=True,
    # Save logs to a file
    log_file_path="training_logs.txt",
    # File where logs will be stored
    sample_weight="weight_column",
    # Column name in dataset that gives weight to each row
    # Example: important samples get higher weight
    weight_evaluation=True,
    # Use sample weights when calculating evaluation metrics
    groups="user_id",
    # Column used for group-based cross validation
    # Example: keep same user's data in same fold
    positive_class=1,
    # For binary classification
    # Defines which class is treated as "positive"
)


####################################
# predictor.fit() — **TRAINING**
####################################

predictor.fit(
    # ---------------- Basic Parameters ----------------
    train_data=train_df,
    # Training dataset (pd.DataFrame or file path)
    tuning_data=val_df,
    # Validation dataset (optional). If None → AutoGluon splits train_data
    time_limit=3600,
    # Maximum training time in seconds
    presets="best_quality",
    # Quality preset:
    # "medium_quality", "good_quality", "high_quality",
    # "best_quality", "extreme_quality",
    # "optimize_for_deployment", "interpretable", "ignore_text"
    hyperparameters="zeroshot",
    # Model configuration preset
    # "default", "zeroshot", "light", "very_light", "toy", "multimodal"
    feature_metadata="infer",
    # Auto detect feature types from dataset
    infer_limit=0.05,
    # Max inference time per row (seconds)
    infer_limit_batch_size=10000,
    # Batch size used to estimate inference speed
    fit_weighted_ensemble=True,
    # Train weighted ensemble models
    fit_full_last_level_weighted_ensemble=True,
    # Final ensemble uses models from all stack layers
    full_weighted_ensemble_additionally=False,
    # Train extra ensemble after stacking
    dynamic_stacking="auto",
    # Automatically decide if stacking should be used
    calibrate_decision_threshold="auto",
    # Optimize decision threshold for binary classification
    num_cpus="auto",
    # CPUs used by AutoGluon
    num_gpus="auto",
    # GPUs used by AutoGluon
    fit_strategy="sequential",
    # Training strategy:
    # "sequential" or "parallel"
    memory_limit="auto",
    # Soft RAM limit in GB
    callbacks=None,
    # Custom callbacks during training
    # ---------------- Bagging / Stacking ----------------
    num_bag_folds=5,
    # Number of bagging folds (0 disables bagging)
    num_bag_sets=1,
    # Number of bagging repetitions
    num_stack_levels=1,
    # Number of stacking layers above base models
    auto_stack=False,
    # Automatically configure bagging + stacking
    use_bag_holdout=False,
    # Use holdout data during bagging
    holdout_frac=0.1,
    # Fraction of training data used as validation
    # ---------------- Hyperparameter Tuning ----------------
    hyperparameter_tune_kwargs={
        "num_trials": 20,
        "scheduler": "local",
        "searcher": "random",
    },
    # Hyperparameter optimization settings
    # ---------------- Model Refit ----------------
    refit_full=False,
    # Refit best models using full training data
    set_best_to_refit_full=False,
    # Set refit model as best model
    # ---------------- Feature Engineering ----------------
    feature_generator=None,
    # Custom feature generator pipeline
    # ---------------- Semi-Supervised Learning ----------------
    unlabeled_data=None,
    # Additional unlabeled dataset
    # ---------------- Model Meta Arguments ----------------
    ag_args={"name_suffix": "_v1", "priority": 0},
    # Model-level metadata arguments
    # ---------------- Model Fit Constraints ----------------
    ag_args_fit={"num_cpus": 4, "max_rows": 100000},
    # Per-model training constraints
    # ---------------- Ensemble Arguments ----------------
    ag_args_ensemble={"use_orig_features": True, "save_bag_folds": False},
    # Ensemble training configuration
    # ---------------- Model Selection ----------------
    excluded_model_types=["KNN"],
    # Skip training specific model types
    # ---------------- Dynamic Stacking Arguments ----------------
    ds_args={"validation_procedure": "holdout"},
    # Dynamic stacking settings
)


#########################################
# predictor.predict() — **INFERENCE**
#########################################

predictions = predictor.predict(
    data=test_df,
    # Data to make predictions on (pd.DataFrame or file path)
    # Must contain the same feature columns used during training
    model="LightGBM",
    # Optional: specify which trained model to use
    # If None → AutoGluon automatically uses the best validation model
    as_pandas=True,
    # True  → return predictions as pandas Series
    # False → return predictions as numpy ndarray
    transform_features=True,
    # True  → AutoGluon performs feature preprocessing automatically
    # False → skip preprocessing if you already ran:
    #         predictor.transform_features(data)
    inverse_transform=True,
    # True  → convert predictions back to original label format
    # Example:
    # If labels were internally encoded (0,1,2),
    # this returns original labels ("cat","dog","bird")
    decision_threshold=0.5,
    # Only used for binary classification
    # If predicted probability > 0.5 → class = 1
    # If predicted probability <= 0.5 → class = 0
)


##########################################################
# predictor.predict_proba() — **PROBABILITY INFERENCE**
##########################################################

probabilities = predictor.predict_proba(
    data=test_df,
    # Data to make predictions on (pd.DataFrame or file path)
    # Must contain the same feature columns used during training
    model="LightGBM",
    # Optional: specify which trained model to use
    # If None → AutoGluon automatically uses the best validation model
    as_pandas=True,
    # True  → return predictions as pandas DataFrame
    # False → return predictions as numpy ndarray
    as_multiclass=True,
    # For binary classification:
    # True  → return two columns (probability of class 0 and class 1)
    # False → return only probability of positive class
    transform_features=True,
    # True  → AutoGluon preprocesses features automatically
    # False → skip preprocessing if already done with:
    # predictor.transform_features(data)
)


###################################################
# predictor.leaderboard() — **MODEL RANKINGS**
###################################################

leaderboard_df = predictor.leaderboard(
    data=test_df,
    # Dataset used to evaluate models
    # Must include the label column used during training
    # If None → only validation scores are shown
    extra_info=True,
    # True → show advanced model information
    # Example: memory usage, features used, model type
    extra_metrics=["accuracy", "roc_auc", "f1"],
    # Additional metrics to compute on the provided dataset
    # Only works when data is provided
    decision_threshold=0.5,
    # Threshold used to convert probabilities to labels
    # Only relevant for binary classification
    score_format="score",
    # "score" → higher is better
    # "error" → converts score to metric error (lower is better)
    only_pareto_frontier=False,
    # True → show only best accuracy vs inference-time tradeoff models
    skip_score=False,
    # True → skip computing test score (faster evaluation)
    refit_full=None,
    # None → show all models
    # True → show only refit models (_FULL)
    # False → show only original models
    set_refit_score_to_parent=False,
    # If True → refit model uses parent's validation score
    display=True,
    # True → print leaderboard directly
)


#########################################################################
# predictor.feature_importance() — **PERMUTATION FEATURE IMPORTANCE**
#########################################################################

feature_importance_df = predictor.feature_importance(
    data=val_df,
    # Dataset used to compute feature importance
    # Must include the label column used during training
    # Recommended: use validation or test data (not training data)
    model=None,
    # Model to compute feature importance for
    # None → automatically selects the best model
    # Example: "LightGBM", "CatBoost", "WeightedEnsemble_L2"
    features=None,
    # List of feature names to compute importance for
    # None → compute importance for ALL features
    # Example: ["age", "income", "education"]
    feature_stage="original",
    # Stage of features to analyze
    # Options:
    # "original" → raw input features
    # "transformed" → after AutoGluon feature engineering
    # "transformed_model" → features used internally by model
    subsample_size=5000,
    # Number of rows sampled for computing importance
    # Larger value → more accurate but slower
    # None → use entire dataset
    time_limit=None,
    # Maximum time allowed for computing feature importance (seconds)
    # None → no time limit
    num_shuffle_sets=None,
    # Number of permutations per feature
    # Higher value → more reliable importance
    # Default:
    # 5 if time_limit=None
    # 10 if time_limit is specified
    include_confidence_band=True,
    # True → include confidence interval columns in result
    confidence_level=0.99,
    # Confidence level for importance interval
    # Example: 0.95 → p95_high, p95_low
    # Example: 0.99 → p99_high, p99_low
    silent=False,
    # True → suppress logging output
)


#########################################################
# predictor.fit_extra() — **Add More Models Post-Fit**
#########################################################

predictor.fit_extra(
    hyperparameters={"GBM": {}, "CAT": {}, "XGB": {}},
    # Models to add after the original fit()
    # Can be string presets like "default", "light", "zeroshot"
    # Or a dictionary specifying model types
    time_limit=1800,
    # Maximum training time in seconds for these additional models
    base_model_names=["LightGBM", "CatBoost", "XGBoost"],
    # Existing trained models used as base models for stacking
    # Their predictions become additional features
    fit_weighted_ensemble=True,
    # Train a WeightedEnsembleModel for each stacking layer
    fit_full_last_level_weighted_ensemble=True,
    # Final ensemble will combine ALL successful models
    # across previous layers
    full_weighted_ensemble_additionally=False,
    # If True → train an additional ensemble after stacking
    num_cpus="auto",
    # Number of CPU cores used
    # "auto" → AutoGluon decides based on system
    num_gpus="auto",
    # Number of GPUs used
    # "auto" → automatically detect available GPUs
    fit_strategy="auto",
    # Training strategy
    # "auto" → same as original fit()
    # "sequential" → models trained one-by-one
    # "parallel" → models trained in parallel (Ray)
    memory_limit="auto",
    # Soft RAM limit in GB
    # "auto" → use all available memory
    ag_args_fit=None,
    # Advanced model fit arguments
    # Example:
    # {"max_memory_usage_ratio": 0.8}
    ag_args_ensemble=None,
    # Arguments for ensemble models
    # Example:
    # {"use_orig_features": True}
    pseudo_data=None,
    # Optional self-labeled dataset added to training
    # Used for semi-supervised learning
)


###################################################
# predictor.fit_summary() — **TRAINING SUMMARY**
###################################################

summary = predictor.fit_summary(
    verbosity=3,
    # Controls the level of detail printed in the summary
    # Options:
    # 0 → no output
    # 1 → high-level summary only
    # 2 → summary + generate plots
    # 3 → full detailed information about models
    show_plot=False,
    # If True → opens the model summary plot in a browser
    # Only works when verbosity > 1
)


###################################################
# Supported Model Keys for **HYPERPARAMETERS**
###################################################

hyperparameters = {
    "GBM": {},  # LightGBM
    "CAT": {},  # CatBoost
    "XGB": {},  # XGBoost
    "RF": {},  # Random Forest
    "XT": {},  # Extra Trees
    "KNN": {},  # K-Nearest Neighbors
    "LR": {},  # Linear / Logistic Regression
    "NN_TORCH": {},  # PyTorch Neural Network
    "FASTAI": {},  # FastAI Neural Network
    "TABM": {},  # TabM
    "REALMLP": {},  # RealMLP
    "EBM": {},  # Explainable Boosting Machine
    "AG_AUTOMM": {},  # MultimodalPredictor (requires GPU)
    "TABPFNV2": {},  # TabPFN v2 (requires GPU)
    "TABICL": {},  # TabICL (requires GPU)
}
