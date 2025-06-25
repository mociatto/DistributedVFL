"""
Centralized Configuration for HybridVFL Project
Consolidates all hyperparameters and settings for better maintainability.
"""

import os

# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================

PROJECT_NAME = "HybridVFL"
VERSION = "4.0.0"  # Post-architectural improvements

# =============================================================================
# DATA CONFIGURATION  
# =============================================================================

DATA_CONFIG = {
    # Dataset settings
    'dataset_name': 'HAM10000',
    'data_percentage': 0.05,
    'num_classes': 7,
    'class_names': ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'],
    
    # Data paths
    'image_dir_1': 'data/HAM10000_images_part_1',
    'image_dir_2': 'data/HAM10000_images_part_2', 
    'metadata_file': 'data/HAM10000_metadata.csv',
    
    # Data splits
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    
    # Image preprocessing
    'image_size': (224, 224),
    'normalize': True,
    'augmentation': True
}

# =============================================================================
# MODEL ARCHITECTURE CONFIGURATION
# =============================================================================

MODEL_CONFIG = {
    # Embedding dimensions
    'embedding_dim': 256,
    'image_embedding_dim': 256,
    'tabular_embedding_dim': 256,
    
    # Image encoder (Step 3 enhanced)
    'image_encoder': {
        'input_shape': (224, 224, 3),
        'base_filters': 32,
        'max_filters': 128,
        'dropout_rate': 0.3,
        'advanced_dropout_rate': 0.2,  # Step 3 feature
        'l2_reg': 0.001,
        'use_batch_norm': True,
        'use_residual': True,
        'activation': 'relu'
    },
    
    # Tabular encoder (Step 3 enhanced)
    'tabular_encoder': {
        'hidden_units': [512, 256, 256],
        'dropout_rate': 0.3,
        'advanced_dropout_rate': 0.2,  # Step 3 feature  
        'l2_reg': 0.001,
        'use_batch_norm': True,
        'use_residual': True,
        'activation': 'relu'
    },
    
    # Fusion model (OPTIMIZED for better learning)
    'fusion_model': {
        'attention_heads': 4,  # Reduced from 8
        'attention_dim': 128,  # Reduced from 256
        'hidden_units': [256, 128, 64],  # Restored from simplified version
        'dropout_rate': 0.3,  # Reduced from 0.7 - was too aggressive
        'advanced_dropout_rate': 0.2,  # Reduced from 0.6 - was too aggressive
        'l2_reg': 0.001,  # Reduced from 0.05 - was too aggressive
        'use_cross_attention': True,  # Re-enabled for better fusion
        'use_ensemble': False,  # Keep disabled to reduce complexity
        'ensemble_size': 1,
        'activation': 'relu'
    }
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

TRAINING_CONFIG = {
    # VFL training settings
    'total_rounds': 5,  # Increased from 2 to 5 for better training
    'epochs_per_round': 12,  # Increased from 8 to 12 for stronger learning
    'batch_size': 16,
    'learning_rate': 0.002,  # Increased from 0.001 for faster learning
    
    # Client training settings
    'client_epochs': {
        'image': 20,  # Increased for stronger client learning
        'tabular': 20  # Increased for stronger client learning
    },
    'client_batch_size': 32,
    'client_learning_rate': 0.002,  # Increased for faster client learning
    
    # Optimization settings
    'optimizer': 'adam',
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-7,
    'gradient_clip_norm': 2.0,  # Increased for more aggressive training
    
    # Learning rate scheduling
    'use_lr_scheduling': True,
    'lr_schedule_type': 'step',  # 'step', 'exponential', 'cosine'
    'lr_decay_factor': 0.7,  # Less aggressive decay
    'lr_decay_epochs': [8, 15],  # Later decay for more aggressive early training
    
    # Early stopping (More aggressive for faster iteration)
    'use_early_stopping': True,
    'patience': 8,  # Increased from 5 to 8 - allow more aggressive training
    'min_delta': 0.002,  # Keep sensitive to improvement
    'restore_best_weights': True,
    
    # Progressive training strategy
    'use_progressive_training': True,
    'start_simple': True,  # Start with simple architecture
    'complexity_schedule': [0.3, 0.6, 1.0]  # Gradually increase complexity
}

# =============================================================================
# LOSS CONFIGURATION (Step 2 Enhancement)
# =============================================================================

LOSS_CONFIG = {
    # Contrastive learning (NT-Xent)
    'use_contrastive_loss': True,
    'contrastive_temperature': 0.3,  # Lower temperature for sharper distributions
    'contrastive_weight': 0.4,  # Increased for stronger representation learning
    'classification_weight': 0.6,  # Balanced with contrastive loss
    
    # Class balancing
    'use_class_weights': True,
    'class_weight_method': 'balanced',  # 'balanced', 'inverse', 'custom'
    
    # Label smoothing
    'use_label_smoothing': True,
    'label_smoothing_factor': 0.15,  # Increased for better generalization
    
    # Regularization (AGGRESSIVE for better defense)
    'l2_lambda': 0.002,  # Increased from 0.001 for stronger regularization
    'use_mixup': True,  # Step 3 feature
    'mixup_alpha': 0.3,  # Increased from 0.2 for stronger augmentation
    'mixup_start_epoch': 1  # Start immediately for aggressive training
}

# =============================================================================
# GENERALIZATION CONFIGURATION (Step 3 Enhancement)
# =============================================================================

GENERALIZATION_CONFIG = {
    # Noise injection (AGGRESSIVE)
    'use_noise_injection': True,
    'noise_stddev': 0.15,  # Increased from 0.1 for stronger noise
    'noise_probability': 0.4,  # Increased from 0.3 for more frequent noise
    
    # Advanced dropout (AGGRESSIVE)
    'use_advanced_dropout': True,
    'spatial_dropout_rate': 0.3,  # Increased from 0.2
    'gaussian_dropout_rate': 0.15,  # Increased from 0.1
    'alpha_dropout_rate': 0.25,  # Increased from 0.2
    
    # Data augmentation (VERY AGGRESSIVE)
    'use_strong_augmentation': True,
    'augmentation_probability': 0.95,  # Increased from 0.9 - nearly always augment
    'rotation_range': 45,  # Increased from 30 for stronger augmentation
    'zoom_range': 0.3,  # Increased from 0.2
    'horizontal_flip': True,
    'vertical_flip': True,
    'brightness_range': [0.7, 1.3],  # Wider range for stronger augmentation
    'shear_range': 0.15,  # Increased from 0.1
    
    # Cross-validation
    'use_cross_validation': True,  # Enable for better generalization
    'cv_folds': 3,  # Reduced to 3 for faster training
    
    # Ensemble methods
    'use_ensemble': True,
    'ensemble_diversity_loss': 0.02  # Increased for stronger diversity
}

# =============================================================================
# PRIVACY CONFIGURATION (AGGRESSIVE ADVERSARIAL TRAINING)
# =============================================================================

PRIVACY_CONFIG = {
    # Adversarial privacy mechanism (ENABLED AND AGGRESSIVE)
    'adversarial_lambda': 0.0,  # Still starts at 0.0 but dashboard can control
    'max_adversarial_lambda': 1.0,  # Maximum lambda for WILD defense
    'adversarial_warmup_rounds': 1,  # Start adversarial training after 1 round
    'use_differential_privacy': False,
    'dp_noise_multiplier': 1.0,
    'dp_l2_norm_clip': 1.0,
    
    # WILD adversarial defense settings
    'wild_defense_enabled': True,  # Enable WILD defense mode
    'wild_perturbation_steps': 8,  # Multi-step perturbations
    'wild_step_size_multiplier': 0.8,  # Aggressive step size
    'wild_noise_multipliers': {
        'gaussian': 0.6,  # Strong Gaussian noise
        'uniform': 0.8,   # Strong uniform noise
        'structured': 1.2,  # Very strong structured noise
        'targeted': 1.5    # Extremely strong targeted noise
    },
    'wild_feature_corruption_ratio': 0.5,  # Corrupt 50% of features
    'wild_chunk_strategies': 4,  # Use 4 different noise strategies
    
    # Secure aggregation
    'use_secure_aggregation': False,
    'encryption_key_size': 2048
}

# =============================================================================
# FEDERATED LEARNING CONFIGURATION (AGGRESSIVE True FL Implementation)
# =============================================================================

FEDERATED_LEARNING_CONFIG = {
    # FL Training Strategy
    'strategy': 'fusion_guided_updates',  # 'fusion_guided_updates', 'federated_avg', 'embeddings_only'
    'enable_true_fl': True,  # Enable iterative client retraining per round
    
    # FL Round Configuration (AGGRESSIVE)
    'total_fl_rounds': 5,  # Keep at 5 for good training
    'client_epochs_per_round': 5,  # Increased from 3 to 5 for stronger client learning
    'collect_fresh_embeddings': True,  # Generate new embeddings each round
    
    # Fusion-Guided Updates Settings (AGGRESSIVE)
    'send_embedding_gradients': True,  # Send gradients w.r.t embeddings to clients
    'send_attention_weights': True,  # Send fusion attention weights as guidance
    'gradient_scaling_factor': 0.2,  # Increased from 0.1 for stronger guidance
    'attention_weight_threshold': 0.005,  # Reduced threshold for more guidance
    
    # Client Update Strategy (AGGRESSIVE)
    'client_learning_rate_multiplier': 0.7,  # Increased from 0.5 for faster adaptation
    'use_guided_training': True,  # Apply server guidance during client training
    'guidance_weight': 0.4,  # Increased from 0.3 for stronger guidance
    
    # Performance Tracking
    'track_round_improvements': True,  # Track accuracy improvement per round
    'min_round_improvement': 0.001,  # Reduced from 0.002 for more sensitivity
    'convergence_patience': 4,  # Increased from 3 for more aggressive training
    
    # Communication Settings
    'max_gradient_norm': 2.0,  # Increased from 1.0 for stronger gradients
    'compress_communications': False,  # Compress large tensors (future feature)
    
    # Debugging and Monitoring
    'save_round_models': True,  # Save server fusion model each round
    'log_client_performance': True,  # Log individual client metrics
    'visualize_attention_evolution': True  # Track attention weight changes
}

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

SYSTEM_CONFIG = {
    # Paths
    'results_dir': 'results',
    'models_dir': 'models', 
    'logs_dir': 'logs',
    'plots_dir': 'plots',
    'embeddings_dir': 'embeddings',
    'communication_dir': 'communication',
    'status_dir': 'status',
    
    # Logging
    'log_level': 'INFO',
    'save_plots': True,
    'save_embeddings': True,
    'save_intermediate_models': True,
    
    # Performance monitoring
    'track_gpu_usage': True,
    'track_memory_usage': True,
    'profiling_enabled': False,
    
    # Reproducibility
    'random_seed': 42,
    'deterministic_training': True
}

# =============================================================================
# PHASE CONFIGURATION
# =============================================================================

PHASE_CONFIG = {
    'current_phase': 3,  # Phase 3: Advanced Generalization & Robustness
    'phase_name': 'Advanced Generalization & Robustness',
    'step': 3,  # Step 3 within Phase 3
    'step_name': 'Advanced Generalization & Robustness',
    
    # Phase-specific features
    'phase_features': {
        1: ['basic_vfl', 'simple_fusion'],
        2: ['transformer_fusion', 'ensemble', 'enhanced_regularization'], 
        3: ['noise_injection', 'advanced_dropout', 'mixup', 'early_stopping']
    }
}

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

EVALUATION_CONFIG = {
    # Metrics to compute
    'metrics': ['accuracy', 'f1_macro', 'f1_weighted', 'precision', 'recall', 'auc'],
    
    # Evaluation settings
    'evaluate_every_round': True,
    'save_predictions': True,
    'save_confusion_matrix': True,
    'save_classification_report': True,
    
    # Performance thresholds
    'target_accuracy': 0.70,  # 70% target accuracy
    'min_improvement': 0.01,  # 1% minimum improvement
    'convergence_patience': 5
}

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_config():
    """Validate configuration settings for consistency."""
    errors = []
    
    # Check data splits sum to 1.0
    splits_sum = (DATA_CONFIG['train_ratio'] + 
                  DATA_CONFIG['val_ratio'] + 
                  DATA_CONFIG['test_ratio'])
    if abs(splits_sum - 1.0) > 1e-6:
        errors.append(f"Data splits sum to {splits_sum}, should be 1.0")
    
    # Check loss weights sum appropriately
    loss_weights_sum = (LOSS_CONFIG['classification_weight'] + 
                       LOSS_CONFIG['contrastive_weight'])
    if abs(loss_weights_sum - 1.0) > 1e-6:
        errors.append(f"Loss weights sum to {loss_weights_sum}, should be 1.0")
    
    # Check embedding dimensions match
    if (MODEL_CONFIG['image_embedding_dim'] != MODEL_CONFIG['embedding_dim'] or
        MODEL_CONFIG['tabular_embedding_dim'] != MODEL_CONFIG['embedding_dim']):
        errors.append("Embedding dimensions must match between modalities")
    
    # Check phase consistency
    if PHASE_CONFIG['current_phase'] not in PHASE_CONFIG['phase_features']:
        errors.append(f"Current phase {PHASE_CONFIG['current_phase']} not defined")
    
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))
    
    return True

# =============================================================================
# CONFIGURATION ACCESS FUNCTIONS
# =============================================================================

def get_config(section=None):
    """Get configuration section or all configs."""
    configs = {
        'data': DATA_CONFIG,
        'model': MODEL_CONFIG, 
        'training': TRAINING_CONFIG,
        'loss': LOSS_CONFIG,
        'generalization': GENERALIZATION_CONFIG,
        'privacy': PRIVACY_CONFIG,
        'federated_learning': FEDERATED_LEARNING_CONFIG,
        'system': SYSTEM_CONFIG,
        'phase': PHASE_CONFIG,
        'evaluation': EVALUATION_CONFIG
    }
    
    if section:
        return configs.get(section, {})
    return configs

def get_current_phase_config():
    """Get configuration for current phase."""
    return {
        'phase': PHASE_CONFIG['current_phase'],
        'step': PHASE_CONFIG['step'],
        'features': PHASE_CONFIG['phase_features'][PHASE_CONFIG['current_phase']],
        'name': PHASE_CONFIG['phase_name']
    }

def update_config(section, key, value):
    """Update a configuration value."""
    configs = get_config()
    if section in configs and key in configs[section]:
        configs[section][key] = value
        return True
    return False

# =============================================================================
# INITIALIZATION
# =============================================================================

# Validate configuration on import
if __name__ == "__main__":
    validate_config()
    print("Configuration validation passed")
    print(f"Current Phase: {PHASE_CONFIG['current_phase']} - {PHASE_CONFIG['phase_name']}")
    print(f"Step: {PHASE_CONFIG['step']} - {PHASE_CONFIG['step_name']}")
else:
    validate_config() 