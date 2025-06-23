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
    'data_percentage': 0.2,  # Use 10% of data for faster experimentation
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
    
    # Fusion model (SIMPLIFIED for better generalization)
    'fusion_model': {
        'attention_heads': 4,  # Reduced from 8
        'attention_dim': 128,  # Reduced from 256
        'hidden_units': [256, 64],  # Simplified from [512, 256, 128]
        'dropout_rate': 0.7,  # Increased to 0.7
        'advanced_dropout_rate': 0.6,  # Increased to 0.6
        'l2_reg': 0.05,  # Increased to 0.05
        'use_cross_attention': False,  # Disabled for simplicity
        'use_ensemble': False,  # Disabled to reduce complexity
        'ensemble_size': 1,
        'activation': 'relu'
    }
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

TRAINING_CONFIG = {
    # VFL training settings
    'total_rounds': 3,
    'epochs_per_round': 8,
    'batch_size': 16,
    'learning_rate': 0.0005,  # Reduced by 50% for better generalization
    
    # Client training settings
    'client_epochs': {
        'image': 15,  # Step 2 & 3 enhanced
        'tabular': 15  # Step 2 & 3 enhanced
    },
    'client_batch_size': 32,
    'client_learning_rate': 0.001,
    
    # Optimization settings
    'optimizer': 'adam',
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-7,
    'gradient_clip_norm': 1.0,
    
    # Learning rate scheduling
    'use_lr_scheduling': True,
    'lr_schedule_type': 'step',  # 'step', 'exponential', 'cosine'
    'lr_decay_factor': 0.5,
    'lr_decay_epochs': [5, 10],
    
    # Early stopping (VERY aggressive)
    'use_early_stopping': True,
    'patience': 1,  # VERY aggressive - stop after 1 epoch without improvement
    'min_delta': 0.01,  # High threshold for meaningful improvement
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
    'contrastive_temperature': 0.5,
    'contrastive_weight': 0.3,  # (1 - alpha) where alpha=0.7 for classification
    'classification_weight': 0.7,  # alpha for classification loss
    
    # Class balancing
    'use_class_weights': True,
    'class_weight_method': 'balanced',  # 'balanced', 'inverse', 'custom'
    
    # Label smoothing
    'use_label_smoothing': True,
    'label_smoothing_factor': 0.1,
    
    # Regularization (MUCH stronger)
    'l2_lambda': 0.05,  # Increased 50x for stronger regularization
    'use_mixup': True,  # Step 3 feature
    'mixup_alpha': 0.4,  # Increased for stronger augmentation
    'mixup_start_epoch': 1  # Start immediately
}

# =============================================================================
# GENERALIZATION CONFIGURATION (Step 3 Enhancement)
# =============================================================================

GENERALIZATION_CONFIG = {
    # Noise injection
    'use_noise_injection': True,
    'noise_stddev': 0.1,
    'noise_probability': 0.3,
    
    # Advanced dropout
    'use_advanced_dropout': True,
    'spatial_dropout_rate': 0.2,
    'gaussian_dropout_rate': 0.1,
    'alpha_dropout_rate': 0.2,
    
    # Data augmentation (enhanced)
    'use_strong_augmentation': True,
    'augmentation_probability': 0.9,  # Increased from 0.8
    'rotation_range': 30,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'vertical_flip': True,
    'brightness_range': [0.8, 1.2],
    'shear_range': 0.1,
    
    # Cross-validation
    'use_cross_validation': True,  # Enable for better generalization
    'cv_folds': 3,  # Reduced to 3 for faster training
    
    # Ensemble methods
    'use_ensemble': True,
    'ensemble_diversity_loss': 0.01
}

# =============================================================================
# PRIVACY CONFIGURATION
# =============================================================================

PRIVACY_CONFIG = {
    # Adversarial privacy mechanism
    'adversarial_lambda': 0.0,  # Disabled for Phase 3 (high performance focus)
    'use_differential_privacy': False,
    'dp_noise_multiplier': 1.0,
    'dp_l2_norm_clip': 1.0,
    
    # Secure aggregation
    'use_secure_aggregation': False,
    'encryption_key_size': 2048
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
    print("✅ Configuration validation passed")
    print(f"📊 Current Phase: {PHASE_CONFIG['current_phase']} - {PHASE_CONFIG['phase_name']}")
    print(f"📈 Step: {PHASE_CONFIG['step']} - {PHASE_CONFIG['step_name']}")
else:
    validate_config() 