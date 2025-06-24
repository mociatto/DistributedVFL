"""
Core training and evaluation logic for multimodal vertical federated learning.
Handles both individual client training and federated server training with fusion.
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score, confusion_matrix, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import os


def compute_class_weights(labels, method='balanced'):
    """
    Compute class weights to handle imbalanced datasets with improved strategies.
    
    Args:
        labels (np.ndarray): Training labels
        method (str): Method for computing weights ('balanced', 'sqrt_balanced', 'uniform')
    
    Returns:
        dict: Class weights dictionary
    """
    unique_classes = np.unique(labels)
    class_counts = np.bincount(labels)
    
    if method == 'balanced':
        # Standard balanced approach - works well for moderately balanced data
        raw_weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
    elif method == 'sqrt_balanced':
        raw_weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
        # Apply square root to moderate extreme weights
        raw_weights = np.sqrt(raw_weights)
    elif method == 'uniform':
        # Equal weights for all classes - good for balanced datasets
        raw_weights = np.ones(len(unique_classes))
    else:
        # Default to balanced
        raw_weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
    
    # Cap extreme weights to prevent training instability
    max_weight = 5.0  # Prevent any weight from being too extreme
    raw_weights = np.clip(raw_weights, 0.1, max_weight)
    
    # Create class weight dictionary
    class_weights = {int(cls): float(weight) for cls, weight in zip(unique_classes, raw_weights)}
    
    return class_weights


def create_image_data_generator(image_paths, labels, batch_size=32, 
                               target_size=(224, 224), augment=True, shuffle=True):
    """
    Create data generator for image data.
    
    Args:
        image_paths (np.ndarray): Array of image paths
        labels (np.ndarray): Corresponding labels
        batch_size (int): Batch size
        target_size (tuple): Target image size
        augment (bool): Whether to apply augmentation
        shuffle (bool): Whether to shuffle data
    
    Yields:
        tuple: (batch_images, batch_labels)
    """
    from data_loader import load_and_preprocess_image
    
    indices = np.arange(len(image_paths))
    
    while True:
        if shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            
            batch_images = []
            batch_labels = []
            
            for idx in batch_indices:
                img = load_and_preprocess_image(
                    image_paths[idx], 
                    target_size=target_size, 
                    augment=augment,
                    use_efficientnet_preprocessing=True  # PHASE 2: Enable EfficientNet preprocessing
                )
                batch_images.append(img)
                batch_labels.append(labels[idx])
            
            yield np.array(batch_images), np.array(batch_labels)


def create_tabular_data_generator(features, labels, batch_size=32, shuffle=True):
    """
    Create data generator for tabular data.
    
    Args:
        features (np.ndarray): Feature array
        labels (np.ndarray): Corresponding labels
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
    
    Yields:
        tuple: (batch_features, batch_labels)
    """
    indices = np.arange(len(features))
    
    while True:
        if shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            yield features[batch_indices], labels[batch_indices]


def train_client_model(model, train_generator, val_data, epochs=10, steps_per_epoch=None,
                      class_weights=None, patience=3, verbose=1):
    """
    Train a client model (image or tabular encoder).
    
    Args:
        model: Model to train
        train_generator: Training data generator
        val_data: Validation data (X, y)
        epochs (int): Number of epochs
        steps_per_epoch (int): Steps per epoch (None for auto-calculation)
        class_weights (dict): Class weights for imbalanced data
        patience (int): Early stopping patience
        verbose (int): Verbosity level
    
    Returns:
        dict: Training history and metrics
    """
    # Callbacks
    callbacks = []
    
    if patience > 0:
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            verbose=verbose
        )
        callbacks.append(early_stopping)
    
    # Reduce learning rate on plateau
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=max(1, patience // 2),
        min_lr=1e-7,
        verbose=verbose
    )
    callbacks.append(lr_reducer)
    
    # Progress callback for better tracking
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            print(f"      🔄 Epoch {epoch + 1}/{epochs} starting...")
        
        def on_epoch_end(self, epoch, logs=None):
            if logs:
                acc = logs.get('accuracy', 0)
                val_acc = logs.get('val_accuracy', 0)
                loss = logs.get('loss', 0)
                val_loss = logs.get('val_loss', 0)
                print(f"      ✅ Epoch {epoch + 1}/{epochs}: acc={acc:.4f}, val_acc={val_acc:.4f}, loss={loss:.4f}, val_loss={val_loss:.4f}")
    
    if verbose > 1:
        callbacks.append(ProgressCallback())
    
    # Training
    val_x, val_y = val_data
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=(val_x, val_y),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=verbose
    )
    
    return history


def evaluate_client_model(model, test_data, class_names=None, verbose=1):
    """
    Evaluate a client model and return detailed metrics.
    
    Args:
        model: Model to evaluate
        test_data: Test data (X, y)
        class_names (list): Class names for reporting
        verbose (int): Verbosity level
    
    Returns:
        dict: Evaluation metrics
    """
    test_x, test_y = test_data
    
    # Predictions
    predictions = model.predict(test_x, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    
    # Metrics
    accuracy = np.mean(pred_classes == test_y)
    f1_macro = f1_score(test_y, pred_classes, average='macro')
    f1_weighted = f1_score(test_y, pred_classes, average='weighted')
    
    if verbose > 0:
        print(f"\nEvaluation Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        print(f"  F1 (weighted): {f1_weighted:.4f}")
        
        if class_names is not None:
            print(f"\nDetailed Classification Report:")
            print(classification_report(test_y, pred_classes, target_names=class_names))
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'predictions': predictions,
        'pred_classes': pred_classes,
        'true_classes': test_y
    }


def extract_embeddings(encoder, data, batch_size=32):
    """
    Extract embeddings from an encoder model.
    
    Args:
        encoder: Encoder model
        data: Input data
        batch_size (int): Batch size for inference
    
    Returns:
        np.ndarray: Extracted embeddings
    """
    embeddings = []
    
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i + batch_size]
        batch_embeddings = encoder.predict(batch_data, verbose=0)
        embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)


def train_fusion_model_with_adversarial(fusion_model, adversarial_model,
                                       image_embeddings, tabular_embeddings, 
                                       labels, sensitive_attrs,
                                       val_image_embeddings, val_tabular_embeddings,
                                       val_labels, val_sensitive_attrs,
                                       adversarial_lambda=0.0, epochs=10,
                                       batch_size=32, verbose=1):
    """
    Train the fusion model with optional adversarial training.
    
    Args:
        fusion_model: Main classification model
        adversarial_model: Adversarial model (can be None)
        image_embeddings: Training image embeddings
        tabular_embeddings: Training tabular embeddings
        labels: Training labels
        sensitive_attrs: Sensitive attributes (sex, age_bin)
        val_image_embeddings: Validation image embeddings
        val_tabular_embeddings: Validation tabular embeddings
        val_labels: Validation labels
        val_sensitive_attrs: Validation sensitive attributes
        adversarial_lambda (float): Adversarial loss weight
        epochs (int): Number of epochs
        batch_size (int): Batch size
        verbose (int): Verbosity level
    
    Returns:
        dict: Training history
    """
    if adversarial_lambda > 0 and adversarial_model is not None:
        return _train_with_adversarial(
            fusion_model, adversarial_model,
            image_embeddings, tabular_embeddings, labels, sensitive_attrs,
            val_image_embeddings, val_tabular_embeddings, val_labels, val_sensitive_attrs,
            adversarial_lambda, epochs, batch_size, verbose
        )
    else:
        return _train_without_adversarial(
            fusion_model,
            image_embeddings, tabular_embeddings, labels,
            val_image_embeddings, val_tabular_embeddings, val_labels,
            epochs, batch_size, verbose
        )


def _train_without_adversarial(fusion_model, image_embeddings, tabular_embeddings, labels,
                              val_image_embeddings, val_tabular_embeddings, val_labels,
                              epochs, batch_size, verbose):
    """Train fusion model without adversarial component."""
    # Compute class weights
    class_weights = compute_class_weights(labels, method='sqrt_balanced')
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=verbose
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=verbose
        )
    ]
    
    # Train
    history = fusion_model.fit(
        [image_embeddings, tabular_embeddings],
        labels,
        validation_data=([val_image_embeddings, val_tabular_embeddings], val_labels),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=verbose
    )
    
    return history


def _train_with_adversarial(fusion_model, adversarial_model,
                           image_embeddings, tabular_embeddings, labels, sensitive_attrs,
                           val_image_embeddings, val_tabular_embeddings, val_labels, val_sensitive_attrs,
                           adversarial_lambda, epochs, batch_size, verbose):
    """Train fusion model with adversarial component."""
    # This is a simplified version - in practice, you'd need alternating training
    # For now, we'll just train the main model since adversarial_lambda is 0 in Phase 1
    
    print(f"Warning: Adversarial training with lambda={adversarial_lambda} not fully implemented.")
    print("Training main fusion model only.")
    
    return _train_without_adversarial(
        fusion_model, image_embeddings, tabular_embeddings, labels,
        val_image_embeddings, val_tabular_embeddings, val_labels,
        epochs, batch_size, verbose
    )


def evaluate_fusion_model(fusion_model, image_embeddings, tabular_embeddings, labels,
                         class_names=None, save_confusion_matrix=False, verbose=1):
    """
    Evaluate the fusion model.
    
    Args:
        fusion_model: Fusion model to evaluate
        image_embeddings: Image embeddings
        tabular_embeddings: Tabular embeddings
        labels: True labels
        class_names (list): Class names
        save_confusion_matrix (bool): Whether to save confusion matrix plot
        verbose (int): Verbosity level
    
    Returns:
        dict: Evaluation metrics
    """
    # Predictions
    predictions = fusion_model.predict([image_embeddings, tabular_embeddings], verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    
    # Metrics
    accuracy = np.mean(pred_classes == labels)
    balanced_accuracy = balanced_accuracy_score(labels, pred_classes)
    f1_macro = f1_score(labels, pred_classes, average='macro')
    f1_weighted = f1_score(labels, pred_classes, average='weighted')
    
    if verbose > 0:
        print(f"\nFusion Model Evaluation:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Balanced Accuracy: {balanced_accuracy:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        print(f"  F1 (weighted): {f1_weighted:.4f}")
        
        if class_names is not None:
            print(f"\nDetailed Classification Report:")
            print(classification_report(labels, pred_classes, target_names=class_names))
    
    # Confusion matrix
    if save_confusion_matrix and class_names is not None:
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-GUI backend for thread safety
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            cm = confusion_matrix(labels, pred_classes)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix - Fusion Model')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('confusion_matrix_fusion.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            if verbose > 0:
                print("Confusion matrix saved as 'confusion_matrix_fusion.png'")
        except Exception as e:
            if verbose > 0:
                print(f"⚠️ Could not save confusion matrix: {e}")
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'predictions': predictions,
        'pred_classes': pred_classes,
        'true_classes': labels
    }


def save_training_plots(history, filename_prefix='training'):
    """
    Save training plots for loss and accuracy.
    
    Args:
        history: Training history from model.fit()
        filename_prefix (str): Prefix for saved files
    """
    if hasattr(history, 'history'):
        history = history.history
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(epochs, history['loss'], 'b-', label='Training Loss')
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy')
    if 'val_accuracy' in history:
        ax2.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_history.png', dpi=300, bbox_inches='tight')
    plt.close()


def check_validation_test_consistency(val_accuracy, test_accuracy, threshold=0.15):
    """
    Check if validation and test accuracies are consistent.
    Large gaps indicate overfitting to validation set.
    
    Args:
        val_accuracy: Validation accuracy
        test_accuracy: Test accuracy  
        threshold: Maximum acceptable gap (default 15%)
    
    Returns:
        dict: Analysis results with warnings and recommendations
    """
    gap = abs(val_accuracy - test_accuracy)
    gap_percentage = gap * 100
    
    analysis = {
        'validation_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'gap': gap,
        'gap_percentage': gap_percentage,
        'is_overfitted': gap > threshold,
        'severity': 'low'
    }
    
    # Determine severity
    if gap > 0.4:  # 40%+ gap
        analysis['severity'] = 'critical'
        analysis['recommendation'] = 'Severe overfitting detected. Increase regularization dramatically.'
    elif gap > 0.25:  # 25%+ gap
        analysis['severity'] = 'high'
        analysis['recommendation'] = 'High overfitting. Increase dropout, L2 reg, and reduce model complexity.'
    elif gap > threshold:  # 15%+ gap
        analysis['severity'] = 'moderate'
        analysis['recommendation'] = 'Moderate overfitting. Consider stronger regularization.'
    else:
        analysis['severity'] = 'low'
        analysis['recommendation'] = 'Good generalization. Model is well-regularized.'
    
    return analysis


def suggest_regularization_improvements(gap_percentage):
    """
    Suggest specific regularization improvements based on val-test gap.
    
    Args:
        gap_percentage: Validation-test accuracy gap as percentage
    
    Returns:
        list: Specific improvement suggestions
    """
    suggestions = []
    
    if gap_percentage > 40:
        suggestions.extend([
            "🔥 CRITICAL: Increase dropout to 0.7-0.8",
            "🔥 CRITICAL: Increase L2 regularization to 0.1",
            "🔥 CRITICAL: Reduce model complexity (fewer layers/units)",
            "🔥 CRITICAL: Use stronger data augmentation",
            "🔥 CRITICAL: Implement early stopping with patience=1"
        ])
    elif gap_percentage > 25:
        suggestions.extend([
            "⚠️  HIGH: Increase dropout to 0.6-0.7",
            "⚠️  HIGH: Increase L2 regularization to 0.01-0.05",
            "⚠️  HIGH: Use mixup augmentation from epoch 1",
            "⚠️  HIGH: Reduce learning rate by 50%"
        ])
    elif gap_percentage > 15:
        suggestions.extend([
            "📊 MODERATE: Increase dropout to 0.5-0.6",
            "📊 MODERATE: Increase L2 regularization to 0.005-0.01",
            "📊 MODERATE: Use label smoothing",
            "📊 MODERATE: Implement gradient clipping"
        ])
    else:
        suggestions.append("✅ GOOD: Model generalization is acceptable")
    
    return suggestions


# REMOVED: federated_averaging function - not needed in true VFL architecture
# VFL uses embedding-based training, not weight averaging 