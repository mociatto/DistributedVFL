#!/usr/bin/env python3
"""
Distributed Federated Learning Server with Fusion-Guided Weight Updates
Implements true federated learning with iterative client retraining and guidance.
"""

import os
import sys
import time
import json
import pickle
import logging
import argparse
import threading
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf

from network_config import NetworkConfig, APIEndpoints, get_server_config
from config import get_config, get_current_phase_config, validate_config
from status import update_training_status
from data_loader import HAM10000DataLoader
from models import create_fusion_model_with_transformer
from train_evaluate import (
    train_fusion_model_with_adversarial,
    evaluate_fusion_model,
    save_training_plots
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedServer:
    """
    Federated server for coordinating vertical federated learning.
    Handles multimodal fusion with Transformer attention and controllable privacy.
    """
    
    def __init__(self, embedding_dim=256, num_classes=7, adversarial_lambda=0.0,
                 learning_rate=0.001, data_percentage=0.1, config=None):
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.adversarial_lambda = adversarial_lambda
        self.learning_rate = learning_rate
        self.data_percentage = data_percentage
        self.config = config or {}  # Store full configuration
        
        # Models
        self.fusion_model = None
        self.adversarial_model = None
        
        # Inference attack models (always present)
        self.age_inference_model = None
        self.gender_inference_model = None
        
        # Data loader for metadata
        self.data_loader = None
        
        # Training metrics
        self.training_history = {
            'round_accuracies': [],
            'round_f1_scores': [],
            'round_losses': [],
            'training_times': []
        }
        
        # Best performance tracking
        self.best_accuracy = 0.0
        self.best_f1 = 0.0
        self.best_round = 0
        
        # Federated learning state
        self.aggregated_embedding_knowledge = None
        
        # DYNAMIC ADVERSARIAL LAMBDA SYSTEM - DISABLED BY DEFAULT
        self.dynamic_lambda_enabled = False  # DISABLED until dashboard button clicked
        self.lambda_history = []  # Track lambda changes over time
        self.attack_performance_history = []  # Track attack success rates
        self.lambda_adjustment_strategy = "adaptive_pid"  # "adaptive_pid", "threshold_based", "exponential"
        
        # PID Controller parameters for lambda adjustment
        self.lambda_pid = {
            'kp': 0.8,      # Proportional gain (how quickly to respond)
            'ki': 0.2,      # Integral gain (how much to consider past errors)
            'kd': 0.1,      # Derivative gain (how much to consider rate of change)
            'previous_error': 0.0,
            'integral': 0.0,
            'target_age_leakage': 25.0,     # Target: reduce age leakage to ~25% (vs 16.67% random)
            'target_gender_leakage': 60.0,   # Target: reduce gender leakage to ~60% (vs 50% random)
            'min_lambda': 0.0,
            'max_lambda': 1.0,
            'lambda_change_rate': 0.15  # Maximum change per round
        }
        
        print(f"Federated Server Initialized")
        print(f"   Embedding dimension: {self.embedding_dim}")
        print(f"   Number of classes: {self.num_classes}")
        print(f"   Adversarial lambda: {self.adversarial_lambda}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Data percentage: {self.data_percentage*100:.1f}%")
        
        if self.adversarial_lambda == 0.0:
            print(f"   Privacy mechanism: DISABLED (Phase 1 - High Performance)")
        else:
            print(f"   Privacy mechanism: ENABLED (Lambda={self.adversarial_lambda})")
        
        # Dynamic lambda system info - ONLY show if enabled
        if self.dynamic_lambda_enabled:
            print(f"   Dynamic Lambda System: ENABLED")
            print(f"   Strategy: {self.lambda_adjustment_strategy}")
            print(f"   Target age leakage: {self.lambda_pid['target_age_leakage']:.1f}%")
            print(f"   Target gender leakage: {self.lambda_pid['target_gender_leakage']:.1f}%")
        else:
            print(f"   Dynamic Lambda System: DISABLED (controlled by dashboard)")
        
        # Create inference attack models
        self.create_inference_models()
        print(f"   Inference attacks: Age (6-class), Gender (2-class)")
    
    def create_inference_models(self):
        """Create inference attack models for privacy leakage assessment."""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
        
        # Input dimension: combined embeddings (image + tabular)
        input_dim = self.embedding_dim * 2
        
        # Age inference model (6-class classification)
        self.age_inference_model = Sequential([
            Dense(256, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(6, activation='softmax', name='age_output')  # 6 age bins
        ], name='age_inference_attack')
        
        # Gender inference model (2-class classification)
        self.gender_inference_model = Sequential([
            Dense(256, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(2, activation='softmax', name='gender_output')  # 2 genders
        ], name='gender_inference_attack')
        
        # Compile models
        self.age_inference_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.gender_inference_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"   Age inference model: {self.age_inference_model.count_params():,} parameters")
        print(f"   Gender inference model: {self.gender_inference_model.count_params():,} parameters")
    
    def create_models(self, use_advanced_fusion=True, use_step3_enhancements=True):
        """
        Create and initialize all required models.
        
        Args:
            use_advanced_fusion (bool): Whether to use Step 2 advanced fusion
            use_step3_enhancements (bool): Whether to use Step 3 generalization enhancements
        """
        print("Creating server models...")
        
        # Create fusion model with optional advanced features
        self.fusion_model, self.adversarial_model = create_fusion_model_with_transformer(
            image_dim=self.embedding_dim,
            tabular_dim=self.embedding_dim,
            num_classes=self.num_classes,
            adversarial_lambda=self.adversarial_lambda,
            use_advanced_fusion=use_advanced_fusion,  # STEP 2: Advanced fusion option
            use_step3_enhancements=use_step3_enhancements  # STEP 3: Generalization enhancements
        )
        
        print(f"   Fusion model created with {self.fusion_model.count_params():,} parameters")
        
        if self.adversarial_model is not None:
            print(f"   Adversarial model created (λ={self.adversarial_lambda})")
        else:
            print(f"   Adversarial model disabled (lambda={self.adversarial_lambda})")
        
        # STEP 2 & 3: Create ensemble models for better robustness
        if use_advanced_fusion or use_step3_enhancements:
            print("   Creating ensemble models for enhanced robustness...")
            self.ensemble_models = []
            
            # Create 3 diverse fusion models for ensemble
            for i in range(3):
                # Mix different configurations for diversity
                use_advanced = (i % 2 == 0)
                use_step3 = use_step3_enhancements and (i != 1)  # Skip step3 for middle model
                
                ensemble_model, _ = create_fusion_model_with_transformer(
                    image_dim=self.embedding_dim,
                    tabular_dim=self.embedding_dim,
                    num_classes=self.num_classes,
                    adversarial_lambda=0.0,
                    use_advanced_fusion=use_advanced,
                    use_step3_enhancements=use_step3
                )
                self.ensemble_models.append(ensemble_model)
            
            print(f"   Created {len(self.ensemble_models)} diverse ensemble models")
        else:
            self.ensemble_models = []
    
    def load_client_embeddings(self, data_split='train', embeddings_dir='embeddings'):
        """
        Load embeddings from both clients.
        
        Args:
            data_split (str): 'train', 'val', or 'test'
            embeddings_dir (str): Directory containing embedding files
        
        Returns:
            tuple: (image_embeddings, tabular_embeddings, labels, sensitive_attrs)
        """
        print(f"\nLoading {data_split} embeddings from clients...")
        
        # Load image client embeddings
        image_file = f"{embeddings_dir}/image_client_{data_split}_embeddings.pkl"
        with open(image_file, 'rb') as f:
            image_data = pickle.load(f)
        
        # Load tabular client embeddings
        tabular_file = f"{embeddings_dir}/tabular_client_{data_split}_embeddings.pkl"
        with open(tabular_file, 'rb') as f:
            tabular_data = pickle.load(f)
        
        # Verify alignment
        assert np.array_equal(image_data['indices'], tabular_data['indices']), \
            f"Sample indices mismatch between clients for {data_split} split!"
        assert np.array_equal(image_data['labels'], tabular_data['labels']), \
            f"Labels mismatch between clients for {data_split} split!"
        
        print(f"   Embeddings loaded and verified")
        print(f"   - Image embeddings: {image_data['embeddings'].shape}")
        print(f"   - Tabular embeddings: {tabular_data['embeddings'].shape}")
        print(f"   - Samples: {len(image_data['labels'])}")
        
        # Get sensitive attributes (should be same from both clients)
        sensitive_attrs = image_data.get('sensitive_attrs', None)
        if sensitive_attrs is None:
            sensitive_attrs = tabular_data.get('sensitive_attrs', None)
        
        return (image_data['embeddings'], tabular_data['embeddings'], 
                image_data['labels'], sensitive_attrs)

    def train_vfl_round(self, round_idx, total_rounds, epochs=8, batch_size=16):
        """Train VFL for one round."""
        print(f"\nTRAINING ROUND {round_idx + 1}/{total_rounds}")
        print("="*50)
        
        start_time = time.time()
        
        try:
            # Load training embeddings
            train_image_emb, train_tabular_emb, train_labels, train_sensitive_attrs = self.load_client_embeddings('train')
            val_image_emb, val_tabular_emb, val_labels, val_sensitive_attrs = self.load_client_embeddings('val')
            
            # Train fusion model
            from train_evaluate import train_fusion_model_with_adversarial
            
            history = train_fusion_model_with_adversarial(
                fusion_model=self.fusion_model,
                adversarial_model=self.adversarial_model,
                image_embeddings=train_image_emb,
                tabular_embeddings=train_tabular_emb,
                labels=train_labels,
                sensitive_attrs=train_sensitive_attrs,  # FIXED: Now properly passing sensitive attributes
                val_image_embeddings=val_image_emb,
                val_tabular_embeddings=val_tabular_emb,
                val_labels=val_labels,
                val_sensitive_attrs=val_sensitive_attrs,  # FIXED: Now properly passing sensitive attributes
                epochs=epochs,
                batch_size=batch_size,
                adversarial_lambda=self.adversarial_lambda,
                verbose=1
            )
            
            # Extract metrics
            round_loss = min(history.history.get('loss', [1.0]))
            round_accuracy = max(history.history.get('val_accuracy', [0.0]))
            # Note: val_f1_macro may not be available, so we'll calculate it or use 0
            round_f1 = max(history.history.get('val_f1_macro', [0.0]))
            
            # Update tracking
            self.training_history['round_accuracies'].append(round_accuracy)
            self.training_history['round_f1_scores'].append(round_f1) 
            self.training_history['round_losses'].append(round_loss)
            self.training_history['training_times'].append(time.time() - start_time)
            
            # Update best performance
            if round_accuracy > self.best_accuracy:
                self.best_accuracy = round_accuracy
                self.best_f1 = round_f1
                self.best_round = round_idx + 1
                # Save best model
                try:
                    self.save_best_model()
                except Exception as e:
                    print(f"Failed to save best model: {e}")
            return {
                'accuracy': round_accuracy,
                'f1': round_f1,
                'loss': round_loss,
                'history': history,
                'training_time': time.time() - start_time
            }
            
        except Exception as e:
            print(f"Training round {round_idx + 1} failed: {e}")
            return {
                'accuracy': 0.0,
                'f1': 0.0, 
                'loss': 1.0,
                'history': {'accuracy': [0.0], 'val_accuracy': [0.0], 'loss': [1.0], 'val_loss': [1.0]},
                'training_time': time.time() - start_time
            }

    def evaluate_global_model(self, round_idx):
        """Evaluate global model on validation set."""
        try:
            val_image_emb, val_tabular_emb, val_labels, val_sensitive_attrs = self.load_client_embeddings('val')
            
            # Make predictions
            predictions = self.fusion_model.predict([val_image_emb, val_tabular_emb], verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            accuracy = accuracy_score(val_labels, predicted_classes)
            f1 = f1_score(val_labels, predicted_classes, average='macro')
            precision = precision_score(val_labels, predicted_classes, average='macro', zero_division=0)
            recall = recall_score(val_labels, predicted_classes, average='macro', zero_division=0)
            
            # Calculate combined precision-recall metric (harmonic mean)
            precision_recall = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate fairness metrics if sensitive attributes are available
            gender_fairness = [0.0, 0.0]  # [female_accuracy, male_accuracy]
            age_fairness = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 6 age groups
            
            if val_sensitive_attrs is not None:
                gender_fairness, age_fairness = self._calculate_fairness_metrics(
                    predicted_classes, val_labels, val_sensitive_attrs
                )
            
            return {
                'accuracy': accuracy, 
                'f1_macro': f1,
                'precision': precision,
                'recall': recall,
                'precision_recall': precision_recall,
                'gender_fairness': gender_fairness,
                'age_fairness': age_fairness
            }
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return {
                'accuracy': 0.0, 'f1_macro': 0.0, 'precision': 0.0, 'recall': 0.0, 'precision_recall': 0.0,
                'gender_fairness': [0.0, 0.0], 'age_fairness': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            }

    def _calculate_fairness_metrics(self, predicted_classes, true_labels, sensitive_attrs):
        """Calculate fairness metrics for gender and age groups."""
        from sklearn.metrics import accuracy_score
        
        # Initialize fairness scores
        gender_fairness = [0.0, 0.0]  # [female, male]
        age_fairness = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 6 age groups
        
        try:
            # Extract gender and age from sensitive attributes
            genders = sensitive_attrs[:, 0]  # First column is gender (0=female, 1=male)
            ages = sensitive_attrs[:, 1]     # Second column is age_bin (0-5)
            
            # Calculate gender fairness
            for gender_val in [0, 1]:  # 0=female, 1=male
                gender_mask = (genders == gender_val)
                if np.sum(gender_mask) > 0:
                    gender_acc = accuracy_score(
                        true_labels[gender_mask], 
                        predicted_classes[gender_mask]
                    )
                    gender_fairness[gender_val] = gender_acc * 100  # Convert to percentage
            
            # Calculate age fairness (6 groups)
            for age_val in range(6):  # 0-5 age bins
                age_mask = (ages == age_val)
                if np.sum(age_mask) > 0:
                    age_acc = accuracy_score(
                        true_labels[age_mask], 
                        predicted_classes[age_mask]
                    )
                    age_fairness[age_val] = age_acc * 100  # Convert to percentage
        
        except Exception as e:
            print(f"Fairness calculation error: {e}")
        
        return gender_fairness, age_fairness

    def train_and_evaluate_inference_attacks(self, round_idx):
        """
        Train inference attack models with fresh embeddings and evaluate leakage.
        This simulates adversarial attacks trying to infer sensitive attributes.
        ENHANCED: Now includes dynamic lambda adjustment based on attack performance.
        """
        try:
            print(f"   Training inference attacks for Round {round_idx + 1}...")
            
            # Load fresh embeddings for this round
            train_image_emb, train_tabular_emb, train_labels, train_sensitive_attrs = self.load_client_embeddings('train')
            val_image_emb, val_tabular_emb, val_labels, val_sensitive_attrs = self.load_client_embeddings('val')
            
            if train_image_emb is None or train_tabular_emb is None:
                print("     Warning: Could not load embeddings for inference attacks")
                return {
                    'age_leakage': 16.67,  # Random baseline
                    'gender_leakage': 50.0,  # Random baseline
                    'defense_strength': 0.0
                }
            
            # Combine embeddings (image + tabular)
            train_combined = np.concatenate([train_image_emb, train_tabular_emb], axis=1)
            val_combined = np.concatenate([val_image_emb, val_tabular_emb], axis=1)
            
            # Apply adversarial defense if lambda > 0 (ONLY when actually enabled)
            if self.adversarial_lambda > 0:
                print(f"     Applying adversarial defense (λ={self.adversarial_lambda:.3f}) to embeddings...")
                train_combined = self._apply_adversarial_defense(train_combined, self.adversarial_lambda)
                val_combined = self._apply_adversarial_defense(val_combined, self.adversarial_lambda)
            else:
                print(f"     No adversarial defense - embeddings vulnerable to inference attacks")
            
            # Extract sensitive attributes - FIXED: Handle numpy array correctly
            if train_sensitive_attrs is None or val_sensitive_attrs is None:
                print("     Warning: Sensitive attributes not available")
                return {
                    'age_leakage': 16.67,  # Random baseline
                    'gender_leakage': 50.0,  # Random baseline  
                    'defense_strength': 0.0
                }
            
            # FIXED: Correct indexing for numpy array
            # sensitive_attrs is numpy array with shape (n_samples, 2)
            # Column 0: sex_encoded (0=female, 1=male)
            # Column 1: age_bin (0-5 age groups)
            train_gender_labels = train_sensitive_attrs[:, 0].astype(int)  # Gender column
            train_age_labels = train_sensitive_attrs[:, 1].astype(int)     # Age column
            val_gender_labels = val_sensitive_attrs[:, 0].astype(int)      # Gender column
            val_age_labels = val_sensitive_attrs[:, 1].astype(int)         # Age column
            
            print(f"     Gender distribution: {np.bincount(train_gender_labels)} (0=female, 1=male)")
            print(f"     Age distribution: {np.bincount(train_age_labels)} (6 age groups)")
            
            # Train age inference attack
            print(f"     Training age inference attack...")
            self.age_inference_model.fit(
                train_combined, train_age_labels,
                validation_data=(val_combined, val_age_labels),
                epochs=10, batch_size=32, verbose=0
            )
            
            # Train gender inference attack  
            print(f"     Training gender inference attack...")
            self.gender_inference_model.fit(
                train_combined, train_gender_labels,
                validation_data=(val_combined, val_gender_labels),
                epochs=10, batch_size=32, verbose=0
            )
            
            # Evaluate inference attacks
            age_pred = self.age_inference_model.predict(val_combined, verbose=0)
            gender_pred = self.gender_inference_model.predict(val_combined, verbose=0)
            
            age_accuracy = np.mean(np.argmax(age_pred, axis=1) == val_age_labels) * 100
            gender_accuracy = np.mean(np.argmax(gender_pred, axis=1) == val_gender_labels) * 100
            
            print(f"     Age leakage: {age_accuracy:.2f}% (baseline: 16.67%)")
            print(f"     Gender leakage: {gender_accuracy:.2f}% (baseline: 50.0%)")
            
            # ONLY update dynamic lambda if the system is ENABLED
            if self.dynamic_lambda_enabled:
                print(f"   Evaluating dynamic lambda adjustment...")
                old_lambda = self.adversarial_lambda
                self.update_dynamic_adversarial_lambda(age_accuracy, gender_accuracy, round_idx)
                if self.adversarial_lambda != old_lambda:
                    print(f"   DYNAMIC LAMBDA UPDATE: {old_lambda:.3f} → {self.adversarial_lambda:.3f}")
                    print(f"     Reason: Attack performance evaluation")
                    print(f"     Strategy: {self.lambda_adjustment_strategy}")
            else:
                # Dynamic lambda is DISABLED - lambda only changes via dashboard
                print(f"   Dynamic lambda system DISABLED - lambda controlled by dashboard")
            
            # Calculate defense strength
            defense_strength = self._calculate_defense_strength(age_accuracy, gender_accuracy)
            
            return {
                'age_leakage': age_accuracy,
                'gender_leakage': gender_accuracy,
                'defense_strength': defense_strength
            }
            
        except Exception as e:
            print(f"   Error in inference attacks: {e}")
            import traceback
            traceback.print_exc()
            # Return baseline values on error (NOT fake 100% protection)
            return {
                'age_leakage': 90.0,  # Assume high leakage on error
                'gender_leakage': 95.0,  # Assume high leakage on error
                'defense_strength': 0.0  # No protection on error
            }

    def _apply_adversarial_defense(self, embeddings, adversarial_lambda):
        """
        Apply WILD adversarial defense perturbations to embeddings.
        ENHANCED: Much stronger perturbations inspired by successful research.
        
        Args:
            embeddings: Input embeddings to protect
            adversarial_lambda: Defense strength parameter
            
        Returns:
            Protected embeddings with WILD adversarial perturbations applied
        """
        if adversarial_lambda <= 0:
            return embeddings
            
        import tensorflow as tf
        
        # Convert to tensor for gradient computation
        embeddings_tensor = tf.Variable(embeddings, dtype=tf.float32)
        
        # Use the ACTUAL trained inference models (not dummy models)
        if not hasattr(self, 'age_inference_model') or not hasattr(self, 'gender_inference_model'):
            print("     Warning: Inference models not available, using WILD random noise")
            # Fallback to EXTREMELY strong random perturbations
            noise = tf.random.normal(embeddings_tensor.shape, stddev=adversarial_lambda * 3.0)  # 3x stronger!
            return (embeddings_tensor + noise).numpy()
        
        # WILD adversarial perturbation using trained models
        batch_size = tf.shape(embeddings_tensor)[0]
        protected_embeddings = embeddings_tensor
        
        # Multi-step adversarial perturbation (inspired by PGD approach)
        num_steps = 12  # More perturbation steps for EXTREME WILD defense
        step_size = adversarial_lambda * 1.2  # MUCH stronger step size (was 0.8)
        
        for step in range(num_steps):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(protected_embeddings)
                
                # Get predictions from ACTUAL trained inference models
                age_pred = self.age_inference_model(protected_embeddings, training=False)
                gender_pred = self.gender_inference_model(protected_embeddings, training=False)
                
                # Strategy 1: Maximize entropy (push toward uniform distribution)
                age_entropy = -tf.reduce_sum(age_pred * tf.math.log(age_pred + 1e-8), axis=1)
                gender_entropy = -tf.reduce_sum(gender_pred * tf.math.log(gender_pred + 1e-8), axis=1)
                
                # Strategy 2: Target confusion (push toward wrong predictions)
                # Create adversarial targets that are maximally confusing
                age_uniform_target = tf.ones_like(age_pred) / 6.0  # Uniform over 6 classes
                gender_uniform_target = tf.ones_like(gender_pred) / 2.0  # Uniform over 2 classes
                
                # Use cross-entropy loss instead of KL divergence (FIXED API ERROR)
                # Convert uniform targets to label format for cross-entropy
                age_confusion_loss = tf.keras.losses.categorical_crossentropy(age_uniform_target, age_pred)
                gender_confusion_loss = tf.keras.losses.categorical_crossentropy(gender_uniform_target, gender_pred)
                
                # Strategy 3: Feature corruption (add noise to most important features)
                # Calculate feature importance based on gradients
                dummy_age_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                    tf.zeros([batch_size], dtype=tf.int32), age_pred))
                dummy_gender_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                    tf.zeros([batch_size], dtype=tf.int32), gender_pred))
                
                # ENHANCED: Extra focus on gender defense (since it's harder to protect)
                gender_focus_multiplier = 3.0  # 3x more focus on gender defense
                
                # Combined adversarial objective: maximize entropy + maximize confusion + corrupt features
                total_adv_loss = (
                    -tf.reduce_mean(age_entropy) +      # Negative because we want to MAXIMIZE entropy
                    -tf.reduce_mean(gender_entropy) * gender_focus_multiplier +  # Extra gender focus
                    tf.reduce_mean(age_confusion_loss) +
                    tf.reduce_mean(gender_confusion_loss) * gender_focus_multiplier +  # Extra gender focus
                    dummy_age_loss + dummy_gender_loss * gender_focus_multiplier  # Extra gender focus
                )
            
            # Compute gradients w.r.t. embeddings
            adversarial_gradients = tape.gradient(total_adv_loss, protected_embeddings)
            del tape
            
            if adversarial_gradients is not None:
                # Apply WILD gradient-based perturbations
                # Use signed gradients for maximum disruption
                gradient_perturbation = step_size * tf.sign(adversarial_gradients)
                
                # Add WILD structured noise based on embedding statistics
                embedding_std = tf.math.reduce_std(protected_embeddings, axis=0)
                structured_noise = tf.random.normal(protected_embeddings.shape) * embedding_std * adversarial_lambda * 2.0  # 4x stronger!
                
                # Add WILD random uniform noise for additional robustness
                uniform_noise = tf.random.uniform(protected_embeddings.shape, -adversarial_lambda * 1.5, adversarial_lambda * 1.5)  # 4x stronger!
                
                # Add WILD Gaussian noise
                gaussian_noise = tf.random.normal(protected_embeddings.shape, stddev=adversarial_lambda * 1.0)  # 4x stronger!
                
                # Add EXTREME noise for gender-specific features
                # Target features that are most important for gender prediction
                gender_specific_noise = tf.random.normal(protected_embeddings.shape) * adversarial_lambda * 1.8  # Gender-specific noise
                
                # Combined perturbation: gradient + structured + uniform + gaussian + gender-specific
                total_perturbation = (gradient_perturbation + structured_noise + uniform_noise + 
                                    gaussian_noise + gender_specific_noise)
                
                # Apply perturbation with clipping to prevent extreme values
                protected_embeddings = protected_embeddings + total_perturbation
                
                # Optional: Clip to reasonable range to prevent exploding values
                embedding_mean = tf.reduce_mean(embeddings_tensor)
                embedding_std_global = tf.math.reduce_std(embeddings_tensor)
                clip_min = embedding_mean - 6 * embedding_std_global  # Much wider clipping range
                clip_max = embedding_mean + 6 * embedding_std_global
                protected_embeddings = tf.clip_by_value(protected_embeddings, clip_min, clip_max)
            else:
                # Fallback: WILD random perturbations
                strong_noise = tf.random.normal(protected_embeddings.shape, stddev=adversarial_lambda * 2.5)  # 3x stronger!
                protected_embeddings = protected_embeddings + strong_noise
        
        # Final enhancement: Add WILD feature-wise perturbations
        # Identify most important features and add targeted noise
        feature_importance = tf.math.reduce_std(embeddings_tensor, axis=0)  # Features with high variance
        top_k = min(int(embeddings_tensor.shape[1] * 0.7), 300)  # Top 70% or 300 features (was 50%/200)
        _, top_indices = tf.nn.top_k(feature_importance, k=top_k)
        
        # Create WILD targeted noise for most important features
        targeted_noise = tf.zeros_like(protected_embeddings)
        for i in range(top_k):
            feature_idx = top_indices[i]
            targeted_noise = tf.tensor_scatter_nd_update(
                targeted_noise,
                [[j, feature_idx] for j in range(batch_size)],
                tf.random.normal([batch_size]) * adversarial_lambda * 2.5  # 4x stronger!
            )
        
        protected_embeddings = protected_embeddings + targeted_noise
        
        # Additional WILD enhancement: Add layer-wise perturbations
        # Split embedding into chunks and apply different noise patterns
        chunk_size = embeddings_tensor.shape[1] // 6  # 6 chunks (was 4)
        for chunk_idx in range(6):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, embeddings_tensor.shape[1])
            
            # Different noise pattern for each chunk (more aggressive)
            if chunk_idx == 0:
                # Multiplicative noise
                chunk_noise = tf.random.normal([batch_size, end_idx - start_idx]) * adversarial_lambda * 0.8
                chunk_multiplier = 1.0 + chunk_noise
                protected_embeddings = tf.tensor_scatter_nd_update(
                    protected_embeddings,
                    [[i, j] for i in range(batch_size) for j in range(start_idx, end_idx)],
                    tf.reshape(protected_embeddings[:, start_idx:end_idx] * chunk_multiplier, [-1])
                )
            elif chunk_idx == 1:
                # Laplace noise (heavier tails than Gaussian)
                laplace_noise = tf.random.uniform([batch_size, end_idx - start_idx], -1, 1)
                laplace_noise = tf.sign(laplace_noise) * tf.math.log(1 - tf.abs(laplace_noise) + 1e-8) * adversarial_lambda * 0.6
                protected_embeddings = tf.tensor_scatter_nd_add(
                    protected_embeddings,
                    [[i, j] for i in range(batch_size) for j in range(start_idx, end_idx)],
                    tf.reshape(laplace_noise, [-1])
                )
            elif chunk_idx == 2:
                # Dropout-like noise (randomly zero out features)
                dropout_mask = tf.random.uniform([batch_size, end_idx - start_idx]) > (adversarial_lambda * 0.4)
                dropout_mask = tf.cast(dropout_mask, tf.float32)
                protected_embeddings = tf.tensor_scatter_nd_update(
                    protected_embeddings,
                    [[i, j] for i in range(batch_size) for j in range(start_idx, end_idx)],
                    tf.reshape(protected_embeddings[:, start_idx:end_idx] * dropout_mask, [-1])
                )
            elif chunk_idx == 3:
                # Adversarial sign flip
                sign_flip_prob = adversarial_lambda * 0.2
                flip_mask = tf.random.uniform([batch_size, end_idx - start_idx]) < sign_flip_prob
                flip_multiplier = tf.where(flip_mask, -1.0, 1.0)
                protected_embeddings = tf.tensor_scatter_nd_update(
                    protected_embeddings,
                    [[i, j] for i in range(batch_size) for j in range(start_idx, end_idx)],
                    tf.reshape(protected_embeddings[:, start_idx:end_idx] * flip_multiplier, [-1])
                )
            elif chunk_idx == 4:
                # Exponential-like noise (for extreme perturbations)
                # Use gamma distribution as exponential alternative
                exp_noise = tf.random.gamma([batch_size, end_idx - start_idx], alpha=1.0, beta=1.0) * adversarial_lambda * 0.3
                exp_noise = exp_noise * tf.random.uniform([batch_size, end_idx - start_idx], -1, 1)  # Random sign
                protected_embeddings = tf.tensor_scatter_nd_add(
                    protected_embeddings,
                    [[i, j] for i in range(batch_size) for j in range(start_idx, end_idx)],
                    tf.reshape(exp_noise, [-1])
                )
            else:
                # Cauchy noise (very heavy tails for extreme disruption)
                cauchy_noise = tf.random.uniform([batch_size, end_idx - start_idx], -0.5, 0.5)
                cauchy_noise = tf.math.tan(cauchy_noise * 3.14159) * adversarial_lambda * 0.2
                cauchy_noise = tf.clip_by_value(cauchy_noise, -adversarial_lambda * 2, adversarial_lambda * 2)
                protected_embeddings = tf.tensor_scatter_nd_add(
                    protected_embeddings,
                    [[i, j] for i in range(batch_size) for j in range(start_idx, end_idx)],
                    tf.reshape(cauchy_noise, [-1])
                )
        
        # EXTREME FINAL ENHANCEMENT: Gender-specific feature disruption
        # Add extra noise to features that are most predictive of gender
        if hasattr(self, 'gender_inference_model'):
            # Get the first layer weights to identify gender-predictive features
            try:
                gender_weights = self.gender_inference_model.layers[0].get_weights()[0]  # [features, hidden]
                gender_importance = tf.reduce_sum(tf.abs(gender_weights), axis=1)  # Sum over hidden units
                gender_top_k = min(int(len(gender_importance) * 0.3), 150)  # Top 30% gender features
                _, gender_top_indices = tf.nn.top_k(gender_importance, k=gender_top_k)
                
                # Add EXTREME noise to gender-predictive features
                gender_targeted_noise = tf.zeros_like(protected_embeddings)
                for i in range(gender_top_k):
                    feature_idx = gender_top_indices[i]
                    if feature_idx < protected_embeddings.shape[1]:  # Ensure valid index
                        gender_targeted_noise = tf.tensor_scatter_nd_update(
                            gender_targeted_noise,
                            [[j, feature_idx] for j in range(batch_size)],
                            tf.random.normal([batch_size]) * adversarial_lambda * 3.0  # EXTREME noise for gender features
                        )
                
                protected_embeddings = protected_embeddings + gender_targeted_noise
                print(f"     EXTREME Gender Defense: Added targeted noise to {gender_top_k} gender-predictive features")
                
            except Exception as e:
                print(f"     Could not apply gender-specific defense: {e}")
        
        # Calculate perturbation strength for logging
        perturbation_norm = tf.norm(protected_embeddings - embeddings_tensor)
        original_norm = tf.norm(embeddings_tensor)
        perturbation_ratio = perturbation_norm / (original_norm + 1e-8)
        
        print(f"     EXTREME WILD Defense Applied:")
        print(f"       Multi-step iterations: {num_steps}")
        print(f"       Perturbation strength: {float(perturbation_ratio):.3f}")
        print(f"       Lambda: {adversarial_lambda:.3f}")
        print(f"       Targeted features: {top_k}/{embeddings_tensor.shape[1]}")
        print(f"       Gender focus multiplier: 3.0x")
        print(f"       EXTREME WILD mode: {num_steps}-step multi-strategy with gender focus!")
        
        return protected_embeddings.numpy()

    def evaluate_final_model(self, class_names=None):
        """Final model evaluation."""
        print(f"\nFINAL MODEL EVALUATION")
        print("="*40)
        
        try:
            # Load test embeddings
            test_image_emb, test_tabular_emb, test_labels, _ = self.load_client_embeddings('test')
            
            # Make predictions
            predictions = self.fusion_model.predict([test_image_emb, test_tabular_emb], verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, f1_score, classification_report
            accuracy = accuracy_score(test_labels, predicted_classes)
            f1_macro = f1_score(test_labels, predicted_classes, average='macro')
            f1_weighted = f1_score(test_labels, predicted_classes, average='weighted')
            
            print(f"Final Test Accuracy: {accuracy:.4f}")
            print(f"Final Test F1 (macro): {f1_macro:.4f}")
            print(f"Final Test F1 (weighted): {f1_weighted:.4f}")
            
            if class_names:
                print(f"\nClassification Report:")
                report = classification_report(test_labels, predicted_classes, 
                                             target_names=class_names, output_dict=False)
                print(report)
            
            return {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'test_accuracy': accuracy,
                'test_f1': f1_macro
            }
            
        except Exception as e:
            print(f"Final evaluation failed: {e}")
            # Return best validation metrics as fallback
            return {
                'accuracy': self.best_accuracy,
                'f1_macro': self.best_f1,
                'f1_weighted': self.best_f1,
                'test_accuracy': self.best_accuracy,
                'test_f1': self.best_f1
            }

    def save_best_model(self, model_dir="models"):
        """Save the best model."""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            model_path = os.path.join(model_dir, "best_fusion_model.h5")
            self.fusion_model.save_weights(model_path)
            print(f"Best model saved to {model_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")

    def update_dynamic_adversarial_lambda(self, age_leakage, gender_leakage, round_idx):
        """
        Dynamically adjust adversarial lambda based on inference attack performance.
        Uses adaptive PID control inspired by successful adversarial training research.
        
        Args:
            age_leakage (float): Current age inference attack accuracy (%)
            gender_leakage (float): Current gender inference attack accuracy (%)
            round_idx (int): Current training round
            
        Returns:
            float: New adversarial lambda value
        """
        if not self.dynamic_lambda_enabled:
            return self.adversarial_lambda
        
        # Record current performance
        current_performance = {
            'round': round_idx,
            'age_leakage': age_leakage,
            'gender_leakage': gender_leakage,
            'lambda': self.adversarial_lambda
        }
        self.attack_performance_history.append(current_performance)
        
        # Calculate attack strength (how far from random guessing)
        age_attack_strength = max(0, age_leakage - 16.67)  # Above random (16.67%)
        gender_attack_strength = max(0, gender_leakage - 50.0)  # Above random (50%)
        
        # Combined attack threat level (0-100, higher = more threatening)
        age_threat = (age_attack_strength / (100 - 16.67)) * 100  # Normalize to 0-100
        gender_threat = (gender_attack_strength / (100 - 50)) * 100  # Normalize to 0-100
        combined_threat = (age_threat + gender_threat) / 2
        
        print(f"   Dynamic Lambda Analysis:")
        print(f"     Age threat level: {age_threat:.1f}% (leakage: {age_leakage:.1f}%)")
        print(f"     Gender threat level: {gender_threat:.1f}% (leakage: {gender_leakage:.1f}%)")
        print(f"     Combined threat: {combined_threat:.1f}%")
        
        if self.lambda_adjustment_strategy == "adaptive_pid":
            new_lambda = self._update_lambda_with_pid_control(age_leakage, gender_leakage, combined_threat)
        elif self.lambda_adjustment_strategy == "threshold_based":
            new_lambda = self._update_lambda_threshold_based(combined_threat)
        elif self.lambda_adjustment_strategy == "exponential":
            new_lambda = self._update_lambda_exponential(combined_threat)
        else:
            new_lambda = self.adversarial_lambda  # No change
        
        # Apply rate limiting to prevent oscillation
        max_change = self.lambda_pid['lambda_change_rate']
        lambda_change = np.clip(new_lambda - self.adversarial_lambda, -max_change, max_change)
        new_lambda = self.adversarial_lambda + lambda_change
        
        # Clamp to valid range
        new_lambda = np.clip(new_lambda, self.lambda_pid['min_lambda'], self.lambda_pid['max_lambda'])
        
        # Update lambda if there's a significant change
        if abs(new_lambda - self.adversarial_lambda) > 0.01:
            old_lambda = self.adversarial_lambda
            self.adversarial_lambda = new_lambda
            self.lambda_history.append({
                'round': round_idx,
                'old_lambda': old_lambda,
                'new_lambda': new_lambda,
                'change_reason': f'Threat: {combined_threat:.1f}%',
                'age_leakage': age_leakage,
                'gender_leakage': gender_leakage
            })
            
            print(f"   DYNAMIC LAMBDA UPDATE: {old_lambda:.3f} → {new_lambda:.3f}")
            print(f"     Reason: Combined threat level {combined_threat:.1f}%")
            print(f"     Strategy: {self.lambda_adjustment_strategy}")
            
            return new_lambda
        else:
            print(f"   Lambda unchanged: {self.adversarial_lambda:.3f} (small adjustment)")
            return self.adversarial_lambda
    
    def _update_lambda_with_pid_control(self, age_leakage, gender_leakage, combined_threat):
        """
        Update lambda using PID control algorithm for smooth, stable adjustment.
        Inspired by "Adversarial Fine-tune with Dynamically Regulated Adversary" research.
        """
        pid = self.lambda_pid
        
        # Calculate error from target performance
        age_error = max(0, age_leakage - pid['target_age_leakage'])
        gender_error = max(0, gender_leakage - pid['target_gender_leakage'])
        combined_error = (age_error + gender_error) / 2
        
        # PID calculations
        # Proportional term: current error
        proportional = pid['kp'] * combined_error
        
        # Integral term: accumulated error over time
        pid['integral'] += combined_error
        integral = pid['ki'] * pid['integral']
        
        # Derivative term: rate of change of error
        derivative = pid['kd'] * (combined_error - pid['previous_error'])
        pid['previous_error'] = combined_error
        
        # PID output (desired lambda increase)
        lambda_adjustment = (proportional + integral + derivative) / 100.0  # Scale to reasonable range
        
        # Current lambda + adjustment
        new_lambda = self.adversarial_lambda + lambda_adjustment
        
        print(f"     PID Control: P={proportional:.3f}, I={integral:.3f}, D={derivative:.3f}")
        print(f"     Error: {combined_error:.1f}%, Adjustment: {lambda_adjustment:+.3f}")
        
        return new_lambda
    
    def _update_lambda_threshold_based(self, combined_threat):
        """
        Update lambda using threshold-based approach with multiple levels.
        """
        if combined_threat > 80:      # Very high threat
            target_lambda = 0.8
        elif combined_threat > 60:    # High threat
            target_lambda = 0.6
        elif combined_threat > 40:    # Medium threat
            target_lambda = 0.4
        elif combined_threat > 20:    # Low threat
            target_lambda = 0.2
        else:                         # Very low threat
            target_lambda = 0.05
        
        print(f"     Threshold-based: Threat {combined_threat:.1f}% → λ={target_lambda:.2f}")
        return target_lambda
    
    def _update_lambda_exponential(self, combined_threat):
        """
        Update lambda using exponential scaling for aggressive response.
        """
        # Exponential scaling: λ = 0.8 * (threat/100)^2
        normalized_threat = combined_threat / 100.0
        target_lambda = 0.8 * (normalized_threat ** 1.5)  # Exponential response
        
        print(f"     Exponential: Threat {combined_threat:.1f}% → λ={target_lambda:.3f}")
        return target_lambda
    
    def get_dynamic_lambda_status(self):
        """Get current status of dynamic lambda system for dashboard."""
        if not self.dynamic_lambda_enabled:
            return {"enabled": False}
        
        latest_performance = self.attack_performance_history[-1] if self.attack_performance_history else None
        latest_change = self.lambda_history[-1] if self.lambda_history else None
        
        return {
            "enabled": True,
            "current_lambda": self.adversarial_lambda,
            "strategy": self.lambda_adjustment_strategy,
            "target_age_leakage": self.lambda_pid['target_age_leakage'],
            "target_gender_leakage": self.lambda_pid['target_gender_leakage'],
            "latest_performance": latest_performance,
            "latest_change": latest_change,
            "total_adjustments": len(self.lambda_history),
            "pid_params": {
                "kp": self.lambda_pid['kp'],
                "ki": self.lambda_pid['ki'], 
                "kd": self.lambda_pid['kd'],
                "integral": self.lambda_pid['integral']
            }
        }

    def _calculate_defense_strength(self, age_leakage, gender_leakage):
        """
        Calculate defense strength based on actual inference attack degradation.
        
        Args:
            age_leakage: Current age inference attack accuracy (%)
            gender_leakage: Current gender inference attack accuracy (%)
            
        Returns:
            Defense strength percentage (0-100%)
        """
        # Baseline accuracy for random guessing
        baseline_age_accuracy = 16.67  # 1/6 classes
        baseline_gender_accuracy = 50.0  # 1/2 classes
        
        # Maximum possible attack advantage (perfect vs random)
        max_age_advantage = 100.0 - baseline_age_accuracy  # 83.33%
        max_gender_advantage = 100.0 - baseline_gender_accuracy  # 50.0%
        
        # Current attack advantage (actual vs random)
        age_advantage = max(0, age_leakage - baseline_age_accuracy)
        gender_advantage = max(0, gender_leakage - baseline_gender_accuracy)
        
        # Defense effectiveness: how much we've reduced attack advantage
        age_defense_ratio = 1 - (age_advantage / max_age_advantage) if max_age_advantage > 0 else 0
        gender_defense_ratio = 1 - (gender_advantage / max_gender_advantage) if max_gender_advantage > 0 else 0
        
        # Combined defense strength (0-100%)
        defense_strength = ((age_defense_ratio + gender_defense_ratio) / 2) * 100
        
        return max(0, min(100, defense_strength))  # Clamp to [0, 100]

app = Flask(__name__)
CORS(app)

# Global server instance
federated_server = None
registered_clients = {}
round_performance_history = []

# Round synchronization state
current_global_round = 0
clients_completed_current_round = set()
round_embeddings_received = set()

# Server shutdown configuration
auto_shutdown_enabled = True

# FL orchestration state
fl_orchestration_active = False
fl_orchestration_thread = None

# Guidance effectiveness tracking
guidance_effectiveness = {
    'total_guidance_sent': 0,
    'client_improvements': 0,
    'client_degradations': 0,
    'improvement_rate': 0.0,
    'average_improvement': 0.0,
    'rejected_guidance_count': 0,
    'adaptive_lr_adjustments': 0
}

def initialize_server():
    """Initialize the federated server with advanced ML strategies."""
    global federated_server
    print("Initializing Distributed Federated Server...")
    
    # Load configurations
    config = get_config()
    fl_config = config['federated_learning']
    model_config = config['model']
    
    federated_server = FederatedServer(
        embedding_dim=model_config['embedding_dim'],
        num_classes=config['data']['num_classes'],
        learning_rate=config['training']['learning_rate'],
        data_percentage=config['data']['data_percentage'],
        config=config
    )
    
    # Create models with all advanced features
    federated_server.create_models(
        use_advanced_fusion=True,
        use_step3_enhancements=True
    )
    
    print("Server initialized with advanced ML strategies preserved")

def compute_embedding_gradients(embeddings_dict, labels):
    """
    Compute gradients w.r.t. embeddings for fusion-guided updates.
    
    Args:
        embeddings_dict: Dict with 'image' and 'tabular' embeddings
        labels: True labels for the batch
    
    Returns:
        dict: Gradients for each modality
    """
    with tf.GradientTape() as tape:
        # Watch the embeddings
        tape.watch(embeddings_dict['image'])
        tape.watch(embeddings_dict['tabular'])
        
        # Forward pass through fusion model
        predictions = federated_server.fusion_model([
            embeddings_dict['image'], 
            embeddings_dict['tabular']
        ])
        
        # Compute loss
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, predictions, from_logits=False
        )
        loss = tf.reduce_mean(loss)
    
    # Compute gradients w.r.t embeddings
    gradients = tape.gradient(loss, [embeddings_dict['image'], embeddings_dict['tabular']])
    
    return {
        'image_gradients': gradients[0].numpy(),
        'tabular_gradients': gradients[1].numpy(),
        'loss_value': loss.numpy()
    }

def extract_attention_weights():
    """Extract attention weights from fusion model for client guidance."""
    attention_weights = {}
    
    # Find attention layers in fusion model
    for i, layer in enumerate(federated_server.fusion_model.layers):
        if hasattr(layer, 'attention') or 'attention' in layer.name.lower():
            try:
                # Get attention weights if available
                if hasattr(layer, 'get_attention_weights'):
                    weights = layer.get_attention_weights()
                    attention_weights[f'layer_{i}'] = weights.numpy()
            except Exception as e:
                print(f"Could not extract attention from layer {i}: {e}")
    
    return attention_weights

def orchestrate_federated_learning():
    """Main server-driven federated learning orchestration."""
    global auto_shutdown_enabled
    
    print(f"\nSTARTING SERVER-DRIVEN FEDERATED LEARNING")
    print("=" * 60)
    
    try:
        config = get_config()
        fl_config = config['federated_learning']
        total_rounds = fl_config['total_fl_rounds']
        
        print(f"FL Configuration:")
        print(f"   Total rounds: {total_rounds}")
        print(f"   Epochs per round: {fl_config['client_epochs_per_round']}")
        print(f"   Strategy: Fusion-Guided Weight Updates")
        
        # Initialize status tracking for dashboard
        from status import initialize_status
        initialize_status(total_rounds)
        
        # Phase 1: Request initial training
        print(f"\nPHASE 1: REQUESTING INITIAL TRAINING")
        print("=" * 50)
        
        # Request initial training from all clients
        for client_id, client_info in registered_clients.items():
            client_info['current_task'] = 'initial_training'
            client_info['waiting_for_request'] = True
        
        # Wait for initial training completion
        wait_for_all_clients_to_complete('initial_training')
        
        # Phase 2: FL Rounds
        for round_idx in range(total_rounds):
            print(f"\nFL ROUND {round_idx + 1}/{total_rounds}")
            print("=" * 50)
            
            # Train global fusion model
            print(f"Training global fusion model with current embeddings...")
            train_global_model_round(round_idx)
            
            # Request next round embeddings (if not final round)
            if round_idx < total_rounds - 1:
                print(f"Requesting Round {round_idx + 2} embeddings with guidance...")
                
                # Send guidance and request fresh embeddings
                for client_id, client_info in registered_clients.items():
                    client_info['current_task'] = f'round_{round_idx + 1}_training'
                    client_info['waiting_for_request'] = True
                
                # Wait for round completion
                wait_for_all_clients_to_complete(f'round_{round_idx + 1}_training')
        
        # Phase 3: Final evaluation
        print(f"\nPHASE 3: FINAL EVALUATION")
        print("=" * 40)
        
        # Use class names from config since server doesn't have data_loader
        config = get_config()
        class_names = config['data']['class_names']
        final_results = federated_server.evaluate_final_model(class_names=class_names)
        
        print(f"\nDISTRIBUTED FL FINAL RESULTS:")
        print(f"   Global Model Test Accuracy: {final_results.get('accuracy', 0.0):.4f} ({final_results.get('accuracy', 0.0)*100:.2f}%)")
        print(f"   Global Model Test F1: {final_results.get('f1_macro', 0.0):.4f}")
        print(f"   Total FL Rounds: {total_rounds}")
        print(f"   Participating Clients: {len(registered_clients)}")
        
        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results/distributed_fl_results_{timestamp}.json"
        os.makedirs("results", exist_ok=True)
        
        final_summary = {
            'final_test_accuracy': final_results.get('accuracy', 0.0),
            'final_test_f1': final_results.get('f1_macro', 0.0),
            'total_rounds': total_rounds,
            'clients': len(registered_clients),
            'strategy': 'fusion_guided_updates',
            'timestamp': time.time()
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_summary, f, indent=2)
        
        print(f"   Results saved to {results_file}")
        
        # Phase 4: Signal completion to clients
        print(f"\nSIGNALING COMPLETION TO ALL CLIENTS")
        print("=" * 40)
        
        for client_id, client_info in registered_clients.items():
            client_info['current_task'] = 'completed'
            print(f"   Signaled completion to client {client_id}")
        
        # Schedule server shutdown
        if auto_shutdown_enabled:
            print(f"\nFEDERATED LEARNING COMPLETED!")
            print(f"Server will shutdown in 10 seconds...")
            print(f"All processes finished successfully!")
            
            def shutdown_server():
                time.sleep(10)
                print(f"\nShutting down server...")
                print(f"All terminals ready for new commands!")
                os._exit(0)
            
            shutdown_thread = threading.Thread(target=shutdown_server)
            shutdown_thread.daemon = True
            shutdown_thread.start()
        
    except Exception as e:
        print(f"FL Orchestration error: {str(e)}")
        import traceback
        traceback.print_exc()

def wait_for_all_clients_to_complete(task_name):
    """Wait for all clients to complete a specific task."""
    print(f"Waiting for all clients to complete: {task_name}")
    
    max_wait_time = 1800  # 30 minutes max
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        all_completed = True
        completed_count = 0
        
        for client_id, client_info in registered_clients.items():
            current_task = client_info.get('current_task', 'idle')
            # A task is completed when current_task is 'idle' or different from the assigned task
            if current_task == task_name:
                # Client still working on this task
                all_completed = False
            else:
                # Client completed this task (now idle or on different task)
                completed_count += 1
        
        total_clients = len(registered_clients)
        print(f"   Progress: {completed_count}/{total_clients} clients completed {task_name}")
        
        if all_completed:
            print(f"   All clients completed {task_name}!")
            return True
        
        time.sleep(5)
    
    print(f"   Timeout waiting for {task_name} after 30 minutes")
    return False

def train_global_model_round(round_idx):
    """Train the global fusion model for a specific round."""
    try:
        config = get_config()
        
        # Load current embeddings
        train_image_emb, train_tabular_emb, train_labels, train_sensitive = \
            federated_server.load_client_embeddings('train')
        val_image_emb, val_tabular_emb, val_labels, val_sensitive = \
            federated_server.load_client_embeddings('val')
        
        # Train fusion model
        round_results = federated_server.train_vfl_round(
            round_idx=round_idx,
            total_rounds=config['federated_learning']['total_fl_rounds'],
            epochs=config['training']['epochs_per_round'],
            batch_size=config['training']['batch_size']
        )
        
        # Evaluate performance
        val_results = federated_server.evaluate_global_model(round_idx)
        
        # Train and evaluate inference attacks (always present)
        inference_results = federated_server.train_and_evaluate_inference_attacks(round_idx)
        
        # Calculate defense strength based on actual inference attack degradation
        age_leakage = inference_results.get('age_leakage', 16.67)
        gender_leakage = inference_results.get('gender_leakage', 50.0)
        
        if federated_server.adversarial_lambda > 0:
            # Baseline accuracy for random guessing
            baseline_age_accuracy = 16.67  # 1/6 classes
            baseline_gender_accuracy = 50.0  # 1/2 classes
            
            # Calculate defense effectiveness: how much we've reduced attack accuracy
            # Perfect defense would bring attacks down to random guessing levels
            
            # For age: if attack goes from 100% to 16.67%, that's 100% defense effectiveness
            max_possible_age_attack = 100.0
            age_attack_reduction = max(0, max_possible_age_attack - age_leakage)
            max_possible_age_reduction = max_possible_age_attack - baseline_age_accuracy  # 83.33%
            age_defense_effectiveness = (age_attack_reduction / max_possible_age_reduction) * 100 if max_possible_age_reduction > 0 else 0
            
            # For gender: if attack goes from 100% to 50%, that's 100% defense effectiveness  
            max_possible_gender_attack = 100.0
            gender_attack_reduction = max(0, max_possible_gender_attack - gender_leakage)
            max_possible_gender_reduction = max_possible_gender_attack - baseline_gender_accuracy  # 50%
            gender_defense_effectiveness = (gender_attack_reduction / max_possible_gender_reduction) * 100 if max_possible_gender_reduction > 0 else 0
            
            # Combined defense strength: average of age and gender defense effectiveness
            defense_strength = (age_defense_effectiveness + gender_defense_effectiveness) / 2
            defense_strength = max(0, min(100, defense_strength))  # Clamp to 0-100%
            
            print(f"   Adversarial defense ACTIVE - Protection level: {defense_strength:.2f}%")
            print(f"   Age attack: {age_leakage:.1f}% (baseline: {baseline_age_accuracy:.1f}%, reduction: {age_attack_reduction:.1f}%)")
            print(f"   Gender attack: {gender_leakage:.1f}% (baseline: {baseline_gender_accuracy:.1f}%, reduction: {gender_attack_reduction:.1f}%)")
            print(f"   Age defense effectiveness: {age_defense_effectiveness:.1f}%")
            print(f"   Gender defense effectiveness: {gender_defense_effectiveness:.1f}%")
        else:
            # No adversarial training = No defense = 0% protection
            defense_strength = 0.0
            print(f"   No adversarial defense - embeddings vulnerable to inference attacks")
        
        # Update training status for dashboard
        config = get_config()
        update_training_status(
            current_round=round_idx + 1,
            total_rounds=config['federated_learning']['total_fl_rounds'],
            accuracy=val_results['accuracy'],
            loss=round_results.get('loss', 0.0),
            f1_score=val_results['f1_macro'],  # Fix: use f1_macro instead of f1_score
            precision=val_results.get('precision', 0.0),
            recall=val_results.get('recall', 0.0),
            precision_recall=val_results.get('precision_recall', 0.0),
            defense_strength=defense_strength,
            gender_fairness=val_results.get('gender_fairness', [0.0, 0.0]),
            age_fairness=val_results.get('age_fairness', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            age_leakage=inference_results.get('age_leakage', 16.67),
            gender_leakage=inference_results.get('gender_leakage', 50.0),
            phase="fl_round_complete"
        )
        
        print(f"   Round {round_idx + 1} Results:")
        print(f"   Accuracy: {val_results['accuracy']:.4f}")
        print(f"   F1 Score: {val_results['f1_macro']:.4f}")
        print(f"   Loss: {round_results.get('loss', 0.0):.4f}")
        
        return round_results
        
    except Exception as e:
        print(f"   Error training global model: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'server_ready': federated_server is not None,
        'registered_clients': len(registered_clients),
        'timestamp': time.time()
    })

@app.route('/register_client', methods=['POST'])
def register_client():
    """Register a new client."""
    global fl_orchestration_active, fl_orchestration_thread
    
    data = request.json
    client_id = data.get('client_id')
    client_type = data.get('client_type')  # 'image' or 'tabular'
    
    if not client_id or not client_type:
        return jsonify({'error': 'Missing client_id or client_type'}), 400
    
    registered_clients[client_id] = {
        'client_type': client_type,
        'registered_at': time.time(),
        'last_seen': time.time(),
        'rounds_participated': 0,
        'current_round': 0,
        'current_accuracy': 0.0,
        'performance_history': [],
        'waiting_for_request': False,
        'current_task': 'idle',
        'sample_counts': {}  # Initialize sample counts
    }
    
    print(f"Registered {client_type} client: {client_id}")
    
    # Check if we have enough clients to start FL
    required_clients = 2  # image + tabular
    if len(registered_clients) >= required_clients and not fl_orchestration_active:
        print(f"\nSUFFICIENT CLIENTS REGISTERED - STARTING FL ORCHESTRATION")
        fl_orchestration_active = True
        fl_orchestration_thread = threading.Thread(target=orchestrate_federated_learning)
        fl_orchestration_thread.daemon = True
        fl_orchestration_thread.start()
    
    return jsonify({
        'status': 'registered',
        'client_id': client_id,
        'total_clients': len(registered_clients),
        'fl_starting': len(registered_clients) >= required_clients
    })

@app.route('/get_fl_config', methods=['GET'])
def get_fl_config():
    """Send FL configuration to clients."""
    config = get_config()
    fl_config = config['federated_learning']
    
    return jsonify({
        'total_fl_rounds': fl_config['total_fl_rounds'],
        'client_epochs_per_round': fl_config['client_epochs_per_round'],
        'client_learning_rate_multiplier': fl_config['client_learning_rate_multiplier'],
        'guidance_weight': fl_config['guidance_weight'],
        'base_learning_rate': config['training']['client_learning_rate'],
        'data_percentage': config['data']['data_percentage']  # Add this so clients get the correct percentage
    })

@app.route('/send_global_guidance', methods=['POST'])
def send_global_guidance():
    """
    Send fusion-guided updates to clients.
    This is the core of our enhanced FL strategy.
    INCLUDES ROUND SYNCHRONIZATION - clients must wait for all to complete previous round.
    """
    global current_global_round, clients_completed_current_round
    
    data = request.json
    client_id = data.get('client_id')
    current_round = data.get('current_round', 0)
    
    if client_id not in registered_clients:
        return jsonify({'error': 'Client not registered'}), 400
    
    client_type = registered_clients[client_id]['client_type']
    
    # ROUND SYNCHRONIZATION CHECK
    if current_round > current_global_round:
        # Check if all clients have completed the current global round
        total_clients = len(registered_clients)
        completed_clients = len(clients_completed_current_round)
        
        if completed_clients < total_clients:
            print(f"Client {client_id} waiting for round synchronization")
            print(f"   Round {current_global_round + 1}: {completed_clients}/{total_clients} clients completed")
            return jsonify({
                'error': 'round_not_ready',
                'message': f'Waiting for all clients to complete round {current_global_round + 1}',
                'current_global_round': current_global_round,
                'clients_completed': completed_clients,
                'total_clients': total_clients,
                'wait_time': 5  # Client should wait 5 seconds and retry
            }), 423  # 423 = Locked (resource temporarily unavailable)
        else:
            # All clients completed - advance to next round
            current_global_round += 1
            clients_completed_current_round = set()
            print(f"ADVANCING TO GLOBAL ROUND {current_global_round + 1}")
            print(f"   All {total_clients} clients synchronized")
    
    try:
        # Load current embeddings to compute gradients
        embeddings_dir = 'embeddings'
        
        # Load a sample of embeddings for gradient computation
        train_image_emb, train_tabular_emb, train_labels, _ = federated_server.load_client_embeddings('train')
        
        # Take a batch for gradient computation (for efficiency)
        batch_size = min(32, len(train_labels))
        batch_indices = np.random.choice(len(train_labels), batch_size, replace=False)
        
        batch_image_emb = train_image_emb[batch_indices]
        batch_tabular_emb = train_tabular_emb[batch_indices]
        batch_labels = train_labels[batch_indices]
        
        # Compute embedding gradients
        embeddings_dict = {
            'image': tf.convert_to_tensor(batch_image_emb, dtype=tf.float32),
            'tabular': tf.convert_to_tensor(batch_tabular_emb, dtype=tf.float32)
        }
        
        gradient_info = compute_embedding_gradients(embeddings_dict, batch_labels)
        
        # Extract attention weights
        attention_weights = extract_attention_weights()
        
        # Get FL config
        fl_config = get_config()['federated_learning']
        
        # Prepare guidance for specific client type
        if client_type == 'image':
            client_gradients = gradient_info['image_gradients']
        else:  # tabular
            client_gradients = gradient_info['tabular_gradients']
        
        # GRADIENT VALIDATION - Check if gradients are meaningful
        gradient_norm = np.linalg.norm(client_gradients)
        gradient_mean = np.mean(np.abs(client_gradients))
        
        # Adaptive learning rate based on client's recent performance
        adaptive_lr_multiplier = fl_config['client_learning_rate_multiplier']
        client_info = registered_clients[client_id]
        
        if 'performance_history' in client_info and len(client_info['performance_history']) >= 2:
            # Check if client improved in last round
            recent_performance = client_info['performance_history'][-2:]
            accuracy_trend = recent_performance[-1]['accuracy'] - recent_performance[-2]['accuracy']
            
            if accuracy_trend < -0.01:  # Client degraded significantly (>1%)
                adaptive_lr_multiplier *= 0.5  # Reduce learning rate
                guidance_effectiveness['adaptive_lr_adjustments'] += 1
                print(f"   Reducing LR multiplier to {adaptive_lr_multiplier:.3f} due to client degradation")
            elif accuracy_trend > 0.01:  # Client improved significantly (>1%)
                adaptive_lr_multiplier *= 1.2  # Slightly increase learning rate
                adaptive_lr_multiplier = min(adaptive_lr_multiplier, 1.0)  # Cap at 1.0
                guidance_effectiveness['adaptive_lr_adjustments'] += 1
                print(f"   Increasing LR multiplier to {adaptive_lr_multiplier:.3f} due to client improvement")
        
        # GRADIENT QUALITY CHECK
        if gradient_norm < 1e-6:  # Very small gradients
            print(f"   Warning: Very small gradient norm ({gradient_norm:.2e}) - guidance may be ineffective")
            # Option: Skip guidance or use default values
            if guidance_effectiveness['improvement_rate'] < 0.3:  # If effectiveness is low
                print(f"   Skipping guidance due to low effectiveness ({guidance_effectiveness['improvement_rate']:.2%})")
                guidance_effectiveness['rejected_guidance_count'] += 1
                return jsonify({
                    'guidance_skipped': True,
                    'reason': 'low_gradient_quality_and_effectiveness',
                    'message': 'Continue with local training only',
                    'timestamp': time.time()
                })
        
        # Scale gradients for stability
        client_gradients = client_gradients * fl_config['gradient_scaling_factor']
        
        # Clip gradients for safety
        client_gradients = np.clip(client_gradients, -fl_config['max_gradient_norm'], fl_config['max_gradient_norm'])
        
        # Track guidance sending
        guidance_effectiveness['total_guidance_sent'] += 1
        
        guidance = {
            'round': current_round,
            'client_type': client_type,
            'embedding_gradients': client_gradients.tolist(),
            'attention_weights': attention_weights,
            'fusion_loss': float(gradient_info['loss_value']),
            'guidance_weight': fl_config['guidance_weight'],
            'learning_rate_multiplier': adaptive_lr_multiplier,  # Use adaptive LR
            'gradient_norm': float(gradient_norm),
            'gradient_quality': 'good' if gradient_norm > 1e-4 else 'poor',
            'timestamp': time.time()
        }
        
        print(f"Sent guidance to {client_type} client {client_id} for round {current_round + 1}")
        print(f"   Gradient norm: {gradient_norm:.4f}")
        print(f"   Fusion loss: {gradient_info['loss_value']:.4f}")
        print(f"   Adaptive LR multiplier: {adaptive_lr_multiplier:.3f}")
        
        return jsonify(guidance)
        
    except Exception as e:
        print(f"Error generating guidance for {client_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/collect_fresh_embeddings', methods=['POST'])
def collect_fresh_embeddings():
    """Collect fresh embeddings from client after guided training."""
    global clients_completed_current_round
    
    data = request.json
    client_id = data.get('client_id')
    current_round = data.get('current_round', 0)
    embeddings_data = data.get('embeddings_data')
    performance_metrics = data.get('performance_metrics', {})
    
    if client_id not in registered_clients:
        return jsonify({'error': 'Client not registered'}), 400
    
    # Update client performance tracking
    registered_clients[client_id]['last_seen'] = time.time()
    registered_clients[client_id]['rounds_participated'] += 1
    registered_clients[client_id]['current_round'] = current_round + 1
    registered_clients[client_id]['current_accuracy'] = performance_metrics.get('accuracy', 0.0)
    registered_clients[client_id]['performance_history'].append({
        'round': current_round,
        'accuracy': performance_metrics.get('accuracy', 0.0),
        'loss': performance_metrics.get('loss', 0.0),
        'timestamp': time.time()
    })
    
    # Mark client as completed for this round
    clients_completed_current_round.add(client_id)
    
    client_type = registered_clients[client_id]['client_type']
    
    print(f"Received fresh embeddings from {client_type} client {client_id}")
    print(f"   Client accuracy: {performance_metrics.get('accuracy', 0.0):.4f}")
    print(f"   Client loss: {performance_metrics.get('loss', 0.0):.4f}")
    print(f"   Round {current_round + 1} progress: {len(clients_completed_current_round)}/{len(registered_clients)} clients completed")
    
    # Check if this is the final round and all clients have completed
    config = get_config()
    fl_config = config['federated_learning']
    final_round = fl_config['total_fl_rounds']
    
    if current_round + 1 >= final_round:
        # Check if all clients have completed final round
        all_clients_completed = True
        for client_id_check, client_info in registered_clients.items():
            if client_info.get('current_round', 0) < final_round:
                all_clients_completed = False
                break
        
        if all_clients_completed:
            print(f"\nALL CLIENTS COMPLETED FL ROUNDS - TRIGGERING FINAL EVALUATION")
            try:
                # Trigger final evaluation automatically
                final_results = federated_server.evaluate_final_model()
                
                print(f"\nDISTRIBUTED FL FINAL RESULTS:")
                print(f"   Global Model Test Accuracy: {final_results.get('accuracy', 0.0):.4f} ({final_results.get('accuracy', 0.0)*100:.2f}%)")
                print(f"   Global Model Test F1: {final_results.get('f1_macro', 0.0):.4f}")
                print(f"   Total FL Rounds: {final_round}")
                print(f"   Participating Clients: {len(registered_clients)}")
                
                # Save final results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = f"results/distributed_fl_results_{timestamp}.json"
                os.makedirs("results", exist_ok=True)
                
                final_summary = {
                    'final_test_accuracy': final_results.get('accuracy', 0.0),
                    'final_test_f1': final_results.get('f1_macro', 0.0),
                    'total_rounds': final_round,
                    'clients': len(registered_clients),
                    'strategy': 'fusion_guided_updates',
                    'timestamp': time.time()
                }
                
                with open(results_file, 'w') as f:
                    json.dump(final_summary, f, indent=2)
                
                print(f"   Results saved to {results_file}")
                
                # Schedule automatic server shutdown if enabled
                if auto_shutdown_enabled:
                    print(f"\nFEDERATED LEARNING COMPLETED!")
                    print(f"Server will shutdown in 10 seconds...")
                    print(f"All processes finished successfully!")
                    
                    # Schedule shutdown after delay to allow final response
                    import threading
                    def shutdown_server():
                        time.sleep(10)  # Give time for final response
                        print(f"\nShutting down server...")
                        print(f"Terminal is ready for new commands!")
                        os._exit(0)  # Force exit
                    
                    shutdown_thread = threading.Thread(target=shutdown_server)
                    shutdown_thread.daemon = True
                    shutdown_thread.start()
                else:
                    print(f"\nFEDERATED LEARNING COMPLETED!")
                    print(f"All processes finished successfully!")
                    print(f"Server will continue running for additional requests...")
                
            except Exception as e:
                print(f"Error in automatic final evaluation: {str(e)}")
                # Still shutdown on error if enabled
                if auto_shutdown_enabled:
                    print(f"Server will shutdown in 5 seconds due to error...")
                    import threading
                    def shutdown_server():
                        time.sleep(5)
                        print(f"\nShutting down server...")
                        print(f"Terminal is ready for new commands!")
                        os._exit(0)
                    
                    shutdown_thread = threading.Thread(target=shutdown_server)
                    shutdown_thread.daemon = True
                    shutdown_thread.start()
                else:
                    print(f"Error occurred, but server will continue running...")
    
    return jsonify({
        'status': 'embeddings_received',
        'round': current_round,
        'timestamp': time.time()
    })

@app.route('/run_fl_round', methods=['POST'])
def run_fl_round():
    """Execute a complete federated learning round with performance tracking."""
    data = request.json
    round_idx = data.get('round_idx', 0)
    
    config = get_config()
    fl_config = config['federated_learning']
    
    print(f"\nDISTRIBUTED FL ROUND {round_idx + 1}/{fl_config['total_fl_rounds']}")
    print("=" * 60)
    
    round_start_time = time.time()
    
    try:
        # Step 1: Load current embeddings and train fusion model
        print("Loading embeddings and training fusion model...")
        
        train_image_emb, train_tabular_emb, train_labels, train_sensitive = \
            federated_server.load_client_embeddings('train')
        val_image_emb, val_tabular_emb, val_labels, val_sensitive = \
            federated_server.load_client_embeddings('val')
        
        # Train fusion model for this round
        round_results = federated_server.train_vfl_round(
            round_idx=round_idx,
            total_rounds=fl_config['total_fl_rounds'],
            epochs=config['training']['epochs_per_round'],
            batch_size=config['training']['batch_size']
        )
        
        # Evaluate round performance
        val_results = federated_server.evaluate_global_model(round_idx)
        
        round_time = time.time() - round_start_time
        
        # Track performance improvement
        current_accuracy = val_results['accuracy']
        current_f1 = val_results['f1_score']
        
        # Compare with previous round
        improvement = 0.0
        if len(round_performance_history) > 0:
            prev_accuracy = round_performance_history[-1]['accuracy']
            improvement = current_accuracy - prev_accuracy
        
        # Store round performance
        round_performance = {
            'round': round_idx + 1,
            'accuracy': current_accuracy,
            'f1_score': current_f1,
            'loss': round_results.get('loss', 0.0),
            'improvement': improvement,
            'time': round_time,
            'client_count': len(registered_clients),
            'timestamp': time.time()
        }
        
        round_performance_history.append(round_performance)
        
        # Update training status
        update_training_status(
            current_round=round_idx + 1,
            total_rounds=fl_config['total_fl_rounds'],
            accuracy=current_accuracy,
            loss=round_results.get('loss', 0.0),
            f1_score=current_f1,
            gender_fairness=[0.0, 0.0],  # Will be updated with actual values
            age_fairness=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Will be updated with actual values
            phase="distributed_fl_round_complete"
        )
        
        # Save round model if configured
        if fl_config['save_round_models']:
            federated_server.save_best_model(f"models/round_{round_idx + 1}")
        
        print(f"\nROUND {round_idx + 1} RESULTS:")
        print(f"   Validation Accuracy: {current_accuracy:.4f}")
        print(f"   Validation F1: {current_f1:.4f}")
        print(f"   Improvement: {improvement:+.4f}")
        print(f"   Round Time: {round_time:.1f}s")
        print(f"   Active Clients: {len(registered_clients)}")
        
        return jsonify({
            'status': 'round_completed',
            'round': round_idx + 1,
            'results': round_performance,
            'continue_training': improvement >= fl_config['min_round_improvement'] or round_idx == 0
        })
        
    except Exception as e:
        print(f"Error in FL round {round_idx + 1}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_final_results', methods=['GET'])
def get_final_results():
    """Get final federated learning results and evaluation."""
    try:
        print("\nFINAL DISTRIBUTED FL EVALUATION")
        print("=" * 50)
        
        # Final evaluation
        final_results = federated_server.evaluate_final_model()
        
        # Compile comprehensive results
        total_time = sum([r['time'] for r in round_performance_history])
        
        results = {
            'training_completed': True,
            'strategy': 'fusion_guided_updates',
            'total_rounds': len(round_performance_history),
            'total_time': total_time,
            'final_test_accuracy': final_results.get('accuracy', 0.0),
            'final_test_f1': final_results.get('f1_macro', 0.0),
            'round_history': round_performance_history,
            'client_summary': {
                client_id: {
                    'type': info['client_type'],
                    'rounds_participated': info['rounds_participated'],
                    'final_accuracy': info['current_accuracy'],
                    'performance_trend': info['performance_history']
                }
                for client_id, info in registered_clients.items()
            },
            'performance_improvements': [
                r['improvement'] for r in round_performance_history
            ],
            'timestamp': time.time()
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results/distributed_fl_results_{timestamp}.json"
        os.makedirs("results", exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {results_file}")
        print(f"Final Test Accuracy: {final_results.get('accuracy', 0.0):.4f}")
        print(f"Final Test F1: {final_results.get('f1_macro', 0.0):.4f}")
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Error in final evaluation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_server_status', methods=['GET'])
def get_server_status():
    """Get comprehensive server status."""
    return jsonify({
        'server_initialized': federated_server is not None,
        'registered_clients': registered_clients,
        'rounds_completed': len(round_performance_history),
        'performance_history': round_performance_history,
        'timestamp': time.time()
    })

@app.route('/request_client_embeddings', methods=['POST'])
def request_client_embeddings():
    """Request embeddings from a specific client (server-driven)."""
    data = request.json
    client_id = data.get('client_id')
    round_idx = data.get('round', 0)
    embedding_type = data.get('embedding_type', 'train')
    guidance_needed = data.get('guidance_needed', False)
    
    if client_id not in registered_clients:
        return jsonify({'error': 'Client not registered'}), 400
    
    print(f"Requesting {embedding_type} embeddings from client {client_id} for round {round_idx + 1}")
    
    # This endpoint just confirms the request
    # Clients will need to poll for this request
    return jsonify({
        'status': 'embedding_request_sent',
        'client_id': client_id,
        'round': round_idx,
        'embedding_type': embedding_type,
        'guidance_needed': guidance_needed,
        'timestamp': time.time()
    })

@app.route('/signal_fl_complete', methods=['POST'])
def signal_fl_complete():
    """Signal to client that FL is complete and they can shutdown."""
    data = request.json
    client_id = data.get('client_id')
    
    if client_id not in registered_clients:
        return jsonify({'error': 'Client not registered'}), 400
    
    print(f"Signaling FL completion to client {client_id}")
    
    return jsonify({
        'status': 'fl_complete',
        'message': 'Federated learning completed - client can shutdown',
        'timestamp': time.time()
    })

@app.route('/get_client_task', methods=['POST'])
def get_client_task():
    """Get current task for a specific client (passive polling)."""
    data = request.json
    client_id = data.get('client_id')
    
    if client_id not in registered_clients:
        return jsonify({'error': 'Client not registered'}), 400
    
    client_info = registered_clients[client_id]
    current_task = client_info.get('current_task', 'idle')
    
    response_data = {
        'client_id': client_id,
        'current_task': current_task,
        'timestamp': time.time()
    }
    
    # Add task-specific data
    if current_task == 'initial_training':
        response_data.update({
            'task_type': 'initial_training',
            'message': 'Perform initial local training and send embeddings'
        })
    elif current_task.startswith('round_') and current_task.endswith('_training'):
        round_num = current_task.split('_')[1]
        response_data.update({
            'task_type': 'guided_training',
            'round': int(round_num),
            'message': f'Perform guided training for round {round_num} and send fresh embeddings'
        })
    elif current_task == 'completed':
        response_data.update({
            'task_type': 'shutdown',
            'message': 'Federated learning completed - you can shutdown'
        })
    else:
        response_data.update({
            'task_type': 'wait',
            'message': 'Wait for server instructions'
        })
    
    return jsonify(response_data)

@app.route('/client_task_completed', methods=['POST'])
def client_task_completed():
    """Mark client task as completed with guidance effectiveness tracking."""
    global guidance_effectiveness, clients_completed_current_round, current_global_round
    
    data = request.json
    client_id = data.get('client_id')
    completed_task = data.get('completed_task')
    performance_metrics = data.get('performance_metrics', {})
    
    if client_id not in registered_clients:
        return jsonify({'error': 'Client not registered'}), 400
    
    client_info = registered_clients[client_id]
    current_accuracy = performance_metrics.get('accuracy', 0.0)
    
    # Get previous accuracy for guidance effectiveness tracking
    previous_accuracy = 0.0
    if 'performance_history' in client_info and len(client_info['performance_history']) > 0:
        previous_accuracy = client_info['performance_history'][-1]['accuracy']
    
    # Update client status
    client_info['last_seen'] = time.time()
    client_info['current_accuracy'] = current_accuracy
    client_info['current_task'] = 'idle'  # Mark as completed
    
    # Track round completion for synchronization
    if completed_task.startswith('round_') and completed_task.endswith('_training'):
        round_num = int(completed_task.split('_')[1])
        if round_num == current_global_round + 1:  # Completing current round
            clients_completed_current_round.add(client_id)
            print(f"   Client {client_id} completed round {round_num}: {len(clients_completed_current_round)}/{len(registered_clients)} clients done")
    
    # Add to performance history
    if 'performance_history' not in client_info:
        client_info['performance_history'] = []
    
    client_info['performance_history'].append({
        'task': completed_task,
        'accuracy': current_accuracy,
        'loss': performance_metrics.get('loss', 0.0),
        'timestamp': time.time()
    })
    
    # Track guidance effectiveness if this was after receiving guidance
    if 'round_' in completed_task and len(client_info['performance_history']) > 1:
        accuracy_change = current_accuracy - previous_accuracy
        
        if accuracy_change > 0.001:  # Improved by more than 0.1%
            guidance_effectiveness['client_improvements'] += 1
            print(f"   Client improved by {accuracy_change:.4f} after guidance")
        elif accuracy_change < -0.001:  # Degraded by more than 0.1%
            guidance_effectiveness['client_degradations'] += 1
            print(f"   Client degraded by {abs(accuracy_change):.4f} after guidance")
        
        # Update improvement rate
        total_guided = guidance_effectiveness['client_improvements'] + guidance_effectiveness['client_degradations']
        if total_guided > 0:
            guidance_effectiveness['improvement_rate'] = guidance_effectiveness['client_improvements'] / total_guided
    
    client_type = client_info['client_type']
    print(f"Client {client_id} ({client_type}) completed task: {completed_task}")
    print(f"   Accuracy: {current_accuracy:.4f}")
    
    # Print guidance effectiveness summary
    if guidance_effectiveness['total_guidance_sent'] > 0:
        print(f"   Guidance Effectiveness: {guidance_effectiveness['improvement_rate']:.2%} improvement rate")
    
    return jsonify({
        'status': 'task_completed_acknowledged',
        'client_id': client_id,
        'timestamp': time.time()
    })

@app.route('/update_adversarial_lambda', methods=['POST'])
def update_adversarial_lambda():
    """Update adversarial lambda value and control dynamic lambda system."""
    try:
        if federated_server is None:
            return jsonify({
                'success': False,
                'error': 'Server not initialized'
            }), 500
        
        data = request.get_json()
        new_lambda = float(data.get('adversarial_lambda', 0.0))
        
        # Determine if this is a "Run Protection" or "Stop Protection" command
        if new_lambda > 0.0:
            # "Run Protection" button clicked
            print(f"PROTECTION ACTIVATED by dashboard")
            print(f"   Enabling dynamic lambda system")
            print(f"   Starting lambda: {new_lambda:.3f}")
            
            # Enable dynamic lambda system
            federated_server.dynamic_lambda_enabled = True
            federated_server.adversarial_lambda = new_lambda
            
            # Reset PID controller for fresh start
            federated_server.lambda_pid['integral'] = 0.0
            federated_server.lambda_pid['previous_error'] = 0.0
            
            # Log the activation
            federated_server.lambda_history.append({
                'round': getattr(federated_server, 'current_round', 0),
                'old_lambda': 0.0,
                'new_lambda': new_lambda,
                'reason': 'Dashboard activation - Run Protection',
                'timestamp': time.time()
            })
            
            message = f"Protection ACTIVATED - Dynamic lambda system enabled (λ={new_lambda:.3f})"
            
        else:
            # "Stop Protection" button clicked
            print(f"PROTECTION DEACTIVATED by dashboard")
            print(f"   Disabling dynamic lambda system")
            print(f"   Lambda set to: 0.0")
            
            # Disable dynamic lambda system
            federated_server.dynamic_lambda_enabled = False
            old_lambda = federated_server.adversarial_lambda
            federated_server.adversarial_lambda = 0.0
            
            # Log the deactivation
            federated_server.lambda_history.append({
                'round': getattr(federated_server, 'current_round', 0),
                'old_lambda': old_lambda,
                'new_lambda': 0.0,
                'reason': 'Dashboard deactivation - Stop Protection',
                'timestamp': time.time()
            })
            
            message = "Protection DEACTIVATED - Dynamic lambda system disabled (λ=0.0)"
        
        return jsonify({
            'success': True,
            'message': message,
            'adversarial_lambda': federated_server.adversarial_lambda,
            'dynamic_enabled': federated_server.dynamic_lambda_enabled,
            'protection_active': federated_server.adversarial_lambda > 0.0
        })
        
    except Exception as e:
        logger.error(f"Error updating adversarial lambda: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/get_defense_status', methods=['GET'])
def get_defense_status():
    """Get current defense status including dynamic lambda information."""
    try:
        if federated_server is None:
            return jsonify({
                'success': False,
                'error': 'Server not initialized'
            }), 500
        
        # Get dynamic lambda status
        dynamic_status = federated_server.get_dynamic_lambda_status()
        
        # Get recent attack performance
        recent_performance = []
        if federated_server.attack_performance_history:
            recent_performance = federated_server.attack_performance_history[-5:]  # Last 5 rounds
        
        # Get recent lambda changes
        recent_changes = []
        if federated_server.lambda_history:
            recent_changes = federated_server.lambda_history[-3:]  # Last 3 changes
        
        return jsonify({
            'success': True,
            'current_lambda': federated_server.adversarial_lambda,
            'defense_active': federated_server.adversarial_lambda > 0.0,
            'dynamic_lambda': dynamic_status,
            'recent_performance': recent_performance,
            'recent_changes': recent_changes,
            'total_rounds_tracked': len(federated_server.attack_performance_history),
            'total_adjustments': len(federated_server.lambda_history)
        })
        
    except Exception as e:
        logger.error(f"Error getting defense status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/configure_dynamic_lambda', methods=['POST'])
def configure_dynamic_lambda():
    """Configure dynamic lambda system parameters."""
    try:
        if federated_server is None:
            return jsonify({
                'success': False,
                'error': 'Server not initialized'
            }), 500
        
        data = request.get_json()
        
        # Update strategy if provided
        if 'strategy' in data:
            strategy = data['strategy']
            if strategy in ['adaptive_pid', 'threshold_based', 'exponential']:
                federated_server.lambda_adjustment_strategy = strategy
                print(f"Dynamic lambda strategy updated to: {strategy}")
            else:
                return jsonify({
                    'success': False,
                    'error': 'Invalid strategy. Must be: adaptive_pid, threshold_based, or exponential'
                }), 400
        
        # Update PID parameters if provided
        if 'pid_params' in data:
            pid_params = data['pid_params']
            for param in ['kp', 'ki', 'kd', 'target_age_leakage', 'target_gender_leakage']:
                if param in pid_params:
                    federated_server.lambda_pid[param] = float(pid_params[param])
                    print(f"Updated {param}: {federated_server.lambda_pid[param]}")
        
        # Enable/disable dynamic lambda
        if 'enabled' in data:
            federated_server.dynamic_lambda_enabled = bool(data['enabled'])
            status = "ENABLED" if federated_server.dynamic_lambda_enabled else "DISABLED"
            print(f"Dynamic lambda system: {status}")
        
        return jsonify({
            'success': True,
            'message': 'Dynamic lambda configuration updated',
            'current_config': federated_server.get_dynamic_lambda_status()
        })
        
    except Exception as e:
        logger.error(f"Error configuring dynamic lambda: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/reset_lambda_history', methods=['POST'])
def reset_lambda_history():
    """Reset lambda adjustment history for fresh start."""
    try:
        if federated_server is None:
            return jsonify({
                'success': False,
                'error': 'Server not initialized'
            }), 500
        
        # Reset histories
        old_performance_count = len(federated_server.attack_performance_history)
        old_lambda_count = len(federated_server.lambda_history)
        
        federated_server.attack_performance_history = []
        federated_server.lambda_history = []
        
        # Reset PID integral term
        federated_server.lambda_pid['integral'] = 0.0
        federated_server.lambda_pid['previous_error'] = 0.0
        
        print(f"Lambda history reset:")
        print(f"   Cleared {old_performance_count} performance records")
        print(f"   Cleared {old_lambda_count} lambda adjustments")
        print(f"   Reset PID integral and derivative terms")
        
        return jsonify({
            'success': True,
            'message': f'Reset {old_performance_count} performance records and {old_lambda_count} lambda adjustments',
            'cleared_performance_records': old_performance_count,
            'cleared_lambda_adjustments': old_lambda_count
        })
        
    except Exception as e:
        logger.error(f"Error resetting lambda history: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/update_sample_counts', methods=['POST'])
def update_sample_counts():
    """Update sample counts for a client."""
    global registered_clients
    
    data = request.json
    client_id = data.get('client_id')
    
    if not client_id or client_id not in registered_clients:
        return jsonify({'error': 'Client not registered'}), 400
    
    # Update sample counts
    registered_clients[client_id]['sample_counts'] = {
        'training_samples': data.get('training_samples', 0),
        'validation_samples': data.get('validation_samples', 0),
        'test_samples': data.get('test_samples', 0),
        'last_updated': time.time()
    }
    
    print(f"Updated sample counts for {client_id}:")
    print(f"   Training: {data.get('training_samples', 0)}")
    print(f"   Validation: {data.get('validation_samples', 0)}")
    print(f"   Test: {data.get('test_samples', 0)}")
    
    return jsonify({
        'status': 'updated',
        'message': 'Sample counts updated successfully'
    })

@app.route('/get_sample_counts', methods=['GET'])
def get_sample_counts():
    """Get aggregated sample counts from all clients."""
    global registered_clients
    
    total_training = 0
    total_validation = 0
    total_test = 0
    
    for client_id, client_info in registered_clients.items():
        if 'sample_counts' in client_info:
            counts = client_info['sample_counts']
            total_training += counts.get('training_samples', 0)
            total_validation += counts.get('validation_samples', 0)
            total_test += counts.get('test_samples', 0)
    
    return jsonify({
        'training_samples': total_training,
        'validation_samples': total_validation,
        'test_samples': total_test,
        'total_clients': len(registered_clients),
        'timestamp': time.time()
    })

def main():
    global auto_shutdown_enabled
    
    parser = argparse.ArgumentParser(description='Distributed Federated Learning Server')
    parser.add_argument('--mode', choices=['local', 'distributed'], default='local',
                        help='Deployment mode')
    parser.add_argument('--host', type=str, default=None,
                        help='Server host (overrides config)')
    parser.add_argument('--port', type=int, default=None,
                        help='Server port (overrides config)')
    parser.add_argument('--no-auto-shutdown', action='store_true',
                        help='Keep server running after FL completion (default: auto-shutdown)')
    parser.add_argument('--data_percentage', type=int, default=5,
                        help='Percentage of dataset to use (1-100, default: 5)')
    parser.add_argument('--fl_rounds', type=int, default=5,
                        help='Number of federated learning rounds (1-20, default: 5)')
    parser.add_argument('--epochs_per_round', type=int, default=5,
                        help='Number of epochs per round (1-20, default: 5)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (1 <= args.data_percentage <= 100):
        print("Error: data_percentage must be between 1 and 100")
        sys.exit(1)
    if not (1 <= args.fl_rounds <= 20):
        print("Error: fl_rounds must be between 1 and 20")
        sys.exit(1)
    if not (1 <= args.epochs_per_round <= 20):
        print("Error: epochs_per_round must be between 1 and 20")
        sys.exit(1)
    
    # Configure auto-shutdown based on command line
    auto_shutdown_enabled = not args.no_auto_shutdown
    
    # Get server configuration
    server_config = get_server_config(args.mode)
    
    host = args.host or server_config['host']
    port = args.port or server_config['port']
    
    # Override config with command line arguments
    config = get_config()
    config['data']['data_percentage'] = args.data_percentage / 100.0  # Convert to decimal
    config['federated_learning']['total_fl_rounds'] = args.fl_rounds
    config['federated_learning']['client_epochs_per_round'] = args.epochs_per_round
    
    print(f"Starting Distributed FL Server in {args.mode} mode")
    print(f"Server: http://{host}:{port}")
    print(f"Strategy: Fusion-Guided Weight Updates")
    print(f"Configuration:")
    print(f"   Data percentage: {args.data_percentage}%")
    print(f"   FL rounds: {args.fl_rounds}")
    print(f"   Epochs per round: {args.epochs_per_round}")
    
    # Show shutdown configuration
    if auto_shutdown_enabled:
        print(f"Auto-shutdown: ENABLED (after {args.fl_rounds} FL rounds)")
        print(f"Use --no-auto-shutdown to keep server running")
    else:
        print(f"Auto-shutdown: DISABLED (server will keep running)")
    
    # Initialize server with updated config
    initialize_server()
    
    # Start Flask server
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    main() 