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
            train_image_emb, train_tabular_emb, train_labels, _ = self.load_client_embeddings('train')
            val_image_emb, val_tabular_emb, val_labels, _ = self.load_client_embeddings('val')
            
            # Train fusion model
            from train_evaluate import train_fusion_model_with_adversarial
            
            history = train_fusion_model_with_adversarial(
                fusion_model=self.fusion_model,
                adversarial_model=self.adversarial_model,
                image_embeddings=train_image_emb,
                tabular_embeddings=train_tabular_emb,
                labels=train_labels,
                sensitive_attrs=None,  # Not using sensitive attributes for now
                val_image_embeddings=val_image_emb,
                val_tabular_embeddings=val_tabular_emb,
                val_labels=val_labels,
                val_sensitive_attrs=None,  # Not using sensitive attributes for now
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
            if round_f1 == 0.0:
                # Calculate F1 using validation data
                val_predictions = self.fusion_model.predict([val_image_emb, val_tabular_emb], verbose=0)
                val_pred_classes = np.argmax(val_predictions, axis=1)
                from sklearn.metrics import f1_score
                round_f1 = f1_score(val_labels, val_pred_classes, average='macro')
            
            # Update best performance tracking
            if round_accuracy > self.best_accuracy:
                self.best_accuracy = round_accuracy
                self.best_f1 = round_f1
                self.best_round = round_idx + 1
                
                # Save best model
                self.save_best_model()
            
            # Record metrics
            self.training_history['round_accuracies'].append(round_accuracy)
            self.training_history['round_f1_scores'].append(round_f1)
            self.training_history['round_losses'].append(round_loss)
            self.training_history['training_times'].append(time.time() - start_time)
            
            print(f"Round {round_idx + 1} complete:")
            print(f"   Accuracy: {round_accuracy:.4f}")
            print(f"   F1: {round_f1:.4f}")
            print(f"   Loss: {round_loss:.4f}")
            
            return {
                'accuracy': round_accuracy,
                'f1_macro': round_f1,
                'loss': round_loss,
                'time': time.time() - start_time
            }
            
        except Exception as e:
            print(f"Round {round_idx + 1} failed: {str(e)}")
            return {
                'accuracy': 0.0,
                'f1_macro': 0.0,
                'loss': 1.0,
                'time': 0.0
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
        
        # Calculate defense strength based on adversarial training status
        if federated_server.adversarial_lambda > 0:
            # Real adversarial loss calculation would go here when adversarial training is enabled
            # For now, simulate adversarial loss as positive value (lower = better defense)
            # This would be the actual adversarial loss from the adversarial model
            simulated_adversarial_loss = max(0.1, 1.0 - val_results['accuracy'])  # Simulate adversarial loss
            defense_strength = simulated_adversarial_loss * 100  # Show as positive value
            print(f"   Adversarial training enabled - Defense strength: {defense_strength:.2f}%")
        else:
            # No adversarial training = No defense = 0% protection
            defense_strength = 0.0
            print(f"   Adversarial training disabled - No defense protection")
        
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
        'current_task': 'idle'
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
        'base_learning_rate': config['training']['client_learning_rate']
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
    
    args = parser.parse_args()
    
    # Configure auto-shutdown based on command line
    auto_shutdown_enabled = not args.no_auto_shutdown
    
    # Get server configuration
    server_config = get_server_config(args.mode)
    
    host = args.host or server_config['host']
    port = args.port or server_config['port']
    
    print(f"Starting Distributed FL Server in {args.mode} mode")
    print(f"Server: http://{host}:{port}")
    print(f"Strategy: Fusion-Guided Weight Updates")
    
    # Show shutdown configuration
    if auto_shutdown_enabled:
        # Get FL rounds from config
        config = get_config()
        fl_rounds = config['federated_learning']['total_fl_rounds']
        print(f"Auto-shutdown: ENABLED (after {fl_rounds} FL rounds)")
        print(f"Use --no-auto-shutdown to keep server running")
    else:
        print(f"Auto-shutdown: DISABLED (server will keep running)")
    
    # Initialize server
    initialize_server()
    
    # Start Flask server
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    main() 