#!/usr/bin/env python3
"""
Distributed Tabular Client with Fusion-Guided Weight Updates
Implements true federated learning with iterative retraining per round.
"""

import os
import sys
import time
import json
import requests
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime

# Import core components
from network_config import get_client_config
from config import get_config
from status import update_client_status
from data_loader import HAM10000DataLoader
from models import create_tabular_encoder
from train_evaluate import (train_client_model, evaluate_client_model, extract_embeddings, 
                           compute_class_weights)

class DistributedTabularClient:
    """
    Distributed Tabular Client with Fusion-Guided Updates.
    Extends TabularClient to support true federated learning.
    """
    
    def __init__(self, server_host='localhost', server_port=8080,
                 client_id="tabular_client", data_percentage=0.1,
                 learning_rate=0.001, embedding_dim=256, **kwargs):
        # Base TabularClient functionality
        self.client_id = client_id
        self.data_percentage = data_percentage
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        
        # Model and data
        self.encoder = None
        self.data_loader = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        # Metrics
        self.current_accuracy = 0.0
        self.current_f1 = 0.0
        self.current_loss = 0.0
        
        # Distributed functionality
        self.server_host = server_host
        self.server_port = server_port
        self.server_url = f"http://{server_host}:{server_port}"
        self.is_registered = False
        self.current_round = 0
        self.fl_config = {}
        
        print(f"Distributed Tabular Client Initialized")
        print(f"Server: {self.server_url}")
        print(f"Client ID: {self.client_id}")
        print(f"Data percentage: {self.data_percentage*100:.1f}%")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def load_data(self, data_dir="data"):
        """Load and preprocess HAM10000 dataset for tabular client."""
        print(f"\nLoading data for {self.client_id}...")
        
        self.data_loader = HAM10000DataLoader(data_dir=data_dir, random_state=42)
        self.data_loader.load_and_preprocess_data(data_percentage=self.data_percentage)
        
        # Get tabular client data
        tabular_data = self.data_loader.get_tabular_client_data()
        
        self.train_data = {
            'features': tabular_data['train']['features'],
            'labels': tabular_data['train']['labels'],
            'sensitive_attrs': tabular_data['train']['sensitive_attrs'],
            'indices': tabular_data['train']['indices']
        }
        
        self.val_data = {
            'features': tabular_data['val']['features'],
            'labels': tabular_data['val']['labels'],
            'sensitive_attrs': tabular_data['val']['sensitive_attrs'],
            'indices': tabular_data['val']['indices']
        }
        
        self.test_data = {
            'features': tabular_data['test']['features'],
            'labels': tabular_data['test']['labels'],
            'sensitive_attrs': tabular_data['test']['sensitive_attrs'],
            'indices': tabular_data['test']['indices']
        }
        
        print(f"Data loaded successfully")
        print(f"   - Train samples: {len(self.train_data['features'])}")
        print(f"   - Validation samples: {len(self.val_data['features'])}")
        print(f"   - Test samples: {len(self.test_data['features'])}")

    def create_model(self, use_step3_enhancements=True, use_lightweight=True):
        """
        Create the tabular encoder model.
        
        Args:
            use_step3_enhancements (bool): Whether to use Step 3 generalization enhancements
            use_lightweight (bool): Whether to use lightweight model
        """
        print("Creating tabular encoder model...")
        
        # Get feature dimension from loaded data
        if hasattr(self, 'train_data') and self.train_data is not None:
            input_dim = self.train_data['features'].shape[1]
        else:
            # Default input dimension for HAM10000 tabular data
            input_dim = 12  # age, sex, localization, dx_type, etc.
        
        # Create enhanced tabular encoder
        self.encoder = create_tabular_encoder(
            input_dim=input_dim,
            embedding_dim=self.embedding_dim,
            use_step3_enhancements=use_step3_enhancements
        )
        
        print(f"Model created with {self.encoder.count_params():,} parameters")

    def generate_embeddings(self, data_split='train'):
        """
        Generate embeddings for a specific data split.
        
        Args:
            data_split (str): 'train', 'val', or 'test'
        
        Returns:
            tuple: (embeddings, labels, indices)
        """
        print(f"Generating {data_split} embeddings...")
        
        if data_split == 'train':
            data = self.train_data
        elif data_split == 'val':
            data = self.val_data
        elif data_split == 'test':
            data = self.test_data
        else:
            raise ValueError(f"Invalid data_split: {data_split}")
        
        # Generate embeddings
        features = np.array(data['features'])
        embeddings = extract_embeddings(self.encoder, features, batch_size=32)
        
        print(f"Generated embeddings: {embeddings.shape}")
        return embeddings, np.array(data['labels']), np.array(data['indices'])

    def save_embeddings(self, embeddings, labels, indices, data_split='train', output_dir='embeddings'):
        """Save embeddings to disk."""
        import pickle
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get sensitive attributes for the same indices
        if data_split == 'train':
            sensitive_attrs = self.train_data['sensitive_attrs']
        elif data_split == 'val':
            sensitive_attrs = self.val_data['sensitive_attrs']
        elif data_split == 'test':
            sensitive_attrs = self.test_data['sensitive_attrs']
        else:
            sensitive_attrs = None
        
        embedding_data = {
            'embeddings': embeddings,
            'labels': labels,
            'indices': indices,
            'sensitive_attrs': sensitive_attrs,  # Add sensitive attributes
            'client_id': self.client_id,
            'data_split': data_split,
            'embedding_dim': self.embedding_dim,
            'timestamp': time.time()
        }
        
        filename = f"{self.client_id}_{data_split}_embeddings.pkl"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(embedding_data, f)
        
        print(f"Saved {data_split} embeddings to {filepath}")

    def train_local_model(self, epochs=10, batch_size=16, verbose=1):
        """Train the local model."""
        if not hasattr(self, 'train_data') or self.train_data is None:
            raise ValueError("No training data available. Load data first.")
        
        print(f"Training local model...")
        print(f"   Training samples: {len(self.train_data['labels'])}")
        print(f"   Epochs: {epochs}")
        
        # Prepare training data
        train_features = np.array(self.train_data['features'])
        train_labels = np.array(self.train_data['labels'])
        val_features = np.array(self.val_data['features']) if hasattr(self, 'val_data') else None
        val_labels = np.array(self.val_data['labels']) if hasattr(self, 'val_data') else None
        
        # Create training model with classification head
        encoder_input = self.encoder.input
        encoder_output = self.encoder.output
        
        # Add classification head for training
        from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
        from tensorflow.keras.models import Model
        
        classifier_head = Dense(64, activation='relu', kernel_initializer='he_normal')(encoder_output)
        classifier_head = BatchNormalization()(classifier_head)
        classifier_head = Dropout(0.4)(classifier_head)
        classifier_predictions = Dense(7, activation='softmax', name='predictions')(classifier_head)
        
        # Create training model
        training_model = Model(inputs=encoder_input, outputs=classifier_predictions, name='local_training_model')
        
        # Compile model
        training_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Prepare validation data
        validation_data = None
        if val_features is not None and val_labels is not None:
            validation_data = (val_features, val_labels)
        
        # Train model
        history = training_model.fit(
            train_features, train_labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            verbose=verbose,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy' if validation_data else 'accuracy',
                    patience=3,
                    restore_best_weights=True
                )
            ]
        )
        
        # Update encoder weights from trained model
        encoder_layers = len(self.encoder.layers)
        for i in range(encoder_layers):
            self.encoder.layers[i].set_weights(training_model.layers[i].get_weights())
        
        # Extract metrics
        final_accuracy = 0.0
        final_loss = 0.0
        
        if hasattr(history, 'history'):
            if 'val_accuracy' in history.history:
                final_accuracy = max(history.history['val_accuracy'])
                final_loss = min(history.history['val_loss'])
            elif 'accuracy' in history.history:
                final_accuracy = max(history.history['accuracy'])
                final_loss = min(history.history['loss'])
        
        return {
            'accuracy': final_accuracy,
            'loss': final_loss,
            'history': history.history if hasattr(history, 'history') else {}
        }

    def send_sample_counts(self):
        """Send sample counts to server for dashboard display."""
        try:
            sample_counts = {
                'client_id': self.client_id,
                'client_type': 'tabular',
                'training_samples': len(self.train_data['features']),
                'validation_samples': len(self.val_data['features']),
                'test_samples': len(self.test_data['features'])
            }
            
            response = requests.post(f"{self.server_url}/update_sample_counts", json=sample_counts)
            
            if response.status_code == 200:
                print(f"Sample counts sent to server")
                return True
            else:
                print(f"Failed to send sample counts: {response.text}")
                return False
                
        except Exception as e:
            print(f"Sample counts error: {str(e)}")
            return False

    def register_with_server(self):
        """Register this client with the federated server."""
        try:
            response = requests.post(f"{self.server_url}/register_client", json={
                'client_id': self.client_id,
                'client_type': 'tabular'
            })
            
            if response.status_code == 200:
                result = response.json()
                self.is_registered = True
                print(f"Registered with server")
                print(f"   Total clients: {result['total_clients']}")
                
                # Send sample counts after successful registration
                self.send_sample_counts()
                
                return True
            else:
                print(f"Registration failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"Registration error: {str(e)}")
            return False
    
    def get_fl_config(self):
        """Get federated learning configuration from server."""
        try:
            response = requests.get(f"{self.server_url}/get_fl_config")
            
            if response.status_code == 200:
                self.fl_config = response.json()
                print(f"📥 Received FL config:")
                print(f"   Rounds: {self.fl_config['total_fl_rounds']}")
                print(f"   Epochs per round: {self.fl_config['client_epochs_per_round']}")
                print(f"   Learning rate multiplier: {self.fl_config['client_learning_rate_multiplier']}")
                return True
            else:
                print(f"Failed to get FL config: {response.text}")
                return False
                
        except Exception as e:
            print(f"FL config error: {str(e)}")
            return False
    
    def receive_global_guidance(self, current_round):
        """
        Receive fusion-guided updates from server.
        INCLUDES ROUND SYNCHRONIZATION - waits for all clients to complete previous round.
        
        Args:
            current_round (int): Current FL round
            
        Returns:
            dict: Server guidance including gradients and attention weights
        """
        max_wait_attempts = 20  # Maximum waiting attempts
        wait_attempts = 0
        
        while wait_attempts < max_wait_attempts:
            try:
                response = requests.post(f"{self.server_url}/send_global_guidance", json={
                    'client_id': self.client_id,
                    'current_round': current_round
                })
                
                if response.status_code == 200:
                    guidance = response.json()
                    print(f"Received guidance for round {current_round + 1}")
                    print(f"   Gradient shape: {np.array(guidance['embedding_gradients']).shape}")
                    print(f"   Fusion loss: {guidance['fusion_loss']:.4f}")
                    print(f"   Guidance weight: {guidance['guidance_weight']}")
                    return guidance
                    
                elif response.status_code == 423:  # Round synchronization wait
                    sync_info = response.json()
                    wait_time = sync_info.get('wait_time', 5)
                    completed = sync_info.get('clients_completed', 0)
                    total = sync_info.get('total_clients', 1)
                    
                    print(f"Round synchronization: waiting for other clients...")
                    print(f"   Progress: {completed}/{total} clients completed round {current_round}")
                    print(f"   Waiting {wait_time} seconds... (attempt {wait_attempts + 1}/{max_wait_attempts})")
                    
                    time.sleep(wait_time)
                    wait_attempts += 1
                    continue
                    
                else:
                    print(f"Failed to receive guidance: {response.text}")
                    return None
                    
            except Exception as e:
                print(f"Guidance error: {str(e)}")
                return None
        
        print(f"Timeout waiting for round synchronization after {max_wait_attempts} attempts")
        return None
    
    def apply_guided_training(self, server_guidance, epochs=3):
        """
        Train local model with server guidance.
        
        Args:
            server_guidance (dict): Guidance from server including gradients
            epochs (int): Number of epochs for guided training
            
        Returns:
            dict: Training results
        """
        if not hasattr(self, 'train_data') or self.train_data is None:
            raise ValueError("No training data available. Load data first.")
        
        print(f"\nGUIDED TRAINING - Round {self.current_round + 1}")
        print(f"   Training samples: {len(self.train_data['labels'])}")
        print(f"   Epochs: {epochs}")
        print(f"   Guidance weight: {server_guidance['guidance_weight']}")
        
        # Prepare training data
        train_features = np.array(self.train_data['features'])
        train_labels = np.array(self.train_data['labels'])
        val_features = np.array(self.val_data['features']) if hasattr(self, 'val_data') else None
        val_labels = np.array(self.val_data['labels']) if hasattr(self, 'val_data') else None
        
        # Extract guidance parameters
        embedding_gradients = np.array(server_guidance['embedding_gradients'])
        guidance_weight = server_guidance['guidance_weight']
        lr_multiplier = server_guidance['learning_rate_multiplier']
        
        # Create guided training model
        encoder_input = self.encoder.input
        encoder_output = self.encoder.output
        
        # Add classification head for training
        from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
        from tensorflow.keras.models import Model
        
        classifier_head = Dense(64, activation='relu', kernel_initializer='he_normal')(encoder_output)
        classifier_head = BatchNormalization()(classifier_head)
        classifier_head = Dropout(0.4)(classifier_head)
        classifier_predictions = Dense(7, activation='softmax', name='predictions')(classifier_head)
        
        # Create training model
        training_model = Model(inputs=encoder_input, outputs=classifier_predictions, name='guided_training_model')
        
        # Compile with adjusted learning rate
        base_lr = self.fl_config.get('base_learning_rate', 0.001)
        adjusted_lr = base_lr * lr_multiplier
        
        training_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=adjusted_lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"   Guided training model: {training_model.count_params():,} parameters")
        print(f"   Adjusted learning rate: {adjusted_lr:.6f}")
        
        # Prepare validation data
        validation_data = None
        if val_features is not None and val_labels is not None:
            validation_data = (val_features, val_labels)
        
        # Train with guidance
        print(f"   Starting guided training...")
        try:
            history = training_model.fit(
                train_features, train_labels,
                batch_size=16,
                epochs=epochs,
                validation_data=validation_data,
                verbose=1,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_accuracy' if validation_data else 'accuracy',
                        patience=3,
                        restore_best_weights=True
                    )
                ]
            )
            
            # Extract the trained encoder weights
            for i, layer in enumerate(self.encoder.layers):
                if i < len(training_model.layers) - 3:  # Exclude classification head
                    if len(layer.get_weights()) > 0:
                        layer.set_weights(training_model.layers[i].get_weights())
            
            # Evaluate final performance
            final_train_acc = max(history.history['accuracy'])
            final_val_acc = max(history.history.get('val_accuracy', [0]))
            final_loss = min(history.history['loss'])
            
            print(f"   Guided training completed!")
            print(f"   Best training accuracy: {final_train_acc:.4f}")
            if validation_data:
                print(f"   Best validation accuracy: {final_val_acc:.4f}")
            
            return {
                'accuracy': final_val_acc if validation_data else final_train_acc,
                'loss': final_loss,
                'epochs_completed': len(history.history['loss']),
                'guided_training': True
            }
            
        except Exception as e:
            print(f"   Guided training failed: {str(e)}")
            return {
                'error': str(e),
                'accuracy': 0.0,
                'loss': float('inf'),
                'epochs_completed': 0
            }
    
    def generate_and_send_fresh_embeddings(self, current_round, performance_metrics):
        """
        Generate fresh embeddings and send to server.
        
        Args:
            current_round (int): Current FL round
            performance_metrics (dict): Local training performance
        """
        print(f"   Generating fresh embeddings after guided training...")
        
        # Generate embeddings for all splits
        train_embeddings, train_labels_emb, train_indices = self.generate_embeddings('train')
        val_embeddings, val_labels_emb, val_indices = self.generate_embeddings('val')
        test_embeddings, test_labels_emb, test_indices = self.generate_embeddings('test')
        
        # Save embeddings locally
        self.save_embeddings(train_embeddings, train_labels_emb, train_indices, 'train')
        self.save_embeddings(val_embeddings, val_labels_emb, val_indices, 'val')
        self.save_embeddings(test_embeddings, test_labels_emb, test_indices, 'test')
        
        # Notify server of fresh embeddings
        try:
            response = requests.post(f"{self.server_url}/collect_fresh_embeddings", json={
                'client_id': self.client_id,
                'current_round': current_round,
                'performance_metrics': performance_metrics,
                'embeddings_data': {
                    'train_shape': train_embeddings.shape,
                    'val_shape': val_embeddings.shape,
                    'test_shape': test_embeddings.shape
                }
            })
            
            if response.status_code == 200:
                print(f"   Server notified of fresh embeddings")
            else:
                print(f"   Failed to notify server: {response.text}")
                
        except Exception as e:
            print(f"   Error notifying server: {str(e)}")
    
    def run_passive_client(self):
        """Run as passive client - wait for server tasks and respond."""
        if not self.is_registered:
            print("Client not registered with server")
            return False
        
        print(f"\nSTARTING PASSIVE CLIENT MODE")
        print(f"Waiting for server instructions...")
        print("=" * 50)
        
        while True:
            try:
                # Poll server for current task
                response = requests.post(f"{self.server_url}/get_client_task", json={
                    'client_id': self.client_id
                })
                
                if response.status_code == 200:
                    task_info = response.json()
                    task_type = task_info.get('task_type', 'wait')
                    current_task = task_info.get('current_task', 'idle')
                    
                    if task_type == 'initial_training':
                        print(f"\nRECEIVED TASK: Initial Training")
                        self.perform_initial_training()
                        
                    elif task_type == 'guided_training':
                        round_num = task_info.get('round', 0)
                        print(f"\nRECEIVED TASK: Guided Training - Round {round_num}")
                        self.perform_guided_training(round_num)
                        
                    elif task_type == 'shutdown':
                        print(f"\nRECEIVED SHUTDOWN SIGNAL")
                        print(f"Federated learning completed!")
                        print(f"Tabular client shutting down...")
                        return True
                        
                    else:  # wait
                        time.sleep(5)  # Wait 5 seconds before polling again
                        
                else:
                    print(f"Failed to get task from server: {response.text}")
                    time.sleep(10)
                    
            except Exception as e:
                print(f"Error communicating with server: {e}")
                time.sleep(10)
        
        return False
    
    def perform_initial_training(self):
        """Perform initial training and send embeddings to server."""
        print(f"PERFORMING INITIAL TRAINING")
        
        try:
            # Load FL config if not already loaded
            if not self.fl_config:
                self.get_fl_config()
            
            # Perform initial training
            print(f"   Training samples: {len(self.train_data['labels'])}")
            print(f"   Epochs: 2")
            
            # Train model
            train_results = self.train_local_model(epochs=2, batch_size=64, verbose=1)
            
            if 'error' in train_results:
                print(f"   Training failed: {train_results['error']}")
                self.notify_task_completion('initial_training', {'error': train_results['error']})
                return
            
            # Generate and save embeddings
            self.generate_and_save_embeddings()
            
            # Get performance metrics
            final_accuracy = train_results.get('final_val_acc', 0.0)
            final_loss = train_results.get('final_val_loss', 0.0)
            
            # Notify server of completion
            self.notify_task_completion('initial_training', {
                'accuracy': final_accuracy,
                'loss': final_loss
            })
            
            print(f"   Initial training completed!")
            print(f"   Final accuracy: {final_accuracy:.4f}")
            
        except Exception as e:
            print(f"   Initial training failed: {e}")
            self.notify_task_completion('initial_training', {'error': str(e)})
    
    def perform_guided_training(self, round_num):
        """Perform guided training for a specific round."""
        print(f"PERFORMING GUIDED TRAINING - Round {round_num}")
        
        try:
            # Get guidance from server
            guidance = self.get_guidance_from_server(round_num - 1)
            if not guidance:
                print(f"   Failed to get guidance from server")
                return
            
            # Perform guided training
            results = self.apply_guided_training(guidance, epochs=2)
            
            # Generate fresh embeddings
            self.generate_and_save_embeddings()
            
            # Notify server of completion
            self.notify_task_completion(f'round_{round_num}_training', results)
            
            print(f"   Guided training completed for round {round_num}!")
            
        except Exception as e:
            print(f"   Guided training failed: {e}")
            self.notify_task_completion(f'round_{round_num}_training', {'error': str(e)})
    
    def get_guidance_from_server(self, round_idx):
        """Get guidance from server for guided training."""
        max_retries = 10
        retry_delay = 10  # seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.post(f"{self.server_url}/send_global_guidance", json={
                    'client_id': self.client_id,
                    'current_round': round_idx
                })
                
                if response.status_code == 200:
                    guidance = response.json()
                    print(f"   Received guidance for round {round_idx + 1}")
                    return guidance
                elif response.status_code == 423:  # Locked - need to wait for synchronization
                    response_data = response.json()
                    wait_time = response_data.get('wait_time', 5)
                    print(f"   Round synchronization - waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"   Failed to get guidance: {response.text}")
                    return None
                    
            except Exception as e:
                print(f"   Error getting guidance: {e}")
                if attempt < max_retries - 1:
                    print(f"   Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue
                return None
        
        print(f"   Max retries exceeded waiting for guidance")
        return None
    
    def generate_and_save_embeddings(self):
        """Generate embeddings and save them locally."""
        print(f"   Generating and saving embeddings...")
        
        # Generate embeddings for all splits
        train_embeddings, train_labels_emb, train_indices = self.generate_embeddings('train')
        val_embeddings, val_labels_emb, val_indices = self.generate_embeddings('val')
        test_embeddings, test_labels_emb, test_indices = self.generate_embeddings('test')
        
        # Save embeddings locally
        self.save_embeddings(train_embeddings, train_labels_emb, train_indices, 'train')
        self.save_embeddings(val_embeddings, val_labels_emb, val_indices, 'val')
        self.save_embeddings(test_embeddings, test_labels_emb, test_indices, 'test')
        
        print(f"   Embeddings generated and saved")
    
    def notify_task_completion(self, task_name, performance_metrics):
        """Notify server that a task has been completed."""
        try:
            response = requests.post(f"{self.server_url}/client_task_completed", json={
                'client_id': self.client_id,
                'completed_task': task_name,
                'performance_metrics': performance_metrics
            })
            
            if response.status_code == 200:
                print(f"   Notified server of task completion: {task_name}")
            else:
                print(f"   Failed to notify server: {response.text}")
                
        except Exception as e:
            print(f"   Error notifying server: {e}")

    def run_fl_training(self):
        """Run the complete federated learning training process."""
        if not self.is_registered:
            print("Client not registered with server")
            return False
        
        if not self.fl_config:
            print("FL configuration not loaded")
            return False
        
        print(f"\nSTARTING FEDERATED LEARNING")
        print(f"   Rounds: {self.fl_config['total_fl_rounds']}")
        print(f"   Epochs per round: {self.fl_config['client_epochs_per_round']}")
        print("=" * 50)
        
        # Initial training (Round 0)
        print(f"\nINITIAL TRAINING (Before FL rounds)")
        initial_results = self.train_local_model(
            epochs=self.fl_config['client_epochs_per_round'],
            batch_size=16,
            verbose=1
        )
        
        if 'error' in initial_results:
            print(f"Initial training failed: {initial_results['error']}")
            return False
        
        print(f"Initial training completed")
        print(f"   Initial accuracy: {initial_results['final_val_acc']:.4f}")
        
        # Federated learning rounds
        for round_idx in range(self.fl_config['total_fl_rounds']):
            self.current_round = round_idx
            
            print(f"\nFL ROUND {round_idx + 1}/{self.fl_config['total_fl_rounds']}")
            print("=" * 40)
            
            # Step 1: Receive guidance from server
            guidance = self.receive_global_guidance(round_idx)
            if guidance is None:
                print(f"Failed to receive guidance for round {round_idx + 1}")
                continue
            
            # Step 2: Apply guided training
            guided_results = self.apply_guided_training(
                guidance, 
                epochs=self.fl_config['client_epochs_per_round']
            )
            
            if 'error' in guided_results:
                print(f"Guided training failed: {guided_results['error']}")
                continue
            
            # Step 3: Generate and send fresh embeddings
            self.generate_and_send_fresh_embeddings(round_idx, guided_results)
            
            # Update status
            update_client_status(
                client_id=self.client_id,
                accuracy=guided_results['accuracy'],
                f1_score=0.0,  # Not computed at client level
                loss=guided_results['loss'],
                embeddings_sent=True,
                weights_updated=True
            )
            
            print(f"Round {round_idx + 1} completed")
            print(f"   Accuracy: {guided_results['accuracy']:.4f}")
            print(f"   Loss: {guided_results['loss']:.4f}")
        
        print(f"\nFEDERATED LEARNING COMPLETED!")
        print(f"   Total rounds: {self.fl_config['total_fl_rounds']}")
        print(f"   Final accuracy: {guided_results['accuracy']:.4f}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Distributed Tabular Client')
    parser.add_argument('--mode', choices=['local', 'distributed'], default='local',
                        help='Deployment mode')
    parser.add_argument('--server_host', type=str, default=None,
                        help='Server host (overrides config)')
    parser.add_argument('--server_port', type=int, default=None,
                        help='Server port (overrides config)')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory')
    
    args = parser.parse_args()
    
    # Get client configuration
    client_config = get_client_config('tabular', args.mode)
    config = get_config()
    
    server_host = args.server_host or client_config['server_host']
    server_port = args.server_port or client_config['server_port']
    
    print(f"Starting Distributed Tabular Client in {args.mode} mode")
    print(f"Server: http://{server_host}:{server_port}")
    print(f"Strategy: Fusion-Guided Weight Updates")
    
    # Create client
    client = DistributedTabularClient(
        server_host=server_host,
        server_port=server_port,
        client_id="tabular_client",
        data_percentage=config['data']['data_percentage'],
        learning_rate=config['training']['client_learning_rate'],
        embedding_dim=config['model']['embedding_dim']
    )
    
    try:
        # Step 1: Load data
        print("\nLoading data...")
        client.load_data(data_dir=args.data_dir)
        
        # Step 2: Create model
        print("Creating model...")
        client.create_model(use_step3_enhancements=True)
        
        # Step 3: Register with server
        print("Registering with server...")
        if not client.register_with_server():
            print("Failed to register with server")
            return 1
        
        # Step 4: Get FL configuration
        print("Getting FL configuration...")
        if not client.get_fl_config():
            print("Failed to get FL configuration")
            return 1
        
        # Step 5: Run passive client mode
        print("Starting passive client mode...")
        print(f"   Waiting for server instructions...")
        print(f"   Client ID: {client.client_id}")
        print(f"   Training samples: {len(client.train_data['labels'])}")
        
        if not client.run_passive_client():
            print("Passive client failed")
            return 1
        
        print("Distributed Tabular Client completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 