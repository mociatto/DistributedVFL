"""
Standalone Tabular Client for Vertical Federated Learning.
Processes patient metadata (age, gender, lesion location) and communicates embeddings to the federated server.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from data_loader import HAM10000DataLoader
from models import create_tabular_encoder
from train_evaluate import (
    create_tabular_data_generator,
    train_client_model,
    evaluate_client_model,
    compute_class_weights,
    extract_embeddings
)
from status import update_client_status
import pickle
import argparse
import time
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras import Model

# Conditional Flask import for server mode
try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


class TabularClient:
    """
    Tabular client for vertical federated learning.
    Handles patient metadata processing and embedding generation.
    """
    
    def __init__(self, client_id="tabular_client", data_percentage=0.1,
                 learning_rate=0.001, embedding_dim=256):
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
        
        print(f"📋 Tabular Client Initialized")
        print(f"   Client ID: {self.client_id}")
        print(f"   Data percentage: {self.data_percentage*100:.1f}%")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Embedding dimension: {self.embedding_dim}")
    
    def load_data(self, data_dir="data"):
        """Load and preprocess HAM10000 dataset for tabular client."""
        print(f"\n📊 Loading data for {self.client_id}...")
        
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
        
        print(f"   ✅ Data loaded successfully")
        print(f"   - Train samples: {len(self.train_data['features'])}")
        print(f"   - Validation samples: {len(self.val_data['features'])}")
        print(f"   - Test samples: {len(self.test_data['features'])}")
        print(f"   - Feature dimension: {self.train_data['features'].shape[1]}")
    
    def create_model(self):
        """Create and compile the tabular encoder model."""
        print(f"\n🏗️  Creating tabular encoder model...")
        
        input_dim = self.train_data['features'].shape[1]
        
        self.encoder = create_tabular_encoder(
            input_dim=input_dim,
            embedding_dim=self.embedding_dim
        )
        
        # Compile model
        self.encoder.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"   ✅ Model created with {self.encoder.count_params():,} parameters")
    
    def train(self, epochs=10, batch_size=32, patience=3):
        """Train the tabular encoder model."""
        print(f"\n🎯 Training {self.client_id}...")
        print(f"   📊 Dataset: {len(self.train_data['labels'])} train, {len(self.val_data['labels'])} val samples")
        print(f"   ⚙️  Config: {epochs} epochs, batch size {batch_size}")
        
        train_features = self.train_data['features']
        train_labels = self.train_data['labels']
        
        val_features = self.val_data['features']
        val_labels = self.val_data['labels']
        
        # Compute class weights for balanced dataset
        class_weights = compute_class_weights(train_labels, method='balanced')
        print(f"   ⚖️  Class weights computed for {len(set(train_labels))} classes")
        
        # Create data generators
        train_generator = create_tabular_data_generator(
            train_features,
            train_labels,
            batch_size=batch_size,
            shuffle=True
        )
        
        steps_per_epoch = len(train_labels) // batch_size
        print(f"   🔄 Training setup: {steps_per_epoch} steps per epoch")
        
        # Train model with progress tracking
        print(f"   🚀 Starting training...")
        history = train_client_model(
            model=self.encoder,
            train_generator=train_generator,
            val_data=(val_features, val_labels),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            class_weights=class_weights,
            patience=patience,
            verbose=2  # More verbose output
        )
        
        # Update metrics
        if hasattr(history, 'history') and 'val_accuracy' in history.history:
            self.current_accuracy = max(history.history['val_accuracy'])
            self.current_loss = min(history.history['val_loss'])
            print(f"   📈 Best validation accuracy: {self.current_accuracy:.4f}")
            print(f"   📉 Best validation loss: {self.current_loss:.4f}")
        
        print(f"   ✅ Training completed successfully")
        return history
    
    def evaluate(self):
        """Evaluate the model on test data."""
        print(f"\n📊 Evaluating {self.client_id}...")
        
        test_features = self.test_data['features']
        test_labels = self.test_data['labels']
        
        # Evaluate model
        results = evaluate_client_model(
            model=self.encoder,
            test_data=(test_features, test_labels),
            class_names=self.data_loader.get_class_names(),
            verbose=1
        )
        
        # Update metrics
        self.current_accuracy = results['accuracy']
        self.current_f1 = results['f1_macro']
        
        # Update status
        update_client_status(
            client_id=self.client_id,
            accuracy=self.current_accuracy,
            f1_score=self.current_f1,
            loss=self.current_loss
        )
        
        return results
    
    def generate_embeddings(self, data_split='train'):
        """
        Generate embeddings for a specific data split.
        
        Args:
            data_split (str): 'train', 'val', or 'test'
        
        Returns:
            tuple: (embeddings, labels, indices)
        """
        print(f"\n🔄 Generating {data_split} embeddings...")
        
        if data_split == 'train':
            data = self.train_data
        elif data_split == 'val':
            data = self.val_data
        elif data_split == 'test':
            data = self.test_data
        else:
            raise ValueError(f"Invalid data_split: {data_split}")
        
        # Generate embeddings
        embeddings = extract_embeddings(self.encoder, data['features'], batch_size=32)
        
        print(f"   ✅ Generated embeddings: {embeddings.shape}")
        
        return embeddings, data['labels'], data['indices']
    
    def save_embeddings(self, embeddings, labels, indices, data_split='train',
                       output_dir='embeddings'):
        """Save embeddings to file for server communication."""
        os.makedirs(output_dir, exist_ok=True)
        
        data = {
            'embeddings': embeddings,
            'labels': labels,
            'indices': indices,
            'client_id': self.client_id,
            'data_split': data_split,
            'embedding_dim': self.embedding_dim
        }
        
        filename = f"{output_dir}/{self.client_id}_{data_split}_embeddings.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"   💾 Embeddings saved to {filename}")
        
        # Update status
        update_client_status(
            client_id=self.client_id,
            embeddings_sent=True,
            accuracy=self.current_accuracy,
            f1_score=self.current_f1
        )
    
    def load_embeddings(self, data_split='train', input_dir='embeddings'):
        """Load embeddings from file."""
        filename = f"{input_dir}/{self.client_id}_{data_split}_embeddings.pkl"
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Embeddings file not found: {filename}")
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        print(f"   📁 Embeddings loaded from {filename}")
        return data['embeddings'], data['labels'], data['indices']
    
    def save_model(self, filepath=None):
        """Save the trained model."""
        if filepath is None:
            os.makedirs('models', exist_ok=True)
            filepath = f"models/{self.client_id}_model.h5"
        
        self.encoder.save_weights(filepath)
        print(f"   💾 Model saved to {filepath}")
    
    def load_model_weights(self, filepath=None):
        """Load model weights."""
        if filepath is None:
            filepath = f"models/{self.client_id}_model.h5"
        
        if os.path.exists(filepath):
            self.encoder.load_weights(filepath)
            print(f"   📁 Model weights loaded from {filepath}")
            
            # Update status
            update_client_status(
                client_id=self.client_id,
                weights_updated=True,
                accuracy=self.current_accuracy,
                f1_score=self.current_f1
            )
        else:
            print(f"   ⚠️  Model weights file not found: {filepath}")
    
    def get_performance_metrics(self):
        """Get current performance metrics."""
        return {
            'client_id': self.client_id,
            'accuracy': self.current_accuracy,
            'f1_score': self.current_f1,
            'loss': self.current_loss
        }
    
    def analyze_feature_importance(self):
        """Analyze the importance of tabular features."""
        if self.encoder is None:
            print("   ⚠️  Model not created yet")
            return
        
        print(f"\n🔍 Analyzing feature importance for {self.client_id}...")
        
        # Get a sample of data for analysis
        sample_features = self.test_data['features'][:100]
        
        # Generate embeddings for the sample
        sample_embeddings = extract_embeddings(self.encoder, sample_features, batch_size=32)
        
        # Simple feature importance based on embedding variance
        feature_importance = np.var(sample_embeddings, axis=0)
        
        print(f"   📊 Feature importance statistics:")
        print(f"   - Mean importance: {np.mean(feature_importance):.4f}")
        print(f"   - Std importance: {np.std(feature_importance):.4f}")
        print(f"   - Max importance: {np.max(feature_importance):.4f}")
        print(f"   - Min importance: {np.min(feature_importance):.4f}")
        
        return feature_importance
    
    def load_global_model(self, round_idx):
        """Load global model weights from server for FL round."""
        fl_comm_dir = "communication"
        global_model_file = f"{fl_comm_dir}/global_model_round_{round_idx}.pkl"
        
        if os.path.exists(global_model_file):
            with open(global_model_file, 'rb') as f:
                global_data = pickle.load(f)
            
            # Apply aggregated embedding knowledge if available
            if 'aggregated_embedding_knowledge' in global_data:
                aggregated_bias = global_data['aggregated_embedding_knowledge']
                
                # Update only the bias of the final embedding layer
                current_weights = self.encoder.get_weights()
                current_weights[-1] = aggregated_bias  # Replace only bias (last layer)
                self.encoder.set_weights(current_weights)
                
                print(f"   📁 Global embedding knowledge applied for round {round_idx + 1}")
                print(f"   🔄 Updated embedding bias: {aggregated_bias.shape}")
            else:
                print(f"   📁 Global model loaded for round {round_idx + 1} (no embedding knowledge to apply)")
                
            return global_data
            
        print(f"   ⚠️  Global model file not found: {global_model_file}")
        return None
    
    def save_model_update(self, round_idx, num_samples):
        """Save model update for server aggregation."""
        fl_comm_dir = "communication"
        os.makedirs(fl_comm_dir, exist_ok=True)
        
        # Get only the final embedding layer weights (last 2 layers: weights + bias)
        all_weights = self.encoder.get_weights()
        embedding_weights = all_weights[-2:]  # Final Dense layer weights and bias
        
        # Prepare model update with only embedding layer weights
        update_data = {
            'client_id': self.client_id,
            'round': round_idx,
            'model_weights': embedding_weights,  # Only final embedding layer
            'num_samples': num_samples,
            'performance': {
                'accuracy': self.current_accuracy,
                'f1_score': self.current_f1,
                'loss': self.current_loss
            }
        }
        
        # Save update
        update_file = f"{fl_comm_dir}/{self.client_id}_update_round_{round_idx}.pkl"
        with open(update_file, 'wb') as f:
            pickle.dump(update_data, f)
        
        print(f"   💾 Model update saved: {update_file} (embedding layer only)")
        
        # Update status
        update_client_status(
            client_id=self.client_id,
            weights_updated=True,
            accuracy=self.current_accuracy,
            f1_score=self.current_f1
        )

    def train_local_model(self, epochs=10, batch_size=16, verbose=1):
        """
        Train the local tabular model on client data.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            verbose (int): Verbosity level
        
        Returns:
            dict: Training history and metrics
        """
        if not hasattr(self, 'train_data') or self.train_data is None:
            raise ValueError("No training data available. Load data first.")
        
        print(f"\n📊 TRAINING TABULAR CLIENT MODEL")
        print(f"   📊 Training samples: {len(self.train_data[1])}")
        print(f"   🔄 Epochs: {epochs}")
        print(f"   📦 Batch size: {batch_size}")
        
        # Prepare training data
        train_features, train_labels = self.train_data
        val_features, val_labels = self.val_data if hasattr(self, 'val_data') else (None, None)
        
        # Compute class weights for imbalanced data
        class_weights = compute_class_weights(train_labels, method='balanced')
        print(f"   ⚖️  Class weights computed for {len(set(train_labels))} classes")
        
        # Create a temporary classification model for training
        encoder_input = self.encoder.input
        encoder_output = self.encoder.output
        
        # Add classification head for training
        classifier_head = Dense(64, activation='relu', kernel_initializer='he_normal')(encoder_output)
        classifier_head = BatchNormalization()(classifier_head)
        classifier_head = Dropout(0.4)(classifier_head)
        classifier_predictions = Dense(7, activation='softmax', name='predictions')(classifier_head)
        
        # Create training model
        training_model = Model(inputs=encoder_input, outputs=classifier_predictions, name='tabular_training_model')
        
        # Compile with appropriate optimizer and loss
        training_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"   🏗️  Training model created: {training_model.count_params():,} parameters")
        
        # Prepare validation data
        validation_data = None
        if val_features is not None and val_labels is not None:
            validation_data = (val_features, val_labels)
        
        # Train the model
        print(f"   🚀 Starting local training...")
        history = training_model.fit(
            train_features, train_labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            class_weight=class_weights,
            verbose=verbose,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy' if validation_data else 'accuracy',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if validation_data else 'loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6
                )
            ]
        )
        
        # Extract the trained encoder weights
        for i, layer in enumerate(self.encoder.layers):
            if i < len(training_model.layers) - 3:  # Exclude the classification head layers
                layer.set_weights(training_model.layers[i].get_weights())
        
        # Evaluate final performance
        final_train_acc = max(history.history['accuracy'])
        final_val_acc = max(history.history.get('val_accuracy', [0]))
        
        print(f"   ✅ Training completed!")
        print(f"   🎯 Best training accuracy: {final_train_acc:.4f}")
        if validation_data:
            print(f"   🎯 Best validation accuracy: {final_val_acc:.4f}")
        
        return {
            'history': history.history,
            'final_train_acc': final_train_acc,
            'final_val_acc': final_val_acc,
            'epochs_completed': len(history.history['loss'])
        }


def run_fl_round(args):
    """Run a single federated learning round for tabular client."""
    print(f"📋 Tabular Client - FL Round {args.round_idx + 1}")
    print("=" * 50)
    
    try:
        # Create client
        client = TabularClient(
            client_id="tabular_client",
            data_percentage=args.data_percentage,
            learning_rate=args.learning_rate,
            embedding_dim=args.embedding_dim
        )
        
        # Load data
        client.load_data(data_dir=args.data_dir)
        
        # Create model
        client.create_model()
        
        # Load global model from server
        global_data = client.load_global_model(args.round_idx)
        
        # Train on local data
        print(f"🎯 Training on local data ({args.epochs} epochs)...")
        results = client.train(epochs=args.epochs, batch_size=args.batch_size)
        
        # Get number of training samples
        num_samples = len(client.train_data['labels'])
        
        # Save model update for server
        client.save_model_update(args.round_idx, num_samples)
        
        print(f"✅ FL Round {args.round_idx + 1} completed")
        print(f"   📊 Local accuracy: {client.current_accuracy:.4f}")
        print(f"   📈 Local F1: {client.current_f1:.4f}")
        print(f"   📦 Samples: {num_samples}")
        
        return 0
        
    except Exception as e:
        print(f"❌ FL Round failed: {e}")
        return 1


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(description='Tabular Client for VFL')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing HAM10000 dataset')
    parser.add_argument('--data_percentage', type=float, default=0.1,
                       help='Percentage of data to use (0.0-1.0)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--embedding_dim', type=int, default=256,
                       help='Embedding dimension')
    parser.add_argument('--output_dir', type=str, default='embeddings',
                       help='Output directory for embeddings')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate', 'generate_embeddings', 'analyze'],
                       help='Mode of operation')
    
    # Federated Learning arguments
    parser.add_argument('--fl_mode', type=str, default='false',
                       help='Enable federated learning mode')
    parser.add_argument('--round_idx', type=int, default=0,
                       help='Current FL round index')
    
    args = parser.parse_args()
    
    # Check for FL mode
    if args.fl_mode.lower() == 'true':
        # FL mode - participate in federated round
        return run_fl_round(args)
    
    # Create client
    client = TabularClient(
        client_id="tabular_client",
        data_percentage=args.data_percentage,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim
    )
    
    # Load data
    client.load_data(data_dir=args.data_dir)
    
    # Create model
    client.create_model()
    
    if args.mode == 'train':
        # Train model
        client.train(epochs=args.epochs, batch_size=args.batch_size)
        
        # Evaluate model
        client.evaluate()
        
        # Save model
        client.save_model()
        
        # Generate and save embeddings for all splits
        for split in ['train', 'val', 'test']:
            embeddings, labels, indices = client.generate_embeddings(split)
            client.save_embeddings(embeddings, labels, indices,
                                 data_split=split, output_dir=args.output_dir)
    
    elif args.mode == 'evaluate':
        # Load existing model
        client.load_model_weights()
        
        # Evaluate model
        client.evaluate()
    
    elif args.mode == 'generate_embeddings':
        # Load existing model
        client.load_model_weights()
        
        # Generate embeddings for all splits
        for split in ['train', 'val', 'test']:
            embeddings, labels, indices = client.generate_embeddings(split)
            client.save_embeddings(embeddings, labels, indices,
                                 data_split=split, output_dir=args.output_dir)
    
    elif args.mode == 'analyze':
        # Load existing model
        client.load_model_weights()
        
        # Analyze feature importance
        client.analyze_feature_importance()
    
    print(f"\n✅ {client.client_id} completed successfully!")


if __name__ == "__main__":
    main()

# Flask app for client API (conditional)
if FLASK_AVAILABLE:
    app = Flask(__name__)
    client = None
    
    @app.route('/train_local', methods=['POST'])
    def train_local():
        """Train local model endpoint."""
        try:
            data = request.get_json()
            epochs = data.get('epochs', 10)
            batch_size = data.get('batch_size', 16)
            verbose = data.get('verbose', 1)
            
            # Train the local model
            results = client.train_local_model(epochs=epochs, batch_size=batch_size, verbose=verbose)
            
            return jsonify({
                'status': 'success',
                'final_train_acc': results['final_train_acc'],
                'final_val_acc': results['final_val_acc'],
                'epochs_completed': results['epochs_completed']
            })
            
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/generate_embeddings', methods=['POST'])
    def generate_embeddings():
        """Generate embeddings for given data split."""
        try:
            data = request.get_json()
            data_split = data.get('data_split', 'train')
            
            # Generate embeddings
            results = client.generate_embeddings(data_split)
            
            return jsonify({
                'status': 'success',
                'embeddings_saved': True,
                'split': data_split,
                'samples': results.get('num_samples', 0)
            })
            
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500 