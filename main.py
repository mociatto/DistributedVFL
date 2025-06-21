#!/usr/bin/env python3
"""
HybridVFL Main Orchestrator
Centralized configuration and federated learning coordination for multimodal VFL.

🎯 QUICK CONFIGURATION GUIDE:
============================
To change training parameters, modify the values in get_default_config() method (around line 50):

Key Parameters:
- data_percentage: 0.01 (1%), 0.1 (10%), 1.0 (100%)
- total_rounds: Number of federated learning rounds
- epochs_per_round: Training epochs per round
- batch_size: Batch size for training
- learning_rate: Learning rate for optimization
- adversarial_lambda: Privacy weight (0.0=disabled)

Then run: python3 main.py
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

# Import project modules
from server import FederatedServer
from status import initialize_status, finalize_training_status


class HybridVFLOrchestrator:
    """
    Main orchestrator for HybridVFL federated learning pipeline.
    Manages configuration, coordinates FL rounds, and collects results.
    """
    
    def __init__(self, config=None):
        """Initialize orchestrator with configuration."""
        # Start with default config and update with provided config
        self.config = self.get_default_config()
        if config:
            self.config.update(config)
        
        self.results = {}
        self.start_time = None
        
        # Create necessary directories
        self.setup_directories()
        
        print("🚀 HybridVFL Orchestrator Initialized")
        print("=" * 60)
        self.print_configuration()
    
    @staticmethod
    def get_default_config():
        """
        🎯 MAIN CONFIGURATION SECTION
        ============================
        Adjust these values to control the entire training pipeline.
        All other files will automatically use these settings.
        """
        return {
            # ============================================================================
            # 🎛️  INDUSTRY DASHBOARD CONTROLS - MODIFY THESE VALUES
            # ============================================================================
            
            # === Core Training Parameters ===
            'data_percentage': 0.02,           # 🔢 Dataset portion (0.01=1%, 0.1=10%, 1.0=100%)
            'total_rounds': 2,                 # 🔄 Number of federated learning rounds
            'epochs_per_round': 2,             # 📈 Training epochs per FL round
            'batch_size': 8,                   # 📦 Batch size for training
            'learning_rate': 0.001,            # 🎯 Learning rate for optimization
            'validation_split': 0.2,           # 📊 Validation data split (20%)
            
            # === Client-Specific Training ===
            'client_epochs': {
                'image_client': 2,             # 🖼️  Image client epochs per round
                'tabular_client': 2            # 📋 Tabular client epochs per round
            },
            
            # === Model Architecture ===
            'embedding_dim': 256,              # 🧠 Embedding dimension for both modalities
            'adversarial_lambda': 0.0,         # 🔒 Privacy weight (0.0=disabled, 0.1-1.0=enabled)
            
            # === Performance Tuning ===
            'early_stopping': False,           # ⏹️  Enable early stopping
            'patience': 3,                     # ⏳ Early stopping patience
            'verbose': 2,                      # 📢 Logging level (0=quiet, 1=normal, 2=detailed)
            
            # ============================================================================
            # 🔧 ADVANCED CONFIGURATION - Usually don't need to change
            # ============================================================================
            
            # === Data Configuration ===
            'data_dir': 'data',
            'random_seed': 42,
            'num_classes': 7,
            
            # === FL Strategy Configuration ===
            'aggregation_method': 'fedavg',     # Federated averaging
            'client_selection': 'all',          # Use all clients
            'min_clients': 2,                   # Minimum clients for training
            
            # === Experimental Configuration ===
            'save_intermediate': True,          # Save models after each round
            'save_client_models': True,         # Save individual client models
            
            # === Output Configuration ===
            'model_dir': 'models',
            'results_dir': 'results',
            'plots_dir': 'plots',
            'logs_dir': 'logs',
            
            # === Phase Configuration ===
            'phase': 1,                         # Current implementation phase
            'phase_description': 'High-Performance Baseline'
        }
    
    def setup_directories(self):
        """Create necessary directories for FL pipeline."""
        dirs = [
            self.config['model_dir'],
            self.config['results_dir'], 
            self.config['plots_dir'],
            self.config['logs_dir'],
            'embeddings',
            'communication'  # For FL weight sharing
        ]
        
        for dir_name in dirs:
            Path(dir_name).mkdir(exist_ok=True)
    
    def print_configuration(self):
        """Print current configuration."""
        print("🔧 CONFIGURATION:")
        print(f"   Phase: {self.config['phase']} - {self.config['phase_description']}")
        print(f"   Data: {self.config['data_percentage']*100:.1f}% of HAM10000")
        print(f"   FL Rounds: {self.config['total_rounds']}")
        print(f"   Epochs per Round: {self.config['epochs_per_round']}")
        print(f"   Batch Size: {self.config['batch_size']}")
        print(f"   Learning Rate: {self.config['learning_rate']}")
        print(f"   Embedding Dim: {self.config['embedding_dim']}")
        
        if self.config['adversarial_lambda'] == 0.0:
            print("   🔒 Privacy: DISABLED (Phase 1 - High Performance)")
        else:
            print(f"   🔒 Privacy: ENABLED (λ={self.config['adversarial_lambda']})")
        
        print(f"   Aggregation: {self.config['aggregation_method'].upper()}")
        print(f"   Clients: {self.config['client_selection']}")
    
    def save_configuration(self):
        """Save configuration to file."""
        config_file = f"{self.config['results_dir']}/fl_config.json"
        
        # Add timestamp and system info
        config_with_meta = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'python_version': sys.version,
                'working_directory': os.getcwd()
            },
            'configuration': self.config
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_with_meta, f, indent=2)
        
        print(f"   💾 Configuration saved to {config_file}")
    
    def run_federated_learning(self):
        """Run the complete federated learning pipeline."""
        print(f"\n🚀 STARTING FEDERATED LEARNING PIPELINE")
        print("=" * 70)
        
        self.start_time = time.time()
        
        # Save configuration
        self.save_configuration()
        
        # Initialize status tracking
        initialize_status(self.config['total_rounds'])
        
        try:
            # Create and configure federated server
            server = FederatedServer(
                embedding_dim=self.config['embedding_dim'],
                num_classes=self.config['num_classes'],
                adversarial_lambda=self.config['adversarial_lambda'],
                learning_rate=self.config['learning_rate'],
                data_percentage=self.config['data_percentage'],
                config=self.config  # Pass full configuration
            )
            
            # Initialize server components
            server.create_models()
            server.load_data_loader(data_dir=self.config['data_dir'])
            
            # Run federated training with orchestrator configuration
            final_results = server.run_federated_training(
                total_rounds=self.config['total_rounds'],
                epochs_per_round=self.config['epochs_per_round'],
                batch_size=self.config['batch_size']
            )
            
            # Store results
            self.results = final_results
            
            # Save comprehensive results
            self.save_results(server)
            
            # Print final summary
            self.print_final_summary()
            
            return final_results
            
        except Exception as e:
            print(f"❌ Federated learning failed: {e}")
            raise
        
        finally:
            total_time = time.time() - self.start_time
            finalize_training_status(
                best_accuracy=getattr(self, 'best_accuracy', 0.0),
                best_f1=getattr(self, 'best_f1', 0.0),
                total_time=total_time,
                total_rounds=self.config['total_rounds']
            )
    
    def save_results(self, server):
        """Save comprehensive training results."""
        total_time = time.time() - self.start_time
        
        results = {
            'configuration': self.config,
            'training_results': self.results,
            'training_history': server.training_history,
            'best_performance': {
                'accuracy': server.best_accuracy,
                'f1_score': server.best_f1,
                'round': server.best_round
            },
            'timing': {
                'total_time_seconds': total_time,
                'total_time_formatted': f"{total_time//60:.0f}m {total_time%60:.0f}s",
                'average_round_time': total_time / self.config['total_rounds']
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'phase': self.config['phase'],
                'phase_description': self.config['phase_description']
            }
        }
        
        # Save to JSON
        results_file = f"{self.config['results_dir']}/fl_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"   💾 Results saved to {results_file}")
        
        # Also save server's detailed results
        server.save_training_results(self.results, f"{self.config['results_dir']}/detailed_results.pkl")
    
    def print_final_summary(self):
        """Print comprehensive final summary."""
        total_time = time.time() - self.start_time
        
        print(f"\n" + "="*80)
        print(f"🎉 HYBRIDVFL FEDERATED LEARNING COMPLETED")
        print(f"="*80)
        
        print(f"\n📊 EXPERIMENT SUMMARY:")
        print(f"   Phase: {self.config['phase']} - {self.config['phase_description']}")
        print(f"   Total Time: {total_time//60:.0f}m {total_time%60:.0f}s")
        print(f"   FL Rounds: {self.config['total_rounds']}")
        print(f"   Data Used: {self.config['data_percentage']*100:.1f}% of HAM10000")
        
        if hasattr(self, 'results') and self.results:
            print(f"   Final Test Accuracy: {self.results.get('accuracy', 0):.4f}")
            print(f"   Final Test F1: {self.results.get('f1_macro', 0):.4f}")
        
        print(f"\n🎯 PHASE 1 OBJECTIVES STATUS:")
        print(f"   ✅ Adversarial head disabled (λ={self.config['adversarial_lambda']})")
        print(f"   ✅ FocalLoss implemented for class imbalance")
        print(f"   ✅ Transformer fusion for multimodal data")
        print(f"   ✅ Proper federated learning paradigm")
        
        target_accuracy = 0.75
        if hasattr(self, 'results') and self.results.get('accuracy', 0) > target_accuracy:
            print(f"   ✅ Target accuracy >75% ACHIEVED!")
        else:
            print(f"   🎯 Target accuracy >75% (work in progress)")
        
        print(f"\n📁 OUTPUT FILES:")
        print(f"   - Configuration: {self.config['results_dir']}/fl_config.json")
        print(f"   - Results: {self.config['results_dir']}/fl_results_*.json")
        print(f"   - Models: {self.config['model_dir']}/")
        print(f"   - Plots: {self.config['plots_dir']}/")


def create_config_from_args(args):
    """Create configuration from command line arguments."""
    config = HybridVFLOrchestrator.get_default_config()
    
    # Update config with command line arguments
    if args.data_percentage is not None:
        config['data_percentage'] = args.data_percentage
    if args.total_rounds is not None:
        config['total_rounds'] = args.total_rounds
    if args.epochs_per_round is not None:
        config['epochs_per_round'] = args.epochs_per_round
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    if args.embedding_dim is not None:
        config['embedding_dim'] = args.embedding_dim
    if args.adversarial_lambda is not None:
        config['adversarial_lambda'] = args.adversarial_lambda
    if args.data_dir is not None:
        config['data_dir'] = args.data_dir
    if args.verbose is not None:
        config['verbose'] = args.verbose
    
    return config


def main():
    """Main function for HybridVFL orchestration."""
    parser = argparse.ArgumentParser(
        description='HybridVFL: Multimodal Vertical Federated Learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # === Core FL Parameters ===
    parser.add_argument('--data_percentage', type=float, default=None,
                       help='Percentage of dataset to use (0.0-1.0)')
    parser.add_argument('--total_rounds', type=int, default=None,
                       help='Total number of federated learning rounds')
    parser.add_argument('--epochs_per_round', type=int, default=None,
                       help='Epochs per federated round')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training')
    
    # === Model Parameters ===
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate for training')
    parser.add_argument('--embedding_dim', type=int, default=None,
                       help='Embedding dimension for both modalities')
    parser.add_argument('--adversarial_lambda', type=float, default=None,
                       help='Adversarial loss weight (0 to disable privacy)')
    
    # === Data Parameters ===
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory containing HAM10000 dataset')
    
    # === Experimental Parameters ===
    parser.add_argument('--verbose', type=int, default=None,
                       help='Verbosity level (0=quiet, 1=normal, 2=detailed)')
    parser.add_argument('--config_file', type=str, default=None,
                       help='Load configuration from JSON file')
    
    # === Quick Presets ===
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test with minimal data (1%%, 2 rounds, 2 epochs)')
    parser.add_argument('--full_training', action='store_true',
                       help='Full training with 100%% data (10 rounds, 20 epochs)')
    
    args = parser.parse_args()
    
    # Handle presets
    if args.quick_test:
        args.data_percentage = 0.01
        args.total_rounds = 2
        args.epochs_per_round = 2
        print("🧪 Quick test mode activated")
    
    if args.full_training:
        args.data_percentage = 1.0
        args.total_rounds = 10
        args.epochs_per_round = 20
        print("🚀 Full training mode activated")
    
    # Create configuration
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        print(f"📁 Configuration loaded from {args.config_file}")
    else:
        config = create_config_from_args(args)
    
    # Create and run orchestrator
    orchestrator = HybridVFLOrchestrator(config)
    
    try:
        results = orchestrator.run_federated_learning()
        print(f"\n✅ HybridVFL completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 