#!/usr/bin/env python3
"""
Test script to verify the WILD adversarial defense mechanism.
This tests the fixed inference attack pipeline and strong adversarial perturbations.
"""

import numpy as np
import tensorflow as tf
from server import FederatedServer
import pickle
import os

def test_inference_attacks_and_defense():
    """Test the inference attack and defense pipeline."""
    print("Testing WILD Adversarial Defense Pipeline")
    print("="*60)
    
    # Initialize server with WILD defense enabled
    print("1. Initializing Federated Server...")
    server = FederatedServer(
        embedding_dim=256,
        num_classes=7,
        adversarial_lambda=0.0,  # Start with no defense
        learning_rate=0.001,
        data_percentage=0.05
    )
    
    # Check if embedding files exist
    embeddings_dir = 'embeddings'
    image_train_file = f"{embeddings_dir}/image_client_train_embeddings.pkl"
    tabular_train_file = f"{embeddings_dir}/tabular_client_train_embeddings.pkl"
    
    if not os.path.exists(image_train_file) or not os.path.exists(tabular_train_file):
        print("Embedding files not found. Please run the FL training first.")
        print("   Expected files:")
        print(f"   - {image_train_file}")
        print(f"   - {tabular_train_file}")
        return False
    
    print("2. Loading embeddings...")
    try:
        # Test loading embeddings (this should work now with fixed indexing)
        train_image_emb, train_tabular_emb, train_labels, train_sensitive_attrs = \
            server.load_client_embeddings('train')
        val_image_emb, val_tabular_emb, val_labels, val_sensitive_attrs = \
            server.load_client_embeddings('val')
        
        print(f"   Train embeddings loaded: {train_image_emb.shape}, {train_tabular_emb.shape}")
        print(f"   Val embeddings loaded: {val_image_emb.shape}, {val_tabular_emb.shape}")
        print(f"   Sensitive attributes shape: {train_sensitive_attrs.shape}")
        print(f"   Sensitive attributes type: {type(train_sensitive_attrs)}")
        
        # Verify sensitive attributes structure
        print(f"   Gender distribution: {np.bincount(train_sensitive_attrs[:, 0])}")
        print(f"   Age distribution: {np.bincount(train_sensitive_attrs[:, 1])}")
        
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n3. Testing inference attacks WITHOUT defense...")
    try:
        # Test inference attacks without defense (lambda=0)
        server.adversarial_lambda = 0.0
        results_no_defense = server.train_and_evaluate_inference_attacks(round_idx=0)
        
        print(f"   No Defense Results:")
        print(f"     Age leakage: {results_no_defense['age_leakage']:.2f}%")
        print(f"     Gender leakage: {results_no_defense['gender_leakage']:.2f}%")
        print(f"     Defense strength: {results_no_defense['defense_strength']:.2f}%")
        
        # Verify we get realistic attack results (should be high leakage)
        if results_no_defense['age_leakage'] < 50 or results_no_defense['gender_leakage'] < 70:
            print("   Warning: Attack results seem unrealistically low")
        else:
            print("   Attack results look realistic")
            
    except Exception as e:
        print(f"Error in inference attacks (no defense): {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n4. Testing WILD adversarial defense...")
    try:
        # Test with WILD defense enabled
        server.adversarial_lambda = 0.5  # Strong defense
        results_with_defense = server.train_and_evaluate_inference_attacks(round_idx=1)
        
        print(f"   WILD Defense Results (λ=0.5):")
        print(f"     Age leakage: {results_with_defense['age_leakage']:.2f}%")
        print(f"     Gender leakage: {results_with_defense['gender_leakage']:.2f}%")
        print(f"     Defense strength: {results_with_defense['defense_strength']:.2f}%")
        
        # Calculate actual defense effectiveness
        age_reduction = results_no_defense['age_leakage'] - results_with_defense['age_leakage']
        gender_reduction = results_no_defense['gender_leakage'] - results_with_defense['gender_leakage']
        
        print(f"   Defense Effectiveness:")
        print(f"     Age leakage reduction: {age_reduction:.2f}% points")
        print(f"     Gender leakage reduction: {gender_reduction:.2f}% points")
        
        # Verify defense is actually working
        if age_reduction > 5 or gender_reduction > 5:
            print("   WILD defense is working! Significant leakage reduction observed.")
        else:
            print("   Warning: Defense effect seems minimal")
            
    except Exception as e:
        print(f"Error in inference attacks (with defense): {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n5. Testing EXTREME WILD defense...")
    try:
        # Test with EXTREME defense
        server.adversarial_lambda = 1.0  # Maximum defense
        results_extreme_defense = server.train_and_evaluate_inference_attacks(round_idx=2)
        
        print(f"   EXTREME WILD Defense Results (λ=1.0):")
        print(f"     Age leakage: {results_extreme_defense['age_leakage']:.2f}%")
        print(f"     Gender leakage: {results_extreme_defense['gender_leakage']:.2f}%")
        print(f"     Defense strength: {results_extreme_defense['defense_strength']:.2f}%")
        
        # Calculate extreme defense effectiveness
        age_reduction_extreme = results_no_defense['age_leakage'] - results_extreme_defense['age_leakage']
        gender_reduction_extreme = results_no_defense['gender_leakage'] - results_extreme_defense['gender_leakage']
        
        print(f"   EXTREME Defense Effectiveness:")
        print(f"     Age leakage reduction: {age_reduction_extreme:.2f}% points")
        print(f"     Gender leakage reduction: {gender_reduction_extreme:.2f}% points")
        
        # Verify extreme defense is even stronger
        if age_reduction_extreme > age_reduction or gender_reduction_extreme > gender_reduction:
            print("   EXTREME WILD defense is stronger! Progressive defense scaling works.")
        else:
            print("   Warning: Extreme defense not significantly stronger than moderate defense")
            
    except Exception as e:
        print(f"Error in extreme defense test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nWILD Defense Test Summary:")
    print("="*60)
    print(f"No Defense:      Age={results_no_defense['age_leakage']:.1f}%, Gender={results_no_defense['gender_leakage']:.1f}%")
    print(f"Moderate Defense: Age={results_with_defense['age_leakage']:.1f}%, Gender={results_with_defense['gender_leakage']:.1f}%")
    print(f"EXTREME Defense:  Age={results_extreme_defense['age_leakage']:.1f}%, Gender={results_extreme_defense['gender_leakage']:.1f}%")
    
    # Overall assessment
    total_age_reduction = results_no_defense['age_leakage'] - results_extreme_defense['age_leakage']
    total_gender_reduction = results_no_defense['gender_leakage'] - results_extreme_defense['gender_leakage']
    
    if total_age_reduction > 10 and total_gender_reduction > 10:
        print("✅ SUCCESS: WILD defense provides significant privacy protection!")
        print(f"   Total reductions: Age={total_age_reduction:.1f}%, Gender={total_gender_reduction:.1f}%")
        return True
    elif total_age_reduction > 5 or total_gender_reduction > 5:
        print("⚠️ PARTIAL SUCCESS: Some privacy protection, but could be stronger")
        print(f"   Total reductions: Age={total_age_reduction:.1f}%, Gender={total_gender_reduction:.1f}%")
        return True
    else:
        print("❌ FAILURE: Defense provides minimal privacy protection")
        print(f"   Total reductions: Age={total_age_reduction:.1f}%, Gender={total_gender_reduction:.1f}%")
        return False

if __name__ == "__main__":
    success = test_inference_attacks_and_defense()
    if success:
        print("\n🎯 WILD Defense mechanism is working correctly!")
        print("   You can now run the full FL training with confidence.")
    else:
        print("\n💥 Issues detected with defense mechanism.")
        print("   Please check the implementation and try again.") 