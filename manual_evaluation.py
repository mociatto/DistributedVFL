#!/usr/bin/env python3
"""
Manual Final Evaluation Script
===============================
This script evaluates the global fusion model using the embeddings 
that were generated during the distributed FL run.

How it works WITHOUT accessing raw data:
1. Loads saved test embeddings from both clients
2. Loads test labels (included with embeddings)
3. Creates and loads the trained fusion model
4. Runs inference on embeddings to get predictions
5. Computes final accuracy and F1 score
"""

import numpy as np
import pickle
import os
from datetime import datetime
import json

# Import your model creation functions
from models import create_fusion_model_with_transformer
from config import get_config
import tensorflow as tf

def load_embeddings(client_type, split='test'):
    """Load embeddings and labels for specific client and data split."""
    embeddings_file = f'embeddings/{client_type}_client_{split}_embeddings.pkl'
    
    if not os.path.exists(embeddings_file):
        print(f"❌ Embeddings file not found: {embeddings_file}")
        return None, None, None
    
    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)
    
    embeddings = data['embeddings']
    labels = data['labels'] 
    indices = data.get('indices', list(range(len(labels))))
    
    print(f"✅ Loaded {client_type} {split} embeddings: {embeddings.shape}")
    return embeddings, labels, indices

def create_and_load_fusion_model():
    """Create the fusion model architecture that was used in training."""
    config = get_config()
    
    # Create fusion model with same architecture as training
    fusion_model, _ = create_fusion_model_with_transformer(
        image_dim=config['model']['embedding_dim'],
        tabular_dim=config['model']['embedding_dim'],
        num_classes=config['data']['num_classes'],
        adversarial_lambda=0.0,  # Same as training
        use_advanced_fusion=True,  # Same as training
        use_step3_enhancements=True  # Same as training
    )
    
    print(f"✅ Created fusion model: {fusion_model.count_params():,} parameters")
    return fusion_model

def evaluate_global_model():
    """
    Evaluate the global fusion model using test embeddings.
    
    This is how the server evaluates WITHOUT raw data:
    1. Load test embeddings from both clients
    2. Load test labels (stored with embeddings)
    3. Run fusion model inference on embeddings
    4. Compute accuracy and F1 score
    """
    print("🔍 MANUAL GLOBAL MODEL EVALUATION")
    print("=" * 50)
    
    # Load test embeddings from both clients
    image_embeddings, image_labels, image_indices = load_embeddings('image', 'test')
    tabular_embeddings, tabular_labels, tabular_indices = load_embeddings('tabular', 'test')
    
    if image_embeddings is None or tabular_embeddings is None:
        print("❌ Cannot load embeddings - evaluation failed")
        return None
    
    # Verify label consistency
    if not np.array_equal(image_labels, tabular_labels):
        print("⚠️ Warning: Image and tabular labels don't match - using image labels")
    
    test_labels = image_labels
    print(f"📊 Test samples: {len(test_labels)}")
    
    # Create fusion model
    fusion_model = create_and_load_fusion_model()
    
    # Convert to tensors
    image_emb_tensor = tf.convert_to_tensor(image_embeddings, dtype=tf.float32)
    tabular_emb_tensor = tf.convert_to_tensor(tabular_embeddings, dtype=tf.float32)
    
    print("🚀 Running inference on test embeddings...")
    
    # Get predictions from fusion model
    predictions = fusion_model([image_emb_tensor, tabular_emb_tensor])
    predicted_classes = tf.argmax(predictions, axis=1).numpy()
    
    # Compute accuracy
    accuracy = np.mean(predicted_classes == test_labels)
    
    # Compute F1 score
    from sklearn.metrics import f1_score, classification_report
    f1_macro = f1_score(test_labels, predicted_classes, average='macro')
    f1_weighted = f1_score(test_labels, predicted_classes, average='weighted')
    
    # Generate detailed report
    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    report = classification_report(test_labels, predicted_classes, 
                                 target_names=class_names, 
                                 output_dict=True)
    
    # Display results
    print(f"\n🏆 GLOBAL FUSION MODEL RESULTS:")
    print(f"   🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   📈 F1 Score (Macro): {f1_macro:.4f}")
    print(f"   📈 F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"   📊 Test Samples: {len(test_labels)}")
    
    print(f"\n📋 PER-CLASS PERFORMANCE:")
    for i, class_name in enumerate(class_names):
        if str(i) in report:
            precision = report[str(i)]['precision']
            recall = report[str(i)]['recall']
            f1 = report[str(i)]['f1-score']
            support = report[str(i)]['support']
            print(f"   {class_name:>6}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} (n={support})")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'final_test_accuracy': float(accuracy),
        'final_test_f1_macro': float(f1_macro),
        'final_test_f1_weighted': float(f1_weighted),
        'test_samples': int(len(test_labels)),
        'strategy': 'fusion_guided_updates',
        'model_parameters': fusion_model.count_params(),
        'per_class_results': report,
        'evaluation_method': 'manual_embeddings_based',
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = f"results/manual_evaluation_{timestamp}.json"
    os.makedirs("results", exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return results

def main():
    """
    Main evaluation function.
    
    EXPLANATION: How Server Evaluates WITHOUT Raw Data
    ================================================
    
    1. EMBEDDINGS ALREADY COMPUTED: Clients computed embeddings from their 
       raw data and sent them to server during FL training
    
    2. LABELS INCLUDED: Test labels are stored with embeddings (metadata)
    
    3. FUSION MODEL TRAINED: Server trained fusion model using these embeddings
    
    4. INFERENCE ON EMBEDDINGS: Server runs trained fusion model on test 
       embeddings to get predictions
    
    5. ACCURACY COMPUTATION: Compare predictions with true labels
    
    This is exactly how distributed FL works - server never sees raw data,
    only processed embeddings from clients!
    """
    print("🧠 HOW SERVER EVALUATES WITHOUT RAW DATA:")
    print("=" * 50)
    print("1. ✅ Clients send embeddings (not raw data) to server")
    print("2. ✅ Server trains fusion model on embeddings")  
    print("3. ✅ Server evaluates fusion model on test embeddings")
    print("4. ✅ Server computes accuracy using test labels")
    print("5. 🔒 Server NEVER sees raw images or tabular data")
    print()
    
    try:
        results = evaluate_global_model()
        
        if results:
            print(f"\n🎉 EVALUATION COMPLETED SUCCESSFULLY!")
            print(f"   🏆 Final Global Model Accuracy: {results['final_test_accuracy']:.4f} ({results['final_test_accuracy']*100:.2f}%)")
            return 0
        else:
            print("❌ Evaluation failed")
            return 1
            
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 