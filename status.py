"""
Status management for federated learning dashboard integration.
Handles writing training progress to JSON files for real-time monitoring.
"""

import json
import time
from datetime import datetime
import os


def update_training_status(current_round, total_rounds, accuracy=None, loss=None,
                          f1_score=None, precision=None, recall=None, precision_recall=None,
                          defense_strength=None, client_weights=None, phase="training",
                          gender_fairness=None, age_fairness=None, age_leakage=None, gender_leakage=None):
    """
    Update the training status JSON file with current progress.
    
    Args:
        current_round (int): Current training round
        total_rounds (int): Total number of rounds
        accuracy (float): Current accuracy
        loss (float): Current loss
        f1_score (float): Current F1 score
        precision (float): Current precision
        recall (float): Current recall
        precision_recall (float): Combined precision-recall metric
        defense_strength (float): Defense strength from adversarial loss
        client_weights (list): Attention weights for clients
        phase (str): Current training phase
        gender_fairness (list): Gender fairness scores [female_acc, male_acc]
        age_fairness (list): Age fairness scores for 6 age groups
        age_leakage (float): Age inference attack success rate (%)
        gender_leakage (float): Gender inference attack success rate (%)
    """
    status = {
        "current_round": current_round,
        "total_rounds": total_rounds,
        "progress_percent": round((current_round / total_rounds) * 100, 2),
        "phase": phase,
        "timestamp": datetime.now().isoformat(),
        "unix_timestamp": int(time.time()),
        "metrics": {
            "accuracy": round(accuracy, 4) if accuracy is not None else None,
            "loss": round(loss, 4) if loss is not None else None,
            "f1_score": round(f1_score, 4) if f1_score is not None else None,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "precision_recall": round(precision_recall, 4) if precision_recall is not None else None,
            "defense_strength": round(defense_strength, 4) if defense_strength is not None else None,
            "gender_fairness": [round(g, 2) for g in gender_fairness] if gender_fairness is not None else [0.0, 0.0],
            "age_fairness": [round(a, 2) for a in age_fairness] if age_fairness is not None else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "age_leakage": round(age_leakage, 2) if age_leakage is not None else 16.67,
            "gender_leakage": round(gender_leakage, 2) if gender_leakage is not None else 50.0
        },
        "client_weights": client_weights if client_weights is not None else None,
        "status": "running" if current_round < total_rounds else "completed"
    }
    
    try:
        with open("status/status.json", "w") as f:
            json.dump(status, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not update status file: {e}")


def update_client_status(client_id, accuracy=None, loss=None, f1_score=None,
                        embeddings_sent=False, weights_updated=False):
    """
    Update individual client status.
    
    Args:
        client_id (str): ID of the client ('image_client' or 'tabular_client')
        accuracy (float): Client's accuracy
        loss (float): Client's loss
        f1_score (float): Client's F1 score
        embeddings_sent (bool): Whether embeddings were sent to server
        weights_updated (bool): Whether weights were updated from server
    """
    status_file = f"status/{client_id}_status.json"
    
    client_status = {
        "client_id": client_id,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "accuracy": round(accuracy, 4) if accuracy is not None else None,
            "loss": round(loss, 4) if loss is not None else None,
            "f1_score": round(f1_score, 4) if f1_score is not None else None
        },
        "communication": {
            "embeddings_sent": embeddings_sent,
            "weights_updated": weights_updated
        },
        "status": "active"
    }
    
    try:
        with open(status_file, "w") as f:
            json.dump(client_status, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not update {client_id} status file: {e}")


def finalize_training_status(best_accuracy, best_f1, total_time, total_rounds):
    """
    Write final training results to status file.
    
    Args:
        best_accuracy (float): Best achieved accuracy
        best_f1 (float): Best achieved F1 score
        total_time (float): Total training time in seconds
        total_rounds (int): Total number of rounds completed
    """
    final_status = {
        "status": "completed",
        "completed_at": datetime.now().isoformat(),
        "total_rounds": total_rounds,
        "total_time_seconds": round(total_time, 2),
        "total_time_formatted": f"{total_time // 60:.0f}m {total_time % 60:.0f}s",
        "best_metrics": {
            "accuracy": round(best_accuracy, 4),
            "f1_score": round(best_f1, 4)
        },
        "average_time_per_round": round(total_time / total_rounds, 2) if total_rounds > 0 else 0
    }
    
    try:
        with open("status/final_results.json", "w") as f:
            json.dump(final_status, f, indent=2)
        
        # Also update the main status file
        with open("status/status.json", "r") as f:
            current_status = json.load(f)
        
        current_status.update(final_status)
        
        with open("status/status.json", "w") as f:
            json.dump(current_status, f, indent=2)
            
    except Exception as e:
        print(f"Warning: Could not update final status: {e}")


def initialize_status(total_rounds):
    """
    Initialize the status tracking system.
    
    Args:
        total_rounds (int): Total number of training rounds planned
    """
    initial_status = {
        "status": "initializing",
        "current_round": 0,
        "total_rounds": total_rounds,
        "progress_percent": 0.0,
        "phase": "initialization",
        "timestamp": datetime.now().isoformat(),
        "unix_timestamp": int(time.time()),
        "metrics": {
            "accuracy": None,
            "loss": None,
            "f1_score": None
        },
        "client_weights": None
    }
    
    try:
        with open("status/status.json", "w") as f:
            json.dump(initial_status, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not initialize status file: {e}")


def cleanup_status_files():
    """
    Clean up status files at the end of training.
    """
    status_files = ["status/image_client_status.json", "status/tabular_client_status.json"]
    
    for file in status_files:
        try:
            if os.path.exists(file):
                os.remove(file)
        except Exception as e:
            print(f"Warning: Could not remove {file}: {e}")