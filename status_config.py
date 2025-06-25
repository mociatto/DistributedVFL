# status_config.py
# Dashboard Status Messages Configuration
# Modify these messages to customize what appears in the "CURRENT STATUS" box

STATUS_MESSAGES = {
    # Initial states
    "WAITING_FOR_TRAINING": "Waiting for FL training to start...",
    "DASHBOARD_READY": "Dashboard ready - Click Play to start training",
    "LOADING_DATASET": "Loading HAM10K Dataset...",
    "INITIALIZING_SERVER": "Initializing Federated Learning Server...",
    "WAITING_FOR_CLIENTS": "Waiting for clients to connect...",
    
    # Training phases
    "TRAINING_START": "Initializing {mode} Training...",
    "TRAINING_ROUND": "Training {mode} - Round {round}/{total_rounds}",
    "TRAINING_EPOCH": "Training {mode} - Round {round} Epoch {epoch}/{total_epochs}",
    "ROUND_COMPLETED": "Completed {mode} - Round {round}/{total_rounds}",
    "CLIENT_TRAINING": "Clients performing local training...",
    "GLOBAL_TRAINING": "Training global fusion model...",
    "GUIDANCE_PHASE": "Sending guidance to clients...",
    
    # Evaluation phase
    "EVALUATING": "Evaluating {mode} Model...",
    "CALCULATING_METRICS": "Calculating Performance Metrics...",
    "CALCULATING_LEAKAGE": "Analyzing Privacy Leakage...",
    "FAIRNESS_EVALUATION": "Evaluating Fairness Constraints...",
    "INFERENCE_ATTACKS": "Running Inference Attack Analysis...",
    
    # Federation specific
    "FEDERATION_SYNC": "Synchronizing Federation Clients...",
    "COLLECTING_EMBEDDINGS": "Collecting embeddings from clients...",
    "AGGREGATING_UPDATES": "Aggregating client updates...",
    "DISTRIBUTING_MODEL": "Distributing updated model...",
    
    # Completion states
    "TRAINING_COMPLETED": "Federated Learning Training Completed Successfully",
    "EVALUATION_COMPLETED": "Final Model Evaluation Completed",
    "FINAL_RESULTS": "Generating final results and metrics...",
    "SAVING_RESULTS": "Saving training results and models...",
    "TRAINING_STOPPED": "Training stopped by user",
    
    # Progress states
    "PHASE_1": "Phase 1: Initial Client Training",
    "PHASE_2": "Phase 2: Federated Learning Rounds",
    "PHASE_3": "Phase 3: Final Model Evaluation",
    
    # Error states
    "ERROR_LOADING": "Error: Failed to Load Dataset",
    "ERROR_TRAINING": "Error: Training Failed",
    "ERROR_EVALUATION": "Error: Evaluation Failed",
    "ERROR_CONNECTION": "Error: Client Connection Failed",
    "ERROR_SERVER": "Error: Server Communication Failed",
    
    # Additional detailed states
    "PREPARING_DATA": "Preparing Training Data...",
    "SAVING_MODEL": "Saving Model Checkpoints...",
    "LOADING_CHECKPOINT": "Loading Model Checkpoint...",
    "OPTIMIZING_HYPERPARAMS": "Optimizing Hyperparameters...",
    "ADVERSARIAL_TRAINING": "Training Adversarial Components...",
    "CREATING_MODELS": "Creating neural network models...",
    "VALIDATING_PERFORMANCE": "Validating model performance...",
    "PROCESSING_RESULTS": "Processing training results...",
}

# Status message helper functions
def get_status(status_key, **kwargs):
    """Get a formatted status message"""
    
    if status_key not in STATUS_MESSAGES:
        return f"Unknown Status: {status_key}"
    
    try:
        return STATUS_MESSAGES[status_key].format(**kwargs)
    except KeyError as e:
        return f"Status Error: Missing variable {e}"

# Pre-defined status combinations for common scenarios
def get_training_status(mode, round_num, total_rounds, epoch=None, total_epochs=None):
    """Get training status with automatic message selection"""
    if epoch is not None:
        return get_status("TRAINING_EPOCH", 
                         mode=mode, round=round_num, total_rounds=total_rounds, 
                         epoch=epoch, total_epochs=total_epochs)
    else:
        return get_status("TRAINING_ROUND", 
                         mode=mode, round=round_num, total_rounds=total_rounds)

def get_completion_status(mode, round_num, total_rounds):
    """Get round completion status"""
    return get_status("ROUND_COMPLETED", 
                     mode=mode, round=round_num, total_rounds=total_rounds)

def get_evaluation_status(mode):
    """Get evaluation status"""
    return get_status("EVALUATING", mode=mode)

def get_phase_status(phase_num):
    """Get phase status"""
    return get_status(f"PHASE_{phase_num}")

def get_federation_status(action):
    """Get federation-specific status"""
    federation_actions = {
        'sync': 'FEDERATION_SYNC',
        'collect': 'COLLECTING_EMBEDDINGS',
        'aggregate': 'AGGREGATING_UPDATES',
        'distribute': 'DISTRIBUTING_MODEL'
    }
    return get_status(federation_actions.get(action, 'FEDERATION_SYNC'))

# Status priority levels (for determining which status to show when multiple are active)
STATUS_PRIORITY = {
    'ERROR_': 1,      # Highest priority
    'TRAINING_': 2,
    'EVALUATING': 3,
    'PHASE_': 4,
    'WAITING_': 5,    # Lowest priority
}

def get_status_priority(status_key):
    """Get priority level for status message"""
    for prefix, priority in STATUS_PRIORITY.items():
        if status_key.startswith(prefix):
            return priority
    return 5  # Default to lowest priority 