"""
Network Configuration for Distributed Federated Learning
Supports both single-machine testing and multi-machine deployment.
"""

import os

class NetworkConfig:
    """Network configuration for distributed FL components."""
    
    # Default configuration for single-machine testing
    DEFAULT_CONFIG = {
        'server': {
            'host': 'localhost',
            'port': 8080,
            'timeout': 300,  # 5 minutes
            'max_retries': 3
        },
        'image_client': {
            'host': 'localhost', 
            'port': 8081,
            'client_id': 'image_client',
            'data_type': 'image'
        },
        'tabular_client': {
            'host': 'localhost',
            'port': 8082, 
            'client_id': 'tabular_client',
            'data_type': 'tabular'
        }
    }
    
    # Multi-machine configuration (for actual deployment)
    DISTRIBUTED_CONFIG = {
        'server': {
            'host': '0.0.0.0',  # Listen on all interfaces
            'port': 8080,
            'timeout': 300,
            'max_retries': 3
        },
        'image_client': {
            'host': '192.168.1.101',  # Image client mini-PC IP
            'port': 8081,
            'client_id': 'image_client',
            'data_type': 'image'
        },
        'tabular_client': {
            'host': '192.168.1.102',  # Tabular client mini-PC IP
            'port': 8082,
            'client_id': 'tabular_client', 
            'data_type': 'tabular'
        }
    }
    
    @classmethod
    def get_config(cls, mode='local'):
        """
        Get network configuration based on deployment mode.
        
        Args:
            mode (str): 'local' for single-machine testing, 'distributed' for multi-machine
        
        Returns:
            dict: Network configuration
        """
        if mode == 'distributed':
            return cls.DISTRIBUTED_CONFIG.copy()
        else:
            return cls.DEFAULT_CONFIG.copy()
    
    @classmethod
    def get_server_url(cls, mode='local'):
        """Get server URL for clients to connect to."""
        config = cls.get_config(mode)
        server_config = config['server']
        
        if mode == 'distributed':
            # For distributed mode, clients need to know the actual server IP
            server_ip = os.getenv('SERVER_IP', '192.168.1.100')  # Default server IP
            return f"http://{server_ip}:{server_config['port']}"
        else:
            return f"http://localhost:{server_config['port']}"
    
    @classmethod
    def get_client_config(cls, client_type, mode='local'):
        """Get configuration for specific client type."""
        config = cls.get_config(mode)
        return config.get(f'{client_type}_client', {})

def get_server_config(mode='local'):
    """Get server configuration for the specified mode."""
    config = NetworkConfig.get_config(mode)
    return config['server']

def get_client_config(client_type, mode='local'):
    """Get client configuration for the specified type and mode."""
    config = NetworkConfig.get_config(mode)
    client_config = config.get(f'{client_type}_client', {})
    
    # Add server information for client to connect to
    if mode == 'distributed':
        server_ip = os.getenv('SERVER_IP', '192.168.1.100')
        client_config['server_host'] = server_ip
        client_config['server_port'] = config['server']['port']
    else:
        client_config['server_host'] = 'localhost'
        client_config['server_port'] = config['server']['port']
    
    return client_config

# API Endpoints
class APIEndpoints:
    """REST API endpoints for FL communication."""
    
    # Server endpoints
    SERVER_ENDPOINTS = {
        'register_client': '/api/register',
        'start_training': '/api/start_training',
        'submit_embeddings': '/api/submit_embeddings',
        'get_global_model': '/api/get_global_model',
        'get_training_status': '/api/status',
        'notify_training_complete': '/api/training_complete'
    }
    
    # Client endpoints (for server to call clients)
    CLIENT_ENDPOINTS = {
        'train_local': '/api/train_local',
        'generate_embeddings': '/api/generate_embeddings',
        'health_check': '/api/health'
    }

# Request/Response schemas
class MessageSchemas:
    """Standard message schemas for FL communication."""
    
    CLIENT_REGISTRATION = {
        'client_id': str,
        'client_type': str,  # 'image' or 'tabular'
        'host': str,
        'port': int,
        'capabilities': dict
    }
    
    TRAINING_REQUEST = {
        'round_id': int,
        'epochs': int,
        'batch_size': int,
        'learning_rate': float
    }
    
    EMBEDDINGS_SUBMISSION = {
        'client_id': str,
        'round_id': int,
        'data_split': str,  # 'train', 'val', 'test'
        'embeddings': list,  # Serialized embeddings
        'labels': list,
        'metadata': dict
    } 