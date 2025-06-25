from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import webbrowser
import threading
import time
import os
import random
import json
from datetime import datetime
import subprocess
import signal
import platform
import atexit
import requests

# Import status configuration
from status_config import get_status, get_training_status, get_completion_status, get_evaluation_status, get_phase_status, get_federation_status

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for dashboard state
current_tab = 'home'
training_active = False
training_start_time = None
training_timer = 0
data_monitor_thread = None
last_round_seen = 0

# Timer control variables
timer_running = False
timer_start_time = None
timer_pause_time = None
timer_elapsed_seconds = 0

# Process management variables
server_process = None
image_client_process = None
tabular_client_process = None
processes_running = False

# Configuration state with status tracking
current_config = {
    'data_percentage': 5,      # Default 5%
    'federated_rounds': 5,     # Default 5 rounds
    'epochs_per_round': 5,     # Default 5 epochs
    'current_round': 0,
    'progress': 0,
    'current_status': get_status('DASHBOARD_READY'),
    'detailed_status': 'Ready to start federated learning training'
}

# Live metrics storage
live_metrics = {
    'home': {
        'training_status': 'Waiting for FL training to start...',
        'status_line2': 'Run "python server.py" to start DistributedVFL training',
        'running_time': '00:00:00',
        'sample_counts': {
            'training': 1000,
            'validation': 200,
            'test': 100
        }
    },
    'performance': {
        'live_accuracy': [],     # Start empty - will populate with real FL data
        'live_loss': [],         # Start empty - will populate with real FL data  
        'f1_score': [],          # Start empty - will populate with real FL data
        'precision_recall': [],  # Start empty - will populate with real FL data
        'gender_accuracy': [],
        'age_accuracy': []
    },
    'attack': {
        'gender_leakage': [],  # Start empty - will populate with real FL data
        'age_leakage': [],     # Start empty - will populate with real FL data
        'connection_status': True
    },
    'defence': {
        'protection_active': False,
        'age_protection': 92.8,
        'gender_protection': 92.8,
        'age_leakage': 92.8,
        'gender_leakage_score': 92.8,
        'defence_strength': []
    }
}

def read_fl_status():
    """Read current FL training status from DistributedVFL status file"""
    try:
        status_file = 'status/status.json'
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                data = json.load(f)
            
            # Debug: Print key status info
            print(f"Status: training_active={data.get('training_active')}, round={data.get('current_round')}, accuracy={data.get('accuracy')}")
            
            return {
                'training_active': data.get('training_active', False) or data.get('status') in ['running', 'training'],
                'current_round': data.get('current_round', 0),
                'total_rounds': data.get('total_rounds', 2),
                'accuracy': data.get('accuracy', 0) * 100 if data.get('accuracy') else 0,
                'loss': data.get('loss', 0),
                'f1_score': data.get('f1_score', 0) * 100 if data.get('f1_score') else 0,
                'precision_recall': data.get('precision_recall', 0) * 100 if data.get('precision_recall') else 0,
                'gender_fairness': data.get('gender_fairness', [0.0, 0.0]),
                'age_fairness': data.get('age_fairness', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                'age_leakage': data.get('age_leakage', 16.67),
                'gender_leakage': data.get('gender_leakage', 50.0),
                'defense_strength': data.get('defense_strength', 0) if data.get('defense_strength') else 0,
                'progress_percent': data.get('progress_percent', 0),
                'phase': data.get('phase', 'waiting'),
                'timestamp': data.get('timestamp', datetime.now().isoformat()),
                'status': data.get('status', 'waiting')
            }
        else:
            return None
    except Exception as e:
        print(f"Error reading FL status: {e}")
        return None

def fetch_sample_counts():
    """Fetch sample counts from the server."""
    try:
        response = requests.get('http://localhost:8080/get_sample_counts', timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                'training_samples': data.get('training_samples', 0),
                'validation_samples': data.get('validation_samples', 0),
                'test_samples': data.get('test_samples', 0),
                'total_clients': data.get('total_clients', 0)
            }
        else:
            print(f"Failed to fetch sample counts: {response.status_code}")
            return None
    except Exception as e:
        # Don't print error if server is not running yet
        if "Connection refused" not in str(e):
            print(f"Error fetching sample counts: {e}")
        return None

def monitor_fl_training():
    """Monitor FL training data from status.json file with enhanced status tracking"""
    global training_active, last_round_seen, current_config, live_metrics
    
    while True:
        try:
            fl_status = read_fl_status()
            
            if fl_status:
                # Update training status
                if fl_status['training_active'] and not training_active:
                    training_active = True
                    start_timer()  # Start timer when training begins
                    print("FL Training detected - Dashboard now monitoring")
                    
                    # Update status based on phase
                    if fl_status['phase'] == 'initial_training':
                        update_status('PHASE_1')
                    elif fl_status['phase'] == 'training':
                        update_status('PHASE_2')
                    elif fl_status['phase'] == 'evaluation':
                        update_status('PHASE_3')
                    else:
                        update_status('TRAINING_START', mode='Federated Learning')
                    
                    # Clear mock data when real FL training starts
                    live_metrics['performance']['live_accuracy'] = []
                    live_metrics['performance']['live_loss'] = []
                    live_metrics['performance']['f1_score'] = []
                    live_metrics['performance']['precision_recall'] = []
                    live_metrics['defence']['defence_strength'] = []
                    live_metrics['attack']['age_leakage'] = []
                    live_metrics['attack']['gender_leakage'] = []
                    last_round_seen = 0  # Reset round counter
                    print("Cleared mock data - ready for real FL data")
                    
                elif not fl_status['training_active'] and training_active:
                    training_active = False
                    pause_timer()  # Pause timer when training finishes
                    update_status('TRAINING_COMPLETED')
                    print("FL Training completed - Timer paused")
                
                # Update current round and config
                current_config['current_round'] = fl_status['current_round']
                current_config['federated_rounds'] = fl_status['total_rounds']
                current_config['progress'] = fl_status['progress_percent']
                
                # Update detailed status based on training phase
                if training_active:
                    if fl_status['phase'] == 'initial_training':
                        update_detailed_status("Clients performing initial local training")
                        update_status('CLIENT_TRAINING')
                    elif fl_status['phase'] == 'training':
                        if fl_status['current_round'] > 0:
                            update_status('TRAINING_ROUND', 
                                        mode='Federated Learning',
                                        round=fl_status['current_round'],
                                        total_rounds=fl_status['total_rounds'])
                            update_detailed_status(f"Training global model - Round {fl_status['current_round']}/{fl_status['total_rounds']}")
                        else:
                            update_status('GLOBAL_TRAINING')
                            update_detailed_status("Training global fusion model")
                    elif fl_status['phase'] == 'evaluation':
                        update_status('EVALUATING', mode='Global')
                        update_detailed_status("Evaluating final model performance")
                    elif fl_status['phase'] == 'inference_attacks':
                        update_status('INFERENCE_ATTACKS')
                        update_detailed_status("Analyzing privacy leakage through inference attacks")
                    elif fl_status['phase'] == 'completed':
                        update_status('TRAINING_COMPLETED')
                        update_detailed_status("Federated learning completed successfully")
                else:
                    update_status('WAITING_FOR_TRAINING')
                    update_detailed_status("Ready to start federated learning training")
                
                # Add new data point when a round completes
                if fl_status['current_round'] > last_round_seen and fl_status['current_round'] > 0:
                    print(f"New FL Round {fl_status['current_round']} completed!")
                    print(f"Accuracy: {fl_status['accuracy']:.2f}%")
                    print(f"Loss: {fl_status['loss']:.4f}")
                    print(f"F1 Score: {fl_status['f1_score']:.2f}%")
                    print(f"Precision-Recall: {fl_status['precision_recall']:.2f}%")
                    print(f"Defense Strength: {fl_status['defense_strength']:.2f}%")
                    print(f"Last round seen: {last_round_seen}, Current round: {fl_status['current_round']}")
                    
                    # Update status for round completion
                    update_status('ROUND_COMPLETED', 
                                mode='Federated Learning',
                                round=fl_status['current_round'],
                                total_rounds=fl_status['total_rounds'])
                    
                    # Add to performance metrics
                    live_metrics['performance']['live_accuracy'].append(fl_status['accuracy'])
                    live_metrics['performance']['live_loss'].append(fl_status['loss'])
                    live_metrics['performance']['f1_score'].append(fl_status['f1_score'])
                    live_metrics['performance']['precision_recall'].append(fl_status['precision_recall'])
                    live_metrics['performance']['gender_accuracy'] = fl_status['gender_fairness']  # Replace, not append
                    live_metrics['performance']['age_accuracy'] = fl_status['age_fairness']  # Replace, not append
                    
                    # Debug: Print accumulated metrics
                    print(f"Accumulated accuracy data: {live_metrics['performance']['live_accuracy']}")
                    print(f"Accumulated loss data: {live_metrics['performance']['live_loss']}")
                    print(f"Accumulated F1 data: {live_metrics['performance']['f1_score']}")
                    print(f"Accumulated P-R data: {live_metrics['performance']['precision_recall']}")
                    
                    # Add to attack metrics (leakage data)
                    live_metrics['attack']['age_leakage'].append(fl_status['age_leakage'])
                    live_metrics['attack']['gender_leakage'].append(fl_status['gender_leakage'])
                    
                    # Add to defense metrics
                    live_metrics['defence']['defence_strength'].append(fl_status['defense_strength'])
                    
                    # Keep only last 10 points for better visualization
                    for metric in ['live_accuracy', 'live_loss', 'f1_score', 'precision_recall']:
                        if len(live_metrics['performance'][metric]) > 10:
                            live_metrics['performance'][metric] = live_metrics['performance'][metric][-10:]
                    
                    # Keep attack metrics limited too
                    for metric in ['age_leakage', 'gender_leakage']:
                        if len(live_metrics['attack'][metric]) > 10:
                            live_metrics['attack'][metric] = live_metrics['attack'][metric][-10:]
                    
                    # Keep defense metrics limited too
                    if len(live_metrics['defence']['defence_strength']) > 10:
                        live_metrics['defence']['defence_strength'] = live_metrics['defence']['defence_strength'][-10:]
                    
                    last_round_seen = fl_status['current_round']
                    
                    # Emit update to all connected clients
                    socketio.emit('fl_data_update', {
                        'config': current_config,
                        'metrics': live_metrics,
                        'training_active': training_active,
                        'fl_status': fl_status,
                        'timer_elapsed': get_elapsed_time(),
                        'timer_running': timer_running
                    })
                    
                    # Also emit individual metric updates
                    socketio.emit('fairness_update', {
                        'gender_fairness': fl_status['gender_fairness'],
                        'age_fairness': fl_status['age_fairness']
                    })
                    
                    socketio.emit('leakage_update', {
                        'gender_leakage': fl_status['gender_leakage'],
                        'age_leakage': fl_status['age_leakage']
                    })
            
            else:
                # No status file found
                if training_active:
                    training_active = False
                    pause_timer()
                    update_status('ERROR_LOADING')
                    update_detailed_status("Status file not found - start FL training first")
            
            # Emit regular updates to keep dashboard synchronized
            socketio.emit('fl_data_update', {
                'config': current_config,
                'metrics': live_metrics,
                'training_active': training_active,
                'timer_elapsed': get_elapsed_time(),
                'timer_running': timer_running
            })
            
        except Exception as e:
            print(f"Error in FL monitoring: {e}")
        
        # Fetch and emit sample counts from server
        try:
            sample_counts = fetch_sample_counts()
            if sample_counts:
                socketio.emit('sample_counts_update', sample_counts)
        except Exception as e:
            # Silently ignore connection errors when server is not running
            pass
        
        # Check every 2 seconds
        time.sleep(2)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/asset/<path:filename>')
def serve_asset(filename):
    """Serve files from the asset folder"""
    return send_from_directory('asset', filename)

@socketio.on('connect')
def handle_connect():
    """Send initial dashboard state when client connects"""
    print(f"Client connected")
    emit('dashboard_state', {
        'current_tab': current_tab,
        'config': current_config,
        'metrics': live_metrics,
        'training_active': training_active,
        'timer_elapsed': get_elapsed_time(),
        'timer_running': timer_running
    })

@socketio.on('switch_tab')
def handle_tab_switch(data):
    """Handle tab switching"""
    global current_tab
    current_tab = data.get('tab', 'home')
    emit('tab_switched', {'tab': current_tab}, broadcast=True)

@socketio.on('update_config')
def handle_config_update(data):
    """Handle configuration updates from frontend"""
    global current_config
    config_type = data.get('type')
    value = data.get('value')
    
    # Only allow config changes when not running
    if processes_running:
        emit('config_error', {
            'message': 'Cannot change configuration while training is running',
            'type': config_type
        })
        return
    
    if config_type == 'data_percentage':
        current_config['data_percentage'] = value
        print(f"Data percentage updated to: {value}%")
    elif config_type == 'federated_rounds':
        current_config['federated_rounds'] = value
        print(f"Federated rounds updated to: {value}")
    elif config_type == 'epochs_per_round':
        current_config['epochs_per_round'] = value
        print(f"Epochs per round updated to: {value}")
    
    # Broadcast updated config to all clients
    emit('config_updated', {
        'type': config_type,
        'value': value,
        'full_config': current_config
    }, broadcast=True)
    
    print(f"Current config: {current_config}")

@socketio.on('training_control')
def handle_training_control(data):
    """Handle training control - Start/Stop server and clients"""
    global processes_running
    
    action = data.get('action')
    
    if action == 'play':
        if not processes_running:
            print(f"Starting FL training with config: {current_config}")
            
            # Update status and start timer
            update_status('INITIALIZING_SERVER')
            start_timer()
            
            success = start_server_process()
            if success:
                update_status('WAITING_FOR_CLIENTS')
                emit('training_status', {
                    'status': 'starting',
                    'message': f'Starting server with {current_config["data_percentage"]}% data, {current_config["federated_rounds"]} rounds, {current_config["epochs_per_round"]} epochs',
                    'config': current_config,
                    'timer_running': True,
                    'timer_elapsed': get_elapsed_time()
                }, broadcast=True)
            else:
                # Reset timer on failure
                reset_timer()
                update_status('ERROR_SERVER')
                emit('training_status', {
                    'status': 'error',
                    'message': 'Failed to start server process',
                    'config': current_config,
                    'timer_running': False,
                    'timer_elapsed': get_elapsed_time()
                }, broadcast=True)
        else:
            emit('training_status', {
                'status': 'already_running',
                'message': 'Training is already running',
                'config': current_config,
                'timer_running': timer_running,
                'timer_elapsed': get_elapsed_time()
            })
            
    elif action == 'stop':
        if processes_running:
            print("Stopping FL training")
            
            # Pause timer and update status
            pause_timer()
            update_status('TRAINING_STOPPED')
            
            success = stop_all_processes()
            if success:
                emit('training_status', {
                    'status': 'stopped',
                    'message': 'All processes stopped successfully',
                    'config': current_config,
                    'timer_running': False,
                    'timer_elapsed': get_elapsed_time()
                }, broadcast=True)
            else:
                emit('training_status', {
                    'status': 'error',
                    'message': 'Error stopping some processes',
                    'config': current_config,
                    'timer_running': False,
                    'timer_elapsed': get_elapsed_time()
                }, broadcast=True)
        else:
            # Reset timer when stop is clicked and nothing is running
            reset_timer()
            update_status('DASHBOARD_READY')
            emit('training_status', {
                'status': 'already_stopped',
                'message': 'Training is not running',
                'config': current_config,
                'timer_running': False,
                'timer_elapsed': get_elapsed_time()
            })

@socketio.on('defence_control')
def handle_defence_control(data):
    """Handle defence protection controls with live server communication"""
    action = data.get('action')
    
    try:
        if action == 'run_protection':
            # Send API request to server to activate defense
            response = requests.post('http://localhost:8080/update_adversarial_lambda', 
                                   json={'adversarial_lambda': 0.3},  # Default protection level
                                   timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                live_metrics['defence']['protection_active'] = True
                emit('defence_status', {
                    'protection_active': True,
                    'lambda_value': result.get('new_lambda', 0.3),
                    'message': 'Protection activated - Adversarial training enabled'
                }, broadcast=True)
                print("🛡️ Dashboard activated server defense (λ=0.3)")
            else:
                emit('defence_status', {
                    'protection_active': False,
                    'error': 'Failed to activate server defense',
                    'message': 'Error: Could not communicate with server'
                }, broadcast=True)
                
        elif action == 'stop_protection':
            # Send API request to server to deactivate defense
            response = requests.post('http://localhost:8080/update_adversarial_lambda', 
                                   json={'adversarial_lambda': 0.0},  # Disable protection
                                   timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                live_metrics['defence']['protection_active'] = False
                emit('defence_status', {
                    'protection_active': False,
                    'lambda_value': result.get('new_lambda', 0.0),
                    'message': 'Protection deactivated - Standard training resumed'
                }, broadcast=True)
                print("🛡️ Dashboard deactivated server defense (λ=0.0)")
            else:
                emit('defence_status', {
                    'protection_active': False,
                    'error': 'Failed to deactivate server defense',
                    'message': 'Error: Could not communicate with server'
                }, broadcast=True)
                
    except Exception as e:
        print(f"Error in defense control: {e}")
        emit('defence_status', {
            'protection_active': False,
            'error': str(e),
            'message': f'Error: {str(e)}'
        }, broadcast=True)

def kill_process_on_port(port=5050):
    """Kill any existing process running on the specified port."""
    try:
        import psutil
        print(f"Checking for existing processes on port {port}...")
        
        # Find processes using the port
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                connections = proc.info['connections']
                if connections:
                    for conn in connections:
                        if hasattr(conn, 'laddr') and conn.laddr and conn.laddr.port == port:
                            print(f"Killing process {proc.info['pid']} ({proc.info['name']}) on port {port}")
                            proc.kill()
                            proc.wait(timeout=3)
                            print(f"Process {proc.info['pid']} killed successfully")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
                
    except ImportError:
        # Fallback method using lsof and kill
        print(f"Using lsof to check port {port}...")
        try:
            # Find process using the port
            result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid:
                        print(f"Killing process {pid} on port {port}")
                        subprocess.run(['kill', '-9', pid], timeout=5)
                        print(f"Process {pid} killed successfully")
            else:
                print(f"✅ No existing process found on port {port}")
        except subprocess.TimeoutExpired:
            print(f"Timeout while checking port {port}")
        except FileNotFoundError:
            print(f"lsof command not found, skipping port cleanup")
        except Exception as e:
            print(f"Error during port cleanup: {e}")
    except Exception as e:
        print(f"Error during port cleanup: {e}")

def open_browser():
    """Open browser after a short delay."""
    time.sleep(2)
    webbrowser.open('http://localhost:5050')

def get_terminal_command():
    """Get platform-specific terminal command."""
    system = platform.system().lower()
    if system == 'darwin':  # macOS
        return 'osascript -e \'tell app "Terminal" to do script "{}"\''.format
    elif system == 'linux':
        return 'gnome-terminal -- bash -c "{}"'.format
    elif system == 'windows':
        return 'start cmd /k "{}"'.format
    else:
        return 'bash -c "{}"'.format  # Fallback

def start_server_process():
    """Start the FL server process in a separate terminal."""
    global server_process, processes_running
    
    try:
        # Get current working directory
        current_dir = os.getcwd()
        
        # Create command with current configuration
        cmd = f"cd '{current_dir}' && python3 server.py --data_percentage={current_config['data_percentage']} --fl_rounds={current_config['federated_rounds']} --epochs_per_round={current_config['epochs_per_round']}"
        
        # Get platform-specific terminal command
        terminal_cmd = get_terminal_command()
        full_cmd = terminal_cmd(cmd)
        
        print(f"Starting server with command: {cmd}")
        print(f"Terminal command: {full_cmd}")
        
        # Start server in new terminal
        if platform.system().lower() == 'darwin':  # macOS
            server_process = subprocess.Popen(['osascript', '-e', f'tell app "Terminal" to do script "{cmd}"'])
        elif platform.system().lower() == 'linux':
            server_process = subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', cmd])
        elif platform.system().lower() == 'windows':
            server_process = subprocess.Popen(['start', 'cmd', '/k', cmd], shell=True)
        else:
            # Fallback - run in background
            server_process = subprocess.Popen(['python3', 'server.py', 
                                             f'--data_percentage={current_config["data_percentage"]}',
                                             f'--fl_rounds={current_config["federated_rounds"]}',
                                             f'--epochs_per_round={current_config["epochs_per_round"]}'],
                                           cwd=current_dir)
        
        processes_running = True
        print(f"Server process started with PID: {server_process.pid}")
        
        # Schedule client startup after 5 seconds
        threading.Timer(5.0, start_client_processes).start()
        
        return True
        
    except Exception as e:
        print(f"Error starting server: {e}")
        return False

def start_client_processes():
    """Start client processes 5 seconds after server."""
    global image_client_process, tabular_client_process
    
    try:
        print("Starting client processes...")
        
        # Get current working directory
        current_dir = os.getcwd()
        
        # Get platform-specific terminal command
        terminal_cmd = get_terminal_command()
        
        # Start image client
        image_cmd = f"cd '{current_dir}' && python3 image_client.py"
        if platform.system().lower() == 'darwin':  # macOS
            image_client_process = subprocess.Popen(['osascript', '-e', f'tell app "Terminal" to do script "{image_cmd}"'])
        elif platform.system().lower() == 'linux':
            image_client_process = subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', image_cmd])
        elif platform.system().lower() == 'windows':
            image_client_process = subprocess.Popen(['start', 'cmd', '/k', image_cmd], shell=True)
        else:
            image_client_process = subprocess.Popen(['python3', 'image_client.py'], cwd=current_dir)
        
        print(f"Image client started with PID: {image_client_process.pid}")
        
        # Start tabular client
        tabular_cmd = f"cd '{current_dir}' && python3 tabular_client.py"
        if platform.system().lower() == 'darwin':  # macOS
            tabular_client_process = subprocess.Popen(['osascript', '-e', f'tell app "Terminal" to do script "{tabular_cmd}"'])
        elif platform.system().lower() == 'linux':
            tabular_client_process = subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', tabular_cmd])
        elif platform.system().lower() == 'windows':
            tabular_client_process = subprocess.Popen(['start', 'cmd', '/k', tabular_cmd], shell=True)
        else:
            tabular_client_process = subprocess.Popen(['python3', 'tabular_client.py'], cwd=current_dir)
        
        print(f"Tabular client started with PID: {tabular_client_process.pid}")
        
    except Exception as e:
        print(f"Error starting clients: {e}")

def stop_all_processes():
    """Stop all running processes."""
    global server_process, image_client_process, tabular_client_process, processes_running
    
    try:
        print("Stopping all processes...")
        
        # Stop server process
        if server_process:
            try:
                if platform.system().lower() == 'windows':
                    server_process.terminate()
                else:
                    server_process.kill()
                server_process.wait(timeout=5)
                print("Server process stopped")
            except Exception as e:
                print(f"Error stopping server: {e}")
            server_process = None
        
        # Stop image client
        if image_client_process:
            try:
                if platform.system().lower() == 'windows':
                    image_client_process.terminate()
                else:
                    image_client_process.kill()
                image_client_process.wait(timeout=5)
                print("Image client stopped")
            except Exception as e:
                print(f"Error stopping image client: {e}")
            image_client_process = None
        
        # Stop tabular client
        if tabular_client_process:
            try:
                if platform.system().lower() == 'windows':
                    tabular_client_process.terminate()
                else:
                    tabular_client_process.kill()
                tabular_client_process.wait(timeout=5)
                print("Tabular client stopped")
            except Exception as e:
                print(f"Error stopping tabular client: {e}")
            tabular_client_process = None
        
        processes_running = False
        print("All processes stopped")
        
        return True
        
    except Exception as e:
        print(f"Error stopping processes: {e}")
        return False

def clean_status_files():
    """Clean old status files to prevent interference"""
    status_dir = "status"
    if not os.path.exists(status_dir):
        os.makedirs(status_dir)
    
    # Remove any existing status files
    status_file = os.path.join(status_dir, "status.json")
    if os.path.exists(status_file):
        os.remove(status_file)
        print("Removed old status file")
    
    print("Status directory cleaned for fresh start")

def cleanup_on_exit():
    """Cleanup function to run on dashboard exit."""
    print("Dashboard shutting down - cleaning up processes...")
    stop_all_processes()

# Register cleanup function
atexit.register(cleanup_on_exit)

# Timer control functions
def start_timer():
    """Start or resume the training timer"""
    global timer_running, timer_start_time, timer_pause_time
    
    if not timer_running:
        if timer_pause_time is not None:
            # Resume from pause
            pause_duration = time.time() - timer_pause_time
            timer_start_time += pause_duration
            timer_pause_time = None
        else:
            # Fresh start
            timer_start_time = time.time()
        
        timer_running = True
        print(f"Timer started")

def pause_timer():
    """Pause the training timer"""
    global timer_running, timer_pause_time, timer_elapsed_seconds
    
    if timer_running:
        timer_running = False
        timer_pause_time = time.time()
        timer_elapsed_seconds = timer_pause_time - timer_start_time
        print(f"Timer paused at {format_time(timer_elapsed_seconds)}")

def reset_timer():
    """Reset the training timer to 00:00:00"""
    global timer_running, timer_start_time, timer_pause_time, timer_elapsed_seconds
    
    timer_running = False
    timer_start_time = None
    timer_pause_time = None
    timer_elapsed_seconds = 0
    print(f"Timer reset to 00:00:00")

def get_elapsed_time():
    """Get current elapsed time in seconds"""
    global timer_running, timer_start_time, timer_elapsed_seconds
    
    if timer_start_time is None:
        return 0
    
    if timer_running:
        return time.time() - timer_start_time
    else:
        return timer_elapsed_seconds

def format_time(seconds):
    """Format seconds into HH:MM:SS string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# Status update functions
def update_status(status_key, **kwargs):
    """Update the current status message"""
    global current_config
    
    try:
        new_status = get_status(status_key, **kwargs)
        current_config['current_status'] = new_status
        
        # Emit status update to all connected clients
        socketio.emit('status_update', {
            'status': new_status,
            'status_key': status_key,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"Status: {new_status}")
    except Exception as e:
        print(f"Error updating status: {e}")

def update_detailed_status(message):
    """Update the detailed status message"""
    global current_config
    
    current_config['detailed_status'] = message
    socketio.emit('detailed_status_update', {
        'detailed_status': message,
        'timestamp': datetime.now().isoformat()
    })

# Live metrics with enhanced status tracking

if __name__ == '__main__':
    print("Starting HYBRIDVFL Dashboard...")
    
    # Clean status files for fresh start
    clean_status_files()
    
    # Kill any existing process on port 5050
    kill_process_on_port(5050)
    
    print("Dashboard will be available at: http://localhost:5050")
    print("Monitoring DistributedVFL training status...")
    print("Reading from: status/status.json")
    print("Press Ctrl+C to stop")
    print()
    print("Workflow:")
    print("   1. Start this dashboard: python dashboard.py")
    print("   2. Start FL server: python server.py")
    print("   3. Start clients: python image_client.py & python tabular_client.py")
    print("   4. Watch live accuracy updates in Performance tab!")
    print()
    
    # Start background data monitoring
    data_monitor_thread = threading.Thread(target=monitor_fl_training, daemon=True)
    data_monitor_thread.start()
    
    # Open browser after delay
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Start Flask-SocketIO server
    try:
        socketio.run(app, host='0.0.0.0', port=5050, debug=False)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"\nDashboard error: {e}")
    finally:
        cleanup_on_exit() 