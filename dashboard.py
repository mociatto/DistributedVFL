from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import webbrowser
import threading
import time
import os
import random
import json
from datetime import datetime

app = Flask(__name__, static_folder='static', template_folder='templates')
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for dashboard state
current_tab = 'home'
training_active = False
training_start_time = None
training_timer = 0
data_monitor_thread = None
last_round_seen = 0

# Configuration states
current_config = {
    'data_percentage': 5.0,
    'federated_rounds': 2,
    'epochs_per_round': 2,
    'current_round': 0,
    'progress': 0.0
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
        'gender_leakage': [],
        'age_leakage': [],
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
            
            return {
                'training_active': data.get('status') in ['running', 'training'],
                'current_round': data.get('current_round', 0),
                'total_rounds': data.get('total_rounds', 2),
                'accuracy': data.get('metrics', {}).get('accuracy', 0) * 100 if data.get('metrics', {}).get('accuracy') else 0,
                'loss': data.get('metrics', {}).get('loss', 0),
                'f1_score': data.get('metrics', {}).get('f1_score', 0) * 100 if data.get('metrics', {}).get('f1_score') else 0,
                'precision_recall': data.get('metrics', {}).get('precision_recall', 0) * 100 if data.get('metrics', {}).get('precision_recall') else 0,
                'gender_fairness': data.get('metrics', {}).get('gender_fairness', [0.0, 0.0]),
                'age_fairness': data.get('metrics', {}).get('age_fairness', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                'defense_strength': data.get('metrics', {}).get('defense_strength', 0) if data.get('metrics', {}).get('defense_strength') else 0,
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

def monitor_fl_training():
    """Monitor FL training data from status.json file"""
    global training_active, last_round_seen, current_config, live_metrics
    
    while True:
        try:
            fl_status = read_fl_status()
            
            if fl_status:
                # Update training status
                if fl_status['training_active'] and not training_active:
                    training_active = True
                    print("🚀 FL Training detected - Dashboard now monitoring")
                    # Clear mock data when real FL training starts
                    live_metrics['performance']['live_accuracy'] = []
                    live_metrics['performance']['live_loss'] = []
                    live_metrics['performance']['f1_score'] = []
                    live_metrics['performance']['precision_recall'] = []
                    live_metrics['defence']['defence_strength'] = []
                    last_round_seen = 0  # Reset round counter
                    print("🧹 Cleared mock data - ready for real FL data")
                elif not fl_status['training_active'] and training_active:
                    training_active = False
                    print("⏹️ FL Training stopped")
                
                # Update current round and config
                current_config['current_round'] = fl_status['current_round']
                current_config['federated_rounds'] = fl_status['total_rounds']
                current_config['progress'] = fl_status['progress_percent']
                
                # Update training status messages
                if training_active:
                    live_metrics['home']['training_status'] = f"FL Training Round {fl_status['current_round']} - {fl_status['phase']}"
                    live_metrics['home']['status_line2'] = f"Accuracy: {fl_status['accuracy']:.2f}% | Loss: {fl_status['loss']:.4f} | F1: {fl_status['f1_score']:.2f}% | P-R: {fl_status['precision_recall']:.2f}%"
                else:
                    live_metrics['home']['training_status'] = "Waiting for FL training to start..."
                    live_metrics['home']['status_line2'] = "Run 'python server.py' to start DistributedVFL training"
                
                # Add new data point when a round completes
                if fl_status['current_round'] > last_round_seen and fl_status['current_round'] > 0:
                    print(f"📊 New FL Round {fl_status['current_round']} completed!")
                    print(f"📈 Accuracy: {fl_status['accuracy']:.2f}%")
                    print(f"📉 Loss: {fl_status['loss']:.4f}")
                    print(f"🎯 F1 Score: {fl_status['f1_score']:.2f}%")
                    print(f"🎯 Precision-Recall: {fl_status['precision_recall']:.2f}%")
                    print(f"🛡️ Defense Strength: {fl_status['defense_strength']:.2f}%")
                    
                    # Add to performance metrics
                    live_metrics['performance']['live_accuracy'].append(fl_status['accuracy'])
                    live_metrics['performance']['live_loss'].append(fl_status['loss'])
                    live_metrics['performance']['f1_score'].append(fl_status['f1_score'])
                    live_metrics['performance']['precision_recall'].append(fl_status['precision_recall'])
                    live_metrics['performance']['gender_accuracy'] = fl_status['gender_fairness']  # Replace, not append
                    live_metrics['performance']['age_accuracy'] = fl_status['age_fairness']  # Replace, not append
                    
                    # Add to defense metrics
                    live_metrics['defence']['defence_strength'].append(fl_status['defense_strength'])
                    
                    # Keep only last 10 points for better visualization
                    for metric in ['live_accuracy', 'live_loss', 'f1_score', 'precision_recall']:
                        if len(live_metrics['performance'][metric]) > 10:
                            live_metrics['performance'][metric] = live_metrics['performance'][metric][-10:]
                    
                    # Keep defense metrics limited too
                    if len(live_metrics['defence']['defence_strength']) > 10:
                        live_metrics['defence']['defence_strength'] = live_metrics['defence']['defence_strength'][-10:]
                    
                    last_round_seen = fl_status['current_round']
                    
                    # Emit update to all connected clients
                    socketio.emit('live_update', {
                        'config': current_config,
                        'metrics': live_metrics,
                        'training_active': training_active,
                        'fl_status': fl_status
                    })
                
                # Generate some mock data for attack charts only
                if training_active:
                    # Add mock data for attack metrics only (defense uses real data)
                    if random.random() > 0.7:  # 30% chance to add new data point
                        live_metrics['attack']['gender_leakage'].append(random.randint(10, 40))
                        live_metrics['attack']['age_leakage'].append(random.randint(15, 35))
                        
                        # Keep only last 10 points for attack metrics
                        for metric in live_metrics['attack']:
                            if isinstance(live_metrics['attack'][metric], list) and len(live_metrics['attack'][metric]) > 10:
                                live_metrics['attack'][metric] = live_metrics['attack'][metric][-10:]
            
            else:
                # No status file found
                if training_active:
                    training_active = False
                    live_metrics['home']['training_status'] = "No FL training detected"
                    live_metrics['home']['status_line2'] = "Status file not found - start FL training first"
            
            # Emit regular updates to keep dashboard synchronized
            socketio.emit('dashboard_update', {
                'config': current_config,
                'metrics': live_metrics,
                'training_active': training_active
            })
            
        except Exception as e:
            print(f"Error in FL monitoring: {e}")
        
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
    print(f"🔌 Client connected")
    emit('dashboard_state', {
        'current_tab': current_tab,
        'config': current_config,
        'metrics': live_metrics,
        'training_active': training_active
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
    
    if config_type in ['data_percentage', 'federated_rounds', 'epochs_per_round']:
        current_config[config_type] = value
        emit('config_updated', {
            'type': config_type,
            'value': value
        }, broadcast=True)

@socketio.on('training_control')
def handle_training_control(data):
    """Handle training control - Note: This dashboard monitors external FL training"""
    action = data.get('action')
    
    if action == 'play':
        emit('training_message', {
            'message': 'To start FL training, run "python server.py" in terminal',
            'type': 'info'
        })
    elif action == 'stop':
        emit('training_message', {
            'message': 'To stop FL training, press Ctrl+C in the server terminal',
            'type': 'info'
        })

@socketio.on('defence_control')
def handle_defence_control(data):
    """Handle defence protection controls"""
    action = data.get('action')
    
    if action == 'run_protection':
        live_metrics['defence']['protection_active'] = True
        emit('defence_status', {
            'protection_active': True,
            'message': 'Protection running - Confusing data inference'
        }, broadcast=True)
        
    elif action == 'stop_protection':
        live_metrics['defence']['protection_active'] = False
        emit('defence_status', {
            'protection_active': False,
            'message': 'Protection stopped - Data inference without protection'
        }, broadcast=True)

def open_browser():
    """Open browser after a short delay"""
    time.sleep(1.5)
    webbrowser.open('http://localhost:5050')

if __name__ == '__main__':
    print("🚀 Starting HYBRIDVFL Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:5050")
    print("📈 Monitoring DistributedVFL training status...")
    print("📁 Reading from: status/status.json")
    print("⏹️  Press Ctrl+C to stop")
    print()
    print("🔄 Workflow:")
    print("   1. Start this dashboard: python dashboard.py")
    print("   2. Start FL server: python server.py")
    print("   3. Start clients: python image_client.py & python tabular_client.py")
    print("   4. Watch live accuracy updates in Performance tab!")
    print()
    
    # Start FL monitoring thread
    data_monitor_thread = threading.Thread(target=monitor_fl_training, daemon=True)
    data_monitor_thread.start()
    
    # Open browser in background
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Run the Flask-SocketIO application
    try:
        socketio.run(app, host='0.0.0.0', port=5050, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n⏹️  Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}") 