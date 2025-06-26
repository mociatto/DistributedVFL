# HYBRIDVFL: Distributed Federated Learning for Skin Cancer Detection

A distributed federated learning system for medical image analysis using the HAM10000 skin lesion dataset. This system allows multimodal clients to train AI models together while keeping their data private and separate. All metrics and plots visulized on a beautiful live dashboard.

## What This Does

- Analyzes skin cancer images using advanced AI models
- Keeps data private - each client keeps its own data
- Trains together - clients collaborate without sharing raw data
- Works on multiple machines - perfect for hospitals or research labs
- Automated live dashboard handles full control on model configurations and running.

---

## How It Works

The system has 3 parts that run on separate computers:

```
SERVER (Port 8080)              IMAGE CLIENT (Port 8081)        TABULAR CLIENT (Port 8082)

│ • Coordinates training │       │ • Processes images   │       │ • Processes data     │
│ • Combines AI models   │  ←──→ │ • Has image files    │  ←──→ │ • Has CSV metadata   │
│ • No raw data needed   │       │ • Generates features │       │ • Generates features │
```

Each computer communicates through HTTP.

---

## Setup

### 1. Get the HAM10000 Dataset

Download from [Kaggle HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

You'll get:
- `HAM10000_images_part_1/` (folder with images)
- `HAM10000_images_part_2/` (folder with images)  
- `HAM10000_metadata.csv` (data about each image)

### 2. Install Python Packages

```bash
pip install -r requirements.txt
```

### 3. Distribute Data to Computers

**Server Computer:**
```
DistributedVFL/
├── server.py
├── models.py
├── config.py
└── (other Python files)
```

**Image Client Computer:**
```
DistributedVFL/
├── image_client.py  
├── data/
│   ├── HAM10000_images_part_1/
│   └── HAM10000_images_part_2/
└── (other Python files)
```

**Tabular Client Computer:**
```
DistributedVFL/
├── tabular_client.py
├── data/
│   └── HAM10000_metadata.csv
└── (other Python files)
```

---

## Running the System

### Control Everything on Live Dashboard

If you want to control and monitor the training process through a user-friendly interface:

**Terminal** - Start Dashboard:
```bash
python3 dashboard.py
```

After adjusting the data percentage, federated rounds, and epochs per round on the home tab, click the play button to start the system. All components will launch automatically, allowing you to monitor metrics and plots in real-time. The web dashboard will automatically open in your default browser at:
```
http://localhost:8080
```

### Single Computer Testing

If you want to test everything on one computer without using designed interface:

**Terminal 1** - Start Server:
```bash
python3 server.py
```

**Terminal 2** - Start Image Client:  
```bash
python3 image_client.py
```

**Terminal 3** - Start Tabular Client:
```bash
python3 tabular_client.py
```

The system will automatically start training when both clients connect.

### Multiple Computer Setup

#### Step 1: Configure Network

Edit `network_config.py` with your computer IPs:

```python
# Example IPs - change these to match your network
DISTRIBUTED_CONFIG = {
    'server': {
        'host': '0.0.0.0',  # Server listens to all
        'port': 8080
    }
}
```

#### Step 2: Start Components (in this order)

**On Server Computer:**
```bash
python3 server.py --mode distributed
```

**On Image Client Computer:**
```bash
python3 image_client.py --mode distributed --server_host 192.168.1.100
```

**On Tabular Client Computer:**
```bash
python3 tabular_client.py --mode distributed --server_host 192.168.1.100
```

Replace `192.168.1.100` with your server's actual IP.

---

### Defense Controls

The dashboard includes live defense controls on the Defense tab:

- **Run Protection**: Activates adversarial defense mechanisms during training to reduce inference attacks on sensitive attributes (age/gender). Sets adversarial lambda to 0.3.
- **Stop Protection**: Disables defense mechanisms and sets adversarial lambda to 0.0.

These controls work during active training and affect the next training round.

### Running the Model

1. Start all three components (server, image client, tabular client)
2. Training begins automatically when both clients connect
3. Monitor progress through the dashboard's various tabs
4. Use defense controls to manage privacy protection during training

---

## Network Configuration

### Ports Used
- **8080** - Server (main coordinator)
- **8081** - Image Client (when running locally)
- **8082** - Tabular Client (when running locally)

### Firewall Settings
Make sure these ports are open on your network.

### Finding Your Computer's IP
**Windows:**
```cmd
ipconfig
```

**Mac/Linux:**
```bash
ifconfig
```

---

## Configuration

### Speed vs Accuracy

In `config.py`, you can adjust:

```python
# Use less data for faster testing
'data_percentage': 0.05,  # 5% of data (fastest)
'data_percentage': 1.0,   # 100% of data (slow but best)

# Training rounds
'total_fl_rounds': 2,     # Quick test
'total_fl_rounds': 5,     # Better results
```

### Advanced Features

The system includes:
- EfficientNetV2 - Advanced image processing  
- Transformer Fusion - Smart data combination
- Attention Mechanisms - Focuses on important features
- Class Balance Handling - Works with uneven data
- Transfer Learning - Uses pre-trained models
- Live Defense System - Privacy protection during training

---

## Troubleshooting

### "Connection refused" 
- Start the server first
- Check if the IP address is correct
- Make sure firewall allows the ports

### "Failed to register"
- Server must be running before clients
- Check network connection between computers
- Verify port 8080 is open

### "Training failed"
- Make sure both clients are connected
- Check that data files exist in the right folders
- Ensure enough disk space for models

### Slow Performance
- Reduce `data_percentage` in config for faster testing
- Use fewer `total_fl_rounds` for quicker results

---

## For Medical/Research Use

This system is designed for:
- Hospitals with multiple locations
- Research collaborations between institutions  
- Privacy-sensitive medical data analysis
- Distributed AI without sharing raw patient data

The federated learning approach means each site keeps their data private while contributing to a shared AI model.

---

