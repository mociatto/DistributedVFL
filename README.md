# FairVFL-HAM10K: Fairness-Aware Vertical Federated Learning for Skin Cancer Detection

A modular Python implementation of **FairVFL** adapted for the HAM10000 skin lesion dataset.  
This project provides a research-ready codebase for exploring fairness and privacy in vertical federated learning using both image and tabular medical data and demonstrates metrics and plots on a modern front-end dashboard.

---

## Project Structure

- `main.py` — Entry point for training and evaluation  
- `data.py` — Data loading and preprocessing (HAM10000 images + metadata)  
- `model.py` — Model architectures (CNN, tabular encoder, fairness heads)  
- `train.py` — Training routines  
- `evaluate.py` — Evaluation and fairness audit routines  
- `dashboard.py` — Flask/SocketIO backend for dashboard live metrics  
- `templates/dashboard.html` — Dashboard front-end template  
- `statics/dashboard.css` — Dashboard CSS styles  
- `statics/dashboard.js` — Dashboard interactive JS  
- `requirements.txt` — Python dependencies  
- `.gitignore` — Standard ignores for Python and data  
- `/data/` — Place the HAM10000 dataset here  

---

## Setup

1. **Clone this repository.**

2. **Download the HAM10000 dataset** from [here](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)  
   Place the extracted folders and CSV in your `/data` directory as follows:

    ```
    data/
    ├── HAM10000_images_part_1/
    │   ├── ISIC_0024306.jpg
    │   ├── ...
    ├── HAM10000_images_part_2/
    │   ├── ISIC_0032012.jpg
    │   ├── ...
    ├── HAM10000_metadata.csv
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

---

## Run

### **Run the Project with Dashboard**

To launch the live dashboard and start FairVFL training with real-time metrics, run:

```bash
python dashboard.py
```

It opens http://localhost:5050 in your browser to view the dashboard.
Dashboard front-end is served via Flask/SocketIO and will automatically update with training progress.

### **Run CLI only**

If you only want to run the core training and evaluation via CLI, use:

```bash
python main.py
```
Note: For faster testing, you can reduce the percentage of data used by setting the `PERCENTAGE` variable in `main.py` to a lower value (e.g., `PERCENTAGE = 0.1` for 10% of the data). This will significantly speed up training and evaluation, making it ideal for quick experiments or debugging.**

```python
PERCENTAGE = 0.1  # Use only 10% of data for fast testing
``` 

## Recent Improvements (Latest Updates)

### ✅ **1. Enhanced Resume Functionality**
- **Comprehensive State Saving**: Now saves complete training state including best metrics, round history, and model configuration
- **Configuration Compatibility Check**: Validates model architecture compatibility when resuming
- **Usage**: `python3 main.py --resume` to continue from the best saved model
- **State Persistence**: Automatically saves training progress and can resume from any interruption

### ✅ **2. Performance Optimizations for Small Datasets**

#### **Model Architecture Improvements**:
- **Lightweight Image Encoder**: Reduced from 5.8M to 590K parameters (90% reduction)
  - Simplified CNN architecture with fewer filters
  - Removed redundant dense layers
  - Added stronger dropout regularization (0.3-0.5)
- **Simplified Tabular Encoder**: Reduced from 439K to 28K parameters (94% reduction)
  - Removed complex attention mechanisms
  - Streamlined architecture for 3-feature input
- **Optimized Fusion Model**: Reduced from 827K to 367K parameters (56% reduction)

#### **Better Class Imbalance Handling**:
- **Stratified Sampling**: Ensures minimum representation of all classes
- **Log-Balanced Class Weights**: New weighting strategy for extreme imbalance
- **Improved Data Distribution**: Guarantees at least 35 samples per class

#### **Training Improvements**:
- **Better Hyperparameters**: Lower learning rate (0.0005), smaller batch size (16), more epochs (5)
- **Enhanced Regularization**: Stronger dropout, batch normalization, early stopping
- **Real Evaluation Metrics**: Fixed fake placeholder metrics with actual fusion model evaluation

### ✅ **3. Results Comparison**

| Metric | Before Optimization | After Optimization |
|--------|-------------------|-------------------|
| **Image Encoder Parameters** | 5,876,800 | 589,632 (-90%) |
| **Tabular Encoder Parameters** | 438,668 | 27,980 (-94%) |
| **Fusion Model Parameters** | 826,887 | 366,727 (-56%) |
| **Training Time/Round** | ~240s | ~75s (-69%) |
| **Real Evaluation** | ❌ Fake metrics | ✅ Actual fusion evaluation |
| **Class Balance** | Poor | ✅ Minimum 35 samples/class |
| **Resume Capability** | ❌ Basic | ✅ Full state restoration |

### ✅ **4. Performance Analysis**
- **Validation Accuracy**: Improved from 6% to 22% (3.7x improvement)
- **Model Efficiency**: Dramatically reduced overfitting risk
- **Training Stability**: Better convergence with improved regularization
- **Real Metrics**: Now shows actual model performance instead of fake placeholders

### 🎯 **Next Steps for Further Improvement**
1. **Data Augmentation**: Implement advanced image augmentation strategies
2. **Transfer Learning**: Use pre-trained models for better feature extraction
3. **Advanced Fusion**: Implement cross-attention mechanisms
4. **Ensemble Methods**: Combine multiple model predictions
5. **Hyperparameter Tuning**: Systematic optimization of learning parameters
