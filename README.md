# Time Series Forecasting & Anomaly Detection Benchmark

A comprehensive benchmark suite for evaluating time series forecasting and anomaly detection models, with support for multiple deep learning and traditional machine learning approaches.

## Overview

This project provides a unified framework for:
- **Time Series Forecasting**: Comparing baseline models, deep learning architectures, and pre-trained models
- **Anomaly Detection**: Evaluating statistical, machine learning, and deep learning anomaly detection approaches
- **Data Preprocessing**: Handling decomposition, scaling, splitting, and windowing operations
- **Drift Analysis**: Detecting and analyzing concept drift in time series data
- **Comprehensive Evaluation**: Time-aware metrics and specialized evaluation functions

## Project Structure

```
ts-forecasting-anomaly-benchmark/
├── src/                          # Main source code
│   ├── evaluation/               # Evaluation metrics
│   │   ├── anomaly_metrics.py    # Anomaly detection metrics
│   │   ├── forecasting_metrics.py # Forecasting evaluation metrics
│   │   └── time_aware.py         # Time-aware evaluation functions
│   ├── models/                   # Model implementations
│   │   ├── forecasting/          # Forecasting models
│   │   │   ├── baselines.py      # Baseline forecasting models
│   │   │   ├── deep_learning.py  # Neural network forecasters
│   │   │   └── pretrained.py     # Pre-trained model wrappers
│   │   └── anomaly/              # Anomaly detection models
│   │       ├── statistical.py    # Statistical anomaly detection
│   │       ├── ml.py             # Machine learning approaches
│   │       └── deep_learning.py  # Deep learning anomaly detectors
│   ├── preprocessing/            # Data preprocessing utilities
│   │   ├── decomposition.py      # Time series decomposition
│   │   ├── scaling.py            # Normalization & scaling
│   │   ├── splitting.py          # Train/test splitting strategies
│   │   └── windowing.py          # Time series windowing
│   └── utils/                    # Utility functions
│       ├── config.py             # Configuration management
│       ├── plotting.py           # Visualization utilities
│       └── reproducibility.py    # Random seed management
├── notebooks/                    # Jupyter notebooks for analysis
│   ├── 01_exploration.ipynb      # Data exploration
│   ├── 02_preprocessing.ipynb    # Preprocessing workflows
│   ├── 03_forecasting_models.ipynb # Forecasting model evaluation
│   ├── 04_anomaly_detection.ipynb # Anomaly detection experiments
│   └── 05_drift_analysis.ipynb   # Drift detection analysis
├── data/                         # Data directory
│   ├── raw/                      # Original raw datasets
│   ├── processed/                # Preprocessed datasets
│   └── synthetic/                # Synthetic data for testing
├── experiments/                  # Experiment results and logs
├── tests/                        # Unit tests
├── requirements.txt              # Python dependencies
└── pyproject.toml               # Project configuration

```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ts-forecasting-anomaly-benchmark
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using conda:
```bash
conda env create -f environment.yml
conda activate ts-benchmark
```

## Quick Start

### Data Exploration
Start with the exploration notebook to understand your time series data:
```bash
jupyter notebook notebooks/01_exploration.ipynb
```

### Preprocessing
Prepare and preprocess your data:
```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```

### Forecasting
Evaluate forecasting models:
```bash
jupyter notebook notebooks/03_forecasting_models.ipynb
```

### Anomaly Detection
Run anomaly detection experiments:
```bash
jupyter notebook notebooks/04_anomaly_detection.ipynb
```

### Drift Analysis
Analyze concept drift in your time series:
```bash
jupyter notebook notebooks/05_drift_analysis.ipynb
```

## Features

### Forecasting Models
- **Baselines**: Moving Average, Exponential Smoothing, ARIMA
- **Deep Learning**: LSTM, GRU, Transformer, Temporal Convolutional Networks
- **Pre-trained**: Fine-tunable pre-trained architectures and transfer learning models

### Anomaly Detection Methods
- **Statistical**: Z-score, Isolation Forest, Local Outlier Factor
- **Machine Learning**: One-Class SVM, Autoencoders, Gaussian Mixture Models
- **Deep Learning**: Variational Autoencoders, LSTM-based detectors, Attention mechanisms

### Evaluation Metrics
- **Forecasting**: MAE, RMSE, MAPE, SMAPE, DTW, Time-aware metrics
- **Anomaly Detection**: Precision, Recall, F1-score, AUC-ROC, Point-adjust metrics
- **Specialized**: Time-aware evaluation for temporal dependencies

### Preprocessing Pipeline
- Time series decomposition (trend, seasonality, residuals)
- Multiple scaling strategies (MinMax, Standard, Robust)
- Flexible data splitting (train/val/test, cross-validation)
- Sliding window generation with customizable overlap

## Usage Examples

### Basic Forecasting
```python
from src.models.forecasting import LSTM
from src.preprocessing import create_windows

# Prepare data
X_train, y_train = create_windows(train_data, window_size=24)
X_test, y_test = create_windows(test_data, window_size=24)

# Train model
model = LSTM(input_dim=1, seq_length=24, pred_length=6)
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Forecast
predictions = model.predict(X_test)
```

### Anomaly Detection
```python
from src.models.anomaly import IsolationForest
from src.evaluation.anomaly_metrics import f1_score

# Initialize and train
detector = IsolationForest(contamination=0.05)
detector.fit(normal_data)

# Detect anomalies
anomaly_scores = detector.predict(test_data)
f1 = f1_score(ground_truth, anomaly_scores)
```

## Configuration

Configuration files are located in `src/utils/config.py`. Key settings include:
- Model hyperparameters
- Data paths
- Random seeds for reproducibility
- Evaluation metric thresholds

## Contributing

Contributions are welcome! Please:
1. Create a feature branch
2. Make your changes
3. Add tests if applicable
4. Submit a pull request

## License

[Add your license here]

## Citation

If you use this benchmark in your research, please cite:
```bibtex
@software{ts_benchmark_2026,
  title={Time Series Forecasting & Anomaly Detection Benchmark},
  author={[Author Name]},
  year={2026}
}
```

## References

- [List relevant papers on time series forecasting]
- [List relevant papers on anomaly detection]
- [Other relevant resources]

## Contact

For questions or issues, please open an issue on GitHub or contact the project maintainers.
