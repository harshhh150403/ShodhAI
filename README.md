# Loan Approval Optimization: Deep Learning vs Offline Reinforcement Learning

This project compares two approaches for loan approval decisions:
1. **Deep Learning (MLP)**: Predicts default probability
2. **Offline RL (CQL)**: Maximizes profit directly

**Key Result**: The RL agent generates **$41M more profit (+17%)** than the supervised model.

---

## Project Structure

```
Loan Acc/
├── data/
│   ├── accepted_2007_to_2018Q4.csv    # Raw data (~6GB)
│   └── processed/                     # Feature-engineered data
│       ├── X_train.parquet
│       ├── X_test.parquet
│       ├── y_train.parquet
│       └── y_test.parquet
├── models/
│   ├── mlp_model.pth                  # Trained DL model
│   └── rl_agent.pth                   # Trained RL agent
├── feature_engineering.ipynb          # Data preprocessing
├── model_training.ipynb               # Deep Learning model
├── rl_agent.ipynb                     # Offline RL agent
├── Report/                            # Analysis & comparison
│   ├── final_report.md             
│   └── final_report(pdf).pdf          
├── requirements.txt                   # Dependencies
└── README.md                          # This file
```

---

## Setup

### 1. Clone/Download the Repository

```bash
cd "ShodhAI"
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: For GPU support, install PyTorch with CUDA:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> ```

### 4. Download the Dataset

Download `accepted_2007_to_2018Q4.csv` from Kaggle's Lending Club dataset and place it in `data/`.

---

## Running the Code

Execute the notebooks in order:

### Step 1: Feature Engineering
```bash
jupyter notebook feature_engineering.ipynb
```
- Loads raw data, handles missing values, removes leakage features
- Outputs: `data/processed/X_train.parquet`, etc.

### Step 2: Train Deep Learning Model
```bash
jupyter notebook model_training.ipynb
```
- Trains MLP classifier (~20 epochs)
- Outputs: `models/mlp_model.pth`
- **Metrics**: AUC-ROC = 0.9542, F1 = 0.78

### Step 3: Train RL Agent
```bash
jupyter notebook rl_agent.ipynb
```
- Trains CQL agent (~50 epochs with early stopping)
- Outputs: `models/rl_agent.pth`
- **Metric**: Total Profit = $278.8M

---

## Reproducing Results

After training both models, you can verify the profit comparison:

| Model                  | Total Profit (Test Set) |
| ---------------------- | ----------------------- |
| Deep Learning (MLP)    | $237,872,403            |
| Offline RL (CQL)       | $278,775,601            |
| **RL Advantage**       | **+$40,903,197 (+17%)** |

See `final_report.md` for detailed analysis.

---

## Requirements

- Python 3.8+
- NVIDIA GPU (recommended for faster training)
- 16GB+ RAM (dataset is large)

---




