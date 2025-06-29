 # 🚗 Car Insurance Claim Prediction using MLP

This deep learning project uses a Multi-Layer Perceptron (MLP) to **predict whether a customer will file a car insurance claim** based on their demographic, financial, and driving history data.

---

## 📊 Problem Statement

- **Objective**: Predict the likelihood of an insurance claim (`CLAIM_FLAG`) using customer features.
- **Type**: Binary classification
- **Target**: `CLAIM_FLAG` (0 = No claim, 1 = Claim)

---

## 🧠 Model: Multi-Layer Perceptron (MLP)

We used TensorFlow and Keras to implement an MLP with:

- Multiple hidden layers
- ReLU activations
- Sigmoid output for binary classification

---

## 🔄 Project Pipeline

| Step | Description |
|------|-------------|
| 1. ✅ Data Load         | Load raw CSV data using Pandas |
| 2. 📊 EDA              | Checked for nulls, class imbalance, feature types |
| 3. 🔧 Preprocessing     | Cleaned dollar signs, handled missing values, encoded categorical features |
| 4. 📏 Scaling           | Applied `StandardScaler` to normalize features |
| 5. 🔄 Train-Test Split | Used 80/20 train-test split |
| 6. 🧪 Baseline Model    | Single-layer Perceptron for benchmarking |
| 7. 🏗️ MLP Architecture | Tuned hidden layers, dropout, units, and regularization |
| 8. ⚙️ Optimizer         | Tuned between `adam` and `rmsprop` |
| 9. 📉 Early Stopping    | Prevented overfitting based on validation loss |
| 10. 🔍 Evaluation       | Used accuracy, confusion matrix, ROC AUC |
| 11. 📈 Visualization    | Plotted training vs validation loss curves |
| 12. 💾 Model Saving     | Saved best model and scaler for future predictions |

---

## 📦 Key Features Implemented

### ✅ Weight Initialization Techniques
- **He Initialization** (`he_normal`) for ReLU activation
- Improves convergence and combats vanishing/exploding gradients

### ✅ Vanishing Gradient Handling
- ReLU activation in hidden layers instead of sigmoid/tanh
- Ensures stable gradients during backpropagation



### ✅ Regularization
- **L2 Regularization** using `kernel_regularizer`
- **Dropout** after each hidden layer (0.2 to 0.5)



### ✅ Early Stopping

- Stops training when `val_loss` doesn’t improve for 5 epochs
- Prevents overfitting and saves best weights

### ✅ Hyperparameter Tuning with Keras Tuner
- `RandomSearch` with:
  - Number of layers (1–3)
  - Units per layer (64–256)
  - Optimizer (`adam`, `rmsprop`)
  - Dropout rate (0.2 to 0.5)

---

## 🧪 Evaluation Metrics

- **Accuracy**
- **Confusion Matrix**
- **Classification Report**
- **ROC-AUC Score**
- **Loss Curves**

---

## 🧾 Sample Prediction Workflow

```bash

# Step 1: Save Model and Scaler
model.save("car_insurance_mlp_best_model.h5")
joblib.dump(scaler, "scaler.pkl")

# Step 2: Load for Inference
from tensorflow.keras.models import load_model
import joblib
model = load_model("car_insurance_mlp_best_model.h5")
scaler = joblib.load("scaler.pkl")

# Step 3: Preprocess new input
# Create new sample with correct structure → scale → predict
```

### 📁 Files Included

| File                              | Description                          |
| --------------------------------- | ------------------------------------ |
| `car_insurance_claim.csv`         | Input dataset                        |
| `model_training.ipynb`            | Training notebook with full pipeline |
| `car_insurance_mlp_best_model.h5` | Saved trained model                  |
| `scaler.pkl`                      | Saved StandardScaler                 |
| `README.md`                       | Project documentation                |


