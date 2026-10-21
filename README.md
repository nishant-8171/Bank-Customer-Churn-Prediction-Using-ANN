# Bank Customer Churn Prediction Using ANN

## Project Overview

This is a machine learning and deep learning practice project that uses Artificial Neural Networks (ANN) to predict whether a bank customer is likely to leave the bank.

The project demonstrates data preprocessing, feature encoding, feature scaling, neural network training, model evaluation, and performance visualization using TensorFlow and Keras.

**Project Type:** Practice Project  
**Domain:** Banking and Customer Analytics  
**Task:** Binary Classification  
**Target Variable:** `Exited`

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Keras
- Matplotlib
- Jupyter Notebook

## Dataset

The notebook uses `Churn_Modelling.csv`, which contains bank customer information.

The target column `Exited` indicates whether a customer left the bank.

The notebook removes the following columns before training:

- `RowNumber`
- `CustomerId`
- `Surname`

## Project Workflow

1. Load and explore the dataset.
2. Check duplicate records and category distributions.
3. Remove identifier columns.
4. Separate features and the target variable.
5. Split data into training and testing sets.
6. Encode categorical features using one-hot encoding.
7. Standardize features using `StandardScaler`.
8. Build and train a baseline ANN model.
9. Evaluate predictions using accuracy.
10. Build an enhanced ANN using Dropout and Early Stopping.
11. Compare training and validation performance using plots.

## Model 1: Baseline ANN

The baseline neural network contains:

| Layer | Configuration |
|---|---|
| Input / Dense | 3 neurons, ReLU |
| Hidden Dense | 3 neurons, ReLU |
| Output | 1 neuron, Sigmoid |

**Training configuration:**

- Optimizer: Adam
- Loss function: Binary Cross-Entropy
- Batch size: 50
- Epochs: 50
- Validation split: 20%

## Model 2: Enhanced ANN

The enhanced model uses additional hidden layers and Dropout regularization.

| Layer | Configuration |
|---|---|
| Dense | 64 neurons, ReLU |
| Dropout | 20% |
| Dense | 32 neurons, ReLU |
| Dropout | 20% |
| Dense | 16 neurons, ReLU |
| Output Dense | 1 neuron, Sigmoid |

**Training configuration:**

- Optimizer: Adam
- Loss function: Binary Cross-Entropy
- Batch size: 32
- Maximum epochs: 100
- Validation split: 20%
- Early Stopping patience: 10 epochs
- Best weights restored when training stops

## Model Evaluation

The notebook uses `accuracy_score` to evaluate predictions.

It also generates plots for:

- Baseline training and validation loss
- Enhanced model training and validation loss
- Enhanced model training and validation accuracy

The enhanced model's test accuracy is printed when the notebook runs.

## How to Run the Project

### Install dependencies

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib jupyter
```

### Prepare the dataset

Place `Churn_Modelling.csv` in the appropriate location. The notebook currently uses:

```python
df = pd.read_csv("/content/Churn_Modelling.csv")
```

This path is suitable for Google Colab. For local execution, update it to the actual CSV file path.

### Run the notebook

```bash
jupyter notebook
```

Open `ANNPRACTICE_FOR_Bainory (1).ipynb` and execute the cells in sequence.

## Key Learning Outcomes

- Exploratory data analysis
- Categorical feature encoding
- Feature scaling and preprocessing
- Artificial Neural Networks for binary classification
- ReLU and Sigmoid activation functions
- Binary Cross-Entropy loss
- Dropout regularization
- Early Stopping
- Accuracy evaluation
- Training and validation visualization

## Future Improvements

- Add a confusion matrix and classification report.
- Evaluate precision, recall, F1-score, and ROC-AUC.
- Compare baseline and enhanced model performance.
- Handle class imbalance if required.
- Save the trained model and fitted scaler.
- Improve reproducibility using a fixed random state.

## Project Structure

```text
.
├── ANNPRACTICE_FOR_Bainory (1).ipynb
├── Churn_Modelling.csv
└── README.md
```

## Disclaimer

This is an educational practice project intended to develop foundational skills in deep learning and customer churn prediction. It is not a production-ready banking application.
