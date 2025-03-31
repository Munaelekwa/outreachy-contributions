# The Blood Brain Barrier Permeability Prediction Project (BBBP)
This Project aims to predict the ability of chemical compounds to cross the blood brain barrier based on their structural components using machine learning techniques.

## Table of Contents  
- [Project Overview](#project-overview)  
- [Dataset Information](#dataset-information)  
- [Setup Instructions](#setup-instructions)  
  - [Prerequisites](#prerequisites)  
  - [Download and Installation](#download-and-installation)  
- [Featurisation](#featurisation)  
- [Model Building](#model-building)  
- [Model Evaluation](#model-evaluation)  
- [Results and Analysis](#results-and-analysis)  
- [References](#references)

---

## Project Overview

The Blood brain barrier is a selective, semi-permeable membrane that protects the brain and the central nervous system (CNS) from harmful materials in the blood stream. The blood brain barrier permeability of molecules is very important in drug discovery and research, especially in the development of drugs that act on the CNS and also in identifying neurotoxic drugs.
This project is a binary classification machine learning model that will accept the smiles notation of a compound as input and predict the bbb permeability based on the chemical and molecular properties represented in the model. Permeable drugs as predicted are labelled 1 and impermeaple drugs are labelled 0. 

## Dataset Information  

This project uses the **Blood-Brain Barrier Permeability (BBBP) dataset** from [Therapeutics Data Commons (TDC)](https://tdcommons.ai/).  

- **Dataset Name**: BBBP (Martins et al., 2012)  
- **Source**: [TDC BBBP Dataset](https://tdcommons.ai/single_pred_tasks/adme/#bbbp)  
- **Original Research Paper**:  
  Martins, I. F., et al. (2012). A Bayesian approach to in silico blood–brain barrier penetration modeling. *Journal of Chemical Information and Modeling, 52*(6), 1686-1697. [DOI:10.1021/ci300124c](https://doi.org/10.1021/ci300124c)
- **Characteristics**:

| Field         | Description                          |
|--------------|----------------------------------|
| **Drug_ID**   | Compound name/Identifier        |
| **Drug**      | SMILES Notation of compound as a string|
| **Y**         | Binary classification, 0: Not Permeable, 1: Permeable       | 

- **Statistics**:

| Split      | Class 0 | Class 1  | Total | Class 0 (%) | Class 1 (%) |
|-----------|---------------|---------------|-------|------------|------------|
| Train     | 325          | 1096           | 1421  | 22.87%     | 77.13%     |
| Validation| 51           | 152           | 203   | 25.12%     | 74.88%     |
| Test      | 103            | 303           | 406   | 25.37%     | 74.63%     |
| Total     | 479           | 1551          | 2030  | 23.60%     | 76.40%     |


This dataset consists of **2030 molecules**, labeled as **BBB+ (permeable) or BBB- (non-permeable)** based on experimental permeability measurements.

## Setup Instructions

### Prerequisities

To successfully run this project, the following should be installed:
- Python 3.9 or later
- Ubuntu OS or WSL (If sysyem os is windows)
- Miniconda/Anaconda
- Docker

### Download and Installation: 

To recreate this project, follow these steps:
- Create a fork of this repository
- Open the project repository in your local machine by cloning it:
  ```bash
      git clone https://github.com/your-username/your-repo.git
      cd your-repo
- Set up the conda environment
  This project uses a conda environment to manage dependencies, create and activate all required dependencies by running:
  ```bash
      conda env create --file environment.yml
      conda activate bbbp
- Download the dataset (Already provided in the data folder):
  The dataset is already split into train, test and valid data and saved as csv files in the data folder. it can be manually downloaded by running the `download_data.py` script
  ```bash
      python scripts/download_data.py
      #this downloads the dataset from Therapeutics Data Commons and saves it in the data folder.

## Featurisation  

The downloaded dataset was featurised, i.e., converted from raw molecular data (SMILES notation) into a numerical representation that can be interpreted by machine learning models.  

The featuriser used is the **RDKit Descriptor Model** from [Ersilia Model Hub](https://www.ersilia.io/model-hub).  

### Steps to Featurise Data:
- Fetch and serve the Ersilia model to be used:  
   ```bash
       ersilia fetch eos8a4x
       ersilia serve eos8a4x
- Run the featurisation script and save featurised data
  ```bash
      python scripts/featurisation.py

This will featurise the split datasets and save the featurised data in the data folder.
 
## Model Building

This project uses the **Random Forest Classifier**  from the **Scikit-Learn** Framework as a model to train the featurised data. Featurised data was cleaned before training the model to ensure that the model interpretes data input correctly by:
- Dropping non-numerical columns, as model cannot interpret text based inputs
- Handling missing values and filling them with median values
- Putting a cap on too large values (all values were clipped within -1e6 - 1e6 range)

The model is trained and saved as a pk1 file in the models folder.
To run the model training script or train the model with a different data, replace the dataset in the `train_model.py` script with the featurised data and run the command:
```
python scripts/train_model.py

#This script trains a random forest classifier model with available data, prints the classification report and saves the model as a pk1 file in the models folder.
```
Also, the trained model can be used in a different script to test new data using the joblib library. to do this, load the model in the new script using the code:
```python
import Joblib
model = joblib.load("models/bbbp_model.pkl")
```

## Model Evaluation

The model is evaluated using metrics like AUROC score (since it is a classification problem), precision, Recall, accuracy score. Here's a tabular representation of the metrics used and the purpose:

| **Metric**              | **Purpose** |
|-------------------------|------------|
| **Accuracy**           | Measures the overall correctness of the model by calculating the percentage of correctly classified instances. |
| **Precision**          | Evaluates how many of the predicted positive cases were actually positive, helping to minimize false positives. |
| **Recall (Sensitivity)** | Measures how well the model identifies actual positive cases, focusing on minimizing false negatives. |
| **F1-Score**           | A balance between precision and recall, useful when classes are imbalanced. |
| **AUROC (Area Under ROC Curve)** | Measures the model’s ability to distinguish between classes across different classification thresholds. A higher AUROC means better separability between classes. |
| **Confusion Matrix**   | Provides a breakdown of true positives, true negatives, false positives, and false negatives, helping to analyze misclassifications. |
| **Precision-Recall Curve** | Shows the trade-off between precision and recall at different thresholds, useful for imbalanced datasets. |
| **Feature Importance** | Identifies which molecular properties contribute most to the permeability prediction. |

The results, graphs and all visuals used in the analysis of this model are well represented and documented in the `evaluation.ipynb` file located in the notebooks folder. To view the analysis file, run the command below in the terminal and navigate to the port on your local machine to run the notebook.
```
Jupyter notebook
```

## Results and Analysis
