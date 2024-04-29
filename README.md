**Classification Model Evaluation**

This Python script evaluates the performance of classification models using bootstrapping and various evaluation metrics such as accuracy, kappa score, and neighborhood accuracy. The models evaluated include Logistic Regression (LR), Linear Discriminant Analysis (LDA), and Support Vector Machine (SVM). The files available are PIC.R for feature selection using NPRED package. The classification.py and classification1.py are used for texture classification and report generation. The Required data for the running of the codes could be found at : https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/ternikarr_iisc_ac_in/Em4GMvbekj1EjFJpsrsmds4Bs6c8bWmo7PlvQKIhjjWhGw?e=ahGcQR


**Table of Contents**
1. Overview
2. Requirements
3. Usage
4. File Structure
5. Output
6. Contributing

**Overview**
The script performs the following tasks:
Preprocesses the input data by scaling features using StandardScaler.
Splits the data into training and testing sets using stratified sampling.
Trains LR, LDA, and SVM models on the training data.
Evaluates model performance using accuracy, kappa score, and neighborhood accuracy metrics.
Compiles the evaluation results and writes them to an output file.

**Requirements**
Python 3.x
NumPy
pandas
scikit-learn
sys
os
glob
matplotlib
time
R version 4.3.x 
NPRED
csvread
readxl
doParallel

**Usage**
Clone the repository or download the script file.
Install the required dependencies using pip install in python and install.packages in R from the requirements section.
Prepare your input data and update the script with the correct file paths and configurations.
Run the script using python classification.py.

**File Structure**
classification.py: Main Python script containing the evaluation logic.
output.txt: Output file where evaluation results are written.
README.md: This README file providing an overview of the script.

**Output**
The output file output.txt contains the following information:
Mean and standard deviation of Overall accuracy, kappa score, neighborhood accuracy and added neighborhood accuracy for LR, LDA, and SVM models.
Mean and standard deviation of Confusion matrices from 100 iterations for each model.

**Contributing**
Contributions are welcome! Feel free to open an issue or submit a pull request for any improvements or bug fixes.
