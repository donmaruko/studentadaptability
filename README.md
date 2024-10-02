# Student Adaptability Level Prediction

## Description
This Python script implements a machine learning model to predict student adaptability levels in online education. It uses a Random Forest Classifier along with data preprocessing and model improvement methbods to achieve accurate predictions based on student characteristics and environmental factors.

## Dataset
The program uses the 'student_adaptability.csv' dataset, which contains information about students' adaptability to online learning, including features such as:

- Gender
- Age
- Education Level
- Institution Type
- IT Student status
- Location
- Load-shedding levels
- Financial Condition
- Internet Type
- Network Type
- Class Duration
- Self LMS usage
- Device type used for learning

The target variable is 'Adaptivity Level', which the model aims to predict.

## Architecture and Workflow

1. **Data Loading and Preprocessing**:
   - Load data from 'student_adaptability.csv'
   - Encode categorical variables using LabelEncoder
   - Handle imbalanced data using SMOTE (Synthetic Minority Over-sampling Technique)

2. **Data Splitting**:
   - Split data into training (80%) and testing (20%) sets

3. **Model Pipeline**:
   - StandardScaler for feature normalization
   - Random Forest Classifier for prediction

4. **Hyperparameter Tuning**:
   - Use RandomizedSearchCV to optimize model parameters
   - Parameters tuned include:
     - Number of estimators
     - Maximum tree depth
     - Minimum samples for split
     - Minimum samples per leaf

5. **Model Evaluation**:
   - Evaluate using accuracy score and classification report (precision, recall, F1-score)

6. **Prediction on New Data**:
   - Demonstrate how to use the model for predicting adaptability levels on new student data

## Performance Improvement Techniques

1. **SMOTE (Synthetic Minority Over-sampling Technique)**:
   - Addresses class imbalance by creating synthetic samples for minority classes

2. **Feature Scaling (StandardScaler)**:
   - Normalizes features to ensure equal importance and improve model performance

3. **Random Forest Classifier**:
   - Ensemble learning method that reduces overfitting and handles complex data relationships

4. **Hyperparameter Tuning (RandomizedSearchCV)**:
   - Optimizes model parameters to improve performance

5. **Cross-Validation**:
   - Uses 10-fold cross-validation to ensure robust model evaluation

## Requirements
- pandas
- scikit-learn
- imbalanced-learn (for SMOTE)

## Usage
1. Ensure you have the required libraries installed.
2. Place the 'student_adaptability.csv' file in the same directory as the script.
3. Run the script to train the model and see evaluation results.
4. To predict for new data, use the provided example at the end of the script and modify the input data as needed.

## Note
This script is designed for educational purposes and demonstrates various machine learning techniques. For production use, consider additional error handling, data validation, and possibly more extensive feature engineering.
