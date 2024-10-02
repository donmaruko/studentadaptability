# Student Adaptability Level Prediction

## Description
This Python script implements a machine learning model to predict student adaptability levels in online education. It uses a Random Forest Classifier along with various data preprocessing and model improvement techniques to achieve accurate predictions based on student characteristics and environmental factors.

The .py files contains a heap of notes in the bottom of the file explaining the processes.

## Dataset
The program uses the 'student_adaptability.csv' dataset, which contains information about students' adaptability to online learning. Features include:

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

## Requirements
- Python 3.7+
- pandas
- scikit-learn
- imbalanced-learn (for SMOTE)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/donmaruko/studentadaptability.git
   cd studentadaptability
   ```

2. Create a virtual environment (optional):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install pandas scikit-learn imbalanced-learn
   ```

## Usage

1. Ensure you have the 'student_adaptability.csv' file in the project directory.

2. Run the script:
   ```
   python studentadapt.py
   ```

3. The script will output the model's accuracy and a classification report.

4. To predict adaptability for new data, modify the `new_data` DataFrame in the script:

   ```python
   new_data = pd.DataFrame({
       'Gender': [label_encoders['Gender'].transform(['Girl'])],
       'Age': [label_encoders['Age'].transform(['21-25'])],
       'Education Level': [label_encoders['Education Level'].transform(['University'])],
       # ... (add other features)
   })
   prediction = best_model.predict(new_data)
   print("Predicted Adaptivity Level:", label_encoders['Adaptivity Level'].inverse_transform(prediction))
   ```

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

## Customization

To modify the hyperparameter search space, edit the `param_dist` dictionary in the script:

```python
param_dist = {
    'model__n_estimators': [100, 200, 300, 400, 500],
    'model__max_depth': [None, 10, 20, 30, 40, 50],
    'model__min_samples_split': [2, 5, 10, 15],
    'model__min_samples_leaf': [1, 2, 4, 6]
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
