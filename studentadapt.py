import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV # hyperparameter tuning
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline # for data preprocessing and modeling
from imblearn.over_sampling import SMOTE # handling imbalanced datasets

data = pd.read_csv('student_adaptability.csv')

# encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder() # converts categorical data into numerical data (Male = 1, Female = 2 for example)
    data[column] = label_encoders[column].fit_transform(data[column])
# from the data, we split features(x) and the target variable(y)
X = data.drop('Adaptivity Level', axis=1) # all features (independent variables) except adaptivity level
y = data['Adaptivity Level'] # target variable is adaptivity level, which is what we're predicting
# each categorical column is applied with the label encoder
# ML algorithms work with numbers. RFC for instance expect numbers for computations like decision tree splits
# algorithms don't know how to interpret string values
# encoding also ensures consistency across datasets, so it's easier to scale, compare, and transform data so it's compatible with ML pipelines



# handle imbalanced data with SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
'''
imbalanced data occurs when one category appears much more frequently than others.
90% of the data could be in one category, and 10% in another, this can lead to 
biased models that predict the majority class more often

thus, SMOTE is used to balance the classes by creating synthetic data points for the MINORITY
class in a balanced way. Unlike random oversampling, which duplicates data points, SMOTE
creates new samples by interpolating between existing minority class points, so the minority
class becomes more diverse and more represented

check the bottom of the code for a more detailed explanation
'''


# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 80% training and 20% testing

# create pipeline with scaling and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])
# the pipeline chains preprocessing and modeling steps together, like a pipeline
# StandardScaler normalizes the features by substracting the mean and scaling to unit variance
# RFC is the chosen model for classification, it builds multiple decision trees and combines their predictions
'''
pipeline ensures that data preprocessing and model fitting happen in the right order and prevents data leakage,
which is when information from the test set (or future data) is accidentally used to train the model, leading
to overly optimistic performance during training but poor generalization to unseen data. So the model "cheats"
by learning patterns that wouldn't be present in real-world data. Pipeline also simplifies the workflow and hyperparam tuning

StandardScaler standardizes features by centering them around a mean of 0 and a std. deviation of 1, ensuring
equal importance of features and improving model performanced, even if RFC itself isn't sensitive to scaling

RFC is chosen because it handles complex data well, reduces overfitting through averaging, is versatile for
different datatypes, and performs well on imbalanced datasets

see below for more info
'''



# hyperparameters for randomized search to be tuned
param_dist = {
    'model__n_estimators': [100, 200, 300, 400, 500],
    'model__max_depth': [None, 10, 20, 30, 40, 50],
    'model__min_samples_split': [2, 5, 10, 15],
    'model__min_samples_leaf': [1, 2, 4, 6]
}
# randomized search with cross-validation
random_search = RandomizedSearchCV(pipeline, param_dist, n_iter=100, cv=10, n_jobs=-1, verbose=2, random_state=42)
random_search.fit(X_train, y_train)
'''
RandomizedSearchCV performs random sampling of hyperparameter combinations.
i defined a parameter grid (param_dist) for RFC to tune:
- n_estimators: number of trees in the forest
- max_depth: maximum depth of each tree
- min_samples_split: minimum number of samples required to split an internal node
- min_samples_leaf: minimum number of samples required to be at a leaf node (end node)

it performs 100 iterations of 10-fold cross-validation to find the best parameters (cv=10)
n_jobs=-1 uses all available CPU cores for parallel processing, which boosts computation

see below for more info
'''

# evalute the best model
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
# precision, recall, and f-1 score
'''
after tuning, the best model is chosen with (best_estimator_) as variable best_model
see below for info about classification report stats and accuracy score
'''

# making predictions on new data (example)
new_data = pd.DataFrame({
    'Gender': [label_encoders['Gender'].transform(['Girl'])],
    'Age': [label_encoders['Age'].transform(['21-25'])],
    'Education Level': [label_encoders['Education Level'].transform(['University'])],
    'Institution Type': [label_encoders['Institution Type'].transform(['Non Government'])],
    'IT Student': [label_encoders['IT Student'].transform(['No'])],
    'Location': [label_encoders['Location'].transform(['Yes'])],
    'Load-shedding': [label_encoders['Load-shedding'].transform(['Low'])],
    'Financial Condition': [label_encoders['Financial Condition'].transform(['Mid'])],
    'Internet Type': [label_encoders['Internet Type'].transform(['Wifi'])],
    'Network Type': [label_encoders['Network Type'].transform(['4G'])],
    'Class Duration': [label_encoders['Class Duration'].transform(['3-6'])],
    'Self Lms': [label_encoders['Self Lms'].transform(['No'])],
    'Device': [label_encoders['Device'].transform(['Tab'])]
})
prediction = best_model.predict(new_data) # uses the best model to predict the adaptability level for the new datas
print("Predicted Adaptivity Level:", label_encoders['Adaptivity Level'].inverse_transform(prediction))
# inverse_transform converts the numerical data back to categorical data
'''
a new data point is created to test the model's prediction
the categorical features are encoded using the same LabelEncoder used during training
best_model.predict() is applied to the new data to get the predicted class
the prediction is converted back to the original categorical form using inverse_transform
'''



'''
SUMMARY:
- load data, encode categorical variables while handling imbalanced data
- split data into training and testing sets, create pipeline with scaling and model, and tune hyperparameters
- evaluate the model's performance using accuracy and classification report
- make predictions on new data using the best model
'''


'''
How does SMOTE work?
1. for each data point in the minority class, SMOTE randomly selects one of its nearest neighbors from the same class
2. SMOTE creates a synthetic point somewhere between the chosen data point and its 
nearest neighbor by taking a random point along the line connecting the two
3. repeat this process until the minority class is oversampled to the desired level

random_state=42 ensures reproducibility, it's a seed for the RNG
without the seed, the results of SMOTE would differ every time you run the code,
by fixing the seed value, you ensure that SMOTE generates the same synthetic data points every time,
so that the results are consistent for future testing and debugging

SMOTE is applied to both feature data (X), it generates synthetic feature data for the minority class based on existing data points
it also upsamples the target labels (Y) to ensure that the number of data points for each class in the target variable is balanced.
For example, if Low Adaptivity is the minority class, SMOTE will generate new synthetic feature data in X corresponding to new "Low Adaptivity" labels in Y,
this ensures the minority class is better represented during training, improving the model's ability to generalize across all classes
'''


'''
StandardScaler:
- ML models that rely on distance/gradient-based optimization like SVM, KNN, and neural networks perform better
when features are standardized, although RFC isn't strictiyl sensitive to scaling because it relies on decision trees,
it's still a good practice to scale features, especially when you're pipelining them alongside other models or if you
plan to experiment with different algorithms. So it overall boosts model performance

- equal importance to features. Features in different scales (age in years vs income in thousands) might distort
the model. Without scaling, features with larger values can dominate the learning process, scaling ensures
that all features contribute equally to the model

RFC:
- creates a forest of decision trees during training and outputs the class that is the MODE of the classifications from individual trees
- forests are good at capturing complex, non-linear relationships between features, since predicting Adaptivity Level likely depends on
many interacting factors like internet access and financial condition, RFC is well-suited for this kind of problem
- it's resilient to overfitting, RFs use multiple trees and average their predictions, which reduces the risk of overfitting to training data,
unlike a single decision tree that might overfit, the randomness in RFC helps generalize better to unseen data
- feature importance, RF naturally ranks the importance of each feature during training, 
this is useful for understanding which factors most influence adaptivity level
- RFC can handle both categorical and numerical features, which makes it versatile.
- while SMOTE helps balance the dataset, RF also has built-in mechanisms like class weighting,
which can further improve performance on imbalanced datasets
'''


'''
overfitting (RFC is resilient to this, but still possible): 
- model learns the training data TOO WELL, capturing noise and minor details that don't generalize well to new, unseen data,
thus the model becomes too complex, memorizing the data rather than learning underlying patterns.
- high acc on training data but poor performance on test data
- e.g. a decision tree that grows too deep, perfectly fitting the training set but failing on new data because it's too tailored to a specific dataset

underfitting:
- model is too simple and doesn't capture underlying patterns in the data well enough,
resulting in both the training and test performance being poor because the model fails to generalize and perform well even on the training set
- low acc on both training and test data
- e.g. a linear regression model trying to fit a complex non-linear dataset, unable to capture relationships in the data

the hyperparameter grid (param_dist):
- n_estimators: more trees can improve performance but increase computation time, a moderate number is 100/200,
if the model seems underfitting, try increasing the number, if you see diminishing returns with higher numbers, stop.
- max_depth: deeper trees capture more complex relationships but risk overfitting, while shallower trees may underfit,
if you're overfitting, try reducing the depth (set to 10,20,etc), if it's underfitting, let the trees grow deeper
or leave it as None to allow unrestricted growth
- min_samples_split: a lower value allows the model to make more splits and build complex trees, which cna lead to overfitting,
thus increase this value if the model overfits (from 2 to 5/10), and decrease if it underfits
- min_samples_leaf: a larger leaf size makes the model less complex and can prevent overfitting,
increase the value (from 1 to 2,4,etc) if the model is too complex and overfitting, decrease if its underfitting

how to know which parameter to tweak for better results:
- start with a wide range for each parameter, and let RandomizedSearchCV test different combinations
- evaluate the results, check the best parameters that gave the highest score, look at which params were selected
- refine specific params, if the best params still show under/overfitting, fine-tune individual params by either
narrowing down the grid (closer values) or testing more specific ranges
- focus on the most sensitive params first, like n_estimators and max_depth, then move to the others

cross-validation:
- technique to evaluate how well a model generalizes to unseen data, instead of using just one training/testing split,
it splits the data multiple times to create more robust and reliable evaluations
- it works by splitting the dataset into multiple "folds" or parts, for each fold, one part is
used for testing, and the remaining parts are used for training. This process is repeated for each fold
- model's performance is averaged over all the folds to give an overall estimate of its performance

10-fold cross-validation:
- the data is divided to 10 equal folds, the model is trained on 9 folds and tested on the 10th fold
- process is repeated 10 times, with each fold being used as the test set once
- finally, the average performance across all 10 iterations is reported
- it's used because it helps reduce the VARIANCE of model evaluation compared to a single train-test split
- it ensures that EVERY DATA POINT gets used in both training and testing, giving a more reliable assessment

RandomizedSearchCV:
- n_iter=100: randomly tries 100 combinations of hyperparams, adjust this to try more or fewer combinations
- cv=10: using 10-fold CV to evaluate each combination of hyperparams
- n_jobs=-1: uses all available CPU cores to run the search in parallel, so it's faster
- verbose=2: controls how much info is printed during the process, 2 will
print detailed progress information, which is useful for tracking the search
'''


'''
classification report metrics:
- precision:
    - when the model predicts positive, how often is it correct?
    - percentage of correctly predictive POSITIVE instances of all instances predicted as positive
    - TP / (TP + FP) | TP = true positive, FP = false positive
    - if model predicts that 100 students have high A.L, and 80 actually do, precision is 80%
- recall:
    - out of all actual positives, how many did the model identify correctly?
    - percentage of correctly predictive POSITIVE instances of all actual positive instances
    - TP / (TP + FN) | FN = false negative
    - if there are 100 students with high A.L, and the model correctly identifies 80, recall is 80%
- f1-score:
    - harmonic mean of precision and recall, balances the two metrics
    - 2 * (precision * recall) / (precision + recall)
    - considers both precision and recall, useful when you want to balance between them or when dealing with imbalanced data
    - if prec=0.75 and recall=0.85, f1-score is 0.8, a good balance between the two
- support:
    - number of actual occurrences of the class in the dataset
    - indicates how many samples actually belong to a particular class (how many students have high adaptivity)
    - if there are 150 students with high adaptivity, the support for this class is 150
- accuracy:
    - ratio of correct predictions (both positive and negative) to the total number of predictions
    - (TP + TN) / (TP + TN + FP + FN) | TN = true negative
    - tells the overall percentage of correctly classified instances out of the entire dataset

- precision focuses on minimizing false positives
- recall focuses on minimizing false negatives
- f-1 score provides a balance between precision and recall
- accuracy measures the overall correctness but can be MISLEADING in imbalanced datasets,
like if 90% of students have high adaptability, even a model that predicts everyone has high
adaptability would have 90% accuracy, but it's not useful, and could have poor precision and recall
'''