# -*- coding: utf-8 -*-

import pandas as pd

# Import the necessary libraries
# NOTE: Import linear instead of logistic
# conda update scikit-learn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Read the CSV file into a pandas DataFrame
dataset = pd.read_csv("C:/Users/cakus/Downloads/standardized-student-mat.csv")

# Define the feature columns and target column
#### NOTE: Since the predicted variable is G1.x (portuguese grade for period 1, I've excluded variables that correspond to math/end in .y) 
feature_columns = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet", "guardian.x", "traveltime.x", "studytime.x", "failures.x", "schoolsup.x", "famsup.x", "paid.x", "activities.x", "higher.x", "romantic.x", "famrel.x", "freetime.x", "goout.x", "Dalc.x", "Walc.x", "health.x", "absences.x"]
target_column = "G1.x"

# Split the data into features (X) and target variable (y)
X = dataset[feature_columns]
y = dataset[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the linear regression model
# NOTE: since the dependent variable (portuguese grade for period 1) is continuious, I'm using linear regression instead of logistic)
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model: Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# NOTE: no need to obtain the probability of the predictions because
# in linear regression, there is no need to use metrics like accuracy, 
# F1 score, or AUC-ROC. Instead, evaluate the model's performance 
# using metrics specific to regression, such as MSE.

# Scale the feature columns
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Making gender dummy
dataset['SexM'] = (dataset['sex'] == '-0.962736').astype(int)
dataset['SexF'] = (dataset['sex'] == '1.0359874').astype(int)
print(dataset.head())

# Step 5: Calculate fairness metric - Demographic Parity
# Identify indices for the two groups based on 'sex' attribute
# Step 5: Calculate fairness metric - Demographic Parity
# Identify indices for the two groups based on 'sex' attribute
female_indices = np.where(X_test['SexM'] == 1)[0]
male_indices = np.where(X_test['SexF'] == 1)[0]

# Perform error checking for index validity
if max(female_indices) >= len(y_pred) or max(male_indices) >= len(y_pred):
    print("Error: Invalid index found in 'female_indices' or 'male_indices'.")
    print("Length of y_pred:", len(y_pred))
else:
    # Filter the predictions and true labels for the two groups
    female_predictions = y_pred[female_indices]
    male_predictions = y_pred[male_indices]

    # Calculate demographic parity
    dp = abs(sum(female_predictions) / len(female_predictions) - sum(male_predictions) / len(male_predictions))

    # Step 6: Print the results
    print("Demographic Parity (DP): {:.4f}".format(dp))


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Read the CSV file into a pandas DataFrame
dataset = pd.read_csv("C:/Users/cakus/Downloads/original-student-mat.csv")

# Define the feature columns and target column
#### NOTE: Since the predicted variable is G1.x (portuguese grade for period 1, I've excluded variables that correspond to math/end in .y) 
feature_columns = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet", "guardian.x", "traveltime.x", "studytime.x", "failures.x", "schoolsup.x", "famsup.x", "G1.x", "activities.x", "higher.x", "romantic.x", "famrel.x", "freetime.x", "goout.x", "Dalc.x", "Walc.x", "health.x", "absences.x"]
target_column = "paid.x"


# Split the data into features (X) and target variable (y)
X = dataset[feature_columns]
y = dataset[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Obtain the probability of the predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)

# Print the evaluation metrics
print("Accuracy: {:.4f}, F1-score: {:.4f}, ROC-AUC score: {:.4f}".format(accuracy, f1, auc_roc))

#########################################
#########################################
#########################################

# Scale the feature columns
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the logistic regression model
model = LogisticRegression(max_iter=1000)

# Train the model on the scaled training data
model.fit(X_train_scaled, y_train)

# Make predictions on the scaled test data
y_pred1 = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred1)
f1 = f1_score(y_test, y_pred1)
auc_roc = roc_auc_score(y_test, y_pred1)

# Print the evaluation metrics
print("Accuracy: {:.4f}, F1-score: {:.4f}, ROC-AUC score: {:.4f}".format(accuracy, f1, auc_roc))

#########################################
#########################################
#########################################
# Evaluation Metric: gender
male_indices = dataset[dataset['sex'] == 1].index
valid_male_indices = male_indices[male_indices < len(y_pred)]  # Filter out invalid indices

y_pred_male = y_pred[valid_male_indices]
y_true_male = dataset.loc[valid_male_indices, 'paid.x']

female_indices = dataset[dataset['sex'] == 0].index
valid_female_indices = female_indices[female_indices < len(y_pred)]  # Filter out invalid indices

y_pred_female = y_pred[valid_female_indices]
y_true_female = dataset.loc[valid_female_indices, 'paid.x']

accuracy_male = accuracy_score(y_true_male, y_pred_male)
precision_male = precision_score(y_true_male, y_pred_male)
recall_male = recall_score(y_true_male, y_pred_male)
f1_score_male = f1_score(y_true_male, y_pred_male)
print(accuracy_male)
print(precision_male)
print(recall_male)
print(recall_male)


accuracy_female = accuracy_score(y_true_female, y_pred_female)
precision_female = precision_score(y_true_female, y_pred_female)
recall_female = recall_score(y_true_female, y_pred_female)
f1_score_female = f1_score(y_true_female, y_pred_female)
print(accuracy_female)
print(precision_female)
print(recall_female)
print(f1_score_female)


dp = abs(sum(y_pred_female) / len(y_pred_female) - sum(y_pred_male) / len(y_pred_male))


#########################################
#########################################
#########################################

# Step 5: Calculate fairness metric - Demographic Parity
# Identify indices for the two groups based on 'sex' attribute
# Identify indices for the two groups based on 'sex' attribute
female_indices = np.where(dataset['sex'] == 0)[0]
male_indices = np.where(dataset['sex'] == 1)[0]

# Perform error checking for index validity
if max(female_indices) >= len(y_pred) or max(male_indices) >= len(y_pred):
    print("Error: Invalid index found in 'female_indices' or 'male_indices'.")
    print("Length of y_pred:", len(y_pred))
else:
    # Filter the predictions and true labels for the two groups
    female_predictions = y_pred[female_indices]
    male_predictions = y_pred[male_indices]

    # Calculate demographic parity
    dp = abs(sum(female_predictions) / len(female_predictions) - sum(male_predictions) / len(male_predictions))

    # Step 6: Print the results
    print("Accuracy: {:.4f}".format(accuracy))
    print("F1-score: {:.4f}".format(f1))
    print("ROC-AUC score: {:.4f}".format(auc_roc))
    print("Demographic Parity (DP): {:.4f}".format(dp))
    
    
###############
###############
# Random Forest (binary)
###############
###############

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Load the dataset
dataset = pd.read_csv('C:/Users/cakus/Downloads/original-student-mat.csv')  # Replace 'dataset.csv' with the actual filename

# Split the dataset into features (X) and target variable (y)
X = dataset.drop('paid.x', axis=1)  # Replace 'paid.x' with the actual column name
y = dataset['paid.x']  # Replace 'paid.x' with the actual column name

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Adjust the test_size as needed

# Create a random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  # Adjust the number of estimators as needed

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the utility of the classifier
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("F1-Score:", f1)
print("AUC-ROC:", auc_roc)


###############
###############
# Random Forest Regression (continuous)
###############
###############


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
dataset = pd.read_csv('C:/Users/cakus/Downloads/original-student-mat.csv')  # Replace 'dataset.csv' with the actual filename

# Split the dataset into features (X) and target variable (y)
X = dataset.drop('G1.x', axis=1)  # Exclude the target variable column
y = dataset['G1.x']  # Select the target variable column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Adjust the test_size as needed

# Create a random forest regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)  # Adjust the number of estimators as needed

# Train the regressor
rf_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_regressor.predict(X_test)

# Evaluate the regression performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)







######
######
###### DECISION TREE

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Step 2: Load the dataset
# Replace 'your_dataset.csv' with the actual filename or path of your dataset
data = pd.read_csv('C:/Users/cakus/Downloads/original-student-mat.csv')  # Replace 'dataset.csv' with the actual filename

# Step 3: Preprocess the data (if required)
# If your dataset requires preprocessing (e.g., handling missing values, encoding categorical variables),
# you can perform those steps here.

# Step 4: Split the dataset into training and testing sets
X = data.drop('paid.x', axis=1)  # Features
y = data['paid.x']              # Target variable

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build the decision tree model
# You can tune hyperparameters of DecisionTreeClassifier like max_depth, min_samples_split, etc.
# For simplicity, we are using the default hyperparameters here.
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 7: Calculate accuracy, F1 score, and ROC-AUC score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print("Accuracy Score:", accuracy)
print("F1 Score:", f1)
print("ROC-AUC Score:", roc_auc)

######
###### MULTILAYER PERCEPTON

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Load the data
data = pd.read_csv('C:/Users/cakus/Downloads/original-student-mat.csv')

# Assuming 'paid.x' is the target variable and the rest are features
X = data.drop(columns=['paid.x'])
y = data['paid.x']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

# Predict on the test set
y_pred = mlp.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Calculate the F1 score
f1 = f1_score(y_test, y_pred)

# Calculate the ROC-AUC score
roc_auc = roc_auc_score(y_test, mlp.predict_proba(X_test)[:, 1])

print("Accuracy Score:", accuracy)
print("F1 Score:", f1)
print("ROC-AUC Score:", roc_auc)










import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
dataset = pd.read_csv('C:/Users/cakus/Downloads/original-student-mat.csv')

# Preprocess the data
X = dataset.drop(columns=['paid.x', 'sex'])
y = dataset['paid.x']

# Encode 'sex' attribute to binary values (0 for male and 1 for female)
X['sex'] = (X['sex'] == 2).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the random forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Calculate accuracy on the test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate Demographic Parity for each group
# Group 1: Male (sex = 0)
# Group 2: Female (sex = 1)
group_1_indices = X_test[X_test['sex'] == 0].index
group_2_indices = X_test[X_test['sex'] == 1].index

# Proportion of positive outcomes for each group
group_1_positive_proportion = y_test.loc[group_1_indices].mean()
group_2_positive_proportion = y_test.loc[group_2_indices].mean()

# Calculate Demographic Parity
demographic_parity = abs(group_1_positive_proportion - group_2_positive_proportion)

print("Accuracy:", accuracy)
print("Demographic Parity:", demographic_parity)





import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dataset
dataset = pd.read_csv('C:/Users/cakus/Downloads/original-student-mat.csv')

# Preprocess the data
X = dataset.drop(columns=['paid.x'])
y = dataset['paid.x']

X['sex'] = X['sex'].replace({1: 0, 2: 1}).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the random forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Calculate accuracy on the test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate Demographic Parity for each group
# Group 1: Male (sex = 0)
# Group 2: Female (sex = 1)
group_1_indices = X_test[X_test['sex'] == 0].index
group_2_indices = X_test[X_test['sex'] == 1].index

# Proportion of positive outcomes for each group
group_1_positive_proportion = y_test.loc[group_1_indices].mean()
group_2_positive_proportion = y_test.loc[group_2_indices].mean()

# Calculate Demographic Parity
demographic_parity = abs(group_1_positive_proportion - group_2_positive_proportion)

print("Accuracy:", accuracy)
print("Demographic Parity:", demographic_parity)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
dataset = pd.read_csv('C:/Users/cakus/Downloads/original-student-mat.csv')

# Preprocess the data
X = dataset.drop(columns=['paid.x'])
y = dataset['paid.x']

X['sex'] = X['sex'].replace({1: 0, 2: 1}).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
logreg_model = LogisticRegression(random_state=42)
logreg_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = logreg_model.predict(X_test)

# Calculate accuracy on the test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate Demographic Parity for each group
# Group 1: Male (sex = 0)
# Group 2: Female (sex = 1)
group_1_indices = X_test[X_test['sex'] == 0].index
group_2_indices = X_test[X_test['sex'] == 1].index

# Proportion of positive outcomes for each group
group_1_positive_proportion = y_test.loc[group_1_indices].mean()
group_2_positive_proportion = y_test.loc[group_2_indices].mean()
demographic_parity = abs(group_1_positive_proportion - group_2_positive_proportion)

# Display results
print("Accuracy: {:.2f}".format(accuracy))
print("Group 1 (Male) Positive Proportion: {:.2f}".format(group_1_positive_proportion))
print("Group 2 (Female) Positive Proportion: {:.2f}".format(group_2_positive_proportion))
print("Demographic Parity:", demographic_parity)


# MULTILAYER PERCEPTRON PARITY

from sklearn.neural_network import MLPClassifier

mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)

# Fit the classifier to the training data
mlp_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_mlp = mlp_classifier.predict(X_test)

y_pred_mlp = mlp_classifier.predict(X_test)

# Identify indices for the privileged and unprivileged groups based on the sensitive attribute
privileged_indices_mlp = np.where(X_test['sex'] == 0)[0]
unprivileged_indices_mlp = np.where(X_test['sex'] == 1)[0]

# Filter the predictions for the privileged and unprivileged groups
predictions_privileged_mlp = y_pred_mlp[privileged_indices_mlp]
predictions_unprivileged_mlp = y_pred_mlp[unprivileged_indices_mlp]

# Calculate demographic parity for Multilayer Perceptron
dp_mlp = abs(sum(predictions_privileged_mlp) / len(predictions_privileged_mlp) - sum(predictions_unprivileged_mlp) / len(predictions_unprivileged_mlp))

# Print the demographic parity for Multilayer Perceptron
print("Multilayer Perceptron - Demographic Parity:", dp_mlp)



# DECISION TREE

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

# Create a Decision Tree classifier object
dt_classifier = DecisionTreeClassifier()

# Fit the classifier to the training data
dt_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_dt = dt_classifier.predict(X_test)

# Calculate the predicted probabilities for positive class
y_pred_proba_dt = dt_classifier.predict_proba(X_test)[:, 1]

# Step 5: Calculate fairness metric - Demographic Parity
# Identify indices for the privileged and unprivileged groups based on the sensitive attribute
privileged_indices_rf = np.where(X_test['sex'] == 1)[0]
unprivileged_indices_rf = np.where(X_test['sex'] == 1)[0]

# Identify indices for the privileged and unprivileged groups based on the sensitive attribute
privileged_indices_dt = np.where(X_test['sex'] == 1)[0]
unprivileged_indices_dt = np.where(X_test['sex'] == 1)[0]

# Filter the predictions for the privileged and unprivileged groups
predictions_privileged_dt = y_pred_dt[privileged_indices_dt]
predictions_unprivileged_dt = y_pred_dt[unprivileged_indices_dt]

# Calculate demographic parity for Decision Tree
dp_dt = abs(sum(predictions_privileged_dt) / len(predictions_privileged_dt) - sum(predictions_unprivileged_dt) / len(predictions_unprivileged_dt))
print("Decision Tree - Demographic Parity:", dp_dt)


# LOGISTIC
from sklearn.preprocessing import StandardScaler

X = dataset.drop(columns=['paid.x'])
y = dataset['paid.x']

X['sex'] = X['sex'].replace({1: 0, 2: 1}).astype(int)

# Scale the feature columns
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the logistic regression model
model = LogisticRegression(max_iter=1000)

# Train the model on the scaled training data
model.fit(X_train_scaled, y_train)

# Make predictions on the scaled test data
y_pred1 = model.predict(X_test_scaled)

# Step 5: Calculate fairness metric - Demographic Parity
female_indices = np.where(X_test['sex'] == 1)[0]
male_indices = np.where(X_test['sex'] == 1)[0]

# Perform error checking for index validity
if max(female_indices) >= len(y_pred) or max(male_indices) >= len(y_pred):
    print("Error: Invalid index found in 'female_indices' or 'male_indices'.")
    print("Length of y_pred:", len(y_pred))
else:
    # Filter the predictions and true labels for the two groups
    female_predictions = y_pred[female_indices]
    male_predictions = y_pred[male_indices]

    # Calculate demographic parity
    dp = abs(sum(female_predictions) / len(female_predictions) - sum(male_predictions) / len(male_predictions))

    # Step 6: Print the results
    print("Accuracy: {:.4f}".format(accuracy))
    print("Demographic Parity (DP): {:.4f}".format(dp))



# EQUALIZED ODDS - LOG REG
X['sex'] = X['sex'].replace({1: 0, 2: 1}).astype(int)
sensitive_attributes = ["sex"]

for sensitive_attribute in sensitive_attributes:
    # Identify indices for the privileged and unprivileged groups based on the sensitive attribute
    privileged_indices = np.where(X_test[sensitive_attribute] == 1)[0]
    unprivileged_indices = np.where(X_test[sensitive_attribute] == 0)[0]

    # Filter the predictions and true labels for the privileged and unprivileged groups
    predictions_privileged = y_pred[privileged_indices]
    predictions_unprivileged = y_pred[unprivileged_indices]

    if len(predictions_privileged) != 0 and len(predictions_unprivileged) != 0:
        # Calculate the Equalized Odds metric
        eq_odds = abs(sum(predictions_privileged) / len(predictions_privileged) - sum(predictions_unprivileged) / len(predictions_unprivileged))
        print("Equalized Odds for '{}': {:.4f}".format(sensitive_attribute, eq_odds))


# EQUALIZED ODDS DECISION TREE and RANDOM FOREST
X['sex'] = X['sex'].replace({1: 0, 2: 1}).astype(int)
sensitive_attributes = ["sex"]

for sensitive_attribute in sensitive_attributes:
    # Identify indices for the privileged and unprivileged groups based on the sensitive attribute
    privileged_indices = np.where(X_test[sensitive_attribute] == 1)[0]
    unprivileged_indices = np.where(X_test[sensitive_attribute] == 0)[0]

    # Filter the predictions and true labels for the privileged and unprivileged groups
    predictions_privileged = y_pred[privileged_indices]
    predictions_unprivileged = y_pred[unprivileged_indices]
    
    # Create a Random Forest classifier object
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_rf = rf_classifier.predict(X_test)

# Calculate the predicted probabilities for positive class
y_pred_proba_rf = rf_classifier.predict_proba(X_test)[:, 1]

    # Make predictions on the testing data
y_pred_rf = rf_classifier.predict(X_test)

    
    # Filter the predictions for the privileged and unprivileged groups
predictions_privileged_rf = y_pred_rf[privileged_indices_rf]
predictions_unprivileged_rf = y_pred_rf[unprivileged_indices_rf]

# Calculate demographic parity for Random Forest
dp_rf = abs(sum(predictions_privileged_rf) / len(predictions_privileged_rf) - sum(predictions_unprivileged_rf) / len(predictions_unprivileged_rf))


if len(predictions_privileged_rf) != 0 and len(predictions_unprivileged_rf) != 0:
        # Calculate the Equalized Odds metric for Random Forest
        eq_odds_rf = abs(sum(predictions_privileged_rf) / len(predictions_privileged_rf) - sum(predictions_unprivileged_rf) / len(predictions_unprivileged_rf))
        print("Random Forest - Equalized Odds for '{}': {:.4f}".format(sensitive_attribute, eq_odds_rf))

    # Identify indices for the privileged and unprivileged groups based on the sensitive attribute
privileged_indices_dt = np.where(X_test[sensitive_attribute] == 1)[0]
unprivileged_indices_dt = np.where(X_test[sensitive_attribute] == 0)[0]

    # Filter the predictions for the privileged and unprivileged groups
predictions_privileged_dt = y_pred_dt[privileged_indices_dt]
predictions_unprivileged_dt = y_pred_dt[unprivileged_indices_dt]

if len(predictions_privileged_dt) != 0 and len(predictions_unprivileged_dt) != 0:
        # Calculate the Equalized Odds metric for Decision Tree
        eq_odds_dt = abs(sum(predictions_privileged_dt) / len(predictions_privileged_dt) - sum(predictions_unprivileged_dt) / len(predictions_unprivileged_dt))
        print("Decision Tree - Equalized Odds for '{}': {:.4f}".format(sensitive_attribute, eq_odds_dt))



# EQUALIZED ODDS for MULTILAYER P.

# Step 7: Calculate fairness metric - Equalized Odds
# Identify indices for each race category
sex_category = ["sex"]

X['sex'] = X['sex'].replace({1: 0, 2: 1}).astype(int)
sensitive_attributes = ["sex"]


for sex_category in sex_category:
    # Identify indices for the two groups based on the sensitive attribute
    group_indices_0_mlp = np.where((X_test[sex_category] == 0) & (y_test == 0))[0]
    group_indices_1_mlp = np.where((X_test[sex_category] == 1) & (y_test == 0))[0]

    # Filter the predictions for the two groups
    predictions_0_mlp = y_pred_mlp[(X_test[sex_category] == 0) & (y_test == 0)]
    predictions_1_mlp = y_pred_mlp[(X_test[sex_category] == 1) & (y_test == 0)]

    # Calculate the Equalized Odds metric for Multilayer Perceptron
    eq_odds_0_mlp = abs(sum(predictions_0_mlp) / len(predictions_0_mlp) - sum(predictions_1_mlp) / len(predictions_1_mlp))
    print("Multilayer Perceptron - Equalized Odds for", sex_category + ":", eq_odds_0_mlp)



X['sex'] = X['sex'].replace({1: 0, 2: 1}).astype(int)
sensitive_attributes = ["sex"]

for sensitive_attribute in sensitive_attributes:
    # Identify indices for the privileged and unprivileged groups based on the sensitive attribute
    privileged_indices = np.where(X_test[sensitive_attribute] == 1)[0]
    unprivileged_indices = np.where(X_test[sensitive_attribute] == 0)[0]

    # Filter the predictions and true labels for the privileged and unprivileged groups
    predictions_privileged = y_pred[privileged_indices]
    predictions_unprivileged = y_pred[unprivileged_indices]

    if len(predictions_privileged) != 0 and len(predictions_unprivileged) != 0:
        # Calculate the Equalized Odds metric
        eq_odds_0_mlp = abs(sum(predictions_0_mlp) / len(predictions_0_mlp) - sum(predictions_1_mlp) / len(predictions_1_mlp))
    print("Multilayer Perceptron - Equalized Odds for", sex_category + ":", eq_odds_0_mlp)
