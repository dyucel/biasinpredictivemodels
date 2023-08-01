#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 14:32:02 2023

@author: suheyladenizyucel
"""

# Import the necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


# Read the CSV file into a pandas DataFrame
dataset = pd.read_csv("/Users/suheyladenizyucel/Desktop/adultnum2.csv")

# Define the feature columns and target column
feature_columns = ["sex.Female", "sex.Male", "workclass.Federal.gov", "workclass.Local.gov", "workclass.Never.worked", "workclass.Private", "workclass.Self.emp.inc", "workclass.Self.emp.not.inc", "workclass.State.gov", "workclass.Without.pay", "education.11th", "education.12th", "education.1st.4th", "education.5th.6th", "education.7th.8th", "education.9th", "education.Assoc.acdm", "education.Assoc.voc", "education.Bachelors", "education.Doctorate", "education.HS.grad", "education.Masters", "education.Preschool", "education.Prof.school", "education.Some.college", "marital.status.Married.AF.spouse", "marital.status.Married.civ.spouse", "marital.status.Married.spouse.absent", "marital.status.Never.married", "marital.status.Separated", "marital.status.Widowed", "occupation.Adm.clerical", "occupation.Armed.Forces", "occupation.Craft.repair", "occupation.Exec.managerial", "occupation.Farming.fishing", "occupation.Handlers.cleaners", "occupation.Machine.op.inspct", "occupation.Other.service", "occupation.Priv.house.serv", "occupation.Prof.specialty", "occupation.Protective.serv", "occupation.Sales", "occupation.Tech.support", "occupation.Transport.moving", "relationship.Not.in.family", "relationship.Other.relative", "relationship.Own.child", "relationship.Unmarried", "relationship.Wife", "race.Asian.Pac.Islander", "race.Black", "race.Other", "race.White", "native.country.Cambodia", "native.country.Canada", "native.country.China", "native.country.Columbia", "native.country.Cuba", "native.country.Dominican.Republic", "native.country.Ecuador", "native.country.El.Salvador", "native.country.England", "native.country.France", "native.country.Germany", "native.country.Greece", "native.country.Guatemala", "native.country.Haiti", "native.country.Holand.Netherlands", "native.country.Honduras", "native.country.Hong", "native.country.Hungary", "native.country.India", "native.country.Iran", "native.country.Ireland", "native.country.Italy", "native.country.Jamaica", "native.country.Japan", "native.country.Laos", "native.country.Mexico", "native.country.Nicaragua", "native.country.Outlying.US.Guam.USVI.etc.", "native.country.Peru", "native.country.Philippines", "native.country.Poland", "native.country.Portugal", "native.country.Puerto.Rico", "native.country.Scotland", "native.country.South", "native.country.Taiwan", "native.country.Thailand", "native.country.Trinadad.Tobago", "native.country.United.States", "native.country.Vietnam", "age", "educational.num", "capital.gain", "capital.loss", "hours.per.week"]
target_column = "income..50K"

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

# Step 5: Calculate fairness metric - Demographic Parity
# Identify indices for the two groups based on 'sex' attribute
# Step 5: Calculate fairness metric - Demographic Parity
# Identify indices for the two groups based on 'sex' attribute
female_indices = np.where(X_test['sex.Female'] == 1)[0]
male_indices = np.where(X_test['sex.Male'] == 1)[0]

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
   
# Define the sensitive attributes or groups
sensitive_attributes = ["race.Asian.Pac.Islander", "race.Black", "race.Other", "race.White"]

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

# Identify indices for the two groups based on 'sex' attribute
female_indices = np.where(X_test['sex.Female'] == 1)[0]
male_indices = np.where(X_test['sex.Male'] == 1)[0]

# Perform error checking for index validity
if max(female_indices) >= len(y_pred) or max(male_indices) >= len(y_pred):
    print("Error: Invalid index found in 'female_indices' or 'male_indices'.")
    print("Length of y_pred:", len(y_pred))
else:
    # Filter the predictions and true labels for the two groups
    female_predictions = y_pred[female_indices]
    male_predictions = y_pred[male_indices]

    # Calculate equal opportunity
    equal_opportunity = abs(sum(female_predictions) / len(female_predictions) - sum(male_predictions) / len(male_predictions))

    # Print the equal opportunity score
    print("Equal Opportunity: {:.4f}".format(equal_opportunity))

# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score

# Create a Random Forest classifier object
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_rf = rf_classifier.predict(X_test)

# Evaluate the utility of the model using accuracy and F1-score
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1score_rf = f1_score(y_test, y_pred_rf)

print("Random Forest - Accuracy:", accuracy_rf)
print("Random Forest - F1-score:", f1score_rf)

# Calculate the predicted probabilities for positive class
y_pred_proba_rf = rf_classifier.predict_proba(X_test)[:, 1]

# Calculate the ROC-AUC score
roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
print("Random Forest - ROC-AUC score:", roc_auc_rf)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

# Create a Decision Tree classifier object
dt_classifier = DecisionTreeClassifier()

# Fit the classifier to the training data
dt_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_dt = dt_classifier.predict(X_test)

# Evaluate the utility of the model using accuracy and F1-score
accuracy_dt = accuracy_score(y_test, y_pred_dt)
f1score_dt = f1_score(y_test, y_pred_dt)

print("Decision Tree - Accuracy:", accuracy_dt)
print("Decision Tree - F1-score:", f1score_dt)

# Calculate the predicted probabilities for positive class
y_pred_proba_dt = dt_classifier.predict_proba(X_test)[:, 1]

# Calculate the ROC-AUC score
roc_auc_dt = roc_auc_score(y_test, y_pred_proba_dt)
print("Decision Tree - ROC-AUC score:", roc_auc_dt)

# Step 5: Calculate fairness metric - Demographic Parity
# Identify indices for the privileged and unprivileged groups based on the sensitive attribute
privileged_indices_rf = np.where(X_test['sex.Male'] == 1)[0]
unprivileged_indices_rf = np.where(X_test['sex.Female'] == 1)[0]

# Filter the predictions for the privileged and unprivileged groups
predictions_privileged_rf = y_pred_rf[privileged_indices_rf]
predictions_unprivileged_rf = y_pred_rf[unprivileged_indices_rf]

# Calculate demographic parity for Random Forest
dp_rf = abs(sum(predictions_privileged_rf) / len(predictions_privileged_rf) - sum(predictions_unprivileged_rf) / len(predictions_unprivileged_rf))

# Print the demographic parity for Random Forest
print("Random Forest - Demographic Parity:", dp_rf)

# Identify indices for the privileged and unprivileged groups based on the sensitive attribute
privileged_indices_dt = np.where(X_test['sex.Male'] == 1)[0]
unprivileged_indices_dt = np.where(X_test['sex.Female'] == 1)[0]

# Filter the predictions for the privileged and unprivileged groups
predictions_privileged_dt = y_pred_dt[privileged_indices_dt]
predictions_unprivileged_dt = y_pred_dt[unprivileged_indices_dt]

# Calculate demographic parity for Decision Tree
dp_dt = abs(sum(predictions_privileged_dt) / len(predictions_privileged_dt) - sum(predictions_unprivileged_dt) / len(predictions_unprivileged_dt))

# Print the demographic parity for Decision Tree
print("Decision Tree - Demographic Parity:", dp_dt)

# Step 7: Calculate fairness metric - Equal Opportunity
# Identify indices for the two groups based on the sensitive attribute
group_indices_0_rf = np.where((X_test['sex.Female'] == 1) & (y_test == 1))[0]
group_indices_1_rf = np.where((X_test['sex.Male'] == 1) & (y_test == 1))[0]

# Filter the predictions for the two groups
predictions_0_rf = y_pred_rf[(X_test['sex.Female'] == 1) & (y_test == 1)]
predictions_1_rf = y_pred_rf[(X_test['sex.Male'] == 1) & (y_test == 1)]

# Calculate the Equal Opportunity metric for Random Forest
equal_opportunity_rf = abs(sum(predictions_0_rf) / len(predictions_0_rf) - sum(predictions_1_rf) / len(predictions_1_rf))
print("Random Forest - Equal Opportunity:", equal_opportunity_rf)

# Identify indices for the two groups based on the sensitive attribute
group_indices_0_dt = np.where((X_test['sex.Female'] == 1) & (y_test == 1))[0]
group_indices_1_dt = np.where((X_test['sex.Male'] == 1) & (y_test == 1))[0]

# Filter the predictions for the two groups
predictions_0_dt = y_pred_dt[(X_test['sex.Female'] == 1) & (y_test == 1)]
predictions_1_dt = y_pred_dt[(X_test['sex.Male'] == 1) & (y_test == 1)]

# Calculate the Equal Opportunity metric for Decision Tree
equal_opportunity_dt = abs(sum(predictions_0_dt) / len(predictions_0_dt) - sum(predictions_1_dt) / len(predictions_1_dt))
print("Decision Tree - Equal Opportunity:", equal_opportunity_dt)

# Step 6: Calculate fairness metric - Equalized Odds for different races
# Define the sensitive attribute - race
sensitive_attributes_race = ["race.Asian.Pac.Islander", "race.Black", "race.Other", "race.White"]

for sensitive_attribute in sensitive_attributes_race:
    # Identify indices for the privileged and unprivileged groups based on the sensitive attribute
    privileged_indices_rf = np.where(X_test[sensitive_attribute] == 1)[0]
    unprivileged_indices_rf = np.where(X_test[sensitive_attribute] == 0)[0]

    # Filter the predictions for the privileged and unprivileged groups
    predictions_privileged_rf = y_pred_rf[privileged_indices_rf]
    predictions_unprivileged_rf = y_pred_rf[unprivileged_indices_rf]

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

from sklearn.neural_network import MLPClassifier

# MLP
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)

# Fit the classifier to the training data
mlp_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_mlp = mlp_classifier.predict(X_test)

# Evaluate the utility of the model using accuracy and F1-score
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
f1score_mlp = f1_score(y_test, y_pred_mlp)

print("Multilayer Perceptron - Accuracy:", accuracy_mlp)
print("Multilayer Perceptron - F1-score:", f1score_mlp)

# Calculate the predicted probabilities for positive class
y_pred_proba_mlp = mlp_classifier.predict_proba(X_test)[:, 1]

# Calculate the ROC-AUC score
roc_auc_mlp = roc_auc_score(y_test, y_pred_proba_mlp)
print("Multilayer Perceptron - ROC-AUC score:", roc_auc_mlp)

# Step 5: Calculate fairness metric - Demographic Parity
# Identify indices for the privileged and unprivileged groups based on the sensitive attribute
privileged_indices_mlp = np.where(X_test['sex.Female'] == 0)[0]
unprivileged_indices_mlp = np.where(X_test['sex.Female'] == 1)[0]

# Filter the predictions for the privileged and unprivileged groups
predictions_privileged_mlp = y_pred_mlp[privileged_indices_mlp]
predictions_unprivileged_mlp = y_pred_mlp[unprivileged_indices_mlp]

# Calculate demographic parity for Multilayer Perceptron
dp_mlp = abs(sum(predictions_privileged_mlp) / len(predictions_privileged_mlp) - sum(predictions_unprivileged_mlp) / len(predictions_unprivileged_mlp))

# Print the demographic parity for Multilayer Perceptron
print("Multilayer Perceptron - Demographic Parity:", dp_mlp)

# Step 6: Calculate fairness metric - Equal Opportunity
# Identify indices for the two groups based on the sensitive attribute
group_indices_0_mlp = np.where((X_test['sex.Female'] == 0) & (y_test == 1))[0]
group_indices_1_mlp = np.where((X_test['sex.Female'] == 1) & (y_test == 1))[0]

# Filter the predictions for the two groups
predictions_0_mlp = y_pred_mlp[(X_test['sex.Female'] == 0) & (y_test == 1)]
predictions_1_mlp = y_pred_mlp[(X_test['sex.Female'] == 1) & (y_test == 1)]

# Calculate the Equal Opportunity metric for Multilayer Perceptron
equal_opportunity_mlp = abs(sum(predictions_0_mlp) / len(predictions_0_mlp) - sum(predictions_1_mlp) / len(predictions_1_mlp))
print("Multilayer Perceptron - Equal Opportunity:", equal_opportunity_mlp)

# Step 7: Calculate fairness metric - Equalized Odds
# Identify indices for each race category
race_categories = ['race.Asian.Pac.Islander', 'race.Black', 'race.Other', 'race.White']

for race_category in race_categories:
    # Identify indices for the two groups based on the sensitive attribute
    group_indices_0_mlp = np.where((X_test[race_category] == 0) & (y_test == 0))[0]
    group_indices_1_mlp = np.where((X_test[race_category] == 1) & (y_test == 0))[0]

    # Filter the predictions for the two groups
    predictions_0_mlp = y_pred_mlp[(X_test[race_category] == 0) & (y_test == 0)]
    predictions_1_mlp = y_pred_mlp[(X_test[race_category] == 1) & (y_test == 0)]

    # Calculate the Equalized Odds metric for Multilayer Perceptron
    eq_odds_0_mlp = abs(sum(predictions_0_mlp) / len(predictions_0_mlp) - sum(predictions_1_mlp) / len(predictions_1_mlp))
    print("Multilayer Perceptron - Equalized Odds for", race_category + ":", eq_odds_0_mlp)

