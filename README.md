# Churn Model Project

## The Problem

The problem is that in our telco business, we occasionally have customers leave us. We think that the rate of customers leaving us is too high, and we want a way to (a) predict which customers are going to leave, and (b) identify which factors are causing customers to leave.

## The Value of the Problem Being Solved

If we can predict which customers are going to leave, then we can offer them special promos and ensure they get good customer service. This might encourage them to stay.

Also, if we can understand the underlying factors that cause customers to leave, then we can fix those problems and create more value and a better experience for customers.

## Outline of the Solution

* We cleaned the dataset by inputing a value of 20 for missing TotalCharges values (although there weren't many missing).
* We featured the dataset by:
    1. Dropping the Customer ID column.
    2. Transformed categorical features into binary 0 and 1 features using one-hot encoding.
    3. Dropped highly correlated features, using a cutoff of VIF of 10.
* We experimented with many different models, and settled on using an Adaboost model.
* After trying various hyperparameter values, we ended up using default scikit-learn values except for learning_rate, which we set to 0.6.

## Main Results

* Customer tenure seemed to impact churn rate quite a bit -- this was potentially the most important feature. The longer a customer has been with us, the less likely they are to churn.
* Additionally, these customer attributes were correlated with churn:
    * Having multiple lines
    * Paying with an elctric check
    * Having no phone service
    * Having Streaming TV.
    * Having paperless billing
    * Having fiber optic internet service
* And, these customer attributes were negatively correlated with churn (in other words, correlated with customer retention):
    * Having online security
    * Having no internet service (streaming movies)
    * Having a one-year or two-year contract

## Evaluation of Model on Test Data

We evaluated our model on unseen test data and got a ROC AUC score of 0.858. This leads us to believe that our model will generalize well to future data.