
# STEPS OF MACHINE LEARNING
- Define: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
- ETL (Extract, Transform, Load)
- Fit: Capture patterns from provided data. This is the heart of modeling.
- Predict: Just what it sounds like
- Evaluate: Determine how accurate the model's predictions are.

## Define
You need to define which model do you will use and which metrics you gona choose. Then, treat all this data with some lib/software that make this job.

## ETL
ETL: In this step, we need to extract data from a source, transform it into a format that can be used by the model, and load it into the model.

## FIT
This part is about to literally FIT data into the modeling parameters. For example, if you choose to use sklearn, in the lib Decision Tree, you can use the method fit() to fit the data.

## Predict
As its name says, this part is about to predict the data. You can use the same lib/software that you used in the fit step. For example, if you used sklearn, you can use the method predict() to predict the data.

## Evaluate
This is a step where we can messure how good is our model. We can use some metrics to evaluate the model. There are many Metrics that we can use.

An example of metrics are: MAE is the Mean Absolute Error.

### Example of evaluation metrics:
We can start with train_test_split() to split the data into train and test sets. Then, we can use the method fit() to fit the data. After that, we can use the method predict() to predict the data. Finally, we can use the method mean_absolute_error() to evaluate the model.

### What is MAE?
MAE is the Mean Absolute Error. It is a measure of how close predictions are to the actual outcomes. It is calculated as the average of the absolute differences between predicted and actual values. The lower the MAE, the better the model's performance.

### Coding example

```python
# ------------------- IMPORT -------------------
# import the libraries
import pandas as pd 

# ------------------- ETL ------------------- 
# Example of loading data from a CSV file
csv_file = 'data.csv'

# Load the data into a pandas DataFrame
data = pd.read_csv(csv_file)

# Example of transforming the data
# Here we can do some data cleaning, feature engineering, etc.
# For example: we want to drop any rows with missing values
data = data.dropna()

# ------------------- FIT -------------------
y = data['target'] # target variable
features = ['feature1', 'feature2', 'feature3'] # == columns to be used as features
X = data[features] # features
model.fit(X, y) # fit the model with the data

# ------------------- PREDICT -------------------
# Example of predicting the data
y_pred = model.predict(X) # predict the data

# ------------------- EVALUATE -------------------
# Example of evaluating the model

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Fit the model with the training data
model.fit(X_train, y_train)

# Predict the data with the test data
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# ------------------- END -------------------

```
