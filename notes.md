# STEPS OF MACHINE LEARNING
- Define: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
- Fit: Capture patterns from provided data. This is the heart of modeling.
- Predict: Just what it sounds like
- Evaluate: Determine how accurate the model's predictions are.

## Define
You need to define which model do you will use and which metrics you gona choose. Then, treat all this data with some lib/software that make this job.

## Fit
This part is about to literally FIT data into the modeling parameters.
For example, if you choose to use sklearn, in the lib Decision Tree, you can use the method fit() to fit the data.

## Predict
As its name says, this part is about to predict the data. You can use the same lib/software that you used in the fit step. For example, if you used sklearn, you can use the method predict() to predict the data.

## Evaluate
This is a step where we can messure how good is our model. We can use some metrics to evaluate the model. There are many Metrics that we can use.

An example of metrics are: MAE is the Mean Absolute Error.

### Example of evaluation metrics:
We can start with train_test_split() to split the data into train and test sets. Then, we can use the method fit() to fit the data. After that, we can use the method predict() to predict the data. Finally, we can use the method mean_absolute_error() to evaluate the model.

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd


# Load the data
data = pd.read_csv('data.csv')

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42)

# Define the model
model = DecisionTreeRegressor()

# Fit the model
model.fit(X_train, y_train)

# Predict the data
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
```

### What is MAE?
MAE is the Mean Absolute Error. It is a measure of how close predictions are to the actual outcomes. It is calculated as the average of the absolute differences between predicted and actual values. The lower the MAE, the better the model's performance.

