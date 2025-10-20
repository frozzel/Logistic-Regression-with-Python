import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Load Data from URL
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn_df = pd.read_csv(url)

# print(churn_df.head())

## Data Preprocessing
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
# print(churn_df.head())  

"""For modeling the input fields X and the target field y need to be fixed. 
Since that the target to be predicted is 'churn', 
the data under this field will be stored under the variable 'y'. 
We may use any combination or all of the remaining fields as the input. 
Store these values in the variable 'X'.
"""

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
print(X[0:5])  #print the first 5 values

y = np.asarray(churn_df['churn'])
print(y[0:5]) #print the first 5 values

# Standardize the Data, StandardScalar function in the Scikit-Learn library
X_norm = StandardScaler().fit(X).transform(X)
print(X_norm[0:5])  #print the first 5 values

### Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4)

## Logistic Regression Classifier modeling
LR = LogisticRegression().fit(X_train,y_train)

yhat = LR.predict(X_test)
print(yhat[:10])  #print the first 10 values

coefficients = pd.Series(LR.coef_[0], index=churn_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()

"""Once the predictions have been generated, 
it becomes prudent to evaluate the performance of the model in predicting the target variable.
Log loss (Logarithmic loss), also known as Binary Cross entropy loss, 
is a function that generates a loss value based on the class wise prediction probabilities and the actual class labels.
The lower the log loss value, the better the model is considered to be.
"""

log_loss(y_test, yhat_prob)