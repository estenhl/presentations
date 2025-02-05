import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter

from sklearn.linear_model import LinearRegression, LogisticRegression

df = pd.read_csv('/home/esten/Downloads/Auto.csv')
df = df[df['horsepower'] != '?']

df['horsepower'] = df['horsepower'].astype(int)
df['muscle'] = df['horsepower'] > np.mean(df['horsepower'])
print(Counter(df['muscle']))

train = df.sample(frac=0.8, random_state=42)
test = df.drop(train.index)
print(f'Training on {len(train)} observations, testing on {len(test)}')

singular_model = LinearRegression()
singular_model.fit(df[['horsepower']], df['mpg'])

x = np.linspace(np.amin(df['horsepower']), np.amax(df['horsepower']), 100)
y = singular_model.predict(x.reshape(-1, 1))
plt.scatter(df['horsepower'], df['mpg'])
plt.plot(x, y, c='red')
plt.show()

train_predictions = singular_model.predict(train[['horsepower']])
train_mae = np.mean(np.abs(train_predictions - train['mpg']))
print(f'Singular training MAE: {train_mae}')
test_predictions = singular_model.predict(test[['horsepower']])
test_mae = np.mean(np.abs(test_predictions - test['mpg']))
print(f'Singular test MAE: {test_mae}')

multivariate_model = LinearRegression()
multivariate_model.fit(train[['horsepower', 'weight', 'displacement', 'year']],
                       train['mpg'])
print(f'Intercept: {multivariate_model.intercept_}, coefficients: {multivariate_model.coef_}')

train_predictions = multivariate_model.predict(train[['horsepower', 'weight', 'displacement', 'year']])
train_mae = np.mean(np.abs(train_predictions - train['mpg']))
print(f'Multivariate training MAE: {train_mae}')
test_predictions = multivariate_model.predict(test[['horsepower', 'weight', 'displacement', 'year']])
test_mae = np.mean(np.abs(test_predictions - test['mpg']))
print(f'Multivariate test MAE: {test_mae}')

binary_model = LogisticRegression()
binary_model.fit(train[['weight', 'displacement', 'year']], train['muscle'])

train_predictions = binary_model.predict(train[['weight', 'displacement', 'year']])
train_accuracy = np.mean(train_predictions == train['muscle'])
print(f'Binary training accuracy: {train_accuracy}')
test_predictions = binary_model.predict(test[['weight', 'displacement', 'year']])
test_accuracy = np.mean(test_predictions == test['muscle'])
print(f'Binary test accuracy: {test_accuracy}')
