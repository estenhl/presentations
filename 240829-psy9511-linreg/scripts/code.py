import pandas as pd

df = pd.DataFrame({'manufacturer': ['Chevrolet', 'Ford', 'Pontiac']})
print(df.columns)
df = pd.get_dummies(df)
print(df.columns)
