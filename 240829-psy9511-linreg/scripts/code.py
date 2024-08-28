import statsmodels.api as sm
import pandas as pd

df = pd.DataFrame({"manufacturer": ["ford", "chevrolet", "ford", "ford", "chevrolet", "ford", "chevrolet", "chevrolet", "ford", "chevrolet"]})
print(f'Columns before: {df.columns.values}')
df = pd.get_dummies(df)
print(f'Columns after: {df.columns.values}')

model = sm.OLS(df['mpg'], sm.add_constant(df[['horsepower']]))
fit = model.fit()
new_input = sm.add_constant(pd.DataFrame({'horsepower': [105, 106]}))
intervals = fit.get_prediction(new_input).summary_frame()
print(intervals)

