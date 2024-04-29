
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve
from scipy.stats import gaussian_kde


df = pd.read_csv('scripts/data/hippo.csv', index_col=None)

min_value = np.amin(df['Hippocampus_bl'])
max_value = np.amax(df['Hippocampus_bl'])
bins = np.linspace(min_value - 500, max_value + 500, 100)

traces = {}

for diagnosis in ['AD', 'CN']:
    values = df.loc[df['DX_bl'] == diagnosis, 'Hippocampus_bl']
    hist_values, _, = np.histogram(values, bins)
    smoother = gaussian_kde(values)
    hist_values = smoother(bins)
    plt.plot(bins, hist_values)
    traces[diagnosis] = hist_values

trace_df = pd.DataFrame({'AD': traces['AD'], 'CN': traces['CN'], 'volume': bins})
trace_df.to_csv('data/hippocampus_traces.csv', index=False)
print(np.amax(trace_df[['AD', 'CN']].values))

plt.show()

df['DX'] = df['DX_bl'].apply(lambda x: 0 if x == 'CN' else 1)
model = LogisticRegression()
model.fit(df[['Hippocampus_bl']], df['DX'])
predictions = model.predict_proba(df[['Hippocampus_bl']])[:,1]
print(predictions)
fpr, tpr, threshold = roc_curve(df['DX'], predictions)
accuracy = accuracy_score(df['DX'], np.round(predictions))
print(accuracy)

auc_df = pd.DataFrame({'fpr': np.round(fpr, 3), 'tpr': np.round(tpr, 3)})
auc_df.to_csv('data/hippocampus_roc.csv', index=False)

plt.plot(auc_df['fpr'], auc_df['tpr'])
plt.show()

