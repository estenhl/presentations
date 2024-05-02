import pandas as pd
import matplotlib.pyplot as plt


for modality in ['fmri', 'dmri']:
    all = pd.read_csv(f'scripts/data/{modality}_all.csv', index_col=None, skiprows=1)
    all = all.set_index('Year')
    all.columns = ['all']
    ml = pd.read_csv(f'scripts/data/{modality}_prediction.csv', index_col=None, skiprows=1)
    ml = ml.set_index('Year')
    ml.columns = ['prediction']
    df = pd.merge(all, ml, left_index=True, right_index=True)
    df['ratios'] = df['prediction'] / df['all']
    plt.plot(df.index.values, df['ratios'], label=modality)

plt.legend()
plt.show()

