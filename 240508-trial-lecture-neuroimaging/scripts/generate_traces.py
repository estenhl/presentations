import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

from collections import Counter
from functools import reduce
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from typing import List


colours = [
    'red',
    'blue',
    'green',
    'orange',
    'purple',
    'teal',
    'brown',
    'black'
]

modality_map = {
    'T1': 'sMRI',
    'T2': 'sMRI',
    'FLAIR': 'sMRI',
    'rsfMRI': 'fMRI',
    'tfMRI': 'fMRI',
    'DTI': 'dMRI',
    'PET': 'Molecular',
    'SPECT': 'Molecular',
    'EEG': 'Electrophysiological',
    'MEG': 'Electrophysiological'
}

diagnosis_map = {
    'AD': 'DEM'
}

known_diagnoses = ['DEM', 'MS', 'PD', 'SCZ', 'MDD', 'BP']

xlimit = [0, 1700]

def standardize(df: pd.DataFrame, modalities: bool = False,
                diagnoses: bool = False,
                known_diagnoses: List[str] = known_diagnoses) -> pd.DataFrame:
    df = df.copy()

    if modalities:
        df['modality'] = df['modality'].apply(lambda x: set(x.split('/')))
        df['modality'] = df['modality'].apply(
            lambda x: 'Multimodal' if len(x) > 1 else x.pop())
        for key, value in modality_map.items():
            df.loc[df['modality'] == key, 'modality'] = value

    if diagnoses:
        df['diagnosis'] = df['diagnosis'].apply(lambda x: set(x.split('/')))
        df['diagnosis'] = df['diagnosis'].apply(
            lambda x: [diag for diag in x if diag != 'HC']
        )
        df['diagnosis'] = df['diagnosis'].apply(
            lambda x: 'Multiclass' if len(x) > 1 else x.pop()
        )

        for key, value in diagnosis_map.items():
            df.loc[df['diagnosis'] == key, 'diagnosis'] = value

        unknown_diagnoses = set(df['diagnosis'].values) - set(known_diagnoses)
        print(f'Dropping {unknown_diagnoses}')
        df = df[df['diagnosis'].isin(known_diagnoses)]

    return df

def plot_occurences(df: pd.DataFrame):
    diagnoses = known_diagnoses + ['Multiclass']
    df = standardize(df, diagnoses=True, known_diagnoses=diagnoses)

    df['modality'] = df['modality'].apply(lambda x: x.split('/'))
    df['new_modality'] = df['modality'].apply(lambda x: [modality_map[y] if y in modality_map else y for y in x])
    print(df.loc[df['modality'].apply(lambda x: 'EEG' in x), ['modality', 'new_modality']])
    modalities = ['sMRI', 'fMRI', 'dMRI', 'Molecular', 'Electrophysiological']

    for modality in modalities:
        df[modality] = df['new_modality'].apply(lambda x: modality in x)

    counts = {diagnosis: {modality: np.sum(df.loc[df['diagnosis'] == diagnosis, modality]) \
                          for modality in modalities} \
              for diagnosis in diagnoses}
    modalities = pd.DataFrame([counts[key] for key in diagnoses],
                              index=diagnoses, columns=modalities)
    print(modalities)
    print(np.sum(modalities))

    #fig = px.bar(df, x='modality', y='count', color='diagnosis',
    #             title='Modalities')

    #fig.show()

def plot_accuracy_by_size(df: pd.DataFrame):
    df = standardize(df, diagnoses=True)

    traces = []

    for i, diagnosis in enumerate(['MS']):#np.unique(df['diagnosis'])):
        subset = df[df['diagnosis'] == diagnosis]
        #model = LinearGAM(s(0)).fit(subset[['sample']], subset['accuracy'])
        model = LinearRegression()
        model.fit(subset[['sample']].values, subset['accuracy'])
        print(len(subset))
        def predict(train, model, test):
            return model.predict(np.reshape(test, (-1, 1)))
            max_prediction = model.predict(np.amax(train))
            min_prediction = model.predict(np.amin(train))
            predictions = model.predict(test)
            predictions[np.where(test > np.amax(train))] = max_prediction
            predictions[np.where(test < np.amin(train))] = min_prediction

            return predictions

        traces += [
            go.Scatter(
                x=subset['sample'],
                y=subset['accuracy'],
                mode='markers',
                name=diagnosis,
                marker={
                    'color': colours[i]
                }
            ),
            go.Scatter(
                x=np.linspace(xlimit[0], xlimit[1], 100),
                y=predict(subset['sample'], model, np.linspace(xlimit[0], xlimit[1], 100)),
                mode='lines',
                showlegend=False,
                line={
                    'color': colours[i]
                }
            )
        ]

    fig = go.Figure(traces)
    fig.update_layout(
        xaxis={
            'range': xlimit
        }
    )
    fig.show()

def plot_accuracy_by_modality(df: pd.DataFrame):
    df = standardize(df, modalities=True, diagnoses=True)

    diagnoses = len(np.unique(df['diagnosis']))
    rows = int(np.floor(np.sqrt(diagnoses)))
    cols = int(np.ceil(diagnoses / rows))

    lengths = {key: len(df[df['diagnosis'] == key]) \
               for key in np.unique(df['diagnosis'])}
    diagnoses = sorted(np.unique(df['diagnosis']),
                       key=lambda x: lengths[x], reverse=True)
    modalities = sorted(np.unique(df['modality']))

    fig = make_subplots(rows, cols, subplot_titles=diagnoses)

    for i, diagnosis in enumerate(diagnoses):
        subset = df[df['diagnosis'] == diagnosis]
        row = i // cols
        col = i % cols

        for j, modality in enumerate(modalities):
            inner_subset = subset[subset['modality'] == modality]

            if len(inner_subset) == 0:
                continue

            fig.add_trace(
                go.Scatter(
                    x=inner_subset['sample'],
                    y=inner_subset['accuracy'],
                    mode='markers',
                    name=modality,
                    showlegend=i == 0,
                    marker={
                        'color': colours[j]
                    }
                ), row=int(row) + 1, col=int(col) + 1
            )
            fig.add_trace(
                go.Scatter(
                    x=xlimit,
                    y=[np.mean(inner_subset['accuracy']),
                       np.mean(inner_subset['accuracy'])],
                    mode='lines',
                    showlegend=False,
                    line={
                        'color': colours[j]
                    }
                ), row=int(row) + 1, col=int(col) + 1
            )
    fig.update_layout(
        xaxis={
            'range': xlimit
        }
    )
    fig.show()

def plot_modality(df: pd.DataFrame, modality: str, modalities: str):
    df = standardize(df, diagnoses=True)
    df['modality'] = df['modality'].apply(lambda x: x.split('/'))

    if modalities is not None:
        df['modality'] = df['modality'].apply(
            lambda x: [m if m not in modalities else modality for m in x]
        )

    df = df[df['modality'].apply(lambda x: modality in x)]
    print(f'Found {len(df)} {modality} studies: {Counter(df["diagnosis"])}')

    df[['year', 'sample', 'diagnosis', 'accuracy']].to_csv(f'data/{modality}_studies.csv', index=False)

    for diagnosis in np.unique(df['diagnosis']):
        print(f'{modality} mean accuracy {diagnosis}: '
              f'{np.mean(df.loc[df["diagnosis"] == diagnosis, "accuracy"]):.2f} '
              f'({np.amax(df.loc[df["diagnosis"] == diagnosis, "sample"])})')



def plot_t2(df: pd.DataFrame):
    plot_modality(df, 'T2', ['FLAIR'])

def plot_dmri(df: pd.DataFrame):
    plot_modality(df, 'dMRI', ['DTI'])

def plot_molecular(df: pd.DataFrame):
    plot_modality(df, 'molecular', ['PET', 'SPECT'])

def plot_fmri(df: pd.DataFrame):
    plot_modality(df, 'fMRI', ['tfMRI', 'rsfMRI'])

def plot_boxplots(df: pd.DataFrame):
    df = standardize(df, diagnoses=True)

    fig = px.box(df, x='diagnosis', y='accuracy')
    fig.show()


    for diagnosis in known_diagnoses:
        subset = df[df['diagnosis'] == diagnosis]
        accuracies = subset['accuracy'].values
        upper_quartile = np.percentile(accuracies, 75)
        lower_quartile = np.percentile(accuracies, 25)
        iqr = upper_quartile - lower_quartile
        upper_whisker = min(np.amax(accuracies), upper_quartile + 1.5 * iqr)
        lower_whisker = max(np.amin(accuracies), lower_quartile - 1.5 * iqr)
        print(diagnosis)
        print(f'median: {np.median(accuracies):.2f}')
        print(f'upper quartile: {upper_quartile:.2f}')
        print(f'lower quartile: {lower_quartile:.2f}')
        print(f'upper whisker: {upper_whisker:.2f}')
        print(f'lower whisker: {lower_whisker:.2f}')

def plot_per_disorder(df: pd.DataFrame):
    df = standardize(df, diagnoses=True)

    fig = make_subplots(rows=2, cols=3)

    for i, diagnosis in enumerate(np.unique(df['diagnosis'])):
        row = i // 3
        col = i % 3
        subset = df[df['diagnosis'] == diagnosis]
        model = LinearRegression()
        model.fit(subset[['sample']].values, subset['accuracy'])
        print(diagnosis)
        print(f'Intercept: {model.intercept_:.4f}, coef: {model.coef_[0]:.4f}')

        fig.add_trace(
            go.Scatter(
                x=subset['sample'],
                y=subset['accuracy'],
                mode='markers'
            ), row=row+1, col=col+1
        )
        x_min = np.amin(subset['sample'])
        x_max = np.amax(subset['sample'])
        fig.add_trace(
            go.Scatter(
                x=[x_min, x_max],
                y=model.predict([[x_min], [x_max]]),
                mode='lines'
            ), row=row+1, col=col+1
        )
        subset.to_csv(f'data/{diagnosis}_accuracy_sample.csv', index=False)

    fig.show()

def plot_multimodality(df: pd.DataFrame):
    df = standardize(df, modalities=True)

    df['multimodal'] = df['modality'].apply(lambda x: x == 'Multimodal')
    df = df.groupby('year')['multimodal'].mean().reset_index()
    #plt.plot(df['year'], df['multimodal'])
    #plt.show()

    df.to_csv('data/multimodal_years.csv', index=False)

def plot_future(df: pd.DataFrame):
    df = standardize(df)

    cs = ['red', 'blue']

    for i, method in enumerate(np.unique(df['method'])):
        print(i)
        print(method)
        subset = df[df['method'] == method]
        model = LinearRegression()
        model.fit(subset[['year']], subset['accuracy'])
        plt.scatter(subset['year'], subset['accuracy'], c=cs[i], label=method)
        plt.plot([2005, 2020], model.predict([[2005], [2020]]), c=cs[i])

        print(f'{method}: {model.intercept_:.3f}, {model.coef_[0]:.3f}')
        subset.to_csv(f'data/{method}_accuracies.csv')

    plt.show()

    model = LinearRegression()
    model.fit(df[['year']], df['sample'])
    print(f'Samples: {model.intercept_:.3f}, {model.coef_[0]:.3f}')
    df.to_csv('data/samples_per_year.csv', index=False)
    plt.scatter(df['year'], df['sample'])
    plt.plot([2005, 2020], model.predict([[2005], [2020]]))
    plt.show()

def expand(df: pd.DataFrame):
    df['modality'] = df['modality'].apply(lambda x: x.split('/'))
    df['diagnosis'] = df['diagnosis'].apply(lambda x: x.split('/'))
    df['diagnosis'] = df['diagnosis'].apply(lambda x: [y for y in x if y != 'HC'])
    df = df.explode('modality')
    df = df.explode('diagnosis')
    df['diagnosis'] = df['diagnosis'].apply(
        lambda x: diagnosis_map[x] if x in diagnosis_map else x
    )
    df = df[['source', 'modality', 'diagnosis']]

    modality_df = df.groupby(['source', 'modality']).count().reset_index()
    modality_df = modality_df.rename(columns={'diagnosis': 'count'})
    modality_df.to_csv('data/expanded_modalities.csv', index=False)

    diagnosis_df = df.groupby(['source', 'diagnosis']).count().reset_index()
    diagnosis_df = diagnosis_df.rename(columns={'modality': 'count'})
    diagnosis_df = diagnosis_df[diagnosis_df['diagnosis'].apply(
        lambda x: x in known_diagnoses
    )]

    for author in np.unique(diagnosis_df['source']):
        for diagnosis in known_diagnoses:
            rows = diagnosis_df.loc[(diagnosis_df['source'] == author) & \
                                    (diagnosis_df['diagnosis'] == diagnosis)]

            if len(rows) == 0:
                diagnosis_df = pd.concat([diagnosis_df,
                                          pd.DataFrame({'source': [author],
                                                        'diagnosis': [diagnosis],
                                                        'count': [0]})])

    diagnosis_df['id'] = diagnosis_df['diagnosis'].apply(
        lambda x: known_diagnoses.index(x)
    )

    diagnosis_df.to_csv('data/expanded_diagnoses.csv', index=False)

df = pd.read_csv('scripts/data/trial_lecture_data.csv')
print(f'Originally: {len(df)}')
df = df.drop_duplicates(['author', 'year', 'diagnosis', 'modality'])
print(f'After dropping: {len(df)}')
print(df[pd.isna(df['accuracy'])])
print(Counter(df['modality']))
print(Counter(df['diagnosis']))
#plot_occurences(df)
#plot_accuracy_by_size(df)
#plot_accuracy_by_modality(df)
#plot_t2(df)
#plot_dmri(df)
#plot_molecular(df)
#plot_fmri(df)
#plot_boxplots(df)
#plot_per_disorder(df)
#plot_multimodality(df)
#plot_future(df)
expand(df)


