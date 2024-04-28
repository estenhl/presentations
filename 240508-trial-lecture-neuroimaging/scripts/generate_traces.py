import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from pygam import LinearGAM, s
import statsmodels.formula.api as smf

from collections import Counter
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression


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
    'DTI': 'dMRI'
}

known_diagnoses = ['AD', 'SCZ', 'MDD', 'BP', 'MS', 'Multiclass', 'PD']

xlimit = [0, 1700]

def standardize(df: pd.DataFrame, modalities: bool = False,
                diagnoses: bool = False) -> pd.DataFrame:
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
        unknown_diagnoses = set(df['diagnosis'].values) - set(known_diagnoses)
        print(f'Dropping {unknown_diagnoses}')
        df = df[df['diagnosis'].isin(known_diagnoses)]

    return df

def plot_occurences(df: pd.DataFrame):
    df = standardize(df, modalities=True, diagnoses=True)

    df = df.groupby(['modality', 'diagnosis']).size().reset_index(name='count')

    fig = px.bar(df, x='modality', y='count', color='diagnosis',
                 title='Modalities')

    fig.show()

def plot_accuracy_by_size(df: pd.DataFrame):
    df = standardize(df, diagnoses=True)

    traces = []

    for i, diagnosis in enumerate(np.unique(df['diagnosis'])):
        subset = df[df['diagnosis'] == diagnosis]
        model = LinearGAM(s(0)).fit(subset[['sample']], subset['accuracy'])

        def predict(train, model, test):
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

def plot_t2(df: pd.DataFrame):
    df = standardize(df, diagnoses=True)
    df.loc[df['modality'] == 'FLAIR', 'modality'] = 'T2'
    df = df[df['modality'] == 'T2']

    df[['year', 'sample', 'diagnosis', 'accuracy']].to_csv('data/t2_studies.csv', index=False)

def plot_dmri(df: pd.DataFrame):
    df = standardize(df, diagnoses=True, modalities=True)
    df = df[df['modality'] == 'dMRI']

    df[['year', 'sample', 'diagnosis', 'accuracy']].to_csv('data/dmri_studies.csv', index=False)

def plot_fmri(df: pd.DataFrame):
    df = standardize(df, diagnoses=True, modalities=True)
    df = df[df['modality'] == 'fMRI']

    df[['year', 'sample', 'diagnosis', 'accuracy']].to_csv('data/fmri_studies.csv', index=False)

df = pd.read_csv('scripts/data/trial_lecture_data.csv')
df = df.drop_duplicates(['author', 'year', 'diagnosis', 'modality'])
print(Counter(df['modality']))
print(Counter(df['diagnosis']))
#plot_occurences(df)
#plot_accuracy_by_size(df)
#plot_accuracy_by_modality(df)
#plot_t2(df)
plot_dmri(df)
plot_fmri(df)


