import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import norm

np.random.seed(42)
controls = np.random.normal(5, 3, 100)
controls_x = controls + np.random.normal(1, 0.2, 100)
controls_y = controls + np.random.normal(1, 0.2, 100)

cases = np.random.normal(5, 3, 100)
cases_x = cases + np.random.normal(1.2, 0.2, 100)
cases_y = cases - np.random.normal(1.2, 0.2, 100)

padding = 4
xmin = min(np.amin(controls_x), np.amin(cases_x)) - padding
xmax = max(np.amax(controls_x), np.amax(cases_x)) + padding
ymin = min(np.amin(controls_y), np.amin(cases_y)) - padding
ymax = max(np.amax(controls_y), np.amax(cases_y)) + padding

print(xmin, xmax)
print(ymin, ymax)
xrange = np.linspace(xmin, xmax, 1000)
yrange = np.linspace(ymin, ymax, 1000)

fig, ax = plt.subplots(1, 3, figsize=(10, 10))
ax[0].scatter(controls_x, controls_y)
ax[0].scatter(cases_x, cases_y, c='red')
ax[0].set_xlim([xmin, xmax])
ax[0].set_ylim([ymin, ymax])

controls_x_pdf = norm.pdf(xrange, np.mean(controls_x), np.std(controls_x))
controls_x_pdf = controls_x_pdf / np.amax(controls_x_pdf)
cases_x_pdf = norm.pdf(xrange, np.mean(cases_x), np.std(cases_x))
cases_x_pdf = cases_x_pdf / np.amax(cases_x_pdf)

controls_y_pdf = norm.pdf(yrange, np.mean(controls_y), np.std(controls_y))
controls_y_pdf = controls_y_pdf / np.amax(controls_y_pdf)
cases_y_pdf = norm.pdf(yrange, np.mean(cases_y), np.std(cases_y))
cases_y_pdf = cases_y_pdf / np.amax(cases_y_pdf)

ax[1].plot(xrange, controls_x_pdf)
ax[1].plot(xrange, cases_x_pdf, c='red')

ax[2].plot(yrange, controls_y_pdf)
ax[2].plot(yrange, cases_y_pdf, c='red')

points = pd.DataFrame({'case_x': cases_x, 'case_y': cases_y,
                       'control_x': controls_x, 'control_y': controls_y})
points.to_csv('data/case_control_points.csv', index=False)

x_distributions = pd.DataFrame({'x': xrange, 'case': cases_x_pdf,
                                'control': controls_x_pdf})
x_distributions.to_csv('data/case_control_x_distributions.csv', index=False)
y_distributions = pd.DataFrame({'y': yrange, 'case': cases_y_pdf,
                                'control': controls_y_pdf})
y_distributions.to_csv('data/case_control_y_distributions.csv', index=False)


#plt.show()

df = pd.DataFrame({'x': np.concatenate([controls_x, cases_x]),
                   'y': np.concatenate([controls_y, cases_y]),
                   'label': np.concatenate([np.zeros(100),
                                            np.ones(100)])})

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense

inputs = Input(shape=(2,))
hidden = Dense(5, activation='relu')(inputs)
hidden = Dense(4, activation='relu')(hidden)
hidden = Dense(3, activation='relu')(hidden)
outputs = Dense(1, activation='sigmoid')(hidden)

model = Model(inputs, outputs)
model.compile(loss='binary_crossentropy', optimizer='adam',
            metrics=['accuracy'])

model.fit(df[['x', 'y']].values, df['label'].values, epochs=200)

prev = df[['x', 'y']].values
expected = df['x'].values / df['y'].values
for l in range(1, 4):
    activations = model.layers[l](prev).numpy()
    values = np.concatenate([activations.T, np.expand_dims(expected, 0)])
    print(np.corrcoef(values)[0])
    prev = activations


