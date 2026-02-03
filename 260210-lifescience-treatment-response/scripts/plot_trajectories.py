import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100)
controls = (1 - x)**(1/2)
leqembi_min =  controls + 0.1*x
leqembi =  controls + 0.3*x
leqembi_max =  controls + 0.5*x

fig, ax = plt.subplots(figsize=(10, 3.5))
plt.plot(x, controls)
plt.plot(x, leqembi, color='red')
plt.fill_between(x, leqembi_min, leqembi_max, color='red', alpha=0.25)
plt.xlim([0, 0.9])
plt.axis('off')
fig.patch.set_edgecolor("black")
fig.patch.set_linewidth(2)
plt.savefig('leqembi.png', bbox_inches="tight", pad_inches=0)

leqembi_min =  controls - 0.1*x
leqembi =  controls + 0.1*x
leqembi_max =  controls + 0.3*x

fig, ax = plt.subplots(figsize=(10, 3.5))
plt.plot(x, controls)
plt.plot(x, leqembi, color='red')
plt.fill_between(x, leqembi_min, leqembi_max, color='red', alpha=0.25)
plt.xlim([0, 0.9])
plt.axis('off')
fig.patch.set_edgecolor("black")
fig.patch.set_linewidth(2)
plt.savefig('kisunla.png', bbox_inches="tight", pad_inches=0)