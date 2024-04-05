import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc, rcParams
import niceplots

# rc('text', usetex=True)
# rc('axes', linewidth=2)
# rc('font', weight='bold')

islender = 2
model_df = pd.read_csv(f"slender{islender}-model-fit_model.csv")
rho_0 = model_df['rho_0'].to_numpy()
mean = model_df['mean'].to_numpy()
std_dev = model_df['std_dev'].to_numpy()

data_df = pd.read_csv(f"slender{islender}-model-fit_data.csv")
AR = data_df['AR'].to_numpy()
lam = data_df['lam'].to_numpy()

plt.style.use(niceplots.get_style())
plt.figure("check model", figsize=(8, 6))
plt.margins(x=0.05, y=0.05)
plt.title(f"b/h in [50,100]")
ax = plt.subplot(111)

colors = plt.cm.viridis(np.linspace(0, 1, 5))

ax.fill_between(
    x=rho_0,
    y1=mean - 3 * std_dev,
    y2=mean + 3 * std_dev,
    label='3-sigma',
    color=colors[0]
)
ax.fill_between(
    x=rho_0,
    y1=mean - std_dev,
    y2=mean + std_dev,
    label='1-sigma',
    color=colors[1]
)
ax.plot(
   rho_0, mean, "k", label="mean", linewidth=2
)  # , label=f"D*-[{Dstar_bin[0]},{Dstar_bin[1]}]""
ax.plot(
    AR, lam, "o", markersize=3, label="train-data",
    color=colors[2], markeredgecolor=colors[2]
)  # , label=f"D*-[{Dstar_bin[0]},{Dstar_bin[1]}]""

# outside of for loop save the plot
# font.size : 18
# axes.labelweight : bold

plt.xlabel(r"$\rho_0$")
plt.ylabel(r"$\lambda_{min}^*$")
plt.legend()
plt.xlim(0.1, 10.0)
plt.ylim(0.0, 10.0)
plt.savefig(f"slender{islender}-model-fit.png", dpi=400)