from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from LandingRockets.NeuralNetwork import NeuralNetwork

model = NeuralNetwork()
model.load_existing_model(path="./LandingRockets/",model_name="140k_samples_1024neurons_3layers_l2-0.000001")

results = list()
for vgo in tqdm(range(150, 1000, 10)):
    for vgh in range(150, 1000, 10):
        for vgc in range(150, 1000, 10):
            results.append([vgo/1000, vgh/1000, vgc/1000])
tmp = model.predict_data_point(results)
df = pd.DataFrame(results, columns=["vgo", "vgh", "vgc"])
df["pcc"] = tmp[:, 0]
df["temp"] = tmp[:, 1]
df["rof"] = tmp[:, 3] / tmp[:, 2]
df["thrust"] = tmp[:, 4]
#results.append([vgo / 100, vgh / 100, vgc / 100, tmp[0], tmp[1], tmp[3] / tmp[2]])
#df.plot.scatter("pcc", "rof", c="temp", colormap='plasma')

#df.plot.scatter("vgo", "vgh", c="temp", colormap='plasma')


df2 = df[df["temp"] < 950]
df2 = df2[df2["temp"] > 850]
df2 = df2[df2["rof"] < 5.8]
df2 = df2[df2["rof"] > 5.2]
df2 = df2[df2["thrust"] < 700]
df2 = df2[df2["thrust"] > 650]

plt.rcParams["figure.figsize"] = 12.8, 9.6
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pnt3d = ax.scatter(df2["vgo"], df2["vgh"], df2["vgc"], c=df2["thrust"], cmap='plasma', linewidth=1)
ax.set_xlabel("vgo")
ax.set_ylabel("vgh")
ax.set_zlabel("vgc")

df2.plot.scatter("vgo", "vgh", c="vgc", colormap='plasma')
cbar = plt.colorbar(pnt3d)