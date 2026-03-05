import matplotlib.pyplot as plt
import numpy as np
import ogstools as ot
###########################################################
#6: postprocessing OGSTools
mesh_series = ot.MeshSeries("_out/stimtec.pvd")
print(mesh_series.timevalues)
pressure = ot.variables.pressure

rows = np.array([np.linspace([0, y, 0], [50, y, 0], 6) for y in [-0.5, 0.5]])
ms_pts = [ot.MeshSeries.extract_probe(mesh_series, pts) for pts in rows]
labels = [
    [f"{i}: x={pt[0]: >5} y={pt[1]}" for i, pt in enumerate(pts)]
    for pts in rows
]

fig, axs = plt.subplots(nrows=2, figsize=[16, 10], sharey=True)
ot.plot.line(ms_pts[0], "time", pressure, ax=axs[0], color="k", fontsize=18)
ot.plot.line(ms_pts[1], "time", pressure, ax=axs[1], marker="o", fontsize=18)
# add the mean of the observation point timeseries
for index in range(2):
    values = pressure.transform(ms_pts[index])
    mean_values = np.mean((values), axis=-1)
    ts = ms_pts[index].timevalues
    fig.axes[index].plot(ts, mean_values, "rk"[index], lw=4)
    fig.axes[index].legend(labels[index] + ["mean"], fontsize=15)
    
plt.show()
