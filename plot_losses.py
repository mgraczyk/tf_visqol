import json
import sys
import numpy as np
from matplotlib import pyplot as plt

def moving_average(a, n=17) :
  ret = np.cumsum(a, dtype=float)
  ret[n:] = ret[n:] - ret[:-n]
  return ret[n - 1:] / n

losses = []
with open(sys.argv[1]) as f:
  for line in f:
    losses.append(json.loads(line[8:].replace("'", '"')))

ax = plt.gca()
ax.plot([l["loss"] for l in losses])
ax.plot(moving_average([l["loss"] for l in losses]))
ax.set_xlabel("batch")
ax.set_ylabel("total loss (reg + nsim + energy + clipping)")
ax.legend(("values", "moving avg"), loc="best")
ax.set_title("Training Loss")
plt.show()
