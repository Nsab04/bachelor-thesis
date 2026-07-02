import numpy as np
import matplotlib.pyplot as plt
import pickle

def showcl(cl,ax):
    for c in cl:
        vs = c.points
        x, y = zip(*vs)
        if (c.principale):
            ax.plot(x, y, '-', color='black')
        else:
            ax.plot(x, y, '-', color='red')

PIK = "pickle_test.dat"
with open(PIK, "rb") as f:
    dff=pickle.load(f)

fig,ax=plt.subplots(1,1,figsize=(10,10))
ax.plot(dff.zs,dff.GGSL,'o-')
ax.plot(dff.zs,dff.MI,'o-')
ax.set_xscale('log')
ax.set_yscale('log')
plt.show()

fig,ax=plt.subplots(1,1,figsize=(10,10))
for i in range(len(dff.zs)):
    cl=dff.CauLines[i]
    showcl(cl,ax)

plt.show()


