import h5py
import numpy as np
import matplotlib.pyplot as plt

filename = 'dTable.hdf5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

# Get the data
data = list(f[a_group_key])
c = data[0][0]
bVals = np.array([itm[1] for itm in data])
dVals = np.array([itm[2] for itm in data])
bMin = min(bVals)
bMax = max(bVals)
dMax = max(dVals)

#begin figure and plot
fig = plt.figure()

#make x,y axis
# plt.plot((-1.,0.5),(0,0),'k--')
# plt.plot((0,0),(-1.0,1.0),'k--')
plt.xlabel('b', fontsize=15)
plt.ylabel('dist', fontsize=15)
plt.xlim(bMin,bMax)
plt.ylim(0,dMax)
plt.plot(bVals,dVals,"o")
plt.show()
