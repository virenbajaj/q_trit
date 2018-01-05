import numpy as np
import matplotlib.pyplot as plt
# from sympy.solvers import solve
# from sympy import Symbol
import matplotlib.patches as mpatches

#begin figure and plot
fig = plt.figure()

#make x,y axis
plt.plot((-1.,0.5),(0,0),'k--')
plt.plot((0,0),(-1.0,1.0),'k--')
plt.xlabel('b', fontsize=15)
plt.ylabel('c', fontsize=15)
plt.xlim(-1.0,0.5)
plt.ylim(-.5,1)

#make lines c=b, c=-2b, c=-0.5b
#define functions for lines to be plotted
def f1(x):
    return x
def f2(x):
    return -2*x
def f3(x):
	return -.5*x
xr = np.linspace(-1,1,100)
y1r = f1(xr)
y2r = f2(xr)
y3r = f3(xr)
plt.plot(xr,y1r,'k')
plt.plot(xr,y2r, 'k')
plt.plot(xr,y3r, 'k')

#fill regions
mx = np.linspace(-1,0,50)
px = np.linspace(0,1,50)
my2 = f2(mx)
my1 = f1(mx)
py1 = f1(px)
py2 = f2(px)
plt.fill_between(xr,y1r,y3r, facecolor = 'gray')
plt.fill_between(mx,my2,1, facecolor = 'orange')
plt.fill_between(mx,-1,my1,facecolor = 'orange')
plt.fill_between(px,py1,1,facecolor = 'orange')
plt.fill_between(px,-1,py2,facecolor = 'orange')

#make legend
red_patch = mpatches.Patch(color='gray', label='E P.S.D ')
blue_patch = mpatches.Patch(color='orange', label='D P.S.D')
tr_patch = mpatches.Patch(color='white', label='D,E not P.S.D')
plt.legend(handles=[blue_patch,red_patch, tr_patch])

#title
fig.suptitle('b-c Phase Space', fontsize=20)
# plt.show()
fig.savefig("b-c Phase Space")