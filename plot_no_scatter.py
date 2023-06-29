# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 04:50:53 2023

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

#plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'font.family':'Arial'})

n=500
x = np.array([0.2, 0.6, 1])

def read_y(xi, m):
    ready = np.zeros(n)
    file = open('ordinate_x{}_{}.txt'.format(xi, m), 'r')
    for i in range (n):
        ready[i] = float(file.readline())
    file.close()
    return ready

def read_u(xi, m):
    readu = np.zeros(n)
    file = open('velocity_x{}_{}.txt'.format(xi, m), 'r')
    for i in range (n):
        readu[i] = float(file.readline())
    file.close()
    return readu

plt.figure(figsize = (6.3,4.2))
plt.plot(0.2+5*read_u(x[0], 0.5), read_y(x[0], 0.5))
plt.plot(0.6+5*read_u(x[1], 0.5), read_y(x[1], 0.5))
plt.plot(1+5*read_u(x[2], 0.5), read_y(x[2], 0.5))
plt.grid('minor', ls=":", c="k")
plt.ylim(bottom=-0.0001, top = 0.115)
plt.xlabel('x, м')
plt.ylabel('y, м')
plt.title('Развитие профилей продольной компоненты скорости над пластинкой при n=0.5')
plt.show()
#print(np.shape(get_u(0.5, 1)))
