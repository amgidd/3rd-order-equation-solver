# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 03:38:36 2023

@author: User
"""

import numpy as np
import math

n=500
eta_max = 50.
root_vis = math.sqrt(1.4525e-5)
eta = np.linspace(0, eta_max, n)
x = np.array([0.2, 0.6, 1])

def read_der(m):
    file = open('derivative{}.txt'.format(m), "r")
    der = np.zeros(n)
    for i in range(n):
        der[i] = float(file.readline())
    file.close()
    return der

def y(m, xi):
    return root_vis*xi**((2-m)/3)*eta
def u(m, xi):
    return xi**((2*m-1)/3)*read_der(m)

def write(xi, m):
    file1 = open('ordinate_x{}_{}.txt'.format(xi, m), 'w')
    file2 = open('velocity_x{}_{}.txt'.format(xi, m), 'w')
    for i in range (n):
        file1.write('{}\n'.format(y(m, xi)[i]))
        file2.write('{}\n'.format(u(m, xi)[i]))
    file1.close()
    file2.close()
    return 1

read_der(3)
write(x[2], 3)
