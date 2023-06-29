# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:36:35 2023

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import style
plt.rcParams.update({'font.family':'Arial'})

c0 = 0.06
n = 500
viscosity = 1.4525e-5 #кинематическая вязкость
area_procent = 0.90 #такая доля от площади под графиком нам нужна для калибра
y0 = 0 #значение функции в начале отрезка этта
y1b = 0 #значение первой производной функции в конце отрезка этта(бесконечность)
eta_max = 50.
h = eta_max/n
iter = 1500
max_res = 1e-6
alfa_initial = -0.008
nu1_initial = np.array([-0.01303152, 0.07177461]) 
"""nu - вектор (alfa, c)"""
nu2_initial = np.array([-0.02605679, 0.08789244])
nu3_initial = np.array([-0.03904817, 0.09955905])


#y_initial = np.array([y0, y1, al0]) # - стало

def runge_kutta(h, y_initial, f, m):
    """Находит решение y уравнения вида y'=f(x,y) методом Рунге-Кутта"""
    y = np.zeros((n, 3))
    y[0, :] = y_initial
    for i in range(n-1):
        k1 = f(y[i], m)
        k2 = f(y[i] + k1 * h/2, m)
        k3 = f(y[i] + k2 * h/2, m)
        k4 = f(y[i] + k3 * h, m)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) * h/6
    return y

def f(y, m):
    """вычисляет y' через y в уравнении y'=f(x,y)"""
    p = np.zeros(3)
    p[0] = y[1]
    p[1] = y[2]
    p[2] = ((2 * m - 1) / 3) * y[1] ** 2 - ((1 + m) / 3) * y[0] * y[2]
    return p

def nevyazka1(alfa, m):
    """вычисляет значение невязки первой производной на конце эта"""
    y_initial = np.array([y0, c0, alfa])
    vy = runge_kutta(h, y_initial, f, m)
    return np.abs(vy[-1, 1]) + np.abs(vy[-2, 1]) - 0

def nevyazka1_proizvodnaya(alfa, m, eps=1e-6):
    """вычисляет производную невязки1 по варьируемой альфе в методе Ньютона"""
    return (nevyazka1(alfa + eps, m) - nevyazka1(alfa-eps, m)) / (2*eps)

def Newtone1(alfa, m, nevyazka1, nevyazka1_proizvodnaya):
    """вычисляет альфу, подходящую для ГУ на конце эта, с точностью res"""
    i = 1
    res = abs(nevyazka1(alfa, m))
    while res > max_res and i < iter:
        alfa = alfa - nevyazka1(alfa, m) / nevyazka1_proizvodnaya(alfa, m)
        res = abs(nevyazka1(alfa, m))
        i += 1
        if i % 50 == 0:
            print('Iter = {}'.format(i))
            print('Res = {}'.format(res))
    return alfa

def trap(y):
    """интегрирует y**2 методом трапеции, умножает на корень из кинемат. вязкости"""
    integral = (2*sum(y**2) - y[n-1]**2 - y[0]**2) * h/2 * math.sqrt(viscosity)
    return integral

#вычисление импульса при m=1/2 с помощью интегрирования мет.трапеций
alfa0 = Newtone1(alfa_initial, 1/2, nevyazka1, nevyazka1_proizvodnaya)
y_init0 = np.array([y0, c0, alfa0])
reshenie0 = runge_kutta(h, y_init0, f, 1/2)
k0 = trap(reshenie0[:, 1])
print('Импульс при m=%.1f: %.5f' % (1/2, k0))

"""Далее метод Ньютона 2D для варьирования альфа и c"""

def nevyazka2(alfa, c, m):
    """вычисляет вектор невязки первой производной и импульса"""
    y_initial = np.array([y0, c, alfa])
    vy = runge_kutta(h, y_initial, f, m)
    a1 = np.abs(vy[-1, 1]) + np.abs(vy[-2, 1]) - 0
    a2 = trap(vy[:, 1]) - k0
    return np.array([a1, a2])

def jak2(alfa, c, m, eps=1e-9):
    """вычисляет обратную матрицу Якоби от вектора невязки по альфа и c"""
    a = np.array([nevyazka2(alfa  + eps, c, m) - nevyazka2(alfa - eps, c, m),
                  nevyazka2(alfa, c + eps, m) - nevyazka2(alfa, c - eps, m)]).T/(2*eps)
    return a

def Newtone2(alfa, c, m, nevyazka2, jak2):
    """вычисляет вектор начальных условий (альфа, c), подходящий двум ГУ на конце"""
    tau = 0.01
    i = 1

    nevyazka = nevyazka2(alfa, c, m)
    a = np.array([alfa, c])
    J = jak2(a[0], a[1], m, eps=1e-6)
    da1 = np.linalg.solve(J.T, -tau*nevyazka)
    a = a + da1
    res = np.sqrt(nevyazka.dot(nevyazka))

    while res > max_res and i < iter:
        J = jak2(a[0], a[1], m, eps=1e-9)
        nevyazka = nevyazka2(a[0], a[1], m)
        da2 = np.linalg.solve(J, -tau*nevyazka)
        a = a + da2
        i += 1
        tau = (da2-da1).dot(da2)/((da2-da1).dot(da2-da1))
        #if tau < 0 or np.abs(tau) > 1:
        tau = 0.01 + 0.025*i/iter
        da1 = da2[:]
        res = np.sqrt(nevyazka.dot(nevyazka))
        if i % 50 == 0:
            print('Iter = {}'.format(i))
            print('Res = {}'.format(res))
            #print('tau = {}'.format(tau))

    return a

"""nu - вектор (alfa, c)"""
nu1 = Newtone2(nu1_initial[0], nu1_initial[1], 1, nevyazka2, jak2)
y_init1 = np.array([y0, nu1[1], nu1[0]])
reshenie1 = runge_kutta(h, y_init1, f, 1)
k1 = trap(reshenie1[:, 1])
print('Импульс при m=%.1f: %.5f' % (1, k1))

nu2 = Newtone2(nu2_initial[0], nu2_initial[1], 2, nevyazka2, jak2)
y_init2 = np.array([y0, nu2[1], nu2[0]])
reshenie2 = runge_kutta(h, y_init2, f, 2)
k2 = trap(reshenie2[:, 1])
print('Импульс при m=%.1f: %.5f' % (2, k2))

nu3 = Newtone2(nu3_initial[0], nu3_initial[1], 3, nevyazka2, jak2)
y_init3 = np.array([y0, nu3[1], nu3[0]])
reshenie3 = runge_kutta(h, y_init3, f, 3)
k3 = trap(reshenie3[:, 1])
print('Импульс при m=%.1f: %.5f' % (3, k3))


def grafic(znachenie, m, name, x=n):
    """задает, но не рисует график профиля скорости при фикс. m"""
    plt.title(name)
    plt.ylabel(name)
    plt.grid('minor')
    return plt.plot(znachenie)

def area(znacheny, numberx):
    """вычислит площадь графика под функцией профиля скорости по значениям профиля
    скорости и количеству взятых точек по икс"""
    summ = 0
    for i in range (numberx-1):
        summ = summ + znacheny[i]
    area = (2*summ - znacheny[numberx-1] - znacheny[0]) * h/2 * math.sqrt(viscosity)
    return area

def kalibr(skorost):
    full_area = area(skorost, n)
    for i in range(1, n):
        area1 = area(skorost, i)
        if area1 >= area_procent * full_area:
            numberkalibr = i
            break
    kalibr = h * numberkalibr * math.sqrt(viscosity)
    return kalibr

hight = np.linspace(0, eta_max, n) * math.sqrt(viscosity)

plt.figure(figsize = (6.3,4.2))
plt.title("Профили продольной компоненты скорости на x=1м")
plt.plot(reshenie0[:, 1], hight, label='n = 0.5')
plt.plot(reshenie1[:, 1], hight, label='n = 1')
plt.plot(reshenie2[:, 1], hight, label='n = 2')
plt.plot(reshenie3[:, 1], hight, label='n = 3')
plt.grid('minor', ls=":", c="k")
plt.legend()
plt.ylabel('y, м')
plt.xlabel("$u$, м/с")
#plt.xlabel(r"$df/d\eta$")
plt.show()

plt.figure(figsize = (6.3,4.2))
plt.title("Автомодельные функции тока")
plt.plot(hight, reshenie0[:, 0], label='n = 0.5')
plt.plot(hight, reshenie1[:, 0], label='n = 1')
plt.plot(hight, reshenie2[:, 0], label='n = 2')
plt.plot(hight, reshenie3[:, 0], label='n = 3')
plt.grid('minor', ls=":", c="k")
plt.legend()
plt.ylabel(r"$f$", rotation=0)
#plt.ylabel(r"$f (\eta)$", rotation=0)
plt.xlabel(r'$\eta$')
plt.show()

print(alfa0)
print(nu1)
print(nu2)
print(nu3)

"""
print('area under profil 0.5 = {}'. format(area(reshenie0[:, 1], n)))
print('area under profil 1 = {}'. format(area(reshenie1[:, 1], n)))

kalibr0 = kalibr(reshenie0[:, 1])
kalibr1 = kalibr(reshenie1[:, 1])
kalibr2 = kalibr(reshenie2[:, 1])
kalibr3 = kalibr(reshenie3[:, 1])

print('калибр при m={}: {}'.format(1/2, kalibr0))
print('калибр при m={}: {}'.format(1, kalibr1))
print('калибр при m={}: {}'.format(2, kalibr2))
print('калибр при m={}: {}'.format(3, kalibr3))
"""

"""def write(m, array1, array2):
    file = open("profil{}.txt".format(m), 'w')
    for i in range(np.size(array1)):
        file.write('{},{},{}\n'.format(np.zeros(n)[i],array1[i], array2[i]))
    file.close()
    return(file)

write(0.5, hight, reshenie0[:, 1])
write(1, hight, reshenie1[:, 1])
write(2, hight, reshenie2[:, 1])
write(3, hight, reshenie3[:, 1])"""