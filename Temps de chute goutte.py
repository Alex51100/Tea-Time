import matplotlib.pyplot as plt
import numpy as np
import statistics
from scipy.optimize import curve_fit
from scipy.integrate import odeint


L = 0.225
l = 0.5e-3

V = np.arange(0.1, 2.6, 0.1)*1e-9
R = ((3/(4*np.pi))*V)**(1/3)

T = np.array([250, 150, 114, 107, 96, 93, 90, 86, 86, 82, 75, 78, 72, 70, 67, 67, 65, 64, 62, 62, 60, 61, 58, 58, 55])

t = (T*l)/L

def fit(x, a):
    return a/(x**2)

popt, pcov = curve_fit(fit, R, t)
x = np.linspace(0.000015, 0.001, 1000)

plt.figure()
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.plot(R, t, 'o')
plt.plot(x, fit(x, *popt), '--', color='orangered', label=r'$f(x) = \frac{\alpha}{x^2}$')
plt.grid()
plt.ylim(-0.2, 2)
plt.xlim(0.000015, 0.0009)
plt.xlabel('Rayon (m)')
plt.ylabel('Temps de chute (s)')
plt.legend(prop={'size': 12})
plt.show()

print(l*(1/popt))

# plt.savefig('chute goutte.pdf')