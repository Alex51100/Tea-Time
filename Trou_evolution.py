import matplotlib.pyplot as plt
import numpy as np
import statistics
from scipy.optimize import curve_fit
from scipy.integrate import odeint


a=0.025 #m
b=0.4e-3   #m
alpha = 5e-3   #m
gamma = 20e-3   #N/m
m = (950*alpha**2*np.pi*2*b)*400  #kg



def longueur(x, a, b):
    return 2*b*np.sqrt(1-(x**2/a**2))


def L(x, a, b, alpha):
    return abs(longueur(x-alpha, a, b) - longueur(x+alpha, a, b))

def f(X, t, a, b, alpha, gamma, m):
    x, xdot = X
    return [xdot, (gamma/m)*L(x, a, b, alpha)]

def accéleration(x, a, b, alpha, gamma, m):
    return (gamma/m)*L(x, a, b, alpha)

t = np.linspace(0, 40, 100)
x0, xdot0 = 0.0001, 0.0001
X0 = np.array([x0, xdot0])

sol = odeint(f, X0, t, args=(a, b, alpha, gamma, m))
x, xdot = sol.T

# def fit(t, A, K):
#     return A*np.exp(K*t)

# popt, pcov = curve_fit(fit, t, x)



# fig, axs = plt.subplots(2, 1, figsize=(10,10))




A = []
for i in range(0, len(t)):
    A.append(a)
    


plt.figure()
# plt.plot(t, fit(t, *popt), 'x')
plt.plot(t, x, label='Modèle théorique')
# plt.plot(t, xdot, label='vitesse', color='r')
plt.plot(t, A, '--')

plt.grid()

# position_1 = np.array([0.024328, 0.024328, 0.024328, 0.024328, 0.024328, 0.024328, 0.024328, 0.024328, 0.024328, 0.024328, \
#                      0.024328, 0.024485, 0.024637, 0.024950, 0.025221, 0.025695, 0.026546, 0.026887, 0.027449, 0.028036, \
#                      0.028658, 0.029364, 0.030270, 0.031063, 0.031554, 0.032257, 0.033236, 0.034163, 0.034910, 0.036397, \
#                      0.037502, 0.038575, 0.039912, 0.041399, 0.042747, 0.044133, 0.045766]) - 0.024328

position_1 = np.array([0.024328, 0.024328, 0.024328, 0.024328, 0.024328,  \
                     0.024328, 0.024637, 0.025221, 0.026546, 0.027449, \
                     0.028658, 0.030270, 0.031554, 0.033236,  0.034910, \
                     0.037502, 0.039912, 0.042747, 0.045766]) - 0.024328
    
    
    
temps_1 = np.arange(0, 38, 2)

# position_2 = np.array([0.0000353, 0.0000353, 0.0000353, 0.0000353, 0.0003265, 0.0004854, 0.0008915, 0.0013594, 0.0017212, 0.0018537, \
#                        0.0021891, 0.0025510, 0.0028158, 0.0028158, 0.0031160, 0.0034514, 0.0037780, 0.0041398, 0.0047665, 0.0049960, \
#                        0.0051195, 0.0053843, 0.0058079, 0.0063286, 0.0071848, 0.0072819, 0.0076348, 0.0082879, 0.0092676, 0.0100177, \
#                        0.0109003, 0.0117475, 0.0127183, 0.0136008, 0.0150041, 0.0160808, 0.0173870, 0.0188167, 0.0203434, 0.0216760, \
#                        0.0231675, 0.0247121, 0.0259740, 0.0269447, 0.0272713, 0.0274654, 0.0282331]) - 0.0000353

position_2 = np.array([0.0000353, 0.0000353, 0.0003265,  0.0008915,  0.0017212, \
                       0.0021891,  0.0028158,  0.0031160,  0.0037780,  0.0047665,  \
                       0.0051195,  0.0058079,  0.0071848,  0.0076348, 0.0092676,  \
                       0.0109003,  0.0127183,  0.0150041,  0.0173870,  0.0203434 ]) - 0.0000353 
temps_2 = np.arange(0, 40, 2)




plt.plot(temps_1, position_1, 'x', label='Première migration')
plt.plot(temps_2, position_2, 'v', label='Deuxième migration')
plt.xlabel('Durée écoulée (s)')
plt.ylabel("Distance au centre de la nappe d'huile (m)")



plt.legend()
plt.show()




# plt.figure()
# plt.plot(t, xdot, label='vitesse', color='r')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(x, accéleration(x, a, b, alpha, gamma, m), label='accélération')
# plt.legend()
# plt.show()


# plt.savefig('Migration_graphe.pdf')



