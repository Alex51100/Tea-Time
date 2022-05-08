import matplotlib.pyplot as plt
import numpy as np
import statistics
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from matplotlib import animation


a=0.03 #m
b=0.4e-3   #m
alpha = 4e-3   #m
gamma = 20e-3   #N/m
m = 0.03e-3   #kg
x = np.linspace(0, a, 100)
area = np.pi*alpha**2

def longueur(x, a, b):
    return 2*b*np.sqrt(1-(x**2/a**2))

def L(x, a, b, alpha):
    return abs(longueur(x-alpha, a, b) - longueur(x+alpha, a, b))

def f(X, t,  a, b, alpha, gamma, m):
    x, xdot = X
    return [xdot, (gamma/m)*L(x, a, b, alpha)]

def accéleration(x, a, b, alpha, gamma, m):
    return (gamma/m)*L(x, a, b, alpha)

t = np.linspace(0, 10, 10000) #10000
x0, xdot0 = 0.0001, 0
X0 = np.array([x0, xdot0])

sol = odeint(f, X0, t, args=(a, b, alpha, gamma, m))
x, xdot = sol.T




A = []
for i in range(0, len(t)):
    A.append(a)
    


# plt.figure()
# plt.plot(t, x, label='position')
# plt.plot(t, xdot, label='vitesse', color='r')
# plt.plot(t, A, '--')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(t, xdot, label='vitesse', color='r')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(x, accéleration(x, a, b, alpha, gamma, m), label='accélération')
# plt.legend()
# plt.show()

limites = np.array([2*a, 2*a])
figure = plt.figure(figsize=(5,5))
# plt.xlim(-limites[0],limites[0])
# plt.ylim(-limites[1],limites[1])
# plt.grid()

O = []
nb = len(xdot)
for i in range(0, nb):
    O.append(0) 


positions = np.array([x0, x0])
vitesses = np.array([O, -xdot])





# plt.plot(positions[0], positions[1] , 'o')

ax = plt.axes(xlim=(-limites[0], limites[0]), ylim=(-limites[1], limites[1]))
ax.set_facecolor('lightskyblue')
# scatter = ax.scatter(positions[0], positions[1], marker='o',  color = "w", edgecolor='w', lw=a)




circle1 = plt.Circle((0, 0), a, color='gold')
# ax.add_patch(circle1)


j=0
while x[j]<a:
    j+=1


E_cinetique = 1/2*m*xdot[j-1]**2

Surface_gouttelette = E_cinetique/gamma
rayon_gouttelette = np.sqrt((Surface_gouttelette/(2*np.pi)))
# print(Surface_gouttelette )
print(vitesses)

T = 0

def nouveau(positions, vitesses):
    global T
    T += 1
    positions += vitesses[:, T]

    


def animate(frame):
    plt.cla()
    nouveau(positions, vitesses)
    ax.add_patch(circle1)
    ax.scatter(positions[0], positions[1], marker='o',  color = "lightskyblue",s=250, zorder=2)
    ax.set_xlim(-limites[0],limites[0])
    ax.set_ylim(-limites[1],limites[1])
    ax.set_xlabel('m')
    ax.set_ylabel('m')
    # if positions[0]>a/np.sqrt(2):
    #     circle2 = plt.Circle((a/np.sqrt(2)+a/10, a/np.sqrt(2)+a/10),alpha , color='gold', zorder=1)
    #     ax.add_patch(circle2)
    

anim = animation.FuncAnimation(figure, animate, frames = 60, interval = 1, blit=False)





