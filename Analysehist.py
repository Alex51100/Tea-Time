import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats




A = [1.1, 2.8, 3.033, 3.133, 3.966, 4.033, 4.133, 4.5, \
     4.566, 4.6, 4.666, 4.866, 4.966, 5, 5.966, 6.066, 6.1, 6.5, 6.533, 6.533, \
    6.7, 6.733, 6.733, 6.766, 6.766, 6.866, 7.033, 7.033, 7.1, 7.233, 7.466, 7.5, 7.666, 7.933, 8.066, 8.1, \
    8.133, 8.3, 8.6, 8.666, 8.9, 9, 9.1, 9.133, 9.2, 9.233, 9.5, 9.533, 9.6, 9.666, 9.666, 9.8, 10, \
    10.066, 10.366, 10.6, 10.8, 11.033, 11.433, 11.466, 12.066, 12.5, 13.966, 14.066, 14.6, 14.7, \
    15.166, 15.3, 15.766, 15.8, 15.833, 16.066, 16.816, 17.400, 17.566, 18.366, 19, 19.133, 22, 22.033, 24.266, 25.2,\
    25.333, 25.566, 26.066, 27, 27.2, 27.233, 28.9, 33.866, 35.366, 36.166, 44.1, 44.133, 48.066]
    
bins = 16


def fit_function(x, A, beta, B, mu, sigma):
    return (A * np.exp(-x/beta) + B * np.exp(-1.0 * (x - mu)**2 / (2 * sigma**2)))






données,  bins= np.histogram(A, bins = bins)
binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])

# z = np.array([-1])
# données = np.append(données, z)
# binscenters = np.append(binscenters, z)









popt, pcov = curve_fit(fit_function, xdata=binscenters, ydata=données)
xspace = np.linspace(0, 50, 1000)
#plt.bar(binscenters, données, width=bins[1] - bins[0], color='steelblue', label='Trous observés')
plt.plot(xspace, fit_function(xspace, *popt),'--', color='orangered', linewidth=2.5)







err = données/10 + 1/(données + 2)
plt.xlim(0,50)
plt.xlabel('Temps depuis le depos de la goutte (s)')
plt.ylabel('Nombre de trous')
plt.errorbar(binscenters, données, yerr=err, fmt='o', markersize = 5, capsize=2, color = 'navy')
plt.ylim([0, 30])
plt.grid()
plt.show()

# plt.savefig('Nb de trous.pdf')


