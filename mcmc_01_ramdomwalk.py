import numpy as np
import matplotlib.pyplot as plt

class randomwalk_mcmc():
    def __init__(self,burn_in = 1000,sampling=100000):
        self.burn_in_term = burn_in
        self.sampling_term = sampling


    def __call__(self,mu=4.0,sigma=0.1):
        def gamma(x,alpha=11,l=13):
            return np.exp(-l*x)*(x**(alpha-1))
        # initialize
        theta = 4.0
        self.thetas_burnin = []
        self.thetas = []
        # calc detailed balance condition
        for i in range(self.burn_in_term):
            #print(sigma)
            a = theta + np.random.normal()
            if not a < 0:
                right_term = gamma(theta)
                left_term = gamma(a)
                if right_term > left_term:# True
                    r = left_term / right_term
                    if np.random.binomial(1,r,1):
                        theta = a
                else: # False
                    theta = a
            self.thetas_burnin.append(theta)
        for i in range(self.sampling_term - self.burn_in_term):
            a = theta + np.random.normal()
            #print(a)
            if not a < 0:
                right_term = gamma(theta)
                left_term = gamma(a)
                if right_term >= left_term:# True
                    r = left_term / right_term
                    if np.random.binomial(1,r,1)[0]:
                        theta = a
                else: # False
                    theta = a
            self.thetas.append(theta)
        return 0


plt.figure()
np.random.seed(1)
thetas = np.arange(0,2.5,0.01)
density = []
l = 13
alpha = 11
C = 3628800 #calced by CASIO.
for theta in thetas:
    density.append(((l**alpha)/C) * theta**(alpha-1) * np.exp(-l*theta))
density = np.array(density)

r_mcmc = randomwalk_mcmc(burn_in=100,sampling=1000000)
r_mcmc()
plt.hist(r_mcmc.thetas, bins=500,normed=True)
plt.plot(thetas,density)
plt.xlim(0,2.5)
plt.figure()
plt.plot(r_mcmc.thetas_burnin + r_mcmc.thetas)
from statistics import mean, stdev
print("theoritical average:{}".format(alpha / l))
print("theoritical sd:{}".format(np.sqrt(alpha/(l**2))))
print("average:{}".format(mean(r_mcmc.thetas)))
#print("sd:{}".format(stdev(i_mcmc.thetas)))
print("sd:{}".format(stdev(r_mcmc.thetas)))
print("sd EAP:{}".format(mean(np.sqrt(r_mcmc.thetas))))
print("sd sd:{}".format(stdev(np.sqrt(r_mcmc.thetas))))
print("skewness EAP:{}".format(mean(1/np.sqrt(r_mcmc.thetas))))
print("skewness sd:{}".format(stdev(1/np.sqrt(r_mcmc.thetas))))
print("kurtosis EAP:{}".format(mean(3*np.ones(len(r_mcmc.thetas))+(1/np.sqrt(r_mcmc.thetas)))))
print("kurtosis sd:{}".format(stdev(3*np.ones(len(r_mcmc.thetas))+(1/np.sqrt(r_mcmc.thetas)))))
plt.show()
