import matplotlib.pyplot as plt
import numpy as np

# QASM simulator with no noise:
approx_ratio = [0.925425,0.99855,0.999625,0.989375,0.99245] # No noise
approx_ratio_noisy = [0.837725, 0.77485, 0.784075, 0.7585, 0.757125] 
approx_ratio_noisy2 = [0.900225, 0.916975, 0.88925]

plt.plot(range(1,len(approx_ratio)+1), approx_ratio, 'bo--', markersize=8, label='No noise')
plt.plot(range(1,len(approx_ratio_noisy2)+1), approx_ratio_noisy2, 'ro--', markersize=8, label='Depolarizing error prob = 0.01')
plt.plot(range(1,len(approx_ratio_noisy)+1), approx_ratio_noisy, 'go--', markersize=8, label='Depolarizing error prob = 0.05')

plt.xticks(range(1,len(approx_ratio)+1))
plt.xlabel('Number of layers p')
plt.ylabel('Approximation ratio')
plt.legend()
plt.title('QASM simulator')
plt.show()