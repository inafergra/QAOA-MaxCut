import matplotlib.pyplot as plt
import numpy as np

# QASM simulator with no noise // 4 node 3 regular graph:
approx_ratio = [0.925425,0.99855,0.999625,0.989375,0.99245] # No noise
approx_ratio_noisy2 = [0.904375, 0.9153, 0.8903, 0.872525, 0.867025] # Depolarizing error 0.01
approx_ratio_noisy = [0.837725, 0.77485, 0.784075, 0.7585, 0.757125] # Depolarizing error 0.05

plt.plot(range(1,len(approx_ratio)+1), approx_ratio, 'bo--', markersize=8, label='No noise')
plt.plot(range(1,len(approx_ratio_noisy2)+1), approx_ratio_noisy2, 'ro--', markersize=8, label='Depolarizing error prob = 0.01')
plt.plot(range(1,len(approx_ratio_noisy)+1), approx_ratio_noisy, 'go--', markersize=8, label='Depolarizing error prob = 0.05')

plt.xticks(range(1,len(approx_ratio)+1))
plt.xlabel('Number of layers p')
plt.ylabel('Approximation ratio')
plt.legend()
plt.title('QASM simulator-4 node regular graph')
#plt.savefig('4 node regular graph with qasm')
#plt.show()

# QASM simulator with no noise // 4 node 3 regular graph:
approx_ratio = [0.7585714285714286, 0.7602714285714286, 0.7851, 0.7834714285714286, 0.7804142857142857] # No noise
approx_ratio_noisy2 = [0.7438857142857144, 0.7251000000000001, 0.7312428571428571, 0.7198142857142857, 0.7168714285714285] # Depolarizing error 0.01
approx_ratio_noisy = [0.6986857142857142, 0.6507857142857143, 0.6570428571428572, 0.6494714285714285, 0.6488571428571428] # Depolarizing error 0.05

plt.close()
plt.plot(range(1,len(approx_ratio)+1), approx_ratio, 'bo--', markersize=8, label='No noise')
plt.plot(range(1,len(approx_ratio_noisy2)+1), approx_ratio_noisy2, 'ro--', markersize=8, label='Depolarizing error prob = 0.01')
plt.plot(range(1,len(approx_ratio_noisy)+1), approx_ratio_noisy, 'go--', markersize=8, label='Depolarizing error prob = 0.05')

plt.xticks(range(1,len(approx_ratio)+1))
plt.xlabel('Number of layers p')
plt.ylabel('Approximation ratio')
plt.legend()
plt.title('QASM simulator-6n prism graph')
plt.savefig('6-n prism graph with qasm')
plt.show()
