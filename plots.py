import matplotlib.pyplot as plt
import numpy as np

# QASM simulator and fake vigo// 4 node 3 regular graph:
approx_ratio = [0.924825, 0.9994, 0.999375, 0.989525, 0.991425] # No noise
# [2.01177997 2.33217192 0.83523549 5.95600807 2.72784627] [1.65157151 6.27857961 1.05144207 3.10821409 1.4194117 ]
approx_ratio_dep1 = [0.904375, 0.9153, 0.8903, 0.872525, 0.867025] # Depolarizing error 0.01
approx_ratio_dep5 = [0.837725, 0.77485, 0.784075, 0.7585, 0.757125] # Depolarizing error 0.05
approx_ratio_fakevigo = [0.889739990234375, 0.87359619140625, 0.8653564453125, 0.84576416015625,0.845123291015625]
layer_by_layer_fakevigo = [0.89141845703125, 0.830657958984375, 0.748077392578125, 0.805877685546875, 0.784576416015625]
layer_by_layer_ideal =[0.926575, 0.9343, 0.78225, 0.9582, 0.957125]
real_vigo_nelder = [0.821563720703125, 0.77789306640625, 0.756439208984375, 0.759124755859375, 0.7598876953125]

#plt.plot(range(1,len(approx_ratio)+1), approx_ratio, 'o--', markersize=8, label='QASM ideal')
#plt.plot(range(1,len(approx_ratio_dep1)+1), approx_ratio_dep1, 'o--', markersize=8, label='Depolarizing error prob = 0.01')
#plt.plot(range(1,len(approx_ratio_dep5)+1), approx_ratio_dep5, 'o--', markersize=8, label='Depolarizing error prob = 0.05')
#plt.plot(range(1,len(approx_ratio_fakevigo)+1), approx_ratio_fakevigo, 'o--', markersize=8, label='FakeVigo')
plt.plot(range(1,len(layer_by_layer_fakevigo)+1), layer_by_layer_fakevigo, 'o--', markersize=8, label='FakeVigo- layer by layer')
plt.plot(range(1,len(layer_by_layer_ideal)+1), layer_by_layer_ideal, 'o--', markersize=8, label='QASM ideal- layer by layer')
plt.plot(range(1,len(real_vigo_nelder)+1), real_vigo_nelder, 'o--', markersize=8, label='IBM Vigo- layer by layer NelderMeald')

plt.xticks(range(1,len(approx_ratio)+1))
plt.xlabel('Number of layers p')
plt.ylabel('Approximation ratio')
plt.legend()
plt.title('4 node regular graph')
plt.savefig('plots/4 node regular graph with real vigo')
plt.show()

# QASM simulator with no noise // 6nprism:
approx_ratio = [0.7581571428571429, 0.7612285714285714, 0.7824428571428571, 0.7796000000000001, 0.7810142857142858]
# [5.84361619 3.19135334 3.7000231  0.01877684 1.6515655 ] [2.06240957 0.06109409 0.10826346 4.55452413 4.47434488]
approx_ratio_noisy2 = [0.7438857142857144, 0.7251000000000001, 0.7312428571428571, 0.7198142857142857, 0.7168714285714285] # Depolarizing error 0.01
approx_ratio_noisy = [0.6986857142857142, 0.6507857142857143, 0.6570428571428572, 0.6494714285714285, 0.6488571428571428] # Depolarizing error 0.05

plt.close()
plt.plot(range(1,len(approx_ratio)+1), approx_ratio, 'o--', markersize=8, label='Ideal')
plt.plot(range(1,len(approx_ratio_noisy2)+1), approx_ratio_noisy2, 'o--', markersize=8, label='Depolarizing error prob = 0.01')
plt.plot(range(1,len(approx_ratio_noisy)+1), approx_ratio_noisy, 'o--', markersize=8, label='Depolarizing error prob = 0.05')

plt.xticks(range(1,len(approx_ratio)+1))
plt.xlabel('Number of layers p')
plt.ylabel('Approximation ratio')
plt.legend()
plt.title('QASM simulator-6n prism graph')
#plt.savefig('6-n prism graph with qasm')
plt.show()

# QASM simulator with no noise // Erdos-Renyo graph:
approx_ratio = [0.7455928571428572, 0.6670857142857143, 0.7555071428571428, 0.755742857142857, 0.7561285714285715] # No noise
# [0.03821226 3.21017583 0.21517038 0.11873241 0.79274375] [2.89780661 3.14217104 4.21520146 0.11372762 2.89750008]
approx_ratio_noisy2 = [0.7270357142857142, 0.6614428571428571, 0.7125785714285715, 0.6606928571428572, 0.6618857142857143] # Depolarizing error 0.01
#[3.23163798 4.61422627 2.65065358 0.13752952 0.74402197] [0.93750292 3.05073663 5.16491242 0.75495762 4.70287073]
approx_ratio_noisy = [0.6845214285714285, 0.6456428571428571, 0.6503642857142857, 0.6476357142857143, 0.64885] # Depolarizing error 0.05
# [3.6446848  0.71288267 2.58491384 5.37518451 4.5581894 ] [4.6198482  5.05260994 4.50554244 3.30805374 2.19893019]
plt.close()

plt.plot(range(1,len(approx_ratio)+1), approx_ratio, 'o--', markersize=8, label='Ideal')
plt.plot(range(1,len(approx_ratio_noisy2)+1), approx_ratio_noisy2, 'o--', markersize=8, label='Depolarizing error prob = 0.01')
plt.plot(range(1,len(approx_ratio_noisy)+1), approx_ratio_noisy, 'o--', markersize=8, label='Depolarizing error prob = 0.05')

plt.xticks(range(1,len(approx_ratio)+1))
plt.xlabel('Number of layers p')
plt.ylabel('Approximation ratio')
plt.legend()
plt.title('QASM simulator-Erdos Renyi graph')
#plt.savefig('Erdos graph with qasm')
plt.show()

# Number of iterations of the optimizer to the circuit when running it in fakevigo 
#calls = [20,72,275,523,852] # maxiter = 999
#calls_layer_by_layer = [27,24,21,28,17]
calls_neldermead_standard = [403, 803, 1200, 1604, 2000]
calls_neldermead_laybylay= [402, 401, 400, 402, 401] 
calls_diffevol_standard = [891, 4520, 14534, 35754, 45491]
calls_diffevol_laybylay = [1413, 1355, 2356, 3603, 3231] # second time [903, 1725, 2217, 4026, 3158]
plt.close()
plt.plot(range(1,len(calls_neldermead_standard)+1), calls_neldermead_standard, 'bo--', markersize=8, label = 'NelderMeald standard')
plt.plot(range(1,len(calls_neldermead_laybylay)+1), calls_neldermead_laybylay, 'go--', markersize=8, label = 'NelderMeald layer by layer')
plt.plot(range(1,len(calls_diffevol_standard)+1), calls_diffevol_standard, 'ro--', markersize=8, label = 'DiffEvol standard')
plt.plot(range(1,len(calls_diffevol_laybylay)+1), calls_diffevol_laybylay, 'mo--', markersize=8, label = 'DiffEvol layer by layer')
plt.xticks(range(1,len(approx_ratio)+1))
plt.xlabel('Number of layers p')
plt.ylabel('Calls')
plt.legend()
plt.title('Calls to the objective function in FakeVigo')
plt.savefig('plots_and_graphs/Calls')
plt.show()

''' VIGO INA 4 NODES NELDER MEALD

Warning: Maximum number of function evaluations has been exceeded.
Number of layers: 1
Number of optimizer iterations: 146
Number of calls to the objective function: 402
Optimal gamma, optimal beta =  -0.6771347180850633 0.18742805876428043
The solution is the state [0, 0, 1, 1] with a cost value of 4.0
Expectation value of the cost function =  3.2862548828125
Approximation ratio =  0.821563720703125
Number of calls list: [402]
Average cost list: [3.2862548828125]
Approximation ratio list: [0.821563720703125]
-------------------------------------------------------------------

Warning: Maximum number of function evaluations has been exceeded.
Number of layers: 2
Number of optimizer iterations: 154
Number of calls to the objective function: 400
Optimal gamma, optimal beta =  -1.3378386878727655 -0.5695843454639812
The solution is the state [0, 0, 1, 1] with a cost value of 4.0
Expectation value of the cost function =  3.111572265625
Approximation ratio =  0.77789306640625
Number of calls list: [402, 400]
Average cost list: [3.2862548828125, 3.111572265625]
Approximation ratio list: [0.821563720703125, 0.77789306640625]
-------------------------------------------------------------------

Warning: Maximum number of function evaluations has been exceeded.
Number of layers: 3
Number of optimizer iterations: 146
Number of calls to the objective function: 401
Optimal gamma, optimal beta =  -0.910054556765747 0.03385530060269644
The solution is the state [0, 0, 1, 1] with a cost value of 4.0
Expectation value of the cost function =  3.0257568359375
Approximation ratio =  0.756439208984375
Number of calls list: [402, 400, 401]
Average cost list: [3.2862548828125, 3.111572265625, 3.0257568359375]
Approximation ratio list: [0.821563720703125, 0.77789306640625, 0.756439208984375]
-------------------------------------------------------------------

Warning: Maximum number of function evaluations has been exceeded.
Number of layers: 4
Number of optimizer iterations: 147
Number of calls to the objective function: 400
Optimal gamma, optimal beta =  -0.5028823452779423 0.14194215109627661
The solution is the state [0, 0, 1, 1] with a cost value of 4.0
Expectation value of the cost function =  3.0364990234375
Approximation ratio =  0.759124755859375
Number of calls list: [402, 400, 401, 400]
Average cost list: [3.2862548828125, 3.111572265625, 3.0257568359375, 3.0364990234375]
Approximation ratio list: [0.821563720703125, 0.77789306640625, 0.756439208984375, 0.759124755859375]
-------------------------------------------------------------------

Number of layers: 5
Number of optimizer iterations: 154
Number of calls to the objective function: 400
Optimal gamma, optimal beta =  -0.08440712920732096 -0.9275173555193875
The solution is the state [0, 0, 1, 1] with a cost value of 4.0
Expectation value of the cost function =  3.03955078125
Approximation ratio =  0.7598876953125
  
Number of calls list: [402, 400, 401, 400, 400]
Average cost list: [3.2862548828125, 3.111572265625, 3.0257568359375, 3.0364990234375, 3.03955078125]
Approximation ratio list: [0.821563720703125, 0.77789306640625, 0.756439208984375, 0.759124755859375, 0.7598876953125]

'''