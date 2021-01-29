# QAOA Max-Cut

## Contributors

  + [Ignacio Fernandez Graña](https://github.com/inafergra)
  + [Smit Chaudhary](https://github.com/smitchaudhary)
  + [Luigi Pio Mastrodomenico](https://github.com/Aloisiu)


## Description of the Project

Our goal is to implement a Quantum Approximate Optimization Algorithm (QAOA) to solve the Max-Cut problem(an NP-hard problem). The Max-Cut problem can be briefly stated as follows: given an undirectional graph, partition the nodes into two sets such that the number of edges crossing these sets is maximized.

In order to address this problem we would like to explore the QAOA approach given in the paperAquantum approximate optimization algorithm 2014, [Farhi et al., 2014][1]. The main idea here is to start with a maximal superposition state and apply a combination of gates in order to achieve a state that approximates the solution to the problem, i.e., maximizes the score function (the number of crossed edges). These gates depend on two parameters,γ and β, and they are applied `p` times to the initial state. As these `2p` parameters need to be tuned for the specific problem we are considering, we would like to study the behaviour of the algorithm as a function of these parameters, and to see the number of times `p` that is needed to apply these gates in order to achieve a certain accuracy. The optimal values of the parameters can be achieved, for example, using a variational quantum eigensolver, a quantum algorithm used to find the ground state of a given Hamiltonian.

Lastly, in order to measure the success of our implementation, our idea is to follow the approach given in another paper [Coles et al., 2018][2], pages 41-45, where three types of Max-Cut computations are proposed as a metric performance for the algorithm. Firstly, we would run a simulation of the algorithm in a classical computer to be sure of the mathematical correctness of the implementation. Second, we would implement the algorithm on real quantum hardware, such as the 5-qubit IBMQX4 IBM computer, to study the feasibility of the algorithm when exposed to noise and decoherence. Lastly, we would implement a random computation that assigns each node to one of the two graph partitions randomly, which will be used to make sure the hardware implementation performs better than what one would expect by chance from pure noise.

## Backends

As we mentioned, for simulation, we plan to use IBM’s AER which is included in QISKIT to first run the simulations locally. Then to follow that up, for implementation on a real quantum computer, we plan to use IBM’s IBM-QExperience and utilize their superconducting-qubits based quantum computers through cloud to run the circuit on a noisy system.

## References
[1] [Coles, P., Eidenbenz, S., Pakin, S., Adedoyin, A., Ambrosiano, J., Anisimov, P., Casper, W.,Chennupati, G., Coffrin, C., Djidjev, H., Gunter, D., Karra, S., Lemons, N., Lin, S., Lokhov, A., Malyzhenkov,A., Mascarenas, D., Mniszewski, S., Nadiga, B., and Zhu, W. (2018). Quantum algorithm implementations forbeginners.](https://ui.adsabs.harvard.edu/abs/2018arXiv180403719A/abstract)

[2] [Farhi, E., Goldstone, J., and Gutmann, S. (2014). A quantum approximate optimization algorithm](https://arxiv.org/abs/1411.4028v1)