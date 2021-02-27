# QAOA Max-Cut

## Contributors

  + [Ignacio Fernandez Graña](https://github.com/inafergra)
  + [Smit Chaudhary](https://github.com/smitchaudhary)
  + [Luigi Pio Mastrodomenico](https://github.com/Aloisiu)


## Description of the Project

Our goal is to implement a Quantum Approximate Optimization Algorithm (QAOA) to solve the Max-Cut problem(an NP-hard problem). The Max-Cut problem can be briefly stated as follows: given an undirectional graph, partition the nodes into two sets such that the number of edges crossing these sets is maximized.

In order to address this problem we explored the QAOA approach given in the paper 'A quantum approximate optimization algorithm', 2014, [Farhi et al., 2014][1]. The main idea is to start with a maximal superposition state and apply a combination of gates in order to achieve a state that approximates the solution to the problem, i.e., maximizes the score function (the number of crossed edges). These gates depend on two parameters,γ and β, and they are applied `p` times to the initial state. As these `2p` parameters need to be tuned for the specific problem we are considering, we would like to study the behaviour of the algorithm as a function of these parameters, and to see the number of times `p` that is needed to apply these gates in order to achieve a certain accuracy. The optimal values of the parameters can be achieved, for example, using a variational quantum eigensolver, a quantum algorithm used to find the ground state of a given Hamiltonian.

## Backends

As we mentioned, for simulation, we plan to use IBM’s AER which is included in QISKIT to first run the simulations locally. Then to follow that up, for implementation on a real quantum computer, we plan to use IBM’s IBM-QExperience and utilize their superconducting-qubits based quantum computers through cloud to run the circuit on a noisy system. We also plan to use QuTech's Starmon 5 chip if possible.
