# bayes-linear-odes

This repo contains the code used to generate the example presented in Section 5 of the manuscript 'Bayes Linear Analysis for Ordinary Differential Equations', available at http://www.lancs.ac.uk/~jonathan/BysLnrOde19.pdf . The example is based on an ordinary differential equation (ODE) model for the trajectory of a projectile launched from a given point with uncertain initial velocity and drag coefficient. The code fits a preliminary model for the numerical discrepancy, generates prior expectations and covariances for the projectile positions through numerical simulation, and computes a posterior moment specification for the trajectory positions and parameters through Bayes linear adjustment on a graphical model. All code is written for Matlab (R2019a), and is provided under the GPL-3.0 license (see [LICENSE.md](LICENSE.md) for details).

## Getting Started

Run the script 'generateFigs.m' to generate the plots shown in Figure 4 of the article:
* Code Figure 1 = manuscript Figure 4a: shows the prior trajectory moments (means and variances), alongside the true trajectories, observations and posterior moments for adjustments at a range of different parameter settings.
* Code Figure 2 = manuscript Figure 4b: shows prior and adjusted moments for the parameters 'gam' and 'du0' for the same adjustments as shown in Figure 4a.

## Code Details

There are three main classes which implement the analysis outlined in the manuscript- high-level details are provided below.

### Solver

Class for generating trajectories from both the real and numerical solutions to the underlying ODE (see Section 5 of the manuscript).\
Important functions:
* `true_trajectory()`: evaluate the true ODE solution trajectory at the input times and parameter settings.
* `simulate_trajetory()`: generate a set of simulations from the (first-order Euler) numerical solution to the ODE, with random numerical discrepancy contributions (if required).
* `gen_data()`: generate a data set for use in fitting the numerical discrepancy model, using the procedure outlined in Section 4.1 and Appendix D of the manuscript.
* `fit_discrepancy_model()`: obtain Bayes linear adjusted moments for the parameters and residual process of the numerical discrepancy model (Equation 7 of the manuscript).

### DAG

Class which stores details of a second-order uncertainty specification on a directed acyclic graph (DAG), and contains functionality for conversion from this representation to a junction tree representation.\
Important functions:
* `AdjacencyMatrix()`: creates the (undirected) adjacency matrix associated with the DAG.
* `CovarianceNumerical()`: generates covariances corresponding to edges in the DAG from the samples supplied in the construction of the DAG object.
* `Moralize()`: computes the adjacency matrix for the moral graph associated with the DAG specification.

### JunctionTree

Class to store properties of a second-order specification on an undirected graphical model, store details of the associated junction tree structure, and update moments by propagation of an adjustment through the cliques of the junction tree (as detailed in Appendix B of the manuscript).\
Important functions:
* `makeJTfromDAG()`: convert the DAG specification (as stored on a DAG class object) into a junction tree specification.
* `sequentialAdjustment()`: adjust the moments stored on the junction tree given observations on particular nodes, by propagating the adjustment sequentially through the junction tree (see Appendix B of the manuscript for details).

## Authors

* **Matthew Jones** - [mjj89](https://github.com/mjj89)
* **Philip Jonathan** 
* **Michael Goldstein**
* **David Randell**

## License

The code is made available under the GPL-3.0 License - see the [LICENSE.md](LICENSE.md) file for details