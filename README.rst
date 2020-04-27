The code structure:

basics.basics:
Provide the most basic functions and parameters

fourdvel.fourdvel(basics):
Provide the essential functionality for inversion and data analysis

Reading parameters

Preparation (loading preprocessed data, find data availability)

Linear inversion functionality 
(need to be isolated into the linear solver)
Form design matrix
Form model prior and data error prior matrix
Perform inversion

Result conversion
functionality for converting the obtained results to meaningful outputs

configure.configure(fourdvel):

Give the inverse problem logistics: 
Load the corresponding data and form data vector, either from synthetic data or real data.

tacks.tasks(configure):

Perform different inversion tasks (linear, nonlinear, etc)
Load the data
Form the design matrix

driver_fourdvel.driver_fourdvel(tasks): 
The perform the operations on all tiles


simulation.simulation(basics):
Generate synthetic data
I am unaware of fourdvel

analysis.analysis(configure):

Load all the results and perform analysis 

solvers.Bayesian_Linear(fourdvel)
Need to expanded

solvers.Bayesian_MCMC(fourdvel)
Hamiltonian Monte Carlo solver the non-linear inversion

display.display(fourdvel)
Show the results for a single point

output.output(fourdvel)
Output the results to XYZ file for GMT to plot


