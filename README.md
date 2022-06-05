# fourDvel2 
Routines for inferring time-dependent, 3d displacement fields from displacement data derived from temporally-dense synthetic-apeature radar observations
fourDvel2 is an extention of fourDvel to allow inferring ephemeral grounding on ice shelves

## Requirement
- numpy
- scipy
- multiprocessing
- matplotlib
- pickle


## Routines
- basics:
Provide the most basic functions and parameters

- fourdvel:
Provide the essential functionality for inversion and data analysis
1. Read parameters
2. Load displacement data into memory
3. Linear inversion functionality: construct design matrix, model prior and data error prior; Parameter estimtion
4. Convert results into different formats

- configure:
Construc inverse problem data vector either from synthetic data or real data

- estimate:
1. Perform different estimatie/inversion tasks: linear inversion, nonlinear inversion, etc
2. Construct design matrix

- driver_fourdvel:
The driver the of fourDvel which multithreads the task

- simulation:
Generate synthetic data

- analysis
Load results and perform analysis

- solvers
Solvers for nonlinear inverse problem

- display
Display the results

- output
Output the results as XYZ files (lon, lat, value)
