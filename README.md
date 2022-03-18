# Prioritized Reconstruction of Healthcare Facilities after Earthquakes based on Recovery of Emergency Services

This repository contains the code developed for the paper:

Prioritized Reconstruction of Healthcare Facilities after Earthquakes based on Recovery of Emergency Services. In Review. 

Please cite the paper if you use any data or analyses from this study. 

## Running the Code

- `Code/greedy_hospital.py`: Main Python file containing all functions to run the Greedy Algorithm
- `Code/greedy_hospital_tmp.py`: Main Python file containing all functions to run the Greedy Algorithm considering temporary facilities

### Test One Scenario
To obtain the reconstruction order of a single scenario, open `Code/lima_scenario.ipynb` and run the notebook. This Jupyter Notebook runs a single test scenario using data in `Data/lima_test.csv`.

### Section 4: Evaluation of Greedy Algorithm

To run the results from Section 4 of the paper (Evaluation of Greedy Algorithm), run `Code/greedy_evaluation_mpi.py`.
Example command for 6 damaged buildings and 12 total buildings: 
`python 3 greedy_evaluation_mpi.py 6 12 'output_folder/' 1` (1 represents simulation idx)

To run multiple simulations at the same time, run `Code/greedy_eval_local.sh` in the command line. 
Change parameters (number of damaged building, number of total building, and output folders) as needed. 

To visualize the results of the evaluation, run `Code/greedy_eval_visualize.ipynb`. Make sure to change the folder name containing the output folders from `greedy_eval_local.sh`.

### Section 6: Results and Discussion

To run the results found from Section 6 of the paper (Results and Discussion):
1. Run all 10,000 earthquake scenarios with `Code/lima_simulation_mpi.py`. 
Example command: `python3 lima_simulation_mpi.py 0 9999 'lima_simulations.csv' 'lima_emergency_data.csv' 'travel_times.json' output_folder/'`. 
2. Run the Jupyter Notebook `Code/lima_paper_results.ipynb`
