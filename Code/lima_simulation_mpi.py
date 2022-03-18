import sys
import numpy as np
import pandas as pd
import json
from ast import literal_eval
from mpi4py import MPI
from greedy_hospital_tmp import add_temp_facility, greedy_tmp
from greedy_hospital import greedy, calculate_area, create_times

class NpEncoder(json.JSONEncoder):
	'''
	For JSON writing purposes. Converts numpy arrays and objects into ones that are serializable
	'''
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		if isinstance(obj, np.floating):
			return float(obj)
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return super(NpEncoder, self).default(obj)

def greedy_with_tmp_check(damaged_facilities, func_facilities, cons_time, f, svr, mu, prio = [1,2], avg_speed = 30, interval = 0.1, travel_time = None, dist_dict = None):
    
	all_facilities = np.concatenate([damaged_facilities, func_facilities])
	total_arrival_rate = sum([f[h][k] for h in all_facilities for k in prio])
	total_arrival_rate = sum([f[h][k] for h in all_facilities for k in prio])
	total_capacity = sum([val for key,val in svr.items() if key in func_facilities])*list(mu.values())[0]
	indicator_tmp = False # indicates if temporary facility is available
	tmp_facilities = []

	# check for capacity. If not enough, add a temporary facility
	if total_capacity - total_arrival_rate < 5:
		if dist_dict is not None:
			tmp_facilities, f, svr, mu, cons_time, dist_dict = add_temp_facility(total_capacity, total_arrival_rate, all_facilities, mu, f, svr, cons_time, 
		                                                                         dist_dict = dist_dict)
		elif travel_time is not None:
			tmp_facilities, f, svr, mu, cons_time, travel_time = add_temp_facility(total_capacity, total_arrival_rate, all_facilities, mu, f, svr, cons_time, 
		                                                                         travel_time = travel_time)                
		indicator_tmp = True
		# print('total capacity: {} total demand: {}. Add tmp facilities'.format(total_capacity, total_arrival_rate), flush = True)
		# print('Not enough capacity, add {} temporary facility'.format(len(tmp_facilities)))

	# Greedy Algorithm
	if indicator_tmp:
		greedy_order, results = greedy_tmp(damaged_facilities, func_facilities, tmp_facilities, cons_time, f, svr, mu, prio, avg_speed, interval, 
	                                       dist_dict = dist_dict, travel_time = travel_time)
		greedy_area = calculate_area(greedy_order, results['cost_total'], cons_time)
	else:
		greedy_order, results = greedy(damaged_facilities, func_facilities, cons_time, f, svr, mu, prio, avg_speed, interval, 
	                                  dist_dict = dist_dict, travel_time = travel_time)
		greedy_area = calculate_area(greedy_order, results['cost_total'], cons_time)

	# print('Greedy Order: {}'.format(greedy_order), flush = True)
	# print('Greedy Area: {}'.format(greedy_area), flush = True)
	# print('',flush = True)

	outputs = (greedy_order, greedy_area, results)

	return outputs

def worker_function(args):

	sim_split, data, H_all, svr, f, mu, travel_time, output_folder = args

	for i in sim_split:
		# get damaged buildings
		H = data.loc[data['Sim{}'.format(i)] != 0, 'ID'].values
		H0 = data.loc[data['Sim{}'.format(i)] == 0, 'ID'].values
		# print('Number of damaged buildings {} out of {}'.format(len(H), len(H_all)), flush = True)

		# get construction time
		hazus = {1: 10, 2: 45, 3: 180, 4: 360}
		T = {}
		for h in H_all:
			damage_level = data.loc[data['ID'] == h, 'Sim{}'.format(i)].values[0]
			if h in H:
				T[h] = hazus[damage_level] # using median values
			else:
				T[h] = 0
		        
		# run greedy
		try:
			outputs = greedy_with_tmp_check(H, H0, T, f, svr, mu, travel_time = travel_time)
			(greedy_order, greedy_area, results) = outputs

			times = create_times(greedy_order, T)

			if results['dist_dict'] is not None:
				dist_dict_json = {str(k):v for k, v in results['dist_dict'].items()} 
				travel_time_json = []
			elif results['travel_time'] is not None:
				dist_dict_json = []
				travel_time_json = {str(k):v for k, v in results['travel_time'].items()} 

			if 'tmp_facilities' in results.keys():
				tmp_facilities_json = results['tmp_facilities']
				results_wo_tmp = results['cost_total_wo_tmp']
			else:
				tmp_facilities_json = []
				results_wo_tmp = []


			# output to json
			final_results = {'greedy_order': greedy_order, 'greedy_area': greedy_area, 'results_cost': results['cost_total'], 'results_cost_wo_tmp': results_wo_tmp, 
							  'times': times, 'tmp_facilities': tmp_facilities_json, 'dist_dict': dist_dict_json, 'travel_time': travel_time_json}

			# Save output
			output_file = output_folder + 'result_{}.json'.format(i)
			with open(output_file, 'w') as file_output:
				json.dump(final_results, file_output, cls = NpEncoder)
		except:
			print('does not work {}'.format(i))
			output_file = output_folder + 'fail_{}.json'.format(i)
			with open(output_file, 'w') as file_output:
				json.dump(i, file_output, cls = NpEncoder)
			continue


if __name__ == '__main__':
	# [start simulation #] [end simulation #] [simulation csv] [hospital data] [travel_time] [output folder]
	# example: python3 lima_simulation_mpi.py 1 10 'lima_simulations.csv' 'lima_emergency_data.csv' 'travel_times.json' output/'

	# intialize multi-processing (and multi node)
	comm = MPI.COMM_WORLD
	size = comm.Get_size() # total number of cores/cpus
	rank = comm.Get_rank() # ID number for each core

	# master core
	if rank == 0:
		args = sys.argv[1:]

		# Simulation data
		sim_filepath = args[2]
		data = pd.read_csv(sim_filepath, index_col = 0)
		data = data.drop(columns = ['Name', 'hospital', 'district', 'lat', 'lon', 'Typology', 'Services'])

		# Hospital data
		hosp_filepath = args[3]
		hosp_data = pd.read_csv(hosp_filepath)

		# Combine data together
		data = hosp_data.merge(data, left_on = 'ID', right_on = 'ID',  how ='left')

		# Travel time
		travel_file = args[4]
		with open(travel_file) as json_file:
		    traveltime = json.load(json_file)
		    
		travel_time = {literal_eval(k): v for k, v in traveltime['travel_time'].items()} # hours

		# arrival rate, and capacity
		f = {}; svr = {}; mu = {}
		for i in range(len(data)):
			f[data['ID'].iloc[i]] = {1: data['priority1'].iloc[i], 2: data['priority2'].iloc[i] }
			svr[data['ID'].iloc[i]] = data['capacity'].iloc[i] # capacity
			mu[data['ID'].iloc[i]] = data['mu'].iloc[i] # service rate (/hr)

		# all facilities (set)
		H_all = data['ID'].values

		# number of simulations
		start_sim = int(args[0])
		end_sim = int(args[1])

		# output folder
		output_folder = args[5]

		# splits the work into the number of cores available
		sim_split = np.array_split(np.arange(start_sim,end_sim+1), size)
		data_list = [data for i in range(size)]
		H_all_list = [H_all for i in range(size)]
		f_list = [f for i in range(size)]
		svr_list = [svr for i in range(size)]
		mu_list = [mu for i in range(size)]
		travel_time_list = [travel_time for i in range(size)]
		output_folder_list = [output_folder for i in range(size)]
		work_list = list(zip(sim_split, data_list, H_all_list, svr_list, f_list, mu_list, travel_time_list, output_folder_list))
	else:
		work_list = None

	work_split = comm.scatter(work_list)
	print('This is CPU {} out of total {}'.format(rank+1, size), flush = True)
	worker_function(work_split)

