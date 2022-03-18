import sys
import numpy as np
import random
import itertools
import math
from mpi4py import MPI
import json
from greedy_hospital import greedy, manual_order, calculate_area
from greedy_hospital_tmp import greedy_tmp, manual_order_tmp, add_temp_facility


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

def generate_problem(num_damaged, total_buildings):
    '''
    Generates a single problem (region) with damaged buildings and total buildings. 
    Generates the location, size and capacity of facility, and patient demand

    Input:
    num_damaged: number of damaged buildings
    total_buildings: total number of buildings in the region

    '''
    H = np.arange(num_damaged)
    H0 = np.arange(num_damaged, total_buildings)
    H_all = np.concatenate([H, H0])

    # initialize dict
    T = {}; svr = {}; f = {}; mu = {}; lat = {}; lon = {}

    for h in H_all:
        if h in H:
            med = random.choice([10,45,180,360])
            T[h] = np.round(np.random.lognormal(np.log(med), 0.4),0)
        else:
            T[h] = 0
            
        svr[h] = int(np.random.uniform(1,50))
        mu[h] = 0.3
        
        max_lam = svr[h]*mu[h]*0.9
        total_lam = np.random.uniform(0.1, max_lam)
        f[h] = {1: np.ceil((0.1*total_lam)*10)/10, 2: np.ceil((0.8*total_lam)*10)/10}
    
        lat[h] = np.round(np.random.uniform(0,30),2)
        lon[h] = np.round(np.random.uniform(0,30),2)

    # calculate distance
    DIST = {}
    for pair in list(itertools.permutations(H_all, 2)):
        DIST[pair[0], pair[1]] = np.round(math.sqrt((lat[pair[0]]-lat[pair[1]])**2 + (lon[pair[0]]-lon[pair[1]])**2),2)
    for h in H_all:
        DIST[h,h] = 0
    
    return H, H0, H_all, T, svr, f, mu, lat, lon, DIST


def evaluate_greedy(damaged_facilities, func_facilities, cons_time, dist_dict, f, svr, mu, prio = [1,2], avg_speed = 30, interval = 0.1):
    '''
    From the generated problem, checks if the region has enough capacity to handle all the demand. If not enough, 
    then add temporary facility. Then, calculates the greedy ordering. 

    Input:
    damaged_facilities: array of damaged facilities
    func_facilities: array of functional facilities
    dist_dict: a dictionary between pairs of facilities as the key and distance as the value
    cons_time: dictionary with key: facility ID, and value: construction time (days)
    f: dictionary for demand arrival rate for facility h and priority level k
    svr: dictionary for number of servers for facility h
    mu: dictionary for service rate of facility h
    prio: list with priority levels
    avg_speed: average speed for travel
    interval: for purposes of patient allocation (how detailed partial allocation is)
    '''


    all_facilities = np.concatenate([damaged_facilities, func_facilities])
    total_arrival_rate = sum([f[h][k] for h in all_facilities for k in prio])
    total_capacity = sum([val for key,val in svr.items() if key in func_facilities])*list(mu.values())[0]
    indicator_tmp = False # indicates if temporary facility is available
    tmp_facilities = []

    # check for capacity. If not enough, add a temporary facility
    if total_capacity - total_arrival_rate < 0.5:
        tmp_facilities, f, svr, mu, cons_time, dist_dict = add_temp_facility(total_capacity, total_arrival_rate, all_facilities, mu, f, svr, 
                                                                                cons_time, dist_dict = dist_dict)
        indicator_tmp = True
        # print('total capacity: {} total demand: {}'.format(total_capacity, total_arrival_rate))
        # print('Not enough capacity, add {} temporary facility'.format(len(tmp_facilities)))

    # Greedy Algorithm
    if indicator_tmp:
        greedy_order, results = greedy_tmp(damaged_facilities, func_facilities, tmp_facilities, cons_time, f, svr, mu, prio, avg_speed, interval, dist_dict = dist_dict)
        greedy_area = calculate_area(greedy_order, results['cost_total_wo_tmp'], cons_time)
    else:
        greedy_order, results = greedy(damaged_facilities, func_facilities, cons_time, f, svr, mu, prio, avg_speed, interval, dist_dict = dist_dict)
        greedy_area = calculate_area(greedy_order, results['cost_total'], cons_time)

    print('Greedy Order: {}'.format(greedy_order), flush = True)
    print('Greedy Area: {}'.format(greedy_area), flush = True)

    # sys.stdout.write('Greedy Order: {}'.format(greedy_order))
    # sys.stdout.write('\n')
    # sys.stdout.write('Greedy Area: {}'.format(greedy_area))
    # sys.stdout.write('\n')
    # sys.stdout.flush()

    return greedy_order, greedy_area, tmp_facilities, f, svr, mu, cons_time, dist_dict, indicator_tmp

def output(output_file, result_list, damaged_facilities, func_facilities, indicator_tmp, dist_dict, f, svr, mu, cons_time, tmp_facilities, greedy_order, greedy_area):
    '''
    Function to combine the results obtained from all the workers, and finds the otpimal order and area. 
    Writes the results and problem generated to a JSON file in the output_file

    Input:
    output_file: output file to write the results
    damaged_facilities: array of damaged facilities
    func_facilities: array of functional facilities
    indicator_tmp: bool to indicate whether temporary facility is used or not. 
    dist_dict: a dictionary between pairs of facilities as the key and distance as the value
    f: dictionary for demand arrival rate for facility h and priority level k
    svr: dictionary for number of servers for facility h
    mu: dictionary for service rate of facility h
    cons_time: dictionary with key: facility ID, and value: construction time (days)
    tmp_facilities: array of temporary facilities. if there is no, then empty list []
    greedy_order: the list of ordering obtained from greedy
    greedy_area: the area obtained from greedy

    '''


    # Put results together
    all_areas = [pair[1] for pair in results_list]
    all_orders = [pair[0] for pair in results_list]

    all_areas = [item for sublist in all_areas for item in sublist]
    all_orders = [item for sublist in all_orders for item in sublist]
    optimal_order = all_orders[np.argmin(all_areas)]
    optimal_area = all_areas[np.argmin(all_areas)]
    print('Optimal Order: {}'.format(optimal_order), flush = True)
    print('Optimal Area: {}'.format(optimal_area), flush = True)
    # sys.stdout.write('Optimal Order: {}'.format(optimal_order))
    # sys.stdout.write('\n')
    # sys.stdout.write('Optimal Area: {}'.format(optimal_area))
    # sys.stdout.write('\n')
    # sys.stdout.flush()

    # Formatting for JSON
    dist_dict_json = {str(k):v for k, v in dist_dict.items()} 
    f_json = {int(k):v for k, v in f.items()} 
    svr_json = {int(k):v for k, v in svr.items()} 
    mu_json = {int(k):v for k, v in mu.items()} 
    cons_time_json = {int(k):v for k, v in cons_time.items()} 

    final_results = {'indicator_tmp': indicator_tmp, 'greedy_order': greedy_order, 'greedy_area': greedy_area, 
                      'optimal_order': optimal_order, 'optimal_area': optimal_area, 
                      'damaged_facilities': damaged_facilities, 'func_facilities': func_facilities, 'tmp_facilities': tmp_facilities, 
                      'dist_dict': dist_dict_json, 'f': f_json, 'svr': svr_json, 'mu': mu_json, 'cons_time': cons_time_json}

    # Save output
    with open(output_file, 'w') as file_output:
        json.dump(final_results, file_output, cls = NpEncoder)

def worker_function(args, prio = [1,2], avg_speed = 30, interval = 0.1):
    '''
    Function for individual workers/cores to calculate the area given the manual ordering. 

    Input:
    args: order_list, damaged_facilities, func_facilities, all_facilities, cons_time, svr, f, mu, dist_dict, tmp_facilities
    prio: list with priority levels
    avg_speed: average speed for travel
    interval: for purposes of patient allocation (how detailed partial allocation is)
    '''

    order_list, damaged_facilities, func_facilities, all_facilities, cons_time, svr, f, mu, dist_dict, tmp_facilities = args

    orders_list = []
    order_areas = []
    for order in order_list:
        if len(tmp_facilities) > 0:
            results = manual_order_tmp(order, damaged_facilities, func_facilities, tmp_facilities, f, svr, mu, prio, avg_speed, interval, dist_dict = dist_dict)
            final_area = calculate_area(results['order'], results['cost_total_wo_tmp'], cons_time)
        else:
            results = manual_order(order, damaged_facilities, func_facilities, f, svr, mu, prio, avg_speed, interval, dist_dict = dist_dict)
            final_area = calculate_area(results['order'], results['cost_total'], cons_time)

        orders_list.append(order)
        order_areas.append(final_area)

    return (orders_list, order_areas)

if __name__ == '__main__':
    # args: num_damaged, total_buildings, output_folder, file suffix 
    # example: python3 greedy_evaluation.py 3 5 'output/' 1

    # intialize multi-processing (and multi node)
    comm = MPI.COMM_WORLD
    size = comm.Get_size() # total number of cores/cpus
    rank = comm.Get_rank() # ID number for each core

    # master core
    if rank == 0:
        args = sys.argv[1:]
        num_damaged = int(args[0])
        total_buildings = int(args[1])
        output_folder = args[2]
        i   = int(args[3])
        output_file = output_folder + 'result_{}.json'.format(i)
        print('Total size {} and rank {}'.format(size, rank), flush = True)
        # sys.stdout.write('Total size {} and rank {}'.format(size, rank))
        # sys.stdout.write('\n')
        # sys.stdout.flush()

        # Run evaluation
        H, H0, H_all, T, svr, f, mu, lat, lon, DIST = generate_problem(num_damaged, total_buildings)
        greedy_order, greedy_area, tmp_id, f, svr, mu, T, DIST, indicator_tmp = evaluate_greedy(H,H0, T, DIST, f, svr, mu)

        all_orders = np.array(list(itertools.permutations(H))) # generate the order
        print('Total number of ordering: {}'.format(len(all_orders)), flush = True)
        # sys.stdout.write('Total number of ordering: {}'.format(len(all_orders)))
        # sys.stdout.write('\n')

        # splits the orders into the number of cores available
        all_order_split = np.array_split(all_orders, size)
        H_list = [H for i in range(size)]
        H0_list = [H0 for i in range(size)]
        H_all_list = [H_all for i in range(size)]
        T_list = [T for i in range(size)]
        f_list = [f for i in range(size)]
        svr_list = [svr for i in range(size)]
        mu_list = [mu for i in range(size)]
        DIST_list = [DIST for i in range(size)]
        tmp_id_list = [tmp_id for i in range(size)]
        work_list = list(zip(all_order_split, H_list, H0_list, H_all_list, T_list, svr_list, f_list, mu_list, DIST_list, tmp_id_list))
    else:
        work_list = None

    work_split = comm.scatter(work_list)
    print('This is CPU {} out of total {}'.format(rank, size), flush = True)
    # sys.stdout.write('This is CPU {} out of total {}'.format(rank, size))
    # sys.stdout.write('\n')
    # sys.stdout.flush()
    results_split = worker_function(work_split)
    
    # Combine all results
    results_list = comm.gather(results_split)

    if rank == 0:
        output(output_file, results_list, H, H0, indicator_tmp, DIST, f, svr, mu, T, tmp_id, greedy_order, greedy_area)


