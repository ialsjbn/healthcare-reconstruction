
import numpy as np
from mpmath import nsum, inf, factorial
import copy
import itertools
from geopy import distance

def import_data(data):
    '''
    Imports the data from a pandas dataframe as variables for the greedy algorithm
    Must have columns: ID, Damage, priority1, priority2, capacity, mu, lat, lon

    Input: data: pandas dataframe with information
    '''

    # sets
    H = data.loc[data['Damage'] != 'None', 'ID'].values
    H0 = data.loc[data['Damage'] == 'None', 'ID'].values
    H_all = np.concatenate([H,H0])

    # arrival rate, and capacity
    f = {}; svr = {}; mu = {}
    for i in range(len(data)):
        f[data['ID'].iloc[i]] = {1: data['priority1'].iloc[i], 2: data['priority2'].iloc[i] }
        svr[data['ID'].iloc[i]] = data['capacity'].iloc[i] # capacity
        mu[data['ID'].iloc[i]] = data['mu'].iloc[i] # service rate (/hr)

    # calculate distance
    dist_dict = {}
    for pair in list(itertools.permutations(H_all, 2)):
        origin = (data.loc[data['ID'] == pair[0], 'lat'].values[0], data.loc[data['ID'] == pair[0], 'lon'].values[0])
        dest = (data.loc[data['ID'] == pair[1], 'lat'].values[0], data.loc[data['ID'] == pair[1], 'lon'].values[0])
        dist_dict[pair[0], pair[1]] = distance.distance(origin, dest).km
    for h in H_all:
        dist_dict[h,h] = 0

    # sample construction time
    hazus_med = {'Slight': 10, 'Moderate': 45, 'Complete': 180, 'Extensive': 360}
    hazus_beta = 0.4
    T = {}
    for h in H_all:
        damage_level = data.loc[data['ID'] == h, 'Damage'].values[0]
        if h in H:
            T[h] = hazus_med[damage_level] # using median values
            # T[h] = np.round(np.random.lognormal(np.log(hazus_med[damage_level]), hazus_beta),0)
        else:
            T[h] = 0

    return H, H0, H_all, f, svr, mu, dist_dict, T


def sort_functional_facilities(origin_facility, func_facilities, W_hk_curr, avg_speed, k, dist_dict = None, travel_time = None):
    '''
    Given the origin facility and priority class k, output a list of functional facilities
    in sorted order, from the facility with shortest total wait time (travel + wait) to the longest
    
    Input:
    origin_facility: origin facility ID
    func_facilities: a list of functional facilities
    dist_dict: a dictionary between pairs of facilities as the key and distance as the value
    W_hk_curr: dictionary with key: facility ID, and value: waiting time for facility h and priority level k
    avg_speed: average speed for travel
    k: priority class
    '''
    
    # get travel time
    if dist_dict is not None:
        total_time = {pair[1]:dist/avg_speed for pair,dist in dist_dict.items() if ((pair[0] == origin_facility) 
                                                                      and (pair[1] in func_facilities))}
    elif travel_time is not None:
                total_time = {pair[1]:time for pair,time in travel_time.items() if ((pair[0] == origin_facility) 
                                                                      and (pair[1] in func_facilities))}

    # add waiting time
    for h in func_facilities:
        total_time[h] += W_hk_curr[h,k]
    
    # sort by total waiting time
    sorted_facility = [h[0] for h in sorted(total_time.items(), key = lambda value: value[1])]
    
    return sorted_facility

def calculate_waiting_time_k(s, mu, rho_k, k):
    '''
    Calculate the average waiting time of the kth class for 
    non-preemptive M/M/s system with equal service time for all classes
    (Cobham 1954)
    
    Input:
    s: number of servers for the facility
    mu: service rate
    rho: utilization rate for each class k
    k: priority class k
    '''
    rho = sum(rho_k)
    rho_bar = np.zeros(k+1)
    rho_bar[1:] = [np.sum(rho_k[:j+1]) for j in range(k)]

    if rho < 0.99: # equation only works when not overloaded
        num = (rho*s)**s
        denom = factorial(s)*(1-rho)*(nsum(lambda j: (rho*s)**j/factorial(j), [0,s-1]) + 
         nsum(lambda j: (rho*s)**j/(factorial(s)*(s**(j-s))), [s, inf]))
        pi = num/denom
    else:
        pi = 0.99
        rho_bar[-1] = 0.99 # prevents division by zero
    
    w_k = pi/(s*mu*(1-rho_bar[k])* (1-rho_bar[k-1]))
    
    return float(w_k)

def initial_allocation(damaged_facilities, func_facilities, allocation, lam_hk_curr, rho_hk_curr, 
                       W_hk_curr, lam_h_curr, rho_h_curr, f, svr, mu, prio, avg_speed, interval, dist_dict = None, travel_time = None):
    '''
    Given the functional facilities, reallocate patient demand to the nearest functional facilities
    making sure waiting time and utilization ratio does not exceed threshold.

    Input:
    damaged_facilities: array of damaged facilities
    func_facilities: array of functional facilities
    dist_dict: a dictionary between pairs of facilities as the key and distance as the value
    allocation: dictionary with key: original facility, and value: current facility allocation
    lam_hk_curr: dictionary with key: facility ID, and value: total arrival rate for facility h and priority level k
    rho_hk_curr: dictionary with key: facility ID, and value: total utilization ratio for facility h and priority level k
    W_hk_curr: dictionary with key: facility ID, and value: waiting time for facility h and priority level k
    lam_h_curr: dictionary with key: facility ID, and value: total arrival rate of facility h (all priorities)
    rho_h_curr: dictionary with key: facility ID, and value: total utilization ratio for facility h (all priorities)
    f: dictionary for demand arrival rate for facility h and priority level k
    svr: dictionary for number of servers for facility h
    mu: dictionary for service rate of facility h
    prio: list with priority levels
    avg_speed: average speed for travel
    interval:for purposes of patient allocation (how detailed partial allocation is):
    
    '''
    
    # initialize cost
    all_facilities = np.concatenate([damaged_facilities, func_facilities])
    
    # initial utilization check
    total_arrival_rate = sum([f[h][k] for h in all_facilities for k in prio])
    total_capacity = sum([val for key,val in svr.items() if key in func_facilities])*list(mu.values())[0]
    if total_arrival_rate > total_capacity:
        print('Not enough capacity!')
        print('total capacity', total_capacity)
        print('total demand', total_arrival_rate)
        print()
        raise ValueError('Not enough capacity!')
    
    # demands at each damaged facility needs to be reallocated
    # demand is reallocated per 0.1 rate
    for d in damaged_facilities: 
        # damaged facilities cannot handle any demand
        lam_h_curr[d] = 0; rho_h_curr[d] = 0

        for k in prio:
            potential_facilities = sort_functional_facilities(d, func_facilities, W_hk_curr, avg_speed, k, 
                                                              dist_dict = dist_dict, travel_time = travel_time)
            
            # damaged facilities cannot handle any demand
            lam_hk_curr[d,k] = 0; rho_hk_curr[d,k] = 0
            
            # for each interval arrival rate, allows for partial allocation
            for p in range(int(f[d][k]*(1/interval))): 
                for i in potential_facilities:
                    # update arrival rate and utilization
                    tmp_lam_hk = lam_hk_curr[i,k] + interval
                    tmp_rho_hk = rho_hk_curr.copy()
                    tmp_rho_hk[i,k] = tmp_lam_hk/(svr[i]*mu[i])
                    tmp_rho_h = sum(tmp_rho_hk[i,p] for p in prio)

                    # utilization ratio 
                    # actually assign to the facility
                    if tmp_rho_h < 0.99:
                        # update variables
                        lam_hk_curr[i,k] = tmp_lam_hk
                        rho_hk_curr[i,k] = tmp_rho_hk[i,k]
                        lam_h_curr[i] += interval
                        rho_h_curr[i] = tmp_rho_h

                        # update allocation
                        allocation[i,k].append(d) # d move to i
                        allocation[d,k].remove(d) # d move from d

                        break # no need to check for other facility

    # update waiting time
    for h in all_facilities:
        for k in prio: # need to update waiting time for all priorities
            W_hk_curr[h,k] = calculate_waiting_time_k(svr[h], mu[h], [tmp_rho_hk[h,j] for j in prio],k)

    # Check if all patients have been allocated (steady-state condition)
    for d in damaged_facilities:
        for k in prio:
            if len(allocation[d,k]) > 0:
                print('Not enough capacity!')
                print('total capacity', total_capacity)
                print('total demand', total_arrival_rate)
                print('Utilization of each facility:')
                print(rho_h_curr)
                print()
                raise ValueError('Not enough capacity!')

    output = (lam_hk_curr, rho_hk_curr, W_hk_curr, lam_h_curr, rho_h_curr, allocation)
                
    return output

    
def reallocate_patients(to_reconstruct, all_facilities, curr_damaged, allocation, 
                        lam_hk, rho_hk, W_hk, lam_h, rho_h, f, svr, mu, prio, avg_speed, interval, dist_dict = None, travel_time = None):
    '''
    Given a facility that is reconstructed, reallocate any demand such that the average total time is reduced
    (travel + waiting time)

    Input:
    to_reconstruct: facility ID to be reconstructed
    all_facilities: array of all facilities in region
    curr_damaged: array with currently damaged facilities
    dist_dict: a dictionary between pairs of facilities as the key and distance as the value
    allocation: dictionary with key: original facility, and value: current facility allocation
    lam_hk: dictionary with key: facility ID, and value: total arrival rate for facility h and priority level k
    rho_hk: dictionary with key: facility ID, and value: total utilization ratio for facility h and priority level k
    W_hk: dictionary with key: facility ID, and value: waiting time for facility h and priority level k
    lam_h: dictionary with key: facility ID, and value: total arrival rate of facility h (all priorities)
    rho_h: dictionary with key: facility ID, and value: total utilization ratio for facility h (all priorities)
    f: dictionary for demand arrival rate for facility h and priority level k
    svr: dictionary for number of servers for facility h
    mu: dictionary for service rate of facility h
    prio: list with priority levels
    avg_speed: average speed for travel
    interval:for purposes of patient allocation (how detailed partial allocation is)
    '''
    
    # make sure the original dicts are not modified
    lam_hk_curr = copy.deepcopy(lam_hk)
    rho_hk_curr = copy.deepcopy(rho_hk)
    W_hk_curr = copy.deepcopy(W_hk)
    lam_h_curr = copy.deepcopy(lam_h)
    rho_h_curr = copy.deepcopy(rho_h)
    allocation_curr = copy.deepcopy(allocation)

    # move demand back to the original facility that is currently being reconstructed
    facilities_modified = set()
    for k in prio:
        # get allocation for the specific priority
        alloc_prio = {key:val for key,val in allocation_curr.items() if key[1] == k}

        # get where each demand originally from to_reconstruct is currently located at
        list_to_move_from = []
        for key in alloc_prio.keys():
            for v in alloc_prio[key]:
                if v == to_reconstruct:
                    list_to_move_from.append(key[0])
            facilities_modified.add(key[0])

        # move one by one to original facility
        for curr_facility in list_to_move_from:
            lam_hk_curr[curr_facility,k] -= interval
            lam_hk_curr[to_reconstruct,k] += interval
            rho_hk_curr[curr_facility,k] = lam_hk_curr[curr_facility,k]/(mu[curr_facility]*svr[curr_facility])
            rho_hk_curr[to_reconstruct,k] = lam_hk_curr[to_reconstruct,k]/(mu[to_reconstruct]*svr[to_reconstruct])

            allocation_curr[curr_facility,k].remove(to_reconstruct)
            allocation_curr[to_reconstruct, k].append(to_reconstruct)

    lam_h_curr[curr_facility] = sum(lam_hk_curr[curr_facility,k] for k in prio)
    lam_h_curr[to_reconstruct] = sum(lam_hk_curr[to_reconstruct,k] for k in prio)
    rho_h_curr[curr_facility] = sum(rho_hk_curr[curr_facility,k] for k in prio)
    rho_h_curr[to_reconstruct] = sum(rho_hk_curr[to_reconstruct,k] for k in prio)

    # update waiting time (update only after everything is moved back)
    for k in prio:
        W_hk_curr[to_reconstruct, k] = calculate_waiting_time_k(svr[to_reconstruct], mu[to_reconstruct], [rho_hk_curr[to_reconstruct,j] for j in prio],k)
        for h in facilities_modified:
            W_hk_curr[h, k] = calculate_waiting_time_k(svr[h], mu[h], [rho_hk_curr[h,j] for j in prio],k)

    output = (lam_hk_curr, rho_hk_curr, W_hk_curr, lam_h_curr, rho_h_curr, allocation_curr)
        
    return output

def initialize_model(damaged_facilities, func_facilities, f, svr, mu, prio, avg_speed, interval, dist_dict = None, travel_time = None):
    '''
    Initializes the model by calculating waiting times and allocating patients to the nearest functional facilities. 
    Intializes the cost variables and counts demand in temporary facilities

    Input:
    damaged_facilities: array of damaged facilities
    func_facilities: array of functional facilities
    dist_dict: a dictionary between pairs of facilities as the key and distance as the value
    f: dictionary for demand arrival rate for facility h and priority level k
    svr: dictionary for number of servers for facility h
    mu: dictionary for service rate of facility h
    prio: list with priority levels
    avg_speed: average speed for travel
    interval:for purposes of patient allocation (how detailed partial allocation is):
    '''
    # Initialize arrays
    all_facilities = np.concatenate([damaged_facilities, func_facilities])
    num_damaged = len(damaged_facilities)

    lam_hk = {}
    W_hk = {}
    rho_hk = {}
    lam_h = {}
    rho_h = {}
    allocation = {(h,k): [] for h in all_facilities for k in prio} # key is the facility, and the value is the demand that is allocated to the facility

    # Calculate initial waiting times and utilization
    for h in all_facilities:
        for k in prio:
            lam_hk[h,k] = f[h][k]
            rho_hk[h,k] = lam_hk[h,k]/(mu[h]*svr[h])
            demands = [h for i in range(int(f[h][k]*(1/interval)))]
            allocation[h,k] = allocation[h,k] + demands
        for k in prio: # needs to calculate all rho_hk first
            W_hk[h,k] = calculate_waiting_time_k(svr[h], mu[h], [rho_hk[h,j] for j in prio],k)
        lam_h[h] = sum(lam_hk[h,k] for k in prio)
        rho_h[h] = sum(rho_hk[h,k] for k in prio)


    # Reallocation to the nearest functional facility
    initial_alloc_output = initial_allocation(damaged_facilities, func_facilities, allocation, lam_hk, rho_hk, W_hk, lam_h, rho_h, 
                                               f, svr, mu, prio, avg_speed, interval, dist_dict = dist_dict, travel_time = travel_time)
    (lam_hk_curr, rho_hk_curr, W_hk_curr, lam_h_curr, rho_h_curr, allocation) = initial_alloc_output
    
    outputs = (all_facilities, num_damaged, lam_hk, rho_hk, W_hk, lam_h, rho_h, allocation)
        
    return outputs

def calculate_avg_total_time(allocation_curr, W_hk_curr, avg_speed, prio, dist_dict = None, travel_time = None):
    '''
    Calculates the average total time (travel + wait at facility) for all the patients in the region. 
    Essentially the objective function of the model. 

    Input:
    allocation_curr: dictionary with key: original facility, and value: current facility allocation
    W_hk_curr: dictionary with key: facility ID, and value: waiting time for facility h and priority level k
    dist_dict: a dictionary between pairs of facilities as the key and distance as the value
    prio: list with priority levels
    avg_speed: average speed for travel
    '''

    # get list of demands (originally located)
    list_from = [val for key,val in allocation_curr.items()]
    list_from = np.array([item for sublist in list_from for item in sublist]) # flatten

    # get where each demand is currently located at
    list_to_move_from = []
    for key in allocation_curr.keys():
        list_to_move_from = list_to_move_from + [key[0] for v in allocation_curr[key]]
    list_to_move_from = np.array(list_to_move_from)

    # calculate waiting time
    if dist_dict is not None:
        curr_totaltime_all = [W_hk_curr[list_to_move_from[i],k] + dist_dict[list_from[i],list_to_move_from[i]]/avg_speed for i in range(len(list_to_move_from))
                             for k in prio]
    elif travel_time is not None:
        curr_totaltime_all = [W_hk_curr[list_to_move_from[i],k] + travel_time[list_from[i],list_to_move_from[i]] for i in range(len(list_to_move_from))
                             for k in prio]
    curr_totaltime_all = np.array(curr_totaltime_all)

    mean_wait = np.mean(curr_totaltime_all)

    return mean_wait

def greedy(damaged_facilities, func_facilities, cons_time, f, svr, mu, prio, avg_speed, interval = 0.1, dist_dict = None, travel_time = None):
    '''
    Greedy algorithm to determine the optimal construction ordering based on the gittins index
    (maximum potential cost reduction). Outputs an array of the construction order and detailed results. 

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
    interval:for purposes of patient allocation (how detailed partial allocation is):
    '''
    print('Intializing Arrays... Calculating Initial Allocation')
    initial_output = initialize_model(damaged_facilities, func_facilities, f, svr, mu, prio, avg_speed, interval,
                                      dist_dict = dist_dict, travel_time = travel_time)
    (all_facilities, num_damaged, lam_hk, rho_hk, W_hk, lam_h, rho_h, allocation) = initial_output

    # calculate total avg wait time
    cost_total = np.zeros(num_damaged + 1)
    cost_total[0] = calculate_avg_total_time(allocation, W_hk, avg_speed, prio, dist_dict = dist_dict, travel_time = travel_time)
    # print(cost_total[0])

    # save allocation and waiting time  
    allocation_time = []
    allocation_time.append(allocation)
    W_hk_time = []
    W_hk_time.append(W_hk)

    # Construction Ordering
    print('Calculating Calculation Ordering...')
    construction_order = []
    curr_damaged = damaged_facilities.copy()

    for d in range(num_damaged):
        # calculate gittins index
        potential_gittins = {}
        potential_cost_total = {}
        test = {}
        for i in curr_damaged:
            tmp_output = reallocate_patients(i, all_facilities, curr_damaged, allocation, lam_hk, rho_hk, W_hk, lam_h, 
                                             rho_h, f, svr, mu, prio, avg_speed, interval, dist_dict = dist_dict, travel_time = travel_time)
            (_, _, W_hk_tmp, _, _, alloc_tmp) = tmp_output

            # calculate potential cost
            potential_cost_total[i] = calculate_avg_total_time(alloc_tmp, W_hk_tmp, avg_speed, prio, dist_dict = dist_dict, travel_time = travel_time)
            potential_gittins[i] = (cost_total[d] - potential_cost_total[i])/cons_time[i]
        # print(potential_cost_total)
        # print(potential_gittins) 

        # choose facility with highest gittins index
        chosen_facility = max(potential_gittins, key = potential_gittins.get)
        print('chosen facility', chosen_facility)
        construction_order.append(chosen_facility)

        # update values
        output_realloc = reallocate_patients(chosen_facility, all_facilities, curr_damaged, allocation, 
                                             lam_hk, rho_hk, W_hk, lam_h, rho_h, f, svr, mu, prio, avg_speed, interval,
                                              dist_dict = dist_dict, travel_time = travel_time)
        (lam_hk, rho_hk, W_hk, lam_h, rho_h, allocation) = output_realloc

        # update cost
        cost_total[d+1] = calculate_avg_total_time(allocation, W_hk, avg_speed, prio, dist_dict = dist_dict, travel_time = travel_time)

        curr_damaged = curr_damaged[curr_damaged != chosen_facility]
        allocation_time.append(allocation)
        W_hk_time.append(W_hk)

    print('Construction Order:', construction_order)

    detailed_results = {'order': construction_order, 'W_hk_time': W_hk_time, 'allocation_time': allocation_time, 'cost_total': cost_total,
                        'travel_time': travel_time, 'dist_dict': dist_dict}
    
    return construction_order, detailed_results
    

def manual_order(order, damaged_facilities, func_facilities, f, svr, mu, prio, avg_speed, interval = 0.1, dist_dict = None, travel_time = None):
    '''
    Given the construction ordering, calculate the allocation of patient demand and total cost (distance + waiting time).
    Outputs the detailed results

    Input:
    order: list of the construction ordering
    damaged_facilities: array of damaged facilities
    func_facilities: array of functional facilities
    dist_dict: a dictionary between pairs of facilities as the key and distance as the value
    f: dictionary for demand arrival rate for facility h and priority level k
    svr: dictionary for number of servers for facility h
    mu: dictionary for service rate of facility h
    prio: list with priority levels
    avg_speed: average speed for travel
    interval:for purposes of patient allocation (how detailed partial allocation is):
    '''

    # Initialize arrays   
    initial_output = initialize_model(damaged_facilities, func_facilities, f, svr, mu, prio, avg_speed, interval, 
                                      dist_dict = dist_dict, travel_time = travel_time)
    (all_facilities, num_damaged, lam_hk, rho_hk, W_hk, lam_h, rho_h, allocation) = initial_output

    # calculate total avg wait time
    cost_total = np.zeros(num_damaged + 1)
    cost_total[0] = calculate_avg_total_time(allocation, W_hk, avg_speed, prio, dist_dict = dist_dict, travel_time = travel_time)

    # save allocation    
    allocation_time = []
    allocation_time.append(allocation)
    W_hk_time = []
    W_hk_time.append(W_hk)

    # Construction Ordering
    # print('Calculating Costs based on Order...')
    curr_damaged = damaged_facilities.copy()

    for d in range(len(order)):
        output_realloc = reallocate_patients(order[d], all_facilities, curr_damaged, allocation, 
                                             lam_hk, rho_hk, W_hk, lam_h, rho_h, f, svr, mu, prio, avg_speed, interval,
                                             dist_dict = dist_dict, travel_time = travel_time)
        (lam_hk, rho_hk, W_hk, lam_h, rho_h, allocation) = output_realloc

        # update cost
        cost_total[d+1] = calculate_avg_total_time(allocation, W_hk, avg_speed, prio, dist_dict = dist_dict, travel_time = travel_time)
        
        curr_damaged = curr_damaged[curr_damaged != order[d]]
        allocation_time.append(allocation)
        W_hk_time.append(W_hk)
    
    detailed_results = {'order': order, 'W_hk_time': W_hk_time, 'allocation_time': allocation_time, 'cost_total': cost_total, 
                        'travel_time': travel_time, 'dist_dict': dist_dict}
    
    return detailed_results

def create_times(order, cons_time):
    '''
    Creates a list of times when a construction finishes, assuming that 
    there is only ONE reconstruction crew

    Input: 
    order: the construction ordering
    cons_time: dictionary with key: facility ID, and value: construction time
    '''
    x_time = np.zeros(len(order)+1)
    for i in range(1, len(x_time)):
        x_time[i] = x_time[i-1] + cons_time[order[i-1]]

    return x_time


def calculate_area(order, costs, cons_time, x_time = [], unit = 'day'):
    '''
    Calculates the area under the curve of costs given the construction ordering

    Input:
    order: list with construction order
    costs: list with cost at every construction finish
    times: list of times when every construction finishes. If only one construction crew, no need to pass in. 
    cons_time: dictionary with key: facility ID, and value: construction time
    unit: outputs value in day units or year units
    '''

    if len(x_time) == 0: # empty list
      x_time = create_times(order, cons_time)

    if unit == 'day':
      return np.trapz(costs, x_time)
    if unit == 'year':
      return np.trapz(costs, x_time/365)

def find_optimal(damaged_facilities, func_facilities, cons_time, f, svr, mu, prio, avg_speed, interval = 0.1, dist_dict = None, travel_time = None):
    '''
    Finds the optimal solution through manual permutation of all possible combination. 

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
    interval:for purposes of patient allocation (how detailed partial allocation is):
    '''

    all_orders = list(itertools.permutations(damaged_facilities))
    best_area = 100000000000

    for order in all_orders:
      results = manual_order(order, damaged_facilities, func_facilities, f, svr, mu, prio, avg_speed, interval, dist_dict = dist_dict, travel_time = travel_time)
      area = calculate_area(order, results['cost_total'], cons_time)

      if area < best_area:
        optimal_order = order
        best_area = area


    print('Optimal Order: ', optimal_order)
    print('Area:' , best_area)

    return optimal_order, best_area

def get_cons_order(cons_time, damaged_facilities):
    '''
    Outputs an ordering of damaged facilities starting with the fastest construction
    (ascending order)
    
    Input:
    cons_time: dictionary with key: facility ID, and value: construction time (days)
    damaged_facilities: array of damaged facilities
    '''
    cons_order = [key for key,val in sorted(cons_time.items(), key = lambda value: value[1]) if key in damaged_facilities]
    return cons_order

def get_capacity_order(svr, damaged_facilities):
    '''
    Outputs an ordering of damaged facilities starting with the most servers
    (descending order)
    
    Input:
    svr: dictionary for number of servers for facility h
    damaged_facilities: array of damaged facilities
    '''
    capacity_order = [key for key,val in sorted(svr.items(), key = lambda value: value[1], reverse = True) if key in damaged_facilities]
    return capacity_order

def get_demand_order(f, prio, damaged_facilities):
    '''
    Outputs an ordering of damaged facilities starting with the most demand
    (descending order)
    
    Input:
    f: dictionary for demand arrival rate for facility h and priority level k
    damaged_facilities: array of damaged facilities
    prio: list with priority levels
    '''
    d = {}
    for h in damaged_facilities:
        d[h] = sum(f[h][k] for k in prio)

    demand_order = [key for key,val in sorted(d.items(), key = lambda value: value[1], reverse = True) if key in damaged_facilities]
    
    return demand_order

def patient_allocation(origin_facility, result):
    patient_allocation = []
    for i in range(len(result['allocation_time'])):
        time_step = result['allocation_time'][i]
        time_allocation = set()
        for key in time_step.keys():
            if origin_facility in time_step[key]:
                time_allocation.add(key[0])
        patient_allocation.append(time_allocation)

    return patient_allocation

def wait_disagg(output_option, result, avg_speed = 30, k = None, dist_dict = None, travel_time = None):
    '''
    Function to output the mean total waiting time over the recovery period. 
    Can be disaggregated by travel time, wait time and total time. 
    Can also be disaggregated by priority level

    Input:
    output_option: either 'total', dist', 'wait' to indicate total time, travel time or wait time at facility
    result: the results dict from the greedy algorithm output
    dist_dict: a dictionary between pairs of facilities as the key and distance as the value
    avg_speed: average speed for travel
    k: if not defined, then for all priority. else disaggregated by priority level
    '''
    
    def helper_function(j, k = None):
        if k is None:
            alloc = result['allocation_time'][j]
        else:
            alloc = {key:val for key,val in result['allocation_time'][j].items() if key[1] == k}

        # get list of demands (originally located)
        list_from = [val for key,val in alloc.items()]
        list_from = np.array([item for sublist in list_from for item in sublist]) # flatten

        # get where each demand is currently located at
        list_to_move_from = []
        for key in alloc.keys():
            list_to_move_from = list_to_move_from + [key[0] for v in alloc[key]]
        list_to_move_from = np.array(list_to_move_from)
        
        return list_from, list_to_move_from

    # initialize
    mean_wait = np.zeros(len(result['allocation_time']))
    
    for j in range(len(result['allocation_time'])):
        # get priority levels
        prio = np.unique([key[1] for key in result['allocation_time'][0].keys()])
        
        # wait time at current time step
        W_hk_curr = result['W_hk_time'][j]
        
        if k is None:
            list_from, list_to_move_from = helper_function(j)
            
            # calculate waiting time
            wait_time = np.array([W_hk_curr[list_to_move_from[i],kk] for i in range(len(list_to_move_from)) for kk in prio])

            if dist_dict is not None:
                dist_time = np.array([dist_dict[list_from[i],list_to_move_from[i]]/avg_speed for i in range(len(list_to_move_from)) for kk in prio])
            elif travel_time is not None:
                dist_time = np.array([travel_time[list_from[i],list_to_move_from[i]] for i in range(len(list_to_move_from)) for kk in prio])
            
            total_time = dist_time + wait_time

        else:
            list_from, list_to_move_from = helper_function(j, k = k)
            
            # calculate waiting time
            wait_time = np.array([W_hk_curr[list_to_move_from[i],k] for i in range(len(list_to_move_from))])

            if dist_dict is not None:
                dist_time = np.array([dist_dict[list_from[i],list_to_move_from[i]]/avg_speed for i in range(len(list_to_move_from))])
            elif travel_time is not None:
                dist_time = np.array([travel_time[list_from[i],list_to_move_from[i]] for i in range(len(list_to_move_from))])
            
            total_time = dist_time + wait_time
        
        if output_option == 'total':
            mean_wait[j] = np.mean(total_time)
        elif output_option == 'dist':
            mean_wait[j] = np.mean(dist_time)
        elif output_option == 'wait':
            mean_wait[j] = np.mean(wait_time)

    return mean_wait