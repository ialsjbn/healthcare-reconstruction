import numpy as np
import copy
from greedy_hospital import sort_functional_facilities, calculate_waiting_time_k

def initialize_model(damaged_facilities, func_facilities, tmp_facilities, f, svr, mu, prio, avg_speed, interval, dist_dict = None, travel_time = None):
    '''
    Initializes the model by calculating waiting times and allocating patients to the nearest functional facilities. Includes temporary facilities
    Intializes the cost variables and counts demand in temporary facilities

    Input:
    damaged_facilities: array of damaged facilities
    func_facilities: array of functional facilities
    tmp_facilities: array of temporary facilities
    dist_dict: a dictionary between pairs of facilities as the key and distance as the value
    f: dictionary for demand arrival rate for facility h and priority level k
    svr: dictionary for number of servers for facility h
    mu: dictionary for service rate of facility h
    prio: list with priority levels
    avg_speed: average speed for travel
    interval:for purposes of patient allocation (how detailed partial allocation is):
    '''

    # Initialize arrays
    all_facilities = np.concatenate([damaged_facilities, func_facilities, tmp_facilities])
    original_facilities = np.concatenate([damaged_facilities, func_facilities])
    avail_facilities = np.concatenate([func_facilities, tmp_facilities])
    num_damaged = len(damaged_facilities)
    num_in_tmp = np.ones(num_damaged+1) # number of patients in temporary facility
    
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
            if h in tmp_facilities:
                W_hk[h,k] = 10000  # waiting time in temporary facility is some large number
            else:
                W_hk[h,k] = calculate_waiting_time_k(svr[h], mu[h], [rho_hk[h,j] for j in prio],k)
        lam_h[h] = sum(lam_hk[h,k] for k in prio)
        rho_h[h] = sum(rho_hk[h,k] for k in prio)
    # print('original waiting time')
    # print(W_hk)
    # print('original utilizzation')
    # print(rho_h)
    # print(rho_hk)

    # Reallocation to the nearest functional facility
    initial_alloc_output = initial_allocation(damaged_facilities, func_facilities, tmp_facilities, 
                                               allocation, lam_hk, rho_hk, W_hk, lam_h, rho_h, 
                                                f, svr, mu, prio, avg_speed, interval, dist_dict = dist_dict, travel_time = travel_time)
    (lam_hk_curr, rho_hk_curr, W_hk_curr, lam_h_curr, rho_h_curr, allocation) = initial_alloc_output
    
    # Count demand allocated to temporary facility
    num_in_tmp[0] = sum(len(vals) for key, vals in allocation.items() if key[0] in tmp_facilities)*interval
    
    outputs = (all_facilities, original_facilities, num_damaged, lam_hk, rho_hk, W_hk, lam_h, rho_h, allocation, num_in_tmp)
        
    return outputs

def initial_allocation(damaged_facilities, func_facilities, tmp_facilities, allocation, lam_hk_curr, rho_hk_curr, 
                       W_hk_curr, lam_h_curr, rho_h_curr, f, svr, mu, prio, avg_speed, interval, dist_dict = None, travel_time = None):
    '''
    Given the functional facilities, reallocate patient demand to the nearest functional facilities
    making sure waiting time and utilization ratio does not exceed threshold.

    Input:
    damaged_facilities: array of damaged facilities
    func_facilities: array of functional facilities
    tmp_facilities: array of temporary facilities
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

    # sets
    all_facilities = np.concatenate([damaged_facilities, func_facilities, tmp_facilities])
    original_facilities = np.concatenate([damaged_facilities, func_facilities])
    avail_facilities = np.concatenate([func_facilities, tmp_facilities])
    
    # initial utilization check
    total_arrival_rate = sum([f[h][k] for h in all_facilities for k in prio])
    total_capacity = sum([val for key,val in svr.items() if key in avail_facilities])*list(mu.values())[0]
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
            potential_facilities = sort_functional_facilities(d, avail_facilities, W_hk_curr, avg_speed, k, 
                                                              dist_dict = dist_dict, travel_time = travel_time)
            
            # damaged facilities cannot handle any demand
            lam_hk_curr[d,k] = 0; rho_hk_curr[d,k] = 0; W_hk_curr[d,k] = 0
            
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
                        if i not in tmp_facilities: # only update waiting time if not temporary facility
                            for kk in prio: # need to update waiting time for all priorities
                                W_hk_curr[i,kk] = calculate_waiting_time_k(svr[i], mu[i], [tmp_rho_hk[i,j] for j in prio],kk)
                        rho_hk_curr[i,k] = tmp_rho_hk[i,k]
                        lam_h_curr[i] += interval
                        rho_h_curr[i] = tmp_rho_h

                        # update allocation
                        allocation[i,k].append(d) # d move to i
                        allocation[d,k].remove(d) # d move from d

                        break # no need to check for other facility

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


def reallocate_patients(to_reconstruct, all_facilities, original_facilities, curr_damaged, tmp_facilities, allocation, 
                        lam_hk, rho_hk, W_hk, lam_h, rho_h, f, svr, mu, prio, avg_speed, interval, dist_dict = None, travel_time = None):
    '''
    Given a facility that is reconstructed, reallocate any demand such that the average total time is reduced
    (travel + waiting time)

    Input:
    to_reconstruct: facility ID to be reconstructed
    all_facilities: array of all facilities in region (includes the temporary facilities)
    original_facilities: array of all facilities in region (excludes the temporary facilities)
    curr_damaged: array with currently damaged facilities
    tmp_facilities: array of temporary facilities in the region
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
    interval: for purposes of patient allocation (how detailed partial allocation is)
    '''
    
    # make sure the original dicts are not modified
    lam_hk_curr = copy.deepcopy(lam_hk)
    rho_hk_curr = copy.deepcopy(rho_hk)
    W_hk_curr = copy.deepcopy(W_hk)
    lam_h_curr = copy.deepcopy(lam_h)
    rho_h_curr = copy.deepcopy(rho_h)
    allocation_curr = copy.deepcopy(allocation)

    # 1) move demand back to the original facility that is currently being reconstructed
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


    # 2) move demand from temporary facilities into any available facilities
    # available facilities (undamaged + to_reconstruct)
    avail_facilities = np.concatenate([[h for h in original_facilities if h not in curr_damaged], [to_reconstruct]])
    # if to_reconstruct == 30:
    #     print(avail_facilities)

    # for each priority
    for k in prio:

        # get allocation in the specific priority that is located in the tmp facility
        alloc_prio_tmp = {key:val for key,val in allocation_curr.items() if key[1] == k and key[0] in tmp_facilities}

        # get list of demand (where it is originally located)
        list_from = [val for key,val in alloc_prio_tmp.items()]
        list_from = np.array([item for sublist in list_from for item in sublist]) # flatten
        # if to_reconstruct == 30:
        #     print(list_from)

        # get where each demand is currently located at
        list_to_move_from = []
        for key in alloc_prio_tmp.keys():
            list_to_move_from = list_to_move_from + [key[0] for v in alloc_prio_tmp[key]]

        # get current total_time
        if dist_dict is not None: # use average speed to calculate travel time
            curr_totaltime_all = [W_hk_curr[list_to_move_from[i],k] + dist_dict[list_from[i],list_to_move_from[i]]/avg_speed for i in range(len(list_to_move_from))]

        elif travel_time is not None: # use actual travel time
            curr_totaltime_all = [W_hk_curr[list_to_move_from[i],k] + travel_time[list_from[i],list_to_move_from[i]] for i in range(len(list_to_move_from))]
        
        curr_totaltime_all = np.array(curr_totaltime_all)

        # sort by descending order (from longest waiting time to shortest)
        list_to_move_from = np.array(list_to_move_from)
        list_to_move_from_sorted = list_to_move_from[np.argsort(curr_totaltime_all)[::-1]]
        curr_totaltime_sorted = curr_totaltime_all[np.argsort(curr_totaltime_all)[::-1]]
        list_from_sorted = list_from[np.argsort(curr_totaltime_all)[::-1]]

        # for each patient demand in temporary facility, try to move it to available facilities
        for i in range(len(list_to_move_from_sorted)):

            # loop through functional facilities
            potential_facilities = sort_functional_facilities(list_from[i], avail_facilities, W_hk_curr, avg_speed, k, 
                                                              dist_dict = dist_dict, travel_time = travel_time)
        
            for f in potential_facilities:
                # update arrival rate and utilization
                tmp_lam_hk = lam_hk_curr[f,k] + interval
                tmp_rho_hk = rho_hk_curr.copy()
                tmp_rho_hk[f,k] = tmp_lam_hk/(svr[f]*mu[f])
                tmp_rho_h = sum(tmp_rho_hk[f,p] for p in prio)

                # utilization ratio 
                # actually assign to the facility
                if tmp_rho_h < 0.99:
                    # update variables
                    lam_hk_curr[f,k] = tmp_lam_hk
                    rho_hk_curr[f,k] = tmp_rho_hk[f,k]
                    lam_h_curr[f] += interval
                    rho_h_curr[f] = tmp_rho_h

                    # update allocation
                    allocation_curr[f,k].append(list_from[i]) # list_from[i] move to f
                    allocation_curr[list_to_move_from[i],k].remove(list_from[i]) # list_from[i] move from list_to_move_from[i]

                    # facility modified
                    facilities_modified.add(f)

                    break # no need to check for other facility


    # 3) update waiting time (update only after everything is reallocated)
    facilities_modified.add(to_reconstruct)
    for k in prio:
        for h in facilities_modified:
            W_hk_curr[h, k] = calculate_waiting_time_k(svr[h], mu[h], [rho_hk_curr[h,j] for j in prio],k)
        
    output = (lam_hk_curr, rho_hk_curr, W_hk_curr, lam_h_curr, rho_h_curr, allocation_curr)
        
    return output

def calculate_avg_total_time(allocation_curr, W_hk_curr, tmp_facilities, avg_speed, prio, dist_dict = None, travel_time = None, inc_tmp = False):
    '''
    Calculates the average total time (travel + wait at facility) for all the patients in the region. 
    Essentially the objective function of the model. 

    Input:
    allocation_curr: dictionary with key: original facility, and value: current facility allocation
    W_hk_curr: dictionary with key: facility ID, and value: waiting time for facility h and priority level k
    tmp_facilities: array of temporary facilities in the region
    dist_dict: a dictionary between pairs of facilities as the key and distance as the value
    prio: list with priority levels
    avg_speed: average speed for travel
    '''

    # get list of demands (originally located)
    if inc_tmp:
        list_from = [val for key,val in allocation_curr.items()]
    else:
        list_from = [val for key,val in allocation_curr.items() if key[0] not in tmp_facilities]
    list_from = np.array([item for sublist in list_from for item in sublist]) # flatten

    # get where each demand is currently located at
    list_to_move_from = []
    for key in allocation_curr.keys():
        if inc_tmp:
            list_to_move_from = list_to_move_from + [key[0] for v in allocation_curr[key]]
        else:
            if key[0] not in tmp_facilities:
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

def greedy_tmp(damaged_facilities, func_facilities, tmp_facilities, cons_time, f, svr, mu, prio, avg_speed, interval = 0.1, dist_dict = None, travel_time = None):
    '''
    Greedy algorithm to determine the optimal construction ordering based on the gittins index
    (maximum potential cost reduction). Outputs an array of the construction order and detailed results. 

    Input:
    damaged_facilities: array of damaged facilities
    func_facilities: array of functional facilities
    tmp_facilities: array of temporary facilities
    dist_dict: a dictionary between pairs of facilities as the key and distance as the value
    cons_time: dictionary with key: facility ID, and value: construction time (days)
    f: dictionary for demand arrival rate for facility h and priority level k
    svr: dictionary for number of servers for facility h
    mu: dictionary for service rate of facility h
    prio: list with priority levels
    avg_speed: average speed for travel
    interval: for purposes of patient allocation (how detailed partial allocation is)
    '''
    print('Intializing Arrays... Calculating Initial Allocation')
    initial_output = initialize_model(damaged_facilities, func_facilities, tmp_facilities, f, svr, mu, prio, avg_speed, interval, 
                                      dist_dict = dist_dict, travel_time = travel_time)
    (all_facilities, original_facilities, num_damaged, lam_hk, rho_hk, W_hk, lam_h, rho_h, allocation, num_in_tmp) = initial_output

    # calculate total avg wait time
    cost_total = np.zeros(num_damaged + 1) # includes patients in temporary facility
    cost_total[0] = calculate_avg_total_time(allocation, W_hk, tmp_facilities, avg_speed, prio, dist_dict = dist_dict, travel_time = travel_time, inc_tmp = True)

    cost_total_wo_tmp = np.zeros(num_damaged + 1)
    cost_total_wo_tmp[0] = calculate_avg_total_time(allocation, W_hk, tmp_facilities, avg_speed, prio, dist_dict = dist_dict, travel_time = travel_time)
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
        for i in curr_damaged:
            tmp_output = reallocate_patients(i, all_facilities, original_facilities, curr_damaged, tmp_facilities, 
                                             allocation, lam_hk, rho_hk, W_hk, lam_h, rho_h, f, svr, mu, prio, avg_speed, interval, 
                                             dist_dict = dist_dict, travel_time = travel_time)
            (_, _, W_hk_tmp, _, _, alloc_tmp) = tmp_output

            # calculate potential cost
            potential_cost_total[i] = calculate_avg_total_time(alloc_tmp, W_hk_tmp, tmp_facilities, avg_speed, prio, dist_dict = dist_dict, travel_time = travel_time, inc_tmp = True)
            potential_gittins[i] = (cost_total[d] - potential_cost_total[i])/cons_time[i] # compares cost including patients in tmp facilities
        # print(potential_cost_total)
        # print(potential_gittins)    

        # choose facility with highest gittins index
        chosen_facility = max(potential_gittins, key = potential_gittins.get)
        construction_order.append(chosen_facility)
        print('chosen facility', chosen_facility)

        # update values
        output_realloc = reallocate_patients(chosen_facility, all_facilities, original_facilities, curr_damaged, tmp_facilities, allocation, 
                                             lam_hk, rho_hk, W_hk, lam_h, rho_h, f, svr, mu, prio, avg_speed, interval, 
                                             dist_dict = dist_dict, travel_time = travel_time)
        (lam_hk, rho_hk, W_hk, lam_h, rho_h, allocation) = output_realloc

        # update cost
        cost_total[d+1] = calculate_avg_total_time(allocation, W_hk, tmp_facilities, avg_speed, prio, dist_dict = dist_dict, travel_time = travel_time, inc_tmp = True)
        cost_total_wo_tmp[d+1] = calculate_avg_total_time(allocation, W_hk, tmp_facilities, avg_speed, prio, dist_dict = dist_dict, travel_time = travel_time)
            
        # count demand in temp facilities
        num_in_tmp[d+1] = sum(len(vals) for key, vals in allocation.items() if key[0] in tmp_facilities)*interval

        # check if there is still demand in tmp facility. If not, close tmp facility
        for t in tmp_facilities:
            demand_in_tmp = sum(len(allocation[t,k]) for k in prio)
            if demand_in_tmp == 0:
                all_facilities = all_facilities[all_facilities != t]
        
        curr_damaged = curr_damaged[curr_damaged != chosen_facility]
        allocation_time.append(allocation)
        W_hk_time.append(W_hk)

    print('Construction Order:', construction_order)

    detailed_results = {'order': construction_order, 'W_hk_time': W_hk_time, 'allocation_time': allocation_time, 'num_in_tmp': num_in_tmp, 'cost_total': cost_total, 
                        'cost_total_wo_tmp': cost_total_wo_tmp, 'tmp_facilities': tmp_facilities, 'travel_time': travel_time, 'dist_dict': dist_dict}
    
    return construction_order, detailed_results

def manual_order_tmp(order, damaged_facilities, func_facilities, tmp_facilities, f, svr, mu, prio, avg_speed, interval = 0.1, dist_dict = None, travel_time = None):
    '''
    Given the construction ordering, calculate the allocation of patient demand and total cost (distance + waiting time).
    Outputs the detailed results

    Input:
    order: list of the construction ordering
    damaged_facilities: array of damaged facilities
    func_facilities: array of functional facilities
    tmp_facilities: array of temporary facilities
    dist_dict: a dictionary between pairs of facilities as the key and distance as the value
    f: dictionary for demand arrival rate for facility h and priority level k
    svr: dictionary for number of servers for facility h
    mu: dictionary for service rate of facility h
    prio: list with priority levels
    avg_speed: average speed for travel
    '''

    # Initialize arrays   
    initial_output = initialize_model(damaged_facilities, func_facilities, tmp_facilities, f, svr, mu, prio, avg_speed, interval, 
                                       dist_dict = dist_dict, travel_time = travel_time)
    (all_facilities, original_facilities, num_damaged, lam_hk, rho_hk, W_hk, lam_h, rho_h, allocation, num_in_tmp) = initial_output

    # calculate total avg wait time
    cost_total = np.zeros(num_damaged + 1)
    cost_total[0] = calculate_avg_total_time(allocation, W_hk, tmp_facilities, avg_speed, prio, dist_dict = dist_dict, travel_time = travel_time, inc_tmp = True)

    cost_total_wo_tmp = np.zeros(num_damaged + 1)
    cost_total_wo_tmp[0] = calculate_avg_total_time(allocation, W_hk, tmp_facilities, avg_speed, prio, dist_dict = dist_dict, travel_time = travel_time)
    # print(cost_total[0])

    # save allocation and waiting time   
    allocation_time = []
    allocation_time.append(allocation)
    W_hk_time = []
    W_hk_time.append(W_hk)

    # Construction Ordering
    # print('Calculating Costs based on Order...')
    curr_damaged = damaged_facilities.copy()

    for d in range(len(order)):
        output_realloc = reallocate_patients(order[d], all_facilities, original_facilities, curr_damaged, tmp_facilities, allocation,
                                             lam_hk, rho_hk, W_hk, lam_h, rho_h, f, svr, mu, prio, avg_speed, interval, 
                                             dist_dict = dist_dict, travel_time = travel_time)
        (lam_hk, rho_hk, W_hk, lam_h, rho_h, allocation) = output_realloc
        
        # update cost
        cost_total[d+1] = calculate_avg_total_time(allocation, W_hk, tmp_facilities, avg_speed, prio, dist_dict = dist_dict, travel_time = travel_time, inc_tmp = True)
        cost_total_wo_tmp[d+1] = calculate_avg_total_time(allocation, W_hk, tmp_facilities, avg_speed, prio, dist_dict = dist_dict, travel_time = travel_time)
        
        # count demand in temp facilities
        num_in_tmp[d+1] = sum(len(vals) for key, vals in allocation.items() if key[0] in tmp_facilities)*interval

        # check if there is still demand in tmp facility. If not, close tmp facility
        for t in tmp_facilities:
            demand_in_tmp = sum(len(allocation[t,k]) for k in prio)
            if demand_in_tmp == 0:
                all_facilities = all_facilities[all_facilities != t]
        
        curr_damaged = curr_damaged[curr_damaged != order[d]]
        allocation_time.append(allocation)
        W_hk_time.append(W_hk)
    
    detailed_results = {'order': order, 'W_hk_time': W_hk_time, 'allocation_time': allocation_time, 'num_in_tmp': num_in_tmp, 'cost_total': cost_total,
                        'cost_total_wo_tmp': cost_total_wo_tmp, 'tmp_facilities': tmp_facilities, 'travel_time': travel_time, 'dist_dict': dist_dict}
    
    return detailed_results


def add_temp_facility(total_capacity, total_arrival_rate, all_facilities, mu, f, svr, cons_time, dist_dict = None, max_svr = 100, travel_time = None):
    '''
    Creates temporary facilities to acccomodate the demand. 

    Input:
    total_capacity: current total capacity of the region
    total_arrival_rate: current total arrival rate (demand) of the regions
    all_facilities: array of all facilities in region (damaged and functional)
    f: dictionary for demand arrival rate for facility h and priority level k
    svr: dictionary for number of servers for facility h
    mu: dictionary for service rate of facility h
    cons_time: dictionary with key: facility ID, and value: construction time (days)
    dist_dict: a dictionary between pairs of facilities as the key and distance as the value
    max_svr: maximum number of servers in one temporary facility

    '''
    
    # Need capacity increase
    MU = list(mu.values())[0]
    inc_cap = np.round(total_arrival_rate - total_capacity, 0) + 5
    total_svrs = int(np.ceil(inc_cap/MU))
    num_tmp_facilities = int(np.ceil(total_svrs/max_svr))

    # Create temporary facilities and updates the variables
    tmp_id = np.array([int('999{}'.format(i)) for i in np.arange(num_tmp_facilities)])
    for t in tmp_id:
        f[t] = {1: 0, 2: 0}
        if num_tmp_facilities == 1 and total_svrs < 50:
            svr[t] = total_svrs
        else:
            svr[t] = max_svr
        mu[t] = MU
        cons_time[t] = 0

    all_combo = [(t,h) for t in tmp_id for h in all_facilities]

    if dist_dict is not None: # using Euclidean distance and avg speed
        for combo in all_combo:
            dist_dict[combo[0], combo[1]] = 1000
            dist_dict[combo[1], combo[0]] = 1000
        for t in tmp_id:
            dist_dict[t, t] = 0
    
        return tmp_id, f, svr, mu, cons_time, dist_dict

    elif travel_time is not None: # using travel time
        for combo in all_combo:
            travel_time[combo[0], combo[1]] = 1000
            travel_time[combo[1], combo[0]] = 1000
        for t in tmp_id:
            travel_time[t, t] = 0
    
        return tmp_id, f, svr, mu, cons_time, travel_time

def wait_disagg_tmp(output_option, result, tmp_facilities, avg_speed = 30, k = None, dist_dict = None, travel_time = None, mean = True):
    '''
    Function to output the mean total waiting time over the recovery period. 
    Can be disaggregated by travel time, wait time and total time. 
    Can also be disaggregated by priority level

    Input:
    output_option: either 'total', dist', 'wait' to indicate total time, travel time or wait time at facility
    result: the results dict from the greedy algorithm output
    tmp_facilities: array of temporary facilities
    dist_dict: a dictionary between pairs of facilities as the key and distance as the value
    avg_speed: average speed for travel
    k: if not defined, then for all priority. else disaggregated by priority level
    '''
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 

    def helper_function(j, k = None):
        if k is None:
            alloc = result['allocation_time'][j]
        else:
            alloc = {key:val for key,val in result['allocation_time'][j].items() if key[1] == k}

        # get list of demands (originally located)
        list_from = [val for key,val in alloc.items() if key[0] not in tmp_facilities]
        list_from = np.array([item for sublist in list_from for item in sublist]) # flatten

        # get where each demand is currently located at
        list_to_move_from = []
        for key in alloc.keys():
            if key[0] not in tmp_facilities:
                list_to_move_from = list_to_move_from + [key[0] for v in alloc[key]]
        list_to_move_from = np.array(list_to_move_from)
        
        return list_from, list_to_move_from

    # initialize
    if mean:
        mean_wait = np.zeros(len(result['allocation_time']))
    else:
        mean_wait = []
    
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
                dist_time = np.array([travel_time[list_from[i],list_to_move_from[i]]/avg_speed for i in range(len(list_to_move_from))])
            
            total_time = dist_time + wait_time
        
        if output_option == 'total':
            if mean:
                mean_wait[j] = np.mean(total_time)
            else:
                mean_wait.append(total_time)
        elif output_option == 'dist':
            if mean:
                mean_wait[j] = np.mean(dist_time)
            else:
                mean_wait.append(dist_time)
        elif output_option == 'wait':
            if mean:
                mean_wait[j] = np.mean(wait_time)
            else:
                mean_wait.append(wait_time)

    return mean_wait
