import numpy as np
import pandas as pd
from queue import Queue
import matplotlib.pyplot as plt
import multiprocessing
import functools
import pickle
import sys
import math

experiments = []

# Generate all experiments we want to run
for model in ['lenet5', 'bi-rnn']:
    for batch_size in [64, 128, 256]:
        for total_cores in [1, 2, 4]:
            for core_per in [1, 2, 4]:
                l_high = 20/400
                l_low = 1/400
                num_steps = 16
                experiments.append(
                    {
                        'ID': f'4servers_4cores_{batch_size}-coresTotal{total_cores}-coresPer{core_per}-{model}',
                        'NUM_SERVERS': 4,
                        'SERVER_CPU_COUNT': 4,
                        'CONCURRENT_JOBS': int(16 / (total_cores * core_per)),
                        'L_HIGH': l_high,
                        'L_LOW': l_low,
                        'L_STEP': (l_high - l_low) / num_steps,
                        'BATCH_SIZE': batch_size,
                        'TOTAL_EXECUTOR_CORES': total_cores,
                        'CORES_PER_EXECUTOR': core_per,
                        'MODEL': model
                    }
                )

# Read the service times from the experiments
f = open('service-times.pkl', 'rb')
df_service_times = pickle.load(f)
f.close()

# This function runs the actual simulation
def sim(arrival_rate, num_servers, service_time_mean, service_time_sd):
    try:
        # Initialize queue and clock
        queue = Queue()
        time_elapsed = 0
        arrival_interval = 0
        max_time = 2000000
        dt = 1
        # Initialise arrays for processing status, job end and start times to determine when a job is finished
        processing_arr = [False] * num_servers
        end_time_arr = [0] * num_servers
        start_time_arr = [0] * num_servers
        # Keep track of wait and response times of jobs
        wait_times_arr = [[]] * num_servers
        response_times_arr = [[]] * num_servers
        # Array that the current queue size is stored in every iteration
        queue_size = []
        # Stores the amount of completed jobs
        jobs = 0
        while time_elapsed < max_time:

            queue_size.append(queue.qsize())

            # Generate job
            if arrival_interval > 1/arrival_rate:
                queue.put(time_elapsed)
                jobs += 1
                arrival_interval = 0

            # Update processors
            for cpu_idx, processing in enumerate(processing_arr):
                # If not processing and there is a job available, run new job
                if not processing and not queue.empty():
                    wait_time = time_elapsed - queue.get()
                    wait_times_arr[cpu_idx].append(wait_time)
                    start_time_arr[cpu_idx] = time_elapsed
                    # Generate service time for the job
                    service_time = np.random.normal(service_time_mean, service_time_sd, 1)[0]
                    end_time_arr[cpu_idx] = time_elapsed + service_time
                    response_times_arr[cpu_idx].append(wait_time + service_time)
                    processing_arr[cpu_idx] = True
                # If processing, check if job finished in this iteration and complete it if it does
                elif processing:
                    if time_elapsed >= end_time_arr[cpu_idx]:
                        processing_arr[cpu_idx] = False

            arrival_interval += dt
            time_elapsed += dt
        
        # Calculate the theoretical expected response time
        mu = 1 / service_time_mean
        rho = arrival_rate / (num_servers * mu)
        pi_0 = 0
        k = num_servers
        for i in range(num_servers):
            pi_0 += (((k*rho)**i) / math.factorial(i)) + (((k*rho)**k)/(math.factorial(k)*(1-rho)))

        pi_0 = 1 / pi_0
        P_Q = (((k*rho)**k)*pi_0)/(math.factorial(k)*(1-rho))
        E_n = (P_Q * (rho/(1-rho))) + (k*rho)
        E_t = E_n / arrival_rate

        return {
            'avg_queue_time': sum(map(sum, wait_times_arr))/sum(map(len, wait_times_arr)),
            'avg_queue_length': sum(queue_size)/len(queue_size),
            'avg_response_time': sum(map(sum, response_times_arr))/sum(map(len, response_times_arr)),
            'expected_response_time': E_t,
            'expected_num_of_jobs': E_n,
            'arrival_rate': arrival_rate,
            'mean_service_time': service_time_mean,
            'sd_service_time': service_time_sd,
            'status': 'completed'
        }
    # If an exception is thrown, return status failed as result
    except:
        return {'status': 'failed'}

# Retrieve the correct mean service time from the results of the experiments
def calcMean(settings):
    try:
        return df_service_times['Mean'][settings['BATCH_SIZE']][settings['CORES_PER_EXECUTOR']][settings['TOTAL_EXECUTOR_CORES']][settings['MODEL']]
    except:
        return 0

# Retrieve the correct standard deviation of service time from the results of the experiments
def calcSd(settings):
    try:
        return df_service_times['SD'][settings['BATCH_SIZE']][settings['CORES_PER_EXECUTOR']][settings['TOTAL_EXECUTOR_CORES']][settings['MODEL']]
    except:
        return 0

# Create helper functions, necessary for using subprocesses to run simulations on multiple cores
simmers = list(map(lambda settings: functools.partial(sim, num_servers=settings['CONCURRENT_JOBS'], service_time_mean=calcMean(settings), service_time_sd=calcSd(settings)), experiments))

if __name__ == '__main__':
    # For every experiment, start a pool of workers and run the simulation for all arrival rates
    for ex_idx, settings in enumerate(experiments):
        with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
            x = np.arange(settings['L_LOW'], settings['L_HIGH'], step=settings['L_STEP'])
            res = pool.map(simmers[ex_idx], x)
            pool.terminate()

        # Print the result and save it in a file
        print(res)
        toDump = {
            'settings': settings,
            'res': res
        }
        f = open(f"simulations/sim-{settings['ID']}.pkl", "wb")
        pickle.dump(toDump, f)
        f.close()