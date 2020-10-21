# This file is very similar to simulator.py, for an explanation of the code check the comments in that file.
# This file was used to run some specific experiments in more detail

import numpy as np
import pandas as pd
from queue import Queue
import matplotlib.pyplot as plt
import multiprocessing
import functools
import pickle
import sys
import math

# Create the experiments
experiments = [
    {
        'NUM_SERVERS': 4,
        'SERVER_CPU_COUNT': 4,
        'CONCURRENT_JOBS': 16,
        'L_HIGH': 0,
        'L_LOW': 0,
        'L_STEP': 0,
        'BATCH_SIZE': 256,
        'TOTAL_EXECUTOR_CORES': 1,
        'CORES_PER_EXECUTOR': 1,
        'MODEL': 'lenet5'
    },
    {
        'NUM_SERVERS': 4,
        'SERVER_CPU_COUNT': 4,
        'CONCURRENT_JOBS': 16,
        'L_HIGH': 0,
        'L_LOW': 0,
        'L_STEP': 0,
        'BATCH_SIZE': 128,
        'TOTAL_EXECUTOR_CORES': 1,
        'CORES_PER_EXECUTOR': 1,
        'MODEL': 'lenet5'
    },
    {
        'NUM_SERVERS': 4,
        'SERVER_CPU_COUNT': 4,
        'CONCURRENT_JOBS': 4,
        'L_HIGH': 0,
        'L_LOW': 0,
        'L_STEP': 0,
        'BATCH_SIZE': 128,
        'TOTAL_EXECUTOR_CORES': 4,
        'CORES_PER_EXECUTOR': 1,
        'MODEL': 'lenet5'
    },
    {
        'NUM_SERVERS': 4,
        'SERVER_CPU_COUNT': 4,
        'CONCURRENT_JOBS': 16,
        'L_HIGH': 0,
        'L_LOW': 0,
        'L_STEP': 0,
        'BATCH_SIZE': 64,
        'TOTAL_EXECUTOR_CORES': 1,
        'CORES_PER_EXECUTOR': 1,
        'MODEL': 'lenet5'
    },
    {
        'NUM_SERVERS': 4,
        'SERVER_CPU_COUNT': 4,
        'CONCURRENT_JOBS': 6,
        'L_HIGH': 0,
        'L_LOW': 0,
        'L_STEP': 0,
        'BATCH_SIZE': 128,
        'TOTAL_EXECUTOR_CORES': 2,
        'CORES_PER_EXECUTOR': 1,
        'MODEL': 'lenet5'
    }
]

f = open('service-times.pkl', 'rb')
df_service_times = pickle.load(f)
f.close()

def sim(arrival_rate, num_servers, service_time_mean, service_time_sd):
    try:
        queue = Queue(maxsize=10)
        time_elapsed = 0
        arrival_interval = 0
        max_time = 50000
        dt = 1
        processing_arr = [False] * num_servers
        end_time_arr = [0] * num_servers
        start_time_arr = [0] * num_servers
        wait_times_arr = [[]] * num_servers
        response_times_arr = [[]] * num_servers
        queue_size = []
        jobs = 0
        failed_jobs = 0
        while time_elapsed < max_time:

            queue_size.append(queue.qsize())

            # Generate job
            if arrival_interval > 1/arrival_rate:
                try:
                    queue.put(time_elapsed, block=False)
                    jobs += 1
                except:
                    failed_jobs += 1
                arrival_interval = 0

            for cpu_idx, processing in enumerate(processing_arr):
                if not processing and not queue.empty():
                    wait_time = time_elapsed - queue.get()
                    wait_times_arr[cpu_idx].append(wait_time)
                    start_time_arr[cpu_idx] = time_elapsed
                    service_time = np.random.normal(service_time_mean, service_time_sd, 1)[0]
                    end_time_arr[cpu_idx] = time_elapsed + service_time
                    response_times_arr[cpu_idx].append(wait_time + service_time)
                    processing_arr[cpu_idx] = True
                elif processing:
                    if time_elapsed >= end_time_arr[cpu_idx]:
                        processing_arr[cpu_idx] = False

            arrival_interval += dt
            time_elapsed += dt
        
        # Expected response time
        mu = 1 / service_time_mean
        rho = arrival_rate / (num_servers * mu)
        pi_0 = 0
        k = num_servers
        for i in range(num_servers):
            pi_0 += (((k*rho)**i) / math.factorial(i)) + (((k*rho)**k)/(math.factorial(k)*(1-rho)))

        pi_0 = 1 / pi_0
        P_Q = (((k*rho)**k)*pi_0)/(math.factorial(k)*(1-rho))
        E_n = (P_Q * (rho/(1-rho))) + (k*rho)
        E_t = (1/arrival_rate) * P_Q * (rho/(1-rho)) + (1/mu)

        return {
            'avg_queue_time': sum(map(sum, wait_times_arr))/sum(map(len, wait_times_arr)),
            'avg_queue_length': sum(queue_size)/len(queue_size),
            'avg_response_time': sum(map(sum, response_times_arr))/sum(map(len, response_times_arr)),
            'failure_rate': failed_jobs / jobs,
            'expected_response_time': E_t,
            'expected_num_of_jobs': E_n,
            'arrival_rate': arrival_rate,
            'mean_service_time': service_time_mean,
            'sd_service_time': service_time_sd,
            'status': 'completed'
        }
    except:
        return {'status': 'failed'}

def calcMean(settings):
    try:
        return df_service_times['Mean'][settings['BATCH_SIZE']][settings['CORES_PER_EXECUTOR']][settings['TOTAL_EXECUTOR_CORES']][settings['MODEL']]
    except:
        return 0

def calcSd(settings):
    try:
        return df_service_times['SD'][settings['BATCH_SIZE']][settings['CORES_PER_EXECUTOR']][settings['TOTAL_EXECUTOR_CORES']][settings['MODEL']]
    except:
        return 0

simmers = list(map(lambda settings: functools.partial(sim, num_servers=settings['CONCURRENT_JOBS'], service_time_mean=calcMean(settings), service_time_sd=calcSd(settings)), experiments))

if __name__ == '__main__':
    results = []
    for ex_idx, settings in enumerate(experiments):
        with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
            # Use the mean service time to find arrival rates near the point where the response times would go up dramatically
            service_time_mean = calcMean(settings)
            service_rate = 1/service_time_mean
            lamb_middle = service_rate * settings['CONCURRENT_JOBS']
            diff = 0.1 * service_rate
            start = lamb_middle - diff
            end = lamb_middle + diff
            stepsize = (end - start)/100
            x = np.arange(start, end, stepsize)
            res = pool.map(simmers[ex_idx], x)
            pool.terminate()

        print(res)
        toDump = {
            'settings': settings,
            'res': res
        }
        results.append(toDump)
        f = open(f"results{ex_idx}.pkl", "wb")
        pickle.dump(toDump, f)
        f.close()
    f = open(f"results-all.pkl", "wb")
    pickle.dump(results, f)
    f.close()