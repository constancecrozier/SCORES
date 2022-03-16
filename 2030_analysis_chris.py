# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 14:32:12 2021

@author: whyis
"""
import time
from generation import (OffshoreWindModel, SolarModel, OnshoreWindModel5800)
from storage import (BatteryStorageModel, HydrogenStorageModel,
                      MultipleStorageAssets)
from system import ElectricitySystem, ElectricitySystemGB
import csv
import numpy as np
import matplotlib.pyplot as plt
import aggregatedEVs as aggEV
'''
Initialise generators
'''
ymin = 2016
ymax = 2017

# osw1 = OffshoreWindModel(year_min=ymin, year_max=ymax, sites=[119],
#                         data_path='data/150m/')
# osw2 = OffshoreWindModel(year_min=ymin, year_max=ymax, sites=[174],
#                         data_path='data/150m/')
# osw3 = OffshoreWindModel(year_min=ymin, year_max=ymax, sites=[178],
#                         data_path='data/150m/')
# osw4 = OffshoreWindModel(year_min=ymin, year_max=ymax, sites=[209],
#                         data_path='data/150m/')
# osw5 = OffshoreWindModel(year_min=ymin, year_max=ymax, sites=[364],
#                         data_path='data/150m/')

osw_master = OffshoreWindModel(year_min=ymin, year_max=ymax, sites=[119,174,178,209,364],
                        data_path='data/150m/')
#p = osw.power_out # time-series of the output

# For onshore wind, various turbine sizes are available, this is for 3.6 MW
w = OnshoreWindModel5800(year_min=ymin, year_max=ymax, sites='all',
                        data_path='data/wind/')

#p = w.power_out # time-series of the output

# Note that the solar model is substantially slower than the wind models
s = SolarModel(year_min=ymin, year_max=ymax, sites=[17,23,24],
                        data_path='data/solar/')


'''
Initialise storage
'''
#T = ThermalStorageModel()
B = BatteryStorageModel()
H = HydrogenStorageModel()

'''
System optimisation
'''
# Initialise list of generators
#generators = [osw1,osw2,osw3,osw4,osw5,w,s]
generators = [osw_master,w,s]

# Initialise list of storage
storage = [B,H]

# Initialise electricity sytem with existing GB demand
es = ElectricitySystemGB(generators, storage, year_min = ymin, year_max = ymax,
                         reliability = 99, strategy='ordered', start_up_time=24)

# Search for the optimal system
#start = time.time()
#caps, cost = es.optimise(tic0=200)
#end = time.time()
#print('Old Method Time: ',end - start, 's')

start = time.time()
es.fully_optimise(sum(es.demand)*0.01,fixed_capacities=False)
end = time.time()
es.new_analyse(filename='log/new.txt')
print('New Method Time: ',end - start, 's')
#es.new_analyse(filename='log/opt_results_improved_1.txt')
start=int(1000)
end = int(1500)
es.plot_timeseries(start,end)
es.plot_timeseries(4000,4500)

Dom1 = aggEV.AggregatedEVModel(eff_in=95, eff_out=95, chargertype=np.zeros([3]), chargercost=np.array([500000,800/25,50/25]), max_c_rate=10, max_d_rate=10, min_SOC=0, max_SOC=40, number=2000000,initial_number = 0.9, Ein = 20, Eout = 36, Nin = np.array([0,0,0,0,0,0,0,0,0,0.1,0,0,0,0,0,0.1,0.1,0.1,0.1,0,0,0,0,0]),Nout = np.array([0,0,0,0,0,0,0,0.2,0.2,0,0,0,0,0,0,0.1,0,0,0,0,0,0,0,0]),name = 'Domestic1')

MultsFleets = aggEV.MultipleAggregatedEVs([Dom1])

es = ElectricitySystemGB(generators, storage, year_min = ymin, year_max = ymax, 
                         reliability = 99, strategy='ordered', start_up_time=24,aggEV_list = MultsFleets)
start = time.time()
es.fully_optimise(sum(es.demand)*0.01,fixed_capacities=False)
end = time.time()
print('New with V2G Method Time: ',end - start, 's')

es.new_analyse(filename='log/noV2G.txt')


storage[0].plot_timeseries(1800,2000)
storage[1].plot_timeseries(1800,2000)
MultsFleets.assets[0].plot_timeseries(1800,2000)




    
               
