# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 14:32:12 2021

@author: whyis
"""

from generation import (OffshoreWindModel, SolarModel, OnshoreWindModel5800)
from storage import (BatteryStorageModel, HydrogenStorageModel,
                      MultipleStorageAssets)
from system import ElectricitySystem, ElectricitySystemGB
import csv

'''
Initialise generators
'''
ymin = 2013
ymax = 2014

osw1 = OffshoreWindModel(year_min=ymin, year_max=ymax, sites=[119],
                        data_path='data/150m/')
osw2 = OffshoreWindModel(year_min=ymin, year_max=ymax, sites=[174],
                        data_path='data/150m/')
osw3 = OffshoreWindModel(year_min=ymin, year_max=ymax, sites=[178],
                        data_path='data/150m/')
osw4 = OffshoreWindModel(year_min=ymin, year_max=ymax, sites=[209],
                        data_path='data/150m/')
osw5 = OffshoreWindModel(year_min=ymin, year_max=ymax, sites=[364],
                        data_path='data/150m/')

#p = osw.power_out # time-series of the output

# For onshore wind, various turbine sizes are available, this is for 3.6 MW
w = OnshoreWindModel5800(year_min=ymin, year_max=ymax, sites='all',
                        data_path='data/wind/')

#p = w.power_out # time-series of the output

# Note that the solar model is substantially slower than the wind models
s = SolarModel(year_min=ymin, year_max=ymax, sites='all',
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
generators = [osw1,osw2,osw3,osw4,osw5,w,s]

# Initialise list of storage
storage = [B,H]

# Initialise electricity sytem with existing GB demand
es = ElectricitySystemGB(generators, storage, year_min = ymin, year_max = ymax,
                         reliability = 99, strategy='ordered', start_up_time=24)

# Search for the optimal system
#caps, cost = es.optimise(tic0=200)

#print('Total System Cost Old: £', int(cost*1e3)*1e-3, ' bn/yr' )

es.fully_optimise(sum(es.demand)*0.01,fixed_capacities=False)
es.new_analyse(filename='log/opt_results_improved.txt')
es.plot_timeseries(1000,1500)
es.plot_timeseries(4000,4500)


        



    
               
