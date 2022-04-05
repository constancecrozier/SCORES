import time
from generation import (OffshoreWindModel, SolarModel, OnshoreWindModel5800)
from storage import (BatteryStorageModel, HydrogenStorageModel,
                      MultipleStorageAssets)
from system import ElectricitySystem, ElectricitySystemGB
import csv
import numpy as np
import matplotlib.pyplot as plt
import aggregatedEVs as aggEV
from opt_con_class import System_LinProg_Model
'''
Initialise generators
'''
ymin = 2014
ymax = 2014


osw_master = OffshoreWindModel(year_min=ymin, year_max=ymax, sites=[119,174,178,209,364],
                        data_path='data/150m/')

# For onshore wind, various turbine sizes are available, this is for 3.6 MW
w = OnshoreWindModel5800(year_min=ymin, year_max=ymax, sites='all',
                        data_path='data/wind/')

#p = w.power_out # time-series of the output

# Note that the solar model is substantially slower than the wind models
s = SolarModel(year_min=ymin, year_max=ymax, sites=[17,23,24],
                        data_path='data/solar/')
#s.limits = [0,40000]

'''
Initialise storage
'''
#T = ThermalStorageModel()
B = BatteryStorageModel()
H = HydrogenStorageModel()
#B.limits = [2000000,4000000]

'''
System optimisation
'''
# Initialise list of generators
#generators = [osw1,osw2,osw3,osw4,osw5,w,s]
generators = [osw_master,w,s]

# Initialise list of storage
storage = [B,H]

#EVs
Dom1 = aggEV.AggregatedEVModel(eff_in=95, eff_out=95, chargertype=np.zeros([3]), chargercost=np.array([2000/20,800/20,50/20]), max_c_rate=10, max_d_rate=10, min_SOC=0, max_SOC=40, number=200000,initial_number = 0.9, Ein = 20, Eout = 36, Nin = np.array([0,0,0,0,0,0,0,0,0,0.1,0,0,0,0,0,0.1,0.1,0.1,0.1,0,0,0,0,0]),Nout = np.array([0,0,0,0,0,0,0,0.2,0.2,0,0,0,0,0,0,0.1,0,0,0,0,0,0,0,0]),name = 'Domestic1')
MultsFleets = aggEV.MultipleAggregatedEVs([Dom1])


# Initialise electricity sytem with existing GB demand
es = ElectricitySystemGB(generators, storage, year_min = ymin, year_max = ymax,
                         reliability = 99, strategy='ordered', start_up_time=24)  #,aggEV_list = MultsFleets)



# Search for the optimal system
#start = time.time()
#caps, cost = es.optimise(tic0=200)
#end = time.time()
#print('Old Method Time: ',end - start, 's')


#es.fully_optimise(sum(es.demand)*0.01,SimYears=[2014,2015],YearRange=[ymin,ymax])

x = System_LinProg_Model(-np.asarray(es.demand),sum(es.demand)*0.01,storage,MultsFleets)
x.Form_Model()
