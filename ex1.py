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
ymin = 2013
ymax = 2019


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
B = BatteryStorageModel(capacity = 10000)
H = HydrogenStorageModel(capacity = 1000)
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
Dom1 = aggEV.AggregatedEVModel(eff_in=95, eff_out=95, chargertype=[0.5,0.5,0], chargercost=np.array([2000/20,800/20,50/20]), max_c_rate=10, max_d_rate=10, min_SOC=0, max_SOC=36, number=200000,initial_number = 0.9, Ein = 20, Eout = 36, Nin = np.array([0,0,0,0,0,0,0,0,0,0.1,0,0,0,0,0,0.1,0.1,0.1,0.1,0,0,0,0,0]),Nout = np.array([0,0,0,0,0,0,0,0.2,0.2,0,0,0,0,0,0,0.1,0,0,0,0,0,0,0,0]),name = 'Domestic1')
MultsFleets = aggEV.MultipleAggregatedEVs([Dom1])


# Initialise electricity sytem with existing GB demand
es = ElectricitySystemGB(generators, storage, year_min = ymin, year_max = ymax,
                         reliability = 99, strategy='ordered', start_up_time=0,aggEV_list = MultsFleets)

'''
For Testing the Causal Operation
'''
# multStor = MultipleStorageAssets(storage)
# surplus = -(es.demand[0:50]-np.ones([50])*32000)
# surplus = -np.ones([500])*3200
# x1 = multStor.causal_system_operation(surplus,[0,1,2,3],[0,1,2,3],MultsFleets,plot_timeseries = True,V2G_discharge_threshold = 20.0)

#print(x1[0])


'''
Run Sizing Then Op
'''
                                                           #NB, I THINK THERE IS AN ERROR HERE WITH INPUTTING THE FOSSIL LIMIT!! Should make the % specified instead!
x = System_LinProg_Model(surplus = -np.asarray(es.demand),fossilLimit = sum(es.demand)*0.01,Mult_Stor = storage,Mult_aggEV = MultsFleets, gen_list = generators,YearRange = [ymin,ymax])
x.Form_Model(True)
df1 = x.Run_Sizing_Then_Op(range(ymin,ymax+1),V2G_discharge_threshold = 20.0, c_order=[2,3,0,1],d_order=[0,3,1,2])
df1.to_excel('test.xlsx', sheet_name='sheet1', index=False)

'''
Simple Optimisation
'''
#es.fully_optimise(sum(es.demand)*0.01,SimYears=[2014,2015],YearRange=[ymin,ymax])