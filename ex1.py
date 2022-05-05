import time
from generation import (OffshoreWindModel, SolarModel, OnshoreWindModel5800)
from storage import (BatteryStorageModel, HydrogenStorageModel,
                      MultipleStorageAssets)
from system import ElectricitySystem, ElectricitySystemGB
import csv
import numpy as np
import matplotlib.pyplot as plt
import aggregatedEVs as aggEV
from opt_con_class import (System_LinProg_Model,opt_results_to_df)
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
from datetime import datetime
'''
Initialise generators
'''
ymin = 2014
ymax = 2015


osw_master = OffshoreWindModel(year_min=ymin, year_max=ymax, sites=[119,174,178,209,364],
                        data_path='data/150m/')

# For onshore wind, various turbine sizes are available, this is for 3.6 MW
#w = OnshoreWindModel5800(year_min=ymin, year_max=ymax, sites='all',
#                        data_path='data/wind/')

#p = w.power_out # time-series of the output

# Note that the solar model is substantially slower than the wind models
s = SolarModel(year_min=ymin, year_max=ymax, sites=[17,23,24],
                        data_path='data/solar/')




'''
Initialise storage
'''
#T = ThermalStorageModel()
B = BatteryStorageModel(capacity = 1000)
H = HydrogenStorageModel(capacity = 10000)
#B.limits = [2000000,4000000]

'''
System optimisation
'''
# Initialise list of generators
#generators = [osw1,osw2,osw3,osw4,osw5,w,s]
generators = [osw_master,s]

# Initialise list of storage
storage = [B,H]

#EVs
Dom1 = aggEV.AggregatedEVModel(eff_in=95, eff_out=95, chargertype=[0.5,0.5], chargercost=np.array([2000/20,800/20,50/20]), max_c_rate=10, max_d_rate=10, min_SOC=0, max_SOC=36, number=200000,initial_number = 0.9, Ein = 20, Eout = 36, Nin = np.array([0,0,0,0,0,0,0,0,0,0.1,0,0,0,0,0,0.1,0.2,0.1,0.1,0,0,0,0,0]),Nout = np.array([0,0,0,0,0,0,0,0.3,0.2,0,0,0,0,0,0,0.1,0,0,0,0,0,0,0,0]),Nin_weekend = np.array([0,0,0,0,0,0,0,0.0,0.0,0,0,0,0,0,0,0.0,0,0,0,0,0,0,0,0]),Nout_weekend = np.array([0,0,0,0,0,0,0,0.0,0.0,0,0,0,0,0,0,0.0,0,0,0,0,0,0,0,0]),name = 'Domestic1')
MultsFleets = aggEV.MultipleAggregatedEVs([Dom1])


# Initialise electricity sytem with existing GB demand
es = ElectricitySystemGB(generators, storage, year_min = ymin, year_max = ymax,
                         reliability = 99, strategy='ordered', start_up_time=0,aggEV_list = MultsFleets)




'''
For Operation
'''
multStor = MultipleStorageAssets(storage)

power = np.asarray(osw_master.power_out[0:500])
power = power/power.max() * 200

demand = np.asarray(es.demand[0:500])/1000
surplus = power-demand

#demand = np.zeros([500])
#power = np.zeros([500])

# # #### Non Causal ####
x2 = multStor.non_causal_system_operation(demand,power,MultsFleets,plot_timeseries = True,InitialSOC=[0,1,1,1])
print(x2)

#### Causal ####
# x1 = multStor.causal_system_operation(demand,power,[2,3,0,1],[0,1,3,2],MultsFleets, start = datetime(ymin,1,1,0),end =datetime(ymax,1,1,0),plot_timeseries = True,V2G_discharge_threshold = 20.0,initial_SOC=[0,1,0.0,0.0])
# print(x1['Causal Reliability'][0])




#x1.to_csv('log/test.csv')

'''
Run Sizing Then Op
'''
                                                           
# x = System_LinProg_Model(surplus = -np.asarray(es.demand),fossilLimit = 0.01,Mult_Stor = MultipleStorageAssets(storage),Mult_aggEV = MultsFleets, gen_list = generators,YearRange = [ymin,ymax])
# x.Form_Model(True)
# df1 = x.Run_Sizing_Then_Op(range(ymin,ymax+1),V2G_discharge_threshold = 26.0, c_order=[2,3,0,1],d_order=[0,3,1,2])
# df1.to_csv('log/Reliability1.csv', index=False)
# x.df_capital.to_csv('log/Capital1.csv', index=False)



'''
Simple Optimisation
'''
# x = System_LinProg_Model(surplus = -np.asarray(es.demand),fossilLimit = 0.01,Mult_Stor = MultipleStorageAssets(storage),Mult_aggEV = MultsFleets, gen_list = generators,YearRange = [ymin,ymax])
# x.Form_Model(False)
# x.Run_Sizing()
# x.df_capital.to_csv('log/Capital.csv', index=False)


'''
Adjustable Parameters
'''
# ### Generation Limits ###
# s.limits = [20000, 25000]
# for g in x.model.GenIndex:
#     x.model.Gen_Limit_Param_Lower[g] = x.gen_list[g].limits[0]
#     x.model.Gen_Limit_Param_Upper[g] = x.gen_list[g].limits[1]

# ### Storage Limits ###
# H.limits = [0,35000000]
# for i in x.model.StorageIndex:
#     x.model.Stor_Limit_Param_Lower[i] = x.Mult_Stor.assets[i].limits[0]
#     x.model.Stor_Limit_Param_Upper[i] = x.Mult_Stor.assets[i].limits[1]

# ### Charger Type Limits ###
# Dom1.limits = [0,75000,0,200000]
# for k in x.model.FleetIndex:
#     for b in x.model.ChargeType:
#         if b == 0 :
#             x.model.V2G_Limit_Param_Lower[k] = x.Mult_aggEV.assets[k].limits[0]
#             x.model.V2G_Limit_Param_Upper[k] = x.Mult_aggEV.assets[k].limits[1]
#         if b == 1 :
#             x.model.Uni_Limit_Param_Lower[k] = x.Mult_aggEV.assets[k].limits[2]
#             x.model.Uni_Limit_Param_Upper[k] = x.Mult_aggEV.assets[k].limits[3]





#x.Run_Sizing()
#x.df_capital.to_csv('log/Capital_sol_lim.csv', index=False)


'''
Construct Timeseries
'''

#MultsFleets.construct_connectivity_timeseries(start = datetime(2012,1,1,0),end = datetime(2012,3,4,0))

