#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 12:10:27 2022
This example is to demonstrate how o simulate power system operation including EVs.
2 methods are used, 1 is non-causal, where the storage devices and EVs are charged/discharged in a
pre-specified order depending on whether demand is more or less than renewable output.
The second simulation method is non causal and assumes full knowledge of renewable power output and
EV connectivity.
@author: cormacomalley
"""

from generation import (OffshoreWindModel,SolarModel)
import aggregatedEVs as aggEV
from opt_con_class import (System_LinProg_Model)
import numpy as np
from storage import (BatteryStorageModel, HydrogenStorageModel,
                      MultipleStorageAssets)
from fns import get_GB_demand
from datetime import datetime

ymin = 2015
ymax = 2015

#Define the generators
osw_master = OffshoreWindModel(year_min=ymin, year_max=ymax, 
                               sites=[119,174,178,209,364], data_path='data/150m/')

#System has 150GW of Wind
power = np.asarray(osw_master.power_out)
power = power/max(power) * 150000

#Define a Fleet of 100000 EVs, half have V2G Chargers
Dom1 = aggEV.AggregatedEVModel(eff_in=95, eff_out=95, chargertype=[0.5,0.5], chargercost=np.array([2000/20,800/20,50/20]), 
                               max_c_rate=10, max_d_rate=10, min_SOC=0, max_SOC=36, number=10000000,initial_number = 0.9, 
                               Ein = 20, Eout = 36, 
                               Nin = np.array([0,0,0,0,0,0,0,0,0,0.1,0,0,0,0,0,0.1,0.1,0.1,0.1,0,0,0,0,0]),
                               Nout = np.array([0,0,0,0,0,0,0,0.2,0.2,0,0,0,0,0,0,0.1,0,0,0,0,0,0,0,0]),
                               Nin_weekend = np.array([0,0,0,0,0,0,0,0.0,0.0,0,0,0,0,0,0,0.0,0,0,0,0,0,0,0,0]),
                               Nout_weekend = np.array([0,0,0,0,0,0,0,0.0,0.0,0,0,0,0,0,0,0.0,0,0,0,0,0,0,0,0]),
                               name = 'Domestic1')


#Define Multiple Fleet Object
MultsFleets = aggEV.MultipleAggregatedEVs([Dom1])

#Define Demand
demand = np.asarray(get_GB_demand(ymin,ymax,list(range(1,13)),False,False))


#Storage Units, 100GWh Batteries, 1 TWh Hydrogen
B = BatteryStorageModel(capacity = 100000)
H = HydrogenStorageModel(capacity = 1000000)

Mult_Stor = MultipleStorageAssets([B,H])

# ### Causal ####
# x1 = Mult_Stor.causal_system_operation(demand,power,[2,3,0,1],[0,1,3,2],MultsFleets, start = datetime(ymin,1,1,0),end =datetime(ymax+1,1,1,0),plot_timeseries = True,V2G_discharge_threshold = 20.0,initial_SOC=[0.5,0.75,0.6,1])
# print(x1)


# #### Non Causal ####
x2 = Mult_Stor.non_causal_system_operation(demand,power,MultsFleets,start = datetime(ymin,1,1,0), end =datetime(ymax+1,1,1,0), plot_timeseries = True,InitialSOC=[0.5,0.75,0.6,1])
print(x2)

