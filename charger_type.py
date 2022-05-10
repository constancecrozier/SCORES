#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 15:46:58 2022

@author: cormacomalley
"""

from storage import (BatteryStorageModel, HydrogenStorageModel,
                      MultipleStorageAssets)
from generation import (OffshoreWindModel,SolarModel)
import aggregatedEVs as aggEV
from opt_con_class import (System_LinProg_Model)
import numpy as np
from datetime import datetime
from fns import get_GB_demand
import matplotlib.pyplot as plt
import pandas as pd
'''
Initialise generators
'''
ymin = 2015
ymax = 2015
osw_master = OffshoreWindModel(year_min=ymin, year_max=ymax, sites=[119,174,178,209,364],
                        data_path='data/150m/')

s = SolarModel(year_min=ymin, year_max=ymax, sites=[17,23,24],
                        data_path='data/solar/')

generators = [osw_master,s]
'''
Initialise Fleets
'''
Dom1 = aggEV.AggregatedEVModel(eff_in=95, eff_out=95, chargertype=[0.5,0.5], 
                               chargercost=np.array([2000/20,800/20,50/20]), max_c_rate=10, 
                               max_d_rate=10, min_SOC=0, max_SOC=36, number=400000,
                               initial_number = 0.9, Ein = 20, Eout = 36, 
                               Nin = np.array([0,0,0,0,0,0,0,0,0,0.1,0,0,0,0,0,0.1,0.2,0.1,0.1,0,0,0,0,0]),
                               Nout = np.array([0,0,0,0,0,0,0,0.3,0.2,0,0,0,0,0,0,0.1,0,0,0,0,0,0,0,0]),
                               Nin_weekend = np.array([0,0,0,0,0,0,0,0.0,0.0,0,0,0,0,0,0,0.0,0,0,0,0,0,0,0,0]),
                               Nout_weekend = np.array([0,0,0,0,0,0,0,0.0,0.0,0,0,0,0,0,0,0.0,0,0,0,0,0,0,0,0]),
                               name = 'Domestic1')

Dom_HeavyUse = aggEV.AggregatedEVModel(eff_in=95, eff_out=95, chargertype=[0.5,0.5], 
                                       chargercost=np.array([2000/20,800/20,50/20]), max_c_rate=10, 
                                       max_d_rate=10, min_SOC=0, max_SOC=36, number=4000000,
                                       initial_number = 0.9, Ein = 15, Eout = 36, 
                                       Nin = np.array([0,0,0,0,0,0,0,0,0,0.1,0,0,0,0,0,0.1,0.2,0.3,0.1,0,0,0,0,0]),
                                       Nout = np.array([0,0,0,0,0,0,0.2,0.3,0.2,0,0,0,0,0,0,0.1,0,0,0,0,0,0,0,0]),
                                       Nin_weekend = np.array([0,0,0,0,0,0,0,0,0,0.1,0,0,0,0,0,0.1,0.2,0.1,0.1,0,0,0,0,0]),
                                       Nout_weekend = np.array([0,0,0,0,0,0,0,0.3,0.2,0,0,0,0,0,0,0.1,0,0,0,0,0,0,0,0]),
                                       name = 'Domestic_Heavy')

Work = aggEV.AggregatedEVModel(eff_in=95, eff_out=95, chargertype=[0.5,0.5], 
                                       chargercost=np.array([2000/20,800/20,50/20]), max_c_rate=10, 
                                       max_d_rate=10, min_SOC=0, max_SOC=36, number=2000000,
                                       initial_number = 0.1, Ein = 15, Eout = 36, 
                                       Nin = np.array([0,0,0,0,0,0,0,0.3,0.3,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                                       Nout = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.2,0.3,0.1,0,0,0,0]),
                                       Nin_weekend = np.array([0,0,0,0,0,0,0,0.0,0.0,0,0,0,0,0,0,0.0,0,0,0,0,0,0,0,0]),
                                       Nout_weekend = np.array([0,0,0,0,0,0,0,0.0,0.0,0,0,0,0,0,0,0.0,0,0,0,0,0,0,0,0]),
                                       name = 'Work')

MultsFleets = aggEV.MultipleAggregatedEVs([Dom1,Dom_HeavyUse,Work])

'''
Initialise storage
'''

B = BatteryStorageModel()
H = HydrogenStorageModel()

# Initialise list of storage
storage = [B,H]

'''
Run Case Study
'''
# Initialise electricity sytem with existing GB demand
demand = np.asarray(get_GB_demand(ymin,ymax,list(range(1,13)),False,False))

x = System_LinProg_Model(surplus = -demand,fossilLimit = 0.01,Mult_Stor = MultipleStorageAssets(storage),
                         Mult_aggEV = MultsFleets, gen_list = generators,YearRange = [ymin,ymax])

x.Form_Model(start_EV = datetime(ymin,1,1,0), end_EV = datetime(ymax+1,1,1,0),InitialSOC = [0.75])

#V2G Costs that want to cycle through
Costs = [2000,2500,3000]


Record = []
for iter in range(len(Costs)):
    for k in x.model.FleetIndex:
        x.model.chargercost[k,0] = Costs[iter]/20

    x.Run_Sizing()

    Record.append(x.df_capital)
    
Record = pd.concat(Record,ignore_index=True)
Record['V2G Cost (£)'] = [2000,2500,3000]
Record.to_csv('log/CostSensitivitynew.csv', index=False)