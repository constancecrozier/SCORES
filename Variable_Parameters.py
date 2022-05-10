#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 14:03:52 2022

@author: cormacomalley
"""

from generation import (OffshoreWindModel,SolarModel)
import aggregatedEVs as aggEV
from opt_con_class import (System_LinProg_Model)
from storage import (BatteryStorageModel, HydrogenStorageModel,
                      MultipleStorageAssets)
import numpy as np
from fns import get_GB_demand
import pandas as pd

ymin = 2012
ymax = 2016

#Define the generators
osw_master = OffshoreWindModel(year_min=ymin, year_max=ymax, 
                               sites=[119,174,178,209,364], data_path='data/150m/')

s = SolarModel(year_min=ymin, year_max=ymax, sites=[17,23,24],
                        data_path='data/solar/')
generators = [s,osw_master]

#Define the Storage
H = HydrogenStorageModel()
B = BatteryStorageModel()
storage = [B,H]


#Define Demand
demand = np.asarray(get_GB_demand(ymin,ymax,list(range(1,13)),False,False))


SolarCost = [40000,60000,80000]
Capital_Record = []

x = System_LinProg_Model(surplus = -demand,fossilLimit = 0.01,Mult_Stor = MultipleStorageAssets(storage),
                         Mult_aggEV = aggEV.MultipleAggregatedEVs([]), gen_list = generators,YearRange = [ymin,ymax])

#### Naive Method ####

# for i in range(len(SolarCost)):
#     s.fixed_cost = SolarCost[i]
#     x.Form_Model()
#     x.Run_Sizing()
#     Capital_Record.append(x.df_capital)
    
# Capital_Record = pd.concat(Capital_Record, ignore_index=True)
# Capital_Record['Solar Cost (£/MW/yr)'] = SolarCost
# Capital_Record.to_csv('log/SolPrice.csv', index=False)


#### Fast Method ####
x.Form_Model()

for i in range(len(SolarCost)):
    x.model.GenCosts[0,0] = SolarCost[i]  #First 0 refers to solar generator
                                        #Second 0 refers to fixed_cost
    x.Run_Sizing()
    Capital_Record.append(x.df_capital)
    
Capital_Record = pd.concat(Capital_Record, ignore_index=True)
Capital_Record['Solar Cost (£/MW/yr)'] = SolarCost
Capital_Record.to_csv('log/SolPrice_Fast.csv', index=False)
















# # #Solve the Linear Program
# # x.Run_Sizing()

# # #Plot Results
# # x.PlotSurplus(0,336)
# # B.plot_timeseries(0,336)
# # H.plot_timeseries(0,336)

# # #Store Results
# # x.df_capital.to_csv('log/Capital.csv', index=False)
# # x.df_costs.to_csv('log/Costs.csv', index=False)


# #Sensitivity to amount of Fossil Fuels
# FosLim=[0.04,0.02,0.0]
# Cap_Record = []
# Cost_Record = []

# for b in range(len(FosLim)):
#     x.model.foss_lim_param = FosLim[b] * sum(demand)
#     x.Run_Sizing()
#     Cap_Record.append(x.df_capital)
#     Cost_Record.append(x.df_costs)
  
# Cap_Record1 = pd.concat(Cap_Record,ignore_index=True)
# Cost_Record1 = pd.concat(Cost_Record,ignore_index=True)

# Cost_Record1.to_csv('log/Cost_Rec_1.csv', index=False)
# Cap_Record1.to_csv('log/Cap_Rec_1.csv', index=False)

# #Sensitivity to Limit Solar
# Max_Solar = [75000,50000,25000]
# for s in range(len(Max_Solar)):
#     x.model.Gen_Limit_Param_Upper[0] = Max_Solar[s]
#     x.Run_Sizing()
#     Cap_Record.append(x.df_capital)
#     Cost_Record.append(x.df_costs)
    
#     FosLim.append(0.0)

# Cap_Record2 = pd.concat(Cap_Record,ignore_index=True)
# Cost_Record2 = pd.concat(Cost_Record,ignore_index=True)

# Cap_Record2['Fos Lmit (%)'] = FosLim
# Cap_Record2['Max Solar (GW)'] = [400,400,400,75,50,25]

# Cost_Record2['Fos Lmit (%)'] = FosLim
# Cost_Record2['Max Solar (GW)'] = [400,400,400,75,50,25]

# Cost_Record2.to_csv('log/Cost_Rec_2.csv', index=False)
# Cap_Record2.to_csv('log/Cap_Rec_2.csv', index=False)