# -*- coding: utf-8 -*-

'''
This script is to demonstrate the ability of SCORES to Optimise over a large Number of different renewable energy sources
storage types.
'''

from generation import (OffshoreWindModel,SolarModel)
import aggregatedEVs as aggEV
from opt_con_class import (System_LinProg_Model)
from storage import (BatteryStorageModel, HydrogenStorageModel,
                      MultipleStorageAssets)
import numpy as np
from fns import get_GB_demand
import datetime as dt

ymin = 2013
ymax = 2013

#Define the generators. Sites with particularly good resources are chosen.
osw1 = OffshoreWindModel(year_min=ymin, year_max=ymax, 
                               sites=[119], data_path='data/150m/')
osw1.name = 'W1'

osw2 = OffshoreWindModel(year_min=ymin, year_max=ymax, 
                               sites=[174], data_path='data/150m/')
osw2.name = 'W2'

osw3 = OffshoreWindModel(year_min=ymin, year_max=ymax, 
                               sites=[178], data_path='data/150m/')
osw3.name = 'W3'

osw4 = OffshoreWindModel(year_min=ymin, year_max=ymax, 
                               sites=[209], data_path='data/150m/')
osw4.name = 'W4'

osw5 = OffshoreWindModel(year_min=ymin, year_max=ymax, 
                               sites=[364], data_path='data/150m/')
osw5.name = 'W5'

s1 = SolarModel(year_min=ymin, year_max=ymax, sites=[17],
                        data_path='data/solar/')
s1.name = 's1'

s2 = SolarModel(year_min=ymin, year_max=ymax, sites=[23],
                        data_path='data/solar/')
s2.name = 's2'

s3 = SolarModel(year_min=ymin, year_max=ymax, sites=[24],
                        data_path='data/solar/')
s3.name = 's3'



generators = [s1,s2,s3,osw1,osw2,osw3,osw4,osw5]

#Define the Storage
H = HydrogenStorageModel()
B = BatteryStorageModel()

storage = [B,H]


#Define Demand
demand = np.asarray(get_GB_demand(ymin,ymax,list(range(1,13)),False,False))

#Define EVs
'''
Initialise Fleets
'''
Dom1 = aggEV.AggregatedEVModel(eff_in=95, eff_out=95, chargertype=[0.5,0.5], 
                               chargercost=np.array([2000/20,800/20,50/20]), max_c_rate=10, 
                               max_d_rate=10, min_SOC=0, max_SOC=36, number=4000000,
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
                                       initial_number = 0.01, Ein = 15, Eout = 36, 
                                       Nin = np.array([0,0,0,0,0,0,0,0.3,0.3,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                                       Nout = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.2,0.3,0.1,0,0,0,0]),
                                       Nin_weekend = np.array([0,0,0,0,0,0,0,0.0,0.0,0,0,0,0,0,0,0.0,0,0,0,0,0,0,0,0]),
                                       Nout_weekend = np.array([0,0,0,0,0,0,0,0.0,0.0,0,0,0,0,0,0,0.0,0,0,0,0,0,0,0,0]),
                                       name = 'Work')

MultsFleets = aggEV.MultipleAggregatedEVs([Dom1,Dom_HeavyUse,Work])


x = System_LinProg_Model(surplus = -demand,fossilLimit = 0.02,Mult_Stor = MultipleStorageAssets(storage), Mult_aggEV = MultsFleets, gen_list=generators)
x.Form_Model(start_EV = dt.datetime(ymin,1,1,0),end_EV = dt.datetime(ymax+1,1,1,0))
x.Run_Sizing()

x.PlotSurplus(0,336)
B.plot_timeseries(0,336)
H.plot_timeseries(0,336)
Dom1.plot_timeseries(0,336,True)
Dom_HeavyUse.plot_timeseries(0,336,True)
Work.plot_timeseries(0,336,True)


#Store Results
x.df_capital.to_csv('log/Capital_MultLoc.csv', index=False)
x.df_costs.to_csv('log/Costs_MultLoc.csv', index=False)
