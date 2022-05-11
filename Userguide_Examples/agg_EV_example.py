'''
Optimises the charger types for a small fleet of EVs.
'''

from generation import (OffshoreWindModel,SolarModel)
import aggregatedEVs as aggEV
from opt_con_class import (System_LinProg_Model)
from storage import (MultipleStorageAssets)
import numpy as np
import datetime as dt
from fns import get_GB_demand

ymin = 2015
ymax = 2015

#Define the generators
osw_master = OffshoreWindModel(year_min=ymin, year_max=ymax, 
                               sites=[119,174,178,209,364], data_path='data/150m/')

s = SolarModel(year_min=ymin, year_max=ymax, sites=[17,23,24],
                        data_path='data/solar/')
generators = [s,osw_master]

#Define a Fleet of EVs
Dom1 = aggEV.AggregatedEVModel(eff_in=95, eff_out=95, chargertype=[0.5,0.5], chargercost=np.array([2000/20,800/20,50/20]), 
                               max_c_rate=10, max_d_rate=10, min_SOC=0, max_SOC=36, number=20000,initial_number = 0.9, 
                               Ein = 20, Eout = 36, 
                               Nin = np.array([0,0,0,0,0,0,0,0,0,0.1,0,0,0,0,0,0.1,0.1,0.1,0.1,0,0,0,0,0]),
                               Nout = np.array([0,0,0,0,0,0,0,0.2,0.2,0,0,0,0,0,0,0.1,0,0,0,0,0,0,0,0]),
                               Nin_weekend = np.array([0,0,0,0,0,0,0,0.0,0.0,0,0,0,0,0,0,0.0,0,0,0,0,0,0,0,0]),
                               Nout_weekend = np.array([0,0,0,0,0,0,0,0.0,0.0,0,0,0,0,0,0,0.0,0,0,0,0,0,0,0,0]),
                               name = 'Domestic1')

#Define Multiple Fleet Object
MultsFleets = aggEV.MultipleAggregatedEVs([Dom1])

#Define Demand, normalised to max 15MW
demand = np.asarray(get_GB_demand(ymin,ymax,list(range(1,13)),False,False))
demand = - demand/max(demand) * 30.0

#Form Model and solve allowing 2% of demand from fossil fuels
x = System_LinProg_Model(surplus = demand,fossilLimit = 0.02,Mult_Stor = MultipleStorageAssets([]), Mult_aggEV = MultsFleets, gen_list=generators)
x.Form_Model(start_EV = dt.datetime(ymin,1,1,0),end_EV = dt.datetime(ymax+1,1,1,0))
x.Run_Sizing()

#Save Output DataFrame to csv
x.df_capital.to_csv('log/SomeFossil.csv', index=False)

#Plot A Week in December
MultsFleets.assets[0].plot_timeseries(start = 8160, end =8350,withSOClimits=True)
x.PlotSurplus(start = 8160, end =8350)


