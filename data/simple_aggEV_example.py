from generation import (OffshoreWindModel)
import aggregatedEVs as aggEV
from opt_con_class import (System_LinProg_Model)
import numpy as np

ymin = 2015
ymax = 2015

#Define the generators
osw_master = OffshoreWindModel(year_min=ymin, year_max=ymax, 
                               sites=[119,174,178,209,364], data_path='data/150m/')

generators = [osw_master]

#EVs
Dom1 = aggEV.AggregatedEVModel(eff_in=95, eff_out=95, chargertype=[0.5,0.5], chargercost=np.array([2000/20,800/20,50/20]), 
                               max_c_rate=10, max_d_rate=10, min_SOC=0, max_SOC=36, number=50000,initial_number = 0.9, 
                               Ein = 20, Eout = 36, 
                               Nin = np.array([0,0,0,0,0,0,0,0,0,0.1,0,0,0,0,0,0.1,0.1,0.1,0.1,0,0,0,0,0]),
                               Nout = np.array([0,0,0,0,0,0,0,0.2,0.2,0,0,0,0,0,0,0.1,0,0,0,0,0,0,0,0]),
                               name = 'Domestic1')

MultsFleets = aggEV.MultipleAggregatedEVs([Dom1])

#Define flat demand
demand = -15.0* np.ones(len(osw_master.power_out))

#Form Model and solve allowing 2% of demand from fossil fuels
x = System_LinProg_Model(surplus = demand,fossilLimit = 0.02,Mult_aggEV = MultsFleets, gen_list=generators)
x.Form_Model()
x.Run_Sizing()

x.df_capital.to_csv('SomeFossil.csv', index=False)

MultsFleets.assets[0].plot_timeseries(start = 6600, end =6700,withSOClimits=True)

#Resolve Allowing no Fossil Fuels
x.model.foss_lim_param = 0.0
x.Run_Sizing()
x.df_capital.to_csv('NoFossil.csv', index=False)

MultsFleets.assets[0].plot_timeseries(start = 6600, end =6700, withSOClimits=False)

