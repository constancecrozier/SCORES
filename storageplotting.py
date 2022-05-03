from generation import (OffshoreWindModel)
import aggregatedEVs as aggEV
from opt_con_class import (System_LinProg_Model)
import numpy as np
from storage import (BatteryStorageModel, HydrogenStorageModel,
                      MultipleStorageAssets)

ymin = 2015
ymax = 2015

#Define the generators
osw_master = OffshoreWindModel(year_min=ymin, year_max=ymax, 
                               sites=[119,174,178,209,364], data_path='data/150m/')

generators = [osw_master]

#EVs
B = BatteryStorageModel(capacity = 100000)
H = HydrogenStorageModel(capacity = 1000000)

B.limits = [10,10]

Mult_Stor = MultipleStorageAssets([B,H])

#Define flat demand
demand = -15.0* np.ones(len(osw_master.power_out))

#Form Model and solve allowing 2% of demand from fossil fuels
x = System_LinProg_Model(surplus = demand,fossilLimit = 0.02,Mult_Stor = Mult_Stor, gen_list=generators)
x.Form_Model()
x.Run_Sizing()

x.df_capital.to_csv('log/SomeFossil.csv', index=False)


print('discharge shape:', Mult_Stor.assets[0].discharge.shape)
for i in range(Mult_Stor.n_assets):
    Mult_Stor.assets[i].plot_timeseries(start = 0, end =1000)
    Mult_Stor.assets[i].plot_timeseries(start = 6000, end =-1)
