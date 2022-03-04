from generation import (OffshoreWindModel, SolarModel, OnshoreWindModel3600,
                        OnshoreWindModel2000, OnshoreWindModel4200,
                        OnshoreWindModel5300, OnshoreWindModel5800)
from system import ElectricitySystem, ElectricitySystemGB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import aggregatedEVs as aggEV
import storage as stor
from fns import (_subplot, result_as_txt, get_GB_demand, offset,
                 read_analysis_from_file)

ymin = 2014
ymax = 2014
mm = [1]

w1 = OnshoreWindModel5800(sites=[1],year_min=ymin, year_max=ymax,
                         data_path='data/wind/')

w2 = OnshoreWindModel5800(sites=[16],year_min=ymin, year_max=ymax,
                         data_path='data/wind/')
#w1.power_out = w1.power_out/max(w1.power_out)
plt.plot(w1.power_out)

print(w1.fixed_cost, w1.variable_cost)

gen = [w1,w2]
Stors = [stor.BatteryStorageModel(capacity=1),stor.HydrogenStorageModel(capacity=10)]

print(Stors[0].fixed_cost,Stors[1].fixed_cost,Stors[1].variable_cost)

Dom1 = aggEV.AggregatedEVModel(eff_in=95, eff_out=95, chargertype=np.zeros([3]), chargercost=np.array([4000/25,800/25,50/25]), max_c_rate=10, max_d_rate=10, min_SOC=0, max_SOC=40, number=6000000,initial_number = 0.9, Ein = 20, Eout = 36, Nin = np.array([0,0,0,0,0,0,0,0,0,0.1,0,0,0,0,0,0.1,0.1,0.1,0.1,0,0,0,0,0]),Nout = np.array([0,0,0,0,0,0,0,0.2,0.2,0,0,0,0,0,0,0.1,0,0,0,0,0,0,0,0]),name = 'Domestic1')

MultsFleets = aggEV.MultipleAggregatedEVs([Dom1])

es = ElectricitySystemGB(gen, Stors, year_min = ymin, year_max = ymax,
                         reliability = 99, aggEV_list = MultsFleets)
es.fully_optimise(sum(es.demand)*0.1)



start = 7000
end = 7500
es.plot_timeseries(start,end)
MultStors = stor.MultipleStorageAssets(Stors)
MultStors.assets[0].plot_timeseries(start,end)
MultStors.assets[1].plot_timeseries(start,end)


es.new_analyse()
