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

ymin = 2016
ymax = 2016
mm = [1]



w1 = OnshoreWindModel5800(sites=[1],year_min=ymin, year_max=ymax,
                     data_path='data/wind/')

w2 = OnshoreWindModel5800(sites=[16],year_min=ymin, year_max=ymax,
                         data_path='data/wind/')


gen = [w1,w2]
Stors = [stor.BatteryStorageModel(capacity=1),stor.HydrogenStorageModel(capacity=10)]

Dom1 = aggEV.AggregatedEVModel(eff_in=95, eff_out=95, chargertype=np.zeros([3]), chargercost=np.array([4000/25,800/25,50/25]), max_c_rate=10, max_d_rate=10, min_SOC=0, max_SOC=40, number=6000000,initial_number = 0.9, Ein = 20, Eout = 36, Nin = np.array([0,0,0,0,0,0,0,0,0,0.1,0,0,0,0,0,0.1,0.1,0.1,0.1,0,0,0,0,0]),Nout = np.array([0,0,0,0,0,0,0,0.2,0.2,0,0,0,0,0,0,0.1,0,0,0,0,0,0,0,0]),name = 'Domestic1')

MultsFleets = aggEV.MultipleAggregatedEVs([Dom1])

es = ElectricitySystemGB(gen, Stors, year_min = ymin, year_max = ymax,
                         reliability = 99, aggEV_list = MultsFleets)
es.fully_optimise(sum(es.demand)*0.01,fixed_capacities=False)


start = 7000
end = 7500
es.plot_timeseries(start,end)

es.new_analyse()


#test out configuration on a new year
ymin = 2014
ymax = 2014

#I could include this in the fully optimise method
w01 = OnshoreWindModel5800(sites=[1],year_min=ymin, year_max=ymax,
                     data_path='data/wind/')
w01.total_installed_capacity = max(w1.total_installed_capacity,0.01)
w1 = w01

w02 = OnshoreWindModel5800(sites=[16],year_min=ymin, year_max=ymax,
                         data_path='data/wind/')
w02.total_installed_capacity = max(w2.total_installed_capacity,0.01)
w2 = w02

gen = [w1,w2]

es = ElectricitySystemGB(gen, Stors, year_min = ymin, year_max = ymax,
                         reliability = 99, aggEV_list = MultsFleets)

es.fully_optimise(sum(es.demand)*0.01,fixed_capacities=True)
es.plot_timeseries(start,end)
es.new_analyse(filename='log/2015.txt')

print('New Relaibility: ', (1-sum(MultsFleets.Pfos)/sum(es.demand))*100, '%')
MultsFleets.assets[0].plot_timeseries(start,end)
