# test scripts
from generation import (OffshoreWindModel, SolarModel, OnshoreWindModel3600,
                        OnshoreWindModel2000, OnshoreWindModel4200,
                        OnshoreWindModel5300, OnshoreWindModel5800, TidalStreamTurbineModel)
from storage import (BatteryStorageModel, HydrogenStorageModel,
                     ThermalStorageModel)
#from maps import SolarMap, OnshoreWindMap, OffshoreWindMap, LoadFactorEstimator
from system import ElectricitySystem, ElectricitySystemGB
import matplotlib.pyplot as plt

'''
USE CASE 1: Runnning individual generation models
'''

# First offshore wind, note that all parameters are optional, but if the query
# hasn't been run before, a data_path is required


# For onshore wind, various turbine sizes are available, this is for 3.6 MW
w = OnshoreWindModel3600(year_min=2013, year_max=2014, sites='all',
                         data_path='data/wind/')

p = w.power_out # time-series of the output

# Note that the solar model is substantially slower than the wind models
s = SolarModel(year_min=2013, year_max=2014, sites='all',
               data_path='data/solar/')

p = s.power_out # time-series of the output

ts = TidalStreamTurbineModel(year_min=2016, year_max=2016, sites='all',
               data_path='data/tidal/')

p = ts.power_out[0:168]
plt.plot(p)

print('Load Factor: ', ts.get_load_factor())

'''
USE CASE 3: Calculate/optimise  the whole system cost
'''
# Initialise list of generators
generators = [OffshoreWindModel(), OnshoreWindModel3600(), SolarModel()]


# Initialise list of storage
storage = [ThermalStorageModel(), BatteryStorageModel(), HydrogenStorageModel()]


# Initialise electricity sytem with existing GB demand
es = ElectricitySystemGB(generators, storage, year_min = 2013, year_max = 2014,
                         reliability = 99)


# get the cost at a specific point
x = [60,60,60,0.1,0.4,0.5] # 60 GW of each generator, 10% thermal 40$ li-ion 50% H2
print(es.cost(x))

# analyse that system (results writen into log folder)
es.analyse(x)

# Search for the optimal system
#es.optimise(stor_cap = [1e-6, 1e-6], min_gen_cap = [20,0,20],
#            max_gen_cap = [90,70,90], tic0=130)

plt.plot(es.demand)