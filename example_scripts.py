# test scripts
from generation import (SolarModel, OnshoreWindModel3600, OffshoreWindModel10000)
from storage import (BatteryStorageModel, HydrogenStorageModel,
                     ThermalStorageModel)
from maps import SolarMap, OnshoreWindMap, OffshoreWindMap, LoadFactorEstimator
from system import ElectricitySystem, ElectricitySystemGB

'''
USE CASE 1: Runnning individual generation models
'''

# First offshore wind, note that all parameters are optional, but if the query
# hasn't been run before, a data_path is required

osw = OffshoreWindModel10000(year_min=2013, year_max=2019, sites='all',
                        data_path='data/offshore_wind/')

p = osw.power_out # time-series of the output

# For onshore wind, various turbine sizes are available, this is for 3.6 MW
w = OnshoreWindModel3600(year_min=2013, year_max=2019, sites='all',
                         data_path='data/wind/')

p = w.power_out # time-series of the output

# Note that the solar model is substantially slower than the wind models
s = SolarModel(year_min=2013, year_max=2019, sites='all',
               data_path='data/solar/')

p = s.power_out # time-series of the output


'''
USE CASE 2: Estimating load factors at particular point
'''

# load fator maps
s_map = SolarMap(lat_num=40, lon_num=30, quality='l',data_loc='data/solar/')

s_map.draw_map()


w_map = OnshoreWindMap(lat_num=40, lon_num=30, quality='l',turbine_size=5.8,
                       data_loc='data/wind/')
w_map.draw_map()


osw_map = OffshoreWindMap(lat_num=40, lon_num=30, quality='l',
                          data_loc='data/offshore_wind/')

osw_map.draw_map()

# load factor estimation at a specific location
lfe = LoadFactorEstimator('s') # s - code for solar

lat = 51.48 # greenwich park location
lon = 0.00

load_factor = lfe.estimate(lat,lon)


'''
USE CASE 3: Calculate/optimise  the whole system cost
'''
# Initialise list of generators
generators = [OffshoreWindModel(), OnshoreWindModel3600(), SolarModel()]


# Initialise list of storage
storage = [ThermalStorageModel(), BatteryStorageModel(), HydrogenStorageModel()]


# Initialise electricity sytem with existing GB demand
es = ElectricitySystemGB(generators, storage, year_min = 2013, year_max = 2019,
                         reliability = 99)


# get the cost at a specific point
x = [60,60,60,0.1,0.4] # 60 GW of each generator, 10% thermal 40$ li-ion 50% H2
print(es.cost(x))

# analyse that system (results writen into log folder)
es.analyse(x)

# Search for the optimal system
es.optimise(stor_cap = [1e-6, 1e-6], min_gen_cap = [20,0,20],
            max_gen_cap = [90,70,90], tic0=130)

