# -*- coding: utf-8 -*-

import time
from generation import (OffshoreWindModel, SolarModel, OnshoreWindModel)
from storage import (BatteryStorageModel, HydrogenStorageModel,
                      MultipleStorageAssets)
from system import ElectricitySystem, ElectricitySystemGB
import csv
import numpy as np
import matplotlib.pyplot as plt
import aggregatedEVs as aggEV

gen = OnshoreWindModel(turbine_size=3.6, year_min = 2014, year_max = 2014, sites = [20],
                              data_path = 'data/wind/')

wind_pow = np.array(gen.power_out)

surplus = wind_pow - np.ones([365*24])

stor = MultipleStorageAssets([BatteryStorageModel(capacity=100),HydrogenStorageModel(capacity=10)])

print(stor.charge_specfied_order(surplus, c_order = range(2),d_order = range(2)))

print(stor.analyse_usage())

