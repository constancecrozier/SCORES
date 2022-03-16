import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import aggregatedEVs as aggEV
import storage as stor
from fns import (_subplot, result_as_txt, get_GB_demand, offset,
                 read_analysis_from_file)
from generation import (OffshoreWindModel, SolarModel, OnshoreWindModel3600,
                        OnshoreWindModel2000, OnshoreWindModel4200,
                        OnshoreWindModel5300, OnshoreWindModel5800)
from system import ElectricitySystem, ElectricitySystemGB

ymin = 2016
ymax = 2016
mm = [1]

site_lat = {}
with open('data/solar/site_locs.csv', 'rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        site_lat[int(row[0])] = np.deg2rad(float(row[1]))

np.zeros(400,2)
for site in range(1,400): 
    if(site)
    s = SolarModel(year_min=ymin, year_max=ymax, sites=[2], data_path='data/solar/')
    rec[site,0] = site
    rec[site,1] = sum(s.power_out)