# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 15:28:26 2022

@author: C Quarton
"""

import xarray as xr
import csv
import datetime as dt

datafile = 'cmems_mod_nws_phy-uv_my_7km-2D_PT1H-i_Anglesey.nc'

d = xr.open_dataset(datafile)
latitudes = d.latitude.data
longitudes = d.longitude.data
timepoints = d.time.data


counter = 1
filename = 'site_locs.csv'
with open(filename,'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Site','Latitude','Longitude'])
    for lat in range(len(latitudes)):
        for lon in range(len(longitudes)):
            writer.writerow([counter,latitudes[lat],longitudes[lon]])
            counter = counter+1


counter = 1
for lat in range(len(latitudes)):
    for lon in range(len(longitudes)):
        filename = str(counter)+'.csv'
        uo = d.uo.data[:,lat,lon]
        vo = d.vo.data[:,lat,lon]

        with open(filename,'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Year','Month','Day','Hour','','','Ro'])
            timecounter = dt.datetime(2013,1,1,hour = 0)
            for i in range(len(uo)):
                Ro = (uo[i]**2+vo[i]**2)**0.5
                writer.writerow([timecounter.year,timecounter.month,timecounter.day,timecounter.hour,' ',' ',Ro])
                timecounter = timecounter + dt.timedelta(hours = 1)
        counter = counter + 1
    

        

