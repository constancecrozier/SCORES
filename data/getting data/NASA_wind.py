import csv
from netCDF4 import Dataset
import datetime
import numpy as np

d = datetime.datetime(2011,1,1)
df = datetime.datetime(2020,1,1)


sites = {}
for i in range(20):
    for j in range(16):
        sites[i*16+j] = []
locs = {}

while d < df:
    day = str(d.year)
    if d.month <10:
        day += '0'+str(d.month)
    else:
        day += str(d.month)
    if d.day < 10:
        day += '0'+str(d.day)
    else:
        day += str(d.day)
    data = Dataset('wind/'+day+'.nc4', mode='r')
    U50M = data.variables['U50M']
    V50M = data.variables['V50M']

    if len(locs) == 0:
        lons = data.variables['lon'][:]
        lats = data.variables['lat'][:]
        for i in range(20):
            for j in range(16):
                locs[i*16+j] = [lats[i],lons[j]]
    for i in range(20):
        for j in range(16):
            for t in range(24):
                ws = np.sqrt(U50M[t,i,j]*U50M[t,i,j]+V50M[t,i,j]*V50M[t,i,j])
                sites[i*16+j].append([d,t+1,ws])
    data.close()
    d += datetime.timedelta(1)
    print(d)

with open('wind2/site_locs.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Site','Latitude','Longitude'])
    for i in locs:
        writer.writerow([i+1]+locs[i])

for i in sites:
    with open('wind2/'+str(i+1)+'.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Date','Hour','Wind Speed m/s'])
        for row in sites[i]:
            writer.writerow(row)

