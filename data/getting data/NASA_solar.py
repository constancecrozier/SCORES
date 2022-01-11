import csv
from netCDF4 import Dataset
import datetime

d = datetime.datetime(2011,1,1)
df = datetime.datetime(2020,1,1)

# NOTE: 05/09/2010 missing

sites = {}
for i in range(19):
    for j in range(17):
        sites[i*17+j] = []
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
    data = Dataset('solar/'+day+'.nc4', mode='r')
    SWGDN = data.variables['SWGDN'] #surface incoming shortwave flux [W/m^2]

    if len(locs) == 0:
        lons = data.variables['lon'][:]
        lats = data.variables['lat'][:]
        for i in range(19):
            for j in range(17):
                locs[i*17+j] = [lats[i],lons[j]]
    for i in range(19):
        for j in range(17):
            for t in range(24):
                sites[i*17+j].append([d,t+1,SWGDN[t,i,j]*3.6])
    data.close()
    d += datetime.timedelta(1)
    print(d)

with open('solar2/site_locs.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Site','Latitude','Longitude'])
    for i in locs:
        writer.writerow([i+1]+locs[i])

for i in sites:
    with open('solar2/'+str(i+1)+'.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Date','Hour','Irradiance kJ/m^2'])
        for row in sites[i]:
            writer.writerow(row)

