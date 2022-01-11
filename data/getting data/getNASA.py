import numpy as np
import pandas as pd
import requests
from netCDF4 import Dataset
import timeit
import datetime


################################ FUNCTIONS #####################

def save_merra2_file(URL, FILENAME):
    """From NASA earthdata https://disc.gsfc.nasa.gov/datasets
    Can choose variables, time ranges, geographic ranges on site
    Then download text file with url(s) for specific request
    """
    result = requests.get(URL)
    try:
        result.raise_for_status()
        f = open(FILENAME,'wb')
        f.write(result.content)
        f.close()
        print('contents of URL written to '+FILENAME)
    except:
        print('requests.get() returned an error code '+str(result.status_code))

# put your details in below
payload = {'inUserName': '', 'inUserPass': ''}
url = 'https://disc.gsfc.nasa.gov/auth/log/in'
requests.post(url, data=payload)

i = 0
# replace with your file name
f1 = open('subset_M2I1NXASM_5.12.4_20210618_003229.txt','r')
for url in f1:
    url = url[:-1]
    filename = 'wind'+str(i)+'.nc4'
    if i in range(364,366):
        save_merra2_file(url,filename)
    i += 1
    
i = 0
# replace with your file name
f1 = open('subset_M2T1NXRAD_5.12.4_20210217_180249.txt','r')
for url in f1:
    url = url[:-1]
    filename = 'solar'+str(i)+'.nc4'
    if i in range(210,212):
        save_merra2_file(url,filename)
    i += 1
