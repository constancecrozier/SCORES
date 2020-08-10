import datetime
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import copy

def offset(x):
    sf = 1+0.1*random.random()
    for i in range(len(x)):
        x[i] = x[i]*sf
    return x
        
        
def get_filename(sites,code,year_min,year_max,months):
    mnths = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
             7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    fn = code+'_'
    if sites == 'all' or sites[:2] == 'lf':
        fn +=sites
    else:
        return ''
    fn += '_'+str(year_min)+'to'+str(year_max)
    if len(months) < 12:
        fn += mnths[months[0]]+'to'+mnths[months[-1]]

    return fn + '.csv'

def read_analysis_from_file(filename):
    f = open(filename,'r')
    data = f.readlines()
    f.close()

    res = []

    # first get cost, from first line
    i = 0
    while data[0][i] != '£':
        i += 1
    i += 1
    j = copy.deepcopy(i+1)
    while data[0][j] in ['0','1','2','3','4','5','6','7','8','9','.']:
        j += 1
    cost = float(data[0][i:j])
    res.append(['System Cost',cost])

    r = 1
    for i in range(2):
        while data[r][0] != '-':
            r += 1
        r += 1
        
    # generators
    r += 1
    while len(data[r]) > 4:
        i = 0
        while data[r][i] != ':':
            i += 1
        name = data[r][:i]
        i += 2
        j = copy.deepcopy(i)
        while data[r][j] != 'G':
            j += 1
        val = float(data[r][i:j-1])
        res.append([name,val])
        r += 1
        
    for i in range(2):
        while data[r][0] != '-':
            r += 1
        r += 1
        
    # storage
    r += 1
    while len(data[r]) > 4:
        i = 0
        while data[r][i] != ':':
            i += 1
        name = data[r][:i]
        i += 2
        j = copy.deepcopy(i)
        while data[r][j] != 'T':
            j += 1
        val = float(data[r][i:j-1])
        res.append([name,val])
        r += 1

    return res


def get_GB_demand(year_min,year_max,months,electrify_heat=False,evs=False,
                  units='MW'):
    '''
    Gets the hourly GB electricity demand from the specified time range
    '''
    sf = {'MW':1,'GW':1e-3}
    d = datetime.datetime(year_min,1,1)
    df = datetime.datetime(year_max+1,1,1)
    ms = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,
          'SEP':9,'OCT':10,'NOV':11,'DEC':12,'Jan':1,'Feb':2,'Mar':3,'Apr':4,
          'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
    demand = []

    if evs is True:
        wkday = []
        sat = []
        sun = []
        with open('data/ev_demand.csv','r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                wkday.append(float(row[0]))
                sat.append(float(row[1]))
                sun.append(float(row[2]))
    with open('data/demand.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            dt = datetime.datetime(int(row[0][7:]),ms[row[0][3:6]],
                                   int(row[0][:2]))
            if dt < d:
                continue
            if d > df:
                continue
            if ms[row[0][3:6]] not in months:
                continue
            p = float(row[2])*sf[units]

            if evs is True:
                if dt.isoweekday() < 6:
                    p += wkday[dt.hour]
                elif dt.isoweekday == 6:
                    p += sat[dt.hour]
                else:
                    p += sun[dt.hour]
                    
            demand.append(p)

    if electrify_heat is True:

        gas_profile = [3.5,3.4,3.4,3.4,3.4,3.3,3.8,4.3,4.5,4.5,4.5,4.3,4.3,4.2,
                       4.1,4.1,4.3,4.6,4.8,4.9,4.9,4.9,4.5,4.0]
        s = copy.deepcopy(sum(gas_profile))
        for t in range(24):
            gas_profile[t] = gas_profile[t]/s

        cop = [3.25,3.263073005,3.373259762,3.613242784,3.896179966,4.161375212,
               4.35,4.292105263,4.06893039,3.787860781,3.451697793,3.396604414]
        extra = {}

        with open('data/daily_gas.csv','r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                year = int(row[0][:4])
                if year not in extra:
                    extra[year] = []
                mn = int(row[0][5:7])-1
                if mn+1 not in months:
                    continue
                for t in range(24):
                    extra[year].append(gas_profile[t]*float(row[1])*1e-3/cop[mn])
        p_h = []
        for y in range(year_min,year_max+1):
            if y not in extra:
                p_h += extra[2016]
            else:
                p_h += extra[y]

        for t in range(len(demand)):
            demand[t] += p_h[t]
        


            
                

    return demand

def _subplot(x,n):
    if len(x) <= 4:
        plt.subplot(2,2,n)
    elif len(x) <= 6:
        plt.subplot(2,3,n)
    elif len(x) <= 9:
        plt.subplot(3,3,n)
    plt.grid(ls=':')

def result_as_txt(gen_cap,stor_cap,cost,sc):
    r = str(cost)+','
    for g in gen_cap:
        r += str(g)+','
    for s in stor_cap:
        r += str(s*sc)+','
    r += str((1-sum(stor_cap))*sc)+'\n'
    return r

def lambda_i(tsr, b):
    x = (1 / (tsr + 0.08 * b)) - (0.035 / (np.power(b, 3) + 1))
    return 1/x

def c_p(tsr, b):
    c1 = 0.5176
    c2 = 116
    c3 = 0.4
    c4 = 5
    c5 = 21
    c6 = 0.0068
    l_i = lambda_i(tsr, b)
    c_p = (((c2 / l_i) - c3 * b - c4) * c1 * np.exp(-c5 / l_i) +
           c6*tsr)
    return c_p

