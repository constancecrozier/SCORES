import datetime
import csv
import random
import numpy as np

def offset(x):
    sf = 1+0.02*random.random()-0.01
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


def get_demand(year_min,year_max,months,units='MW'):
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
            demand.append(float(row[2])*sf[units])

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

