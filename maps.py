"""
Created: 07/05/2020 by C.CROZIER

File description:
This file contains code that performs analysis using the generation and storage
models.

Pre-requisite modules: csv, numpy, matplotlib, basemap
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from generation import (OffshoreWindModel,OnshoreWindModel,
                        OnshoreWindModel5300, OnshoreWindModel3600,
                        OnshoreWindModel2000, OnshoreWindModel1500,
                        OnshoreWindModel5800, OnshoreWindModel4200, SolarModel)

class LoadFactorEstimator:
    
    def __init__(self, gen_type, data_loc=None):
        self.gen_type = gen_type
        self.load_factors = {}

        self.datapath = data_loc
        
        self.filepath = 'stored_model_runs/'+gen_type+'_load_factors.csv'

        recover = self.check_for_saved_run()
        if recover is False:
            self.calculate_load_factors()

    def check_for_saved_run(self):
        '''
        == description ==
        This function checks to see whether this simulation has been previously
        run, and if so sets power_out to the stored values.

        == parameters ==
        path: (str) location the csv file would stored if it exists

        == returns ==
        True if a previous run has been recovered
        False otherwise
        '''
        try:
            with open(self.filepath, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                n = 1
                for row in reader:
                    self.load_factors[n] = [float(row[0]),float(row[1]),
                                            float(row[2])*100]
                    n += 1 
            return True
        except:
            return False
        
    def store_results(self):
        with open(self.filepath,'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Latitude','Longitude','Load Factor'])
            for n in self.load_factors:
                writer.writerow(self.load_factors[n][:2] +
                                [self.load_factors[n][2]/100])               

    def calculate_load_factors(self):
        if self.datapath is None:
            raise Exception('a data location is required')
        
        gen_model = {'osw':OffshoreWindModel,
                     'w5.8':OnshoreWindModel5800,
                     'w5.3':OnshoreWindModel5300,
                     'w3.6':OnshoreWindModel3600,
                     'w4.2':OnshoreWindModel4200,
                     'w2.0':OnshoreWindModel2000,
                     'w1.5':OnshoreWindModel1500,
                     's':SolarModel}
        locs = {}
        with open(self.datapath+'site_locs.csv','r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                locs[int(row[0])] = [float(row[1]),float(row[2])]

        for site in locs:
            if site == 0:
                continue
            gm = gen_model[self.gen_type](sites=[site],data_path=self.datapath,
                                          save=False,year_min=2016,
                                          year_max=2019)
            
            try:
                lf = gm.get_load_factor()
            except:
                continue
            self.load_factors[site] = locs[site]+[lf]
        self.store_results()

    def estimate(self,lat,lon,max_dist=1,num_pts=3):
        pts = []
        for n in self.load_factors:
            loc = self.load_factors[n][:2]
            d = np.sqrt(np.power(lat-loc[0],2)+np.power(lon-loc[1],2))
            if d > max_dist:
                continue
            pts.append([d,self.load_factors[n][2]])

        pts = sorted(pts)
        f = 0
        n = 0
        if len(pts) < num_pts:
            for i in range(len(pts)):
                w = max_dist-pts[i][0]
                f += pts[i][1]*w
                n += w
        else:
            for i in range(num_pts):
                w = max_dist-pts[i][0]
                f += pts[i][1]*w
                n += w
        if n == 0:
            f = None
        else:
            f = f/n
        return f
        
# first here is the code for drawing maps
class LoadFactorMap:

    def __init__(self, load_factor_estimator, lat_min, lat_max, lon_min,
                 lon_max, lat_num, lon_num, quality, is_land):
        self.load_factor_estimator = load_factor_estimator
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lon_num = lon_num
        self.lat_num = lat_num
        self.quality = quality
        self.is_land = is_land
        
    def draw_map(self, show=True, savepath='', cmap=None, vmax=None, vmin=None):
        if cmap is None:
            # get standard spectral colormap
            spec = cm.get_cmap('Spectral', 1000)
            # reverse so that red is largest
            new = spec(np.linspace(0, 1, 1000))
            new = np.array(list(reversed(new)))
            # set zero to be white so that unknown areas will not be shaded
            new[:1,:] =  np.array([1,1,1,1])
            cmap = ListedColormap(new)
            
        # create new figure, axes instances.
        fig=plt.figure(figsize=(6,8))
        plt.rcParams["font.family"] = 'serif'
        plt.rcParams['font.size'] = 12
        ax=fig.add_axes([0.1,0.1,0.8,0.8])
        # setup mercator map projection.
        m = Basemap(llcrnrlon=self.lon_min, llcrnrlat=self.lat_min,
                    urcrnrlon=self.lon_max, urcrnrlat=self.lat_max,
                    resolution=self.quality, projection='merc',
                    lat_0=40., lon_0=-20., lat_ts=20.)

        x = np.linspace(self.lon_min,self.lon_max,num=self.lon_num)
        y = np.linspace(self.lat_min,self.lat_max,num=self.lat_num)

        Z = np.zeros((len(x),len(y)))
        X = np.zeros((len(x),len(y)))
        Y = np.zeros((len(x),len(y)))
        minz=100
        maxz=0
        m.drawcoastlines()

        for i in range(len(x)):
            for j in range(len(y)):
                xpt,ypt = m(x[i],y[j])
                X[i,j] = xpt
                Y[i,j] = ypt
                if m.is_land(xpt,ypt) == self.is_land:
                    if self.is_land is True:
                        # Ireland
                        if ((xpt < 200000) and (ypt < 930000) and
                            (ypt > 340000)):
                            continue
                        # France
                        if xpt > 930000 and ypt < 190000:
                            continue
                    Z[i,j] = self.load_factor_estimator.estimate(y[j],x[i])
                    if Z[i,j] > maxz:
                        maxz = Z[i,j]
                    if Z[i,j] < minz:
                        minz = Z[i,j]
                else:
                    Z[i,j] = None

        if vmin is None:
            vmin = minz*0.99
        if vmax is None:
            vmax = maxz

        m.pcolor(X,Y,Z,vmin=vmin,vmax=vmax,cmap=cmap)
        plt.colorbar()

        if savepath != '':
            plt.savefig(savepath+'lf_map.pdf', format='pdf',
                        dpi=300, bbox_inches='tight', pad_inches=0)
        if show is True:
            plt.show()

class OffshoreWindMap(LoadFactorMap):

    def __init__(self, lat_min=48.2, lat_max=61.2, lon_min=-10.0, 
                 lon_max=4.0, lat_num=400, lon_num=300, quality='h',
                 data_loc=None):
        lfe = LoadFactorEstimator('osw',data_loc=data_loc)
        
        super().__init__(lfe, lat_min, lat_max, lon_min, lon_max, 
                         lat_num, lon_num, quality, is_land=False)


class OnshoreWindMap(LoadFactorMap):

    def __init__(self, lat_min=49.9, lat_max=59.0, lon_min=-7.5, lon_max=2.0,
                 lat_num=400, lon_num=300, quality='h', turbine_size=3.6,
                 data_loc=None):
        lfe = LoadFactorEstimator('w'+str(float(turbine_size)),
                                  data_loc=data_loc)
        
        super().__init__(lfe, lat_min, lat_max, lon_min, lon_max, 
                         lat_num, lon_num, quality, is_land=True)


class SolarMap(LoadFactorMap):

    def __init__(self, lat_min=49.9, lat_max=59.0, lon_min=-7.5, lon_max=2.0,
                 lat_num=400, lon_num=300, quality='h',data_loc=None):
        lfe = LoadFactorEstimator('s',data_loc=data_loc)
        
        super().__init__(lfe, lat_min, lat_max, lon_min, lon_max,
                         lat_num, lon_num, quality, is_land=True)

#class StorageSizing:
