"""
Created: 28/07/2020 by C.CROZIER

File description: This file contains the classes for a single type of energy
storage and an aggregated portfolio of storage assets.

Notes: The storage base class is technologically agnostic, but child classes are
icluded that are parameterised for Li-Ion, hydrogen, and thermal storage.
"""
import copy
import numpy as np
import datetime
import matplotlib.pyplot as plt
import optimise_configuration as opt_con
#optimisation high level language, help found at https://www.ima.umn.edu/materials/2017-2018.2/W8.21-25.17/26326/3_PyomoFundamentals.pdf
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import aggregatedEVs as aggEV

class StorageModel:
    def __init__(self, eff_in, eff_out, self_dis, variable_cost, fixed_cost,
                 max_c_rate, max_d_rate, name, capacity=1, limits = []):
        '''
        == description ==
        .

        == parameters ==
        eff_in: (float) charging efficiency in % (0-100)
        eff_out: (float) discharge efficiency in % (0-100)
        self_dis: (float) self discharge rate in % per month (0-100)
        fixed_cost: (float) cost incurred per MWh-year of installation in GBP
        variable_cost: (float) cost incurred per MWh of throughput in GBP
        max_c_rate: (float) the maximum charging rate (% per hour)
        max_d_rate: (float) the maximum discharging rate (% per hour)
        name: (str) the name of the asset - for use in graph plotting
        capacity: (float) MWh of storage installed
        limits: array[(float)] the [min,max] capacity in MWh

        NOTE: both c and d rate defined FROM THE GRID SIDE

        == returns ==
        None
        '''
        self.eff_in = eff_in
        self.eff_out = eff_out
        self.self_dis = self_dis
        self.variable_cost = variable_cost
        self.fixed_cost = fixed_cost
        self.max_c_rate = max_c_rate
        self.max_d_rate = max_d_rate
        self.capacity = capacity
        self.name = name
        self.limits = limits

        # These will be used to monitor storage usage
        self.en_in = 0 # total energy into storage (grid side)
        self.en_out = 0 # total energy out of storage (grid side)
        self.curt = 0 # total supply that could not be stored
        
        # from optimise setting only (added by Mac)
        self.discharge = np.empty([]) #timeseries of discharge rate (grid side) MW
        self.charge = np.empty([]) #timeseries of charge rate (grid side) MW
        self.SOC = np.empty([]) #timeseries of Storage State of Charge (SOC) MWh

    def reset(self):
        '''
        == description ==
        Resets the parameters recording the use of the storage assets.

        == parameters ==
        None

        == returns ==
        None
        '''
        self.en_in = 0
        self.en_out = 0
        self.curt = 0
        
        self.discharge = np.empty([]) 
        self.charge = np.empty([]) 
        self.SOC = np.empty([]) 

    def set_capacity(self, capacity):
        '''
        == description ==
        Sets the installed  storage capacity to the specified value.

        == parameters ==
        capacity: (float) MWh of storage installed

        == returns ==
        None
        '''
        self.capacity = capacity
        self.reset()

    def get_cost(self):
        '''
        == description ==
        Gets the total cost of running the storage system.

        == parameters ==
        None

        == returns ==
        (float) cost in GBP/yr of the storage unit
        '''
        if self.capacity == np.inf:
            return np.inf
        else:
            return (self.capacity*self.fixed_cost
                    + self.en_out*self.variable_cost*100/(self.eff_out
                                                          *self.n_years))

    def plot_timeseries(self,start=0,end=-1):
            
        '''   
        == parameters ==
        start: (int) start time of plot
        end: (int) end time of plot
        '''
        
        if(self.discharge.shape == ()):
            print('Charging timeseries not avaialable, try running MultipleStorageAssets.optimise_storage().')
        else:
            if(end<=0):
                timehorizon = self.discharge.size
            else:
                timehorizon = end
            plt.rc('font', size=12)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(int(start),int(timehorizon+1)), self.SOC[int(start):int(timehorizon+1)], color='tab:red', label='SOC')
            ax.plot(range(start,timehorizon), self.charge[start:timehorizon], color='tab:blue', label='Charge')
            ax.plot(range(start,timehorizon), self.discharge[start:timehorizon], color='tab:orange', label='Discharge')

            # Same as above
            ax.set_xlabel('Time (h)')
            ax.set_ylabel('Power (MW), Energy (MWh)')
            ax.set_title(self.name+' Charging Timeseries')
            ax.grid(True)
            ax.legend(loc='upper left');
        
    
    def self_discharge_timestep(self):
        '''
        == description ==
        Reduces stored charge due to self-discharge over one time-step

        == parameters ==
        None

        == returns ==
        None
        '''
        # conversion factors because self.dis measured in %/month not MWh/hr
        self.charge -= (self.self_dis*self.capacity)*self.t_res/(100*24*30)
        if self.charge < 0:
            self.charge = 0.0
    
    def charge_timestep(self, t, surplus):
        '''
        == description ==
        Charges the asset for one timestep - either until all the surplus is
        used, the asset is full, or the charging rate limit is reached (which
        ever comes first)

        == parameters ==
        t: (int) the current timestep - so that the output vector can be updated
        suplus: (float) the excess available energy in MWh for that timestep

        == returns ==
        None
        '''
        # amount required to fill storage
        to_fill = (copy.deepcopy(self.capacity) - self.charge)*100/self.eff_in
        if to_fill > self.max_c:
            largest_in = copy.deepcopy(self.max_c)
        else:
            largest_in = copy.deepcopy(to_fill)
                
        if surplus*self.t_res > largest_in:
            # not all surplus can be stored
            self.charge += largest_in*self.eff_in/100
            self.en_in += largest_in
            self.curt += surplus*self.t_res - largest_in
            self.output[t] = surplus - largest_in/self.t_res
                
        else:
            # all of surplus transfterred to storage
            self.charge += surplus*self.t_res*self.eff_in/100
            self.en_in += surplus*self.t_res
            self.output[t] = 0.0

    def discharge_timestep(self, t, surplus):
        '''
        == description ==
        Charges the asset for one timestep - either until all the surplus is
        used, the asset is full, or the charging rate limit is reached (which
        ever comes first)

        == parameters ==
        t: (int) the current timestep - so that the output vector can be updated
        suplus: (float) the excess available energy in MWh for that timestep

        == returns ==
        None
        '''
        # amount that can be extracted from storage
        to_empty = self.charge*self.eff_out/100
        if to_empty > self.max_d:
            largest_out = copy.deepcopy(self.max_d)
        else:
            largest_out = copy.deepcopy(to_empty)
                
        if surplus*self.t_res*(-1) < largest_out:
            # sufficent storage can be discharged to meet shortfall
            self.charge += surplus*self.t_res*100/self.eff_out
            self.en_out -= surplus*self.t_res
            self.output[t] = 0.0

        else:
            # there is insufficient storage to meet shortfall
            self.en_out += largest_out
            self.output[t] = (surplus +
                              largest_out*self.eff_out/(100*self.t_res))
            if t >= self.start_up_time:
                shortfall = True
                self.charge -= largest_out*100/self.eff_out

    def time_step(self, t, surplus):
        '''
        == description ==
        This executes a timestep of the charge simulation. If the surplus is
        positive it charges storage and if it is negative it discharges.

        == parameters ==
        t: (int) the current timestep - so that the output vector can be updated
        suplus: (float) the excess available energy in MWh for that timestep

        == returns ==
        None
        '''
        self.self_discharge_timestep()
        
        if surplus > 0:
            self.charge_timestep(t, surplus)
        elif surplus < 0:
            self.discharge_timestep(t, surplus)

    def charge_sim(self, surplus, t_res=1, return_output=False, return_soc=False,
                   start_up_time=0):
        '''
        == description ==
        Runs a simulation using opportunistic charging the storage asset.

        == parameters ==
        surplus: (Array<float>) the surplus generation to be smoothed in MW
        t_res: (float) the size of time intervals in hours
        return_output: (boo) whether the smoothed profile should be returned
        start_up_time: (int) number of first time intervals to be ignored when
            calculating the % of met demand (to allow for start up effects).

        == returns ==
        reliability: (float) the percentage of time without shortfalls (0-100)
        output: (Array<float>) the stabilised output profile in MW
        '''
        self.reset()
        self.t_res = t_res
        self.start_up_time = start_up_time
        self.charge = 0.0 # intialise stosrage as empty
        self.output = [0]*len(surplus)
        self.n_years = len(surplus)/(365.25*24/t_res)

        shortfalls = 0 # timesteps where demand could not be met
        soc = []
        # for convenience, these are the ramp rates in MWh 
        self.max_c = self.capacity*self.max_c_rate*t_res/100
        self.max_d = self.capacity*self.max_d_rate*t_res/100
        
        for t in range(len(surplus)):
            self.time_step(t, surplus[t])
            soc.append(self.charge/self.capacity)
            if self.output[t] < 0:
                if t > start_up_time:                
                    shortfalls += 1
            
        reliability = 100 - ((shortfalls*100)/(len(surplus)
                                               -self.start_up_time))
        
        if return_output is False and return_soc is False:
            return reliability
        elif return_soc is True:
            return [reliability, soc]
        else:
            return [reliability, output]

    def analyse_usage(self):
        '''
        == description ==
        .

        == parameters ==
        None

        == returns ==
        en_in (float): the energy put into storage during the simulation (MWh)
        en_out (float): energy recovered from storage during simulation (MWh)
        curt (float): the energy curtailed during the simulation (MWh)
        '''

        return [self.en_in, self.en_out, self.curt]

    def size_storage(self, surplus, reliability, initial_capacity=0,
                     req_res=1e3,t_res=1, max_capacity=1e8,
                     start_up_time=0):
        '''
        == description ==
        Sizes storage or a required system reliability using bisection. Returns
        np.inf if the amount required is above max_storage.

        == parameters ==
        surplus: (Array<float>) the surplus generation to be smoothed in MW
        reliability: (float) required reliability in % (0-100)
        initial_capacity: (float) intital capacity to try in MWh
        req_res: (float) the required capacity resolution in MWh
        t_res: (float) the size of time intervals in hours
        max_storage: (float) the maximum size of storage in MWh to consider
        start_up_time: (int) number of first time intervals to be ignored when
            calculating the % of met demand (to allow for start up effects).

        == returns ==
        (float) required storage capacity (MWh)
        '''
        lower = initial_capacity
        upper = max_capacity
        

        self.set_capacity(upper)
        rel3 = self.charge_sim(surplus,t_res=t_res,start_up_time=start_up_time)
        if rel3 < reliability:
            self.capacity = np.inf
            return np.inf

        self.set_capacity(lower)
        rel1 = self.charge_sim(surplus,t_res=t_res,start_up_time=start_up_time)
        if rel1 > reliability:
            print('Initial capacity too high')
            if initial_capacity == 0:
                return 0.0
            else:
                self.size_storage(surplus, reliability, initial_capacity=0,
                                  req_res=req_res,t_res=t_res,
                                  max_capacity=max_capacity,
                                  start_up_time=start_up_time,
                                  strategy=strategy)

        while upper-lower > req_res:
            mid = (lower+upper)/2
            self.set_capacity(mid)
            rel2 = self.charge_sim(surplus,t_res=t_res,
                                   start_up_time=start_up_time)

            if rel2 < reliability:
                lower = mid
                rel1 = rel2
            else:
                upper = mid
                rel3 = rel2
                
        return (upper+lower)/2
        
class BatteryStorageModel(StorageModel):

    def __init__(self, eff_in=95, eff_out=95, self_dis=2,
                 variable_cost=0,
                 fixed_cost=16000, max_c_rate=100, max_d_rate=100,
                 capacity=1):
        
        super().__init__(eff_in, eff_out, self_dis, variable_cost, fixed_cost,
                         max_c_rate, max_d_rate, 'Li-Ion Battery',
                         capacity=capacity)

class HydrogenStorageModel(StorageModel):

    def __init__(self, eff_in=67, eff_out=56, self_dis=0, variable_cost=6.5,
                 fixed_cost=120, max_c_rate=0.032, max_d_rate=0.15,
                 capacity=1):
        
        super().__init__(eff_in, eff_out, self_dis, variable_cost, fixed_cost,
                         max_c_rate, max_d_rate, 'Hydrogen', capacity=capacity)


class ThermalStorageModel(StorageModel):

    def __init__(self, eff_in=80, eff_out=47, self_dis=9.66,variable_cost=331.6,
                 fixed_cost=773.5, max_c_rate=8.56, max_d_rate=6.82,
                 capacity=1):
        
        super().__init__(eff_in, eff_out, self_dis, variable_cost, fixed_cost,
                         max_c_rate, max_d_rate, 'Thermal', capacity=capacity)
        

class MultipleStorageAssets:

    def __init__(self, assets, c_order=None, d_order=None):
        '''
        == description ==
        Initialisation of a multiple storage object. Note that if charging or
        discharging orders are not specified the model defaults to discharge in
        the order of the asset list, and charge in the reverse.

        == parameters ==
        assets: (Array<StorageModel>) a list of storage model objects
        c_order: (Array<int>) a list of the order which assets should be
            prioritised for charging under 'ordered' operation
        d_order: (Array<int>) a list of the order which assets should be
            prioritised for discharging under 'ordered' operation

        == returns ==
        None
        '''
        self.assets = assets
        self.n_assets = len(assets)
        self.rel_capacity = [0.0]*len(assets)
        self.units = {}
        
        #added by cormac for plotting timeseries from optimisation
        self.surplus = np.empty([]) #the last surplus used as input for optimise
        self.Pfos = np.empty([]) #the necessary fossil fuel generation timeseries from the last optimise run
        self.Shed = np.empty([]) #timeseries of surplus shedding

        if c_order is None:
            c_order = list(range(self.n_assets))
            
        if d_order is None:
            d_order = list(range(self.n_assets))

        self.c_order = c_order
        self.d_order = d_order
            
        for i in range(self.n_assets):
            self.units[i] = assets[i]
            self.rel_capacity[i] = copy.deepcopy(assets[i].capacity)
            
        total_capacity = sum(self.rel_capacity)
        self.capacity = total_capacity
        for i in range(self.n_assets):
            self.rel_capacity[i] = float(self.rel_capacity[i])/total_capacity

    def reset(self):
        '''
        == description ==
        Resets the measurement on all storage units.

        == parameters ==
        None

        == returns ==
        None
        '''
        for i in range(self.n_assets):
            self.units[i].reset()
        
        self.surplus = np.empty([]) 
        self.Pfos = np.empty([])
        self.Shed = np.empty([])

    def set_capacity(self, capacity):
        '''
        == description ==
        Scales the total installed capacity to the specified value, the
        relative capacity of the individual assets remains the same.

        == parameters ==
        capacity: (float) The total installed capacity in MWh

        == returns ==
        None
        '''
        for i in range(self.n_assets):
            self.units[i].set_capacity(capacity*self.rel_capacity[i])
        self.capacity = capacity

    def self_discharge_timestep(self):
        '''
        == description ==
        Self-discharge all assets for one timestep.

        == parameters ==
        None

        == returns ==
        None
        '''
        for i in range(self.n_assets):
            self.units[i].self_discharge_timestep()
            
    def is_MultipleStorageAssets(self):
        '''
        == description ==
        Returns True if it is a Multiple Storage Asset

        == parameters ==
        None

        == returns ==
        (float) True
        '''
        return True

    def get_cost(self):
        '''
        == description ==
        Gets the cumulative cost of all of the storage assets.

        == parameters ==
        None

        == returns ==
        (float) total cost of all storage units in GBP/yr
        '''
        if self.capacity == np.inf:
            return np.inf
        else:
            total = 0.0
            for i in range(self.n_assets):
                total += self.units[i].get_cost()
            return total

    def charge_emptiest(self, surplus, t_res=1, return_output=False,
                        start_up_time=0):
        '''
        == description ==
        .

        == parameters ==
        None

        == returns ==
        None
        '''
        return ''

    def charge_specfied_order(self, surplus, c_order, d_order, t_res=1,
                              return_output=False,start_up_time=0,
                              return_di_av=False):
        '''
        == description ==
        .

        == parameters ==
        None

        == returns ==
        None
        '''
        if len(c_order) != self.n_assets:
            raise Exception('c_order wrong length')
        if len(d_order) != self.n_assets:
            raise Exception('d_order wrong length')
        
        shortfalls = 0
        output = [0]*len(surplus)
        soc = []
        for i in range(self.n_assets):
            soc.append([i])
        self.curt = 0.0
        di_profiles = {}
        T = int(24/t_res)
        for i in range(len(c_order)):
            di_profiles[i] = {'c':[0.0]*T,'d':[0.0]*T}

        # initialise all storage units
        for i in range(self.n_assets):
            self.units[i].max_c = (self.units[i].capacity
                                   *self.units[i].max_c_rate*t_res/100)
            self.units[i].max_d = (self.units[i].capacity
                                   *self.units[i].max_d_rate*t_res/100)
            self.units[i].t_res = t_res
            self.units[i].start_up_time = start_up_time
            self.units[i].charge = 0.0
            self.units[i].n_years = len(surplus)/(365.25*24/t_res)
            self.units[i].output = [0]*len(surplus)
        
        for t in range(len(surplus)):
            # self discharge all assets
            self.self_discharge_timestep()
            

            t_surplus = copy.deepcopy(surplus[t])
    
            if t_surplus > 0:
                for i in range(self.n_assets):
                    if t_surplus > 0:
                        self.units[c_order[i]].charge_timestep(t, t_surplus)
                        output[t] = self.units[c_order[i]].output[t]
                        if t > start_up_time:
                            di_profiles[i]['c'][t%T] += output[t]-t_surplus
                        t_surplus = self.units[c_order[i]].output[t]
                self.curt += output[t]
                
            elif t_surplus < 0:
                for i in range(self.n_assets):
                    if t_surplus < 0:
                        self.units[d_order[i]].discharge_timestep(t,t_surplus)
                        output[t] = self.units[d_order[i]].output[t]
                        if t > start_up_time:
                            di_profiles[i]['d'][t%T] += output[t]-t_surplus
                        t_surplus = self.units[d_order[i]].output[t]
                if output[t] < 0:
                    if t > start_up_time:                    
                        shortfalls += 1
            #soc[i].append(self.charge)
            
        reliability = 100 - ((shortfalls*100)/(len(surplus)-start_up_time))

        if return_output is False and return_di_av is False:
            return reliability

        ret = [reliability]
        
        if return_output is True:
            ret += [output]

        if return_di_av is True:
            sf = (len(surplus)-start_up_time)/T
            for i in di_profiles:
                for t in range(T):
                    di_profiles[i]['c'][t] = float(di_profiles[i]['c'][t])/sf
                    di_profiles[i]['d'][t] = float(di_profiles[i]['d'][t])/sf
            ret += [di_profiles]

        return ret
    
    def charge_sim(self, surplus, t_res=1, return_output=False,
                   start_up_time=0,strategy='ordered',return_di_av=False):
        '''
        == description ==
        This will 

        == parameters ==
        surplus: (Array<float>) the surplus generation to be smoothed in MW
        t_res: (float) the size of time intervals in hours
        return_output: (boo) whether the smoothed profile should be returned
        start_up_time: (int) number of first time intervals to be ignored when
            calculating the % of met demand (to allow for start up effects).
        strategy: (str) the strategy for operating the assets. Options:
                'ordered' - charges/discharges according to self.c_order/d_order
                'balanced' - ?

        == returns ==
        reliability: (float) the percentage of time without shortfalls (0-100)
        output: (Array<float>) the stabilised output profile in MW
        '''
        
        if strategy == 'ordered':
            res = self.charge_specfied_order(surplus, self.c_order,
                                             self.d_order,t_res=t_res,
                                             return_output=return_output,
                                             start_up_time=start_up_time,
                                             return_di_av=return_di_av)
        return res

    def analyse_usage(self):
        '''
        == description ==
        Get the usage of each storage asset following a simulation.
        
        == parameters ==
        None

        == returns ==
        en_in (Array<float>): the energy put into each storage asset during the
            simulation (MWh)
        en_out (Array<float>): energy recovered from each storage asset during
            the simulation (MWh)
        curt (float): the energy curtailed during the simulation (MWh)
        '''
        stored = []
        recovered = []
        for i in range(self.n_assets):
            stored.append(self.units[i].en_in/self.units[i].n_years)
            recovered.append(self.units[i].en_out/self.units[i].n_years)
        curtailed = self.curt/self.units[i].n_years

        return [stored, recovered, curtailed]

    def causal_system_operation(self, surplus, c_order, d_order, Mult_aggEV, t_res=1,
                              start_up_time=0, plot_timeseries = False,V2G_discharge_threshold = 0.0):
        '''
        == description ==
        This function is similiar to charge specified order but with two key differences:
            1) It allows aggregated EVs to be operated also (the order of their discharge specified). To do this it
                it splits each EV fleet into two batteries, one for V2G and one for Smart, these are then operated seperately.
            2) It outputs two new outputs: system reliability based on amount of demand served by renewables
               rather than the old reliability metric based on time where renewables don't cover everything; and 
               EV reliability which gives the % under delivery of power to the EVs. 

        == parameters ==
        Mult_aggEV: (MultipleAggregatedEVs) different fleets of EVs with defined chargertype ratios!
        c_order: list [int] of order of the discharge, where the first entry is for the storage units, later half is for the agg EV fleets.+
        plot_timeseries: (bool), if true will plot the storage SOCs and charge/discharge, as well as the surplus before and after adjustement
        V2G_discharge_threshold: (float), The kWh limit for the EV batteries, below whcih V2G will not discharge. The state of charge can still drop below this due to driving energy, but V2G will not discharge when the SOC is less than this value.
        
        == returns ==
        [power_deficit,EV_Reliability]: Power deficit is the amount of energy in MWh that needs to be supplied by fossil fuels (including the energy needed for topping up the EVs if they are set to unplug below their plugout SOC)
                                        EV_Reliability: is the % of driving energy met by renewable energy. Given as [Fleet1 V2g, Fleet1 Unidirectional, Fleet2 V2G, ...] 
                                        For V2G this can be -ve, as when the EVs are plugged back in they can be discharged to zero again, thus they will need to be charged to 90% from zero rather than from about 30% as for the Unidirectional. 
                                        Thus the energy needed from fossil fuels is larger that the driving energy.
        '''
        units={}
        counter = 0
        for i in range(self.n_assets):
            units[i] = self.assets[i]
            counter = counter+1
        
        #split the EV fleets into a battery for Smart and a Battery for non smart
        for k in range(Mult_aggEV.n_assets):
            for b in range(2):
                units[counter] = BatteryStorageModel()
                counter = counter+1
        
        Num_units = len(units)
        
        if len(c_order) != Num_units:
            raise Exception('c_order wrong length, need two entries for every agg fleet object')
        if len(d_order) != Num_units:
            raise Exception('d_order wrong length, need two entries for every agg fleet object')
        
        power_deficit = 0.0 #this is the energy in MWh met by fossil fuels, including for EV driving demand!
        output = [0]*len(surplus)  #this is the surplus after charging!
        self.curt = 0.0
        di_profiles = {}
        T = int(24/t_res)
        for i in range(len(c_order)):
            di_profiles[i] = {'c':[0.0]*len(surplus),'d':[0.0]*len(surplus)}

        # initialise all storage units (EVs updated at each timestep)
        counter = 0
        for i in range(self.n_assets):
            units[i].max_c = (units[i].capacity
                                    *units[i].max_c_rate*t_res/100)
            units[i].max_d = (units[i].capacity
                                    *units[i].max_d_rate*t_res/100)
            units[i].t_res = t_res
            
            #assumed all units start at 50% charge
            units[i].charge = 0.5*units[i].capacity 
            counter = counter+1
        
        for i in range(Num_units):
            units[i].start_up_time = 0
            units[i].n_years = len(surplus)/(365.25*24/t_res)
            units[i].output = [0]*len(surplus) #this is the left over defecit after the charge action on asset i
            units[i].t_res = t_res
            
    # Begin simulating system #
        EV_Energy_Underserve = np.zeros([Mult_aggEV.n_assets*2]) # this is the total energy for the EVs that needs to be supplied by fossil fuels
        Total_Driving_Energy = np.zeros([Mult_aggEV.n_assets*2]) #this is the total desired plugout energy of the EVs
        charge_hist = np.zeros([Num_units,len(surplus)])
        V2G = True
        for t in range(len(surplus)):
            # self discharge all assets
            for i in range(Num_units):
                units[i].self_discharge_timestep()
        # Update State of the Aggregated EV batteries #
                if(i >= self.n_assets):
                    
                    if(V2G):
                        k = int((i+1 - self.n_assets)/2)
                        b=0
                    else:
                        b=1
                                            
                    #work out the energy remaining after the EVs unplug
                    if(t==0):
                        if Mult_aggEV.assets[k].Eout != Mult_aggEV.assets[k].max_SOC:
                            raise Exception('The max SOC does not equal the plugout SOC. This leads to errors in the causal system operation. Make these the same or improve code.')
                        N = Mult_aggEV.assets[k].N[t]
                        units[i].charge = 0.5 * N * Mult_aggEV.assets[k].chargertype[b] * Mult_aggEV.assets[k].number * Mult_aggEV.assets[k].max_SOC/1000
                    
                    Energy_Remaining = units[i].charge - Mult_aggEV.assets[k].Nout[t]* Mult_aggEV.assets[k].chargertype[b] * Mult_aggEV.assets[k].number * Mult_aggEV.assets[k].Eout/1000 #work out the energy remaining after the EVs have plugged out
                    Total_Driving_Energy[k+b] += Mult_aggEV.assets[k].Nout[t]* Mult_aggEV.assets[k].chargertype[b] * Mult_aggEV.assets[k].number * Mult_aggEV.assets[k].Eout/1000 - Mult_aggEV.assets[k].Nin[t]* Mult_aggEV.assets[k].chargertype[b] * Mult_aggEV.assets[k].number * Mult_aggEV.assets[k].Ein/1000
                    
                    #if there is sufficient for the driving, update the SOC and continue
                    if Energy_Remaining > 0:
                        units[i].charge = Energy_Remaining + Mult_aggEV.assets[k].Nin[t]* Mult_aggEV.assets[k].chargertype[b] * Mult_aggEV.assets[k].number * Mult_aggEV.assets[k].Ein/1000
                    #if there is not, set the charge to 0 and record the underserve
                    else:
                        units[i].charge = Mult_aggEV.assets[k].Nin[t]* Mult_aggEV.assets[k].chargertype[b] * Mult_aggEV.assets[k].number * Mult_aggEV.assets[k].Ein/1000
                        EV_Energy_Underserve[k+b] += -Energy_Remaining 
                        power_deficit += -Energy_Remaining
    
                    #update the max charge limit
                    if(V2G):
                        N = N + Mult_aggEV.assets[k].Nin[t] - Mult_aggEV.assets[k].Nout[t]
                        V2G = False
                        discharge_threshold = N * Mult_aggEV.assets[k].chargertype[b] * Mult_aggEV.assets[k].number * V2G_discharge_threshold/1000
                        
                        if(units[i].charge <= discharge_threshold):
                            units[i].max_d = 0.0
                        else:
                            units[i].max_d = min(N * Mult_aggEV.assets[k].chargertype[b] * Mult_aggEV.assets[k].number * Mult_aggEV.assets[k].max_d_rate/1000*t_res,units[i].charge - discharge_threshold)
                    else:
                        V2G = True
                        units[i].max_d = 0.0

                    units[i].max_c = N * Mult_aggEV.assets[k].chargertype[b] * Mult_aggEV.assets[k].number * Mult_aggEV.assets[k].max_c_rate*t_res/1000
                    units[i].capacity = N * Mult_aggEV.assets[k].chargertype[b] * Mult_aggEV.assets[k].number * Mult_aggEV.assets[k].max_SOC/1000

            t_surplus = copy.deepcopy(surplus[t])
    
            if t_surplus >= 0:
                for i in range(Num_units):                  
                    units[c_order[i]].charge_timestep(t, t_surplus)
                    output[t] = units[c_order[i]].output[t]
                    di_profiles[c_order[i]]['c'][t] = output[t]-t_surplus
                    t_surplus = units[c_order[i]].output[t]
                    charge_hist[c_order[i],t] = units[c_order[i]].charge
                self.curt += output[t]
 
            elif t_surplus < 0:
                for i in range(Num_units):
                    units[d_order[i]].discharge_timestep(t,t_surplus)
                    output[t] = units[d_order[i]].output[t]
                    di_profiles[d_order[i]]['d'][t] = output[t]-t_surplus
                    t_surplus = units[d_order[i]].output[t]
                    
                    charge_hist[d_order[i],t] = units[d_order[i]].charge
                power_deficit += output[t] #this is the power that needs to be supplied by fossil fuels
        
        if(plot_timeseries):
            timehorizon = len(surplus)
            plt.rc('font', size=12)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(timehorizon), surplus, color='k', label='Surplus')
            ax.plot(range(timehorizon), output, color='b', label='Surplus post Charging')
            ax.set_title('Surplus Timeseries')
            ax.legend(loc='upper left')

            for i in range(Num_units):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(range(timehorizon), charge_hist[i,:], color='k', label='SOC')
                ax.plot(range(timehorizon), di_profiles[i]['c'][:], color='r', label='Charge')
                ax.plot(range(timehorizon), di_profiles[i]['d'][:], color='b', label='Discharge')
                ax.set_title(str(i))
                ax.legend(loc='upper left')
                    
        EV_Reliability = np.ones([Mult_aggEV.n_assets*2])*100
        for i in range(Mult_aggEV.n_assets*2): 
            if Total_Driving_Energy[i] > 0:
                EV_Reliability[i] = ((Total_Driving_Energy[i]-EV_Energy_Underserve[i])/Total_Driving_Energy[i])*100 #the % of driving energy met with renewables
        ret = [int(power_deficit),EV_Reliability]
        
        return ret
    
    def optimise_storage(self,surplus,fossilLimit):
        
        '''
        == description ==
        For a given surplus, returns the cost optimal storage mix to meet the specified reliability. Charge order not relevant here.

        == parameters ==
        surplus: (Array<float>) the surplus generation to be smoothed in MW
        fossilLimit: (float) max acceptable amount of fossil fuel generated energy (MWh)

        == returns ==
        '''
        
        opt_con.optimise_configuration(surplus,fossilLimit,self,aggEV.MultipleAggregatedEVs([]))
        
        
       

    def plot_timeseries(self,start=0,end=-1):
        
        '''   
        == parameters ==
        start: (int) start time of plot
        end: (int) end time of plot
        
        '''

        if (self.Pfos.shape == ()):
                print('Charging timeseries not avaialable, try running MultipleStorageAssets.optimise_storage().')
        else:
            if(end<=0):
                timehorizon = self.Pfos.size
            else:
                timehorizon = end
            plt.rc('font', size=12)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(start,timehorizon), self.Pfos[start:timehorizon], color='tab:red', label='FF Power')
            ax.plot(range(start,timehorizon), self.Shed[start:timehorizon], color='tab:blue', label='Renewable Shed')
            ax.plot(range(start,timehorizon), self.surplus[start:timehorizon], color='tab:orange', label='Surplus')
    
            # Same as above
            ax.set_xlabel('Time (h)')
            ax.set_ylabel('Power (MW), Energy (MWh)')
            ax.set_title(' Power Timeseries')
            ax.grid(True)
            ax.legend(loc='upper left');   

    def size_storage(self, surplus, reliability, initial_capacity=None,
                     req_res=1e5,t_res=1, max_capacity=1e9,
                     start_up_time=0,strategy='ordered'):
        '''
        == description ==
        For a fixed relative size of storage assets, this funciton finds the
        total storage required to meet a certain level of reliability.

        == parameters ==
        surplus: (Array<float>) the surplus generation to be smoothed in MW
        reliability: (float) required reliability in % (0-100)
        initial_capacity: (float) intital capacity to try in MWh
        req_res: (float) the required capacity resolution in MWh
        t_res: (float) the size of time intervals in hours
        max_storage: (float) the maximum size of storage in MWh to consider
        start_up_time: (int) number of first time intervals to be ignored when
            calculating the % of met demand (to allow for start up effects).

        == returns ==
        capacity: the required total storage capacity in MWh
        '''

        if initial_capacity is None:
            initial_capacity = min(surplus)*-1
            
        lower = initial_capacity
        upper = max_capacity

        self.set_capacity(upper)
        rel3 = self.charge_sim(surplus,t_res=t_res,start_up_time=start_up_time,
                               strategy=strategy)
        if rel3 < reliability:
            self.capacity = np.inf
            return np.inf

        self.set_capacity(lower)
        rel1 = self.charge_sim(surplus,t_res=t_res,
                                 start_up_time=start_up_time,
                                 strategy=strategy)
        if rel1 > reliability:
            print('Initial capacity too high')
            if initial_capacity == 0:
                return 0.0
            else:
                self.size_storage(surplus, reliability, initial_capacity=0,
                                  req_res=req_res,t_res=t_res,
                                  max_capacity=max_capacity,
                                  start_up_time=start_up_time,
                                  strategy=strategy)

        while upper-lower > req_res:
            mid = (lower+upper)/2
            self.set_capacity(mid)
            rel2 = self.charge_sim(surplus,t_res=t_res,
                                     start_up_time=start_up_time,
                                     strategy=strategy)
            if rel2 < reliability:
                lower = mid
                rel1 = rel2
            else:
                upper = mid
                rel3 = rel2
                
        return (upper+lower)/2

        
