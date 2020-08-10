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

class StorageModel:
    def __init__(self, eff_in, eff_out, self_dis, variable_cost, fixed_cost,
                 max_c_rate, max_d_rate, name, capacity=1):
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

        # These will be used to monitor storage usage
        self.en_in = 0 # total energy into storage (grid side)
        self.en_out = 0 # total energy out of storage (grid side)
        self.curt = 0 # total supply that could not be stored

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

    def charge_sim(self, surplus, t_res=1, return_output=False,start_up_time=0):
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
        
        # for convenience, these are the ramp rates in MWh 
        self.max_c = self.capacity*self.max_c_rate*t_res/100
        self.max_d = self.capacity*self.max_d_rate*t_res/100
        
        for t in range(len(surplus)):
            self.time_step(t, surplus[t])
            if self.output[t] < 0:
                shortfalls += 1
            
        reliability = 100 - ((shortfalls*100)/(len(surplus)
                                               -self.start_up_time))
        
        if return_output is False:
            return reliability
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

    def size_storage(self, surplus, reliability, initial_capacity=None,
                     req_res=1e3,t_res=1, max_storage=1e8,
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

        if initial_capacity is None:
            initial_capacity = max(surplus)

        rel = 0

        lower = initial_capacity
        upper = max_storage

        while upper-lower > req_res:
            cap = np.linspace(lower,upper,num=10)
            i = -1
            while rel < reliability and i < len(cap):
                self.capacity = cap[i]
                rel = self.charge_sim(surplus,t_res=t_res,
                                        start_up_time=start_up_time)
                i += 1
            if i == len(cap) and rel < reliability:
                return np.inf
            
            lower = cap[i-2]
            upper = cap[i-1]

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

    def __init__(self, eff_in=67, eff_out=56, self_dis=0, variable_cost=42.5,
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

        self.n_assets = len(assets)
        self.rel_capacity = [0.0]*len(assets)
        self.units = {}

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
                    shortfalls += 1
            
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
        rel3 = self.charge_sim(surplus,t_res=t_res,
                                 start_up_time=start_up_time,
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

        
