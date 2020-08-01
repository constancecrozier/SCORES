'''
Created: 28/07/2020 by C.CROZIER

File description: This file contains the classes for modelling and optimising
an energy system.

Notes: The following packages must be installed before use:
    - csv
    - datetime
    - copy
    - matplotlib
    - numpy
    - scipy
    - pyDOE
'''
import csv
import datetime
import matplotlib.pyplot as plt
import copy
import numpy as np

from scipy.optimize import Bounds, LinearConstraint, minimize
from pyDOE import lhs

from storage import MultipleStorageAssets
from fns import _subplot, result_as_txt, get_demand, offset
        
class ElectricitySystem:

    def __init__(self, gen_list, stor_list, demand, t_res=1, reliability=99,
                 start_up_time=0,strategy='ordered'):
        '''
        == description ==
        Initiates base class
        
        == parameters ==
        gen_list: (Array<GenerationModel>) a list containing generator objects
        stor_list: (Array<StorageModel>)
        demand: (Array<float)>) the demand in MW
        t_res: (float) the length of one demand time step in hours
        reliability: (float) The percentage of demand that will be met
        start_up_time: (int) number of first time intervals to be ignored when
            calculating the % of met demand (to allow for start up effects).
        strategy: (str) the strategy for operating the assets. Options:
                'ordered' - charges/discharges according to self.c_order/d_order
                'balanced' - ?
        
        == returns ==
        None
        
        '''
        
        for gen in gen_list:
            if len(gen.power_out) != len(demand):
                raise Exception('supply and demand have different lengths')

        self.gen_list = gen_list
        self.demand = demand
        self.t_res = t_res
        self.len = len(demand)
        self.total_installed_generation = 0
        self.n_storage = len(stor_list)
        self.reliability = reliability
        self.start_up_time = start_up_time
        self.strategy = strategy

        # initialise bounds that are used for optimisation
        self.min_gen_cap = [0.0]*len(gen_list)
        self.max_gen_cap = [np.inf]*len(gen_list)
        self.storage = MultipleStorageAssets(stor_list)

        for gen in self.gen_list:
            if max(gen.power_out_scaled) == 0:
                gen.scale_output(1)
            self.total_installed_generation += gen.scaled_installed_capacity
        
    def scale_generation(self, gen_cap):
        '''
        == description ==
        Scales the outputs of the generation objects to the specified installed
        capacities (in order that they are specified)
        
        == parameters ==
        gen_cap: (Array<float)>) installed capacity of each object in GW
        
        == returns ==
        (float) cost in GBP/year of newly scaled generation
        
        '''
        total = 0.0
        self.total_installed_generation = 0
        for i in range(len(self.gen_list)):
            self.gen_list[i].scale_output(gen_cap[i]*1000)
            self.total_installed_generation += gen_cap[i]*1000
            total += self.gen_list[i].get_cost()
        return total

    def scale_generation_tic(self,tic):
        '''
        == description ==
        Keeps the relative size of the existing generation objects, but scales
        them proportionately to the specified total installed capacity (tic)
        
        == parameters ==
        tic: (float) total installed capacity in GW
        
        == returns ==
        (float) cost in GBP/year of newly scaled generation
        
        '''
        gen_cap = []
        for i in range(len(self.gen_list)):
            gen_cap.append(self.gen_list[i].scaled_installed_capacity*tic/
                           self.total_installed_generation)
        total = self.scale_generation(gen_cap)
        self.total_installed_generation = tic*1000
        return total

    def update_surplus(self):
        '''
        == description ==
        Updates the surplus vector following a change in generation
        
        == parameters ==
        None
        
        == returns ==
        None
        
        '''
        self.surplus = [0]*self.len
        for t in range(self.len):
            for gen in self.gen_list:
                self.surplus[t] += gen.power_out_scaled[t]
            self.surplus[t] -= self.demand[t]
        
    def get_reliability(self, start_up_time=0, return_output=False,
                        strategy='ordered'):
        '''
        == description ==
        Calculates the percentage of demand that can be met with current system

        == parameters ==
        None

        == returns ==
        (float) percentage of demand met (0-100)

        if return_output is True:
        (Array<float>) the smoothed supply profile
        '''
        if self.n_storage == 1:
            rel = self.storage.charge_sim(self.surplus, t_res=self.t_res,
                                          return_output=return_output,
                                          start_up_time=start_up_time)
        else:
            rel = self.storage.charge_sim(self.surplus,t_res=self.t_res,
                                          start_up_time=start_up_time,
                                          return_output=return_output,
                                          strategy=strategy)
        return rel

    def cost(self, x): 
        '''
        == description ==
        Calculates the total system cost for a given set of generation and
        relative storage sizes.

        == parameters ==
        x: (Array<float>) the first n_gen elements contain the installed
            capacity in GW of each generation unit (in the order they are
            specified in gen_list). The other elements are the proportion of the
            total storage capacity that each unit comprises. Note that there
            will be one fewer values than number of storage assets (as the
            remaining proportion of 1.0 is allocated to the last one.

        == returns ==
        (float) total system cost £bn /year
        '''                 
        gen_cap = x[:len(self.gen_list)]
        stor_cap = list(x[len(self.gen_list):])
        stor_cap.append(1-sum(stor_cap))

        total =  self.scale_generation(gen_cap)
        self.storage.rel_capacity = stor_cap
        
        self.update_surplus()
        sc = self.storage.size_storage(self.surplus, self.reliability,
                                       start_up_time=self.start_up_time,
                                       strategy=self.strategy)
        
        if sc == np.inf:
            return np.inf
        else:
            total += self.storage.get_cost()
            return total*1e-9

    def analyse(self,x,filename='log/system_analysis.txt'):
        c = self.cost(x)
        f = open(filename,'w')
        f.write('System cost: £'+str(c)+' bn/yr\n\n')
        f.write('---------------------\n')
        f.write('INSTALLED GENERATION\n')
        f.write('---------------------\n\n')
        for i in range(len(self.gen_list)):
            f.write(self.gen_list[i].name+': '+str(x[i])+' GW\n')
        f.write('\n>>TOTAL: '+str(sum(x[:len(self.gen_list)]))+' GW\n\n')
        f.write('------------------\n')
        f.write('INSTALLED STORAGE\n')
        f.write('------------------\n\n')
        for i in range(len(self.storage.units)):
            f.write(self.storage.units[i].name+': '+
                    str(self.storage.units[i].capacity*1e-6)+' TWh\n')
            
        f.write('\n>>TOTAL: '+str(self.storage.capacity*1e-6)+' TWh\n')

        f.write('\n--------------------\n')
        f.write('STORAGE UTILISATION\n')
        f.write('--------------------\n\n')
        use = self.storage.analyse_usage()
        
        n_years = self.storage.units[0].n_years
        curt = use[2]/n_years
        for i in range(self.n_storage):
            f.write('>> '+self.storage.units[i].name+' <<\n\n')
            f.write(str(use[0][i]/n_years*1e-6)+ ' TWh/yr in (grid side)\n')
            f.write(str(use[1][i]/self.storage.units[i].n_years*1e-6)
                    +' TWh/yr out (grid side)\n')
            cycles = (use[1][i]*100/(self.storage.units[i].eff_out*n_years
                                     *self.storage.units[i].capacity))
            f.write(str(cycles)+' cycles per year\n\n')

        f.write('-------------------\n')
        f.write('ENERGY UTILISATION\n')
        f.write('-------------------\n\n')
        f.write('Total Demand: '
                + str((sum(self.demand)*1e-6)/(self.t_res*n_years))
                + ' TWh/yr\n')
        f.write('Total Supply: '
                + str(((sum(self.surplus)+sum(self.demand))*1e-6)
                      /(self.t_res*n_years))
                + ' TWh/yr\n') 
        f.write('Curtailment: '+str(curt*1e-6)+' TWh/yr\n\n')

        f.write('---------------\n')
        f.write('COST BREAKDOWN\n')
        f.write('---------------\n\n')
        for i in range(len(self.gen_list)):
            f.write(self.gen_list[i].name+': £'
                    +str(1e-9*self.gen_list[i].get_cost())+' bn/yr\n')
        for i in range(self.n_storage):
            f.write(self.storage.units[i].name+': £'
                    +str(1e-9*self.storage.units[i].get_cost())+' bn/yr\n')
        f.close()
        
    def cost_fixed_gen_ratio(self, x): 
        '''
        == description ==
        Calculates the total system cost for a given total installed generation
        capacity and a set of relative storage sizes - but with the ratio
        between generation units fixed.

        == parameters ==
        x: (Array<float>) the first element contains a scaling factor for the
        previous total installed generation capacity (tic0) and the remaining
        contain the proportion of the first n-1 storage capacities (the final
        storage asset is allocated the remaining share).

        == returns ==
        (float) total system cost £bn /year
        '''
        tic = x[0]*self.tic0
        total = self.scale_generation_tic(tic)
        
        stor_cap = list(x[1:])
        stor_cap.append(1-sum(stor_cap))
        self.storage.rel_capacity = stor_cap
        
        self.update_surplus()
        sc = self.storage.size_storage(self.surplus, self.reliability,
                                       start_up_time=self.start_up_time,
                                       strategy=self.strategy)
        if sc == np.inf:
            return np.inf
        else:
            total += self.storage.get_cost()
            return total*1e-9

    def search_gen_scale_factor(self):
        '''
        == description ==
        Manually searches through the best generation scale factor, for use
        when there is only one storage.

        == parameters ==
        None

        == returns ==
        (Array<float>) best generation scale factor
        (float) total system cost £bn /year
        '''

        lwst  = np.inf
        best = None
        for x in np.arange(0.7,1.3,0.01):
            f = self.cost_fixed_gen_ratio([x])
            if f < lwst:
                best = x
                lwst = f
            if f > lwst*1.1:
                break

        return [best], lwst

    def lhs_generation(self, tic, number_test_points=20, stor_cap=None): 
        '''
        == description ==
        This function performs a random search over different generation
        capacity ratios. Latin hypercube sampling is used to select dissimilar
        start points and then the generators are scaled such to the specified
        total installed capacity. Points violating the generation limits are
        ignored, otherwise the cost is calculated.

        == parameters ==
        tic: (float) The total installed capacity in GW of all generation units
        number_test_points: (int) The number of test points
        stor_cap: (Array<float>) The ratio of storage capacities to use

        == returns ==
        (float) lowest found total system cost £bn /year
        (Array<float>) the corresponding installed generation capacities in GW
        '''
        # if storage ratio not specified, assume a uniform distribution 
        if stor_cap is None:
            if self.n_storage == 1:
                stor_cap = []
            else:
                stor_cap = [1.0/self.n_storage]*(self.n_storage-1)
                
        x = lhs(len(self.gen_list),samples=number_test_points)

        best = None
        lwst = np.inf
        for i in range(number_test_points):
            gen_cap = []
            violation = False
            # first re-scale to be within the limits
            for j in range(len(self.min_gen_cap)):
                gen_cap.append((x[i][j]*(self.max_gen_cap[j] -
                                         self.min_gen_cap[j])
                                + self.min_gen_cap[j]))
            # then re-scale to correct total installed capacity
            sf = tic/copy.deepcopy(sum(gen_cap))
            for j in range(len(gen_cap)):
                gen_cap[j] = gen_cap[j]*sf

                # chec that specified bounds have not been violated
                if (gen_cap[j] < self.min_gen_cap[j] or
                    gen_cap[j] > self.max_gen_cap[j]):
                    violation = True

            if violation is True:
                continue
            
            f = self.cost(gen_cap+stor_cap)
            if f < lwst:
                lwst = f
                best = gen_cap

        return best, lwst
    
    def optimise_fixed_gen_ratio(self, installed_gen, x0):
        '''
        == description ==
        This function optimises over the total installed capacity and the
        relative storage capacities, but with the relative generation capacities
        

        == parameters ==
        tic: (float) The total installed capacity in GW of all generation units
        number_test_points: (int) The number of test points
        stor_cap: (Array<float>) The ratio of storage capacities to use

        == returns ==
        (float) lowest found total system cost £bn /year
        (Array<float>) the corresponding installed generation capacities in GW
        '''
        self.scale_generation(installed_gen)
        self.tic0 = sum(installed_gen)
        
        bounds = Bounds([0.55]+[0.0]*(self.n_storage-1),
                        [1.45]+[1.0]*(self.n_storage-1))

        if self.n_storage > 2:
            # constraint to ensure that storage ratios add up to less than 1
            linear_constraint = LinearConstraint([0]+[1.0]*(self.n_storage-1),
                                                 [0], [1])
            constraints = [linear_constraint]
        else:
            constraints = None

        if self.n_storage == 1:
            x, cost = self.search_gen_scale_factor()
            stor_cap = []
            tic = x[0]*self.tic0
        else:
            res = minimize(self.cost_fixed_gen_ratio, [1.0]+x0,
                           constraints=constraints, bounds=bounds, tol=1e-3)
            stor_cap = list(res.x)[1:]
            tic = list(res.x)[0]*self.tic0
            cost = res.fun

        gen_cap = []
        for i in range(len(installed_gen)):
            gen_cap.append(installed_gen[i]*tic/sum(installed_gen))
        
        
        return gen_cap, stor_cap, cost

    def optimise(self, reliability=None, tic0=None, stor_cap=None, gen_cap=None,
                 min_gen_cap=None, max_gen_cap=None, analyse=True,
                 start_up_time=30*24*3, strategy=None):
        '''
        == description ==
        Searches for the lowest cost electricity system that meets the
        specified reliability requirement. If an initial gset of generation
        capacities are not specified a lhs search is performed to find a good
        starting point

        == parameters ==
        reliability: (float) The required system reliability (0-100)
        tic0: (float) The total installed generation capacity used for search
        stor_cap: (Array<float>) the ratio of storage asset's capacity
        gen_cap: (Array<float>) the installed generation capacities in GW
        min_gen_cap: (Array<float>) lower limits on the size of each generation
            unit in GW
        max_gen_cap: (Array<float>) upper limits on the size of each generation
            unit in GW
        analyse: (boo) Whether or not to store analysis of optimal system
        start_up_time: (int) number of first time intervals to be ignored when
            calculating the % of met demand (to allow for start up effects).
        strategy: (str) the strategy for operating the assets. Options:
                'ordered' - charges/discharges according to self.c_order/d_order
                'balanced' - ?

        == returns ==
        (Array<float>) the best system sizing as vector "x"
        (float) lowest found total system cost £bn /year
        '''
        if min_gen_cap is not None:
            self.min_gen_cap = min_gen_cap
        if max_gen_cap is not None:
            self.max_gen_cap = max_gen_cap

        if reliability is not None:
            self.reliability = reliability
            self.start_up_time = start_up_time
        if strategy is not None:
            self.strategy = strategy

        if stor_cap is None:
            print('Specifying an initial relative storange capacity will '+
                  'imporve speed and accuracy')
            stor_cap = [1.0/self.n_storage]*(self.n_storage-1)

        if gen_cap is None:
            # perform a lhs search over different generator ratios
            if tic0 is None:
               tic0 = 2.8e-3*max(self.demand) # arbitrary 
            gen_cap,cost = self.lhs_generation(tic0,number_test_points=20,
                                               stor_cap=stor_cap)
        else:
            cost = self.cost(gen_cap+stor_cap)
            
        gen_cap,stor_cap,cost = self.optimise_fixed_gen_ratio(gen_cap,stor_cap)

        if analyse is True:
            self.analyse(list(gen_cap)+list(stor_cap),
                         filename='log/opt_results.txt')

        return list(gen_cap)+list(stor_cap), cost

    def sensitivity_analysis(self,var,mult_facts,var_name,stor_cap=None):
        orig = copy.deepcopy(var)
        x = []
        y = []
        gen_cap=None
        for mf in mult_facts:
            var = orig*mf
            filename = 'log/sens_'+var_name+str(var)+'.txt'
            _x,cost = self.optimise1(stor_cap=stor_cap,gen_cap=gen_cap,
                                     filename=filename)

            # offset new start points to prevent getting stuck in local optima
            gen_cap = offset(_x[:len(self.min_gen_cap)])
            stor_cap = offset(_x[len(self.min_gen_cap):])
            x.append(var)
            y.append(cost)

        plt.figure()
        plt.plot(x,y)
        plt.xlabel(var_name)
        plt.ylabel('Minimum system cost')
        plt.tight_layout()
        plt.show()

        
class ElectricitySystemGB(ElectricitySystem):

    def __init__(self, gen_list, stor_list, year_min=2013, year_max=2019,
                 months=list(range(1,13)), reliability=99,
                 start_up_time=30*24*3):

        demand = get_demand(year_min, year_max, months)
        super().__init__(gen_list, stor_list, demand, reliability=reliability,
                         start_up_time=start_up_time)
    
