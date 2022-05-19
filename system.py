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
from os import listdir

from storage import MultipleStorageAssets
from fns import (_subplot, result_as_txt, get_GB_demand, offset,
                 read_analysis_from_file)
import aggregatedEVs as aggEV
        
class ElectricitySystem:

    def __init__(self, gen_list, stor_list, demand, t_res=1, reliability=99,
                 start_up_time=0,strategy='ordered',aggEV_list = aggEV.MultipleAggregatedEVs([])):
        '''
        == description ==
        Initiates base class
        
        == parameters ==
        gen_list: (Array<GenerationModel>) a list containing generator objects
        stor_list: (Array<StorageModel>)
        aggEV_list: (MultipleAggregatedEVs) class of multiple agg EV fleets from teh aggEVs class, default arg is an empty set
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
                print(len(gen.power_out),len(demand))
                raise Exception('supply and demand have different lengths')

        self.gen_list = gen_list
        self.aggEV_list = aggEV_list 
        self.demand = demand
        self.t_res = t_res
        self.len = len(demand)
        self.total_installed_generation = 0
        self.n_storage = len(stor_list)
        self.reliability = reliability
        self.start_up_time = start_up_time
        self.strategy = strategy
        self.stor_list = stor_list

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
        print(gen_cap)
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
        
    def get_reliability(self, start_up_time=0, return_output=False, return_soc=False,
                        strategy='ordered'):
        '''
        == description ==
        Calculates the percentage of time that demand that can be met with current system

        == parameters ==
        None

        == returns ==
        (float) percentage of demand met (0-100)

        if return_output is True:
        (Array<float>) the smoothed supply profile
        '''
        if self.n_storage == 1:
            rel = self.storage.units[0].charge_sim(self.surplus, t_res=self.t_res,
                                          return_output=return_output,
                                          return_soc=return_soc,
                                          start_up_time=start_up_time)
        else:
            rel = self.storage.charge_sim(self.surplus,t_res=self.t_res,
                                          start_up_time=start_up_time,
                                          return_output=return_output,
                                          strategy=strategy)
        return rel

    def cost(self, x, preoptimised = False): 
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
        preoptimised: (binvar) true if the optimise_conf has been run, false elsewise. User must input

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
            print(total*1e-9)
            return total*1e-9

    def analyse(self,x,filename='log/system_analysis.txt'):
        '''
        == description ==
        Stores analysis of the described system as a text file

        == parameters ==
        x: (Array<float>) the first n_gen elements contain the installed
            capacity in GW of each generation unit (in the order they are
            specified in gen_list). The other elements are the proportion of the
            total storage capacity that each unit comprises. Note that there
            will be one fewer values than number of storage assets (as the
            remaining proportion of 1.0 is allocated to the last one.
        filename: (str) path for the analysis to be stored

        == returns ==
        None
        '''
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

    def get_diurnal_profile(self,gen_cap,stor_cap):
        '''
        == description ==
        Plots the load profiles of the average day

        == parameters ==
        gen_cap: (Array<float>) The installed capacity in GW of each generation
            unit (in the order they are specified in gen_list)
        stor_cap: (Array<float>) The ratio of storage capacities to use

        == returns ==
        None
        '''

        self.scale_generation(gen_cap)

        gen = {}
        gs = []
        for g in self.gen_list:
            gen[g.name] = g.get_diurnal_profile()
            gs.append(g.name)
            
        stor_cap += [1-sum(stor_cap)]
        
        self.storage.rel_capacity = stor_cap
        
        self.update_surplus()
        sc = self.storage.size_storage(self.surplus, self.reliability,
                                       start_up_time=self.start_up_time,
                                       strategy=self.strategy)
        self.storage.capacity = sc
        res = self.storage.charge_sim(self.surplus,
                                      start_up_time=self.start_up_time,
                                      strategy=self.strategy,return_di_av=True)
        stor = res[1]

        # get demand profile
        d = [0.0]*24
        for t in range(len(self.demand)):
            d[t%24] += self.demand[t]*24*1e-3/len(self.demand)

        plt.figure(figsize=(7.7,4.5))
        plt.rcParams["font.family"] = 'serif'
        plt.rcParams['font.size'] = 10
        b = [0.0]*24
        u = [0.0]*24
        for i in range(self.n_storage):
            for t in range(24):
                b[t] += stor[i]['c'][t]*1e-3
        
        plt.ylim(1.2*min(b),2*max(d))
        
        plt.fill_between(range(24),b,u,label='Storage',zorder=3,color='#FFE033')
        st = copy.deepcopy(b)
        total = [0.0]*24
        for i in range(len(gs)):
            for t in range(24):
                total[t] += gen[gs[i]][t]

        gens = []
        for i in range(len(gs)):
            p = gen[gs[i]]
            for t in range(len(p)):
                p[t] = p[t]*1e-3-st[t]*p[t]/total[t]
            gens.append(p)

        i = 0
        while i < len(gens):
            for t in range(24):
                u[t] = b[t]+gens[i][t]
            
            plt.fill_between(range(len(p)),b,u,label=gs[i],zorder=2,
                             color='#33'+str(4*i+1)+str(4*i+1)+'FF')
            b = copy.deepcopy(u)
            i += 1
        plt.plot(d,zorder=4,c='k',ls="--",label='Demand')

        for i in range(self.n_storage):
            for t in range(24):
                u[t] += stor[i]['d'][t]*1e-3

        plt.fill_between(range(24),b,u,zorder=3,color='#FFE033')
                
        plt.grid(ls=':',zorder=0)
        plt.ylabel('Power (GW)')
        plt.xticks(np.linspace(2,21,num=6),['02:00','06:00','10:00','14:00',
                                            '18:00','22:00'])
        plt.xlim(0,23)
        plt.legend(ncol=len(self.gen_list)+2)
        plt.tight_layout()

        
        plt.figure(figsize=(7.7,4.5))
        for i in range(self.n_storage):
            plt.subplot(1,self.n_storage,i+1)
            plt.title(self.storage.units[i].name)
            
            plt.plot(stor[i]['c'],label='Charging',c='b',zorder=3)
            plt.plot(stor[i]['d'],label='Discharging',c='r',zorder=3)
            plt.xticks(np.linspace(2,21,num=6),['02:00','06:00','10:00','14:00',
                                                '18:00','22:00'])
            if -1*min(stor[i]['c']) > max(stor[i]['d']):
                plt.ylim(1.1*min(stor[i]['c']),-1.1*min(stor[i]['c']))
            else:
                plt.ylim(-1.1*max(stor[i]['d']),1.1*max(stor[i]['d']))
            plt.plot(range(24),[0]*24,c='k',ls=':')
            plt.xlim(0,23)
            plt.grid(ls=':',zorder=1)
            if i == 0:
                plt.ylabel('Grid-side power (MW)')
        
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def cost_fixed_gen_and_stor_ratios(self,x):
        '''
        == description ==
        Using existing relative generation and storage sizing, works out the
        cost for a given total installed generation capacity

        == parameters ==
        x: (Array<float>) only one element - the total installed capacity

        == returns ==
        (float) total system cost £bn /year
        '''

        total = self.scale_generation_tic(x[0])
        self.update_surplus()
        sc = self.storage.size_storage(self.surplus, self.reliability,
                                       start_up_time=self.start_up_time,
                                       strategy=self.strategy)
        if sc == np.inf:
            return np.inf
        else:
            total += self.storage.get_cost()
            print(total*1e-9)
            return total*1e-9

    def cost_fixed_gen(self, x): 
        '''
        == description ==
        Calculates the cost using the existing generation capacity, but for the
        stated relative storage sizes.

        == parameters ==
        x: (Array<float>) the proportion of the first n-1 storage capacities
        (the final storage asset is allocated the remaining share).

        == returns ==
        (float) total system cost £bn /year
        '''
        stor_cap = list(x)
        stor_cap.append(1-sum(stor_cap))
        self.storage.rel_capacity = stor_cap
        
        self.update_surplus()
        sc = self.storage.size_storage(self.surplus, self.reliability,
                                       start_up_time=self.start_up_time,
                                       strategy=self.strategy)
        if sc == np.inf:
            return np.inf
        else:
            total = self.storage.get_cost()
            print(total*1e-9)
            return total*1e-9


    def optimise_total_installed_capacity(self,tic,stor_cap):
        '''
        == description ==
        Optimises the total installed capacity for a given relative sizes of
        generators and storage

        == parameters ==
        tic: (float) initial total installed capacity (GW)
        stor_cap: (Array<float>) The ratio of storage capacities to use

        == returns ==
        (flaot) optimal total installed capacity (GW)
        (float) total system cost £bn /year
        '''

        stor_cap.append(1-sum(stor_cap))
        self.storage.rel_capacity = stor_cap

        bounds = Bounds([0.1],[np.inf])

        res = minimize(self.cost_fixed_gen_and_stor_ratios, [tic],
                       bounds=bounds, jac='2-point',method='SLSQP',#L-BFGS-B',
                       options={'finite_diff_rel_step':[1e-2],'ftol':1e-3})

        cost = res.fun
        tic = res.x[0]

        return tic, cost

    def lhs_generation(self, tic, number_test_points=20, stor_cap=None): 
        '''
        == description ==
        This function performs a random search over different generation
        capacity ratios. Latin hypercube sampling is used to select dissimilar
        start points and then the generators are scaled such to the specified
        total installed capacity. Points violating the generation limits are
        ignored, otherwise the cost is calculated.

        ==  parameters ==
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
            if max(self.max_gen_cap) < np.inf:
                for j in range(len(self.min_gen_cap)):
                    gen_cap.append((x[i][j]*(self.max_gen_cap[j] -
                                             self.min_gen_cap[j])
                                    + self.min_gen_cap[j]))
            else:
                for j in range(len(x[i])):
                    gen_cap.append(x[i][j])
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
        
        bounds = Bounds([0.85]+[0.0]*(self.n_storage-1),
                        [1.15]+[1.0]*(self.n_storage-1))

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
                           constraints=constraints, bounds=bounds)
            stor_cap = list(res.x)[1:]
            tic = list(res.x)[0]*self.tic0
            cost = res.fun

        gen_cap = []
        for i in range(len(installed_gen)):
            gen_cap.append(installed_gen[i]*tic/sum(installed_gen))
        
        
        return gen_cap, stor_cap, cost

    def optimise_storage_ratio(self, gen_cap, stor_cap):
        self.scale_generation(gen_cap)

        bounds = Bounds([0.0]*(self.n_storage-1),[1.0]*(self.n_storage-1))

        if self.n_storage > 2:
            # constraint to ensure that storage ratios add up to less than 1
            linear_constraint = LinearConstraint([1.0]*(self.n_storage-1),
                                                 [0], [1])
            constraints = [linear_constraint]
        else:
            constraints = ()

        res = minimize(self.cost_fixed_gen, stor_cap,constraints=constraints,
                       bounds=bounds, jac='2-point',method='SLSQP',
                       options={'ftol':1e-3,
                                'finite_diff_rel_step':[1e-1]*(self.n_storage-1)})
        stor_cap = list(res.x)
        
        return stor_cap


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

        # ok, how about a three stage process: opimise generation ratio,
        # optimise storage, optimise amount of
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
            print('Specifying an initial relative storage capacity will '+
                  'improve speed and accuracy')
            stor_cap = [1.0/self.n_storage]*(self.n_storage-1)

        # If a set of generation capacities not given, search for one
        if gen_cap is None:
            print('Searching for a starting set of generation capacities...')
            # perform a lhs search over different generator ratios
            if tic0 is None:
               tic0 = 2.8e-3*max(self.demand) # arbitrary 
            gen_cap,cost = self.lhs_generation(tic0,number_test_points=15,
                                               stor_cap=stor_cap)

        # next find the optimal storage relative sizes for that generation
        if self.n_storage > 1:
            print('Starting generation capacities are: ',gen_cap)
            print('Searching for an optimal storage ratio for the given generation capacities...')
            print('Starting storage ratio is: ',stor_cap)
            stor_cap = self.optimise_storage_ratio(gen_cap,stor_cap)
            print('Optimal storage ratio is: ', stor_cap)
        else:
            self.cost(gen_cap)

        # finally optimise the total installed generators
        print('Optimising the total installed capacities of generation and storage (ratios fixed)...')
        tic, cost = self.optimise_total_installed_capacity(sum(gen_cap),
                                                           stor_cap)

        tic_sf = copy.deepcopy(tic/sum(gen_cap))
        for i in range(len(gen_cap)):
            gen_cap[i] = gen_cap[i]*tic_sf
        
        if analyse is True:
            self.analyse(list(gen_cap)+list(stor_cap),
                         filename='log/opt_results.txt')

        return list(gen_cap)+list(stor_cap[:-1]), cost

    def sensitivity_analysis(self,var,mult_facts,var_name,max_gen_cap,
                             min_gen_cap,stor_cap=None,tic0=None):
        orig = copy.deepcopy(var)
        gen_cap=None
        for mf in mult_facts:
            var = orig*mf
            filename = 'log/sens_'+var_name+str(var)+'.txt'
            _x,cost = self.optimise(stor_cap=stor_cap,gen_cap=gen_cap,tic0=tic0,
                                    min_gen_cap=min_gen_cap,analyse=False,
                                    max_gen_cap=max_gen_cap)

            self.analyse(_x,filename=filename)

            print(_x)

            # offset new start points to prevent getting stuck in local optima
            gen_cap = offset(_x[:len(self.min_gen_cap)])
            stor_cap = offset(_x[len(self.min_gen_cap):])

            print(gen_cap)
            print(stor_cap)


    def plot_sensitivity_results(self,var_name):
        
        
        plt.figure(figsize=(7.7,4.5))
        plt.rcParams["font.family"] = 'serif'
        plt.rcParams['font.size'] = 10
        
        key = 'sens_'+str(var_name)
        found = []
        vals = []

        lst = listdir('log/')
        for f in lst:
            if f[:len(key)] == key:
                i = len(key)
                j = copy.deepcopy(i)
                while f[j:j+2] != '.t':
                    j += 1
                found.append([float(f[i:j]),f])
        found = sorted(found)
        
        y = {}
        order = []
        for v in range(len(found)):
            vals.append(found[v][0])
            res = read_analysis_from_file('log/'+found[v][1])
            for i in range(len(res)):
                if res[i][0] not in y:
                    y[res[i][0]] = []
                    order.append(res[i][0])
                y[res[i][0]].append(res[i][1])

        plt.figure()
        i = 0
        for i in range(len(order)):
            _subplot(order,i+1)
            plt.plot(vals,y[order[i]])
            plt.ylabel(order[i])
            plt.xlabel(var_name)
            plt.grid()

        plt.tight_layout()
        plt.show()
    
    def simulate(self, ordered_charging = False):
        '''
        == description ==
        Simulates the system operation for the timeperiod

        == parameters ==
        ordered_charging: (<bool>) if True then the system is simulated semi-causally, with EVs being charged optimally based on 24hr forecasts and then the storage is charged in a specified order from the remaining surplus
                                    if False, then the system operation is optimised over the entire year using the given capacities of generator, storage and charger types, outputting the new reliability

        == returns ==
        None
        '''
        
        #if(not ordered_charging):
            
            
    
    def new_analyse(self,filename='log/system_analysis.txt'):
        '''
        == description ==
        Stores analysis of the described system as a text file. This was added by Mac to include EVs, done after optimise system

        == parameters ==
        

        == returns ==
        None
        '''
        
        f = open(filename,'w')
#        f.write('System cost: £'+str(c)+' bn/yr\n\n')
        f.write('---------------------\n')
        f.write('INSTALLED GENERATION\n')
        f.write('---------------------\n\n')
        for i in range(len(self.gen_list)):
            f.write(self.gen_list[i].name+': '+str(int(self.gen_list[i].total_installed_capacity)*1e-3)+' GW\n\n')
#        f.write('\n>>TOTAL: '+str(sum(x[:len(self.gen_list)]))+' GW\n\n')
        f.write('------------------\n')
        f.write('INSTALLED STORAGE\n')
        f.write('------------------\n\n')
        for i in range(len(self.stor_list)):
            f.write(self.stor_list[i].name+': '+
                    str(int(self.stor_list[i].capacity*1e-2)*1e-4)+' TWh\n')
            
        f.write('\n>>TOTAL: '+str(int(sum(self.stor_list[i].capacity for i in range(len(self.stor_list)) )*1e-2)*1e-4)+' TWh\n\n')

        f.write('------------------\n')
        f.write('INSTALLED EV CHARGERS\n')
        f.write('------------------\n\n')
        for i in range(self.aggEV_list.n_assets):
            f.write(self.aggEV_list.assets[i].name+': #V2G: '+
                    str(int(self.aggEV_list.assets[i].chargertype[0]*self.aggEV_list.assets[i].number))+', #Smart: '+str(int(self.aggEV_list.assets[i].chargertype[1]*self.aggEV_list.assets[i].number)) + '\n')
            


        f.write('\n--------------------\n')
        f.write('STORAGE UTILISATION\n')
        f.write('--------------------\n\n')
 #      use = self.storage.analyse_usage()
        
        n_years = len(self.demand)/(365*24)
  #      curt = use[2]/n_years
        for i in range(self.n_storage):
            f.write('>> '+self.stor_list[i].name+' <<\n\n')
            f.write(str(int(self.stor_list[i].en_in*1e-3/n_years)*1e-3) + ' TWh/yr in (grid side)\n')
            f.write(str(-int(self.stor_list[i].en_out*1e-3/n_years)*1e-3)
                    +' TWh/yr out (grid side)\n\n')
            
        for i in range(self.aggEV_list.n_assets):
            f.write('>> '+self.aggEV_list.assets[i].name+' <<\n\n')
            f.write('V2G: '+
                str(int(self.aggEV_list.assets[i].V2G_en_in*1e-3)*1e-3)+' TWh/yr in (grid side)\n')
            f.write('V2G: '+
                str(int(self.aggEV_list.assets[i].V2G_en_out*1e-3)*1e-3)+' TWh/yr out (grid side)\n\n')
            f.write('Smart: '+
                str(int(self.aggEV_list.assets[i].Smart_en_in*1e-3)*1e-3)+' TWh/yr in (grid side)\n\n')

   #         cycles = (use[1][i]*100/(self.storage.units[i].eff_out*n_years
   #                                  *self.storage.units[i].capacity))
   #         f.write(str(cycles)+' cycles per year\n\n')

        f.write('-------------------\n')
        f.write('ENERGY UTILISATION\n')
        f.write('-------------------\n\n')
        f.write('Total Passive Demand: '
                + str(int(sum(self.demand)*1e-3/(self.t_res*n_years))*1e-3)
                + ' TWh/yr\n')
        f.write('Total EV Driving Demand: '
                + str(int(self.aggEV_list.driving_energy*1e-3/(self.t_res*n_years))*1e-3)
                + ' TWh/yr\n')
        f.write('Total Supply: '
                + str( int((sum(sum(self.gen_list[i].power_out)for i in range(len(self.gen_list)))*1e-3)*1e-3)
                      /(self.t_res*n_years) )
                + ' TWh/yr\n') 
        f.write('Curtailment: '+str(int(sum(self.aggEV_list.Shed*1e-3)/n_years)*1e-3)+' TWh/yr\n\n')

        f.write('---------------\n')
        f.write('COST BREAKDOWN\n')
        f.write('---------------\n\n')
        for i in range(len(self.gen_list)):
            f.write(self.gen_list[i].name+': £'
                    +str(int(1e-6*self.gen_list[i].total_installed_capacity*self.gen_list[i].fixed_cost)*1e-3)+' bn/yr\n')
        for i in range(len(self.stor_list)):
            f.write(self.stor_list[i].name+': £'
                    +str(int(1e-6*self.stor_list[i].capacity*self.stor_list[i].fixed_cost)*1e-3)+' bn/yr\n')
        f.close()

        
    def plot_timeseries(self,start=0,end=-1):
        
        '''   
        == parameters ==
        start: (int) start time of plot
        end: (int) end time of plot
        
        '''

        if (self.aggEV_list.Pfos.shape == ()):
            print('Charging timeseries not avaialable, try running MultipleAggregatedEVs.optimise_charger_type().')
        else:
            if(end<=0):
                timehorizon = self.aggEV_list.Pfos.size
            else:
                timehorizon = end
            plt.rc('font', size=12)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(start,timehorizon), self.aggEV_list.Pfos[start:timehorizon], color='tab:red', label='FF Power')
            ax.plot(range(start,timehorizon), self.aggEV_list.Shed[start:timehorizon], color='tab:blue', label='Renewable Shed')
            ax.plot(range(start,timehorizon), self.aggEV_list.surplus[start:timehorizon], color='tab:orange', label='Surplus')
    
            # Same as above
            ax.set_xlabel('Time (h)')
            ax.set_ylabel('Power (MW)')
            ax.set_title(' Power Timeseries')
            ax.grid(True)
            ax.legend(loc='upper left');   
        
        
            plt.rc('font', size=12)
            fig, ax = plt.subplots(figsize=(10, 6))
            colours = ['b','g','r','c','m','y','b','g','r','c','m','y']
            for g in range(len(self.gen_list)):                
                ax.plot(range(start,timehorizon), self.gen_list[g].power_out[start:timehorizon], color=colours[g], label='Gen '+self.gen_list[g].name)
            ax.plot(range(start,timehorizon), self.demand[start:timehorizon], color='tab:orange', label='Demand')
    
            # Same as above
            ax.set_xlabel('Time (h)')
            ax.set_ylabel('Power (MW)')
            ax.set_title(' Generation')
            ax.grid(True)
            ax.legend(loc='upper left'); 
        

        
class ElectricitySystemGB(ElectricitySystem):

    def __init__(self, gen_list, stor_list, year_min=2013, year_max=2019,
                 months=list(range(1,13)), reliability=99,strategy='ordered',
                 start_up_time=30*24*3,electrify_heat=False,evs=False,aggEV_list = aggEV.MultipleAggregatedEVs([])):

        demand = get_GB_demand(year_min, year_max, months,
                               electrify_heat=electrify_heat, evs=evs)
        super().__init__(gen_list, stor_list, demand, reliability=reliability,
                         start_up_time=start_up_time,aggEV_list=aggEV_list)

        
class DispatchableOutput(ElectricitySystem):

    def __init__(self, generator, storage):
        super().__init__([generator], storage, [0.0]*len(generator.power_out))



    def plot_reliability_curve(self, target_load_factor, min_installed=0,
                               max_installed=100, step=1, hold_on=False,
                               plot_number=1, label=None, start_up_time=0):
        if label is None:
            label=str(int(target_load_factor))+'% target'
            
        goal = target_load_factor*self.total_installed_generation/100
        self.demand = [goal]*self.len
        self.update_surplus()

        storage_capacity = np.arange(min_installed, max_installed, step)
        reliability = []
        for sc in storage_capacity:
            self.storage.set_capacity(sc*self.total_installed_generation)
            reliability.append(self.get_reliability(start_up_time=start_up_time))
            
        plt.figure(plot_number)
        plt.rcParams["font.family"] = 'serif'
        plt.rcParams['font.size'] = 10
        plt.plot(storage_capacity,reliability,label=label)
        if hold_on is False:
            plt.ylabel('Reliability')
            plt.xlabel('Storage installed per unit generation (MWh per MW)')
            plt.grid(ls=':')
            plt.legend()
            plt.tight_layout()
            plt.show()
        
    def plot_reliability_cost_curve(self, target_load_factor, min_installed=0,
                                    max_installed=100, step=1, hold_on=False,
                                    plot_number=1, label=None, start_up_time=0):
        if label is None:
            label=str(int(target_load_factor))+'% target'
            
        goal = target_load_factor*self.total_installed_generation/100
        self.demand = [goal]*self.len
        self.update_surplus()

        storage_capacity = np.arange(min_installed, max_installed, step)
        reliability = []
        cost = []
        for sc in storage_capacity:
            self.storage.set_capacity(sc*self.total_installed_generation)
            reliability.append(self.get_reliability(start_up_time=start_up_time))
            cost.append(self.storage.get_cost())
            if sc == storage_capacity[-1]:
                print(self.storage.analyse_usage())
            
        plt.figure(plot_number)
        plt.rcParams["font.family"] = 'serif'
        plt.rcParams['font.size'] = 10
        plt.subplot(2,1,1)
        plt.plot(storage_capacity,reliability,label=label)
        plt.subplot(2,1,2)
        plt.plot(cost,reliability,label=label)
        if hold_on is False:
            plt.ylabel('Reliability')
            plt.grid(ls=':')
            plt.xlabel('Storage cost per unit generation (£bn per MW-year)')
            plt.subplot(2,1,1)
            plt.xlabel('Storage installed per unit generation (MWh per MW)')
            plt.grid(ls=':')
            plt.legend()
            plt.tight_layout()
            plt.show()

class CostOptimisation(ElectricitySystem):

    def __init__(self, gen_list, stor_list, demand='GB', min_gen_cap=[],
                 max_gen_cap = [], reliability=99,
                 start_up_time=30*24*3, year_min=2013, year_max=2019,
                 months=list(range(1,13))):

        if demand == 'GB':
            demand = get_demand(year_min, year_max, months)
        super().__init__(gen_list, stor_list, demand)

        self.reliability = reliability
        self.demand = demand
        self.start_up_time = start_up_time
        self.min_gen_cap = min_gen_cap
        self.max_gen_cap = max_gen_cap

    def cost_generation_only(self, gen_cap):
        total =  self.scale_generation(gen_cap)
        self.update_surplus()
        sc = self.storage.size_storage(self.surplus, self.reliability,
                                       start_up_time=self.start_up_time)
        if sc == np.inf:
            return np.inf
        else:
            total += self.storage.get_cost()
            return total*1e-9
        
    def cost_storage_only(self, x):
        total = 0
        stor_cap = list(x)
        stor_cap.append(1-sum(stor_cap))

        self.storage.rel_capacity = stor_cap
        
        self.update_surplus()
        sc = self.storage.size_storage(self.surplus, self.reliability,
                                       start_up_time=self.start_up_time)
        
        if sc == np.inf:
            return np.inf
        else:
            total += self.storage.get_cost()
            return total*1e-9

    def optimise_generation(self,number_test_points=20,stor_cap=None,
                            refine=False,x0=None):
        if stor_cap is None:
            if self.n_storage == 1:
                stor_cap = []
            else:
                stor_cap = [1.0/self.n_storage]*(self.n_storage-1)

        if refine is False or x0 is None:
            x = lhs(len(self.gen_list),samples=number_test_points)

            best = None
            lwst = np.inf
            for i in range(number_test_points):
                for j in range(len(self.min_gen_cap)):
                    # now scale the test points
                    x[i][j] = (x[i][j]*(self.max_gen_cap[j] - self.min_gen_cap[j])
                               + self.min_gen_cap[j])
                f = self.cost(list(x[i])+stor_cap)
                if f < lwst:
                    lwst = f
                    best = list(x[i])
            x0 = best

        if refine is True:
            bounds = Bounds(self.min_gen_cap,self.max_gen_cap)
            res = minimize(self.cost_generation_only, x0, bounds=bounds,
                           tol=1e-3)
            best = res.x
            lwst = res.fun

        return best, lwst

    def optimise0(self, stor_cap=None):
        # need an exit clause for the single storage case
        # three stage process

        gen_cap,cost = self.optimise_generation(number_test_points=10,
                                                refine=False,stor_cap=stor_cap)
        stor_cap,cost = self.optimise_storage(gen_cap,stor_cap)
        gen_cap,cost = self.optimise_generation(stor_cap=stor_cap,refine=True,
                                                x0=gen_cap)

        return list(gen_cap)+list(stor_cap), cost

    def optimise(self, stor_cap=None, gen_cap=None, show_slices=False,
                 analyse_results=True,filename='log/system_analysis.txt'):
        # need an exit clause for the single storage case
        # three stage process

        if gen_cap is None:
            gen_cap,cost = self.optimise_generation(number_test_points=5,
                                                    refine=False,
                                                    stor_cap=stor_cap)
        else:
            cost = self.cost(gen_cap+stor_cap)
        print(gen_cap)
        print(stor_cap)
        gen_cap,stor_cap,cost = self.optimise_storage(gen_cap,stor_cap)
        print(gen_cap)
        print(stor_cap)
        gen_cap,cost = self.optimise_generation(refine=True,x0=gen_cap,
                                                stor_cap=stor_cap)
        print(gen_cap)
        print(stor_cap)

        f = open('log/opt_results.txt','w')
        res = 'MIN:'+result_as_txt(gen_cap,stor_cap,cost,self.storage.capacity)
        f.write(res)
        f.close()

        if show_slices is True:
            self.plot_slices(gen_cap,stor_cap,cost)

        if analyse_results is True:
            self.analyse(list(gen_cap)+list(stor_cap),filename=filename)

        return list(gen_cap)+list(stor_cap), cost

    def optimise1(self, tic0=None, stor_cap=None, gen_cap=None,
                  show_slices=False, analyse_results=True,
                  filename='log/system_analysis.txt'):
        # need an exit clause for the single storage case
        # three stage process
        if tic0 is None:
            tic0 = 2.8e-3*max(self.demand) # arbitrary fudge

        if gen_cap is None:
            gen_cap,cost = self.lhs_generation(tic0,number_test_points=20,
                                               stor_cap=stor_cap)
        else:
            cost = self.cost(gen_cap+stor_cap)
            
        gen_cap,stor_cap,cost = self.optimise_storage(gen_cap,stor_cap)
        
        f = open('log/opt_results.txt','w')
        res = 'MIN:'+result_as_txt(gen_cap,stor_cap,cost,self.storage.capacity)
        f.write(res)
        f.close()

        if show_slices is True:
            self.plot_slices(gen_cap,stor_cap,cost)

        if analyse_results is True:
            self.analyse(list(gen_cap)+list(stor_cap),filename=filename)

        return list(gen_cap)+list(stor_cap), cost
            
    def plot_slices(self,gen_cap,stor_cap,cost):
        for i in range(len(gen_cap)):
            x = []
            y = []
            for mf in [0.8,0.9,0.95,1,1.05,1.1,1.2]:            
                g = copy.deepcopy(gen_cap)
                g[i] = g[i]*mf
                x.append(g[i])
                if mf == 1:
                    c = cost
                else:
                    c = self.cost(g+stor_cap)
                y.append(c)
                f = open('log/opt_results.txt','a+')
                res = 'G'+str(i)+'-'+str(mf)+':'+result_as_txt(g,stor_cap,c,self.storage.capacity)
                f.write(res)
                f.close()

                _subplot(gen_cap+stor_cap,i+1)
                plt.plot(x,y)
                plt.scatter([gen_cap[i]],[cost],marker='x',c='r')
                plt.ylabel('Cost (£bn/yr)')
                plt.xlabel(self.gen_list[i].name+' (GWh)')

        for i in range(len(stor_cap)):
            x = []
            y = []
            if stor_cap[i] < 1e-2:
                for af in [-0.5e-2,-1e-3,0,1e-3,5e-3,1e-2]:
                    s = copy.deepcopy(stor_cap)
                    s[i] += af
                    if s[i] < 0:
                        s[i] = 0
                    x.append(s[i]*self.storage.capacity*1e-3)
                    if af == 0:
                        c = cost
                    else:
                        c = self.cost(gen_cap+s)
                    y.append(c)
                    f = open('log/opt_results.txt','a+')
                    res = 'S'+str(i)+'-'+str(mf)+':'+result_as_txt(gen_cap,s,c,self.storage.capacity)
                    f.write(res)
                    f.close()

                    _subplot(gen_cap+stor_cap,i+1+len(gen_cap))
                    plt.plot(x,y)
                    plt.ylabel('Cost (£bn/yr)')
                    plt.xlabel(self.storage.units[i].name+' (GWh)')

            else:
                for mf in [0.8,0.9,0.95,1.05,1.1,1.2]: 
                    s = copy.deepcopy(stor_cap)
                    s[i] = mf*s[i]
                    x.append(s[i]*self.storage.capacity*1e-3)
                    if mf == 1:
                        c = cost
                    else:
                        c = self.cost(gen_cap+s)
                    y.append(c)
                    f = open('log/opt_results.txt','a+')
                    res = 'S'+str(i)+'-'+str(mf)+':'+result_as_txt(gen_cap,s,c,self.storage.capacity)
                    f.write(res)
                    f.close()

                    _subplot(gen_cap+stor_cap,i+1+len(gen_cap))
                    plt.plot(x,y)
                    plt.ylabel('Cost (£bn/yr)')
                    plt.xlabel(self.storage.units[i].name+' (GWh)')
                    
        plt.tight_layout()
        plt.show()

    def search_single_storage_ratio(self,gen_cap,first=1e-2,scale_factor=2):
        gen_cost = self.scale_generation(gen_cap)
        ratios = [0.0]
        r = first
        while r < 1.0:
            ratios.append(r)
            r = r*scale_factor
        ratios.append(1.0)

        cost = [0]*len(ratios)
        capacity = [0]*len(ratios)
        sc = None
        for i in np.arange(len(ratios)-1,-1,-1):
            r = ratios[i]
            self.storage.rel_capacity = [r,1-r]
            self.update_surplus()
            if sc == np.inf:
                sc = None
            sc = self.storage.size_storage(self.surplus, self.reliability,
                                           start_up_time=self.start_up_time,
                                           initial_capacity=sc)
            cost[i] = (gen_cost+self.storage.get_cost())*1e-9
            capacity[i] = self.storage.capacity*1e-6

        
        plt.figure()
        plt.subplot(2,1,1)
        title = ''
        for i in range(len(gen_cap)):
            title += str(int(gen_cap[i])) + ' GW gen' + str(i+1) + ', '
        plt.title(title)
        plt.plot(ratios,cost)
        plt.ylabel('Total System Cost (£bn/yr)')
        plt.subplot(2,1,2)
        plt.plot(ratios,capacity)
        plt.ylabel('Storage Capacity (TWh)')
        plt.xlabel('Ratio of storage1:storage2')

        for i in range(1,3):
            plt.subplot(2,1,i)
            plt.grid(ls=':')
            plt.xlim(0,1.01)
        plt.tight_layout()
        plt.show()


    def get_cost_breakdown(self):
        s = self.storage.get_cost()
        g = 0.0
        for i in range(len(self.gen_list)):
            g += self.gen_list[i].get_cost()

        return 100*s/(s+g)
        
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
        print(x)
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
            print(total*1e-9)
            return total*1e-9


    def cost_fixed_gen_ratio1(self, x): 
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
        print(x)
        tic = x[0]
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
            print(total*1e-9)
            return total*1e-9

    def optimise_fixed_gen_ratio1(self, installed_gen, x0):
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
        tic = sum(installed_gen)
        self.scale_generation(installed_gen)
        
        bounds = Bounds([0.0]*(self.n_storage),
                        [sum(self.max_gen_cap)]+[1.0]*(self.n_storage-1))

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
            tic = x#help, don't know
        else:
            res = minimize(self.cost_fixed_gen_ratio1, [tic]+x0,
                           constraints=constraints, bounds=bounds)
            stor_cap = list(res.x[1:])
            tic = x[0]#list(res.x)[0]*self.tic0
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
    
    def optimise1(self, reliability=None, tic0=None, stor_cap=None, gen_cap=None,
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

        # ok, how about a three stage process: opimise generation ratio,
        # optimise storage, optimise amount of
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
            gen_cap,cost = self.lhs_generation(tic0,number_test_points=5,
                                               stor_cap=stor_cap)
        else:
            cost = self.cost(gen_cap+stor_cap)
            
        gen_cap,stor_cap,cost = self.optimise_fixed_gen_ratio1(gen_cap,stor_cap)

        if analyse is True:
            self.analyse(list(gen_cap)+list(stor_cap),
                         filename='log/opt_results.txt')

        return list(gen_cap)+list(stor_cap), cost
        
    def search_gen_scale_factor(self,stor_cap):
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
            f = self.cost_fixed_gen_ratio1([x]+stor_cap)
            if f < lwst:
                best = x
                lwst = f
            if f > lwst*1.1:
                break

        return [best], lwst
    
