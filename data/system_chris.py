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

from scipy.optimize import Bounds, LinearConstraint, minimize, shgo
from pyDOE import lhs
from os import listdir

from storage import MultipleStorageAssets
from fns import (_subplot, result_as_txt, get_GB_demand, offset,
                 read_analysis_from_file)
        
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
        #print(gen_cap)
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
            gen[g.name] = g.get_dirunal_profile()
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
               tic0 = 2*2.8e-3*max(self.demand) # arbitrary 
            gen_cap,cost = self.lhs_generation(tic0,number_test_points=8,
                                               stor_cap=stor_cap)

        # next find the optimal storage relative sizes for that generation
        if self.n_storage > 1:
            print('Starting generation capacities are: ',gen_cap)
            print('Searching for an optimal storage ratio for the given generation capacities...')
            print('Starting storage ratio is: ',stor_cap)
            stor_cap = self.optimise_storage_ratio(gen_cap,stor_cap)
            print('Optimal storage ratio is: ', stor_cap)

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

        
class ElectricitySystemGB(ElectricitySystem):

    def __init__(self, gen_list, stor_list, year_min=2013, year_max=2019,
                 months=list(range(1,13)), reliability=99,strategy='ordered',
                 start_up_time=30*24*3,electrify_heat=False,evs=False):

        demand = get_GB_demand(year_min, year_max, months,
                               electrify_heat=electrify_heat, evs=evs)
        super().__init__(gen_list, stor_list, demand, reliability=reliability,
                         start_up_time=start_up_time)

class ElectricitySystem2030(ElectricitySystem):

    def __init__(self, gen_list, stor_list, year_min=2018, year_max=2019,
                 months=list(range(1,13)), reliability=99,strategy='ordered',
                 start_up_time=365,elec_demand=228.0,heat_demand=30.1,
                 ev_demand=0):

        demand = get_GB_demand(year_min, year_max, months,
                               elec_demand, heat_demand, ev_demand)
                              # heat_demand=heat_demand,ev_demand=ev_demand)
        super().__init__(gen_list, stor_list, demand, reliability=reliability,
                         start_up_time=start_up_time)
        
class ElectricitySystemXLinks(ElectricitySystem):

    def __init__(self, gen_list, stor_list, year_min=2013, year_max=2019,
                 months=list(range(1,13)), reliability=100,strategy='ordered',
                 start_up_time=0,elec_demand=3.6,heat_demand=0,
                 ev_demand=0):

        demand = [elec_demand*1000]*61344
        super().__init__(gen_list, stor_list, demand, reliability=reliability,
                         start_up_time=start_up_time)
        
class ElectricitySystemInputDemand(ElectricitySystem):
    
    def __init__(self, gen_list, stor_list, year_min=2013, year_max=2019,
                 months=list(range(1,13)), reliability=100,strategy='ordered',
                 start_up_time=0,elec_demand=3.6,heat_demand=0,
                 ev_demand=0):

        demand = []
        with open('FIF_data/dairy_demand.csv','r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                demand.append(float((row[0])))
        super().__init__(gen_list, stor_list, demand, reliability=reliability,
                         start_up_time=start_up_time)    
    
