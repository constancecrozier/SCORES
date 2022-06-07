import numpy as np
#optimisation high level language, help found at https://www.ima.umn.edu/materials/2017-2018.2/W8.21-25.17/26326/3_PyomoFundamentals.pdf
#low level algorithmic solve performed by solver
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import time
import datetime as dt
from pandas import DataFrame
import matplotlib.pyplot as plt


def opt_results_to_df(model):
        '''
        == description ==
        Takes a solved model and outputs the results of its most recent solve as 2 data frames, one summarising
        the built capacities (df.capital), the other expenditure in pounds (df.costs)!

        == parameters ==
        model: (pyo.ConcreteModel()) This is a pyomo model that has been constructed then solved. The solved parameter values can be extracted from it.
        
        == returns ==
        '''
        
    # Record Built Decisions #
        df1 = DataFrame({'Total Demand (GWh)': [int(sum(-pyo.value(model.Demand[t]) for t in model.TimeIndex)/1000.0)], 'Total Fossil Fuel (GWh)': [int(sum(pyo.value(model.Pfos[t]) for t in model.TimeIndex)/1000.0)],'Total Curtailement (GWh)': [int(sum(pyo.value(model.Shed[t]) for t in model.TimeIndex)/1000.0)]})
        
    #Need to Work out how to get the price as an output (could do with changing the costs to model parameters instead)
        for g in model.GenIndex:
                df1['Gen '+str(g) + ' Cap (GW)'] = int(pyo.value(model.GenCapacity[g])/10.0)/100.0
        
        for i in model.StorageIndex:
                df1['Stor '+str(i) + ' Cap (GWh)'] = int(pyo.value(model.BuiltCapacity[i])/10.0)/100.0
        
        for k in model.FleetIndex:
            for b in model.ChargeType:
                if b == 0:
                    df1['Fleet '+str(k) + ' V2G'] = int(pyo.value(model.EV_TypeBuiltCapacity[k,b]))
                elif b ==1:
                    df1['Fleet '+str(k) + ' Uni'] = int(pyo.value(model.EV_TypeBuiltCapacity[k,b]))
        
    # Record Pricing #
        
        #print('Objective: £',int(pyo.value(model.obj)/1000000),'m')
        #print('Fossil Fuel Cost Penalty: £',int(sum(pyo.value(model.fossilfuelpenalty) * pyo.value(model.Pfos[t]) for t in model.TimeIndex)/1000000),'m')
        #print('Discharge Penalty for Storage: £',int(sum(sum(0.05 * pyo.value(model.D[i,t]) for t in model.TimeIndex)for i in model.StorageIndex)/1000000),'m')
        #print('Discharge Penalty for EVs: £',int(sum( sum((0.05) * pyo.value(model.EV_D[k,t,0]) for t in model.TimeIndex) for k in model.FleetIndex)/1000000),'m')
        
        #print('Number of Years:',max(model.TimeIndex)/(365*24))
        n_yr = max(model.TimeIndex)/(366*24+0.1) #number of years needed to annualize the operating costs
        df2 = DataFrame({'Length of Sizing (yr)': [int(n_yr)]})
        
        
        cum_op = 0.0
        cum_cap = 0.0
        
        for g in model.GenIndex:
            df2['Gen '+str(g) + ' Capital (£m/yr)'] = int(pyo.value(model.GenCapacity[g])*pyo.value(model.GenCosts[g,0])/1000000)
            df2['Gen '+str(g) + ' Operation (£m/yr)'] = int(pyo.value(model.GenCapacity[g])*pyo.value(model.GenCosts[g,1])*sum(model.NormalisedGen[g,t] for t in model.TimeIndex)/(n_yr*1000000))
            cum_cap += pyo.value(model.GenCapacity[g])*pyo.value(model.GenCosts[g,0])
            cum_op += pyo.value(model.GenCapacity[g])*pyo.value(model.GenCosts[g,1])*sum(model.NormalisedGen[g,t] for t in model.TimeIndex)
            
            #print('Gen Capital: (£m/yr)',int(pyo.value(model.GenCapacity[g])*pyo.value(model.GenCosts[g,0])/1000000))
            #print('Gen Use: (£m/yr)',int(pyo.value(model.GenCapacity[g])*pyo.value(model.GenCosts[g,1])*sum(model.NormalisedGen[g,t] for t in model.TimeIndex)/(n_yr*1000000)))
        
        for i in model.StorageIndex:
            df2['Stor '+str(i) + ' Capital (£m/yr)'] =  int(pyo.value(model.StorCosts[i,0]) * pyo.value(model.BuiltCapacity[i])/1000000)
            df2['Stor '+str(i) + ' Operation (£m/yr)'] = int(sum(pyo.value(model.StorCosts[i,1]) * pyo.value(model.D[i,t]) for t in model.TimeIndex)/(n_yr*1000000))
            cum_cap += pyo.value(model.StorCosts[i,0]) * pyo.value(model.BuiltCapacity[i])
            cum_op +=sum(pyo.value(model.StorCosts[i,1]) * pyo.value(model.D[i,t]) for t in model.TimeIndex)
            
        for k in model.FleetIndex:
            for b in model.ChargeType:
                if b == 0:
                    df2['Fleet '+str(k) + ' V2G (£m/yr)'] = int(pyo.value(model.chargercost[k,b]) * pyo.value(model.EV_TypeBuiltCapacity[k,b])/1000000)
                    cum_cap += pyo.value(model.chargercost[k,b]) * pyo.value(model.EV_TypeBuiltCapacity[k,b])
                    #print('V2G Cost (£m/yr):', int(pyo.value(model.chargercost[k,b]) * pyo.value(model.EV_TypeBuiltCapacity[k,b]))/1000000)
                elif b ==1:
                    df2['Fleet '+str(k) + ' Uni (£m/yr)'] = int(pyo.value(model.chargercost[k,b]) * pyo.value(model.EV_TypeBuiltCapacity[k,b])/1000000)
                    cum_cap += pyo.value(model.chargercost[k,b]) * pyo.value(model.EV_TypeBuiltCapacity[k,b])
                    #print('Uni Cost (£m/yr):',int(pyo.value(model.chargercost[k,b]) * pyo.value(model.EV_TypeBuiltCapacity[k,b]))/1000000)
        
        df2['Total Capital (£m/yr)'] = int(cum_cap/1000000)
        df2['Total Operation (£m/yr)'] = int(cum_op/1000000)

        return [df1,df2]
    
  
def store_optimisation_results(model, Mult_aggEV, Mult_Stor, gen_list=[]):
        '''
        == description ==
        #stores the results of a sizing optimisation in the EV, storage and generator objects. Useful for plotting and keeping a record  
        == parameters ==
        model: (pyo.ConcreteModel()) This is a pyomo model that has been constructed then solved. The solved parameter values can be extracted from it.
        Mult_aggEV: The mutliple aggregated fleet object that has been optimised
        Mult_Stor: Multiple storage object that has been optimised
        gen_list: List of Generators used within the optimisation.
        
        == returns ==
        '''
        timehorizon = max(model.TimeIndex) + 1
    #Agg EV Fleets
        EV_SOC_results = np.empty([Mult_aggEV.n_assets,timehorizon+1,2])
        EV_D_results = np.empty([Mult_aggEV.n_assets,timehorizon]) #as only V2G has discharge variable
        EV_C_results = np.empty([Mult_aggEV.n_assets,timehorizon,2])
        
        for k in range(Mult_aggEV.n_assets):
            Mult_aggEV.assets[k].reset()
            
            for b in range(2):
                Mult_aggEV.assets[k].chargertype[b] = pyo.value(model.EV_TypeBuiltCapacity[k,b])/Mult_aggEV.assets[k].number
                #EV_SOC_results[k,0,b] = 0.5 * pyo.value(model.EV_TypeBuiltCapacity[k,b])*Mult_aggEV.assets[k].N[0]*Mult_aggEV.assets[k].max_SOC/1000
                EV_SOC_results[k,0,b] = pyo.value(model.EV_SOC[k,0,b])
                
                for t in range(0,timehorizon):
                    EV_SOC_results[k,t+1,b] = pyo.value(model.EV_SOC[k,t,b])
                    EV_C_results[k,t,b] = pyo.value(model.EV_C[k,t,b])
                    
                    if(b==0):
                        EV_D_results[k,t] = pyo.value(model.EV_D[k,t,b])                               
                        
                
            Mult_aggEV.assets[k].discharge = EV_D_results[k,:]
            Mult_aggEV.assets[k].charge = EV_C_results[k,:,:]
            Mult_aggEV.assets[k].SOC = EV_SOC_results[k,:,:]
            
        for k in range(Mult_aggEV.n_assets):
            for b in range(2):
                
                if(b==0):
                    Mult_aggEV.assets[k].V2G_en_in = sum(EV_C_results[k,:,b]*100/Mult_aggEV.assets[k].eff_in)
                    Mult_aggEV.assets[k].V2G_en_out = sum(EV_D_results[k,:]*Mult_aggEV.assets[k].eff_out/100)
                elif(b==1):
                    Mult_aggEV.assets[k].Smart_en_in = sum(EV_C_results[k,:,b]*100/Mult_aggEV.assets[k].eff_in) 
                    
    #Storage
        SOC_results = np.empty([Mult_Stor.n_assets, timehorizon+1])
        D_results = np.empty([Mult_Stor.n_assets, timehorizon])
        C_results = np.empty([Mult_Stor.n_assets, timehorizon])
        
        for i in range(Mult_Stor.n_assets):
            Mult_Stor.assets[i].reset()
            Mult_Stor.assets[i].set_capacity(pyo.value(model.BuiltCapacity[i]))
            
            
            SOC_results[i,0] = pyo.value(model.SOC[i,0]);
            
            for t in range(0,timehorizon):
                SOC_results[i,t+1]=pyo.value(model.SOC[i,t])
                D_results[i,t] = -pyo.value(model.D[i,t])
                C_results[i,t] = pyo.value(model.C[i,t])
        
            Mult_Stor.assets[i].en_in = sum(C_results[i,:]*100/Mult_Stor.assets[i].eff_in)
            Mult_Stor.assets[i].en_out = sum(D_results[i,:]*Mult_Stor.assets[i].eff_out/100)
            Mult_Stor.assets[i].discharge = D_results[i,:]
            Mult_Stor.assets[i].charge = C_results[i,:]
            Mult_Stor.assets[i].SOC = SOC_results[i,:]
            
    #Generators
        for g in model.GenIndex:  
            if (max(gen_list[g].power_out) < 0.001):
                gen_list[g].power_out = pyo.value(model.GenCapacity[g])*np.asarray(gen_list[g].power_out)
            else:
                gen_list[g].power_out = pyo.value(model.GenCapacity[g])*(np.asarray(gen_list[g].power_out)/max(gen_list[g].power_out))
            
            gen_list[g].total_installed_capacity = (pyo.value(model.GenCapacity[g]))
            
    

#this class produces an object of the linear program for optimal sizing of the system over the specified timelength
class System_LinProg_Model:
    def __init__(self,surplus,fossilLimit,Mult_Stor,Mult_aggEV=[],gen_list=[],YearRange=[]):
        '''
        == description ==
        Initialises the system linear programme

        == parameters ==
        surplus: (Array<float>) the surplus generation to be smoothed in MW, must be an np array!! (can just be demand also (demand should be -ve))
        fossilLimit: (float) fraction of demand (i.e. -ve surplus) that can come from fossil fuels (expected values between 0:0.05....)
        StorAssets: (mult_stor): MUST be a multiple storage object, even if it has a length of zero!
        Mult_aggEV: (Mult_aggEV_Object): Must be multiple fleet object. (even if of length zero!)
        gen_list: (array<generation>): list of the potential renewable generators to build
        YearRange: (array<int>): [MinYear,MaxYear]
        == returns ==
        '''
        
        self.surplus = surplus
        self.fossilLimit = fossilLimit
        self.gen_list= gen_list
        self.YearRange= YearRange
        
        self.df_capital = DataFrame() #this will contain the built capacities post optimisation run
        self.df_costs = DataFrame() #this contains the operational and investement costs of the system
        

        #Check that the correct storage and aggEV objects have been input   
        if isinstance(Mult_Stor, list):
            raise Exception('Error, the Mult_Stor must be a MultipleStorageAssets object rather than just a list. Even when n_assets=0.')
        self.Mult_Stor = Mult_Stor  
            
        if isinstance(Mult_aggEV, list):
            raise Exception('Error, the Mult_aggEV object must be a MultipleAggregatedEVs object, not a list. Even when n_assets=0.')                
        self.Mult_aggEV = Mult_aggEV
        
    def Form_Model(self,start_EV=-1,end_EV=-1, SizingThenOperation = False, includeleapdays=True, fossilfuelpenalty = 1.0, StartSOCEqualsEndSOC=True, InitialSOC = [-1]):     

        '''
        == description ==
        This creates the model object of a linear programme. Useful because Pyomo programmes take a long time to form initially.
        Once formed though they can be 'solved' repeatedly whilst only changing specific parameters, this reduces construction time massively.

        == parameters ==
        SizingThenOperation: (bool): Make this True when the intention is to use the model for repeated system sizing and
                                     then operational simulation (using Run_Sizing_Then_Op() method). This means that the timehorizon
                                     will be one year less than the year range states and that the leap years will be removed.
        includeleapdays: (bool): When set to False the model will ignore leap days. Can be useful if want to run repeated sims with different years data without having to reform the model to include a leap year.
        fossilfuelpenalty: (float): cost using fossil fuels (£/MWh) default is set to one as 0 gives unatural results.
        StartSOCEqualsEndSOC: (bool): when true ensures the SOC for storage and AggEVs is the same for the first and last timestep
        initial_SOC:  array<floats>, float value between 0:1, determines the start SOC of the EVs and batteries (i.e. 0.5 corresponds to them starting 50% fully charged)
                            if single float given, all storage + EVs start on it, if given as array, allows choosing of individual storage start SOCs, specified in order: [stor0,stor1…,Fleet0 V2G, Fleet0 Uni, Fleet1 V2G…]
        start_EV: datetime : This is used to setup the EV plugin timeseries, making sure that weekdays and weekends are properly aligned.
                                Must be set correctly to first hour of simulation (usually midnight on Jan 1st ymin). When no EVs are being optimised it can remain at -1, otherwise an error will output.
        end_EV: datetime : This is used to setup the EV plugin timeseries, making sure that weekdays and weekends are properly aligned.
                                Must be set correctly to first hour of simulation (usually midnight on Jan 1st ymax+1). When no EVs are being optimised it can remain at -1, otherwise an error will output.
        == returns ==
        '''
        self.SizingThenOperation = SizingThenOperation
        self.fossilfuelpenalty = fossilfuelpenalty
        
        print('Forming Optimisation Model...')
        start = time.time()
        
        #if only one Start SOC Value entered, it is used for all storage
        if len(InitialSOC) == 1:
            for x in range(self.Mult_Stor.n_assets + self.Mult_aggEV.n_assets*2 -1):
                InitialSOC.append(InitialSOC[0])
        elif len(InitialSOC) != self.Mult_Stor.n_assets + self.Mult_aggEV.n_assets*2:
            raise Exception('InitialSOC must either be of length 1 or self.Mult_Stor.n_assets + self.Mult_aggEV.n_assets*2')
        
        if(not StartSOCEqualsEndSOC and InitialSOC[0] < 0):
            raise Exception('The start State of Charge is undefined, either set StartSOCEqualsEndSOC=True and/or define InitialSOC between 0-1.')
        
        if(self.Mult_aggEV.n_assets > 0 and (start_EV == -1 or end_EV == -1)):
            raise Exception('When using EVs, the start and end time of the simulation must be entered as datetimes in the start_EV and end_EV inputs.')
        
    # Create model #
        model=pyo.ConcreteModel()  #create the model object, concrete because parameters are known
        #model=pyo.AbstractModel()
        
    # Declare Indexs #        
        model.StorageIndex=range(self.Mult_Stor.n_assets) #Index over the storage assets
        model.FleetIndex=range(self.Mult_aggEV.n_assets) #Index over Fleets
        model.ChargeType = range(2) #Index over Type of Charger, 0: V2G, 1: Smart, 2: Unmanaged
        model.GenIndex = range(len(self.gen_list)) #Index over possible generators
            
        
    # Remove Leap Days if Used For Repeated Simulations #       
        if(self.SizingThenOperation):
            #check for full years as input
            for g in model.GenIndex:
                if(not(len(self.gen_list[g].months)==12)):
                    print('To use for system sizing then subsequent operation, must use complete years not months.')
                    return
            #check that the YearRange Has Been Defined
            if(not(len(self.YearRange)==2)):
                print('Please Specify the YearRange input to use system sizing then operation.')
                return
            
            if(self.YearRange[1]-self.YearRange[0] <= 0):
                print('YearRange[1] must be larger than YearRange[0]')
                return
            
            #assign datetimes to the demand and generation inputs
            x = dt.timedelta(hours = 1)
            N1 = dt.datetime(self.YearRange[0], 1, 1, hour = 0)
            
            date_range = []
            for t in range(len(self.surplus)):
                date_range.append(N1)
                N1=N1+x
            
            #delete the correct values from power_out, demand and date_range!
            for t in range(len(self.surplus)-1,-1,-1):
                if date_range[t].month == 2 and date_range[t].day == 29:
                    date_range.pop(t)                #list
                    self.surplus = np.delete(self.surplus,[t]) #np.array
                    for g in model.GenIndex:
                        self.gen_list[g].power_out.pop(t) #list
            
            self.date_range = date_range                                                    
            #then get the new timehorizon, will be 365*24*(YearRange[1]-YearRange[0] - 1)
            timehorizon = len(self.surplus) - 365*24
            
        # Extend EV Connectivity Data #
            if(self.Mult_aggEV.n_assets > 0):
                self.Mult_aggEV.construct_connectivity_timeseries(start_EV,dt.datetime(end_EV.year-1,1,1,0),includeleapdays=False)
            
            
        else: #if not repeatedly reforming the model then no need to delete leap days
            timehorizon = len(self.surplus)
        # Extend EV Connectivity Data #
            if(self.Mult_aggEV.n_assets > 0):
                self.Mult_aggEV.construct_connectivity_timeseries(start_EV,end_EV)
                       
        model.TimeIndex=range(timehorizon)  #NB, index for time of day, starts at 0    
        


        
        
    # Declare decision variables #
        
        #General
        model.Pfos=pyo.Var(model.TimeIndex, within = pyo.NonNegativeReals) #power from fossil fuels needed at time t
        model.Shed=pyo.Var(model.TimeIndex, within = pyo.NonNegativeReals) #amount of surplus shed
        
        #Storage
        model.BuiltCapacity=pyo.Var(model.StorageIndex, within = pyo.NonNegativeReals) #chosen capacity to build of each storage (MWh)
        model.C=pyo.Var(model.StorageIndex, model.TimeIndex, within = pyo.NonNegativeReals) #charging rate of each storage asset (MW) (from battery side, so energy into battety)
        model.D=pyo.Var(model.StorageIndex, model.TimeIndex, within = pyo.NonNegativeReals) #discharging rate of each storage asset (MW) (from battery side, so energy out of battery, this will be larger than than the energy into the grid)
        model.SOC=pyo.Var(model.StorageIndex, model.TimeIndex, within = pyo.NonNegativeReals) #state of charge of the asset at end of timestep t (MWh)    
        
        #EVs
        model.EV_TypeBuiltCapacity=pyo.Var(model.FleetIndex, model.ChargeType, within = pyo.NonNegativeReals) #number of built chargers for each type, V2G,Smart,Dumb
        model.EV_C=pyo.Var(model.FleetIndex, model.TimeIndex, model.ChargeType, within = pyo.NonNegativeReals) #hourly charging rate of each subset of charger types within each fleet(MW) (battery side, so energy into batteries)
        model.EV_D=pyo.Var(model.FleetIndex, model.TimeIndex, model.ChargeType, within = pyo.NonNegativeReals) #hourly discharging rate of each subset of charger types within each fleet(MW) (battery side, so energy out of battery)
        model.EV_SOC=pyo.Var(model.FleetIndex, model.TimeIndex, model.ChargeType, within = pyo.NonNegativeReals) #state of charge of the asset at end of timestep t (MWh)    
        
        #Generators
        model.GenCapacity = pyo.Var(model.GenIndex,within = pyo.NonNegativeReals) #the built capacity (MW) of each generator type
        
             
    # Declare Parameters #
        
        #Parameters save time on setting up the model for repeated runs, as they can be adjusted without having to rebuild the entire model
        #Declaring Parameters is fiddly, needs to be done with an enumerated dictionary of tuples as below
        
        #if I am running repeated simulations, then don't initialize parameters
        if(self.SizingThenOperation):
            model.Demand = pyo.Param(model.TimeIndex, within = pyo.Reals, mutable = True)    
            model.NormalisedGen = pyo.Param(model.GenIndex, model.TimeIndex, within = pyo.NonNegativeReals, mutable = True)
        else:                
            model.Demand = pyo.Param(model.TimeIndex, initialize = dict(enumerate(self.surplus)), within = pyo.Reals, mutable = True)                    
            ref = {} #this gives the normalised power output of generator g at time t (multiplied by the built capacity to give the MW)
            for g in model.GenIndex:
                for t in model.TimeIndex:
                    ref[g,model.TimeIndex[t]] = self.gen_list[g].power_out[t]/max(self.gen_list[g].power_out)
            model.NormalisedGen = pyo.Param(model.GenIndex, model.TimeIndex, within = pyo.NonNegativeReals, initialize = ref)
        
        #Declare the Plugin Data as parameters
        ref_Nin={}
        ref_Nout={}
        ref_N={}
        for k in model.FleetIndex:
            for t in model.TimeIndex:
                ref_Nin[model.TimeIndex[t],k] = self.Mult_aggEV.assets[k].Nin[t]
                ref_Nout[model.TimeIndex[t],k] = self.Mult_aggEV.assets[k].Nout[t]
                ref_N[model.TimeIndex[t],k] = self.Mult_aggEV.assets[k].N[t]
        model.Nin = pyo.Param(model.TimeIndex, model.FleetIndex, within = pyo.NonNegativeReals, mutable = True,initialize = ref_Nin)
        model.Nout = pyo.Param(model.TimeIndex, model.FleetIndex, within = pyo.NonNegativeReals, mutable = True,initialize = ref_Nout)
        model.N = pyo.Param(model.TimeIndex, model.FleetIndex, within = pyo.NonNegativeReals, mutable = True,initialize = ref_N)
          

        
        #the limit on energy from fossil fuels is specified as a % of the -ve given surplus (i.e. demand), 
        #this can be updated for repeated sims or not!
        limit = -self.fossilLimit * sum(self.surplus * (self.surplus < 0))
        model.foss_lim_param = pyo.Param(within = pyo.NonNegativeReals,mutable = True,initialize = limit)
        
        model.fossilfuelpenalty = pyo.Param(within = pyo.NonNegativeReals, mutable = True,initialize = fossilfuelpenalty)
        
        #### Limits ####
            # Generation Limits #
            #these two parameters limit the upper and lower bounds of the generation capacities
        gen_limits_lower = []
        gen_limits_upper = []
        for g in model.GenIndex:
            gen_limits_lower.append(self.gen_list[g].limits[0])
            gen_limits_upper.append(self.gen_list[g].limits[1])
        model.Gen_Limit_Param_Lower = pyo.Param(model.GenIndex, within = pyo.NonNegativeReals,mutable = True,initialize = dict(enumerate(gen_limits_lower)))
        model.Gen_Limit_Param_Upper = pyo.Param(model.GenIndex, within = pyo.NonNegativeReals,mutable = True,initialize = dict(enumerate(gen_limits_upper)))
        
            # Storage Limits #
        stor_limits_lower = []
        stor_limits_upper = []
        for i in model.StorageIndex:
            stor_limits_lower.append(self.Mult_Stor.assets[i].limits[0])
            stor_limits_upper.append(self.Mult_Stor.assets[i].limits[1])
        model.Stor_Limit_Param_Lower = pyo.Param(model.StorageIndex, within = pyo.NonNegativeReals,mutable = True,initialize = dict(enumerate(stor_limits_lower)))
        model.Stor_Limit_Param_Upper = pyo.Param(model.StorageIndex, within = pyo.NonNegativeReals,mutable = True,initialize = dict(enumerate(stor_limits_upper)))
    
            # Charger Type Limits #
        V2G_limits_lower = []
        V2G_limits_upper = []
        Uni_limits_lower = []
        Uni_limits_upper = []
        for k in model.FleetIndex:
            for b in model.ChargeType:
                if(b==0):
                    V2G_limits_lower.append(self.Mult_aggEV.assets[k].limits[0])
                    V2G_limits_upper.append(self.Mult_aggEV.assets[k].limits[1])
                if(b==1):
                    Uni_limits_lower.append(self.Mult_aggEV.assets[k].limits[2])
                    Uni_limits_upper.append(self.Mult_aggEV.assets[k].limits[3])
        model.V2G_Limit_Param_Lower = pyo.Param(model.FleetIndex,within = pyo.NonNegativeReals,mutable = True,initialize = dict(enumerate(V2G_limits_lower)))
        model.V2G_Limit_Param_Upper = pyo.Param(model.FleetIndex,within = pyo.NonNegativeReals,mutable = True,initialize = dict(enumerate(V2G_limits_upper)))
        model.Uni_Limit_Param_Lower = pyo.Param(model.FleetIndex,within = pyo.NonNegativeReals,mutable = True,initialize = dict(enumerate(Uni_limits_lower)))
        model.Uni_Limit_Param_Upper = pyo.Param(model.FleetIndex,within = pyo.NonNegativeReals,mutable = True,initialize = dict(enumerate(Uni_limits_upper))) 

        #### COST PARAMETERS ####
        # ChargerCosts
        ref = {}
        for k in model.FleetIndex:
            for b in model.ChargeType:
                ref[k,b] = self.Mult_aggEV.assets[k].chargercost[b]
        model.chargercost = pyo.Param(model.FleetIndex,model.ChargeType,within = pyo.NonNegativeReals,mutable = True,initialize = ref)
        
        # GeneratorCosts
        ref_g = {}
        for g in model.GenIndex:
            ref_g[g,0] = self.gen_list[g].fixed_cost
            ref_g[g,1] = self.gen_list[g].variable_cost
        model.GenCosts = pyo.Param(model.GenIndex, range(2),within = pyo.NonNegativeReals,mutable = True,initialize = ref_g)
        
        # Storage Costs
        ref_s = {}
        for s in model.StorageIndex:
            ref_s[s,0] = self.Mult_Stor.assets[s].fixed_cost
            ref_s[s,1] = self.Mult_Stor.assets[s].variable_cost
        model.StorCosts = pyo.Param(model.StorageIndex, range(2),within = pyo.NonNegativeReals,mutable = True,initialize = ref_s)
            
            
    # Declare constraints #
        
        #General Constraints
        #Power Balance
        model.PowerBalance = pyo.ConstraintList()
        for t in range(timehorizon):                                                    #this is a normalised power output between 0-1
            #model.PowerBalance.add(surplus[t] + sum(model.GenCapacity[g]*(gen_list[g].power_out[t]/max(gen_list[g].power_out)) for g in model.GenIndex)- model.Shed[t] + sum(model.D[i,t] * Mult_Stor.assets[i].eff_out/100.0 - model.C[i,t] * 100.0/Mult_Stor.assets[i].eff_in for i in model.StorageIndex) + model.Pfos[t] +  sum(model.EV_D[k,t,0] * Mult_aggEV.assets[k].eff_out/100  - sum( model.EV_C[k,t,b] * 100/Mult_aggEV.assets[k].eff_in for b in model.ChargeType)  for k in model.FleetIndex)== 0)
            model.PowerBalance.add(model.Demand[t] + sum(model.GenCapacity[g]*model.NormalisedGen[g,t] for g in model.GenIndex)- model.Shed[t] + sum(model.D[i,t] * self.Mult_Stor.assets[i].eff_out/100.0 - model.C[i,t] * 100.0/self.Mult_Stor.assets[i].eff_in for i in model.StorageIndex) + model.Pfos[t] +  sum(model.EV_D[k,t,0] * self.Mult_aggEV.assets[k].eff_out/100  - sum( model.EV_C[k,t,b] * 100/self.Mult_aggEV.assets[k].eff_in for b in model.ChargeType)  for k in model.FleetIndex)== 0)
            
        #Specified Amount of Fossil Fuel Input
        model.FossilLimit = pyo.ConstraintList()
        model.FossilLimit.add(sum(model.Pfos[t] for t in model.TimeIndex) <= model.foss_lim_param)
        
    
        #Generator Constraints
        model.genlimits = pyo.ConstraintList()
        for g in model.GenIndex:
            model.genlimits.add(model.GenCapacity[g] >= model.Gen_Limit_Param_Lower[g])
            model.genlimits.add(model.GenCapacity[g] <= model.Gen_Limit_Param_Upper[g])
    
        #Storage Constraints
        model.maxSOC = pyo.ConstraintList()
        model.battery_charge_level = pyo.ConstraintList()
        model.maxD = pyo.ConstraintList()
        model.maxC = pyo.ConstraintList()
        model.storagelimits = pyo.ConstraintList()
        for i in range(self.Mult_Stor.n_assets):
            model.storagelimits.add(model.BuiltCapacity[i] >=  model.Stor_Limit_Param_Lower[i])
            model.storagelimits.add(model.BuiltCapacity[i] <=  model.Stor_Limit_Param_Upper[i])
                
            for t in range(timehorizon):
                model.maxSOC.add(model.SOC[i,t] <= model.BuiltCapacity[i]) #SOC less than maximum
                model.maxD.add(model.D[i,t] * self.Mult_Stor.assets[i].eff_out/100.0 <= model.BuiltCapacity[i] * self.Mult_Stor.assets[i].max_d_rate/100)
                model.maxC.add(model.C[i,t] * 100.0/self.Mult_Stor.assets[i].eff_in <= model.BuiltCapacity[i] * self.Mult_Stor.assets[i].max_c_rate/100)               

                if t == 0:
                    if(InitialSOC[0] >= 0.0):                        
                        model.battery_charge_level.add(model.SOC[i,t]== InitialSOC[i]*model.BuiltCapacity[i] + model.C[i,t] - model.D[i,t]) #batteries start at specified charge level
                    if(StartSOCEqualsEndSOC):
                        model.battery_charge_level.add(model.SOC[i,t]== model.SOC[i,timehorizon-1] + model.C[i,t] - model.D[i,t]) #batteries start and end on same Charge
                else:
                    model.battery_charge_level.add(model.SOC[i,t]== model.SOC[i,t-1]*(1-(self.Mult_Stor.assets[i].self_dis/(100*30*24))) + model.C[i,t] - model.D[i,t])
            
        
        #EV Constraints
        model.EV_maxSOC = pyo.ConstraintList()
        model.EV_minSOC = pyo.ConstraintList()
        model.EV_battery_charge_level = pyo.ConstraintList()
        model.EV_maxD = pyo.ConstraintList()
        model.EV_maxC = pyo.ConstraintList()
        model.Built_Asset_Sum = pyo.ConstraintList()
        model.ChargerTypeLimits = pyo.ConstraintList() #allows the user to specify the max number of a certain charger type
        
        for k in range(self.Mult_aggEV.n_assets):
            #constraint to make sure all the different built capacities add to one
            model.Built_Asset_Sum.add(sum(model.EV_TypeBuiltCapacity[k,b] for b in model.ChargeType) == self.Mult_aggEV.assets[k].number)
            
            for b in model.ChargeType:
                if(b==0):
                    model.ChargerTypeLimits.add(model.EV_TypeBuiltCapacity[k,b] >= model.V2G_Limit_Param_Lower[k])
                    model.ChargerTypeLimits.add(model.EV_TypeBuiltCapacity[k,b] <= model.V2G_Limit_Param_Upper[k])
                elif(b==1):
                    model.ChargerTypeLimits.add(model.EV_TypeBuiltCapacity[k,b] >= model.Uni_Limit_Param_Lower[k])
                    model.ChargerTypeLimits.add(model.EV_TypeBuiltCapacity[k,b] <= model.Uni_Limit_Param_Upper[k])
                    
                for t in range(timehorizon):
                    model.EV_maxSOC.add(model.EV_SOC[k,t,b] <= model.EV_TypeBuiltCapacity[k,b]*model.N[t,k]*self.Mult_aggEV.assets[k].max_SOC/1000) #constraint to limit the max SOC
                    
                    if(t>0): #as the EV_SOC decision variable refers to the end of the timestep!
                        model.EV_minSOC.add(model.EV_SOC[k,t-1,b] >= model.EV_TypeBuiltCapacity[k,b]*model.Nout[t,k]*self.Mult_aggEV.assets[k].Eout/1000) #constraint to make sure that there is always enough charge for the EVs to plug out
              
                    model.EV_maxC.add(model.EV_C[k,t,b] * 100/self.Mult_aggEV.assets[k].eff_in <= model.EV_TypeBuiltCapacity[k,b]*model.N[t,k]*self.Mult_aggEV.assets[k].max_c_rate/1000)
                    
                    if(b==0):
                        #V2G specific Constraints
                        model.EV_maxD.add(model.EV_D[k,t,b] * self.Mult_aggEV.assets[k].eff_out/100 <= model.EV_TypeBuiltCapacity[k,b]*model.N[t,k]*self.Mult_aggEV.assets[k].max_d_rate/1000)
                        
                        if t == 0:
                            if(StartSOCEqualsEndSOC):
                                model.EV_battery_charge_level.add(model.EV_SOC[k,t,b] == model.EV_SOC[k,timehorizon-1,b] + model.EV_C[k,t,b] - model.EV_D[k,t,b] - model.EV_TypeBuiltCapacity[k,b]*model.Nout[t,k]*self.Mult_aggEV.assets[k].Eout/1000 + model.EV_TypeBuiltCapacity[k,b]*model.Nin[t,k]*self.Mult_aggEV.assets[k].Ein/1000)                         
                            if(InitialSOC[0] >= 0.0): 
                                model.EV_battery_charge_level.add(model.EV_SOC[k,t,b] == InitialSOC[self.Mult_Stor.n_assets + 2*k + b] * model.EV_TypeBuiltCapacity[k,b]*model.N[t,k]*self.Mult_aggEV.assets[k].max_SOC/1000 + model.EV_C[k,t,b] - model.EV_D[k,t,b] - model.EV_TypeBuiltCapacity[k,b]*model.Nout[t,k]*self.Mult_aggEV.assets[k].Eout/1000 + model.EV_TypeBuiltCapacity[k,b]*model.Nin[t,k]*self.Mult_aggEV.assets[k].Ein/1000) 
                        
                        else:
                            model.EV_battery_charge_level.add(model.EV_SOC[k,t,b] == model.EV_SOC[k,t-1,b] + model.EV_C[k,t,b] - model.EV_D[k,t,b] - model.EV_TypeBuiltCapacity[k,b]*model.Nout[t,k]*self.Mult_aggEV.assets[k].Eout/1000 + model.EV_TypeBuiltCapacity[k,b]*model.Nin[t,k]*self.Mult_aggEV.assets[k].Ein/1000) 

                    elif b==1:
                        #Smart Unidirectional
                        if t == 0:
                            if(StartSOCEqualsEndSOC):
                                model.EV_battery_charge_level.add(model.EV_SOC[k,t,b] == model.EV_SOC[k,timehorizon-1,b] + model.EV_C[k,t,b] - model.EV_TypeBuiltCapacity[k,b]*model.Nout[t,k]*self.Mult_aggEV.assets[k].Eout/1000 + model.EV_TypeBuiltCapacity[k,b]*model.Nin[t,k]*self.Mult_aggEV.assets[k].Ein/1000)                         
                            if(InitialSOC[0] >= 0.0): 
                                model.EV_battery_charge_level.add(model.EV_SOC[k,t,b] == InitialSOC[self.Mult_Stor.n_assets + 2*k + b] * model.EV_TypeBuiltCapacity[k,b]*model.N[t,k]*self.Mult_aggEV.assets[k].max_SOC/1000 + model.EV_C[k,t,b] - model.EV_TypeBuiltCapacity[k,b]*model.Nout[t,k]*self.Mult_aggEV.assets[k].Eout/1000 + model.EV_TypeBuiltCapacity[k,b]*model.Nin[t,k]*self.Mult_aggEV.assets[k].Ein/1000) 
                        
                        else:
                            model.EV_battery_charge_level.add(model.EV_SOC[k,t,b] == model.EV_SOC[k,t-1,b] + model.EV_C[k,t,b] - model.EV_TypeBuiltCapacity[k,b]*model.Nout[t,k]*self.Mult_aggEV.assets[k].Eout/1000 + model.EV_TypeBuiltCapacity[k,b]*model.Nin[t,k]*self.Mult_aggEV.assets[k].Ein/1000) 

    # Declare objective function #
        model.obj=pyo.Objective(                                                                                                            #adding this small 0.05 stops the model from charging and discharging simultaneously unecessarily                                                                                                                                      this penalises fossil fuel use to encourage healthier charging behaviour                                                                                                                                                                                                          this penalty is very small and encourages more 'normal' charging behaviour                         penalty to encourage healthier EV charging
        expr=sum((timehorizon/(365*24))*model.GenCapacity[g]*model.GenCosts[g,0] + model.GenCapacity[g]*model.GenCosts[g,1]*sum(model.NormalisedGen[g,t] for t in model.TimeIndex) for g in model.GenIndex) + sum((timehorizon/(365*24))*model.StorCosts[i,0] * model.BuiltCapacity[i] for i in model.StorageIndex) + sum( sum((model.StorCosts[i,1]+0.05) * model.D[i,t] for t in model.TimeIndex) for i in model.StorageIndex) + sum(sum((timehorizon/(365*24)) * model.chargercost[k,b] * model.EV_TypeBuiltCapacity[k,b] for b in model.ChargeType)for k in model.FleetIndex) + sum(model.fossilfuelpenalty * model.Pfos[t] for t in model.TimeIndex) + sum( sum((0.05) * model.EV_D[k,t,0] for t in model.TimeIndex) for k in model.FleetIndex),   
        sense=pyo.minimize)
        
        self.model = model
        
        end = time.time()
        print('Model Formation Complete after: ',int(end - start), 's')

    def Run_Sizing(self, solver = 'glpk'):
        '''
        == description ==
        Solve the linear programme specified by Form_Model. Results recorded by updating df_capital. The operational timeseries
        for generators and storage are saved to their respective objects.
        
        == parameters ==       
       
        
        == returns ==

        
        '''
        if self.SizingThenOperation:
            print('Sizing  Requires Model Constructed with SizingThenOperation = False')
            return        
        
    # Create a solver #
        opt = pyo.SolverFactory(solver)

        
    # Solve #
        print('Finding Optimal System ...')
        start = time.time()
        
        opt.solve(self.model)  
        
        end = time.time()
        print('Solved after: ',int(end - start),'s')
        
    # Update System Characteristics #
        
        x = opt_results_to_df(self.model)
        self.df_capital = x[0]
        self.df_costs = x[1]
        store_optimisation_results(self.model,self.Mult_aggEV,self.Mult_Stor,self.gen_list)
        

        
    def Run_Sizing_Then_Op(self,SimYears,c_order=[],d_order=[], V2G_discharge_threshold = 0.0, solver = 'glpk'):
        '''
        == description ==
        Runs repeated simulations, the system will be sized based on set of years, then simulated both causally and non-causally
        on the remaining year. During simulations the Agg EVs and Storage devices will start at the average SOC for 00:00 1st Jan 
        during the sizing optimisation. SimYears dictates which years will be simulated after being excluded from the system sizing.
        
        == parameters ==
        SimYears: (array<int>): list of the years that will be excluded from the optimisation and operated causally
        c_order/d_order: The charge/discharge order during the causal system operation. Bare in mind, the order of 'storage' during operation corresponding to the index in c_order/d_order is: [MultStor.assets[0],MultStor.assets[1]...,V2G Mult_aggEV.assets[0],unidirectional Mult_aggEV.assets[0],V2G Mult_aggEV.assets[1]...  ]
                        Remember, need two entries for every fleet because the V2G and Smart virtual batteries are distinct. 
        V2G_discharge_threshold: (float), the SOC threshold (KWh) of the EV batteries, below which V2G will not discharge during causal system operation
       
        
        == returns ==
        df: (DataFrame): this contains the reliability of the causal sim runs and the % of EV driving energy met by renewables!
        
        '''
        self.SimYears= SimYears
        
        if len(c_order) == 0:
            c_order = range(self.Mult_Stor.n_assets + 2 * self.Mult_aggEV.n_assets)
        
        if len(d_order) == 0:
            d_order = range(self.Mult_Stor.n_assets + 2 * self.Mult_aggEV.n_assets)
        
    # Create a solver #
        opt = pyo.SolverFactory(solver)

        
        if not(self.SizingThenOperation):
            print('Sizing then Op Requires Model Constructed with SizingThenOperation = True')
            return
        
        for y in range(len(self.SimYears)):
            if(self.YearRange[0] > self.SimYears[y] or self.YearRange[1] < self.SimYears[y] ):
                print('The specified causally operated years must be within the year range')
                return
    # Create Output Recording objects #
        
        Causal_Reliability = [] #The % of demand met by renewable power
        Causal_EV_Reliability = [] #The % of driving Energy Provided by Renewable energy (if no chargers of a certain type this is set to 100%)
        for x in range(2*self.Mult_aggEV.n_assets):
            Causal_EV_Reliability.append([])
            
        Non_Causal_Reliability = [] #The % of demand met by renewable power
        
        
        
    # Run Sizing Operation #
    
        #rerun connectivity timeseries so it is llong enouh for the simulations
        if(len(SimYears) > 0):
            for y in range(len(SimYears)):
                self.Mult_aggEV.construct_connectivity_timeseries(dt.datetime(SimYears[0],1,1,0),dt.datetime(SimYears[-1]+1,1,1,0),includeleapdays=False)
                #these value are uses within the sizing operation ot find the ideal system
                surplus_instance=np.zeros([365*24*(self.YearRange[1]-self.YearRange[0])])
                gen_power_instance = np.zeros([len(self.gen_list),365*24*(self.YearRange[1]-self.YearRange[0])])
            #find the plugin schedules for the EV fleets
                N_instance=np.zeros([self.Mult_aggEV.n_assets,365*24*(self.YearRange[1]-self.YearRange[0])])
                Nin_instance=np.zeros([self.Mult_aggEV.n_assets,365*24*(self.YearRange[1]-self.YearRange[0])])
                Nout_instance=np.zeros([self.Mult_aggEV.n_assets,365*24*(self.YearRange[1]-self.YearRange[0])])
                
                #these values are for simulating the operation of the previously sized system
                surplus_simulation = np.zeros([365*24])
                gen_power_simulation = np.zeros([len(self.gen_list),365*24])
            #Plugin Schedules
                N_simulation = np.zeros([self.Mult_aggEV.n_assets,365*24])
                Nin_simulation = np.zeros([self.Mult_aggEV.n_assets,365*24])
                Nout_simulation = np.zeros([self.Mult_aggEV.n_assets,365*24])
                
                #remove simulation year from the parameter input data
                years_encompassed = range(self.YearRange[0],self.YearRange[1]+1) 
                counter = 0
                for x in range(self.YearRange[1]+1-self.YearRange[0]):
                    if(not(years_encompassed[x] ==SimYears[y])):
                        surplus_instance[counter*365*24:(counter+1)*365*24] = self.surplus[x*365*24:(x+1)*365*24]
                        for g in self.model.GenIndex:
                            gen_power_instance[g,counter*365*24:(counter+1)*365*24] = np.asarray(self.gen_list[g].power_out[x*365*24:(x+1)*365*24])/max(self.gen_list[g].power_out)
                        
                        for k in range(self.Mult_aggEV.n_assets):
                            N_instance[k,counter*365*24:(counter+1)*365*24] = self.Mult_aggEV.assets[k].N[x*365*24:(x+1)*365*24]
                            Nin_instance[k,counter*365*24:(counter+1)*365*24] = self.Mult_aggEV.assets[k].Nin[x*365*24:(x+1)*365*24]
                            Nout_instance[k,counter*365*24:(counter+1)*365*24] = self.Mult_aggEV.assets[k].Nout[x*365*24:(x+1)*365*24]
                            
                        counter = counter + 1
                    else:
                        surplus_simulation = self.surplus[x*365*24:(x+1)*365*24]
                        for g in self.model.GenIndex:
                            gen_power_simulation[g,:] = np.asarray(self.gen_list[g].power_out[x*365*24:(x+1)*365*24])/max(self.gen_list[g].power_out)
                        
                        for k in range(self.Mult_aggEV.n_assets):
                            N_simulation[k,:] = self.Mult_aggEV.assets[k].N[x*365*24:(x+1)*365*24]
                            Nin_simulation[k,:] = self.Mult_aggEV.assets[k].Nin[x*365*24:(x+1)*365*24]
                            Nout_simulation[k,:] = self.Mult_aggEV.assets[k].Nout[x*365*24:(x+1)*365*24]
                        
                #use instance surplus to limit fossil fuel power
                limit = -self.fossilLimit * sum(surplus_instance * (surplus_instance < 0))
                self.model.foss_lim_param = limit

                
                #fill in the correct parameters before running the model
                for t in self.model.TimeIndex:
                    self.model.Demand[t] = surplus_instance[t]
                    for g in self.model.GenIndex:
                        #print(gen_power_instance[g][t])
                        self.model.NormalisedGen[g,t] = gen_power_instance[g,t]
                    for k in self.model.FleetIndex:
                        self.model.Nin[t,k] = Nin_instance[k,t]
                        self.model.Nout[t,k] = Nout_instance[k,t]
                        self.model.N[t,k] = N_instance[k,t]
                 
                
                
            # Solve #
                print('Finding Optimal System for ', self.YearRange[0], '-', self.YearRange[1], ', excluding', SimYears[y],'...')
                start = time.time()
                
                opt.solve(self.model)  
                
                end = time.time()
                print('Solved after: ',int(end - start), 's. Now causally simulating operation on this system for ', SimYears[y], ' ...')
                
            # Update System Characteristics #
                    
                for i in range(self.Mult_Stor.n_assets):
                    self.Mult_Stor.assets[i].set_capacity(pyo.value(self.model.BuiltCapacity[i]))
                    
                for k in range(self.Mult_aggEV.n_assets):
                    self.Mult_aggEV.assets[k].reset()                
                    for b in range(2):
                        self.Mult_aggEV.assets[k].chargertype[b] = pyo.value(self.model.EV_TypeBuiltCapacity[k,b])/self.Mult_aggEV.assets[k].number
            
            # Find Average SOC on 00:00 Jan for Initial_SOC input for simulations #
                #find number of years within sizing operation
                yrs = int(len(surplus_instance)%(365*24 - 1))
                
                Av_Stor_SOC = np.zeros([self.Mult_Stor.n_assets,yrs])
                Av_EV_SOC = np.zeros([self.Mult_aggEV.n_assets * 2,yrs]) #Fleet0 V2G, Fleet0 Smart, Fleet1 V2g...
                
                yr = 0
                #add the SOC from each 1st Jan to storage variables
                for t in range(len(surplus_instance)):
                   if self.date_range[t].month == 1 and self.date_range[t].day == 1 and self.date_range[t].hour == 0:
                        for i in range(self.Mult_Stor.n_assets):
                            Av_Stor_SOC[i,yr] = pyo.value(self.model.SOC[i,t])
                        for k in range(self.Mult_aggEV.n_assets):
                            for b in range(2):
                                Av_EV_SOC[k*2+b,yr] = pyo.value(self.model.EV_SOC[k,t,b])
                                  
                        yr = yr + 1

                initialSOC = []
                for i in range(self.Mult_Stor.n_assets):
                    initialSOC.append((sum(Av_Stor_SOC[i,:])/yrs)/self.Mult_Stor.assets[i].capacity)
                
                for k in range(self.Mult_aggEV.n_assets):
                    for b in range(2):
                        if self.Mult_aggEV.assets[k].chargertype[b] > 0:
                            initialSOC.append((sum(Av_EV_SOC[2*k+b,:])/yrs)/(self.Mult_aggEV.assets[k].chargertype[b]*self.Mult_aggEV.assets[k].number*self.Mult_aggEV.assets[k].max_SOC/1000))
                        else:
                            initialSOC.append(0)

                
            # Simulate System Operation Causally #
                
                start = time.time()
                #First Method is ordered Charging
                #Find the input surplus:
                demand = -surplus_simulation 
                power = sum(pyo.value(self.model.GenCapacity[g])*gen_power_simulation[g,:] for g in self.model.GenIndex)
                #Simulate the system Causally
                x1 = self.Mult_Stor.causal_system_operation(demand,power,c_order,d_order,self.Mult_aggEV,start = dt.datetime(SimYears[y],1,1,0),end = dt.datetime(SimYears[y]+1,1,1,0),IncludeEVLeapDay=False,plot_timeseries = True,V2G_discharge_threshold = V2G_discharge_threshold,initial_SOC = initialSOC)
                end = time.time()
                print('Simulation Complete: ',int(end - start), 's. Now Non-Causally Simulating System Operation:')
                             
            # Simulate System Operation Non-Causally #
                #To save time only want to form model on 1st non-causal sim, after that just update the surplus, stor_capacity and EV Charger type parameters to save time!           
                if y ==0:
                    Non_Causal_Reliability = np.zeros([len(SimYears)])
                    sim_Mult_Stor = self.Mult_Stor
                    sim_Mult_aggEV = self.Mult_aggEV
                    Non_Causal_Reliability[y] = sim_Mult_Stor.non_causal_system_operation(demand, power, sim_Mult_aggEV, start = dt.datetime(SimYears[y],1,1,0),end = dt.datetime(SimYears[y]+1,1,1,0), includeleapdaysEVs = False,
                                  plot_timeseries = True, InitialSOC = initialSOC, form_model = True)
                else:
                    #update the storage asset sizes to reflect the new system
                    for i in range(sim_Mult_Stor.n_assets):
                        sim_Mult_Stor.non_causal_linprog.model.Stor_Limit_Param_Lower[i] = self.Mult_Stor.assets[i].capacity
                        sim_Mult_Stor.non_causal_linprog.model.Stor_Limit_Param_Upper[i] = self.Mult_Stor.assets[i].capacity
                        
                    #update the EV Charger Types to reflect the new system
                    for k in range(sim_Mult_aggEV.n_assets):
                        #update with correct plugin characteristics
                        for t in sim_Mult_Stor.non_causal_linprog.model.TimeIndex:                            
                            sim_Mult_Stor.non_causal_linprog.model.Nin[t,k] = Nin_instance[k,t]
                            sim_Mult_Stor.non_causal_linprog.model.Nout[t,k] = Nout_instance[k,t]
                            sim_Mult_Stor.non_causal_linprog.model.N[t,k] = N_instance[k,t]
                        for b in range(2):
                            if b == 0 :
                                sim_Mult_Stor.non_causal_linprog.model.V2G_Limit_Param_Lower[k] = self.Mult_aggEV.assets[k].chargertype[b] * self.Mult_aggEV.assets[k].number
                                sim_Mult_Stor.non_causal_linprog.model.V2G_Limit_Param_Upper[k] = self.Mult_aggEV.assets[k].chargertype[b] * self.Mult_aggEV.assets[k].number
                            if b == 1 :
                                lim = max(0,self.Mult_aggEV.assets[k].chargertype[b] * self.Mult_aggEV.assets[k].number)
                                sim_Mult_Stor.non_causal_linprog.model.Uni_Limit_Param_Lower[k] = lim
                                sim_Mult_Stor.non_causal_linprog.model.Uni_Limit_Param_Upper[k] = lim
                    
                    #demand and power parameters updated within the model
                    Non_Causal_Reliability[y] = sim_Mult_Stor.non_causal_system_operation(demand, power, sim_Mult_aggEV, start = dt.datetime(SimYears[y],1,1,0),end = dt.datetime(SimYears[y]+1,1,1,0), includeleapdaysEVs = False,
                                  plot_timeseries = True, InitialSOC = initialSOC, form_model = False)
                    

                
                if y == 0:
                    x = opt_results_to_df(self.model)
                    self.df_capital = x[0]
                else:
                    x = opt_results_to_df(self.model)
                    self.df_capital = self.df_capital.append(x[0])
                    
                if y == 0:
                    df = x1
                else:
                    df = df.append(x1,ignore_index=True) 
                    
            df['Non Causal Reliability'] = Non_Causal_Reliability
            df['Simulation Year'] = SimYears
            self.df_capital['Simulation Year'] = SimYears
                
            return df
        
        
    def PlotSurplus(self,start = 0, end = -1): 
        '''
        == description ==
        Plots the surplus before and after the EV/battery charging actions. Must be run after a sizing run!
        '''
        
        store_optimisation_results(self.model, self.Mult_aggEV, self.Mult_Stor, self.gen_list)
        
        if end < 0:
            timehorizon = max(self.model.TimeIndex)-start
        else:
            timehorizon = end-start
        
        
    
        surplus = np.zeros([timehorizon])    
        name = '. '
        for g in range(len(self.gen_list)):
            surplus += np.asarray(self.gen_list[g].power_out[start:(start+timehorizon)])
            #print('L1', surplus[0:24])
            name += self.gen_list[g].name + ' ' +  str(int(pyo.value(self.model.GenCapacity[g])/1000)) + 'GW, '
        
        surplus += self.surplus[start:start+timehorizon]
        #print('L2', surplus[0:24])
        #print('self.surp', self.surplus[0:24])
        
        plt.rc('font', size=12)            
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(range(timehorizon), surplus, color='k', label='Generation-Demand')
        
        #work out the surplus post charging
        surplus_pc = np.zeros([timehorizon])
        model = self.model
        for t in range(timehorizon):
            #surplus_pc[t] = -pyo.value(self.model.Pfos[t]) + pyo.value(self.model.Shed[t])
            #surplus_pc[t] = pyo.value(model.Demand[t]) + sum(pyo.value(model.GenCapacity[g])*pyo.value(model.NormalisedGen[g,t]) for g in model.GenIndex)- pyo.value(model.Shed[t]) + sum(pyo.value(model.D[i,t]) * self.Mult_Stor.assets[i].eff_out/100.0 -pyo.value(model.C[i,t]) * 100.0/self.Mult_Stor.assets[i].eff_in for i in model.StorageIndex) + pyo.value(model.Pfos[t])
            surplus_pc[t] = pyo.value(model.Demand[start+t]) + sum(pyo.value(model.GenCapacity[g])*pyo.value(model.NormalisedGen[g,start+t]) for g in model.GenIndex) + sum(pyo.value(model.D[i,start+t]) * self.Mult_Stor.assets[i].eff_out/100.0 -pyo.value(model.C[i,start+t]) * 100.0/self.Mult_Stor.assets[i].eff_in for i in model.StorageIndex) 
            
        ax.plot(range(timehorizon), surplus_pc, color='b', label='Generation-Demand after Charging')
        ax.set_title('Surplus Timeseries'+name)
        ax.legend(loc='upper left')
        

        
        
        
        
        
        
        
        
        
        
        