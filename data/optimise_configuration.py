import numpy as np
#optimisation high level language, help found at https://www.ima.umn.edu/materials/2017-2018.2/W8.21-25.17/26326/3_PyomoFundamentals.pdf
#low level algorithmic solve performed by solver
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import time
import datetime as dt



def optimise_configuration(surplus,fossilLimit,Mult_Stor,Mult_aggEV,gen_list=[],SimYears=[],YearRange=[]):
        '''
        == description ==
        For a given surplus, returns the cost optimal storage mix to meet the specified reliability. Charge order not relevant here.

        == parameters ==
        surplus: (Array<float>) the surplus generation to be smoothed in MW, must be an np array!!
        fossilLimit: (float) max acceptable amount of fossil fuel generated energy (MWh)
        StorAssets: (mult_stor)
        gen_list: (array<generation>): list of the potential renewable generators to build
        SimYears: (array<int>): list of the years that will be excluded from the optimisation and operated causally
        YearRange: (array<int>): [MinYear,MaxYear]
        == returns ==
        '''
        
        #print('Remember, surplus timeseries must have hourly entries for this to work. ')
        print('Forming Optimisation Model...')
        start = time.time()
            
            
        # Create a solver #
        opt = pyo.SolverFactory('mosek')
        
        # Create model #
        model=pyo.ConcreteModel()  #create the model object, concrete because parameters are known
        #model=pyo.AbstractModel()
        
        # Declare Indexs #
        
        model.StorageIndex=range(Mult_Stor.n_assets) #Index over the storage assets
        model.FleetIndex=range(Mult_aggEV.n_assets) #Index over Fleets
        model.ChargeType = range(2) #Index over Type of Charger, 0: V2G, 1: Smart, 2: Unmanaged
        model.GenIndex = range(len(gen_list)) #Index over possible generators
            
        
        #remove leap days, necessary for re-running model without re-defining it, as need the same number of decision variables no matter the years
        if(len(SimYears) > 0):
            #check for full years as input
            for g in model.GenIndex:
                if(not(len(gen_list[g].months)==12)):
                    print('To use for system sizing then subsequent operation, must use complete years not months.')
                    return
            #check that the YearRange Has Been Defined
            if(not(len(YearRange)==2)):
                print('Please Specify the YearRange input to use system sizing then operation.')
                return
            
            if(YearRange[1]-YearRange[0] <= 0):
                print('YearRange[1] must be larger than YearRange[0]')
                return
            
            for y in range(len(SimYears)):
                if(YearRange[0] > SimYears[y] or YearRange[1] < SimYears[y] ):
                    print('The specified causally operated years must be within the year range')
                    return
            
            #assign datetimes to the demand and generation inputs
            x = dt.timedelta(hours = 1)
            N = dt.datetime(YearRange[0], 1, 1, hour = 0)
            
            date_range = []
            for t in range(len(surplus)):
                date_range.append(N)
                N=N+x
            
            #delete the correct values from power_out, demand and date_range!
            for t in range(len(surplus)-1,-1,-1):
                if date_range[t].month == 2 and date_range[t].day == 29:
                    date_range.pop(t)                #list
                    surplus = np.delete(surplus,[t]) #np.array
                    for g in model.GenIndex:
                        gen_list[g].power_out.pop(t) #list
                                                                              
            #then get the new timehorizon, will be 365*24*(YearRange[1]-YearRange[0] - 1)
            timehorizon = len(surplus) - 365*24
            
        else: #if not repeatedly reforming the model then no need to delete leap days
            timehorizon = len(surplus)
                       
        model.TimeIndex=range(timehorizon)  #NB, index for time of day, starts at 0         
        
        
        # Declare decision variables #
        
        #General
        model.Pfos=pyo.Var(model.TimeIndex, within = pyo.NonNegativeReals) #power from fossil fuels needed at time t
        model.Shed=pyo.Var(model.TimeIndex, within = pyo.NonNegativeReals) #amount of surplus shed
        
        #Storage
        model.BuiltCapacity=pyo.Var(model.StorageIndex, within = pyo.NonNegativeReals) #chosen capacity to build of each storage (MWh)
        model.C=pyo.Var(model.StorageIndex, model.TimeIndex, within = pyo.NonNegativeReals) #charging rate of each storage asset (MW)
        model.D=pyo.Var(model.StorageIndex, model.TimeIndex, within = pyo.NonNegativeReals) #discharging rate of each storage asset (MW)
        model.SOC=pyo.Var(model.StorageIndex, model.TimeIndex, within = pyo.NonNegativeReals) #state of charge of the asset at end of timestep t (MWh)    
        
        #EVs
        model.EV_TypeBuiltCapacity=pyo.Var(model.FleetIndex, model.ChargeType, within = pyo.NonNegativeReals) #number of built chargers for each type, V2G,Smart,Dumb
        model.EV_C=pyo.Var(model.FleetIndex, model.TimeIndex, model.ChargeType, within = pyo.NonNegativeReals) #hourly charging rate of each subset of charger types within each fleet(MW)
        model.EV_D=pyo.Var(model.FleetIndex, model.TimeIndex, model.ChargeType, within = pyo.NonNegativeReals) #hourly discharging rate of each subset of charger types within each fleet(MW)
        model.EV_SOC=pyo.Var(model.FleetIndex, model.TimeIndex, model.ChargeType, within = pyo.NonNegativeReals) #state of charge of the asset at end of timestep t (MWh)    
        
        #Generators
        model.GenCapacity = pyo.Var(model.GenIndex,within = pyo.NonNegativeReals) #the built capacity (MW) of each generator type
        
             
        # Declare Parameters #
        
        #Parameters save time on setting up the model for repeated runs, as they can be adjusted without having to rebuild the entire model
        #Declaring Parameters is fiddly, needs to be done with an enumerated dictionary of tuples as below
        
        #if I am running repeated simulations, then don't initialize parameters
        if(len(SimYears) > 0):
            model.Demand = pyo.Param(model.TimeIndex, within = pyo.Reals, mutable = True)    
            model.NormalisedGen = pyo.Param(model.GenIndex, model.TimeIndex, within = pyo.NonNegativeReals, mutable = True)
        else:
                
            model.Demand = pyo.Param(model.TimeIndex, initialize = dict(enumerate(surplus)), within = pyo.Reals, mutable = True)        
            
            ref = {} #this gives the normalised power output of generator g at time t (multiplied by the built capacity to give the MW)
            for g in model.GenIndex:
                for t in model.TimeIndex:
                    ref[g,model.TimeIndex[t]] = gen_list[g].power_out[t]/max(gen_list[g].power_out)
            model.NormalisedGen = pyo.Param(model.GenIndex, model.TimeIndex, within = pyo.NonNegativeReals, initialize = ref)
        
        
        N = np.empty([Mult_aggEV.n_assets,timehorizon]) #the normalised number of EVs connected at a given time (EV connections/disconnections are assumed to occur at teh start of the timestep)
        for k in range(Mult_aggEV.n_assets):
            for t in range(timehorizon):
                if t == 0:
                    N[k,t] = Mult_aggEV.assets[k].initial_number
                else:
                    N[k,t] = N[k,t-1] + Mult_aggEV.assets[k].Nin[t] - Mult_aggEV.assets[k].Nout[t]
        
        
        # Declare constraints #
        
        #General Constraints
        #Power Balance
        model.PowerBalance = pyo.ConstraintList()
        for t in range(timehorizon):                                                    #this is a normalised power output between 0-1
            #model.PowerBalance.add(surplus[t] + sum(model.GenCapacity[g]*(gen_list[g].power_out[t]/max(gen_list[g].power_out)) for g in model.GenIndex)- model.Shed[t] + sum(model.D[i,t] * Mult_Stor.assets[i].eff_out/100.0 - model.C[i,t] * 100.0/Mult_Stor.assets[i].eff_in for i in model.StorageIndex) + model.Pfos[t] +  sum(model.EV_D[k,t,0] * Mult_aggEV.assets[k].eff_out/100  - sum( model.EV_C[k,t,b] * 100/Mult_aggEV.assets[k].eff_in for b in model.ChargeType)  for k in model.FleetIndex)== 0)
            model.PowerBalance.add(model.Demand[t] + sum(model.GenCapacity[g]*model.NormalisedGen[g,t] for g in model.GenIndex)- model.Shed[t] + sum(model.D[i,t] * Mult_Stor.assets[i].eff_out/100.0 - model.C[i,t] * 100.0/Mult_Stor.assets[i].eff_in for i in model.StorageIndex) + model.Pfos[t] +  sum(model.EV_D[k,t,0] * Mult_aggEV.assets[k].eff_out/100  - sum( model.EV_C[k,t,b] * 100/Mult_aggEV.assets[k].eff_in for b in model.ChargeType)  for k in model.FleetIndex)== 0)
            
        #Specified Amount of Fossil Fuel Input
        model.FossilLimit = pyo.ConstraintList()
        model.FossilLimit.add(sum(model.Pfos[t] for t in model.TimeIndex) <= fossilLimit)
    
        #Generator Constraints
        model.genlimits = pyo.ConstraintList()
        for g in model.GenIndex:
            if(len(gen_list[g].limits)>0):
                model.genlimits.add(model.GenCapacity[g] >= gen_list[g].limits[0])
                model.genlimits.add(model.GenCapacity[g] <= gen_list[g].limits[1])
    
        #Storage Constraints
        model.maxSOC = pyo.ConstraintList()
        model.battery_charge_level = pyo.ConstraintList()
        model.maxD = pyo.ConstraintList()
        model.maxC = pyo.ConstraintList()
        model.storagelimits = pyo.ConstraintList()
        for i in range(Mult_Stor.n_assets):
            if(len(Mult_Stor.assets[i].limits)>0):
                model.storagelimits.add(model.BuiltCapacity[i] >=  Mult_Stor.assets[i].limits[0])
                model.storagelimits.add(model.BuiltCapacity[i] <=  Mult_Stor.assets[i].limits[1])
                
            for t in range(timehorizon):
                model.maxSOC.add(model.SOC[i,t] <= model.BuiltCapacity[i]) #SOC less than maximum
                model.maxD.add(model.D[i,t] * Mult_Stor.assets[i].eff_out/100.0 <= model.BuiltCapacity[i] * Mult_Stor.assets[i].max_d_rate/100)
                model.maxC.add(model.C[i,t] * 100.0/Mult_Stor.assets[i].eff_in <= model.BuiltCapacity[i] * Mult_Stor.assets[i].max_c_rate/100)               

                if t == 0:
                    model.battery_charge_level.add(model.SOC[i,t]== 0.5*model.BuiltCapacity[i] + model.C[i,t] - model.D[i,t]) #batteries all start at half charge
                else:
                    model.battery_charge_level.add(model.SOC[i,t]== model.SOC[i,t-1]*(1-(Mult_Stor.assets[i].self_dis/(100*30*24))) + model.C[i,t] - model.D[i,t])
            
        
        #EV Constraints
        model.EV_maxSOC = pyo.ConstraintList()
        model.EV_minSOC = pyo.ConstraintList()
        model.EV_battery_charge_level = pyo.ConstraintList()
        model.EV_maxD = pyo.ConstraintList()
        model.EV_maxC = pyo.ConstraintList()
        model.Built_Asset_Sum = pyo.ConstraintList()
        model.ChargerTypeLimits = pyo.ConstraintList() #allows the user to specify the max number of a certain charger type
        
        for k in range(Mult_aggEV.n_assets):
            #constraint to make sure all the different built capacities add to one
            model.Built_Asset_Sum.add(sum(model.EV_TypeBuiltCapacity[k,b] for b in model.ChargeType) == Mult_aggEV.assets[k].number)
            
            for b in model.ChargeType:
                if (len(Mult_aggEV.assets[k].limits) > 0):
                    model.ChargerTypeLimits.add(model.EV_TypeBuiltCapacity[k,b] >= Mult_aggEV.assets[k].limits[2*b])
                    model.ChargerTypeLimits.add(model.EV_TypeBuiltCapacity[k,b] <= Mult_aggEV.assets[k].limits[2*b+1])
                    
                for t in range(timehorizon):
                    model.EV_maxSOC.add(model.EV_SOC[k,t,b] <= model.EV_TypeBuiltCapacity[k,b]*N[k,t]*Mult_aggEV.assets[k].max_SOC/1000) #constraint to limit the max SOC
                    model.EV_minSOC.add(model.EV_SOC[k,t,b] >= model.EV_TypeBuiltCapacity[k,b]*Mult_aggEV.assets[k].Nout[t]*Mult_aggEV.assets[k].Eout/1000 - model.EV_TypeBuiltCapacity[k,b]*Mult_aggEV.assets[k].Nin[t]*Mult_aggEV.assets[k].Ein/1000) #constraint to make sure that there is always enough charge for the EVs to plug out
                    model.EV_maxC.add(model.EV_C[k,t,b] * 100/Mult_aggEV.assets[k].eff_in <= model.EV_TypeBuiltCapacity[k,b]*N[k,t]*Mult_aggEV.assets[k].max_c_rate/1000)
                    
                    if(b==0):
                        #V2G specific Constraints
                        model.EV_maxD.add(model.EV_D[k,t,b] * Mult_aggEV.assets[k].eff_out/100 <= model.EV_TypeBuiltCapacity[k,b]*N[k,t]*Mult_aggEV.assets[k].max_d_rate/1000)
                        
                        if t == 0:
                            model.EV_battery_charge_level.add(model.EV_SOC[k,t,b] == 0.5 * model.EV_TypeBuiltCapacity[k,b]*N[k,t]*Mult_aggEV.assets[k].max_SOC/1000 + model.EV_C[k,t,b] - model.EV_D[k,t,b] - model.EV_TypeBuiltCapacity[k,b]*Mult_aggEV.assets[k].Nout[t]*Mult_aggEV.assets[k].Eout/1000 + model.EV_TypeBuiltCapacity[k,b]*Mult_aggEV.assets[k].Nin[t]*Mult_aggEV.assets[k].Ein/1000) 
                        else:
                            model.EV_battery_charge_level.add(model.EV_SOC[k,t,b] == model.EV_SOC[k,t-1,b] + model.EV_C[k,t,b] - model.EV_D[k,t,b] - model.EV_TypeBuiltCapacity[k,b]*Mult_aggEV.assets[k].Nout[t]*Mult_aggEV.assets[k].Eout/1000 + model.EV_TypeBuiltCapacity[k,b]*Mult_aggEV.assets[k].Nin[t]*Mult_aggEV.assets[k].Ein/1000) 

                    elif b==1:
                        #Smart Unidirectional
                        if t == 0:
                            model.EV_battery_charge_level.add(model.EV_SOC[k,t,b] == 0.5 * model.EV_TypeBuiltCapacity[k,b]*N[k,t]*Mult_aggEV.assets[k].max_SOC/1000 + model.EV_C[k,t,b] - model.EV_TypeBuiltCapacity[k,b]*Mult_aggEV.assets[k].Nout[t]*Mult_aggEV.assets[k].Eout/1000 + model.EV_TypeBuiltCapacity[k,b]*Mult_aggEV.assets[k].Nin[t]*Mult_aggEV.assets[k].Ein/1000) 
                        else:
                            model.EV_battery_charge_level.add(model.EV_SOC[k,t,b] == model.EV_SOC[k,t-1,b] + model.EV_C[k,t,b] - model.EV_TypeBuiltCapacity[k,b]*Mult_aggEV.assets[k].Nout[t]*Mult_aggEV.assets[k].Eout/1000 + model.EV_TypeBuiltCapacity[k,b]*Mult_aggEV.assets[k].Nin[t]*Mult_aggEV.assets[k].Ein/1000) 

        # Declare objective function #
        model.obj=pyo.Objective(                                                                                                            #adding this small 0.05 stops the model from charging and discharging simultaneously unecessarily                                                                                                                                      this penalises fossil fuel use to encourage healthier charging behaviour
        expr=sum((timehorizon/(365*24))*model.GenCapacity[g]*gen_list[g].fixed_cost + model.GenCapacity[g]*gen_list[g].variable_cost*sum(gen_list[g].power_out[t]/max(gen_list[g].power_out) for t in model.TimeIndex) for g in model.GenIndex) + sum((timehorizon/(365*24))*Mult_Stor.assets[i].fixed_cost * model.BuiltCapacity[i] for i in model.StorageIndex) + sum( sum((Mult_Stor.assets[i].variable_cost+0.05) * model.D[i,t] for t in model.TimeIndex) for i in model.StorageIndex) + sum(sum((timehorizon/(365*24)) * Mult_aggEV.assets[k].chargercost[b] * model.EV_TypeBuiltCapacity[k,b] for b in model.ChargeType)for k in model.FleetIndex) + sum(model.Pfos[t] for t in model.TimeIndex) + sum( sum((0.05) * model.EV_D[k,t,0] for t in model.TimeIndex) for k in model.FleetIndex),   
        sense=pyo.minimize)
        
        
        end = time.time()
        print('Model Formation Complete after: ',int(end - start), 's')

# Two Options Depending on if Repeated Solves Are needed
        #set range of times that are learning from, remember, surplus is a year longer than timehorizon, so only selecting those entries that are relevant to us
        #need to extend to an iteration
        if(len(SimYears) > 0):
            for y in range(len(SimYears)):
                
                print('Forming New Instance...')
                start = time.time()
                # surplus_instance=[]
                # gen_power_instance = []
                # for g in range(len(gen_list)):#make the gen list the correct size
                #     gen_power_instance.append([])
                
                # #remove simulation year from the parameter input data
                # years_encompassed = range(YearRange[0],YearRange[1]+1) 
                # print(years_encompassed,'    ',SimYears[0] )
                # for x in range(YearRange[1]+1-YearRange[0]):
                #     if(not(years_encompassed[x] ==SimYears[0])):
                #         surplus_instance.append(surplus[x*365*24:(x+1)*365*24-1].tolist())
                #         for t in model.TimeIndex:
                #             for g in model.GenIndex:
                #                 gen_power_instance[g].append(gen_list[g].power_out[x*365*24+t]/max(gen_list[g].power_out))  
                
                surplus_instance=np.zeros([365*24*(YearRange[1]-YearRange[0])])
                gen_power_instance = np.zeros([len(gen_list),365*24*(YearRange[1]-YearRange[0])])
                
                #these values are for simulating the operation of the previously sized system
                surplus_simulation = np.zeros([365*24])
                gen_power_simulation = np.zeros([len(gen_list),365*24])
                
                #remove simulation year from the parameter input data
                years_encompassed = range(YearRange[0],YearRange[1]+1) 
                counter = 0
                for x in range(YearRange[1]+1-YearRange[0]):
                    if(not(years_encompassed[x] ==SimYears[y])):
                        surplus_instance[counter*365*24:(counter+1)*365*24] = surplus[x*365*24:(x+1)*365*24]
                        for g in model.GenIndex:
                            gen_power_instance[g,counter*365*24:(counter+1)*365*24] = np.asarray(gen_list[g].power_out[x*365*24:(x+1)*365*24])/max(gen_list[g].power_out)
                            
                        counter = counter + 1
                    else:
                        surplus_simulation = surplus[x*365*24:(x+1)*365*24]
                        for g in model.GenIndex:
                            gen_power_simulation[g,:] = np.asarray(gen_list[g].power_out[x*365*24:(x+1)*365*24])/max(gen_list[g].power_out)
                
                #fill in the correct parameters before running the model
                for t in model.TimeIndex:
                    model.Demand[t] = surplus_instance[t]
                    for g in model.GenIndex:
                        #print(gen_power_instance[g][t])
                        model.NormalisedGen[g,t] = gen_power_instance[g,t]
                
                end = time.time()
                print('Instance Complete: ',int(end - start), 's')   
    
    
                # Solve #
                print('Searching for Optimal Solution...')
                start = time.time()
                
                opt.solve(model)  
                
                end = time.time()
                print('Solved after: ',int(end - start), 's')
                
                # Store Results #
                # for g in model.GenIndex:           
                #     gen_list[g].total_installed_capacity = (pyo.value(model.GenCapacity[g]))
                #     #gen_list[g].power_out = pyo.value(model.GenCapacity[g])*(np.asarray(gen_list[g].power_out)/max(gen_list[g].power_out))
                    
                for i in range(Mult_Stor.n_assets):
                    Mult_Stor.assets[i].set_capacity(pyo.value(model.BuiltCapacity[i]))
                    
                for k in range(Mult_aggEV.n_assets):
                    Mult_aggEV.assets[k].reset()                
                    for b in range(2):
                        Mult_aggEV.assets[k].chargertype[b] = pyo.value(model.EV_TypeBuiltCapacity[k,b])/Mult_aggEV.assets[k].number


                # Simulate System Operation #
                
                print('Simulate Causal Operation...')
                start = time.time()
                #First Method is ordered Charging
                input1 = surplus_simulation + sum(pyo.value(model.GenCapacity[g])*gen_power_simulation[g,:] for g in model.GenIndex)
                Old_Reliability = Mult_Stor.charge_specfied_order(surplus = input1, c_order = Mult_Stor.c_order, d_order = Mult_Stor.d_order, t_res=1,
                              return_output=False,start_up_time=168,
                              return_di_av=False)
                end = time.time()
                print('Simulation Complete: ',int(end - start), 's')
                print('Reliability = ', Old_Reliability)
                print('Usage: ', Mult_Stor.analyse_usage())
        else:
            # Solve #
            print('Searching for Optimal Solution...')
            start = time.time()
            
            opt.solve(model)  
            
            end = time.time()
            print('Solved after: ',int(end - start), 's')
        
        # Store Results #
            #Generator
            for g in model.GenIndex:           
                gen_list[g].power_out = pyo.value(model.GenCapacity[g])*(np.asarray(gen_list[g].power_out)/max(gen_list[g].power_out))
                gen_list[g].total_installed_capacity = (pyo.value(model.GenCapacity[g]))
            
            #Storage
            SOC_results = np.empty([Mult_Stor.n_assets, timehorizon+1])
            D_results = np.empty([Mult_Stor.n_assets, timehorizon])
            C_results = np.empty([Mult_Stor.n_assets, timehorizon])
            
            for i in range(Mult_Stor.n_assets):
                Mult_Stor.assets[i].reset()
                Mult_Stor.assets[i].set_capacity(pyo.value(model.BuiltCapacity[i]))
                
                SOC_results[i,0] = 0.0;
                
                for t in range(0,timehorizon):
                    SOC_results[i,t+1]=pyo.value(model.SOC[i,t])
                    D_results[i,t] = -pyo.value(model.D[i,t])
                    C_results[i,t] = pyo.value(model.C[i,t])
            
                Mult_Stor.assets[i].en_in = sum(C_results[i,:]*100/Mult_Stor.assets[i].eff_in)
                Mult_Stor.assets[i].en_out = sum(D_results[i,:]*Mult_Stor.assets[i].eff_out/100)
                Mult_Stor.assets[i].discharge = D_results[i,:]
                Mult_Stor.assets[i].charge = C_results[i,:]
                Mult_Stor.assets[i].SOC = SOC_results[i,:]
                
    
            Pfos_results = np.empty([timehorizon])
            Shed_results = np.empty([timehorizon])
            for t in range(0,timehorizon):
                Pfos_results[t] = pyo.value(model.Pfos[t])
                Shed_results[t] = pyo.value(model.Shed[t])
            
            Mult_Stor.Pfos = Pfos_results #timeseries of the power from fossil fuels (MW)
            Mult_Stor.Shed = Shed_results #timeseries of the shed renewable power
            Mult_Stor.surplus = surplus +sum((gen_list[g].power_out) for g in model.GenIndex)
            
            
            #Agg EV Fleets
            EV_SOC_results = np.empty([Mult_aggEV.n_assets,timehorizon+1,2])
            EV_D_results = np.empty([Mult_aggEV.n_assets,timehorizon]) #as only V2G has discharge variable
            EV_C_results = np.empty([Mult_aggEV.n_assets,timehorizon,2])
            
            for k in range(Mult_aggEV.n_assets):
                Mult_aggEV.assets[k].reset()
                
                for b in range(2):
                    Mult_aggEV.assets[k].chargertype[b] = pyo.value(model.EV_TypeBuiltCapacity[k,b])/Mult_aggEV.assets[k].number
                    EV_SOC_results[k,0,b] = 0.5 * pyo.value(model.EV_TypeBuiltCapacity[k,b])*N[k,t]*Mult_aggEV.assets[k].max_SOC/1000
                    
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
                
            
            Mult_aggEV.Pfos = Pfos_results #timeseries of the power from fossil fuels (MW)
            Mult_aggEV.Shed = Shed_results #timeseries of the shed renewable power
            Mult_aggEV.surplus = surplus +sum((gen_list[g].power_out) for g in model.GenIndex)
            Mult_aggEV.driving_energy = sum(Mult_aggEV.assets[k].number*sum(Mult_aggEV.assets[k].Nout[t]*(Mult_aggEV.assets[k].Eout-Mult_aggEV.assets[k].Ein)/1000  for t in model.TimeIndex)for k in model.FleetIndex)
        