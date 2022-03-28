import numpy as np
#optimisation high level language, help found at https://www.ima.umn.edu/materials/2017-2018.2/W8.21-25.17/26326/3_PyomoFundamentals.pdf
#low level algorithmic solve performed by solver
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

def optimise_configuration(surplus,fossilLimit,Mult_Stor,Mult_aggEV,gen_list=[],fixed_capacities =  False):
        '''
        == description ==
        For a given surplus, returns the cost optimal storage mix to meet the specified reliability. Charge order not relevant here.

        == parameters ==
        surplus: (Array<float>) the surplus generation to be smoothed in MW
        fossilLimit: (float) max acceptable amount of fossil fuel generated energy (MWh)
        StorAssets: (Multiple Storage Assets )
        gen_list: (array<generation>): list of the potential renewable generators to build
        fixed_capacities: (bool) when true, this is run as an operational optimiser rather than an investement tool.
        == returns ==
        '''
        print('Remember, surplus timeseries must have hourly entries for this to work. ')
        print('Assuming hourly timeseries, Forming Optimisation Model...')

            
            
        # Create a solver #
        opt = pyo.SolverFactory('mosek')
        
        # Create model #
        model=pyo.ConcreteModel()  #create the model object, concrete because parameters are known
        
        # Declare Indexs #
        timehorizon = len(surplus)
        model.TimeIndex=range(timehorizon)  #NB, index for time of day, starts at 0 
        model.StorageIndex=range(Mult_Stor.n_assets) #Index over the storage assets
        model.FleetIndex=range(Mult_aggEV.n_assets) #Index over Fleets
        model.ChargeType = range(2) #Index over Type of Charger, 0: V2G, 1: Smart, 2: Unmanaged
        model.GenIndex = range(len(gen_list)) #Index over possible generators
        
        
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
        
        N = np.empty([Mult_aggEV.n_assets,timehorizon]) #the normalised number of EVs connected at a given time (EV connections/disconnections are assumed to occur at teh start of the timestep)
        for k in range(Mult_aggEV.n_assets):
            for t in range(timehorizon):
                if t == 0:
                    N[k,t] = Mult_aggEV.assets[k].initial_number
                else:
                    N[k,t] = N[k,t-1] + Mult_aggEV.assets[k].Nin[t] - Mult_aggEV.assets[k].Nout[t]
        
        #Generators
        model.GenCapacity = pyo.Var(model.GenIndex,within = pyo.NonNegativeReals) #the built capacity (MW) of each generator type
        
        # Declare constraints #
        
        #General Constraints
        #Power Balance
        model.PowerBalance = pyo.ConstraintList()
        for t in range(timehorizon):                                                    #this is a normalised power output between 0-1
            model.PowerBalance.add(surplus[t] + sum(model.GenCapacity[g]*(gen_list[g].power_out[t]/max(gen_list[g].power_out)) for g in model.GenIndex)- model.Shed[t] + sum(model.D[i,t] * Mult_Stor.assets[i].eff_out/100.0 - model.C[i,t] * 100.0/Mult_Stor.assets[i].eff_in for i in model.StorageIndex) + model.Pfos[t] +  sum(model.EV_D[k,t,0] * Mult_aggEV.assets[k].eff_out/100  - sum( model.EV_C[k,t,b] * 100/Mult_aggEV.assets[k].eff_in for b in model.ChargeType)  for k in model.FleetIndex)== 0)
            
        #Specified Amount of Fossil Fuel Input
        model.FossilLimit = pyo.ConstraintList()
        if(not fixed_capacities):
            model.FossilLimit.add(sum(model.Pfos[t] for t in model.TimeIndex) <= fossilLimit)
        
        #fix capacities if needed
        model.fixed_cap = pyo.ConstraintList()
        if(fixed_capacities):
            for i in range(Mult_Stor.n_assets):
                model.fixed_cap.add(model.BuiltCapacity[i] == Mult_Stor.assets[i].capacity)
                
            for g in model.GenIndex:
                model.fixed_cap.add(model.GenCapacity[g] == gen_list[g].total_installed_capacity)
                
            for k in range(Mult_aggEV.n_assets):
                for b in model.ChargeType:
                    model.fixed_cap.add(model.EV_TypeBuiltCapacity[k,b] == Mult_aggEV.assets[k].chargertype[b] * Mult_aggEV.assets[k].number)
                    
        
        
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
        for k in range(Mult_aggEV.n_assets):
            #constraint to make sure all the different built capacities add to one
            model.Built_Asset_Sum.add(sum(model.EV_TypeBuiltCapacity[k,b] for b in model.ChargeType) == Mult_aggEV.assets[k].number)
            
            for b in model.ChargeType:
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
        if(not fixed_capacities):
            model.obj=pyo.Objective(                                                                                                            #adding this small 0.05 stops the model from charging and discharging simultaneously unecessarily                                                                                                                                      this penalises fossil fuel use to encourage healthier charging behaviour
            expr=sum((timehorizon/(365*24))*model.GenCapacity[g]*gen_list[g].fixed_cost + model.GenCapacity[g]*gen_list[g].variable_cost*sum(gen_list[g].power_out[t]/max(gen_list[g].power_out) for t in model.TimeIndex) for g in model.GenIndex) + sum((timehorizon/(365*24))*Mult_Stor.assets[i].fixed_cost * model.BuiltCapacity[i] for i in model.StorageIndex) + sum( sum((Mult_Stor.assets[i].variable_cost+0.05) * model.D[i,t] for t in model.TimeIndex) for i in model.StorageIndex) + sum(sum((timehorizon/(365*24)) * Mult_aggEV.assets[k].chargercost[b] * model.EV_TypeBuiltCapacity[k,b] for b in model.ChargeType)for k in model.FleetIndex) + sum(model.Pfos[t] for t in model.TimeIndex) + sum( sum((0.05) * model.EV_D[k,t,0] for t in model.TimeIndex) for k in model.FleetIndex),   
            sense=pyo.minimize)
        else:
            #with fixed capacities, the only thing to minimise is the amount of fossil fuels used
            model.obj=pyo.Objective(                                
            expr=sum(100 * model.Pfos[t] for t in model.TimeIndex), 
            sense=pyo.minimize)
        
        print('Searching for Optimal Solution...')
        
        # Solve #
        opt.solve(model)    
        print('Solved')
        
        # Store Results #  
        #print('Total Cost System Cost: £',int(pyo.value(model.obj)*1e-6)*1e-3,'bn')
        
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
        