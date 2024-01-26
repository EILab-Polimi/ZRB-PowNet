# coding: utf-8
from __future__ import division  # convert int or long division arguments to floating point values before division
from pyomo.environ import *

gd_nodes = ['STAF', 'ZAM', 'ZIM', 'BTSW', 'NAM', 'SWA', 'DRC', 'ANG', 'TAN', 'MAL', 'MOZ', 'LES']

g_nodes = gd_nodes  # + gn_nodes
print('Gen_Nodes:', len(g_nodes))

model = AbstractModel()

######=================================================########
######               Segment B.1                       ########
######=================================================########

## string indentifiers for the set of generators (in the order of g_nodes list)
model.GD1Gens = Set()
model.GD2Gens = Set()
model.GD3Gens = Set()
model.GD4Gens = Set()
model.GD5Gens = Set()
model.GD6Gens = Set()
model.GD7Gens = Set()
model.GD8Gens = Set()
model.GD9Gens = Set()
model.GD10Gens = Set()
model.GD11Gens = Set()
model.GD12Gens = Set()

model.Generators = model.GD1Gens | model.GD2Gens | model.GD3Gens | model.GD4Gens | \
                   model.GD5Gens | model.GD6Gens | model.GD7Gens | model.GD8Gens | \
                   model.GD9Gens | model.GD10Gens | model.GD11Gens | model.GD12Gens

### Generators by fuel-type
model.Coal_st = Set()
model.Oil_ic = Set()
model.Biomass_st = Set()
model.Gas_st = Set()
model.Nuclear = Set()

###Allocate generators that will ensure minimum reserves
model.ResGenerators = model.Coal_st | model.Oil_ic | model.Nuclear

######=================================================########
######               Segment B.2                       ########
######=================================================########

### Nodal sets
model.nodes = Set()
model.sources = Set(within=model.nodes)
model.sinks = Set(within=model.nodes)

model.h_nodes = Set()
model.s_nodes = Set()
model.w_nodes = Set()
model.d_nodes = Set()
model.gd_nodes = Set()

######=================================================########
######               Segment B.3                       ########
######=================================================########

#####==== Parameters for dispatchable resources ===####

# Generator type
model.typ = Param(model.Generators, within=Any)

# Node name
model.node = Param(model.Generators, within=Any)

# Max capacity
model.maxcap = Param(model.Generators, within=Any)

# Min capacity
model.mincap = Param(model.Generators, within=Any)

# Heat rate
model.heat_rate = Param(model.Generators, within=Any)

# Variable O&M
model.var_om = Param(model.Generators, within=Any)

# Fixed O&M cost
model.fix_om = Param(model.Generators, within=Any)

# Start cost
model.st_cost = Param(model.Generators, within=Any)

# Ramp rate
model.ramp = Param(model.Generators, within=Any)

# Minimun up time
model.minup = Param(model.Generators, within=Any)

# Minmun down time
model.mindn = Param(model.Generators, within=Any)

# Derate_factor as percent of maximum capacity of water-dependant generators
model.deratef = Param(model.Generators, within=NonNegativeReals)

# heat rates and import unit costs
model.gen_cost = Param(model.Generators, within=NonNegativeReals)


######=================================================########
######               Segment B.4                       ########
######=================================================########

######==== Transmission line parameters =======#######
model.linemva = Param(model.sources, model.sinks, mutable=True)
model.linesus = Param(model.sources, model.sinks)

### Transmission Loss as a %discount on production
model.TransLoss = Param(within=NonNegativeReals)

### Maximum line-usage as a percent of line-capacity
model.n1criterion = Param(within=NonNegativeReals)

### Minimum spinning reserve as a percent of total reserve
model.spin_margin = Param(within=NonNegativeReals)

# todo modificato per oncon riga 274
model.m = Param(initialize=5e5)

######=================================================########
######               Segment B.5                       ########
######=================================================########

######===== Parameters/initial_conditions to run simulation ======####### 
## Full range of time series information
model.SimHours = Param(within=PositiveIntegers)
model.SH_periods = RangeSet(1, model.SimHours + 1)
model.SimDays = Param(within=PositiveIntegers)
model.SD_periods = RangeSet(1, model.SimDays + 1)

# Operating horizon information 
model.HorizonHours = Param(within=PositiveIntegers)  # HorizonHours=24
model.HH_periods = RangeSet(0, model.HorizonHours)
model.hh_periods = RangeSet(1, model.HorizonHours)
model.ramp_periods = RangeSet(2, 24)

######=================================================########
######               Segment B.6                       ########
######=================================================########

# Demand over simulation period
model.SimDemand = Param(model.d_nodes * model.SH_periods, within=NonNegativeReals)
# Horizon demand
model.HorizonDemand = Param(model.d_nodes * model.hh_periods, within=NonNegativeReals, mutable=True)

# Reserve for the entire system
model.SimReserves = Param(model.SH_periods, within=NonNegativeReals)
model.HorizonReserves = Param(model.hh_periods, within=NonNegativeReals, mutable=True)

##Variable resources over simulation period
model.SimHydro = Param(model.h_nodes, model.SH_periods, within=NonNegativeReals)
model.SimSolar = Param(model.s_nodes, model.SH_periods, within=NonNegativeReals)
model.SimWind = Param(model.w_nodes, model.SH_periods, within=NonNegativeReals)

# Variable resources over horizon
model.HorizonHydro = Param(model.h_nodes, model.hh_periods, within=NonNegativeReals, mutable=True)
model.HorizonSolar = Param(model.s_nodes, model.hh_periods, within=NonNegativeReals, mutable=True)
model.HorizonWind = Param(model.w_nodes, model.hh_periods, within=NonNegativeReals, mutable=True)

# Max hydropower over horizon
model.HorizonHydroMax = Param(model.h_nodes, within=NonNegativeReals, mutable=True)

##Initial conditions
model.ini_on = Param(model.Generators, within=NonNegativeReals, mutable=True)

######=================================================########
######               Segment B.7                       ########
######=================================================########

######=======================Decision variables======================########
##Amount of day-ahead energy generated by each generator at each hour
model.mwh = Var(model.Generators, model.HH_periods, within=NonNegativeReals)

# 1 if unit is on in hour i, otherwise 0
model.on = Var(model.Generators, model.HH_periods, within=Binary)

# 1 if unit is switching on in hour i, otherwise 0
model.switch = Var(model.Generators, model.HH_periods, within=Binary)

# Amount of spining reserve offered by an unit in each hour
model.srsv = Var(model.Generators, model.HH_periods, within=NonNegativeReals)

# Amount of non-sping reserve offered by an unit in each hour
model.nrsv = Var(model.Generators, model.HH_periods, within=NonNegativeReals)

# dispatch of hydropower from each domestic dam in each hour
model.hydro = Var(model.h_nodes, model.HH_periods, within=NonNegativeReals)

# dispatch of solar-power in each hour
model.solar = Var(model.s_nodes, model.HH_periods, within=NonNegativeReals)

# dispatch of wind-power in each hour
model.wind = Var(model.w_nodes, model.HH_periods, within=NonNegativeReals)

# Voltage angle at each node in each hour
model.vlt_angle = Var(model.nodes, model.HH_periods)


######=================================================########
######               Segment B.8                       ########
######=================================================########

######================Objective function=============########
# todo implementare un costo fittizio delle res per farle rientrare nella funzione obiettivo ? ## CHECK ##

def SysCost(model):
    fixed = sum(model.maxcap[j] * model.fix_om[j] * model.on[j, i] for i in model.hh_periods for j in model.Generators)

    starts = sum(
        model.maxcap[j] * model.st_cost[j] * model.switch[j, i] for i in model.hh_periods for j in model.Generators)

    coal_st = sum(
        model.mwh[j, i] * (model.heat_rate[j] * model.gen_cost[j] + model.var_om[j]) for i in model.hh_periods for j in
        model.Coal_st)
    oil_ic = sum(
        model.mwh[j, i] * (model.heat_rate[j] * model.gen_cost[j] + model.var_om[j]) for i in model.hh_periods for j in
        model.Oil_ic)
    biomass_st = sum(
        model.mwh[j, i] * (model.heat_rate[j] * model.gen_cost[j] + model.var_om[j]) for i in model.hh_periods for j in
        model.Biomass_st)
    gas_st = sum(
        model.mwh[j, i] * (model.heat_rate[j] * model.gen_cost[j] + model.var_om[j]) for i in model.hh_periods for j in
        model.Gas_st)
    nuclear = sum(
        model.mwh[j, i] * (model.heat_rate[j] * model.gen_cost[j] + model.var_om[j]) for i in model.hh_periods for j in
        model.Nuclear)

    infinite_solar = sum(
        model.solar['INF_SOURCE_s', i] * 1e6 for i in model.hh_periods)

    hydro_nudge = sum(
        model.hydro[j, i] * 1 for i in model.hh_periods for j in
            model.h_nodes)


    return fixed + starts + coal_st + oil_ic + gas_st + biomass_st + nuclear + infinite_solar + hydro_nudge

model.SystemCost = Objective(rule=SysCost, sense=minimize)

######=================================================########
######               Segment B.9                      ########
######=================================================########

######========== Logical Constraint =========#############
def OnCon(model, j, i):
    return model.mwh[j, i] <= model.on[j, i] * model.m


model.OnConstraint = Constraint(model.Generators, model.HH_periods, rule=OnCon)


def OnCon_initial(model, j, i):
    if i == 0:
        return (model.on[j, i] == model.ini_on[j])
    else:
        return Constraint.Skip


model.initial_value_constr = Constraint(model.Generators, model.HH_periods, rule=OnCon_initial)


def SwitchCon2(model, j, i):
    return model.switch[j, i] <= model.on[j, i] * model.m


model.Switch2Constraint = Constraint(model.Generators, model.hh_periods, rule=SwitchCon2)


def SwitchCon3(model, j, i):
    return model.switch[j, i] <= (1 - model.on[j, i - 1]) * model.m


model.Switch3Constraint = Constraint(model.Generators, model.hh_periods, rule=SwitchCon3)


def SwitchCon4(model, j, i):
    return model.on[j, i] - model.on[j, i - 1] <= model.switch[j, i]


model.Switch4Constraint = Constraint(model.Generators, model.hh_periods, rule=SwitchCon4)


######========== Up/Down Time Constraint =========#############
##Min Up time
def MinUp(model, j, i, k):
    if i > 0 and k > i and k < min(i + model.minup[j] - 1, model.HorizonHours):
        return model.on[j, i] - model.on[j, i - 1] <= model.on[j, k]
    else:
        return Constraint.Skip


model.MinimumUp = Constraint(model.Generators, model.HH_periods, model.HH_periods, rule=MinUp)


##Min Down time
def MinDown(model, j, i, k):
    if i > 0 and k > i and k < min(i + model.mindn[j] - 1, model.HorizonHours):
        return model.on[j, i - 1] - model.on[j, i] <= 1 - model.on[j, k]
    else:
        return Constraint.Skip


model.MinimumDown = Constraint(model.Generators, model.HH_periods, model.HH_periods, rule=MinDown)


######==========Ramp Rate Constraints =========#############
def Ramp1(model, j, i):
    a = model.mwh[j, i]
    b = model.mwh[j, i - 1]
    return a - b <= model.ramp[j]


model.RampCon1 = Constraint(model.Generators, model.ramp_periods, rule=Ramp1)


def Ramp2(model, j, i):
    a = model.mwh[j, i]
    b = model.mwh[j, i - 1]
    return b - a <= model.ramp[j]


model.RampCon2 = Constraint(model.Generators, model.ramp_periods, rule=Ramp2)

# def Ramp3(model, j, i):
#     a = model.hydro[j, i]
#     b = model.hydro[j, i - 1]
#     return a - b <= 0.3 * b


# model.RampCon3 = Constraint(model.h_nodes, model.ramp_periods, rule=Ramp3)


# def Ramp4(model, j, i):
#     a = model.hydro[j, i]
#     b = model.hydro[j, i - 1]
#     return b - a <= 0.3 * b


# model.RampCon4 = Constraint(model.h_nodes, model.ramp_periods, rule=Ramp4)

######=================================================########
######               Segment B.10                      ########
######=================================================########

######=========== Capacity Constraints ============##########
# Constraints for Max & Min Capacity of dispatchable resources
# derate factor can be below 1 for dry years, otherwise 1
# todo implementare la condizione che la produzione di solare e wind sia >0 ?
def MaxC(model, j, i):
    return model.mwh[j, i] <= model.on[j, i] * model.maxcap[j] * model.deratef[j]


model.MaxCap = Constraint(model.Generators, model.hh_periods, rule=MaxC)


def MinC(model, j, i):
    return model.mwh[j, i] >= model.on[j, i] * model.mincap[j]


model.MinCap = Constraint(model.Generators, model.hh_periods, rule=MinC)


# Max hourly capacity constraints on domestic hydropower
def HydroC(model, z, i):
    return model.hydro[z, i] <= model.HorizonHydro[z, i]


model.HydroConstraint = Constraint(model.h_nodes, model.hh_periods, rule=HydroC)


# Max daily capacity constraint on domestic hydropower
def HydroC_max_horizon(model, z):
    return sum(model.hydro[z, i] for i in model.hh_periods) <= model.HorizonHydroMax[z]


model.HydroConstraint_max_horizon = Constraint(model.h_nodes, rule=HydroC_max_horizon)


# Max capacity constraints on solar
def SolarC(model, z, i):
    return model.solar[z, i] <= model.HorizonSolar[z, i]


model.SolarConstraint = Constraint(model.s_nodes, model.hh_periods, rule=SolarC)


# Max capacity constraints on wind
def WindC(model, z, i):
    return model.wind[z, i] <= model.HorizonWind[z, i]


model.WindConstraint = Constraint(model.w_nodes, model.hh_periods, rule=WindC)



######=================================================########
######               Segment B.11.2                    ########
######=================================================########

######=================== Power balance in nodes of variable resources (without demand in this case) =================########

###Hydropower Plants
# todo non posso  considerare questi vincoli perche vanno in conflitto con
#  Power balance in nodes of dispatchable resources with demand per il voltAngle
def HPnodes_Balance(model, z, i):
    dis_hydro = model.hydro[z, i]
    impedance = sum(model.linesus[z, k] * (model.vlt_angle[z, i] - model.vlt_angle[k, i]) for k in model.sinks)
    return (1 - model.TransLoss) * dis_hydro == impedance  # - demand


model.HPnodes_BalConstraint = Constraint(model.h_nodes, model.hh_periods, rule=HPnodes_Balance)

####Solar Plants
def Solarnodes_Balance(model, z, i):
    dis_solar = model.solar[z, i]
    impedance = sum(model.linesus[z, k] * (model.vlt_angle[z, i] - model.vlt_angle[k, i]) for k in model.sinks)
    return (1 - model.TransLoss) * dis_solar == impedance  # - demand


model.Solarnodes_BalConstraint = Constraint(model.s_nodes, model.hh_periods, rule=Solarnodes_Balance)


#####Wind Plants
def Windnodes_Balance(model, z, i):
    dis_wind = model.wind[z, i]
    impedance = sum(model.linesus[z, k] * (model.vlt_angle[z, i] - model.vlt_angle[k, i]) for k in model.sinks)
    return (1 - model.TransLoss) * dis_wind == impedance  # - demand


model.Windnodes_BalConstraint = Constraint(model.w_nodes, model.hh_periods, rule=Windnodes_Balance)


######=================================================########
######               Segment B.11.3                    ########
######=================================================########

##########============ Power balance in nodes of dispatchable resources with demand ==============############
def GD1_Balance(model, i):
    gd = 1
    thermo = sum(model.mwh[j, i] for j in model.GD1Gens)
    demand = model.HorizonDemand[gd_nodes[gd - 1], i]
    impedance = sum(
        model.linesus[gd_nodes[gd - 1], k] * (model.vlt_angle[gd_nodes[gd - 1], i] - model.vlt_angle[k, i]) for k in
        model.sinks)
    return (1 - model.TransLoss) * thermo - demand == impedance


model.GD1_BalConstraint = Constraint(model.hh_periods, rule=GD1_Balance)


def GD2_Balance(model, i):
    gd = 2
    thermo = sum(model.mwh[j, i] for j in model.GD2Gens)
    demand = model.HorizonDemand[gd_nodes[gd - 1], i]
    impedance = sum(
        model.linesus[gd_nodes[gd - 1], k] * (model.vlt_angle[gd_nodes[gd - 1], i] - model.vlt_angle[k, i]) for k in
        model.sinks)
    return (1 - model.TransLoss) * thermo - demand == impedance


model.GD2_BalConstraint = Constraint(model.hh_periods, rule=GD2_Balance)


def GD3_Balance(model, i):
    gd = 3
    thermo = sum(model.mwh[j, i] for j in model.GD3Gens)
    demand = model.HorizonDemand[gd_nodes[gd - 1], i]
    impedance = sum(
        model.linesus[gd_nodes[gd - 1], k] * (model.vlt_angle[gd_nodes[gd - 1], i] - model.vlt_angle[k, i]) for k in
        model.sinks)
    return (1 - model.TransLoss) * thermo - demand == impedance


model.GD3_BalConstraint = Constraint(model.hh_periods, rule=GD3_Balance)


def GD4_Balance(model, i):
    gd = 4
    thermo = sum(model.mwh[j, i] for j in model.GD4Gens)
    demand = model.HorizonDemand[gd_nodes[gd - 1], i]
    impedance = sum(
        model.linesus[gd_nodes[gd - 1], k] * (model.vlt_angle[gd_nodes[gd - 1], i] - model.vlt_angle[k, i]) for k in
        model.sinks)
    return (1 - model.TransLoss) * thermo - demand == impedance


model.GD4_BalConstraint = Constraint(model.hh_periods, rule=GD4_Balance)


def GD5_Balance(model, i):
    gd = 5
    thermo = sum(model.mwh[j, i] for j in model.GD5Gens)
    demand = model.HorizonDemand[gd_nodes[gd - 1], i]
    impedance = sum(
        model.linesus[gd_nodes[gd - 1], k] * (model.vlt_angle[gd_nodes[gd - 1], i] - model.vlt_angle[k, i]) for k in
        model.sinks)
    return (1 - model.TransLoss) * thermo - demand == impedance


model.GD5_BalConstraint = Constraint(model.hh_periods, rule=GD5_Balance)


def GD6_Balance(model, i):
    gd = 6
    thermo = sum(model.mwh[j, i] for j in model.GD6Gens)
    demand = model.HorizonDemand[gd_nodes[gd - 1], i]
    impedance = sum(
        model.linesus[gd_nodes[gd - 1], k] * (model.vlt_angle[gd_nodes[gd - 1], i] - model.vlt_angle[k, i]) for k in
        model.sinks)
    return (1 - model.TransLoss) * thermo - demand == impedance


model.GD6_BalConstraint = Constraint(model.hh_periods, rule=GD6_Balance)


def GD7_Balance(model, i):
    gd = 7
    thermo = sum(model.mwh[j, i] for j in model.GD7Gens)
    demand = model.HorizonDemand[gd_nodes[gd - 1], i]
    impedance = sum(
        model.linesus[gd_nodes[gd - 1], k] * (model.vlt_angle[gd_nodes[gd - 1], i] - model.vlt_angle[k, i]) for k in
        model.sinks)
    return (1 - model.TransLoss) * thermo - demand == impedance


model.GD7_BalConstraint = Constraint(model.hh_periods, rule=GD7_Balance)


def GD8_Balance(model, i):
    gd = 8
    thermo = sum(model.mwh[j, i] for j in model.GD8Gens)
    demand = model.HorizonDemand[gd_nodes[gd - 1], i]
    impedance = sum(
        model.linesus[gd_nodes[gd - 1], k] * (model.vlt_angle[gd_nodes[gd - 1], i] - model.vlt_angle[k, i]) for k in
        model.sinks)
    return (1 - model.TransLoss) * thermo - demand == impedance


model.GD8_BalConstraint = Constraint(model.hh_periods, rule=GD8_Balance)


def GD9_Balance(model, i):
    gd = 9
    thermo = sum(model.mwh[j, i] for j in model.GD9Gens)
    demand = model.HorizonDemand[gd_nodes[gd - 1], i]
    impedance = sum(
        model.linesus[gd_nodes[gd - 1], k] * (model.vlt_angle[gd_nodes[gd - 1], i] - model.vlt_angle[k, i]) for k in
        model.sinks)
    return (1 - model.TransLoss) * thermo - demand == impedance


model.GD9_BalConstraint = Constraint(model.hh_periods, rule=GD9_Balance)


def GD10_Balance(model, i):
    gd = 10
    thermo = sum(model.mwh[j, i] for j in model.GD10Gens)
    demand = model.HorizonDemand[gd_nodes[gd - 1], i]
    impedance = sum(
        model.linesus[gd_nodes[gd - 1], k] * (model.vlt_angle[gd_nodes[gd - 1], i] - model.vlt_angle[k, i]) for k in
        model.sinks)
    return (1 - model.TransLoss) * thermo - demand == impedance


model.GD10_BalConstraint = Constraint(model.hh_periods, rule=GD10_Balance)


def GD11_Balance(model, i):
    gd = 11
    thermo = sum(model.mwh[j, i] for j in model.GD11Gens)
    demand = model.HorizonDemand[gd_nodes[gd - 1], i]
    impedance = sum(
        model.linesus[gd_nodes[gd - 1], k] * (model.vlt_angle[gd_nodes[gd - 1], i] - model.vlt_angle[k, i]) for k in
        model.sinks)
    return (1 - model.TransLoss) * thermo - demand == impedance


model.GD11_BalConstraint = Constraint(model.hh_periods, rule=GD11_Balance)


def GD12_Balance(model, i):
    gd = 12
    thermo = sum(model.mwh[j, i] for j in model.GD12Gens)
    demand = model.HorizonDemand[gd_nodes[gd - 1], i]
    impedance = sum(
        model.linesus[gd_nodes[gd - 1], k] * (model.vlt_angle[gd_nodes[gd - 1], i] - model.vlt_angle[k, i]) for k in
        model.sinks)
    return (1 - model.TransLoss) * thermo - demand == impedance


model.GD12_BalConstraint = Constraint(model.hh_periods, rule=GD12_Balance)


######=================================================########
######               Segment B.12                    ########
######=================================================########

######==================Transmission  constraints==================########

####=== Reference Node =====#####
# todo definire il reference node, ovvero quello con la demand piu alta (a cui si aggiunge anche il contributo dello
#  deficit)
def ref_node(model, i):
    return model.vlt_angle['STAF', i] == 0


model.Ref_NodeConstraint = Constraint(model.hh_periods, rule=ref_node)


######========== Transmission Capacity Constraints (N-1 Criterion) =========#############
def MaxLine(model, s, k, i):
    if model.linesus[s, k] > 0:
        return (model.n1criterion) * model.linemva[s, k] >= model.linesus[s, k] * (
                    model.vlt_angle[s, i] - model.vlt_angle[k, i])
    else:
        return Constraint.Skip


model.MaxLineConstraint = Constraint(model.sources, model.sinks, model.hh_periods, rule=MaxLine)


def MinLine(model, s, k, i):
    if model.linesus[s, k] > 0:
        return (-model.n1criterion) * model.linemva[s, k] <= model.linesus[s, k] * (
                    model.vlt_angle[s, i] - model.vlt_angle[k, i])
    else:
        return Constraint.Skip


model.MinLineConstraint = Constraint(model.sources, model.sinks, model.hh_periods, rule=MinLine)


######=================================================########
######               Segment B.13                      ########
######=================================================########

######===================Reserve and zero-sum constraints ==================########

##System Reserve Requirement
def SysReserve(model, i):
    return sum(model.srsv[j, i] for j in model.ResGenerators) + sum(model.nrsv[j, i] for j in model.ResGenerators) >= \
           model.HorizonReserves[i]


model.SystemReserve = Constraint(model.hh_periods, rule=SysReserve)


##Spinning Reserve Requirement
def SpinningReq(model, i):
    return sum(model.srsv[j, i] for j in model.ResGenerators) >= model.spin_margin * model.HorizonReserves[i]


model.SpinReq = Constraint(model.hh_periods, rule=SpinningReq)


##Spinning reserve can only be offered by units that are online
def SpinningReq2(model, j, i):
    return model.srsv[j, i] <= model.on[j, i] * model.maxcap[j] * model.deratef[j]


model.SpinReq2 = Constraint(model.Generators, model.hh_periods, rule=SpinningReq2)


##Non-Spinning reserve can only be offered by units that are offline
def NonSpinningReq(model, j, i):
    return model.nrsv[j, i] <= (1 - model.on[j, i]) * model.maxcap[j] * model.deratef[j]


model.NonSpinReq = Constraint(model.Generators, model.hh_periods, rule=NonSpinningReq)


######========== Zero Sum Constraint =========#############
def ZeroSum(model, j, i):
    return model.mwh[j, i] + model.srsv[j, i] + model.nrsv[j, i] <= model.maxcap[j]


model.ZeroSumConstraint = Constraint(model.Generators, model.hh_periods, rule=ZeroSum)

#### ****** #####
#Access duals
# model.dual = Suffix(direction=Suffix.IMPORT)

######======================================#############
######==========        End        =========#############
######=======================================############
