import os

import pandas as pd
import numpy as np

######=================================================########
######               Segment A.1                       ########
######=================================================########

SimDays = 1
SimHours = SimDays * 24
HorizonHours = 24  ##planning horizon (e.g., 24, 48, 72 hours etc.)
TransLoss = 0.075  ##transmission loss as a percent of generation (0.075)
n1criterion = 0.75  ##maximum line-usage as a percent of line-capacity (0.75)
res_margin = 0.15  ##minimum reserve as a percent of system demand (0.15)
spin_margin = 0.50  ##minimum spinning reserve as a percent of total reserve (0.5)

# todo check Unit cost of generation / import of each fuel type [$/MMbtu]
gen_cost = {'coal_st': 5.2,
            'oil_ic': 6.0,
            'biomass_st': 5.0,
            'gas_st': 5.0,
            'nuclear': 1.0}

data_name = 'pownet_SAPP_24hr'

######=================================================########
######               Segment A.2                       ########
######=================================================########

# read parameters for dispatchable resources (coal/gas/oil/biomass generators, imports)
df_gen = pd.read_csv(os.path.join(os.path.dirname(__file__), 'input/genparams.csv'), header=0)
df_gen['gen_cost'] = df_gen['typ'].map(gen_cost)
df_gen['ini_on'] = 1

# read derate factors of dispatchable units for the simulation year
df_gen_deratef = pd.read_csv(os.path.join(os.path.dirname(__file__), 'input/genparams_deratef.csv'), header=0)
df_gen['deratef'] = df_gen_deratef['deratef']

##maximum hourly ts of dispatchable hydropower at each domestic dam
df_hydro = pd.read_csv(os.path.join(os.path.dirname(__file__), 'input/hydro.csv'), header=0)

##maximum daily dispatchable hydropower at each domestic dam
df_hydro_24_max = pd.read_csv(os.path.join(os.path.dirname(__file__), 'input/hydro_day_limit.csv'), index_col='name')

##hourly ts of dispatchable solar-power at each plant
df_solar = pd.read_csv(os.path.join(os.path.dirname(__file__), 'input/solar.csv'), header=0)

##hourly ts of dispatchable wind-power at each plant
df_wind = pd.read_csv(os.path.join(os.path.dirname(__file__), 'input/wind.csv'), header=0)

##hourly ts of load at substation-level
df_load = pd.read_csv(os.path.join(os.path.dirname(__file__), 'input/load.csv'), header=0)

# capacity and susceptence of each transmission line (one direction)
df_trans1 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'input/transparam.csv'), header=0)

# hourly minimum reserve as a function of load (e.g., 15% of current load)
df_reserves = pd.DataFrame((df_load.iloc[:, 4:].sum(axis=1) * res_margin).values, columns=['Reserve'])

# capacity and susceptence of each transmission line (both directions)
df_trans2 = pd.DataFrame(
    [df_trans1['sink'], df_trans1['source'], df_trans1['linemva'], df_trans1['linesus']]).transpose()
df_trans2.columns = ['source', 'sink', 'linemva', 'linesus']
df_paths = pd.concat([df_trans1, df_trans2], axis=0)
df_paths.index = np.arange(len(df_paths))

######=================================================########
######               Segment A.3                       ########
######=================================================########

####======== Lists of Nodes of the Power System ========########

h_nodes = ['ANG_h', 'TAN_h', 'MAL_h', 'LES_h', 'SWA_h', 'DRC_h', 'NAM_h', 'STAF_h', 'MOZ_CAH.BAS', 'MOZ_CHI', 'MOZ_COR',
           'ZAM_ITE.TEZ', 'ZAM_KAF.GO.L', 'ZAM_KAF.GO.U', 'ZAM_KARIBA', 'ZAM_LUS', 'ZAM_VICTORIA', 'ZIM_KARIBA',
           'ZAM_DEVIL.GO', 'ZIM_DEVIL.GO', 'MOZ_MPHANDA', 'ZAM_BATOKA.GO', 'ZIM_BATOKA.GO']
s_nodes = ['CAHORA_s', 'KARIBA_s', 'BATOKA.GO_s', 'DEVIL.GO_s', 'MOZ_MPHANDA_s', 'INF_SOURCE_s']
w_nodes = []
gd_nodes = ['STAF', 'ZAM', 'ZIM', 'BTSW', 'NAM', 'SWA', 'DRC', 'ANG', 'TAN', 'MAL', 'MOZ', 'LES']
g_nodes = gd_nodes
d_nodes = gd_nodes
all_nodes = h_nodes + gd_nodes + s_nodes + w_nodes

##list of types of dispatchable units
types = ['coal_st', 'oil_ic', 'biomass_st', 'gas_st', 'nuclear']  # gas_cc , oil_st

######=================================================########
######               Segment A.4                       ########
######=================================================########

######====== write data.dat file ======########
with open(os.path.join(os.path.dirname(__file__), 'input', str(data_name) + '.dat'), 'w') as f:
    ###### generator sets by generator nodes
    for z in gd_nodes:
        # node string
        z_int = gd_nodes.index(z)
        f.write('set GD%dGens :=\n' % (z_int + 1))
        # pull relevant generators
        for gen in range(0, len(df_gen)):
            if df_gen.loc[gen, 'node'] == z:
                unit_name = df_gen.loc[gen, 'name']
                unit_name = unit_name.replace(' ', '_')
                f.write(unit_name + ' ')
        f.write(';\n\n')

    ####### generator sets by type
    # Coal
    f.write('set Coal_st :=\n')
    # pull relevant generators
    for gen in range(0, len(df_gen)):
        if df_gen.loc[gen, 'typ'] == 'coal_st':
            unit_name = df_gen.loc[gen, 'name']
            unit_name = unit_name.replace(' ', '_')
            f.write(unit_name + ' ')
    f.write(';\n\n')

    # Oil_ic
    f.write('set Oil_ic :=\n')
    # pull relevant generators
    for gen in range(0, len(df_gen)):
        if df_gen.loc[gen, 'typ'] == 'oil_ic':
            unit_name = df_gen.loc[gen, 'name']
            unit_name = unit_name.replace(' ', '_')
            f.write(unit_name + ' ')
    f.write(';\n\n')

    # Biomass
    f.write('set Biomass_st :=\n')
    # pull relevant generators
    for gen in range(0, len(df_gen)):
        if df_gen.loc[gen, 'typ'] == 'biomass_st':
            unit_name = df_gen.loc[gen, 'name']
            unit_name = unit_name.replace(' ', '_')
            f.write(unit_name + ' ')
    f.write(';\n\n')

    # Gas_st
    f.write('set Gas_st :=\n')
    # pull relevant generators
    for gen in range(0, len(df_gen)):
        if df_gen.loc[gen, 'typ'] == 'gas_st':
            unit_name = df_gen.loc[gen, 'name']
            unit_name = unit_name.replace(' ', '_')
            f.write(unit_name + ' ')
    f.write(';\n\n')

    # Nuclear
    f.write('set Nuclear :=\n')
    # pull relevant generators
    for gen in range(0, len(df_gen)):
        if df_gen.loc[gen, 'typ'] == 'nuclear':
            unit_name = df_gen.loc[gen, 'name']
            unit_name = unit_name.replace(' ', '_')
            f.write(unit_name + ' ')
    f.write(';\n\n')


    ######=================================================########
    ######               Segment A.5                       ########
    ######=================================================########

    ######Set nodes, sources and sinks
    # nodes
    f.write('set nodes :=\n')
    for z in all_nodes:
        f.write(z + ' ')
    f.write(';\n\n')

    # sources
    f.write('set sources :=\n')
    for z in all_nodes:
        f.write(z + ' ')
    f.write(';\n\n')

    # sinks
    f.write('set sinks :=\n')
    for z in all_nodes:
        f.write(z + ' ')
    f.write(';\n\n')

    # hydro_nodes
    f.write('set h_nodes :=\n')
    for z in h_nodes:
        f.write(z + ' ')
    f.write(';\n\n')

    # solar_nodes
    f.write('set s_nodes :=\n')
    for z in s_nodes:
        f.write(z + ' ')
    f.write(';\n\n')

    # wind_nodes
    f.write('set w_nodes :=\n')
    for z in w_nodes:
        f.write(z + ' ')
    f.write(';\n\n')

    # all demand nodes
    f.write('set d_nodes :=\n')
    for z in d_nodes:
        f.write(z + ' ')
    f.write(';\n\n')

    # generator with demand nodes
    f.write('set gd_nodes :=\n')
    for z in gd_nodes:
        f.write(z + ' ')
    f.write(';\n\n')

    ######=================================================########
    ######               Segment A.6                       ########
    ######=================================================########

    ####### simulation period and horizon
    f.write('param SimHours := %d;' % SimHours)
    f.write('\n')
    f.write('param SimDays:= %d;' % SimDays)
    f.write('\n\n')
    f.write('param HorizonHours := %d;' % HorizonHours)
    f.write('\n\n')
    f.write('param TransLoss := %0.3f;' % TransLoss)
    f.write('\n\n')
    f.write('param n1criterion := %0.3f;' % n1criterion)
    f.write('\n\n')
    f.write('param spin_margin := %0.3f;' % spin_margin)
    f.write('\n\n')

    ######=================================================########
    ######               Segment A.7                       ########
    ######=================================================########

    ####### create parameter matrix for generators
    f.write('param:' + '\t')
    for c in df_gen.columns:
        if c != 'name':
            f.write(c + '\t')
    f.write(':=\n\n')
    for i in range(0, len(df_gen)):
        for c in df_gen.columns:
            if c == 'name':
                unit_name = df_gen.loc[i, 'name']
                unit_name = unit_name.replace(' ', '_')
                f.write(unit_name + '\t')
            else:
                f.write(str((df_gen.loc[i, c])) + '\t')
        f.write('\n')
    f.write(';\n\n')

    ######=================================================########
    ######               Segment A.8                       ########
    ######=================================================########

    ####### create parameter matrix for transmission paths (source and sink connections)

    f.write('param:' + '\t' + 'linemva' + '\t' + 'linesus :=' + '\n')
    for z in all_nodes:
        for x in all_nodes:
            f.write(z + '\t' + x + '\t')
            match = 0
            for p in range(0, len(df_paths)):
                source = df_paths.loc[p, 'source']
                sink = df_paths.loc[p, 'sink']
                if source == z and sink == x:
                    match = 1
                    p_match = p
            if match > 0:
                f.write(str(df_paths.loc[p_match, 'linemva']) + '\t' + str(df_paths.loc[p_match, 'linesus']) + '\n')
            else:
                f.write('0' + '\t' + '0' + '\n')
    f.write(';\n\n')

    ######=================================================########
    ######               Segment A.9                       ########
    ######=================================================########

    ####### Hourly timeseries (load, hydro, solar, wind, reserve)
    # load (hourly)
    f.write('param:' + '\t' + 'SimDemand:=' + '\n')
    for z in d_nodes:
        for h in range(0, len(df_load)):
            f.write(z + '\t' + str(h + 1) + '\t' + str(df_load.loc[h, z]) + '\n')
    f.write(';\n\n')

    # hydro (max daily)
    f.write('param:' + '\t' + 'HorizonHydroMax:=' + '\n')
    for z in h_nodes:
        f.write(z + '\t' + str(float(df_hydro_24_max.loc[z])) + '\n')
    f.write(';\n\n')

    # hydro (hourly)
    f.write('param:' + '\t' + 'SimHydro:=' + '\n')
    for z in h_nodes:
        for h in range(0, len(df_hydro)):
            f.write(z + '\t' + str(h + 1) + '\t' + str(df_hydro.loc[h, z]) + '\n')
    f.write(';\n\n')

    # solar (hourly)
    f.write('param:' + '\t' + 'SimSolar:=' + '\n')
    for z in s_nodes:
        for h in range(0, len(df_solar)):
            f.write(z + '\t' + str(h + 1) + '\t' + str(df_solar.loc[h, z]) + '\n')
    f.write(';\n\n')

    # wind (hourly)
    f.write('param:' + '\t' + 'SimWind:=' + '\n')
    for z in w_nodes:
        for h in range(0, len(df_wind)):
            f.write(z + '\t' + str(h + 1) + '\t' + str(df_wind.loc[h, z]) + '\n')
    f.write(';\n\n')

    ###### System-wide hourly reserve
    f.write('param' + '\t' + 'SimReserves:=' + '\n')
    for h in range(0, len(df_load)):
        f.write(str(h + 1) + '\t' + str(df_reserves.loc[h, 'Reserve']) + '\n')
    f.write(';\n\n')

    print('Complete:', data_name)
