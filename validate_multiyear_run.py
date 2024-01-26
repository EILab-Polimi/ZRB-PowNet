#!/usr/bin/env python
import sys, os
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from model.pownet_model import model

# e.g.,python pownet_multiyear_run.py "test 0, 332 333 334 335 336, KA CB BG DG MN"

# set solution
args = [x.strip() for x in sys.argv[1].split(",")]
print(args)
solution = args[0].split()
cols = [int(x) for x in args[1].split()]
names = args[2].split()

np.random.seed(2)


# model
def init_pownet(instance):
    h = instance.HorizonHours
    k = range(1, h + 1)

    # Initialize demands and renewables
    for z in instance.d_nodes:
        # load Demand and Reserve time series data
        for i in k:
            instance.HorizonDemand[z, i] = instance.SimDemand[z, i]
            instance.HorizonReserves[i] = instance.SimReserves[i]

    for z in instance.h_nodes:
        # load Hydropower time series data
        for i in k:
            instance.HorizonHydro[z, i] = instance.SimHydro[z, i]

    for z in instance.s_nodes:
        # load Solar time series data
        for i in k:
            instance.HorizonSolar[z, i] = instance.SimSolar[z, i]

    for z in instance.w_nodes:
        # load Wind time series data
        for i in k:
            instance.HorizonWind[z, i] = instance.SimWind[z, i]

    return instance


instance = init_pownet(model.create_instance("./model/input/pownet_SAPP_24hr.dat"))

opt = SolverFactory("cplex_direct")

h_max24 = {
    "ITT": 5194,
    "KGU": 20963,
    "KGL": 18434,
    "KA": 25741,
    "CB": 49719,
    "MN": 31281,
    "DG": 32487,
    "BG": 40318,
}

h_index = dict(
    [
        ("MOZ_CAH.BAS", "CB"),
        ("ZAM_ITE.TEZ", "ITT"),
        ("ZAM_KAF.GO.L", "KGL"),
        ("ZAM_KAF.GO.U", "KGU"),
        ("ZAM_KARIBA", "KA"),
        ("ZIM_KARIBA", "KA"),
        ("ZAM_BATOKA.GO", "BG"),
        ("ZIM_BATOKA.GO", "BG"),
        ("ZAM_DEVIL.GO", "DG"),
        ("ZIM_DEVIL.GO", "DG"),
        ("MOZ_MPHANDA", "MN"),
    ]
)

days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

solar_k = {"KA": 0, "CB": 0, "BG": 0, "DG": 0, "MN": 0}

solar_system_losses = 0.14

# load solar decisions
zrb_policies = pd.read_csv(
    os.path.join("./data/zrb_emodps_policies", solution[0] + ".txt"),
    header=None,
    sep="\s+",
    usecols=cols,
    names=names,
)
for col in zrb_policies.columns:
    solar_k[col] = zrb_policies.iloc[int(solution[1])][col]
print("Solar Capacities:", solar_k)

zrb_sim = pd.read_csv(
    os.path.join("./data/zrb_emodps_sim", solution[0] + ".sim"),
    header=None,
    usecols=[0, 25, 26, 27, 28, 29, 30, 31, 32],
    names=["idx", "ITT", "KGU", "KGL", "KA", "CB", "BG", "DG", "MN"],
)
hydro = zrb_sim.loc[zrb_sim.idx == int(solution[1]) + 1]
hydro.insert(1, "month", value=np.tile(days_in_month, int(len(hydro) / 12)))
for col in hydro.columns[2:]:
    hydro[col] = hydro.apply(lambda x: x[col] * 1e6 / 12 / x["month"], axis=1)

# Solar potentials
ka_solar_potential = np.loadtxt(
    "./data/pvgis_1kwp/KA_2529Wp_-16.627_28.763_SA2_crystSi_0_22deg_-179deg_2005_2020.csv",
    skiprows=11,
    delimiter=",",
    usecols=[1],
    max_rows=140257,
)
ka_solar_potential = np.concatenate((ka_solar_potential, ka_solar_potential))
ka_solar_potential = pd.DataFrame(ka_solar_potential)
ka_solar_obs = ka_solar_potential * solar_k["KA"] * (1 - solar_system_losses)

bgdg_solar_potential = np.loadtxt(
    "./data/pvgis_1kwp/BGDG_1839Wp_-17.860_25.498_SA2_1kWp_crystSi_0_21deg_-131deg_2005_2020.csv",
    skiprows=11,
    delimiter=",",
    usecols=[1],
    max_rows=140257,
)
bgdg_solar_potential = np.concatenate((bgdg_solar_potential, bgdg_solar_potential))
bgdg_solar_potential = pd.DataFrame(bgdg_solar_potential)
bg_solar_obs = bgdg_solar_potential * solar_k["BG"] * (1 - solar_system_losses)
dg_solar_obs = bgdg_solar_potential * solar_k["DG"] * (1 - solar_system_losses)

cb_solar_potential = np.loadtxt(
    "./data/pvgis_1kwp/CB_2050Wp_-15.624_32.280_SA2_crystSi_0_20deg_-179deg_2005_2020.csv",
    skiprows=11,
    delimiter=",",
    usecols=[1],
    max_rows=140257,
)
cb_solar_potential = np.concatenate((cb_solar_potential, cb_solar_potential))
cb_solar_potential = pd.DataFrame(cb_solar_potential)
cb_solar_obs = cb_solar_potential * solar_k["CB"] * (1 - solar_system_losses)

mn_solar_potential = np.loadtxt(
    "./data/pvgis_1kwp/MN_1864Wp_-15.917_33.320_SA2_1kWp_crystSi_0_19deg_-179deg_2005_2020.csv",
    skiprows=11,
    delimiter=",",
    usecols=[1],
    max_rows=140257,
)
mn_solar_potential = np.concatenate((mn_solar_potential, mn_solar_potential))
mn_solar_potential = pd.DataFrame(mn_solar_potential)
mn_solar_obs = mn_solar_potential * solar_k["MN"] * (1 - solar_system_losses)

# load curve
load = pd.read_csv("./data/load/load_2016.csv")

# base hydro max
df_hydro_24_max = pd.read_csv("./model/input/hydro_day_limit.csv", index_col="name")

# peak solar curve
df_solar = pd.read_csv("./model/input/solar.csv", header=0)

# simulation
sample_df = pd.DataFrame(
    columns=[
        "obj",
        "cost",
        "deficit",
        "load",
        "zrb_load",
        "hydro",
        "solar",
        "wind",
        "coal",
        "oil",
        "biomass",
        "gas",
        "nuclear",
        "zrb_hydro",
        "zrb_solar",
        "zrb_oil",
        "zrb_gas",
        "zrb_coal",
        "zrb_biomass",
        "itt_hydro",
        "kgu_hydro",
        "kgl_hydro",
        "vic_hydro",
        "ka_n_hydro",
        "ka_s_hydro",
        "ka_solar",
        "bg_n_hydro",
        "bg_s_hydro",
        "bg_solar",
        "dg_n_hydro",
        "dg_s_hydro",
        "dg_solar",
        "cb_hydro",
        "cb_solar",
        "mn_hydro",
        "mn_solar",
    ]
)

yrs = int(len(hydro) / 12)
h = 24
k = range(1, h + 1)

n = 0
for y in range(yrs):
    print(solution[0], solution[1], ":", y + 1)
    # doy_max = 0
    doy_init = 7  # start on 7th day of each month
    for m in range(12):
        doy = doy_init + (m * 30)
        for dow in np.arange(0, 7):  # simulate 1 week
            # for dom in np.arange(0, days_in_month[m]):

            # set load
            for d in instance.d_nodes:
                # load Demand and Reserve time series data
                for i in k:
                    instance.HorizonDemand[d, i] = load.iloc[doy * 24 + i - 1][d]
                    instance.HorizonReserves[i] = (
                        load.iloc[doy * 24 + i - 1].sum() * 0.15
                    )

            # set hydro
            for hp in h_index.keys():
                hp_col = h_index[hp]
                max24 = hydro.iloc[y * 12 + m][hp_col]
                if hp_col in ["KA", "BG", "DG"]:
                    max24 = max24 / 2
                instance.HorizonHydroMax[hp] = max24

            # set solar
            for i in k:
                instance.HorizonSolar["KARIBA_s", i] = ka_solar_obs.iloc[
                    (((y * 365) + doy) * 24) + i
                ].values[0]
                instance.HorizonSolar["CAHORA_s", i] = cb_solar_obs.iloc[
                    (((y * 365) + doy) * 24) + i
                ].values[0]
                instance.HorizonSolar["MOZ_MPHANDA_s", i] = mn_solar_obs.iloc[
                    (((y * 365) + doy) * 24) + i
                ].values[0]
                instance.HorizonSolar["BATOKA.GO_s", i] = bg_solar_obs.iloc[
                    (((y * 365) + doy) * 24) + i
                ].values[0]
                instance.HorizonSolar["DEVIL.GO_s", i] = dg_solar_obs.iloc[
                    (((y * 365) + doy) * 24) + i
                ].values[0]

            # solve
            results = opt.solve(instance, load_solutions=False)

            # results
            if (
                results.solver.termination_condition
                == pyo.TerminationCondition.infeasible
            ):
                print("infeasible")
                break
            else:
                instance.solutions.load_from(results)

                obj = pyo.value(instance.SystemCost)

                line_slacks = []
                for i in instance.MaxLineConstraint:
                    line_slacks.append(instance.MaxLineConstraint[i].uslack())
                n1active = line_slacks.count(0)

                total_hydro = pyo.value(
                    sum(
                        instance.hydro[j, i]
                        for i in instance.hh_periods
                        for j in instance.h_nodes
                    )
                )

                itt_hydro = pyo.value(
                    sum(
                        instance.hydro[j, i]
                        for i in instance.hh_periods
                        for j in ["ZAM_ITE.TEZ"]
                    )
                )

                kgu_hydro = pyo.value(
                    sum(
                        instance.hydro[j, i]
                        for i in instance.hh_periods
                        for j in ["ZAM_KAF.GO.U"]
                    )
                )

                kgl_hydro = pyo.value(
                    sum(
                        instance.hydro[j, i]
                        for i in instance.hh_periods
                        for j in ["ZAM_KAF.GO.L"]
                    )
                )

                vic_hydro = pyo.value(
                    sum(
                        instance.hydro[j, i]
                        for i in instance.hh_periods
                        for j in ["ZAM_VICTORIA"]
                    )
                )

                ka_n_hydro = pyo.value(
                    sum(
                        instance.hydro[j, i]
                        for i in instance.hh_periods
                        for j in ["ZAM_KARIBA"]
                    )
                )

                ka_s_hydro = pyo.value(
                    sum(
                        instance.hydro[j, i]
                        for i in instance.hh_periods
                        for j in ["ZIM_KARIBA"]
                    )
                )

                ka_solar = pyo.value(
                    sum(
                        instance.solar[j, i]
                        for i in instance.hh_periods
                        for j in ["KARIBA_s"]
                    )
                )

                bg_n_hydro = pyo.value(
                    sum(
                        instance.hydro[j, i]
                        for i in instance.hh_periods
                        for j in ["ZAM_BATOKA.GO"]
                    )
                )

                bg_s_hydro = pyo.value(
                    sum(
                        instance.hydro[j, i]
                        for i in instance.hh_periods
                        for j in ["ZIM_BATOKA.GO"]
                    )
                )

                bg_solar = pyo.value(
                    sum(
                        instance.solar[j, i]
                        for i in instance.hh_periods
                        for j in ["BATOKA.GO_s"]
                    )
                )

                dg_n_hydro = pyo.value(
                    sum(
                        instance.hydro[j, i]
                        for i in instance.hh_periods
                        for j in ["ZAM_DEVIL.GO"]
                    )
                )

                dg_s_hydro = pyo.value(
                    sum(
                        instance.hydro[j, i]
                        for i in instance.hh_periods
                        for j in ["ZIM_DEVIL.GO"]
                    )
                )

                dg_solar = pyo.value(
                    sum(
                        instance.solar[j, i]
                        for i in instance.hh_periods
                        for j in ["DEVIL.GO_s"]
                    )
                )

                cb_hydro = pyo.value(
                    sum(
                        instance.hydro[j, i]
                        for i in instance.hh_periods
                        for j in ["MOZ_CAH.BAS"]
                    )
                )

                cb_solar = pyo.value(
                    sum(
                        instance.solar[j, i]
                        for i in instance.hh_periods
                        for j in ["CAHORA_s"]
                    )
                )

                mn_hydro = pyo.value(
                    sum(
                        instance.hydro[j, i]
                        for i in instance.hh_periods
                        for j in ["MOZ_MPHANDA"]
                    )
                )

                mn_solar = pyo.value(
                    sum(
                        instance.solar[j, i]
                        for i in instance.hh_periods
                        for j in ["MOZ_MPHANDA_s"]
                    )
                )

                total_solar = pyo.value(
                    sum(
                        instance.solar[j, i]
                        for i in instance.hh_periods
                        for j in instance.s_nodes
                    )
                )

                total_wind = pyo.value(
                    sum(
                        instance.wind[j, i]
                        for i in instance.hh_periods
                        for j in instance.w_nodes
                    )
                )

                total_ZRB_hydro = (
                    itt_hydro
                    + kgu_hydro
                    + kgl_hydro
                    + vic_hydro
                    + ka_n_hydro
                    + ka_s_hydro
                    + bg_n_hydro
                    + bg_s_hydro
                    + dg_n_hydro
                    + dg_s_hydro
                    + cb_hydro
                    + mn_hydro
                )

                total_ZRB_solar = ka_solar + bg_solar + dg_solar + cb_solar + mn_solar

                total_ZRB_oil = pyo.value(
                    sum(
                        instance.mwh[j, i]
                        for i in instance.hh_periods
                        for j in ["ZAM_oil", "NAM_oil"]
                    )
                )

                total_ZRB_gas = pyo.value(
                    sum(
                        instance.mwh[j, i]
                        for i in instance.hh_periods
                        for j in ["MAL_gas"]
                    )
                )

                total_ZRB_coal = pyo.value(
                    sum(
                        instance.mwh[j, i]
                        for i in instance.hh_periods
                        for j in ["ZAM_coal", "ZIM_Coal", "NAM_coal", "BTSW_coal"]
                    )
                )

                total_ZRB_biomass = pyo.value(
                    sum(
                        instance.mwh[j, i]
                        for i in instance.hh_periods
                        for j in ["ZAM_biomass", "MAL_biomass"]
                    )
                )

                coal_st = pyo.value(
                    sum(
                        instance.mwh[j, i]
                        for i in instance.hh_periods
                        for j in instance.Coal_st
                    )
                )
                oil_ic = (
                    pyo.value(
                        sum(
                            instance.mwh[j, i]
                            for i in instance.hh_periods
                            for j in instance.Oil_ic
                        )
                    )
                    / 24
                )
                biomass_st = pyo.value(
                    sum(
                        instance.mwh[j, i]
                        for i in instance.hh_periods
                        for j in instance.Biomass_st
                    )
                )
                gas_st = (
                    pyo.value(
                        sum(
                            instance.mwh[j, i]
                            for i in instance.hh_periods
                            for j in instance.Gas_st
                        )
                    )
                    / 24
                )
                nuclear = pyo.value(
                    sum(
                        instance.mwh[j, i]
                        for i in instance.hh_periods
                        for j in instance.Nuclear
                    )
                )

                infinite_solar_penalty = pyo.value(
                    sum(
                        instance.solar["INF_SOURCE_s", i] * 1e6
                        for i in instance.hh_periods
                    )
                )
                infinite_solar = pyo.value(
                    sum(instance.solar["INF_SOURCE_s", i] for i in instance.hh_periods)
                )

                total_solar = total_solar - infinite_solar

                sample_df.loc[n] = [
                    obj,
                    obj - infinite_solar_penalty,
                    infinite_solar,
                    load.iloc[doy * 24 : doy * 24 + 24].sum().sum(),
                    load.iloc[doy * 24 : doy * 24 + 24][
                        ["BTSW", "MAL", "MOZ", "NAM", "ZAM", "ZIM"]
                    ]
                    .sum()
                    .sum(),
                    total_hydro,
                    total_solar,
                    total_wind,
                    coal_st,
                    oil_ic,
                    biomass_st,
                    gas_st,
                    nuclear,
                    total_ZRB_hydro,
                    total_ZRB_solar,
                    total_ZRB_oil,
                    total_ZRB_gas,
                    total_ZRB_coal,
                    total_ZRB_biomass,
                    itt_hydro,
                    kgu_hydro,
                    kgl_hydro,
                    vic_hydro,
                    ka_n_hydro,
                    ka_s_hydro,
                    ka_solar,
                    bg_n_hydro,
                    bg_s_hydro,
                    bg_solar,
                    dg_n_hydro,
                    dg_s_hydro,
                    dg_solar,
                    cb_hydro,
                    cb_solar,
                    mn_hydro,
                    mn_solar,
                ]

                for z in instance.Generators:
                    switch = max(0, int(pyo.value(instance.on[z, 24])))
                    instance.ini_on[z] = int(switch)

            doy += 1
            n += 1

sample_df.to_csv("./output/sim/" + solution[0] + "_" + solution[1] + ".csv")
