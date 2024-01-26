#!/usr/bin/env python

import sys, os
import pandas as pd

pd.options.mode.chained_assignment = None
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from model.pownet_model import model

# e.g.,python pownet_multiyear_run.py "test 0, 332 333 334 335 336, KA CB BG DG MN, 3, 7"

# set solution
args = [x.strip() for x in sys.argv[1].split(",")]
print(args)
solution = args[0].split()
cols = [int(x) for x in args[4].split()]
names = args[1].split()

year = int(args[2])
month = int(args[3])
print(solution, year, month)


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


instance = init_pownet(
    model.create_instance("./model/input/pownet_SAPP_24hr.dat")
)

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
sample_df = pd.DataFrame()

h = 24
k = range(1, h + 1)
n = 365 * year
doy = 0

for m in range(month - 1):
    doy += days_in_month[m]

for dom in np.arange(0, days_in_month[month - 1])[:]:
    print("day: " + str(dom + 1))
    # set load
    for d in instance.d_nodes:
        # load Demand and Reserve time series data
        for i in k:
            instance.HorizonDemand[d, i] = load.iloc[doy * 24 + i - 1][d]
            instance.HorizonReserves[i] = load.iloc[doy * 24 + i - 1].sum() * 0.15

    # set hydro
    for hp in h_index.keys():
        hp_col = h_index[hp]
        max24 = hydro.iloc[year * 12 + month - 1][hp_col]
        if hp_col == "KA":
            max24 = max24 / 2
        instance.HorizonHydroMax[hp] = max24

    # set solar
    for i in k:
        instance.HorizonSolar["KARIBA_s", i] = ka_solar_obs.iloc[(n * 24) + i].values[0]
        instance.HorizonSolar["CAHORA_s", i] = cb_solar_obs.iloc[(n * 24) + i].values[0]
        instance.HorizonSolar["MOZ_MPHANDA_s", i] = mn_solar_obs.iloc[
            (n * 24) + i
        ].values[0]
        instance.HorizonSolar["BATOKA.GO_s", i] = bg_solar_obs.iloc[
            (n * 24) + i
        ].values[0]
        instance.HorizonSolar["DEVIL.GO_s", i] = dg_solar_obs.iloc[(n * 24) + i].values[
            0
        ]

    # solve
    results = opt.solve(instance, load_solutions=False)

    # results
    if results.solver.termination_condition == pyo.TerminationCondition.infeasible:
        print("infeasible")
        break
    else:
        instance.solutions.load_from(results)

        obj = pyo.value(instance.SystemCost)

        line_slacks = []
        for i in instance.MaxLineConstraint:
            line_slacks.append(instance.MaxLineConstraint[i].uslack())

        itt_hydro = [
            pyo.value(instance.hydro["ZAM_ITE.TEZ", i]) for i in instance.hh_periods
        ]

        kgu_hydro = [
            pyo.value(instance.hydro["ZAM_KAF.GO.U", i]) for i in instance.hh_periods
        ]

        kgl_hydro = [
            pyo.value(instance.hydro["ZAM_KAF.GO.L", i]) for i in instance.hh_periods
        ]

        vic_hydro = [
            pyo.value(instance.hydro["ZAM_VICTORIA", i]) for i in instance.hh_periods
        ]

        chi_hydro = [
            pyo.value(instance.hydro["MOZ_CHI", i]) for i in instance.hh_periods
        ]

        cor_hydro = [
            pyo.value(instance.hydro["MOZ_COR", i]) for i in instance.hh_periods
        ]

        ka_n_hydro = [
            pyo.value(instance.hydro["ZAM_KARIBA", i]) for i in instance.hh_periods
        ]

        ka_s_hydro = [
            pyo.value(instance.hydro["ZIM_KARIBA", i]) for i in instance.hh_periods
        ]

        ka_solar = [
            pyo.value(instance.solar["KARIBA_s", i]) for i in instance.hh_periods
        ]

        bg_n_hydro = [
            pyo.value(instance.hydro["ZAM_BATOKA.GO", i]) for i in instance.hh_periods
        ]

        bg_s_hydro = [
            pyo.value(instance.hydro["ZIM_BATOKA.GO", i]) for i in instance.hh_periods
        ]

        bg_solar = [
            pyo.value(instance.solar["BATOKA.GO_s", i]) for i in instance.hh_periods
        ]

        dg_n_hydro = [
            pyo.value(instance.hydro["ZAM_DEVIL.GO", i]) for i in instance.hh_periods
        ]

        dg_s_hydro = [
            pyo.value(instance.hydro["ZIM_DEVIL.GO", i]) for i in instance.hh_periods
        ]

        dg_solar = [
            pyo.value(instance.solar["DEVIL.GO_s", i]) for i in instance.hh_periods
        ]

        cb_hydro = [
            pyo.value(instance.hydro["MOZ_CAH.BAS", i]) for i in instance.hh_periods
        ]

        cb_solar = [
            pyo.value(instance.solar["CAHORA_s", i]) for i in instance.hh_periods
        ]

        mn_hydro = [
            pyo.value(instance.hydro["MOZ_MPHANDA", i]) for i in instance.hh_periods
        ]

        mn_solar = [
            pyo.value(instance.solar["MOZ_MPHANDA_s", i]) for i in instance.hh_periods
        ]

        total_ZAM_oil = []
        for i in instance.hh_periods:
            total_ZAM_oil.append(
                pyo.value(sum(instance.mwh[j, i] for j in ["ZAM_oil"]))
            )

        total_NAM_oil = []
        for i in instance.hh_periods:
            total_NAM_oil.append(
                pyo.value(sum(instance.mwh[j, i] for j in ["NAM_oil"]))
            )

        total_MAL_gas = []
        for i in instance.hh_periods:
            total_MAL_gas.append(
                pyo.value(sum(instance.mwh[j, i] for j in ["MAL_gas"]))
            )
        total_ZAM_coal = []
        for i in instance.hh_periods:
            total_ZAM_coal.append(
                pyo.value(sum(instance.mwh[j, i] for j in ["ZAM_coal"]))
            )
        total_ZIM_coal = []
        for i in instance.hh_periods:
            total_ZIM_coal.append(
                pyo.value(sum(instance.mwh[j, i] for j in ["ZIM_Coal"]))
            )
        total_NAM_coal = []
        for i in instance.hh_periods:
            total_NAM_coal.append(
                pyo.value(sum(instance.mwh[j, i] for j in ["NAM_coal"]))
            )
        total_BTSW_coal = []
        for i in instance.hh_periods:
            total_BTSW_coal.append(
                pyo.value(sum(instance.mwh[j, i] for j in ["BTSW_coal"]))
            )
        total_ZAM_biomass = []
        for i in instance.hh_periods:
            total_ZAM_biomass.append(
                pyo.value(sum(instance.mwh[j, i] for j in ["ZAM_biomass"]))
            )
        total_MAL_biomass = []
        for i in instance.hh_periods:
            total_MAL_biomass.append(
                pyo.value(sum(instance.mwh[j, i] for j in ["MAL_biomass"]))
            )

        coal_st = []
        for i in instance.hh_periods:
            coal_st.append(pyo.value(sum(instance.mwh[j, i] for j in instance.Coal_st)))

        oil_ic = []
        for i in instance.hh_periods:
            oil_ic.append(pyo.value(sum(instance.mwh[j, i] for j in instance.Oil_ic)))

        biomass_st = []
        for i in instance.hh_periods:
            biomass_st.append(
                pyo.value(sum(instance.mwh[j, i] for j in instance.Biomass_st))
            )

        gas_st = []
        for i in instance.hh_periods:
            gas_st.append(pyo.value(sum(instance.mwh[j, i] for j in instance.Gas_st)))

        nuclear = []
        for i in instance.hh_periods:
            nuclear.append(pyo.value(sum(instance.mwh[j, i] for j in instance.Nuclear)))

        solar_dis = []
        for i in instance.hh_periods:
            solar_dis.append(
                pyo.value(sum(instance.solar[j, i] for j in instance.s_nodes))
            )

        wind_dis = []
        for i in instance.hh_periods:
            wind_dis.append(
                pyo.value(sum(instance.wind[j, i] for j in instance.w_nodes))
            )

        hydro_dis = []
        for i in instance.hh_periods:
            hydro_dis.append(
                pyo.value(sum(instance.hydro[j, i] for j in instance.h_nodes))
            )

        infinite_solar_penalty = [
            pyo.value(instance.solar["INF_SOURCE_s", i]) * 1e6
            for i in instance.hh_periods
        ]

        infinite_solar = [
            pyo.value(instance.solar["INF_SOURCE_s", i]) for i in instance.hh_periods
        ]

        df = pd.DataFrame(
            {
                "year": np.repeat(year, 24),
                "month": np.repeat(month, 24),
                "dom": np.repeat(dom, 24),
                "hour": np.arange(1, 24 + 1),
                "obj": np.repeat(obj, 24),
                "infinite_solar_penalty": infinite_solar_penalty,
                "infinite_solar": infinite_solar,
                "load": load.iloc[doy * 24 : doy * 24 + 24].sum(axis=1).to_list(),
                "coal": coal_st,
                "oil": oil_ic,
                "gas": gas_st,
                "nuclear": nuclear,
                "biomass": biomass_st,
                "wind": wind_dis,
                "solar": solar_dis,
                "hydro": hydro_dis,
                "BTSW_load": load.iloc[doy * 24 : doy * 24 + 24][["BTSW"]]
                .sum(axis=1)
                .to_list(),
                "MAL_load": load.iloc[doy * 24 : doy * 24 + 24][["MAL"]]
                .sum(axis=1)
                .to_list(),
                "MOZ_load": load.iloc[doy * 24 : doy * 24 + 24][["MOZ"]]
                .sum(axis=1)
                .to_list(),
                "NAM_load": load.iloc[doy * 24 : doy * 24 + 24][["NAM"]]
                .sum(axis=1)
                .to_list(),
                "ZAM_load": load.iloc[doy * 24 : doy * 24 + 24][["ZAM"]]
                .sum(axis=1)
                .to_list(),
                "ZIM_load": load.iloc[doy * 24 : doy * 24 + 24][["ZIM"]]
                .sum(axis=1)
                .to_list(),
                "zam_oil": total_ZAM_oil,
                "nam_oil": total_NAM_oil,
                "mal_gas": total_MAL_gas,
                "zam_coal": total_ZAM_coal,
                "zim_coal": total_ZIM_coal,
                "nam_coal": total_NAM_coal,
                "btsw_coal": total_BTSW_coal,
                "zam_biomass": total_ZAM_biomass,
                "mal_biomass": total_MAL_biomass,
                "itt_hydro": itt_hydro,
                "kgu_hydro": kgu_hydro,
                "kgl_hydro": kgl_hydro,
                "vic_hydro": vic_hydro,
                "chi_hydro": chi_hydro,
                "cor_hydro": cor_hydro,
                "ka_n_hydro": ka_n_hydro,
                "ka_s_hydro": ka_s_hydro,
                "ka_solar": ka_solar,
                "bg_n_hydro": bg_n_hydro,
                "bg_s_hydro": bg_s_hydro,
                "bg_solar": bg_solar,
                "dg_n_hydro": dg_n_hydro,
                "dg_s_hydro": dg_s_hydro,
                "dg_solar": dg_solar,
                "cb_hydro": cb_hydro,
                "cb_solar": cb_solar,
                "mn_hydro": mn_hydro,
                "mn_solar": mn_solar,
            }
        )

        sample_df = pd.concat([sample_df, df])

        for z in instance.Generators:
            instance.ini_on[z] = pyo.value(instance.on[z, 24])

    doy += 1
    n += 1

sample_df.insert(
    12,
    "zam_hydro",
    value=sample_df["itt_hydro"]
    + sample_df["kgu_hydro"]
    + sample_df["kgl_hydro"]
    + sample_df["ka_n_hydro"]
    + sample_df["bg_n_hydro"]
    + sample_df["dg_n_hydro"]
    + sample_df["vic_hydro"],
)

sample_df.insert(
    12,
    "zam_balance",
    value=sample_df["zam_load"]
    - sample_df["zam_hydro"]
    - sample_df["zam_coal"]
    - sample_df["zam_oil"]
    - sample_df["zam_biomass"],
)

sample_df.insert(
    12,
    "zim_hydro",
    value=sample_df["ka_s_hydro"] + sample_df["bg_s_hydro"] + +sample_df["dg_s_hydro"],
)

sample_df.insert(
    12,
    "zim_balance",
    value=sample_df["zim_load"] - sample_df["zim_hydro"] - sample_df["zim_coal"],
)

sample_df.insert(
    12,
    "moz_hydro",
    value=sample_df["cb_hydro"]
    + sample_df["mn_hydro"]
    + sample_df["chi_hydro"]
    + sample_df["cor_hydro"],
)

sample_df.insert(
    12,
    "moz_balance",
    value=sample_df["moz_load"] - sample_df["moz_hydro"] - sample_df["zim_coal"],
)

sample_df.insert(
    12,
    "upper_solar",
    value=sample_df["ka_solar"] + sample_df["dg_solar"] + sample_df["bg_solar"],
)
sample_df.insert(12, "lower_solar", value=sample_df["cb_solar"] + sample_df["mn_solar"])

sample_df.to_csv(
    "./output/sim/{}_{}_hourly_{}_{}.csv".format(solution[0], solution[1], year, month),
    index=False,
)
