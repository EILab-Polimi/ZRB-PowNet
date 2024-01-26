# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mat
mat.rcParams["figure.dpi"] = 300

from model.pownet_model import model
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# %% pownet functions
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

def pownet_sample(
    r_labels,
    X,
    X_h_index,
    hydro_24_max,
    X_s_index,
    solar_24,
    instance,
    opt,
    printout=10,
):
    sample_df = pd.DataFrame(
        columns=r_labels
        + [
            "obj",
            "cost",
            "deficit",
            "n1active",
            "hydro",
            "solar",
            "wind",
            "coal",
            "oil",
            "biomass",
            "gas",
            "nuclear",
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

    h = instance.HorizonHours
    k = range(1, h + 1)

    for d in instance.d_nodes:
        # load Demand and Reserve time series data
        for i in k:
            instance.HorizonDemand[d, i] = instance.HorizonDemand[d, i] * 0.8
            instance.HorizonReserves[i] = instance.HorizonDemand[d, i] * 0.15

    for n, x in enumerate(X):
        if n % printout == 0:
            print("Processed through: ", n)

        # set hydro
        for hp in X_h_index:
            instance.HorizonHydroMax[hp] = (
                hydro_24_max.loc[hp]["max24prod"] * x[X_h_index[hp]]
            )

        # set solar
        for s in X_s_index:
            for i in k:
                instance.HorizonSolar[s, i] = solar_24.loc[i - 1, s] * x[X_s_index[s]]

        # solve
        results = opt.solve(instance, load_solutions=False)

        # results
        if results.solver.termination_condition == pyo.TerminationCondition.infeasible:
            sample_df.loc[n] = list(x) + list(
                np.repeat(0, len(sample_df.columns) - len(r_labels))
            )  # +len(deficit_labels)))

        else:
            instance.solutions.load_from(results)

            obj = pyo.value(instance.SystemCost)

            line_slacks = []
            for i in instance.MaxLineConstraint:
                line_slacks.append(instance.MaxLineConstraint[i].uslack())
            n1active = line_slacks.count(0)

            hydro = pyo.value(
                sum(
                    instance.hydro[j, i]
                    for i in instance.hh_periods
                    for j in instance.h_nodes
                )
            )

            productions = []
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

            solar = pyo.value(
                sum(
                    instance.solar[j, i]
                    for i in instance.hh_periods
                    for j in instance.s_nodes
                )
            )
            wind = pyo.value(
                sum(
                    instance.wind[j, i]
                    for i in instance.hh_periods
                    for j in instance.w_nodes
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
                    instance.solar["INF_SOURCE_s", i] * 1e6 for i in instance.hh_periods
                )
            )
            infinite_solar = (
                pyo.value(
                    sum(instance.solar["INF_SOURCE_s", i] for i in instance.hh_periods)
                )
                / 24
            )

            sample_df.loc[n] = list(x) + [
                obj,
                obj - infinite_solar_penalty,
                infinite_solar,
                n1active,
                hydro,
                solar,
                wind,
                coal_st,
                oil_ic,
                biomass_st,
                gas_st,
                nuclear,
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
            ]  #  + deficits

    return sample_df

# %% instantiate pownet
instance = init_pownet(model.create_instance("./model/input/pownet_SAPP_24hr.dat"))
opt = SolverFactory("cplex_direct")
opt.options["threads"] = 4

# %% labels
r_labels = [
    "CB",
    "ITT",
    "KGL",
    "KGU",
    "KA",
    "VIC",
    "BG",
    "DG",
    "MN",
    "KA_s",
    "CB_s",
    "BG_s",
    "DG_s",
    "MN_s",
]
n_hydro = 9

# %%  dictionaries of the PowNet id to the column index of the sample array
# HP indexes
X_h_index = dict(
    [
        ("MOZ_CAH.BAS", 0),
        ("ZAM_ITE.TEZ", 1),
        ("ZAM_KAF.GO.L", 2),
        ("ZAM_KAF.GO.U", 3),
        ("ZAM_KARIBA", 4),
        ("ZIM_KARIBA", 4),
        ("ZAM_VICTORIA", 5),
        ("ZAM_BATOKA.GO", 6),
        ("ZIM_BATOKA.GO", 6),
        ("ZAM_DEVIL.GO", 7),
        ("ZIM_DEVIL.GO", 7),
        ("MOZ_MPHANDA", 8),
    ]
)

# solar indexes
X_s_index = dict(
    [
        ("KARIBA_s", 9),
        ("CAHORA_s", 10),
        ("BATOKA.GO_s", 11),
        ("DEVIL.GO_s", 12),
        ("MOZ_MPHANDA_s", 13),
    ]
)

# %% load the maximum generation from HP and solar generators
df_hydro_24_max = pd.read_csv("./model/input/hydro_day_limit.csv", index_col="name")
df_solar = pd.read_csv("./model/input/solar.csv", header=0)

# %% solar multiplier sampling intervals
solar_test = np.arange(0, 2.1, 0.1)
print(solar_test.T)
# print(solar_test.T[int(1.87/0.05)])

# %% transmission line capacity intervals
trans_step = np.arange(200, 3000, 600)
print(trans_step)

# %% hydro multiplier sampling intervals
hydro_mult = [0]

# %% total matrix sampling points
len(solar_test) * len(trans_step)

# %% solar transmision hydro sensitivity pownet sampling for each reservoir
def solar_trans_sensitivity_ka(
    trans_step,
    solar_test,
    hydro_mult,
    n_hydro,
    r_labels,
    X_h_index,
    df_hydro_24_max,
    X_s_index,
    df_solar,
    instance,
    opt,
):
    X_test_solar = np.zeros((len(hydro_mult), len(solar_test), len(r_labels)))

    for i, hm in enumerate(hydro_mult):
        for j, lvl in enumerate(solar_test):
            X_test_solar[i][j][:] = np.repeat(1, n_hydro).tolist() + [lvl, 0, 0, 0, 0]
            X_test_solar[i][j][4] = hm
    X_test_solar = X_test_solar.reshape(
        int(len(solar_test) * len(hydro_mult)), len(r_labels)
    )

    y_test_df_solar = pd.DataFrame()

    y_df = pownet_sample(
        r_labels,
        X_test_solar,
        X_h_index,
        df_hydro_24_max,
        X_s_index,
        df_solar,
        instance,
        opt,
        printout=10
    )
    y_df["trans"] = "Existing"
    y_test_df_solar = y_test_df_solar.append(y_df)

    for i, trans in enumerate(trans_step):
        # print(trans)
        instance.linemva["ZAM_KARIBA", "ZIM_KARIBA"] = trans
        instance.linemva["ZIM_KARIBA", "ZAM_KARIBA"] = trans
        instance.linemva["ZAM_KARIBA", "ZAM"] = trans
        instance.linemva["ZIM_KARIBA", "ZIM"] = trans
        instance.linemva["ZAM", "ZAM_KARIBA"] = trans
        instance.linemva["ZIM", "ZIM_KARIBA"] = trans
        instance.linemva["ZIM_BATOKA.GO", "ZAM_BATOKA.GO"] = 2.0
        instance.linemva["ZAM_BATOKA.GO", "ZIM_BATOKA.GO"] = 2.0
        instance.linemva["ZIM_DEVIL.GO", "ZAM_DEVIL.GO"] = 2.0
        instance.linemva["ZAM_DEVIL.GO", "ZIM_DEVIL.GO"] = 2.0

        y_df = pownet_sample(
            r_labels,
            X_test_solar,
            X_h_index,
            df_hydro_24_max,
            X_s_index,
            df_solar,
            instance,
            opt,
            printout=10
        )
        y_df["trans"] = trans
        y_test_df_solar = y_test_df_solar.append(y_df)

    return y_test_df_solar

def solar_trans_sensitivity_cb(
    trans_step,
    solar_test,
    hydro_mult,
    n_hydro,
    r_labels,
    X_h_index,
    df_hydro_24_max,
    X_s_index,
    df_solar,
    instance,
    opt,
):
    X_test_solar = np.zeros((len(hydro_mult), len(solar_test), len(r_labels)))

    for i, hm in enumerate(hydro_mult):
        for j, lvl in enumerate(solar_test):
            X_test_solar[i][j][:] = np.repeat(1, n_hydro).tolist() + [0, lvl, 0, 0, 0]
            X_test_solar[i][j][0] = hm
    X_test_solar = X_test_solar.reshape(
        int(len(solar_test) * len(hydro_mult)), len(r_labels)
    )

    y_test_df_solar = pd.DataFrame()

    y_df = pownet_sample(
        r_labels,
        X_test_solar,
        X_h_index,
        df_hydro_24_max,
        X_s_index,
        df_solar,
        instance,
        opt,
    )
    y_df["trans"] = "Existing"
    y_test_df_solar = y_test_df_solar.append(y_df)

    for i, trans in enumerate(trans_step):
        instance.linemva["MOZ_CAH.BAS", "MOZ"] = trans
        instance.linemva["MOZ", "MOZ_CAH.BAS"] = trans
        instance.linemva["ZIM_BATOKA.GO", "ZAM_BATOKA.GO"] = 2.0
        instance.linemva["ZAM_BATOKA.GO", "ZIM_BATOKA.GO"] = 2.0
        instance.linemva["ZIM_DEVIL.GO", "ZAM_DEVIL.GO"] = 2.0
        instance.linemva["ZAM_DEVIL.GO", "ZIM_DEVIL.GO"] = 2.0

        y_df = pownet_sample(
            r_labels,
            X_test_solar,
            X_h_index,
            df_hydro_24_max,
            X_s_index,
            df_solar,
            instance,
            opt,
        )
        y_df["trans"] = trans
        y_test_df_solar = y_test_df_solar.append(y_df)

    return y_test_df_solar

def solar_trans_sensitivity_mn(
    trans_step,
    solar_test,
    hydro_mult,
    n_hydro,
    r_labels,
    X_h_index,
    df_hydro_24_max,
    X_s_index,
    df_solar,
    instance,
    opt,
):
    X_test_solar = np.zeros((len(hydro_mult), len(solar_test), len(r_labels)))

    for i, hm in enumerate(hydro_mult):
        for j, lvl in enumerate(solar_test):
            X_test_solar[i][j][:] = np.repeat(1, n_hydro).tolist() + [0, 0, 0, 0, lvl]
            X_test_solar[i][j][8] = hm
    X_test_solar = X_test_solar.reshape(
        int(len(solar_test) * len(hydro_mult)), len(r_labels)
    )

    y_test_df_solar = pd.DataFrame()

    y_df = pownet_sample(
        r_labels,
        X_test_solar,
        X_h_index,
        df_hydro_24_max,
        X_s_index,
        df_solar,
        instance,
        opt,
    )
    y_df["trans"] = "Existing"
    y_test_df_solar = y_test_df_solar.append(y_df)

    for i, trans in enumerate(trans_step):
        instance.linemva["MOZ_MPHANDA", "MOZ"] = trans
        instance.linemva["MOZ", "MOZ_MPHANDA"] = trans
        instance.linemva["ZIM_BATOKA.GO", "ZAM_BATOKA.GO"] = 2.0
        instance.linemva["ZAM_BATOKA.GO", "ZIM_BATOKA.GO"] = 2.0
        instance.linemva["ZIM_DEVIL.GO", "ZAM_DEVIL.GO"] = 2.0
        instance.linemva["ZAM_DEVIL.GO", "ZIM_DEVIL.GO"] = 2.0

        y_df = pownet_sample(
            r_labels,
            X_test_solar,
            X_h_index,
            df_hydro_24_max,
            X_s_index,
            df_solar,
            instance,
            opt,
        )
        y_df["trans"] = trans
        y_test_df_solar = y_test_df_solar.append(y_df)

    return y_test_df_solar

def solar_trans_sensitivity_bg(
    trans_step,
    solar_test,
    hydro_mult,
    n_hydro,
    r_labels,
    X_h_index,
    df_hydro_24_max,
    X_s_index,
    df_solar,
    instance,
    opt,
):
    X_test_solar = np.zeros((len(hydro_mult), len(solar_test), len(r_labels)))

    for i, hm in enumerate(hydro_mult):
        for j, lvl in enumerate(solar_test):
            X_test_solar[i][j][:] = np.repeat(1, n_hydro).tolist() + [0, 0, lvl, 0, 0]
            X_test_solar[i][j][6] = hm
    X_test_solar = X_test_solar.reshape(
        int(len(solar_test) * len(hydro_mult)), len(r_labels)
    )

    y_test_df_solar = pd.DataFrame()

    y_df = pownet_sample(
        r_labels,
        X_test_solar,
        X_h_index,
        df_hydro_24_max,
        X_s_index,
        df_solar,
        instance,
        opt,
    )
    y_df["trans"] = "Existing"
    y_test_df_solar = y_test_df_solar.append(y_df)

    for i, trans in enumerate(trans_step):
        instance.linemva["ZIM_BATOKA.GO", "ZAM_BATOKA.GO"] = trans
        instance.linemva["ZAM_BATOKA.GO", "ZIM_BATOKA.GO"] = trans
        instance.linemva["ZAM_BATOKA.GO", "ZAM"] = trans
        instance.linemva["ZIM_BATOKA.GO", "ZIM"] = trans
        instance.linemva["ZAM", "ZAM_BATOKA.GO"] = trans
        instance.linemva["ZIM", "ZIM_BATOKA.GO"] = trans
        instance.linemva["ZIM_DEVIL.GO", "ZAM_DEVIL.GO"] = 2.0
        instance.linemva["ZAM_DEVIL.GO", "ZIM_DEVIL.GO"] = 2.0

        y_df = pownet_sample(
            r_labels,
            X_test_solar,
            X_h_index,
            df_hydro_24_max,
            X_s_index,
            df_solar,
            instance,
            opt,
        )
        y_df["trans"] = trans
        y_test_df_solar = y_test_df_solar.append(y_df)

    return y_test_df_solar

def solar_trans_sensitivity_dg(
    trans_step,
    solar_test,
    hydro_mult,
    n_hydro,
    r_labels,
    X_h_index,
    df_hydro_24_max,
    X_s_index,
    df_solar,
    instance,
    opt,
):
    X_test_solar = np.zeros((len(hydro_mult), len(solar_test), len(r_labels)))

    for i, hm in enumerate(hydro_mult):
        for j, lvl in enumerate(solar_test):
            X_test_solar[i][j][:] = np.repeat(1, n_hydro).tolist() + [0, 0, 0, lvl, 0]
            X_test_solar[i][j][7] = hm
    X_test_solar = X_test_solar.reshape(
        int(len(solar_test) * len(hydro_mult)), len(r_labels)
    )

    y_test_df_solar = pd.DataFrame()

    y_df = pownet_sample(
        r_labels,
        X_test_solar,
        X_h_index,
        df_hydro_24_max,
        X_s_index,
        df_solar,
        instance,
        opt,
    )
    y_df["trans"] = "Existing"
    y_test_df_solar = y_test_df_solar.append(y_df)

    for i, trans in enumerate(trans_step):
        instance.linemva["ZIM_DEVIL.GO", "ZAM_DEVIL.GO"] = trans
        instance.linemva["ZAM_DEVIL.GO", "ZIM_DEVIL.GO"] = trans
        instance.linemva["ZAM_DEVIL.GO", "ZAM"] = trans
        instance.linemva["ZIM_DEVIL.GO", "ZIM"] = trans
        instance.linemva["ZAM", "ZAM_DEVIL.GO"] = trans
        instance.linemva["ZIM", "ZIM_DEVIL.GO"] = trans
        instance.linemva["ZIM_BATOKA.GO", "ZAM_BATOKA.GO"] = 2.0
        instance.linemva["ZAM_BATOKA.GO", "ZIM_BATOKA.GO"] = 2.0

        y_df = pownet_sample(
            r_labels,
            X_test_solar,
            X_h_index,
            df_hydro_24_max,
            X_s_index,
            df_solar,
            instance,
            opt,
        )
        y_df["trans"] = trans
        y_test_df_solar = y_test_df_solar.append(y_df)

    return y_test_df_solar

# %% sample each reservoir's transmission sensitivity
instance = init_pownet(model.create_instance("./model/input/pownet_SAPP_24hr.dat"))
y_test_df_solar_ka = solar_trans_sensitivity_ka(
    trans_step,
    solar_test,
    hydro_mult,
    n_hydro,
    r_labels,
    X_h_index,
    df_hydro_24_max,
    X_s_index,
    df_solar,
    instance,
    opt,
)
# %% save
y_test_df_solar_ka.to_csv('./output/response_tables/solar_trans_sensitivity_ka.csv',index=False)
# %% cahora
instance = init_pownet(model.create_instance("./model/input/pownet_SAPP_24hr.dat"))
y_test_df_solar_cb = solar_trans_sensitivity_cb(
    trans_step,
    solar_test,
    hydro_mult,
    n_hydro,
    r_labels,
    X_h_index,
    df_hydro_24_max,
    X_s_index,
    df_solar,
    instance,
    opt,
)
# %% save
y_test_df_solar_cb.to_csv('./output/response_tables/solar_trans_sensitivity_cb.csv',index=False)
# %% mphanda
instance = init_pownet(model.create_instance("./model/input/pownet_SAPP_24hr.dat"))
y_test_df_solar_mn = solar_trans_sensitivity_mn(
    trans_step,
    solar_test,
    hydro_mult[0:1],
    n_hydro,
    r_labels,
    X_h_index,
    df_hydro_24_max,
    X_s_index,
    df_solar,
    instance,
    opt,
)
# %% save
y_test_df_solar_mn.to_csv('./output/response_tables/solar_trans_sensitivity_mn.csv',index=False)
# %% batoka
instance = init_pownet(model.create_instance("./model/input/pownet_SAPP_24hr.dat"))
y_test_df_solar_bg = solar_trans_sensitivity_bg(
    trans_step,
    solar_test,
    hydro_mult[0:1],
    n_hydro,
    r_labels,
    X_h_index,
    df_hydro_24_max,
    X_s_index,
    df_solar,
    instance,
    opt,
)
# %% save
y_test_df_solar_bg.to_csv('./output/response_tables/solar_trans_sensitivity_bg.csv',index=False)
# %% devils
instance = init_pownet(model.create_instance("./model/input/pownet_SAPP_24hr.dat"))
y_test_df_solar_dg = solar_trans_sensitivity_dg(
    trans_step,
    solar_test,
    hydro_mult[0:1],
    n_hydro,
    r_labels,
    X_h_index,
    df_hydro_24_max,
    X_s_index,
    df_solar,
    instance,
    opt,
)
# %% save
y_test_df_solar_dg.to_csv('./output/response_tables/solar_trans_sensitivity_dg.csv',index=False)

# %% load previous
y_test_df_solar_ka = pd.read_csv("./output/response_tables/solar_trans_sensitivity_ka.csv")
y_test_df_solar_cb = pd.read_csv("./output/response_tables/solar_trans_sensitivity_cb.csv")
y_test_df_solar_bg = pd.read_csv("./output/response_tables/solar_trans_sensitivity_bg.csv")
y_test_df_solar_dg = pd.read_csv("./output/response_tables/solar_trans_sensitivity_dg.csv")
y_test_df_solar_mn = pd.read_csv("./output/response_tables/solar_trans_sensitivity_mn.csv")


# %% transmission sensitivity plot
ncols = 5
fig, ax = plt.subplots(ncols=ncols, sharey=True, figsize=(12, 2.5))
plt.subplots_adjust(wspace=0.1)

z_solar = ["ka_solar", "cb_solar", "bg_solar", "dg_solar", "mn_solar"]
peak_solar = [2.5, 2.0, 1.8, 1.8, 1.8]

responses = [
    y_test_df_solar_ka,
    y_test_df_solar_cb,
    y_test_df_solar_bg,
    y_test_df_solar_dg,
    y_test_df_solar_mn,
]

for n in np.arange(0, ncols):
    data = responses[n]
    for i, trans in enumerate(data.trans.unique()[::-1]):
        data1 = data.loc[data.trans == trans].copy()
        if trans == "Existing":
            ax[n].plot(
                data1.iloc[:, 9 + n],
                data1[z_solar[n]] / 1e3,
                color="k",
                label=trans,
                lw=2,
            )
        else:
            ax[n].plot(
                data1.iloc[:, 9 + n],
                data1[z_solar[n]] / 1e3,
                color="C{}".format(i),
                label=trans,
                lw=3,
                linestyle="dashed",
            )
        if trans == "Existing":
            ax[n].axhline(
                data1[z_solar[n]].iloc[20] / 1e3, c="k", lw=0.5, linestyle="dashed"
            )

lgd = ax[1].legend(title="Trans. (MW)", fontsize="x-small", loc="upper left")
plt.setp(lgd.get_title(), fontsize="x-small")
ax[0].set_ylabel("Daily dispatch (GWh/day)")
for j in np.arange(0, ncols):
    ax[j].axvline(1, c="k", lw=0.5, linestyle="dashed")
    ax[j].set_xlabel(
        r_labels[9 + j] + " Multiplier\n" + "(1x={}GWh-peak)".format(peak_solar[j])
    )

# %%
