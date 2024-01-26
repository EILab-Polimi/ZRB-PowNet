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
    printout=1000,
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
solar_test = np.arange(0, 2.05, 0.05)
print(solar_test.T)
# print(solar_test.T[int(1.87/0.05)])

# %% hydro multiplier sampling intervals
hydro_mult = np.arange(0, 1.025, 0.025)
print(hydro_mult.T)

# %% total matrix sampling points
len(solar_test) * len(hydro_mult)

# %% solar hydro sensitivity pownet sampling for each reservoir
def solar_hydro_sensitivity(
    res_idx,
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
    solar = [0, 0, 0, 0, 0]
    for i, hm in enumerate(hydro_mult):
        for j, lvl in enumerate(solar_test):
            solar[res_idx] = lvl
            X_test_solar[i][j][:] = np.repeat(1, n_hydro).tolist() + solar
            X_test_solar[i][j][4] = hm
    X_test_solar = X_test_solar.reshape(
        int(len(solar_test) * len(hydro_mult)), len(r_labels)
    )

    y_test_df_solar = pownet_sample(
        r_labels,
        X_test_solar,
        X_h_index,
        df_hydro_24_max,
        X_s_index,
        df_solar,
        instance,
        opt,
        printout=100
    )
    y_test_df_solar["trans"] = "Existing"

    return y_test_df_solar

# %% kariba
y_test_df_solar_ka = solar_hydro_sensitivity(
    res_idx=0,
    solar_test=solar_test,
    hydro_mult=hydro_mult,
    n_hydro=n_hydro,
    r_labels=r_labels,
    X_h_index=X_h_index,
    df_hydro_24_max=df_hydro_24_max,
    X_s_index=X_s_index,
    df_solar=df_solar,
    instance=instance,
    opt=opt,
)
# y_test_df_solar_ka.to_csv('./output/response_tables/solar_hydro_response_ka.csv',index=False)
# %% cahora
y_test_df_solar_cb = solar_hydro_sensitivity(
    res_idx=1,
    solar_test=solar_test,
    hydro_mult=hydro_mult,
    n_hydro=n_hydro,
    r_labels=r_labels,
    X_h_index=X_h_index,
    df_hydro_24_max=df_hydro_24_max,
    X_s_index=X_s_index,
    df_solar=df_solar,
    instance=instance,
    opt=opt,
)
# y_test_df_solar_cb.to_csv('./output/response_tables/solar_hydro_response_cb.csv',index=False)
# %% mphanda
y_test_df_solar_mn = solar_hydro_sensitivity(
    res_idx=4,
    solar_test=solar_test,
    hydro_mult=hydro_mult,
    n_hydro=n_hydro,
    r_labels=r_labels,
    X_h_index=X_h_index,
    df_hydro_24_max=df_hydro_24_max,
    X_s_index=X_s_index,
    df_solar=df_solar,
    instance=instance,
    opt=opt,
)
# y_test_df_solar_mn.to_csv('./output/response_tables/solar_hydro_response_mn.csv',index=False)
# %% batoka
y_test_df_solar_bg = solar_hydro_sensitivity(
    res_idx=2,
    solar_test=solar_test,
    hydro_mult=hydro_mult,
    n_hydro=n_hydro,
    r_labels=r_labels,
    X_h_index=X_h_index,
    df_hydro_24_max=df_hydro_24_max,
    X_s_index=X_s_index,
    df_solar=df_solar,
    instance=instance,
    opt=opt,
)
# y_test_df_solar_bg.to_csv('./output/response_tables/solar_hydro_response_bg.csv',index=False)
# %% devils
y_test_df_solar_dg = solar_hydro_sensitivity(
    res_idx=3,
    solar_test=solar_test,
    hydro_mult=hydro_mult,
    n_hydro=n_hydro,
    r_labels=r_labels,
    X_h_index=X_h_index,
    df_hydro_24_max=df_hydro_24_max,
    X_s_index=X_s_index,
    df_solar=df_solar,
    instance=instance,
    opt=opt,
)
# y_test_df_solar_dg.to_csv('./output/response_tables/solar_hydro_response_dg.csv',index=False)

# %% load previous
y_test_df_solar_ka = pd.read_csv("./output/response_tables/solar_hydro_response_ka.csv")
y_test_df_solar_cb = pd.read_csv("./output/response_tables/solar_hydro_response_cb.csv")
y_test_df_solar_bg = pd.read_csv("./output/response_tables/solar_hydro_response_bg.csv")
y_test_df_solar_dg = pd.read_csv("./output/response_tables/solar_hydro_response_dg.csv")
y_test_df_solar_mn = pd.read_csv("./output/response_tables/solar_hydro_response_mn.csv")

# %% reservoir max hydro
ka_n_max_hydro = 14640
ka_s_max_hydro = 11088
bg_max_hydro = 20160
cb_max_hydro = 49728
mn_max_hydro = 31272
dg_max_hydro = 16248

ka_peak_solar = 2.5
cb_peak_solar = 2.0
bg_peak_solar = 1.8
dg_peak_solar = 1.8
mn_peak_solar = 1.8

# %%
res = "KA"
res_solar = "ka_solar"
hyd = "ka_n_hydro"
max_hydro = ka_n_max_hydro
peak_s = ka_peak_solar
solar_sample_df = y_test_df_solar_ka.copy()

# %% solar production lookup table
solar_table = solar_sample_df.query('trans=="Existing" and {}==1.0'.format(res))[
    ["{}_s".format(res), res_solar]
]
# np.savetxt('{}_lookup_{}.txt'.format(res_solar,'base'),solar_table[res_solar].values/24, fmt='%1.1f')

# %% plot solar production curve
fig, ax = plt.subplots(1, 1)
data = solar_sample_df.query('trans=="Existing"')[
    [res, "{}_s".format(res), res_solar]
].melt(value_vars=[res_solar], id_vars=[res, "{}_s".format(res)])
sns.lineplot(data=data, x=data["{}_s".format(res)] * peak_s, y=data.value / 1e3)
ax.set_title("{} Solar Production Curve".format(res))
ax.set_ylabel("GWh/day")
ax.set_xlabel("GWh-peak")

# %% hydropower curtailment lookup table
data = solar_sample_df.query('trans=="Existing"')[[res_solar, res, hyd]]
data[hyd] = (data[res] - data[hyd] / max_hydro).round(2)
curtailment = pd.pivot_table(data, columns=res_solar, index=res)
# np.savetxt('{}_solar_curtailment_{}.txt'.format(hyd,'base'),curtailment.values,fmt='%1.2f')

# %% plot hydropower curtailment surface
fig, ax = plt.subplots(1, 1)
x, y, z = data[res].values * 100, data[res_solar].values / 1e3, data[hyd].values * 100
idx = np.lexsort((y, x)).reshape(len(hydro_mult), len(solar_test))
ax.contour(x[idx], y[idx], z[idx], levels=11, linewidths=0.5, colors="k")
cntr1 = ax.contourf(x[idx], y[idx], z[idx], levels=11, cmap="RdBu_r")
fig.colorbar(cntr1, ax=ax, label="Curtailment (%)")
ax.set_title("{} Hydropower Curtailment".format(res))
ax.set_xlabel("Hydropower Availability (%)")
ax.set_ylabel("Solar Production (GWh/day)")

# %%
