# %%
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats.distributions import truncnorm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mat

mat.rcParams["figure.dpi"] = 300

# %% import pownet model
from model.pownet_model import model
from pyomo.opt import SolverFactory
import pyomo.environ as pyo

# %% my functions
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
            )

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
r_agg_labels = [
    "ka_total",
    "bg_total",
    "dg_total",
    "cb_total",
    "mn_total",
    "vic_total",
    "kg_total",
]

# %%  Create dictionaries of the PowNet feature to the index of the sampling list of coefficients
# HP generator indexes
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
# solar generator indexes
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

# %% lhs sample generator
seed = 42
lhs = stats.qmc.LatinHypercube(d=14)

# %% sample lhs with increased (bias) sampling frequency of lower generation
X_test = lhs.random(n=int(2.5e2))
u, sd = 0.05, 0.1
uvs = []
for i in range(11):
    a, b = (0 - u) / sd, (1 - u) / sd
    X_test[:, i] = truncnorm(a=a, b=b, loc=u, scale=sd).ppf(X_test[:, i])

# %% plot multiplier lhs sampling histogram
plt.hist((X_test[:, 0]), density=False, histtype="step")
plt.ylabel("Frequency")
plt.xlabel("Multiplier")

# %% run pownet for all samples
X_test_df = pownet_sample(
    r_labels,
    X_test,
    X_h_index,
    df_hydro_24_max,
    X_s_index,
    df_solar,
    instance,
    opt,
)

# %% save response sample
# X_test_df.to_csv('./output/response_tables/lhs_24hr_pownet_sample25k.csv'.format(zrb_path),index=False)

# %% read previous sampling
X_test_df = pd.read_csv("./output/response_tables/lhs_24hr_pownet_sample25k.csv")

# %% add indicators (aggregate of variables)
def add_indicators(df):
    try:
        df.insert(
            0, "ka_total", value=df[["ka_s_hydro", "ka_n_hydro", "ka_solar"]].sum(axis=1)
        )
        df.insert(
            0, "bg_total", value=df[["bg_s_hydro", "bg_n_hydro", "bg_solar"]].sum(axis=1)
        )
        df.insert(
            0, "dg_total", value=df[["dg_s_hydro", "dg_n_hydro", "dg_solar"]].sum(axis=1)
        )
        df.insert(0, "cb_total", value=df[["cb_hydro", "cb_solar"]].sum(axis=1))
        df.insert(0, "mn_total", value=df[["mn_hydro", "mn_solar"]].sum(axis=1))
        df.insert(0, "vic_total", value=df[["vic_hydro"]].sum(axis=1))
        df.insert(
            0, "kg_total", value=df[["itt_hydro", "kgu_hydro", "kgl_hydro"]].sum(axis=1)
        )
    except:
        print('res totals already added')
    try:
        df.insert(
            0,
            "zrb_hydro",
            value=df[
                [
                    "ka_s_hydro",
                    "ka_n_hydro",
                    "bg_s_hydro",
                    "dg_s_hydro",
                    "dg_n_hydro",
                    "cb_hydro",
                    "mn_hydro",
                    "vic_hydro",
                    "itt_hydro",
                    "kgu_hydro",
                    "kgl_hydro",
                ]
            ].sum(axis=1),
        )
    except:
        print('zrb hydro already added')
    try:
        df.insert(
            0,
            "zrb_hydro_solar",
            value=df[
                [
                    "kg_total",
                    "vic_total",
                    "mn_total",
                    "cb_total",
                    "dg_total",
                    "bg_total",
                    "ka_total",
                ]
            ].sum(axis=1),
        )
    except:
        print('zrb hydro-solar already added')

add_indicators(X_test_df)

# %% arrays for regression model
X_test_1 = X_test_df[r_agg_labels]
y_test_1 = X_test_df["cost"]

# %% 1st order gen/cost r correlations
for r in r_labels:
    data = X_test_df.copy()
    corr = stats.pearsonr(data[r], data["cost"])
    print(r, ": ", round(corr[0], 2))

# %% max deficit sampled
X_test_df.deficit.max()

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_test_1, y_test_1, test_size=0.33, random_state=seed
)

# %% linear model
model = LinearRegression()

# %% fit model
model.fit(X_train, y_train)

# %% score model
model.score(X_test, y_test)

# %% coefficients of linear model (upscale to annual $bil *30 days/month * 12 months/yr / 1e9 )
print(np.round(model.intercept_ * 30 * 12 / 1e9, 2))
print(np.round(model.coef_ / 1e6, 6))

# %% plot hydro-solar / cost
fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
x_data = X_test_df["zrb_hydro_solar"] / 24
y_data = X_test_df["cost"] / 1e6
ax.scatter(x_data, y_data, s=5)
ax.set_ylabel("SAPP PowNet Operation Cost [$Mil/day]")
ax.set_xlabel("ZRB Solar + Hydropower [MWh]")

# %% residuals calculation
y_pred = model.predict(X_test_1)
residuals = ((y_pred - y_test_1) / y_test_1) * 100
y_pred = y_pred / 1e6

def cost_bin(y, bins):
    if y < bins[0]:
        return 1
    elif y < bins[1]:
        return 2
    elif y < bins[2]:
        return 3
    elif y < bins[3]:
        return 4
    elif y < bins[4]:
        return 5
    elif y < bins[5]:
        return 6
    elif y < bins[6]:
        return 7
    elif y < bins[7]:
        return 8
    elif y > bins[7]:
        return 9

y_test_binned = np.array(
    list(
        map(
            lambda x: cost_bin(
                x, bins=[n for n in [46.5, 48.0, 49.0, 51.0, 52.5, 54.0, 55.5, 57.0]]
            ),
            y_test_1 / 1e6,
        )
    )
)

# %% plot residuals
g = sns.jointplot(
    y=y_test_1 / 1e6,
    x=residuals,
    s=5,
    legend=False,
    palette="coolwarm",
    height=3.5,
    ratio=4,
)
g.ax_joint.axvline(0, color="k")
g.ax_marg_x.axhline(0, color="k")
g.ax_joint.set_xlabel(r"Residuals ($\frac{\hat{y}-y}{y}*100$)")
g.ax_joint.set_ylabel("SAPP PowNet Operation Cost [$Mil/day]")
