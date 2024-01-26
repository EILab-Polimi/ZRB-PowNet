# %%
import os, glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mat
from scipy import stats
from sklearn.metrics import mean_squared_error

mat.rcParams["figure.dpi"] = 300

# %% labels
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


# %% convert opex to annual
def opex_yr(data):
    return data["cost"].sum() / (len(data) / 365) / 1e9


# %% solutions
val_solutions = glob.glob("./output/sim/validation/*.csv")
bm_solutions = glob.glob("./output/sim/benchmark/*.csv")

# %% load solution simulations into merged ZRB-PowNet dataframes at monthly and annual summarized time step
def load_solutions(solutions, benchmark=False):
    monthly = pd.DataFrame()
    annual = pd.DataFrame()

    for each in solutions:
        # solution markers
        s = each.split("/")[-1].split(".")[0]
        print(s)
        count_ = s.count("_")
        solution_name = "_".join(s.split("_")[0:count_])
        solution_id = s.split("_")[count_]
        if count_ == 4:
            solution_res = "_".join(s.split("_")[2:4])
        else:
            solution_res = "_".join(s.split("_")[2:5])

        # ZRB simulation
        months = np.tile(np.arange(1, 13), 20)  # monthly over 20 years
        years = np.repeat(np.arange(1985, 2005), 12)
        df_sim = pd.read_csv(
            os.path.join("./data/zrb_emodps_sim", "{}.sim".format(solution_name)),
            header=None,
            usecols=[0, 25, 26, 27, 28, 29, 30, 31, 32, 60, 61, 62, 63, 64, 94],
            names=[
                "idx",
                "ITT",
                "KGU",
                "KGL",
                "KA",
                "CB",
                "BG",
                "DG",
                "MN",
                "KA_s",
                "CB_s",
                "BG_s",
                "DG_s",
                "MN_s",
                "OPEX",
            ],
        )

        df_sim = df_sim.loc[df_sim.idx == int(solution_id) + 1]
        df_sim.insert(0, "month", value=months)
        df_sim.insert(0, "year", value=years)
        df_sim.insert(
            2,
            "ZRB Hydro",
            value=df_sim["ITT"]
            + df_sim["KGU"]
            + df_sim["KGL"]
            + df_sim["KA"]
            + df_sim["CB"]
            + df_sim["BG"]
            + df_sim["DG"]
            + df_sim["MN"],
        )
        df_sim.insert(
            2,
            "ZRB Solar",
            value=df_sim["KA_s"]
            + df_sim["CB_s"]
            + df_sim["BG_s"]
            + df_sim["DG_s"]
            + df_sim["MN_s"],
        )
        df_sim_y = df_sim.groupby("year", as_index=False).mean()
        df_sim_y.insert(0, "solution", value=solution_name)
        df_sim_y.insert(0, "res", value=solution_res)
        df_sim.insert(0, "solution", value=solution_name)
        df_sim.insert(0, "res", value=solution_res)

        # PowNet simulation
        months = np.tile(
            np.repeat(np.arange(1, 13), 7), 20
        )  # 1 week per month over 20 years
        years = np.repeat(np.arange(1985, 2005), 7 * 12)
        subpath = "benchmark" if benchmark else "validation"
        df_pownet = pd.read_csv(
            "./output/sim/{}/{}.csv".format(
                subpath, solution_name + "_" + str(solution_id)
            )
        )
        df_pownet.drop("Unnamed: 0", axis=1, inplace=True)
        df_pownet["cost"] = df_pownet["cost"] / 1e9
        df_pownet["zrb_solar"] = df_pownet["zrb_solar"] / 1e6
        df_pownet["zrb_hydro"] = (
            df_pownet["zrb_hydro"] - 3096
        ) / 1e6  # subtract victoria 129MW*24hr
        df_pownet["ka_solar"] = df_pownet["ka_solar"] / 1e6
        df_pownet["cb_solar"] = df_pownet["cb_solar"] / 1e6
        df_pownet["bg_solar"] = df_pownet["bg_solar"] / 1e6
        df_pownet["dg_solar"] = df_pownet["dg_solar"] / 1e6
        df_pownet["mn_solar"] = df_pownet["mn_solar"] / 1e6
        df_pownet.insert(
            0,
            "ka_hydro",
            value=(df_pownet["ka_n_hydro"] + df_pownet["ka_s_hydro"]) / 1e6,
        )
        df_pownet["cb_hydro"] = df_pownet["cb_hydro"] / 1e6
        df_pownet.insert(
            0,
            "bg_hydro",
            value=(df_pownet["bg_n_hydro"] + df_pownet["bg_s_hydro"]) / 1e6,
        )
        df_pownet.insert(
            0,
            "dg_hydro",
            value=(df_pownet["dg_n_hydro"] + df_pownet["dg_s_hydro"]) / 1e6,
        )
        df_pownet["mn_hydro"] = df_pownet["mn_hydro"] / 1e6
        df_pownet["itt_hydro"] = df_pownet["itt_hydro"] / 1e6
        df_pownet["kgu_hydro"] = df_pownet["kgu_hydro"] / 1e6
        df_pownet["ka_s_hydro"] = df_pownet["ka_s_hydro"] / 1e6
        df_pownet["ka_n_hydro"] = df_pownet["ka_n_hydro"] / 1e6
        df_pownet = df_pownet * 365
        df_pownet.insert(0, "month", value=months)
        df_pownet.insert(0, "year", value=years)
        df_pownet_m = df_pownet.groupby(["month", "year"], as_index=False).mean()
        df_pownet_y = df_pownet_m.groupby(["year"], as_index=False).mean()
        df_pownet_y.insert(0, "solution", value=solution_name)
        df_pownet_m.insert(0, "solution", value=solution_name)
        df_pownet.insert(0, "solution", value=solution_name)

        # Combine ZRB with PowNet
        monthly = pd.concat(
            [monthly, df_sim.merge(df_pownet_m, on=["month", "year", "solution"])],
            axis=0,
        )
        annual = pd.concat(
            [annual, df_sim_y.merge(df_pownet_y, on=["year", "solution"])], axis=0
        )

    return monthly, annual

# %% validation solutions
monthly, annual = load_solutions(val_solutions, benchmark=False)

# %% sort
annual.sort_values(['year','solution'],inplace=True)
monthly.sort_values(['year','solution'],inplace=True)

# %% benchmark solutions
# monthly, annual = load_solutions(bm_solutions, benchmark=True) 
# annual.to_csv('FPV_all_select_annual.csv')
# monthly.to_csv('FPV_all_select_monthly.csv')

# %% mse zrb-pownet opex
np.sqrt(mean_squared_error(annual["OPEX"], annual["cost"]))

# %% plot ZRB vs PowNET OPEX across all solutions
fig, ax = plt.subplots(figsize=(5, 5))
# get coeffs of linear fit
slope, intercept, r_value, p_value, std_err = stats.linregress(
    annual["OPEX"], annual["cost"]
)
sns.regplot(
    x="OPEX",
    y="cost",
    data=annual,
    color="b",
    line_kws={"label": "y={0:.2f}x+{1:.2f}".format(slope, intercept)},
    scatter_kws={"s": 2, "color": "grey"},
    x_ci="None",
    ax=ax,
)
sns.scatterplot(x="OPEX", y="cost", data=annual, hue="res", s=10, ax=ax)
ax.legend(fontsize='small')
ax.set_xlabel("ZRB Design Model")
ax.set_ylabel("PowNet Simulation")
ax.set_title("OPEX [$Bil/yr]")
plt.show()

# %% plot each reservoir's solar
fig, ax = plt.subplots(figsize=(16, 2.8), nrows=1, ncols=5)
plt.subplots_adjust(wspace=0.3)

# use line_kws to set line label for legend
slope, intercept, r_value, p_value, std_err = stats.linregress(
    annual["KA_s"], annual["ka_solar"]
)
sns.regplot(
    x="KA_s",
    y="ka_solar",
    data=annual,
    color="b",
    line_kws={"label": "y={0:.2f}x+{1:.2f}".format(slope, intercept)},
    scatter_kws={"s": 2, "color": "grey"},
    x_ci="None",
    ax=ax[0],
)

slope, intercept, r_value, p_value, std_err = stats.linregress(
    annual["CB_s"], annual["cb_solar"]
)
sns.regplot(
    x="CB_s",
    y="cb_solar",
    data=annual,
    color="b",
    line_kws={"label": "y={0:.2f}x+{1:.2f}".format(slope, intercept)},
    scatter_kws={"s": 2, "color": "grey"},
    x_ci="None",
    ax=ax[1],
)

slope, intercept, r_value, p_value, std_err = stats.linregress(
    annual["BG_s"], annual["bg_solar"]
)
sns.regplot(
    x="BG_s",
    y="bg_solar",
    color="b",
    data=annual.loc[
        (annual.res.str.contains("BG")) | (annual.res.str.contains("8_res"))
    ],
    line_kws={"label": "y={0:.2f}x+{1:.2f}".format(slope, intercept)},
    scatter_kws={"s": 2, "color": "grey"},
    x_ci="None",
    ax=ax[2],
)

slope, intercept, r_value, p_value, std_err = stats.linregress(
    annual["DG_s"], annual["dg_solar"]
)
sns.regplot(
    x="DG_s",
    y="dg_solar",
    color="b",
    data=annual.loc[
        (annual.res.str.contains("DG")) | (annual.res.str.contains("8_res"))
    ],
    line_kws={"label": "y={0:.2f}x+{1:.2f}".format(slope, intercept)},
    scatter_kws={"s": 2, "color": "grey"},
    x_ci="None",
    ax=ax[3],
)

slope, intercept, r_value, p_value, std_err = stats.linregress(
    annual["MN_s"], annual["mn_solar"]
)
sns.regplot(
    x="MN_s",
    y="mn_solar",
    color="b",
    data=annual.loc[
        (annual.res.str.contains("MN")) | (annual.res.str.contains("8_res"))
    ],
    line_kws={"label": "y={0:.2f}x+{1:.2f}".format(slope, intercept)},
    scatter_kws={"s": 2, "color": "grey"},
    x_ci="None",
    ax=ax[4],
)

# plot legend
for i in range(5):
    ax[i].legend()
    ax[i].set_xlabel("ZRB Design Model")
    ax[i].set_ylabel("PowNet Simulation")  # if i < 1 else ax[i].set_ylabel('')
    ax[i].set_xlim(ax[i].get_ylim())
    ax[i].set_xticks(ax[i].get_yticks())
    ax[i].set_yticks(ax[i].get_xticks())
    ax[i].set_xlim(left=-0.2)
    ax[i].set_ylim(bottom=-0.2)

ax[0].set_title("KA Solar [MWh/yr]")
ax[1].set_title("CB Solar [MWh/yr]")
ax[2].set_title("BG Solar [MWh/yr]")
ax[3].set_title("DG Solar [MWh/yr]")
ax[4].set_title("MN Solar [MWh/yr]")

plt.show()


# %% plot each reservoir's hydro
fig, ax = plt.subplots(figsize=(16, 2.8), nrows=1, ncols=5)
plt.subplots_adjust(wspace=0.3)

# use line_kws to set line label for legend
slope, intercept, r_value, p_value, std_err = stats.linregress(
    annual["KA"], annual["ka_hydro"]
)
sns.regplot(
    x="KA",
    y="ka_hydro",
    data=annual,
    color="b",
    line_kws={"label": "y={0:.2f}x+{1:.2f}".format(slope, intercept)},
    scatter_kws={"s": 5, "color": "grey"},
    x_ci="None",
    ax=ax[0],
)

slope, intercept, r_value, p_value, std_err = stats.linregress(
    annual["CB"], annual["cb_hydro"]
)
sns.regplot(
    x="CB",
    y="cb_hydro",
    data=annual,
    color="b",
    line_kws={"label": "y={0:.2f}x+{1:.2f}".format(slope, intercept)},
    scatter_kws={"s": 5, "color": "grey"},
    x_ci="None",
    ax=ax[1],
)

slope, intercept, r_value, p_value, std_err = stats.linregress(
    annual["BG"], annual["bg_hydro"]
)
sns.regplot(
    x="BG",
    y="bg_hydro",
    color="b",
    data=annual.loc[
        (annual.res.str.contains("BG")) | (annual.res.str.contains("8_res"))
    ],
    line_kws={"label": "y={0:.2f}x+{1:.2f}".format(slope, intercept)},
    scatter_kws={"s": 5, "color": "grey"},
    x_ci="None",
    ax=ax[2],
)

slope, intercept, r_value, p_value, std_err = stats.linregress(
    annual["DG"], annual["dg_hydro"]
)
sns.regplot(
    x="DG",
    y="dg_hydro",
    color="b",
    data=annual.loc[
        (annual.res.str.contains("DG")) | (annual.res.str.contains("8_res"))
    ],
    line_kws={"label": "y={0:.2f}x+{1:.2f}".format(slope, intercept)},
    scatter_kws={"s": 5, "color": "grey"},
    x_ci="None",
    ax=ax[3],
)
slope, intercept, r_value, p_value, std_err = stats.linregress(
    annual["MN"], annual["mn_hydro"]
)
sns.regplot(
    x="MN",
    y="mn_hydro",
    color="b",
    data=annual.loc[
        (annual.res.str.contains("MN")) | (annual.res.str.contains("8_res"))
    ],
    line_kws={"label": "y={0:.2f}x+{1:.2f}".format(slope, intercept)},
    scatter_kws={"s": 5, "color": "grey"},
    x_ci="None",
    ax=ax[4],
)

# plot legend
for i in range(5):
    ax[i].legend()
    ax[i].set_xlabel("ZRB Design Model")
    ax[i].set_ylabel("PowNet Simulation")  # if i < 1 else ax[i].set_ylabel('')
    ax[i].set_xlim(ax[i].get_ylim())
    ax[i].set_xticks(ax[i].get_yticks())
    ax[i].set_yticks(ax[i].get_xticks())
    ax[i].set_xlim(left=-0.2)
    ax[i].set_ylim(bottom=-0.2)


ax[0].set_title("KA Hydro [MWh/yr]")
ax[1].set_title("CB Hydro [MWh/yr]")
ax[2].set_title("BG Hydro [MWh/yr]")
ax[3].set_title("DG Hydro [MWh/yr]")
ax[4].set_title("MN Hydro [MWh/yr]")

plt.show()

# %%
