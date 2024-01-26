import os
import multiprocessing
import numpy as np
import pandas as pd


def run_command(cmd):
    """execute cmd via the shell."""
    print("starting `{}` ...".format(cmd))
    os.system(cmd)
    print("end `{}`".format(cmd))


def run_commands(commands, n_parallel):
    """run commands (up to n_parallel in parallel)."""
    worker = multiprocessing.Pool(n_parallel)
    worker.map(run_command, commands)


if __name__ == "__main__":
    seed = 1
    np.random.seed(seed)

    sets = {  #'FPV_all_5_res':'230 231, KA CB',
        # "FPV_all_6_res_BG": "264 265 266, KA CB BG",
        "FPV_all_6_res_DG": "264 265 266, KA CB DG",
        "FPV_all_6_res_MN": "264 265 266, KA CB MN",
        "FPV_all_7_res_DGMN": "298 299 300 301, KA CB DG MN",
        "FPV_all_7_res_BGMN": "298 299 300 301, KA CB BG MN",
        "FPV_all_7_res_BGDG": "298 299 300 301, KA CB BG DG",
        "FPV_all_8_res": "332 333 334 335 336, KA CB BG DG MN",
    }

    for set in sets:
        policies = pd.read_csv(
            os.path.join("./data/zrb_emodps_policies", set + ".txt"),
            header=None,
            sep="\s+",
        )
        n_policies = len(policies)
        idxs = np.random.randint(low=0, high=n_policies, size=7)
        runs = [
            "python validate_multiyear_run.py "
            + '"{} {}, {}"'.format(set, i, sets[set])
            for i in idxs
        ]
        run_commands(runs, n_parallel=7)
