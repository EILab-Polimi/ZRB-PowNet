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
    solutions = [
         'python pownet_multiyear_run_yearmonth_select.py "FPV_all_6_res_MN 204, 264 265 266, KA CB MN, 10, 10"',
         'python pownet_multiyear_run_yearmonth_select.py "FPV_all_6_res_DG 366, 264 265 266, KA CB DG, 10, 10"',
         'python pownet_multiyear_run_yearmonth_select.py "FPV_all_8_res 116, 332 333 334 335 336, KA CB BG DG MN, 10, 10"'
    ]

    run_commands(solutions, n_parallel=3)
