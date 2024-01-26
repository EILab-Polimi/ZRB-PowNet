# ZRB-PowNet Response Model
This repository develops a version of the PowNet model ([Chowdhury et al. (2020)](https://openresearchsoftware.metajnl.com/articles/10.5334/jors.302/)) for the South African Power Pool (SAPP) for soft-linking to the Zambezi watercourse (ZRB) reservoir operations model and EMODPS optimization.

# How to use
There are four main parts to using this package:
1. PowNet model construction and data setup
2. Sampling the PowNet model to build a response model of OPEX, solar production, and hydropower curtailment
3. Validating the response model
4. Running the PowNet model for select solutions to output high-resolution time series

## 1. PowNet Setup
The PowNet model configured for the Zambezi-SAPP case study is located in the ```/model``` directory. To build the model, open a terminal, navigate into ```/model``` directory and run ```python pownet_datasetup.py```. It will output a data file (```/model/input/pownet_SAPP_24hr.dat```) which is used as input to ```/model/pownet_model.py``` to construct a Pyomo linear program of the SAPP power grid.

## 2. PowNet Sampling
Three scripts are used for sampling PowNet:
- ```sample_24hr_OPEX_response.py``` : develops a linear model predicting the OPEX objective value in PowNet regressed on total hydropower and solar production from the Zambezi watercourse
- ```sample_24hr_dispatch_response.py``` : develops lookup tables of a) solar availability --> solar dispatch (for each reservoir site) and b) solar dispatch & hydropower availability --> hydropower curtailment (for each reservoir site)
- ```sample_24hr_transmission_sensitivity.py``` :  test the sensitivity of solar dispatch to transmission line capacity at various levels of solar peak capacity for each reservoir 

## 3. Validation of Response Model
These scripts are used to validate the response model by comparing the difference between  simulated outputs of the Zambezi watercourse reservoir operations model to simulated outputs of the PowNet model  
- ```validate_multiyear_run.py``` : run the simulated output (i.e., monthly hydropower and solar production) of a pareto-efficient solution from the Zambezi watercourse model as a constraint on hydropower availability and peak solar capacity for a PowNet simulation. The PowNet simulation is run for 1 week per month over the 1985-2004 period using the PVGIS daily solar insolation dataset.
- ```validate_multiyear_run_main.py``` : wrapper for above script to run multiple solutions in parallel
- ```validate_multiyear_run_post.py``` : post-process the ZW and PowNet model outputs

## 4. Validation of Response Model
- ```hourly_multiyear_run_yearmonth.py``` : for a given month and year, runs and outputs at high-resolution (daily) PowNet simulation for a chosen ZW model solution
- ```hourly_multiyear_run_yearmonth_main.py``` : wrapper for above script to run multiple solutions in parallel

# Reference
The Zambezi watercourse reservoir operations simulation model contains sensitive hydrologic data, along with hydropower plant characteristics from the Zambezi River Authority (ZRA), Zambia Electricity Supply Corporation (ZESCO) and Hidroeléctrica de Cahora Bassa (HCB), thus it cannot be made public.

*PVGIS data:* European Commission Joint Research Centre. 2022. “PVGIS 5.2.” EU Science Hub. https://joint-research-centre.ec.europa.eu/pvgis-photovoltaic-geographical-information-system/pvgis-releases/pvgis-52_en.

*PowNet: Unit Commitment / Economic Dispatch model in Python:* PowNet is a least-cost optimization model for simulating the Unit Commitment and Economic Dispatch (UC/ED) of large-scale (regional to country) power systems. More details about the functionalities of PowNet are provided in [Chowdhury et al. (2020)](https://openresearchsoftware.metajnl.com/articles/10.5334/jors.302/). PowNet is written in Python 3.5. It requires the following Python packages: (i) Pyomo, (ii) NumPy, (iii) Pandas, and (iv) Matplotlib (optional for plotting). It also requires an optimization solver (e.g., Gurobi, CPLEX). See: Chowdhury, A.F.M.K., Kern, J., Dang, T.D. and Galelli, S., 2020. PowNet: A Network-Constrained Unit Commitment/Economic Dispatch Model for Large-Scale Power Systems Analysis. Journal of Open Research Software, 8(1), p.5. DOI: http://doi.org/10.5334/jors.302.
