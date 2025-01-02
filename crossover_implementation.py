#importing
from slim_gsgp.main_slim import slim  # import the slim library
from slim_gsgp.config.slim_config import slim_gsgp_parameters
from slim_gsgp.datasets.data_loader import load_ld50,load_istanbul,load_ppb, load_concrete_slump, load_concrete_strength,load_resid_build_sale_price # import the loader for the datasets
from slim_gsgp.algorithms.SLIM_GSGP.operators.crossover_operators import donor_xo, donor_n_xo, best_donor_xo, best_donor_n_xo, new_donor_n_xo, new_best_donor_n_xo , dif_donor_n_xo, dif_best_donor_n_xo
from slim_gsgp.evaluators.fitness_functions import rmse  # import the rmse fitness metric
from slim_gsgp.utils.utils import train_test_split, show_individual  # import the train-test split function
from slim_gsgp.utils.utils import generate_random_uniform  # import the mutation step function
import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


#we will start by testing donor_xo and donor_n_xo

#Testing for all datasets means changing parameters for each one

dfs ={'toxicity': load_ld50(),
           'istanbul': load_istanbul(),
           'ppb': load_ppb(),
           'concrete_slump': load_concrete_slump(),
           'concrete_strength': load_concrete_strength(),
           'resid_build_sale_price': load_resid_build_sale_price()
           }

toxicity = {'ms': [0, 0.1],
            'p_inflate': 0.1}

concrete = {'ms': [0, 3],
            'p_inflate': 0.5}

#Create a function that runs slim for specified parameters and crossovers:
def run_slim(xo,xo_name, datasets):
    slim_gsgp_parameters['crossover'] = xo

    for df_name, df_function in datasets.items():
        print(df_name)
        if df_name == 'toxicity':
            slim_gsgp_parameters['ms'] = toxicity['ms']
            slim_gsgp_parameters['p_inflate'] = toxicity['p_inflate']
            ms_l = toxicity['ms'][0]
            ms_u = toxicity['ms'][1]
        elif df_name in ['concrete_slump', 'concrete_strength']:
            slim_gsgp_parameters['ms'] = concrete['ms']
            slim_gsgp_parameters['p_inflate'] = concrete['p_inflate']
            ms_l = concrete['ms'][0]
            ms_u = concrete['ms'][1]
        else:
            ms_l = 0
            ms_u = 1

        # Load dataset:
        X, y = df_function

        # Set the number of splits (30 partitions)
        n_splits = 30
        test_size = 0.3  # 30% test, 70% train

        # Store the rmse results
        rmse_list = []

        # Store best_individual information to select best individual of n_splits
        best_ind = []

        # Perform Monte Carlo Cross-Validation with 30 different splits
        for i in range(n_splits):
            print('split',i)
            # Randomly shuffle the dataset and split into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=test_size, seed=i)
            X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5, seed=i)

            # Apply the SLIM GSGP algorithm
            for algorithm in ["SLIM+SIG2", "SLIM*ABS"]:
                print(algorithm)

                final_tree = slim(X_train=X_train, y_train=y_train,
                                  X_test=X_val, y_test=y_val,
                                  dataset_name=df_name, slim_version=algorithm,pop_size=100, n_iter=1000,
                                  ms_lower=ms_l, ms_upper=ms_u, p_inflate=0.3, log_path=os.path.join(os.getcwd(),
                                                                                                     "log",
                                                                                                     f"{xo_name}",
                                                                                                     f"{df_name}_{xo_name}.csv"), )



crossovers = ["dif_best_donor_n_xo"]
#crossovers = ["dif_donor_n_xo"]
for cross_name in crossovers:
    print(cross_name)
    if cross_name == 'donor_xo' :
        run_slim(donor_xo, f"{cross_name}", dfs)

    elif cross_name == 'new_donor_n_xo':
        for n_d in [1,5,25]:
            print('n',n_d)
            run_slim(new_donor_n_xo(n_d),f"{n_d}_{cross_name}", dfs)

    elif cross_name == 'dif_donor_n_xo':
        for n_d in [25,5,1]:
            print('n',n_d)
            run_slim(dif_donor_n_xo(n_d),f"{n_d}_{cross_name}", dfs)

    elif cross_name == 'best_donor_xo':
        for fit in ['biggest', 'min_fitness']:
            print('fit',fit)
            run_slim(best_donor_xo(fit), f"{fit}_{cross_name}", dfs)

    elif cross_name == 'new_best_donor_n_xo':
        for fit in ['min_fitness']:
            print('fit',fit)
            for n_d in [25,5,1]:
                print('n:',n_d)
                run_slim(new_best_donor_n_xo(n_d, fit), f"{fit}_{n_d}_{cross_name}", dfs)

    elif cross_name == 'dif_best_donor_n_xo':
        for fit in ['biggest','min_fitness']:
            print('fit',fit)
            for n_d in [25,5,1]:
                print('n:',n_d)
                run_slim(dif_best_donor_n_xo(n_d, fit), f"{fit}_{n_d}_{cross_name}", dfs)

    print('-'*40)