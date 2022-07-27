import subprocess
import pandas as pd
import numpy as np
import time
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import argparse
import os

parser = argparse.ArgumentParser(description='Bayesian optimization of hyperparameters for a pinnPotentialFoam case.')
parser.add_argument('N_iter', metavar='N_iter', type=int,
                    help='Number of optimization iterations.')
parser.add_argument("-case","--case", help="Foam case, default is cwd", default = os.getcwd())
parser.add_argument("-epochs","--epochs", help="Training epochs", default = os.getcwd())
args = parser.parse_args()


N_iter=args.N_iter
max_iterations=args.epochs
case = args.case
#space_lists=['layers', 'nodes_per_layer', 'optimizer_step']
#pbounds = {space_lists[0]: (4, 7),
#           space_lists[1]: (10, 30),
#           space_lists[2]: (1e-4, 1e-3)}

space_lists=['layers', 'nodes_per_layer']
pbounds = {space_lists[0]: (3, 10),
           space_lists[1]: (5, 50)}

cases = ["run_orig"] 
max_parallel_processes = 1

optimizer = BayesianOptimization(
    f=None,
    pbounds=pbounds,
    verbose=2,
    random_state=1,
)

utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

def make_integer(dict_):
    new_dict=dict_.copy()
    for key in dict_.keys():
        if key=='optimizer_step':
            break
        if key=='nodes_per_layer':
            dict_[key]=5 * round(dict_[key]/5)
        new_dict[key]=int(dict_[key])

    return (new_dict)

suggested=[]
history=pd.DataFrame()
header=pd.DataFrame(np.array([['iter', 'searching', 'target']]))
header.to_csv('history.csv', mode='w', header=False, index=False)


active_processes = []
for i in range(N_iter):
    print(i)
    next_point_to_probe=optimizer.suggest(utility)
    next_point_to_probe=make_integer(next_point_to_probe)

    # Construct the hiddenLayers option for pinnFoam
    hidden_layers = ("(")
    for layers in range(next_point_to_probe['layers']):
        hidden_layers = hidden_layers + str(next_point_to_probe['nodes_per_layer']) + " "
    hidden_layers = hidden_layers + ")"

    # Start the training
    call_list = ['pinnPotentialFoam',
        '-case', case,
        '-hiddenLayers', hidden_layers,
        '-maxIterations', max_iterations]
        #'-optimizerStep', str(next_point_to_probe['optimizer_step'])]

    call_string = " ".join(call_list)
    print(call_list)
    Det=True
    for j in range(len(suggested)):
        if suggested[j] == next_point_to_probe:
            print(suggested[j], next_point_to_probe)
            print('already did it')
            Det=False

    if Det:
        subproc=subprocess.Popen(call_list), #stdout=subprocess.DEVNULL)
        active_processes.append(subproc)
        subproc.communicate()

        time.sleep(3)
        file_name='pinnPotentialFoam-{:08d}.csv'.format(i)
        df = pd.read_csv(file_name)
        target_value=df['TRAINING_MSE'].iloc[-1]
        #print(next_point_to_probe)
        #print(target_value)

        print('Doing it')
        suggested.append(next_point_to_probe)
        optimizer.register(params=next_point_to_probe, target=-target_value)
    
    history=pd.DataFrame(np.array([[i, next_point_to_probe, target_value]]))
    history.to_csv('./history.csv', mode='a', header=False, index=False)

    # If max number of parallel processes is reached
    if (i % max_parallel_processes == 0):
        # Wait for active processes to finish
        for process in active_processes:
            process.wait()
        active_processes.clear()

print ("Done.")

df = pd.read_csv('history.csv')
print(df[['target']].idxmin())
