from typing import List
import numpy as np
import pickle
import os
import requests
from scipy.optimize import minimize
from mergenetic import PROJECT_ROOT

import logging
logger = logging.getLogger(__name__)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def item_curve(theta, a, b):
    z = np.clip(a*theta - b, -30, 30).sum(axis=1)
    return sigmoid(z)

# function for loading pandas DataFrame of evaluation results
def load_results(results_fname):
    import pandas as pd
    from pathlib import Path

    path = Path("../experiments/0.1-evaluate-base-gsm8k/results/") / results_fname

    if path.exists():
        return pd.read_csv(path)
    else:
        logger.info(f"File {path} not found.")
        return None

# %%
### NEW FUNCTIONS

def get_tinybenchmarks_from_path(path):
    """
    Returns the opened pkl file.
    """
    import pickle
    
    with open(path, 'rb') as handle:
        tinyBenchmarks = pickle.load(handle)
    
    return tinyBenchmarks

def get_tinybenchmarks():
    """
    Downloads the tinyBenchmarks.pkl file from the repository and return the opened pkl file.
    """
    try: 
        import pickle
        path = str(PROJECT_ROOT) + "/tinyBenchmarks.pkl"
        with open(path, 'rb') as handle: tinyBenchmarks = pickle.load(handle) 
        return tinyBenchmarks
    except:
        import requests
        import pickle
        import os

        if not os.path.isfile("tinyBenchmarks.pkl"):
            url = "https://raw.githubusercontent.com/felipemaiapolo/tinyBenchmarks/main/tinyBenchmarks/tinyBenchmarks.pkl"
            response = requests.get(url)
            if response.status_code == 200:
                # Write the content to a file
                with open("tinyBenchmarks.pkl", "wb") as file:
                    file.write(response.content)
        
        with open('tinyBenchmarks.pkl', 'rb') as handle:
            tinyBenchmarks = pickle.load(handle)
        
        return tinyBenchmarks

def calculate_example_weights(bench_name, scenarios_position):
    """
    Calculate the example weights for the specified benchmark.
    """
    # Initialize the dictionary to store the example weights
    examples_weights = {}
    for scenario in scenarios_position.keys():
        N_sce = len(scenarios_position[scenario])
        examples_weights[scenario] = 1/N_sce
    return examples_weights

def estimate_theta(y_input, bench, number_of_examples=100, tinyBenchmarks=None):
    # List of scenarios used in the leaderboard evaluation and other benchmarks
    lb_scenarios = ['truthfulqa', 'gsm8k', 'winogrande', 'arc', 'hellaswag']
    # Ensure the input `y_input` is a 1D numpy array
    assert len(y_input.shape) == 1, "y_input must be a unidimensional numpy array."
    # Ensure the provided `bench` is in either the leaderboard scenarios or benchmark list
    assert bench in lb_scenarios
    # Assign the correct benchmark name based on whether it is in the leaderboard scenarios or not
    bench_name = 'lb'
    
    # load the data for estimation
    if tinyBenchmarks is None:
        tinyBenchmarks = get_tinybenchmarks()
        logger.info("Loaded official tinyBenchmarks...")

    # Extract key variables from the loaded benchmark data
    irt_parameters = tinyBenchmarks[bench_name]['irt_parameters']     
    A, B = irt_parameters['A'], irt_parameters['B']
    seen_examples = tinyBenchmarks[bench_name]['seen_examples']

    # Get the index of the scenario in the list of scenarios
    scenarios_position = tinyBenchmarks[bench_name]['scenarios_position'] # still useful

    # assert we have all the answers
    ind_scenario = number_of_examples*([i for i,s in enumerate(scenarios_position.keys()) if s==bench][0])
    seen_examples = seen_examples[ind_scenario:ind_scenario+number_of_examples]
    # shift the examples to the scenario
    seen_examples = [s + scenarios_position[bench][0] for s in seen_examples]

    ### Creating the response vector `y` and populating with values from `y_input`
    y = np.zeros(np.max([np.max(x) for x in scenarios_position.values()])+1)
    for i, j in enumerate(seen_examples):
        # Assign values from `y_input` to the corresponding seen examples
        y[j] = y_input[i]

    ### Estimating the theta parameter using the fitted IRT model
    theta = fit_theta(y, seen_examples, A, B)
    
    return theta

def fit_theta(responses_test, seen_items, A, B, theta_init=None, eps=1e-10, optimizer="BFGS"):
    D = A.shape[1]
    # Define the negative log likelihood function
    def neg_log_like(x):
        P = item_curve(x.reshape(1, D, 1), A[:, :, seen_items], B[:, :, seen_items]).squeeze()
        log_likelihood = np.sum(responses_test[seen_items] * np.log(P + eps) + (1 - responses_test[seen_items]) * np.log(1 - P + eps))
        return -log_likelihood
    # Use the minimize function to find the ability parameters that minimize the negative log likelihood
    optimal_theta = minimize(neg_log_like, np.zeros(D), method = optimizer).x[None,:,None]
    return optimal_theta

def fit_lambda(responses_test, seen_items, A, B, thetas: List[np.array], lambda_init=None, eps=1e-10, optimizer="BFGS"):
    D = A.shape[1]  # Number of ability dimensions

    # Define the negative log-likelihood function
    def neg_log_like(lambdas):
        # Compute theta as a linear combination of theta1 and theta2
        est_thetas = [lambdas[i] * thetas[i] for i in range(len(thetas))]
        theta = np.sum(est_thetas, axis=0)

        # Compute the probabilities using the item_curve function
        P = item_curve(theta.reshape(1, D, 1), A[:, :, seen_items], B[:, :, seen_items]).squeeze()

        # Compute the negative log-likelihood
        log_likelihood = np.sum(
            responses_test[seen_items] * np.log(P + eps) +
            (1 - responses_test[seen_items]) * np.log(1 - P + eps)
        )
        return -log_likelihood  # We minimize the negative log-likelihood

    # Initial guess for lambdas if not provided
    if lambda_init is None:
        #lambda_init = np.zeros(len(thetas))
        lambda_init = np.ones(len(thetas)) / len(thetas)

    # Use the minimize function to find the lambdas that minimize the negative log-likelihood
    result = minimize(neg_log_like, lambda_init, method=optimizer)
    optimal_lambdas = result.x  # Optimal values of lambda1 and lambda2
    # softmax normalization
    #optimal_lambdas = np.exp(optimal_lambdas) / np.sum(np.exp(optimal_lambdas))

    return optimal_lambdas

def estimate_theta_linear(y_input, bench, thetas: List[np.array], seen_examples: List[int], tinyBenchmarks=None):
    # List of scenarios used in the leaderboard evaluation and other benchmarks
    lb_scenarios = ['truthfulqa', 'gsm8k', 'winogrande', 'arc', 'hellaswag']
    # Ensure the input `y_input` is a 1D numpy array
    assert len(y_input.shape) == 1, "y_input must be a unidimensional numpy array."
    # Ensure the provided `bench` is in the list of scenarios
    assert bench in lb_scenarios, f"Benchmark '{bench}' not recognized."
    # Assign the correct benchmark name based on whether it is in the leaderboard scenarios or not
    bench_name = 'lb'

    # Load the data for estimation
    if tinyBenchmarks is None:
        tinyBenchmarks = get_tinybenchmarks()
        logger.info("Loaded official tinyBenchmarks in estimate_theta_linear...")

    # Extract key variables from the loaded benchmark data
    irt_parameters = tinyBenchmarks[bench_name]['irt_parameters']     
    A, B = irt_parameters['A'], irt_parameters['B']

    scenarios_position = tinyBenchmarks[bench_name]['scenarios_position']

    ### Creating the response vector `y` and populating it with values from `y_input`
    num_items = np.max([np.max(x) for x in scenarios_position.values()]) + 1
    y = np.zeros(num_items)
    for i, j in enumerate(seen_examples):
        # Assign values from `y_input` to the corresponding seen examples
        y[j] = y_input[i]

    ### Estimating the lambda parameters using the fitted IRT model
    optimal_lambdas = fit_lambda(y, seen_examples, A, B, thetas)
    
    # Compute the estimated theta using the optimal lambdas
    thetas_estimated = [optimal_lambdas[i] * thetas[i] for i in range(len(thetas))]

    return np.sum(thetas_estimated, axis=0)

def estimate_theta_anchors(y_input, bench, anchor_ids: List[int], tinyBenchmarks=None):
    # List of scenarios used in the leaderboard evaluation and other benchmarks
    lb_scenarios = ['truthfulqa', 'gsm8k', 'winogrande', 'arc', 'hellaswag']
    # Ensure the input `y_input` is a 1D numpy array
    assert len(y_input.shape) == 1, "y_input must be a unidimensional numpy array."
    # Ensure the provided `bench` is in the list of scenarios
    assert bench in lb_scenarios, f"Benchmark '{bench}' not recognized."
    # Assign the correct benchmark name based on whether it is in the leaderboard scenarios or not
    bench_name = 'lb'

    # Load the data for estimation
    if tinyBenchmarks is None:
        tinyBenchmarks = get_tinybenchmarks()

    # Extract key variables from the loaded benchmark data
    irt_parameters = tinyBenchmarks[bench_name]['irt_parameters']     
    A, B = irt_parameters['A'], irt_parameters['B']

    # Get the indices of the seen examples for the specified benchmark
    shift = np.array(tinyBenchmarks[bench_name]['scenarios_position'][bench]).min()
    seen_examples = np.array(anchor_ids) + shift

    scenarios_position = tinyBenchmarks[bench_name]['scenarios_position']

    ### Creating the response vector `y` and populating it with values from `y_input`
    num_items = np.max([np.max(x) for x in scenarios_position.values()]) + 1
    y = np.zeros(num_items)
    for i, j in enumerate(seen_examples):
        # Assign values from `y_input` to the corresponding seen examples
        y[j] = y_input[i]

    optimal_thetas = fit_theta(y, seen_examples, A, B)

    return optimal_thetas

def is_approx_linear_combination(v1, v2, v3, tol=1e-5):
    """
    Check if vector v3 is approximately a linear combination of vectors v1 and v2.
    
    Args:
    v1 (numpy array): First vector of floats.
    v2 (numpy array): Second vector of floats.
    v3 (numpy array): Third vector of floats to check for linear combination.
    tol (float): Tolerance level for the approximation. Default is 1e-5.
    
    Returns:
    bool: True if v3 is approximately a linear combination of v1 and v2, False otherwise.
    """
    v1 = v1[0,:,0]
    v2 = v2[0,:,0]
    v3 = v3[0,:,0]
    # Stack v1 and v2 into a matrix (column-wise)
    A = np.vstack([v1, v2]).T  # Transpose to create a 2-column matrix
    
    # Solve the least squares problem to find the coefficients (alpha, beta)
    coeffs, residuals, rank, s = np.linalg.lstsq(A, v3, rcond=None)
    
    # Reconstruct the approximation of v3
    v3_approx = coeffs[0] * v1 + coeffs[1] * v2
    
    # Check if the approximation is within the tolerance
    return np.allclose(v3_approx, v3, atol=tol)

def estimate_theta_subset(y_input, bench):
    
    number_of_examples = 20
    lb_scenarios = ['truthfulqa', 'gsm8k', 'winogrande', 'arc', 'hellaswag']
    benchs = ['lb', 'mmlu', 'helm_lite', 'alpaca']
    
    assert len(y_input.shape)==1, "y_input must be a unidimensional numpy array."
    assert bench in benchs + lb_scenarios
    
    if bench in lb_scenarios: bench_name = 'lb'
    else: bench_name = bench
        
    # Downloading files
    if not os.path.isfile("tinyBenchmarks.pkl"):
        url = "https://raw.githubusercontent.com/felipemaiapolo/tinyBenchmarks/main/tinyBenchmarks/tinyBenchmarks.pkl"
        response = requests.get(url)
        if response.status_code == 200:
            # Write the content to a file
            with open("tinyBenchmarks.pkl", "wb") as file:
                file.write(response.content)

    ### Loading and creating important objects
    with open('tinyBenchmarks.pkl', 'rb') as handle:
        tinyBenchmarks = pickle.load(handle)

    seen_examples = tinyBenchmarks[bench_name]['seen_examples']
    examples_weights = tinyBenchmarks[bench_name]['examples_weights']
    irt_parameters = tinyBenchmarks[bench_name]['irt_parameters']
    A, B = irt_parameters['A'], irt_parameters['B']
    optimal_lambdas = tinyBenchmarks[bench_name]['optimal_lambdas']
    scenarios_position = tinyBenchmarks[bench_name]['scenarios_position']
    subscenarios_position = tinyBenchmarks[bench_name]['subscenarios_position']

    N = np.max([np.max(x) for x in scenarios_position.values()])+1
    balance_weights = np.ones(N)
    for scenario in scenarios_position.keys():
        N_sce = len(scenarios_position[scenario])
        n_sub = len(subscenarios_position[scenario])
        for sub in subscenarios_position[scenario].keys():
            n_i = len(subscenarios_position[scenario][sub])
            balance_weights[subscenarios_position[scenario][sub]] = N_sce/(n_sub*n_i) 

    ### In case we use the big IRT model to estimate the performance of individual scenarios
    if bench not in benchs:
        scenarios = [bench]
        ind_scenario = number_of_examples*([i for i,s in enumerate(scenarios_position.keys()) if s==bench][0])
        seen_examples = seen_examples[ind_scenario:ind_scenario+number_of_examples]
    else:
        scenarios = list(scenarios_position.keys())
        
    ### Creating vector y and estimating theta
    y = np.zeros(N)
    for i, j in enumerate(seen_examples):
        y[j] = y_input[i]

    ### Getting estimates
    theta = fit_theta(y, seen_examples, A, B)
    return theta

def get_estimate_theta_subset(y_superset, benchmark, head_size = 20):
    # Load the data for estimation
    tinyBenchmarks = get_tinybenchmarks()
    anchor_points = tinyBenchmarks['lb']['seen_examples']
    all_benchmark = tinyBenchmarks['lb']['scenarios_position'][benchmark]
    anchor_points_benchmark = [i - all_benchmark[0] for i in anchor_points if i in all_benchmark]
    y = y_superset.iloc[anchor_points_benchmark]['is_answer_correct'].head(head_size).values

    theta = estimate_theta_subset(y, benchmark)
    return theta

def get_estimate_lambda_subset(y_superset, benchmark, theta1, theta2, theta3, head_size = 20):
    # Load the data for estimation
    tinyBenchmarks = get_tinybenchmarks()
    anchor_points = tinyBenchmarks['lb']['seen_examples']
    all_benchmark = tinyBenchmarks['lb']['scenarios_position'][benchmark]
    anchor_points_benchmark = [i - all_benchmark[0] for i in anchor_points if i in all_benchmark]
    y = y_superset.iloc[anchor_points_benchmark]['is_answer_correct'].head(head_size).values

    theta = estimate_theta_linear(y, benchmark, [theta1, theta2, theta3], ids=anchor_points_benchmark)
    return theta

def euclidean_distance(theta1, theta2):
    # Step 1: Flatten the arrays to 1D vectors
    vector1 = theta1.flatten()
    vector2 = theta2.flatten()

    # Step 2: Compute the difference between the vectors
    difference = vector1 - vector2

    # Step 3: Compute the Euclidean distance
    euclidean_distance = np.linalg.norm(difference)
    return euclidean_distance

def eval_mpirt_on_anchors(y_input, thetas:List[np.array], bench, anchors_idx, tinyBenchmarks, examples_weights=None):
    num_of_examples = y_input.shape[0]
    SCENARIOS = ['truthfulqa', 'gsm8k', 'winogrande', 'arc', 'hellaswag']

    assert len(y_input.shape)==1, "y_input must be a unidimensional numpy array."
    assert bench in SCENARIOS

    bench_name = 'lb'

    if tinyBenchmarks is None:
        tinyBenchmarks = get_tinybenchmarks()
    
    irt_parameters = tinyBenchmarks[bench_name]['irt_parameters']
    A, B = irt_parameters['A'], irt_parameters['B']
    scenarios_position = tinyBenchmarks[bench_name]['scenarios_position']

    # seen_examples = tinyBenchmarks[bench_name]['seen_examples']
    # ind_scenario = num_of_examples*([i for i,s in enumerate(scenarios_position.keys()) if s==bench][0])
    # seen_examples = seen_examples[ind_scenario:ind_scenario+num_of_examples]
    seen_examples = anchors_idx

    # shift the examples to the scenario
    seen_examples = [s + scenarios_position[bench][0] for s in seen_examples]
    
    if examples_weights is None:
        examples_weights = tinyBenchmarks[bench_name]['examples_weights'][scenario]

    subscenarios_position = tinyBenchmarks[bench_name]['subscenarios_position']

    N = np.max([np.max(x) for x in scenarios_position.values()])+1
    balance_weights = np.ones(N)
    for scenario in scenarios_position.keys():
        N_sce = len(scenarios_position[scenario])
        n_sub = len(subscenarios_position[scenario])
        for sub in subscenarios_position[scenario].keys():
            n_i = len(subscenarios_position[scenario][sub])
            balance_weights[subscenarios_position[scenario][sub]] = N_sce/(n_sub*n_i)

    y = np.zeros(N)
    for i, j in enumerate(seen_examples):
        y[j] = y_input[i]

    if len(thetas) == 1:
        thetas_interp = thetas[0]
    else:
        thetas_interp = estimate_theta_linear(y_input, bench, thetas, seen_examples=seen_examples, tinyBenchmarks=tinyBenchmarks)
    unseen_examples = [i for i in range(N) if i not in seen_examples]

    N_sce = len(scenarios_position[bench])
    unseen_examples_sce = [s for s in unseen_examples if s in scenarios_position[bench]]

    seen_examples_sce = [s for s in seen_examples if s in scenarios_position[bench]]
    
    #data_part_IRTp = ((balance_weights*y)[seen_examples]).mean()
    irt_part = (balance_weights*item_curve(thetas_interp.reshape(1, A.shape[1], 1), A, B))[0, [unseen_examples_sce]].mean()
    IRT = (examples_weights*y[seen_examples_sce]).sum()
    IRTp_lambd = num_of_examples/N_sce
    IRTp = IRTp_lambd * IRT + (1 - IRTp_lambd) * irt_part

    return IRTp

def tb_eval_mpirt(y_input, thetas:List[np.array], bench, anchors_idx):
    lb_scenarios = ['truthfulqa', 'gsm8k', 'winogrande', 'arc', 'hellaswag']
    benchs = ['lb', 'mmlu', 'helm_lite', 'alpaca']

    number_of_examples = y_input.shape[0]
    
    assert len(y_input.shape)==1, "y_input must be a unidimensional numpy array."
    assert bench in benchs + lb_scenarios
    
    if bench in lb_scenarios: bench_name = 'lb'
    else: bench_name = bench

    tinyBenchmarks = get_tinybenchmarks()
    
    irt_parameters = tinyBenchmarks[bench_name]['irt_parameters']
    A, B = irt_parameters['A'], irt_parameters['B']
    scenarios_position = tinyBenchmarks[bench_name]['scenarios_position']
    subscenarios_position = tinyBenchmarks[bench_name]['subscenarios_position']
    seen_examples = anchors_idx
    seen_examples = [s + scenarios_position[bench][0] for s in seen_examples]

    N = np.max([np.max(x) for x in scenarios_position.values()])+1
    balance_weights = np.ones(N)
    for scenario in scenarios_position.keys():
        N_sce = len(scenarios_position[scenario])
        n_sub = len(subscenarios_position[scenario])
        for sub in subscenarios_position[scenario].keys():
            n_i = len(subscenarios_position[scenario][sub])
            balance_weights[subscenarios_position[scenario][sub]] = N_sce/(n_sub*n_i) 

    ### In case we use the big IRT model to estimate the performance of individual scenarios
    if bench not in benchs:
        scenarios = [bench]
    else:
        scenarios = list(scenarios_position.keys())

    subscenarios_position = tinyBenchmarks[bench_name]['subscenarios_position']

    N = np.max([np.max(x) for x in scenarios_position.values()])+1
    balance_weights = np.ones(N)
    for scenario in scenarios_position.keys():
        N_sce = len(scenarios_position[scenario])
        n_sub = len(subscenarios_position[scenario])
        for sub in subscenarios_position[scenario].keys():
            n_i = len(subscenarios_position[scenario][sub])
            balance_weights[subscenarios_position[scenario][sub]] = N_sce/(n_sub*n_i)

    y = np.zeros(N)
    for i, j in enumerate(anchors_idx):
        y[j] = y_input[i]

    if len(thetas) == 1:
        thetas_interp = thetas[0]
    else:
        thetas_interp = estimate_theta_linear(y_input, bench, thetas, seen_examples=seen_examples, tinyBenchmarks=tinyBenchmarks)
    unseen_examples = [i for i in range(N) if i not in seen_examples]
    estimates = {}

    for scenario in scenarios:

        N_sce = len(scenarios_position[scenario])
        seen_examples_sce = [s for s in seen_examples if s in scenarios_position[scenario]]
        unseen_examples_sce = [s for s in unseen_examples if s in scenarios_position[scenario]]

        N_sce = len(scenarios_position[bench])
        unseen_examples_sce = [s for s in unseen_examples if s in scenarios_position[bench]]

        seen_examples_sce = [s for s in seen_examples if s in scenarios_position[bench]]
        
        #data_part_IRTp = ((balance_weights*y)[seen_examples]).mean()
        irt_part = (balance_weights*item_curve(thetas_interp.reshape(1, A.shape[1], 1), A, B))[0, [unseen_examples_sce]].mean()
        IRT = (y[seen_examples_sce]).mean()
        IRTp_lambd = number_of_examples/N_sce
        IRTp = IRTp_lambd * IRT + (1 - IRTp_lambd) * irt_part

        estimates[scenario] = {}
        estimates[scenario]['irt'] = IRT
        estimates[scenario]['mpirt'] = IRTp

    return estimates

def evaluate_merge_on_anchors(y_input, thetas:List[np.array], bench, anchors_idx, example_weights, tinybenchmarks, delta=0.5):
    mpirt = eval_mpirt_on_anchors(y_input, thetas, bench, anchors_idx, tinybenchmarks, examples_weights=example_weights)
    rc = (example_weights * y_input).sum()
    gmpirt = delta * rc + (1 - delta) * mpirt

    return {
        "mpirt": mpirt,
        "rc": rc,
        "gmpirt": gmpirt
    }

def tb_evaluate_merge(y_input, thetas:List[np.array], bench, anchors_idx, delta=0.5):
    estimates = tb_eval_mpirt(y_input, thetas, bench, anchors_idx)
    
    if len(list(estimates.keys())) == 1:
        mpirt = estimates[bench]['mpirt']
        random = y_input.mean()
        gmpirt = delta * random + (1 - delta) * mpirt
        irt = estimates[bench]['irt']
    else:
        mpirt_values = [estimates[scenario]['mpirt'] for scenario in list(estimates.keys())]
        random = y_input.mean()
        gmpirt_values = [delta * random + (1 - delta) * mpirt for mpirt in mpirt_values]
        gmpirt = np.mean(gmpirt_values)
        irt_values = [estimates[scenario]['irt'] for scenario in list(estimates.keys())]
        irt = np.mean(irt_values)

    return {
        "irt": irt,
        "mpirt": mpirt,
        "random": random,
        "gmpirt": gmpirt
    }

def evaluate_on_anchors(y_input, bench, anchors_idx, tinyBenchmarks, anchor_weights=None):
    '''
    Evaluate the performance of the model on the specified anchors.
    :param y_input: numpy array of model predictions
    :param bench: benchmark name
    :param anchors_idx: list of indices of the anchor examples
    :param anchor_weights: numpy array of weights for the anchor examples
    '''
    num_of_examples = y_input.shape[0]
    SCENARIOS = ['truthfulqa', 'gsm8k', 'winogrande', 'arc', 'hellaswag']

    assert len(y_input.shape)==1, "y_input must be a unidimensional numpy array."
    assert bench in SCENARIOS

    bench_name = 'lb'

    if tinyBenchmarks is None:
        tinyBenchmarks = get_tinybenchmarks()
    
    if anchor_weights is None:
        examples_weights = np.full(num_of_examples, 1/num_of_examples)
    else:
        examples_weights = anchor_weights #

    irt_parameters = tinyBenchmarks[bench_name]['irt_parameters']
    A, B = irt_parameters['A'], irt_parameters['B']
    optimal_lambdas = tinyBenchmarks[bench_name]['optimal_lambdas']
    scenarios_position = tinyBenchmarks[bench_name]['scenarios_position']

    seen_examples = [i + scenarios_position[bench][0] for i in anchors_idx]

    N = np.max([np.max(x) for x in scenarios_position.values()])+1

    y = np.zeros(N)
    for i, j in enumerate(seen_examples):
        y[j] = y_input[i]

    theta = fit_theta(y, seen_examples, A, B)
    estimates = {}
    unseen_examples = [i for i in range(N) if i not in seen_examples]

    N_sce = len(scenarios_position[bench])
    seen_examples_sce = [s for s in seen_examples if s in scenarios_position[bench]]
    unseen_examples_sce = [s for s in unseen_examples if s in scenarios_position[bench]]

    balance_weights = np.ones(N)
    data_part_IRTp = ((balance_weights*y)[seen_examples_sce]).mean()
    irt_part = (balance_weights*item_curve(theta.reshape(1, A.shape[1], 1), A, B))[0, [unseen_examples_sce]].mean()
    IRTp_lambd = num_of_examples/N_sce
    IRTp = IRTp_lambd * data_part_IRTp + (1 - IRTp_lambd) * irt_part
    
    wIRT = (examples_weights*y[seen_examples_sce]).sum() # weighted IRT, FIXME: we don't have these weights
    IRTpp = optimal_lambdas[bench]*wIRT + (1-optimal_lambdas[bench])*IRTp

    estimates[bench] = {}
    estimates[bench]['irt'] = wIRT # not significant at the moment, since we don't have the weights
    estimates[bench]['pirt'] = IRTp
    estimates[bench]['gpirt'] = IRTpp

    return estimates