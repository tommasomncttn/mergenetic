import logging
import os
import pickle
from typing import List, Optional

import numpy as np
import requests
from scipy.optimize import minimize

from mergenetic import PROJECT_ROOT

logger = logging.getLogger(__name__)

# ==========================
#  UTILITY FUNCTIONS
# ==========================


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Computes the sigmoid function on the given array.

    Parameters
    ----------
    z : np.ndarray
        The input array on which to compute the sigmoid.

    Returns
    -------
    np.ndarray
        The element-wise sigmoid of z.
    """
    return 1 / (1 + np.exp(-z))


def item_curve(theta: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes the item response curve using a logistic (sigmoid) function,
    given ability parameters theta, discrimination A, and difficulty B.

    Parameters
    ----------
    theta : np.ndarray
        Ability parameter, shaped (1, D, 1) or (N, D, 1).
    A : np.ndarray
        Discrimination parameters, shaped (D, D, N) or (D, D, ...).
    B : np.ndarray
        Difficulty parameters, shaped (D, D, N) or (D, D, ...).

    Returns
    -------
    np.ndarray
        The probability of a correct response under the logistic model.
    """
    # Clip the linear combination to avoid numerical overflow in exp()
    z = np.clip(A * theta - B, -30, 30).sum(axis=1)
    return sigmoid(z)


def fit_theta(
    responses_test: np.ndarray,
    seen_items: List[int],
    A: np.ndarray,
    B: np.ndarray,
    theta_init: Optional[np.ndarray] = None,
    eps: float = 1e-10,
    optimizer: str = "BFGS",
) -> np.ndarray:
    """
    Fits the ability parameter theta by minimizing the negative log-likelihood.

    Parameters
    ----------
    responses_test : np.ndarray
        1D array of observed correct (1) / incorrect (0) responses.
    seen_items : List[int]
        Indices of the items that have been "seen" (answered).
    A : np.ndarray
        Discrimination parameters.
    B : np.ndarray
        Difficulty parameters.
    theta_init : np.ndarray, optional
        Initial guess for theta. Defaults to zeros if not provided.
    eps : float, optional
        A small constant to avoid log(0). Defaults to 1e-10.
    optimizer : str, optional
        The optimization method for scipy.minimize. Defaults to "BFGS".

    Returns
    -------
    np.ndarray
        The fitted theta parameter, shaped (1, D, 1).
    """
    D = A.shape[1]

    def neg_log_like(x):
        P = item_curve(
            x.reshape(1, D, 1), A[:, :, seen_items], B[:, :, seen_items]
        ).squeeze()
        ll = np.sum(
            responses_test[seen_items] * np.log(P + eps)
            + (1 - responses_test[seen_items]) * np.log(1 - P + eps)
        )
        return -ll

    if theta_init is None:
        theta_init = np.zeros(D)

    optimal_theta = minimize(neg_log_like, theta_init, method=optimizer).x
    return optimal_theta[None, :, None]


def fit_lambda(
    responses_test: np.ndarray,
    seen_items: List[int],
    A: np.ndarray,
    B: np.ndarray,
    thetas: List[np.ndarray],
    lambda_init: Optional[np.ndarray] = None,
    eps: float = 1e-10,
    optimizer: str = "BFGS",
) -> np.ndarray:
    """
    Finds the linear combination of multiple theta vectors that best fits
    the observed responses.

    Parameters
    ----------
    responses_test : np.ndarray
        1D array of observed correct (1) / incorrect (0) responses.
    seen_items : List[int]
        Indices of the items that have been "seen" (answered).
    A : np.ndarray
        Discrimination parameters.
    B : np.ndarray
        Difficulty parameters.
    thetas : List[np.ndarray]
        List of candidate theta parameters for linear combination.
    lambda_init : np.ndarray, optional
        Initial guess for the combination weights. Defaults to an even split.
    eps : float, optional
        Small constant for numerical stability. Defaults to 1e-10.
    optimizer : str, optional
        Optimization method. Defaults to "BFGS".

    Returns
    -------
    np.ndarray
        The optimal weights (lambdas) for combining the given thetas.
    """
    D = A.shape[1]

    def neg_log_like(lambdas):
        # Combine thetas with the given lambdas
        combined_theta = sum(l * t for l, t in zip(lambdas, thetas))
        P = item_curve(
            combined_theta.reshape(1, D, 1), A[:, :, seen_items], B[:, :, seen_items]
        ).squeeze()

        log_likelihood = np.sum(
            responses_test[seen_items] * np.log(P + eps)
            + (1 - responses_test[seen_items]) * np.log(1 - P + eps)
        )
        return -log_likelihood

    if lambda_init is None:
        lambda_init = np.ones(len(thetas)) / len(thetas)

    result = minimize(neg_log_like, lambda_init, method=optimizer)
    return result.x


# ==========================
#  TINYBENCHMARKS I/O
# ==========================


def download_tinybenchmarks_if_needed(
    filename: str = f"{PROJECT_ROOT}/tinyBenchmarks.pkl",
) -> None:
    """
    Downloads the tinyBenchmarks.pkl file from the repository if it is not present locally.

    Parameters
    ----------
    filename : str, optional
        Name of the local file to store the data. Defaults to "tinyBenchmarks.pkl".
    """
    if not os.path.isfile(filename):
        url = "https://raw.githubusercontent.com/felipemaiapolo/tinyBenchmarks/main/tinyBenchmarks/tinyBenchmarks.pkl"
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, "wb") as file:
                file.write(response.content)


def get_tinybenchmarks(filename: str = f"{PROJECT_ROOT}/tinyBenchmarks.pkl") -> dict:
    """
    Downloads the tinyBenchmarks.pkl file from the repository if it is not present locally.

    Parameters
    ----------
    filename : str, optional
        Local name of the tinyBenchmarks file. Defaults to "tinyBenchmarks.pkl".

    Returns
    -------
    dict
        The loaded dictionary containing tinyBenchmarks data.
    """
    download_tinybenchmarks_if_needed(filename)
    with open(filename, "rb") as handle:
        return pickle.load(handle)


# ==========================
#  TOP-LEVEL EVALUATION FUNCTIONS
# ==========================


def evaluate(
    y_input: np.ndarray, bench: str, tinyBenchmarks: Optional[dict] = None
) -> dict:
    """
    Main function that evaluates performance on the chosen benchmark using a combination
    of IRT-based scoring and anchor-based weighting.

    Parameters
    ----------
    y_input : np.ndarray
        A 1D array of binary outcomes (correct=1, incorrect=0) for seen examples.
    bench : str
        The name of the benchmark to evaluate. e.g., "gsm8k".
    tinyBenchmarks : dict, optional
        A loaded tinyBenchmarks dictionary. If None, it will be downloaded or loaded from disk.

    Returns
    -------
    dict
        Dictionary containing the computed estimates (IRT, PIRT, GPIRT) per scenario.
    """
    number_of_examples = 20
    lb_scenarios = ["truthfulqa", "gsm8k", "winogrande", "arc", "hellaswag"]
    benchs = ["lb", "mmlu", "helm_lite", "alpaca"]

    assert len(y_input.shape) == 1, "y_input must be a unidimensional numpy array."
    assert bench in benchs + lb_scenarios, f"Unknown benchmark '{bench}'."

    # If it's a "leaderboard scenario", map bench to "lb"
    bench_name = "lb" if bench in lb_scenarios else bench

    # Load tinyBenchmarks if needed
    if tinyBenchmarks is None:
        tinyBenchmarks = get_tinybenchmarks()

    # Extract relevant data
    data = tinyBenchmarks[bench_name]
    seen_examples = data["seen_examples"]
    examples_weights = data["examples_weights"]
    irt_parameters = data["irt_parameters"]
    A, B = irt_parameters["A"], irt_parameters["B"]
    optimal_lambdas = data["optimal_lambdas"]
    scenarios_position = data["scenarios_position"]
    subscenarios_position = data["subscenarios_position"]

    N = max(np.max(x) for x in scenarios_position.values()) + 1

    # Precompute scenario-based balancing
    balance_weights = np.ones(N)
    for scenario in scenarios_position.keys():
        N_sce = len(scenarios_position[scenario])
        n_sub = len(subscenarios_position[scenario])
        for sub in subscenarios_position[scenario].keys():
            n_i = len(subscenarios_position[scenario][sub])
            balance_weights[subscenarios_position[scenario][sub]] = N_sce / (
                n_sub * n_i
            )

    # If using the big IRT model for a single scenario
    if bench not in benchs:
        scenario_idx = [
            i for i, s in enumerate(scenarios_position.keys()) if s == bench
        ][0]
        start = number_of_examples * scenario_idx
        seen_examples = seen_examples[start : start + number_of_examples]
    else:
        # Evaluate all scenarios in "lb", "mmlu", etc.
        pass

    # Create y vector from the input data
    y = np.zeros(N)
    for i, ex_id in enumerate(seen_examples):
        y[ex_id] = y_input[i]

    # Fit user ability parameter
    theta = fit_theta(y, seen_examples, A, B)

    # Compute estimates for each scenario
    estimates = {}
    unseen_examples = [i for i in range(N) if i not in seen_examples]

    for scenario in scenarios_position.keys():
        # Scenario info
        N_sce = len(scenarios_position[scenario])
        seen_sce = [s for s in seen_examples if s in scenarios_position[scenario]]
        unseen_sce = [s for s in unseen_examples if s in scenarios_position[scenario]]

        data_part_IRTp = ((balance_weights * y)[seen_sce]).mean() if seen_sce else 0.0
        irt_part = (
            (balance_weights * item_curve(theta.reshape(1, A.shape[1], 1), A, B))[
                0, unseen_sce
            ].mean()
            if unseen_sce
            else 0.0
        )

        # Weighted blending
        IRTp_lambd = number_of_examples / N_sce
        IRTp = IRTp_lambd * data_part_IRTp + (1 - IRTp_lambd) * irt_part

        # Weighted sum of correct answers
        IRT = (examples_weights[scenario] * y[seen_sce]).sum() if seen_sce else 0.0

        # Combine with the scenario's optimal lambda
        IRTpp = optimal_lambdas[scenario] * IRT + (1 - optimal_lambdas[scenario]) * IRTp

        estimates[scenario] = {"irt": IRT, "pirt": IRTp, "gpirt": IRTpp}

    return estimates


def estimate_theta(
    y_input: np.ndarray,
    bench: str,
    number_of_examples: int = 100,
    tinyBenchmarks: Optional[dict] = None,
) -> np.ndarray:
    """
    Estimates the ability parameter theta for the chosen scenario in the LB set.

    Parameters
    ----------
    y_input : np.ndarray
        1D array of binary outcomes.
    bench : str
        The scenario in lb_scenarios, e.g., "gsm8k".
    number_of_examples : int, optional
        Number of examples used for the scenario. Defaults to 100.
    tinyBenchmarks : dict, optional
        Data dictionary. If None, it is loaded from disk or downloaded.

    Returns
    -------
    np.ndarray
        The estimated theta parameter, shaped (1, D, 1).
    """
    lb_scenarios = ["truthfulqa", "gsm8k", "winogrande", "arc", "hellaswag"]

    assert len(y_input.shape) == 1, "y_input must be unidimensional."
    assert bench in lb_scenarios, f"Benchmark '{bench}' not recognized."
    bench_name = "lb"  # For LB-based scenarios

    if tinyBenchmarks is None:
        tinyBenchmarks = get_tinybenchmarks()
        logger.info("Loaded official tinyBenchmarks in estimate_theta...")

    data = tinyBenchmarks[bench_name]
    A, B = data["irt_parameters"]["A"], data["irt_parameters"]["B"]
    seen_examples = data["seen_examples"]
    scenarios_position = data["scenarios_position"]

    # Identify the relevant scenario block
    scenario_idx = [i for i, s in enumerate(scenarios_position.keys()) if s == bench][0]
    start = number_of_examples * scenario_idx
    scenario_seen_examples = seen_examples[start : start + number_of_examples]

    # Shift to the scenario's index range
    shift = scenarios_position[bench][0]
    scenario_seen_examples = [ex + shift for ex in scenario_seen_examples]

    # Construct the y vector
    N = max(np.max(x) for x in scenarios_position.values()) + 1
    y = np.zeros(N)
    for i, ex_id in enumerate(scenario_seen_examples):
        y[ex_id] = y_input[i]

    return fit_theta(y, scenario_seen_examples, A, B)


def estimate_theta_linear(
    y_input: np.ndarray,
    bench: str,
    thetas: List[np.ndarray],
    seen_examples: List[int],
    tinyBenchmarks: Optional[dict] = None,
) -> np.ndarray:
    """
    Estimates a combined theta by finding the optimal linear combination of thetas
    that best fits the observed data.

    Parameters
    ----------
    y_input : np.ndarray
        1D array of binary outcomes (correct=1, incorrect=0).
    bench : str
        A scenario from the LB set (e.g., "gsm8k").
    thetas : List[np.ndarray]
        Candidate theta parameters.
    seen_examples : List[int]
        Indices of items that have been answered (seen).
    tinyBenchmarks : dict, optional
        Data dictionary. If None, it will be loaded from disk.

    Returns
    -------
    np.ndarray
        The combined theta array, shaped (1, D, 1).
    """
    lb_scenarios = ["truthfulqa", "gsm8k", "winogrande", "arc", "hellaswag"]

    assert len(y_input.shape) == 1, "y_input must be unidimensional."
    assert bench in lb_scenarios, f"Benchmark '{bench}' not recognized."
    bench_name = "lb"

    if tinyBenchmarks is None:
        tinyBenchmarks = get_tinybenchmarks()

    data = tinyBenchmarks[bench_name]
    A, B = data["irt_parameters"]["A"], data["irt_parameters"]["B"]
    scenarios_position = data["scenarios_position"]

    # Construct the y vector
    N = max(np.max(x) for x in scenarios_position.values()) + 1
    y = np.zeros(N)
    for i, ex_id in enumerate(seen_examples):
        y[ex_id] = y_input[i]

    # Fit the lambda weights
    lambdas_opt = fit_lambda(y, seen_examples, A, B, thetas)
    # Weighted sum of the candidate thetas
    combined_theta = sum(l * t for l, t in zip(lambdas_opt, thetas))

    return combined_theta


def estimate_theta_anchors(
    y_input: np.ndarray,
    bench: str,
    anchor_ids: List[int],
    tinyBenchmarks: Optional[dict] = None,
) -> np.ndarray:
    """
    Estimate the ability parameter (theta) using only the specified anchor IDs.

    Parameters
    ----------
    y_input : np.ndarray
        1D array of binary outcomes for anchor items.
    bench : str
        A scenario from the LB set (e.g., "gsm8k").
    anchor_ids : List[int]
        The local item indices used as anchors for fitting.
    tinyBenchmarks : dict, optional
        Data dictionary. If None, it will be loaded from disk.

    Returns
    -------
    np.ndarray
        The estimated theta array, shaped (1, D, 1).
    """
    lb_scenarios = ["truthfulqa", "gsm8k", "winogrande", "arc", "hellaswag"]
    assert len(y_input.shape) == 1, "y_input must be unidimensional."
    assert bench in lb_scenarios, f"Benchmark '{bench}' not recognized."

    bench_name = "lb"
    if tinyBenchmarks is None:
        tinyBenchmarks = get_tinybenchmarks()

    data = tinyBenchmarks[bench_name]
    A, B = data["irt_parameters"]["A"], data["irt_parameters"]["B"]
    scenarios_position = data["scenarios_position"]

    shift = np.array(scenarios_position[bench]).min()
    seen_examples = list(np.array(anchor_ids) + shift)

    N = max(np.max(x) for x in scenarios_position.values()) + 1
    y = np.zeros(N)
    for i, ex_id in enumerate(seen_examples):
        y[ex_id] = y_input[i]

    return fit_theta(y, seen_examples, A, B)


def eval_mpirt_on_anchors(
    y_input: np.ndarray,
    thetas: List[np.ndarray],
    bench: str,
    anchors_idx: List[int],
    tinyBenchmarks: dict = None,
    examples_weights: Optional[np.ndarray] = None,
) -> float:
    """
    Evaluate merged-performance IRT on anchor items, combining candidate thetas linearly.

    Parameters
    ----------
    y_input : np.ndarray
        1D array of binary outcomes on anchors.
    thetas : List[np.ndarray]
        Candidate thetas to combine.
    bench : str
        The scenario name (from lb_scenarios).
    anchors_idx : List[int]
        Local indices of anchor items.
    tinyBenchmarks : dict
        The loaded tinyBenchmarks data.
    examples_weights : np.ndarray, optional
        Weights for the anchor items. Defaults to None.

    Returns
    -------
    float
        The partial IRT estimate.
    """
    num_of_examples = y_input.shape[0]
    SCENARIOS = ["truthfulqa", "gsm8k", "winogrande", "arc", "hellaswag"]

    assert len(y_input.shape) == 1, "y_input must be a unidimensional numpy array."
    assert bench in SCENARIOS, f"Unknown scenario '{bench}'"

    if tinyBenchmarks is None:
        tinyBenchmarks = get_tinybenchmarks()

    bench_name = "lb"
    irt_params = tinyBenchmarks[bench_name]["irt_parameters"]
    A, B = irt_params["A"], irt_params["B"]
    scenarios_position = tinyBenchmarks[bench_name]["scenarios_position"]

    shift = scenarios_position[bench][0]
    seen_examples = [idx + shift for idx in anchors_idx]

    if examples_weights is None:
        # Possibly rely on some default weighting or user-supplied weights
        examples_weights = np.ones_like(y_input) / num_of_examples

    # Construct full y
    N = max(np.max(x) for x in scenarios_position.values()) + 1
    y = np.zeros(N)
    for i, ex_id in enumerate(seen_examples):
        y[ex_id] = y_input[i]

    # Combine or pick thetas
    if len(thetas) == 1:
        final_theta = thetas[0]
    else:
        final_theta = estimate_theta_linear(
            y_input, bench, thetas, seen_examples, tinyBenchmarks
        )

    # Identify seen/unseen for scenario
    unseen_examples = [i for i in range(N) if i not in seen_examples]
    unseen_sce = [s for s in unseen_examples if s in scenarios_position[bench]]
    # Weighted item-curve
    if len(unseen_sce) > 0:
        mpirt_part = (item_curve(final_theta.reshape(1, A.shape[1], 1), A, B))[
            0, unseen_sce
        ].mean()
    else:
        mpirt_part = 0.0

    # Weighted "observed" portion
    seen_sce = [s for s in seen_examples if s in scenarios_position[bench]]
    w_irt = (examples_weights * y_input).sum() if len(seen_sce) > 0 else 0.0

    # Weighted combination
    lambd = num_of_examples / len(scenarios_position[bench])
    result = lambd * w_irt + (1 - lambd) * mpirt_part
    return result


def estimate_fitness(
    y_input: np.ndarray,
    thetas: List[np.ndarray],
    bench: str,
    anchors_idx: List[int],
    example_weights: np.ndarray,
    tinybenchmarks: dict = None,
    delta: float = 0.5,
) -> dict:
    """
    Evaluate merge performance on anchors, returning mpirt, rc, and a combination gmpirt.

    Parameters
    ----------
    y_input : np.ndarray
        1D array of binary outcomes for anchor items.
    thetas : List[np.ndarray]
        Candidate thetas for combination.
    bench : str
        Scenario name from lb_scenarios.
    anchors_idx : List[int]
        Local indices of anchor items.
    example_weights : np.ndarray
        Weights for the anchor items.
    tinybenchmarks : dict
        The loaded tinyBenchmarks data.
    delta : float, optional
        Blend factor for combining anchor-based correctness and mpirt. Defaults to 0.5.

    Returns
    -------
    dict
        Contains "mpirt", "weighted_avg", and "gmpirt" measures.
    """
    mpirt_val = eval_mpirt_on_anchors(
        y_input,
        thetas,
        bench,
        anchors_idx,
        tinybenchmarks,
        examples_weights=example_weights,
    )
    weighted_avg = (example_weights * y_input).sum()
    gmpirt = delta * weighted_avg + (1 - delta) * mpirt_val

    names = ["mpirt", "gmpirt"] if len(thetas) > 1 else ["pirt", "gpirt"]

    return {names[0]: mpirt_val, "weighted_avg": weighted_avg, names[1]: gmpirt}
