import numpy as np
from math import isclose

def bisect(function, lower:float, upper:float) ->  float:
    """finds the root of a function within the range specified.
    Note this function only finds one root, so define
    a good range to avoid problems.

    Args:
        function (callable): The equation to find the root of.
            takes only one parameter as input.
        lower (float): Lower bound on optimization range.
        upper (float): Upper bound on optimization range.

    Returns:
        float: value that minimizes the objective function. 
    """
    working_upper = upper
    working_lower = lower
    in_value = ((upper-lower)/2) + lower #begin at midpoint
    out_value = function(in_value)

    while not isclose(out_value,0.0):
        if out_value > 0:
            working_upper = in_value
        elif out_value < 0:
            working_lower = in_value
        else:
            print("ERROR: Invalid out_value in bisect!")
            exit(1)
        in_value = ((working_upper-working_lower)/2) + working_lower
        out_value = function(in_value)
    return in_value


def binary_entropy(p):
    """Binary entropy function h(p) = -p log p - (1-p) log(1-p)."""
    if p in [0,1]:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def fano_inequality_root(entropy_x_given_y, num_states):
    """Solve Fano's inequality for P(e) given H(X|Y) and M."""
    def equation(p):
        return binary_entropy(p) + p * np.log2(num_states - 1) - entropy_x_given_y # = 0

    # Solve for P(e) using bisection in range [0, 1]
    p_e = bisect(equation, 0, 1)
    return p_e

# Example Usage
entropy_game_given_feature = 0.5  # conditional entropy of the game given a feature (or features)
num_games = 4  # How many distinct games are we trying to predict?
probability_of_error = fano_inequality_root(entropy_game_given_feature, num_games)
print(f"Estimated P(e): {probability_of_error:.4f}")
