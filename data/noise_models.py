"""
Author : Victor Fairon
Date : 2026-09-09

This file simply defines both types of noise model's parameters used in the generation of the datasets. The parameters can be 
"""

def generate_noise_constants(
        noise_type="Depolarizing",
        after_clifford_depolarization=None,
        after_reset_flip_probability=None,
        before_measure_flip_probability=None,
        before_round_data_depolarization=None,
        default=0.05,
        ):

    """
    Function that returns a dict with the noise model params that can direclty be used in the stim circuit generation.
    The function takes the four diffenrent noise model parameters as input, and if they are not specified, it will use the default value (also given as argument) for the parameters needed for the specified noise model type. The function will raise an error if the noise_type is not recognized.

    @args:
        noise_type : str, optional
            The type of noise model to use. Can be either "Depolarizing" or "CircuitLevel". Default is "Depolarizing".
        after_clifford_depolarization : float, optional
            The depolarization probability after each Clifford gate. Only used if noise_type is "CircuitLevel". Default is None.
        after_reset_flip_probability : float, optional
            The flip probability after each qubit reset. Only used if noise_type is "CircuitLevel". Default is None.
        before_measure_flip_probability : float, optional
            The flip probability before each qubit measurement. Only used if noise_type is "CircuitLevel  ". Default is None.
        before_round_data_depolarization : float, optional
            The depolarization (uniform on X, Y, Z) probability before each round of data qubit operations. Used in both noise models. Default is None.
        default : float, optional
            The default value to use for any unspecified noise model parameters. Default is 0.05
    @returns:
        dict
            A dictionary containing the noise model parameters, with keys corresponding to the parameter names and values corresponding to the specified or default values. 
    
    """
    def d(v):
        return default if v is None else v

    #Use only before_round_data_depolarization with default value parameter
    if noise_type == "Depolarizing":
        return {
            "before_round_data_depolarization": d(before_round_data_depolarization),
            # "before_measure_flip_probability": d(before_measure_flip_probability), 
        }
    elif noise_type == "CircuitLevel":
        return {
            "after_clifford_depolarization": d(after_clifford_depolarization),
            "after_reset_flip_probability": d(after_reset_flip_probability),
            "before_measure_flip_probability": d(before_measure_flip_probability),
            "before_round_data_depolarization": d(before_round_data_depolarization),
        }
    else:
        raise ValueError(f"unknown noise_type: {noise_type}")