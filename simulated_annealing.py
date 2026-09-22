import numpy as np
import torch

def extract_error_instructions_from_dem(dem):
    """
    Extract error instructions from dem. 

    The detector error model (dem) is a list of all the independent error mechanisms in a circuit
    as well as their symptoms (which detectors the fire/set off).

    A detector is a comparison between 2 ancillas measurment.

    Parameters
        ----------
        dem              : the detector error model
    
        Returns
        -------
        e_instructions : error instructions list
    """
    e_instructions = []

    for instr in dem.flattened(): #Flattened because we need to extract error from within the circuit rounds
        if instr.type == "error":
             e_instructions.append(instr)

    return e_instructions

def E_to_synds_and_label(dem, E):
    """
    Given a binary error pattern E over DEM error mechanisms,
    compute the resulting syndrome and logical observable by XOR-ing
    the effects of all active error mechanisms.

    Parameters
        ----------
        dem              : the detector error model
        E                : binary vector, shape (L,), one entry per DEM error mechanism

        Returns
        -------
        syndrome : binary array, shape (num_detectors,)
        label    : int in {0, 1}  — 1 if logical X error occurred
        L        : total number of error locations (len of E which is the len of dem error instructions )
    """
    num_detectors = dem.num_detectors # N_ancillas X N_rounds
    num_observables = dem.num_observables # 1

    synds = np.zeros(num_detectors, dtype=np.int8)
    logical = np.zeros(num_observables, dtype=np.int8)

    e_instructions = extract_error_instructions_from_dem(dem)

    for i, e_instr in enumerate(e_instructions):
        if E[i] == 1:
            # XOR the detectors this error flips. A target is a detector
            for target in e_instr.targets_copy():
                if target.is_relative_detector_id(): 
                    synds[target.val] ^= 1
                elif target.is_logical_observable_id():
                    
                    logical[target.val] ^= 1

    label = float(logical[0])  # 1 = logical X error occurred
    return synds.astype(np.float32), label

def attack(
    model,
    dem,
    criterion,
    W: int,                    # error budget. Hamming weight of E, fixed
    T0: float     = 2.0,       # initial annealing temperature
    T_min: float  = 0.1,       # stopping temperature
    cooling: float = 0.995,    # geometric cooling rate gamma
    steps_per_T: int = 20,     # Metropolis steps per temperature level
    n_restarts: int  = 5,      # independent SA chains
    seed: int        = 0,
):
    """
    Finds the error pattern E* in A_W = {E : |E| = W} that maximises
    the Binary Cross Entropy (BCE) loss of the decoder, i.e. the most adversarial X-error pattern
    within the given budget.

    Move set: swap (remove one active fault, add one inactive fault)
    → preserves |E| = W exactly throughout.

    Returns
    -------
    best_E      : binary array, shape (L,)  — worst-case error pattern
    best_loss   : float                     — BCE loss at best_E
    best_S      : float32 array             — syndrome produced by best_E
    best_label  : int                       — ground-truth label at best_E
    """
    rng = np.random.default_rng(seed)

    best_E     = None
    best_loss  = -np.inf
    best_S     = None
    best_label = None
    n_detectors = dem.num_detectors # N_ancillas X N_rounds

    for restart in range(n_restarts):

        # ── initialise: random pattern of exact weight W 
        L = len(extract_error_instructions_from_dem(dem))
        E = np.zeros(L, dtype=np.int8)
        init_idx = rng.choice(L, size=W, replace=False)
        E[init_idx] = 1

        S, label = E_to_synds_and_label(dem, E)

        #Transform the syndrome vector S and the ground truth label to fit in the model
        label = torch.tensor([label])
        S = torch.from_numpy(S)
        S = S.view(n_detectors)
        label = label.view(1)

        #Compute the BCE loss
        curr_loss  = criterion(model(S), label)

        local_best_E    = E.copy()
        local_best_loss = curr_loss
        local_best_S    = S.clone()
        local_best_label = label

        T = T0

         # ── annealing loop 
        while T > T_min:
            for _ in range(steps_per_T):

                # swap move: keeps |E| = W
                on_idx  = np.flatnonzero(E == 1)
                off_idx = np.flatnonzero(E == 0)
                i_remove = rng.choice(on_idx)
                i_add    = rng.choice(off_idx)

                E_new = E.copy()
                E_new[i_remove] = 0
                E_new[i_add]    = 1

                S_new, label_new = E_to_synds_and_label(
                    dem, E_new)

                #Transform the syndrome vector S and the ground truth label to fit in the model
                label_new = torch.tensor([label_new])
                S_new = torch.from_numpy(S_new)
                S_new = S_new.view(n_detectors)
                label_new = label_new.view(1)

                new_loss = criterion(model(S_new), label_new)

                # Metropolis–Hastings acceptance
                delta = new_loss - curr_loss
                delta = delta.item()
                if delta > 0 or rng.random() < np.exp(delta / T):
                    E, curr_loss = E_new, new_loss
                    S, label     = S_new, label_new

                    if curr_loss > local_best_loss:
                        local_best_E     = E.copy()
                        local_best_loss  = curr_loss
                        local_best_S     = S.clone()
                        local_best_label = label

            T *= cooling   # geometric cooling
            #print(local_best_loss.item(), '\n', local_best_S, '\n', local_best_label)

             # update global best across restarts
        if local_best_loss > best_loss:
            best_loss  = local_best_loss
            best_E     = local_best_E.copy()
            best_S     = local_best_S.clone()
            best_label = local_best_label

        if (restart + 1) % 1 == 0:
            print(f"  restart {restart+1}/{n_restarts} | "
                  f"best BCE loss so far: {best_loss:.4f}")

    return best_E, best_loss, best_S, best_label