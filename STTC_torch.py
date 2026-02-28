import torch
import numpy as np

def generate_spike_train(N, T, sparsity):
    '''
    Creates an array of spike trains of size (N,T) at a given spike density (sparsity, or total spikes/length of signal)

    returns spike_train (np.array)
    '''
    
    # Check if sparsity is between 0 and 1
    if not (0 <= sparsity <= 1):
        raise ValueError("Spike density must be between 0 and 1.")
    if sparsity == 0:
        return np.zeros((N, T), dtype=int)
    elif sparsity == 1:
        return np.ones((N, T), dtype=int)
        
    num_non_zero = int(N * T * sparsity)
    indices = np.random.choice(N * T, num_non_zero, replace=False)
    spike_train = np.zeros((N, T), dtype=int)
    spike_train.flat[indices] = 1

    return spike_train
    
def STTC_formula(TA, TB, PA, PB):
    '''
    Formula of STTC from the Cutts and Eglen (2014) Paper
    '''
    x = (PA - TB)/(1 - (PA*TB))
    y = (PB - TA)/(1 - (PB*TA))
    STTC = 0.5*(x + y)
    return STTC

def add_dt(signal, dt=0, device='cuda'):
    '''
    Create tiled signal. Make 0->1 in the surrounding (+/- dt) values around each spike (1)
    This is executed using a 1-d convolution with a rectangle window across each row in the signal.

    For example:
    for dt = 1,
    signal = [1 0 0 0 1 0 0 0 0 1 1 0 0]
    output = [1 1 0 1 1 1 0 0 1 1 1 1 0]

    Inputs
    - signal : (torch.Tensor). Can be 1-d or 2-d array of signals. Output is the same type of Tensor as input. 
                Only put in signals in the format (n_cells, n_samples in time)
    - dt : (int, default = 0)

    Output
    - convolved_signal (torch.Tensor) in the same type and shape as input Tensor.
    '''
    if not torch.is_tensor(signal):
        raise ValueError("Input must be a PyTorch tensor.")

    
    if dt == 0:
        return signal
    
    change_ndim = False
    if signal.ndim==1:
        signal = signal.view(1,-1)
        change_ndim = True


    # Pad the signal
    padded_signal = torch.nn.functional.pad(signal, (dt, dt), value=0)

    # Create a window with ones of size (2*dt+1)
    N,T = signal.shape
    window = torch.ones(N, 2 * dt + 1, device=device, dtype=signal.dtype)

    # Convolve the padded signal with the window
    convolved_signal = torch.nn.functional.conv1d(padded_signal, 
                                                  window.unsqueeze(1), padding=dt, groups=N)

    if change_ndim:
        return ((convolved_signal > 0)*1)[0, dt:-dt].type(signal.dtype)
    return ((convolved_signal > 0)*1)[:, dt:-dt].type(signal.dtype)



def get_STTC(signals, dt=0, ret_all = False):
    '''
    Calculates pairwise STTC of all rows in the signals array.

    Inputs
    - signals : (torch.Tensor). 2-d array of signals.
                Only put in signals in the format (n_cells, n_samples in time)
    - dt : (int, default = 0)
    - ret_all : (boolean, default = False) if set to True, returns dictionary with the intermediate calculations of 
                                            TA, TB, PA and PB along with STTC

    Outputs
    - STTC : (torch.Tensor) retruns a (n_cells x n_cells) matrix with each cell giving the paired STTC calculated at set dt.
    - TA, TB, PA, PB : intermediate calculated values that go into the STTC.
    '''
    if not torch.is_tensor(signals):
        raise ValueError("Input must be a PyTorch tensor.")

    if not signals.is_floating_point():
        raise ValueError("Input must be a float tensor, dtype=torch.float, torch.float32 or torch.float64.")
        
    N,L = signals.shape
    dt_signals = add_dt(signals, dt=dt)
    NB = torch.tile(signals.sum(axis=1), (N,1))
    TB = torch.tile(dt_signals.mean(axis=1), (N,1))
    TA = TB.T
    
    DAB = signals @ dt_signals.T

    PA = DAB/(NB.T)
    PB = DAB.T/(NB)
    
    STTC = STTC_formula(TA, TB, PA, PB)
    
    if ret_all:
        return dict(zip(['STTC', 'TA', 'TB', 'PA', 'PB'], [STTC, TA, TB, PA, PB]))
    
    return STTC

def shift_signal(signals, shifts=None, shuffle=False):
    '''
    shuffle or circular shifts signal according to the second dimension
    '''
    if shuffle:
        return signals[:, torch.randperm(signals.size(1))]
    if shifts is None:
        shifts = np.random.randint(15, signals.shape[1])
    return torch.roll(signals, shifts=shifts, dims=1)

def get_null(signals, n_shifts, dt=0, shuffle=False):
    '''
    Calculates null where each row's (cell) STTC is calculated against a circular-shifted array of every other cell. 
    This is done for (ns = 1 to n_shifts + 1) values

    If shuffle=True, instead of circular shifting, the signals are instead randomly shuffled 'n_shift' times.

    The output is shaped as (ns, n_cell, n_cell).
    Each cell can be calculated as STTC(A, torch.roll(B, shifts=ns)) (if shuffle=False)

    Inputs
    - signals : (torch.Tensor). 2-d array of signals.
                Only put in signals in the format (n_cells, n_samples in time)
    - n_shifts : (int). Maximal value of shifts to be calculated.
    - dt : (int, default = 0)
    - ret_all : (boolean, default = False) if set to True, returns dictionary with the intermediate calculations of 
                                            TA, TB, PA and PB along with STTC

    Outputs
    - null : (torch.Tensor) retruns a (ns x n_cells x n_cells) matrix with each cell giving the 
                            paired STTC calculated at set dt with the j-index being shifted by ns.

    '''
    if not torch.is_tensor(signals):
        raise ValueError("Input must be a PyTorch tensor.")

    if signals.dtype!=torch.float:
        raise ValueError("Input must be a float tensor, dtype=torch.float, torch.float32 or torch.float64.")
        
    N,L = signals.shape

    null = []

    dt_signals = add_dt(signals, dt=dt)
    NB = torch.tile(signals.sum(axis=1), (N,1))
    TA = torch.tile(dt_signals.mean(axis=1), (N,1)).T

    rng = np.random.default_rng()
    shift_range = np.concatenate([rng.integers(-(L-40), -40, size=n_shifts//2), rng.integers(40, L-40, size=n_shifts//2)])
    for ns in shift_range:
        shifted = shift_signal(signals, shifts=ns, shuffle=shuffle)
        dt_shifted = add_dt(shifted, dt=dt)
        TB = torch.tile(dt_shifted.mean(axis=1), (N,1))

        DAB = signals @ dt_shifted.T
        DBA = dt_signals @ shifted.T

        PA = DAB/NB.T
        PB = DBA/NB

        null.append(STTC_formula(TA, TB, PA, PB))
        
    null = torch.stack(null, dim=0)
    return null


def get_null_nodt(signals, n_shifts=100, shuffle=False):
    if not torch.is_tensor(signals):
        raise ValueError("Input must be a PyTorch tensor.")

    if not signals.is_floating_point():
        raise ValueError("Input must be a float tensor, dtype=torch.float, torch.float32 or torch.float64.")
        
    N,L = signals.shape

    null = []

    
    NB = torch.tile(signals.sum(axis=1), (N,1))
    TA = torch.tile(signals.mean(axis=1), (N,1)).T

    rng = np.random.default_rng()
    shift_range = np.concatenate([rng.integers(-(L-40), -40, size=n_shifts//2), rng.integers(40, L-40, size=n_shifts//2)])

    for ns in shift_range:
        shifted = shift_signal(signals, shifts=ns, shuffle=shuffle)
        DAB = signals @ shifted.T
        TB = torch.tile(shifted.mean(axis=1), (N,1))
        
        PA = DAB/NB.T
        PB = DAB/NB
        del shifted, DAB
        null.append(STTC_formula(TA, TB, PA, PB))
        
    null = torch.stack(null, dim=0)
    return null

# def get_null_nodt_batched(signals, n_shifts = 100, shuffle=False, batch_split = 5):
#     if not torch.is_tensor(signals):
#         raise ValueError("Input must be a PyTorch tensor.")

#     if signals.dtype!=torch.float:
#         raise ValueError("Input must be a float tensor, dtype=torch.float, torch.float32 or torch.float64.")

#     batches = np.arange(0, n_shifts,batch_split)

#     null_batches = []

#     for bch in batches:
#         print(bch, end = ' ')
#         null_batches.append(get_null_nodt(signals, list(range(bch, bch+batch_split))))

#     return torch.cat(null_batches)

def STTC_AB_nodt(signals_A, signals_B, ret_all = False):
    if not torch.is_tensor(signals_A) or not torch.is_tensor(signals_B):
        raise ValueError("Input must be a PyTorch tensor.")

    if not signals_A.is_floating_point() or not signals_B.is_floating_point():
        raise ValueError("Input must be a float tensor, dtype=torch.float, torch.float32 or torch.float64.")
        
    nA, L = signals_A.shape
    nB = signals_B.shape[0]
    
    if L != signals_B.shape[1]:
        raise ValueError("Input signals must have same timesteps.")

    NA = torch.tile(signals_A.sum(axis=1), (nB, 1)).T
    NB = torch.tile(signals_B.sum(axis=1), (nA, 1)).T
    TA = torch.tile(signals_A.mean(axis=1), (nB, 1)).T
    TB = torch.tile(signals_B.mean(axis=1), (nA, 1))
    
    DAB = signals_A @ signals_B.T

    PA = DAB/NA
    PB = DAB/NB.T
    
    STTC = STTC_formula(TA, TB, PA, PB)
    
    if ret_all:
        return dict(zip(['STTC', 'TA', 'TB', 'PA', 'PB'], [STTC, TA, TB, PA, PB]))
    
    return STTC

    
def get_null_AB_nodt(signals_A, signals_B, n_shifts = 100, shuffle=False):
    
    if not torch.is_tensor(signals_A) or not torch.is_tensor(signals_B):
        raise ValueError("Input must be a PyTorch tensor.")

    if not signals_A.is_floating_point() or not signals_B.is_floating_point():
        raise ValueError("Input must be a float tensor, dtype=torch.float, torch.float32 or torch.float64.")
        
        
    nA, L = signals_A.shape
    nB = signals_B.shape[0]
    
    if L != signals_B.shape[1]:
        raise ValueError("Input signals must have same timesteps.")
    
   
    AB_mean = torch.zeros(nA, nB).to(signals_A.device).to(signals_A.dtype)
    AB_M2 = torch.zeros(nA, nB).to(signals_A.device).to(signals_A.dtype)

    count = 0

    rng = np.random.default_rng()
    shift_range = np.concatenate([rng.integers(-(L-40), -40, size=n_shifts//2), rng.integers(40, L-40, size=n_shifts//2)])
    for nx, ns in enumerate(shift_range):
        
        shifted_B = shift_signal(signals_B, shifts=ns, shuffle=shuffle)
        count += 1
        
        x = STTC_AB_nodt(signals_A, shifted_B)
        delta = x - AB_mean
        AB_mean += delta / count
        AB_M2 += delta * (x - AB_mean)

    rand_shift = rng.choice([rng.integers(-(L-40), -40), rng.integers(40, L-40)])
    shifted_B = shift_signal(signals_B, shifts=rand_shift, shuffle=shuffle)

    AB_std = torch.sqrt(AB_M2 / (count - 1))
    rand_null = STTC_AB_nodt(signals_A, shifted_B)

    return rand_null, AB_mean, AB_std


# def get_null_AB_nodt_TRIAL(signals_A, signals_B, n_shifts = 100, shuffle=False, opp=True):
    
#     if not torch.is_tensor(signals_A) or not torch.is_tensor(signals_B):
#         raise ValueError("Input must be a PyTorch tensor.")

#     if signals_A.dtype!=torch.float or signals_B.dtype!=torch.float:
#         raise ValueError("Input must be a float tensor, dtype=torch.float, torch.float32 or torch.float64.")
        
        
#     nA, L = signals_A.shape
#     nB = signals_B.shape[0]
    
#     if L != signals_B.shape[1]:
#         raise ValueError("Input signals must have same timesteps.")
    
   
#     AB_mean = torch.zeros(nA, nB).to(signals_A.device).to(signals_A.dtype)
#     AB_M2 = torch.zeros(nA, nB).to(signals_A.device).to(signals_A.dtype)

#     if opp:
#         opp_mean = torch.zeros(nA, nB).to(signals_A.device).to(signals_A.dtype)
#         opp_M2 = torch.zeros(nA, nB).to(signals_A.device).to(signals_A.dtype)
#     count = 0

#     rng = np.random.default_rng()
#     shift_range = rng.integers(20, L - 20, size = n_shifts)
#     for nx, ns in enumerate(shift_range):

#         if nx>=n_shifts // 2:
#             shifted_B = shift_signal(signals_B, shifts=0, shuffle=shuffle)
#             shifted_A = shift_signal(signals_A, shifts=ns, shuffle=shuffle)
#         else:
#             shifted_B = shift_signal(signals_B, shifts=ns, shuffle=shuffle)
#             shifted_A = shift_signal(signals_A, shifts=0, shuffle=shuffle)
#         count += 1
        
#         x = STTC_AB_nodt(shifted_A, shifted_B)
#         delta = x - AB_mean
#         AB_mean += delta / count
#         AB_M2 += delta * (x - AB_mean)

#         if opp:
#             x_opp = STTC_AB_nodt(shifted_B, signals_A)
#             delta_opp = x_opp - opp_mean
#             opp_mean += delta_opp / count
#             opp_M2 += delta_opp * (x_opp - opp_mean)
    
#     # AB_std = torch.sqrt(AB_M2 / (count - 1))
#     # opp_std = torch.sqrt(opp_M2 / (count - 1))

#     rand_shift = rng.integers(20, L-20)
#     shifted_B = shift_signal(signals_B, shifts=rand_shift, shuffle=shuffle)
#     shifted_A = shift_signal(signals_A, shifts=rand_shift, shuffle=shuffle)

#     AB_std = torch.sqrt(AB_M2 / (count - 1))
#     rand_null = STTC_AB_nodt(signals_A, shifted_B)

#     if opp:
#         opp_std = torch.sqrt(opp_M2 / (count - 1))
#         rand_null_opp = STTC_AB_nodt(shifted_A, signals_B)
    
#         return rand_null, rand_null_opp, AB_mean, opp_mean, AB_std, opp_std

#     return rand_null, AB_mean, AB_std
    
    