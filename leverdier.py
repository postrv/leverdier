import numpy as np
import scipy.signal as signal
from scipy.optimize import minimize, approx_fprime
import matplotlib.pyplot as plt
from scipy.interpolate import LSQUnivariateSpline, BSpline

# -------------------------------------
# Mock Input Signal
# -------------------------------------
# Create a test signal composed of two slowly varying "harmonic-like" modes:
# x(t) = C1(t)*cos(Θ1(t)) + C2(t)*cos(Θ2(t)) + noise

Fs = 1000  # Sampling frequency in Hz
T = 1.0  # Duration in seconds
t = np.linspace(0, T, int(Fs * T), endpoint=False)

# Construct Mode 1: Slowly decreasing frequency from 50 Hz to 45 Hz
f1 = 50 - 5 * t  # Frequency drift
phase1 = 2 * np.pi * np.cumsum(f1) / Fs
C1 = 1.0 + 0.2 * np.sin(2 * np.pi * 2 * t)  # Amplitude modulation at 2 Hz
x1 = C1 * np.cos(phase1)

# Construct Mode 2: Amplitude-modulated mode at around 100 Hz
f2 = 100 + 10 * np.sin(2 * np.pi * 0.5 * t)  # Frequency modulation at 0.5 Hz
phase2 = 2 * np.pi * np.cumsum(f2) / Fs
C2 = 0.5 + 0.1 * np.cos(2 * np.pi * t)  # Amplitude modulation at 1 Hz
x2 = C2 * np.cos(phase2)

# Combine modes and add white Gaussian noise
x = x1 + x2 + 0.1 * np.random.randn(len(t))

# -------------------------------------
# Initial Guess Using STFT
# -------------------------------------
# Identify two dominant frequency ridges in the STFT.
f_stft, t_stft, Zxx = signal.stft(x, fs=Fs, nperseg=256, noverlap=128)
magnitude = np.abs(Zxx)

# Identify the two highest energy frequency components per time frame
freq_indices = np.argsort(magnitude, axis=0)[-2:]  # Indices of two highest peaks
freq_tracks = f_stft[freq_indices]

# Handle cases where there might be fewer time bins than expected
if freq_tracks.shape[1] == 0:
    raise ValueError("No frequency tracks found. Check the STFT parameters.")

# Extract frequency estimates for both modes
mode1_freq_est = freq_tracks[0, :]  # Shape: (n_time_bins,)
mode2_freq_est = freq_tracks[1, :]  # Shape: (n_time_bins,)

# Interpolate frequency estimates to match the original signal's time vector
mode1_freq_full = np.interp(t, t_stft, mode1_freq_est)
mode2_freq_full = np.interp(t, t_stft, mode2_freq_est)

# Convert to instantaneous phase by integrating frequency
phase_mode1_init = 2 * np.pi * np.cumsum(mode1_freq_full) / Fs
phase_mode2_init = 2 * np.pi * np.cumsum(mode2_freq_full) / Fs

# Initialize amplitudes
C1_init = np.real(x * np.exp(-1j * phase_mode1_init))
C2_init = np.real(x * np.exp(-1j * phase_mode2_init))

# Smooth initial estimates
C1_init = signal.savgol_filter(C1_init, window_length=51, polyorder=3)
C2_init = signal.savgol_filter(C2_init, window_length=51, polyorder=3)

# -------------------------------------
# Spline Setup with Proper Knot Placement
# -------------------------------------
spline_degree = 3
num_internal_knots = 20


def generate_internal_knots(x, num_knots, degree):
    """Generate internal knots that satisfy Schoenberg-Whitney conditions."""
    # Need at least (degree + 1) points between consecutive knots
    min_points = degree + 1
    n = len(x)

    # Calculate number of intervals that can accommodate minimum points
    num_intervals = min(num_knots + 1, n // min_points)

    if num_intervals < 1:
        raise ValueError("Not enough data points to place internal knots.")

    # Generate indices for internal knots
    indices = np.linspace(min_points, n - min_points, num_intervals + 1)[1:-1]
    indices = indices.astype(int)

    return x[indices]


# Generate internal knots
internal_knots = generate_internal_knots(t, num_internal_knots, spline_degree)


# Define full knot vector with clamped ends
def create_full_knots(internal_knots, degree, t_start, t_end):
    """Create full knot vector with clamped ends."""
    # For clamped splines, repeat the first and last internal knots degree times
    start_knots = np.repeat(t_start, degree)
    end_knots = np.repeat(t_end, degree)
    full_knots = np.concatenate((start_knots, internal_knots, end_knots))
    return full_knots


full_knots = create_full_knots(internal_knots, spline_degree, t[0], t[-1])


# Create initial splines and get coefficients
def create_initial_coeffs(data, knots, degree):
    """Create initial spline coefficients using LSQUnivariateSpline."""
    spline = LSQUnivariateSpline(t, data, knots, k=degree)
    return spline.get_coeffs()


C1_coeffs_init = create_initial_coeffs(C1_init, internal_knots, spline_degree)
C2_coeffs_init = create_initial_coeffs(C2_init, internal_knots, spline_degree)
P1_coeffs_init = create_initial_coeffs(phase_mode1_init, internal_knots, spline_degree)
P2_coeffs_init = create_initial_coeffs(phase_mode2_init, internal_knots, spline_degree)


# -------------------------------------
# Define Parameter Packing and Unpacking
# -------------------------------------
# We store all spline coefficients in a single vector: [C1_coeffs, C2_coeffs, P1_coeffs, P2_coeffs]
def pack_params(C1_coeffs, C2_coeffs, P1_coeffs, P2_coeffs):
    """Pack all parameters into a single array."""
    return np.concatenate([C1_coeffs, C2_coeffs, P1_coeffs, P2_coeffs])


def unpack_params(params):
    """Unpack parameters into individual coefficient arrays."""
    n = len(params) // 4
    return params[:n], params[n:2 * n], params[2 * n:3 * n], params[3 * n:]


# Initialize parameters
params_init = pack_params(C1_coeffs_init, C2_coeffs_init, P1_coeffs_init, P2_coeffs_init)


# -------------------------------------
# Reconstruct Signal Function
# -------------------------------------
def reconstruct_signal(params, full_knots, k=3, t_eval=t):
    """Reconstruct the signal using the current parameter values."""
    C1_coeffs, C2_coeffs, P1_coeffs, P2_coeffs = unpack_params(params)

    # Create BSpline objects
    C1_spline = BSpline(full_knots, C1_coeffs, k, extrapolate=True)
    C2_spline = BSpline(full_knots, C2_coeffs, k, extrapolate=True)
    P1_spline = BSpline(full_knots, P1_coeffs, k, extrapolate=True)
    P2_spline = BSpline(full_knots, P2_coeffs, k, extrapolate=True)

    # Evaluate splines
    C1 = C1_spline(t_eval)
    C2 = C2_spline(t_eval)
    P1 = P1_spline(t_eval)
    P2 = P2_spline(t_eval)

    # Reconstruct signal
    recon = C1 * np.cos(P1) + C2 * np.cos(P2)
    return recon


# -------------------------------------
# Cost Function and Gradient
# -------------------------------------
alpha = 0.1  # Regularization parameter


def cost_function(params):
    """Compute the cost function value for the current parameters."""
    try:
        recon = reconstruct_signal(params, full_knots, k=spline_degree, t_eval=t)
        error = x - recon
        data_term = 0.5 * np.sum(error ** 2)

        # Regularization: smoothness of spline coefficients via second differences
        C1_coeffs, C2_coeffs, P1_coeffs, P2_coeffs = unpack_params(params)
        reg_C1 = np.sum(np.diff(C1_coeffs, n=2) ** 2)
        reg_C2 = np.sum(np.diff(C2_coeffs, n=2) ** 2)
        reg_P1 = np.sum(np.diff(P1_coeffs, n=2) ** 2)
        reg_P2 = np.sum(np.diff(P2_coeffs, n=2) ** 2)
        reg_term = alpha * (reg_C1 + reg_C2 + reg_P1 + reg_P2)

        return data_term + reg_term
    except Exception as e:
        print(f"Error in cost_function: {e}")
        raise


def cost_gradient(params):
    """Compute the gradient of the cost function using finite differences."""
    try:
        epsilon = 1e-5
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += epsilon
            f_plus = cost_function(params_plus)

            params_minus = params.copy()
            params_minus[i] -= epsilon
            f_minus = cost_function(params_minus)

            grad[i] = (f_plus - f_minus) / (2 * epsilon)
        return grad
    except Exception as e:
        print(f"Error in cost_gradient: {e}")
        raise


# -------------------------------------
# Gradient Check
# -------------------------------------
def gradient_check():
    # Select a small subset of parameters for gradient checking
    subset_size = 10  # Number of parameters to check
    np.random.seed(0)  # For reproducibility
    subset_indices = np.random.choice(len(params_init), size=subset_size, replace=False)
    subset_params = params_init[subset_indices]

    def cost_func_subset(subset):
        # Create a full parameter vector with the subset replaced
        full_params = np.copy(params_init)
        full_params[subset_indices] = subset
        return cost_function(full_params)

    def grad_subset(subset):
        # Compute gradient for the subset
        full_grad = cost_gradient(params_init)
        return full_grad[subset_indices]

    # Compute numerical gradient
    epsilon = 1e-8
    num_grad = approx_fprime(subset_params, cost_func_subset, epsilon)

    # Compute analytical gradient
    anal_grad = grad_subset(subset_params)

    # Compute relative error
    rel_error = np.linalg.norm(num_grad - anal_grad) / (np.linalg.norm(num_grad) + np.linalg.norm(anal_grad))
    print(f"Gradient check relative error (subset size {subset_size}): {rel_error:.2e}")

    if rel_error < 1e-5:
        print("Gradient check PASSED.")
    else:
        print("Gradient check FAILED. Please verify gradient computations.")


# Perform gradient check
print("Performing gradient check on a random subset of parameters...")
gradient_check()

# -------------------------------------
# Optimization
# -------------------------------------
print("Starting Leverdier optimization...")
print(f"Initial parameters shape: {params_init.shape}")
print(f"Number of internal knots: {len(internal_knots)}")
print(f"Number of full knots: {len(full_knots)}")
print(f"Spline degree: {spline_degree}")

# Define optimization options
optimizer_options = {
    'maxiter': 500,
    'disp': True,
    'ftol': 1e-8,
    'gtol': 1e-8
}

try:
    res = minimize(
        fun=cost_function,
        x0=params_init,
        jac=cost_gradient,
        method='L-BFGS-B',
        options=optimizer_options
    )

    if res.success:
        print("Optimization succeeded!")
    else:
        print("Optimization failed:", res.message)

    # -------------------------------------
    # Results Processing
    # -------------------------------------
    # Extract optimized parameters
    C1_coeffs_opt, C2_coeffs_opt, P1_coeffs_opt, P2_coeffs_opt = unpack_params(res.x)

    # Reconstruct signal and components
    reconstructed = reconstruct_signal(res.x, full_knots, k=spline_degree, t_eval=t)

    # Get individual components
    C1_final = BSpline(full_knots, C1_coeffs_opt, spline_degree, extrapolate=True)(t)
    C2_final = BSpline(full_knots, C2_coeffs_opt, spline_degree, extrapolate=True)(t)
    P1_final = BSpline(full_knots, P1_coeffs_opt, spline_degree, extrapolate=True)(t)
    P2_final = BSpline(full_knots, P2_coeffs_opt, spline_degree, extrapolate=True)(t)

    # Compute instantaneous frequencies
    freq1 = np.gradient(P1_final, t) / (2 * np.pi)
    freq2 = np.gradient(P2_final, t) / (2 * np.pi)

    # -------------------------------------
    # Visualization
    # -------------------------------------
    plt.figure(figsize=(15, 15))

    # Original vs Reconstructed Signal
    plt.subplot(5, 1, 1)
    plt.title("Original and Reconstructed Signals")
    plt.plot(t, x, 'b', label='Original', alpha=0.6)
    plt.plot(t, reconstructed, 'r', label='Reconstructed', alpha=0.8)
    plt.legend()

    # Mode 1 Components
    plt.subplot(5, 1, 2)
    plt.title("Mode 1: Amplitude and Phase")
    plt.plot(t, C1_final, label='Amplitude (C1)')
    plt.plot(t, P1_final, label='Phase (P1)')
    plt.legend()

    # Mode 2 Components
    plt.subplot(5, 1, 3)
    plt.title("Mode 2: Amplitude and Phase")
    plt.plot(t, C2_final, label='Amplitude (C2)')
    plt.plot(t, P2_final, label='Phase (P2)')
    plt.legend()

    # Reconstruction Error
    plt.subplot(5, 1, 4)
    plt.title("Reconstruction Error")
    plt.plot(t, x - reconstructed, label='Error', color='red')
    plt.legend()

    # Instantaneous Frequencies
    plt.subplot(5, 1, 5)
    plt.title("Instantaneous Frequencies")
    plt.plot(t, freq1, label='Mode 1 Frequency')
    plt.plot(t, freq2, label='Mode 2 Frequency')
    plt.plot(t, f1, '--', label='True f1', alpha=0.5)
    plt.plot(t, f2, '--', label='True f2', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print error metrics
    rmse = np.sqrt(np.mean((x - reconstructed) ** 2))
    print(f"Root Mean Square Error: {rmse:.6f}")

except Exception as e:
    print(f"Error during optimization: {e}")
    raise
