import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set the style
plt.style.use('seaborn-v0_8-darkgrid')

# Create a figure with a grid layout
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 2, figure=fig)

# Set random seed for reproducibility
np.random.seed(42)

# 1. Generate a continuous-time signal
# Let's create a signal with two frequency components
def generate_continuous_signal(t):
    # Main frequency component at 5 Hz
    signal_1 = np.sin(2 * np.pi * 5 * t)
    # Second frequency component at 20 Hz
    signal_2 = 0.5 * np.sin(2 * np.pi * 20 * t)
    return signal_1 + signal_2

# Create a high-resolution time axis for "continuous" signal
t_continuous = np.linspace(0, 1, 10000)
continuous_signal = generate_continuous_signal(t_continuous)

# Plot the continuous signal
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t_continuous, continuous_signal, 'b-', label='Continuous Signal')
ax1.set_title('Original Continuous Signal (5 Hz + 20 Hz components)')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ax1.grid(True)
ax1.legend()

# 2. Sample the signal at different rates
# According to Nyquist, we need a sampling rate of at least 2*20 Hz = 40 Hz

# Above Nyquist: 100 Hz sampling rate (adequate sampling)
fs_adequate = 100  # Hz
t_adequate = np.arange(0, 1, 1/fs_adequate)
sampled_adequate = generate_continuous_signal(t_adequate)

# Just above Nyquist: 50 Hz (minimum proper sampling rate)
fs_minimum = 50  # Hz
t_minimum = np.arange(0, 1, 1/fs_minimum)
sampled_minimum = generate_continuous_signal(t_minimum)

# Below Nyquist: 30 Hz (aliasing will occur)
fs_aliasing = 30  # Hz
t_aliasing = np.arange(0, 1, 1/fs_aliasing)
sampled_aliasing = generate_continuous_signal(t_aliasing)

# Very low sampling rate: 8 Hz (severe aliasing)
fs_severe = 8  # Hz
t_severe = np.arange(0, 1, 1/fs_severe)
sampled_severe = generate_continuous_signal(t_severe)

# Plot the different sampling rates
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(t_continuous, continuous_signal, 'b-', alpha=0.3, label='Original')
ax2.stem(t_adequate, sampled_adequate, 'g-', markerfmt='go', basefmt=" ",
         label=f'Sampling at {fs_adequate} Hz (>Nyquist)')
ax2.set_title('Adequate Sampling (Above Nyquist Rate)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Amplitude')
ax2.grid(True)
ax2.legend()
ax2.set_xlim(0, 0.5)  # Focus on the first half to see details better

ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(t_continuous, continuous_signal, 'b-', alpha=0.3, label='Original')
ax3.stem(t_aliasing, sampled_aliasing, 'r-', markerfmt='ro', basefmt=" ",
         label=f'Sampling at {fs_aliasing} Hz (<Nyquist)')
ax3.set_title('Aliasing Due to Undersampling')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Amplitude')
ax3.grid(True)
ax3.legend()
ax3.set_xlim(0, 0.5)  # Focus on the first half to see details better

# 3. Analyze in the frequency domain using Fourier Transform
def plot_spectrum(ax, signal_data, fs, title):
    n = len(signal_data)
    # Compute FFT
    yf = np.fft.rfft(signal_data)
    # Compute frequency points
    xf = np.fft.rfftfreq(n, 1/fs)
    # Plot single-sided spectrum
    ax.plot(xf, 2.0/n * np.abs(yf))
    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.grid(True)
    # Add vertical lines for the original frequencies
    ax.axvline(x=5, color='g', linestyle='--', alpha=0.7, label='5 Hz Component')
    ax.axvline(x=20, color='g', linestyle='--', alpha=0.7, label='20 Hz Component')
    # Show Nyquist frequency
    ax.axvline(x=fs/2, color='r', linestyle=':', alpha=0.7, label='Nyquist Frequency')
    ax.legend()
    ax.set_xlim(0, max(50, fs/2 + 5))  # Show up to Nyquist frequency + padding

# Plot frequency domain analysis
ax4 = fig.add_subplot(gs[2, 0])
plot_spectrum(ax4, sampled_adequate, fs_adequate, f'Spectrum with {fs_adequate} Hz Sampling')

ax5 = fig.add_subplot(gs[2, 1])
plot_spectrum(ax5, sampled_severe, fs_severe, f'Spectrum with {fs_severe} Hz Sampling (Aliasing)')

plt.tight_layout()
plt.show()

# Additional demonstration: Signal reconstruction with different sampling rates
plt.figure(figsize=(12, 12))

# Original signal
plt.subplot(3, 1, 1)
plt.plot(t_continuous, continuous_signal, 'b-', label='Original Signal')
plt.title('Original Continuous Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# Proper reconstruction with adequate sampling
plt.subplot(3, 1, 2)
plt.plot(t_continuous, continuous_signal, 'b-', alpha=0.5, label='Original Signal')
plt.stem(t_adequate, sampled_adequate, 'g-', markerfmt='go', basefmt=" ",
         label=f'Samples at {fs_adequate} Hz')

# Reconstruct signal using sinc interpolation (ideal lowpass filtering)
t_recon = t_continuous
reconstructed_signal = np.zeros_like(t_recon)
for i, t_i in enumerate(t_adequate):
    reconstructed_signal += sampled_adequate[i] * np.sinc(fs_adequate * (t_recon - t_i))

plt.plot(t_recon, reconstructed_signal, 'r-', label='Reconstructed Signal')
plt.title(f'Reconstruction with {fs_adequate} Hz Sampling (Above Nyquist)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.xlim(0, 0.3)  # Focus on a portion to see details better

# Failed reconstruction with severe undersampling
plt.subplot(3, 1, 3)
plt.plot(t_continuous, continuous_signal, 'b-', alpha=0.5, label='Original Signal')
plt.stem(t_severe, sampled_severe, 'r-', markerfmt='ro', basefmt=" ",
         label=f'Samples at {fs_severe} Hz')

# Attempt to reconstruct (will demonstrate aliasing)
reconstructed_aliased = np.zeros_like(t_recon)
for i, t_i in enumerate(t_severe):
    reconstructed_aliased += sampled_severe[i] * np.sinc(fs_severe * (t_recon - t_i))

plt.plot(t_recon, reconstructed_aliased, 'r-', label='Incorrectly Reconstructed Signal')
plt.title(f'Failed Reconstruction with {fs_severe} Hz Sampling (Below Nyquist)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.xlim(0, 0.3)

plt.tight_layout()
plt.show()
