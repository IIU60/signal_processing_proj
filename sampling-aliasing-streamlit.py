import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Sampling & Aliasing Demo",
    page_icon="📊",
    layout="wide"
)

st.title("Interactive Sampling and Aliasing Demonstration")
st.markdown("""
This application demonstrates the effects of sampling rates on signal reconstruction 
and the phenomenon of aliasing. Adjust the parameters to see how they impact signal sampling and reconstruction.
""")

# Sidebar for all user inputs
st.sidebar.header("Signal Parameters")

# Primary signal frequency
primary_freq = st.sidebar.slider(
    "Primary Signal Frequency (Hz)",
    min_value=1,
    max_value=20,
    value=5,
    step=1,
    help="Frequency of the main signal component"
)

# Secondary signal frequency
secondary_freq = st.sidebar.slider(
    "Secondary Signal Frequency (Hz)",
    min_value=1,
    max_value=50,
    value=20,
    step=1,
    help="Frequency of the secondary signal component"
)

# Secondary signal amplitude
secondary_amp = st.sidebar.slider(
    "Secondary Signal Amplitude",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1,
    help="Amplitude of the secondary signal component relative to the primary (1.0)"
)

st.sidebar.header("Sampling Parameters")

# Adequate sampling rate
fs_adequate_default = 2 * (primary_freq + secondary_freq) + 10
fs_adequate = st.sidebar.slider(
    "Adequate Sampling Rate (Hz)",
    min_value=max(primary_freq, secondary_freq) * 2 + 1,
    max_value=200,
    value=min(fs_adequate_default, 200),
    step=1,
    help="Sampling rate well above the Nyquist rate"
)

# Aliasing sampling rate
nyquist_rate = 2 * max(primary_freq, secondary_freq)
fs_aliasing = st.sidebar.slider(
    "Aliasing Sampling Rate (Hz)",
    min_value=1,
    max_value=int(nyquist_rate) - 1 if int(nyquist_rate) > 2 else 1,
    value=min(int(nyquist_rate) - 5, max(1, int(nyquist_rate) // 2)),
    step=1,
    help="Sampling rate below the Nyquist rate (will cause aliasing)"
)

# Function to generate the continuous signal
def generate_continuous_signal(t, f1, f2, amp2):
    # Main frequency component
    signal_1 = np.sin(2 * np.pi * f1 * t)
    # Second frequency component
    signal_2 = amp2 * np.sin(2 * np.pi * f2 * t)
    return signal_1 + signal_2

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Time Domain", "Frequency Domain", "Signal Reconstruction"])

# Create time axis for "continuous" signal
t_continuous = np.linspace(0, 1, 10000)
continuous_signal = generate_continuous_signal(t_continuous, primary_freq, secondary_freq, secondary_amp)

# Sample at adequate rate
t_adequate = np.arange(0, 1, 1/fs_adequate)
sampled_adequate = generate_continuous_signal(t_adequate, primary_freq, secondary_freq, secondary_amp)

# Sample at aliasing rate
t_aliasing = np.arange(0, 1, 1/fs_aliasing)
sampled_aliasing = generate_continuous_signal(t_aliasing, primary_freq, secondary_freq, secondary_amp)

with tab1:
    st.header("Time Domain Analysis")
    
    # Create time domain figure
    fig_time = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Original Continuous Signal",
            f"Adequate Sampling ({fs_adequate} Hz, above Nyquist)",
            f"Aliasing Sampling ({fs_aliasing} Hz, below Nyquist)"
        ),
        vertical_spacing=0.1
    )
    
    # Plot original continuous signal
    fig_time.add_trace(
        go.Scatter(x=t_continuous, y=continuous_signal, mode='lines', name='Original Signal'),
        row=1, col=1
    )
    
    # Plot adequate sampling
    fig_time.add_trace(
        go.Scatter(x=t_continuous, y=continuous_signal, mode='lines', 
                   line=dict(color='blue', width=1, dash='dot'), showlegend=False),
        row=2, col=1
    )
    fig_time.add_trace(
        go.Scatter(x=t_adequate, y=sampled_adequate, mode='markers', 
                   marker=dict(color='green', size=10), name='Adequate Samples'),
        row=2, col=1
    )
    
    # Plot aliasing sampling
    fig_time.add_trace(
        go.Scatter(x=t_continuous, y=continuous_signal, mode='lines', 
                   line=dict(color='blue', width=1, dash='dot'), showlegend=False),
        row=3, col=1
    )
    fig_time.add_trace(
        go.Scatter(x=t_aliasing, y=sampled_aliasing, mode='markers', 
                   marker=dict(color='red', size=10), name='Aliasing Samples'),
        row=3, col=1
    )
    
    # Update layout for better visualization
    fig_time.update_layout(
        height=800,
        xaxis_title="Time (s)",
        xaxis2_title="Time (s)",
        xaxis3_title="Time (s)",
        yaxis_title="Amplitude",
        yaxis2_title="Amplitude",
        yaxis3_title="Amplitude",
        legend=dict(orientation="h", y=1.1),
    )
    
    # Set x-axis range for better visualization
    fig_time.update_xaxes(range=[0, 0.5], row=1, col=1)
    fig_time.update_xaxes(range=[0, 0.5], row=2, col=1)
    fig_time.update_xaxes(range=[0, 0.5], row=3, col=1)
    
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Display Nyquist rate information
    st.info(f"""
        **Nyquist Information:**
        - Primary Signal: {primary_freq} Hz
        - Secondary Signal: {secondary_freq} Hz
        - Highest Frequency: {max(primary_freq, secondary_freq)} Hz
        - Nyquist Rate: {nyquist_rate} Hz (2 × highest frequency)
    """)

with tab2:
    st.header("Frequency Domain Analysis")
    
    # Function to calculate spectrum
    def calculate_spectrum(signal_data, fs):
        n = len(signal_data)
        yf = np.fft.rfft(signal_data) 
        xf = np.fft.rfftfreq(n, 1/fs)
        return xf, 2.0/n * np.abs(yf)
    
    # Calculate spectrum for adequate sampling
    xf_adequate, yf_adequate = calculate_spectrum(sampled_adequate, fs_adequate)
    
    # Calculate spectrum for aliasing sampling
    xf_aliasing, yf_aliasing = calculate_spectrum(sampled_aliasing, fs_aliasing)
    
    # Create frequency domain figure
    fig_freq = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f"Frequency Spectrum with {fs_adequate} Hz Sampling (Above Nyquist)",
            f"Frequency Spectrum with {fs_aliasing} Hz Sampling (Below Nyquist)"
        ),
        vertical_spacing=0.2
    )
    
    # Plot adequate sampling spectrum
    fig_freq.add_trace(
        go.Scatter(x=xf_adequate, y=yf_adequate, mode='lines', name=f'Spectrum at {fs_adequate} Hz'),
        row=1, col=1
    )
    
    # Add vertical lines for reference frequencies (adequate)
    fig_freq.add_trace(
        go.Scatter(x=[primary_freq, primary_freq], y=[0, max(yf_adequate)*1.1], 
                   mode='lines', line=dict(color='green', width=2, dash='dash'),
                   name='Primary Frequency'),
        row=1, col=1
    )
    fig_freq.add_trace(
        go.Scatter(x=[secondary_freq, secondary_freq], y=[0, max(yf_adequate)*1.1], 
                   mode='lines', line=dict(color='orange', width=2, dash='dash'),
                   name='Secondary Frequency'),
        row=1, col=1
    )
    fig_freq.add_trace(
        go.Scatter(x=[fs_adequate/2, fs_adequate/2], y=[0, max(yf_adequate)*1.1], 
                   mode='lines', line=dict(color='red', width=2, dash='dash'),
                   name='Nyquist Frequency'),
        row=1, col=1
    )
    
    # Plot aliasing sampling spectrum
    fig_freq.add_trace(
        go.Scatter(x=xf_aliasing, y=yf_aliasing, mode='lines', name=f'Spectrum at {fs_aliasing} Hz'),
        row=2, col=1
    )
    
    # Add vertical lines for reference frequencies (aliasing)
    fig_freq.add_trace(
        go.Scatter(x=[primary_freq, primary_freq], y=[0, max(yf_aliasing)*1.1], 
                   mode='lines', line=dict(color='green', width=2, dash='dash'), showlegend=False),
        row=2, col=1
    )
    fig_freq.add_trace(
        go.Scatter(x=[secondary_freq, secondary_freq], y=[0, max(yf_aliasing)*1.1], 
                   mode='lines', line=dict(color='orange', width=2, dash='dash'), showlegend=False),
        row=2, col=1
    )
    # Only add the secondary frequency line if it's below the Nyquist frequency for aliasing
    if secondary_freq < fs_aliasing/2:
        fig_freq.add_trace(
            go.Scatter(x=[secondary_freq, secondary_freq], y=[0, max(yf_aliasing)*1.1], 
                      mode='lines', line=dict(color='orange', width=2, dash='dash'), showlegend=False),
            row=2, col=1
        )
    fig_freq.add_trace(
        go.Scatter(x=[fs_aliasing/2, fs_aliasing/2], y=[0, max(yf_aliasing)*1.1], 
                   mode='lines', line=dict(color='red', width=2, dash='dash'), showlegend=False),
        row=2, col=1
    )
    
    # Highlight aliased frequencies if applicable
    if secondary_freq > fs_aliasing/2:
        # Calculate the aliased frequency
        aliased_freq = fs_aliasing - (secondary_freq % fs_aliasing)
        if aliased_freq > fs_aliasing/2:
            aliased_freq = fs_aliasing - aliased_freq
            
        fig_freq.add_trace(
            go.Scatter(x=[aliased_freq, aliased_freq], y=[0, max(yf_aliasing)*1.1], 
                      mode='lines', line=dict(color='purple', width=2, dash='dash'),
                      name=f'Aliased Frequency ({aliased_freq:.1f} Hz)'),
            row=2, col=1
        )
    
    # Update layout for better visualization
    fig_freq.update_layout(
        height=700,
        xaxis_title="Frequency (Hz)",
        xaxis2_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        yaxis2_title="Magnitude",
        legend=dict(orientation="h", y=1.1)
    )
    
    # Set x-axis range for better visualization
    x_max_adequate = min(fs_adequate/2 + 5, max(100, secondary_freq*1.5))
    x_max_aliasing = min(fs_aliasing/2 + 5, max(50, secondary_freq*0.75))
    
    fig_freq.update_xaxes(range=[0, x_max_adequate], row=1, col=1)
    fig_freq.update_xaxes(range=[0, x_max_aliasing], row=2, col=1)
    
    st.plotly_chart(fig_freq, use_container_width=True)
    
    # Display aliasing explanation
    if secondary_freq > fs_aliasing/2:
        aliased_freq = fs_aliasing - (secondary_freq % fs_aliasing)
        if aliased_freq > fs_aliasing/2:
            aliased_freq = fs_aliasing - aliased_freq
            
        st.warning(f"""
            **Aliasing Detected!**
            
            The secondary signal at {secondary_freq} Hz is above the Nyquist frequency ({fs_aliasing/2} Hz) 
            for your chosen sampling rate of {fs_aliasing} Hz.
            
            This causes the frequency to "fold back" and appear as a lower frequency at approximately 
            {aliased_freq:.1f} Hz in the sampled signal.
        """)
    else:
        st.success(f"""
            All signal frequencies are below the Nyquist frequency ({fs_aliasing/2} Hz) for your chosen 
            aliasing sampling rate of {fs_aliasing} Hz.
            
            Try increasing the secondary frequency to see aliasing effects!
        """)

with tab3:
    st.header("Signal Reconstruction")
    
    # Define function for sinc interpolation
    def sinc_interp(x, s, u):
        """
        Interpolate signal s at positions u using sinc interpolation
        x: original sample positions
        s: original sample values
        u: new positions for interpolation
        """
        if len(x) != len(s):
            raise ValueError('x and s must have the same length')
        
        # Find the spacing of the original samples
        T = (x[-1] - x[0]) / (len(x) - 1)
        
        # Calculate sinc interpolation
        sincM = np.tile(u, (len(x), 1)) - np.tile(x[:, np.newaxis], (1, len(u)))
        sincM = np.sinc(sincM / T)
        return np.dot(s, sincM)
    
    # Create reconstruction time axis (less dense for performance)
    t_recon = np.linspace(0, 1, 1000)
    
    # Reconstruct signals using sinc interpolation
    try:
        # Sometimes sinc_interp can be unstable, so we'll catch any errors
        reconstructed_adequate = sinc_interp(t_adequate, sampled_adequate, t_recon)
        reconstructed_aliasing = sinc_interp(t_aliasing, sampled_aliasing, t_recon)
    except Exception as e:
        st.error(f"Error in sinc interpolation: {e}")
        # Fall back to a simpler but less accurate interpolation
        from scipy import interpolate
        f_adequate = interpolate.interp1d(t_adequate, sampled_adequate, kind='cubic', bounds_error=False, fill_value="extrapolate")
        f_aliasing = interpolate.interp1d(t_aliasing, sampled_aliasing, kind='cubic', bounds_error=False, fill_value="extrapolate")
        reconstructed_adequate = f_adequate(t_recon)
        reconstructed_aliasing = f_aliasing(t_recon)
    
    # Sample the continuous signal at the reconstruction points for comparison
    continuous_at_recon = generate_continuous_signal(t_recon, primary_freq, secondary_freq, secondary_amp)
    
    # Create reconstruction figure
    fig_recon = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f"Reconstruction with {fs_adequate} Hz Sampling (Above Nyquist)",
            f"Reconstruction with {fs_aliasing} Hz Sampling (Below Nyquist)"
        ),
        vertical_spacing=0.2
    )
    
    # Plot adequate sampling reconstruction
    fig_recon.add_trace(
        go.Scatter(x=t_recon, y=continuous_at_recon, mode='lines', 
                   line=dict(color='blue', width=2), name='Original Signal'),
        row=1, col=1
    )
    fig_recon.add_trace(
        go.Scatter(x=t_adequate, y=sampled_adequate, mode='markers', 
                   marker=dict(color='green', size=8), name='Samples'),
        row=1, col=1
    )
    fig_recon.add_trace(
        go.Scatter(x=t_recon, y=reconstructed_adequate, mode='lines', 
                   line=dict(color='red', width=2), name='Reconstructed'),
        row=1, col=1
    )
    
    # Plot aliasing sampling reconstruction
    fig_recon.add_trace(
        go.Scatter(x=t_recon, y=continuous_at_recon, mode='lines', 
                   line=dict(color='blue', width=2), showlegend=False),
        row=2, col=1
    )
    fig_recon.add_trace(
        go.Scatter(x=t_aliasing, y=sampled_aliasing, mode='markers', 
                   marker=dict(color='green', size=8), showlegend=False),
        row=2, col=1
    )
    fig_recon.add_trace(
        go.Scatter(x=t_recon, y=reconstructed_aliasing, mode='lines', 
                   line=dict(color='red', width=2), showlegend=False),
        row=2, col=1
    )
    
    # Update layout for better visualization
    fig_recon.update_layout(
        height=700,
        xaxis_title="Time (s)",
        xaxis2_title="Time (s)",
        yaxis_title="Amplitude",
        yaxis2_title="Amplitude",
        legend=dict(orientation="h", y=1.1)
    )
    
    # Set x-axis range for better visualization
    fig_recon.update_xaxes(range=[0, 0.3], row=1, col=1)
    fig_recon.update_xaxes(range=[0, 0.3], row=2, col=1)
    
    st.plotly_chart(fig_recon, use_container_width=True)
    
    # Calculate reconstruction error
    error_adequate = np.mean((reconstructed_adequate - continuous_at_recon)**2)
    error_aliasing = np.mean((reconstructed_aliasing - continuous_at_recon)**2)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Square Error (Adequate)", f"{error_adequate:.6f}")
    with col2:
        st.metric("Mean Square Error (Aliasing)", f"{error_aliasing:.6f}")
    
    # Explanation
    st.markdown("""
    ### Signal Reconstruction Explanation
    
    The **Nyquist-Shannon sampling theorem** states that a continuous bandlimited signal can be perfectly 
    reconstructed from its samples if the sampling rate is greater than twice the highest frequency in the signal.
    
    - In the **top graph**, we sample at a rate well above the Nyquist rate, allowing for accurate reconstruction.
    - In the **bottom graph**, we sample below the Nyquist rate, causing aliasing and preventing accurate reconstruction.
    
    The reconstruction is performed using **sinc interpolation**, which is the theoretically perfect 
    reconstruction method for bandlimited signals sampled above the Nyquist rate.
    """)

# Add educational resources at the bottom
st.markdown("---")
st.header("Educational Resources")
st.markdown("""
### Key Concepts
- **Sampling**: The process of converting a continuous signal into a discrete sequence of values.
- **Nyquist Rate**: To accurately represent a signal, you must sample at least twice as fast as the highest frequency component in the signal.
- **Aliasing**: When sampling below the Nyquist rate, high-frequency components appear as lower frequencies in the sampled signal.
- **Signal Reconstruction**: The process of recreating a continuous signal from its samples.

### Further Reading
- [Nyquist-Shannon Sampling Theorem](https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem)
- [Aliasing in Signal Processing](https://en.wikipedia.org/wiki/Aliasing)
- [Digital Signal Processing Basics](https://www.analog.com/en/design-center/landing-pages/001/beginners-guide-to-dsp.html)
""")
