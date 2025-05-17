# Sampling and Aliasing Demonstration

This interactive web application demonstrates the concepts of signal sampling and aliasing in digital signal processing. The app allows users to explore how different sampling rates affect signal reconstruction and observe the effects of aliasing when sampling below the Nyquist rate.

## Features

- Interactive signal parameter controls (frequency and amplitude)
- Adjustable sampling rates to visualize both proper sampling and aliasing effects
- Time domain visualization of original and sampled signals
- Frequency domain analysis with spectrum visualization
- Signal reconstruction demonstration with error metrics
- Educational explanations of key concepts

## Live Demo

The application is deployed and accessible at:
<https://sampling-aliasing-hugo-approximation-project.streamlit.app/>

## How to Run Locally

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run sampling-aliasing-streamlit.py
   ```

## Requirements

The application requires Python 3.6+ and the packages listed in requirements.txt, primarily:

- streamlit
- numpy
- plotly

## How to Use

1. Adjust the signal parameters in the sidebar to define your signal composition
2. Set different sampling rates to see how they affect signal reconstruction
3. Explore each tab to understand different aspects of sampling theory:
   - Time Domain: Visualize the original signal and how it's sampled
   - Frequency Domain: See how sampling affects the frequency spectrum
   - Signal Reconstruction: Observe how well the signal can be reconstructed from samples.
