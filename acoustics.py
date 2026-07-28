import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Acoustic Signal Analysis", layout="wide")

st.title("Acoustic Signal Analysis & Harmonic Ratio Study")

# Audio mapping
SOUND_FILES = {
    'Temple Bell': 'Templebell.mp3',
    'Shankh (Conch)': 'shell.mp3',
    'Tanpura (Sa)': 'tanpura.mp3'
}

# Sidebar / Radio button selection
selected_name = st.radio("Select Instrument / Audio Source:", list(SOUND_FILES.keys()))
file_path = SOUND_FILES[selected_name]

# Play Audio player in browser
st.audio(file_path)

# Process Signal using Librosa
@st.cache_data
def analyze_signal(path):
    data, sample_rate = librosa.load(path, sr=None)
    fft_size = 2048
    chunk = data[:fft_size]
    fft_spectrum = np.abs(np.fft.rfft(chunk))
    freqs = np.fft.rfftfreq(fft_size, 1 / sample_rate)
    
    peak_idx = np.argmax(fft_spectrum)
    observed_f0 = freqs[peak_idx]
    return data, fft_spectrum, freqs, observed_f0

try:
    data, fft_spectrum, freqs, observed_f0 = analyze_signal(file_path)

    # Plot Waveform and Spectrum
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    fig.patch.set_facecolor('#0f111a')
    
    # Waveform
    ax1.set_facecolor('#181c2b')
    ax1.plot(data[:1000], color='#6366f1')
    ax1.set_title("WAVEFORM ANALYSIS: Time-Domain Signal x(t)", color='white')
    ax1.tick_params(colors='white')
    
    # Spectrum
    ax2.set_facecolor('#181c2b')
    ax2.bar(freqs[:70], fft_spectrum[:70], color='#10b981', width=freqs[1]-freqs[0])
    ax2.set_title("FREQUENCY SPECTRUM: Fast Fourier Transform X(f)", color='white')
    ax2.tick_params(colors='white')
    
    plt.tight_layout()
    st.pyplot(fig)

    # Display Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Observed Base Frequency (Sa)", f"{observed_f0:.1f} Hz")
    col2.metric("Calculated Pa Ratio (1.5x)", f"{(observed_f0 * 1.5):.1f} Hz")
    col3.metric("Calculated Taar Sa Ratio (2.0x)", f"{(observed_f0 * 2.0):.1f} Hz")

except Exception as e:
    st.error(f"Error loading audio file '{file_path}'. Please verify the file is pushed to GitHub. Details: {e}")
