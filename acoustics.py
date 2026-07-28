import streamlit as st
import numpy as np
from scipy.io import wavfile
import librosa

st.set_page_config(page_title="Acoustic Signal Analysis", layout="wide")

st.title("Acoustic Signal Analysis & Harmonic Ratio Study")

# Audio file mapping
SOUND_FILES = {
    'Temple Bell': 'Templebell.mp3',
    'Shankh (Conch)': 'shell.mp3',
    'Tanpura (Sa)': 'tanpura.mp3'
}

# 1. Select Instrument
selected_name = st.selectbox("Select Instrument / Audio Source:", list(SOUND_FILES.keys()))
file_path = SOUND_FILES[selected_name]

# 2. Audio Player
st.audio(file_path)

# 3. Fast Signal Processing Function
@st.cache_data
def load_audio_fast(path):
    # Fast load audio data
    data, sr = librosa.load(path, sr=22050)
    return data, sr

try:
    data, sr = load_audio_fast(file_path)
    total_duration = len(data) / sr

    # 4. Interactive Time Slider (To view dynamic waveform across time)
    st.subheader("Signal Timeline Control")
    start_time = st.slider("Analyze Audio at Timestamp (seconds):", 0.0, float(max(0.1, total_duration - 0.1)), 0.0, step=0.1)

    # Extract 2048 samples at the selected timestamp
    start_sample = int(start_time * sr)
    chunk_size = 2048
    chunk = data[start_sample : start_sample + chunk_size]

    if len(chunk) < chunk_size:
        chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

    # Compute FFT
    fft_spectrum = np.abs(np.fft.rfft(chunk))
    freqs = np.fft.rfftfreq(chunk_size, 1.0 / sr)

    # Find peak frequency (Sa)
    peak_idx = np.argmax(fft_spectrum)
    observed_f0 = freqs[peak_idx]

    # 5. Display Waveform Chart
    st.subheader("WAVEFORM ANALYSIS: Time-Domain Signal x(t)")
    st.line_chart(chunk[:1000], height=180)

    # 6. Display Frequency Spectrum Chart
    st.subheader("FREQUENCY SPECTRUM: Fast Fourier Transform X(f)")
    st.bar_chart(fft_spectrum[:100], height=200)

    # 7. Display Calculated Ratios
    st.markdown("### Harmonic Ratio Calculations")
    col1, col2, col3 = st.columns(3)
    col1.metric("Observed Base Frequency (Sa)", f"{observed_f0:.1f} Hz")
    col2.metric("Calculated Pa Ratio (1.5x)", f"{(observed_f0 * 1.5):.1f} Hz")
    col3.metric("Calculated Taar Sa Ratio (2.0x)", f"{(observed_f0 * 2.0):.1f} Hz")

except Exception as e:
    st.error(f"Error loading audio file '{file_path}'. Please ensure the file exists in your repository. Details: {e}")
