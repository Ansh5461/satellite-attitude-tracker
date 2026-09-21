import soundfile as sf
import numpy as np
from scipy.signal import spectrogram
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = "/media/ansh/SDA1/its-onnn/data/move2_351020_2018-12-05/raw/satnogs_351020_2018-12-05T14-45-34.ogg"  # fill in

data, sr = sf.read(path)  # full read, mono float64
print("sr:", sr, "n_samples:", len(data), "duration:", len(data)/sr)

# Narrowband spectrogram — CW tone is likely a few hundred Hz wide at most,
# so we want good frequency resolution, time resolution can be coarser (~1s is fine)
nperseg = 8192
noverlap = int(nperseg * 0.75)

f, t, Sxx = spectrogram(data, fs=sr, nperseg=nperseg, noverlap=noverlap, window="hann")
Sxx_db = 10 * np.log10(Sxx + 1e-12)

# Restrict to plausible CW audio tone range for a first look — adjust after seeing the plot
freq_mask = f < 3000
plt.figure(figsize=(6, 10))
plt.pcolormesh(f[freq_mask], t, Sxx_db[freq_mask].T, shading="auto", cmap="viridis")
plt.xlabel("Audio Frequency (Hz)")
plt.ylabel("Time (s)")
plt.colorbar(label="Power (dB)")
plt.title("Audio spectrogram — 359214")
plt.tight_layout()
plt.savefig("audio_spectrogram_359214.png", dpi=130)
print("saved audio_spectrogram_359214.png")

# Numeric peak-frequency track: for each time bin, find the strongest bin
# within the plausible tone range, but only trust it above a power threshold
band = Sxx_db[freq_mask]
f_band = f[freq_mask]
peak_idx = band.argmax(axis=0)
peak_freq = f_band[peak_idx]
peak_power = band.max(axis=0)

np.save("audio_peak_t.npy", t)
np.save("audio_peak_freq.npy", peak_freq)
np.save("audio_peak_power.npy", peak_power)

print("power range:", peak_power.min(), peak_power.max())
print("freq range where power > median+10dB:",
      peak_freq[peak_power > np.median(peak_power)+10].min(),
      peak_freq[peak_power > np.median(peak_power)+10].max())