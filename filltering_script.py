import soundfile as sf
import numpy as np
from scipy.signal import spectrogram
import json, csv
from pathlib import Path

DATA_ROOT = Path("/media/ansh/SDA1/its-onnn/data")
OUT_CSV = Path("/media/ansh/SDA1/its-onnn/data/vetting_report.csv")

def analyze_observation(obs_dir):
    obs_id = obs_dir.name.split("_")[1]
    meta_files = list((obs_dir / "meta").glob("observation_*.json"))
    raw_files = list((obs_dir / "raw").glob("*.ogg"))
    if not meta_files or not raw_files:
        return None

    meta = json.loads(meta_files[0].read_text())
    audio_path = raw_files[0]

    data, sr = sf.read(str(audio_path))
    if data.ndim > 1:
        data = data[:, 0]  # just in case some ARE stereo I/Q

    nperseg = 8192
    noverlap = int(nperseg * 0.75)
    f, t, Sxx = spectrogram(data, fs=sr, nperseg=nperseg, noverlap=noverlap, window="hann")
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    # restrict to plausible CW audio tone range
    band_mask = f < 3000
    band = Sxx_db[band_mask]
    f_band = f[band_mask]

    peak_idx = band.argmax(axis=0)
    peak_freq = f_band[peak_idx]
    peak_power = band.max(axis=0)

    noise_floor = np.median(peak_power)
    strong = peak_power > noise_floor + 10

    freq_excursion_hz = (peak_freq[strong].max() - peak_freq[strong].min()) if strong.any() else 0.0
    snr_db = peak_power.max() - noise_floor
    frac_time_strong = strong.mean()

    radio_meta = {}
    try:
        radio_meta = json.loads(meta.get("client_metadata", "{}")).get("radio", {})
    except Exception:
        pass

    return {
        "obs_id": obs_id,
        "station": meta.get("station_name"),
        "max_altitude": meta.get("max_altitude"),
        "duration_s": len(data) / sr,
        "rf_gain": radio_meta.get("rf_gain"),
        "if_gain": radio_meta.get("if_gain"),
        "bb_gain": radio_meta.get("bb_gain"),
        "fixed_gain": radio_meta.get("rf_gain") is not None,  # crude proxy for "not AGC"
        "freq_excursion_hz": round(freq_excursion_hz, 2),
        "doppler_corrected_guess": freq_excursion_hz < 200,  # tune threshold once you see the distribution
        "snr_db": round(snr_db, 2),
        "frac_time_strong": round(frac_time_strong, 3),
        "vetted_status": meta.get("vetted_status"),
        "waterfall_status": meta.get("waterfall_status"),
    }

rows = []
for obs_dir in sorted(DATA_ROOT.glob("move2_*")):
    try:
        r = analyze_observation(obs_dir)
        if r:
            rows.append(r)
            print(r["obs_id"], r["freq_excursion_hz"], r["snr_db"], r["fixed_gain"])
    except Exception as e:
        print("FAILED:", obs_dir.name, e)

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nWrote {len(rows)} rows to {OUT_CSV}")