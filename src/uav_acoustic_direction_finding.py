# -*- coding: utf-8 -*-
"""
This program analyzes direction of arrival (DOA) and acoustic features from audio
recorded with a multichannel microphone array.

The algorithm divides the input WAV file into short time windows (snapshots) and
calculates signal energy, spectral features, and a direction estimate for each
window. Direction finding uses SRP-PHAT (Steered Response Power with Phase
Transform) in both a primary frequency band and a multiband configuration.

Snapshots are evaluated using energy and confidence criteria and filtered at two
levels: soft and strict. Rejecting noisy or low-confidence estimates provides more
stable direction tracking.

The program also produces:
- Direction of arrival over time
- A beamforming DOA heatmap
- Spectrogram and average-spectrum analyses
- Confidence scores and a detection timeline

The final overall DOA is calculated from the selected high-confidence snapshots
using a weighted average. Frequency components are also analyzed to characterize
the UAV acoustic signature.

The system is designed for real-time or post-processing analysis of moving targets,
including UAVs following circular flight paths. It has been developed using
recordings collected with different hardware in multiple environments so that it
can accommodate devices with different acoustic harmonics. This version processes
WAV recordings; a live-radar variant remains under development and testing.

Current hardware:
reSpeaker V3 USB 4Mic-ARRAY XVF3000 (the array geometry is defined below).
Version: 25.03.2026 V1.12
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.signal import butter, filtfilt, get_window, stft, medfilt, find_peaks

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "input.wav")
OPTIONAL_NOISE_PATH = os.path.join(BASE_DIR, "data", "analiz.wav")
OUTPUT_ROOT = os.path.join(BASE_DIR, "results")

MODE = "circle"

CHANNELS_TO_USE = [1, 2, 3, 4]

WINDOW_TYPE = "hann"
ANGLES = np.linspace(-180, 180, 361)

IGNORE_FIRST_SEC = 4.0

DISPLAY_SPEC_MAX_HZ = 3000.0

SAVE_PLOTS = True
SHOW_PLOTS = True

EXPORT_PRESENTATION_NOTES = True

# Mode configuration: hover or circle
def configure_mode(mode):
    global MODE, SNAPSHOT_SEC, HOP_SEC, MAIN_BAND, SELECTION_BAND
    global MULTI_BANDS, USE_TRACKING, TRACK_LAMBDA, USE_SMOOTHING
    global SMOOTH_KERNEL, STRICT_TOP_ENERGY_PERCENT, SOFT_TOP_ENERGY_PERCENT
    global MIN_RUN_LENGTH_STRICT, MIN_RUN_LENGTH_SOFT
    global USE_SPECTRAL_FLATNESS_FILTER, MAX_FLATNESS_STRICT
    global MAX_FLATNESS_SOFT, USE_PEAK_RATIO_FILTER, PEAK_RATIO_THR
    global USE_SNR_FILTER, MIN_SNR_DB, USE_PHAT, PHAT_ALPHA

    MODE = mode
    MAIN_BAND = (400.0, 1200.0)
    MULTI_BANDS = [
        (400.0, 800.0),
        (800.0, 1400.0),
        (1400.0, 2200.0),
    ]
    USE_TRACKING = True
    USE_SMOOTHING = True
    USE_SPECTRAL_FLATNESS_FILTER = True
    USE_PEAK_RATIO_FILTER = False
    USE_SNR_FILTER = False
    USE_PHAT = True

    if mode == "hover":
        SNAPSHOT_SEC = 0.10
        HOP_SEC = 0.05
        SELECTION_BAND = (500.0, 1500.0)
        TRACK_LAMBDA = 0.003
        SMOOTH_KERNEL = 7
        STRICT_TOP_ENERGY_PERCENT = 35
        SOFT_TOP_ENERGY_PERCENT = 60
        MIN_RUN_LENGTH_STRICT = 2
        MIN_RUN_LENGTH_SOFT = 1
        MAX_FLATNESS_STRICT = 0.75
        MAX_FLATNESS_SOFT = 0.90
        PEAK_RATIO_THR = 1.08
        MIN_SNR_DB = 3.0
        PHAT_ALPHA = 0.6
    elif mode == "circle":
        SNAPSHOT_SEC = 0.05
        HOP_SEC = 0.025
        SELECTION_BAND = (500.0, 1700.0)
        TRACK_LAMBDA = 0.0005
        SMOOTH_KERNEL = 3
        STRICT_TOP_ENERGY_PERCENT = 45
        SOFT_TOP_ENERGY_PERCENT = 70
        MIN_RUN_LENGTH_STRICT = 1
        MIN_RUN_LENGTH_SOFT = 1
        MAX_FLATNESS_STRICT = 0.90
        MAX_FLATNESS_SOFT = 0.98
        PEAK_RATIO_THR = 1.04
        MIN_SNR_DB = 2.0
        PHAT_ALPHA = 0.5
    else:
        raise ValueError(f"Unsupported mode: {mode}")


configure_mode(MODE)

# Microphone array geometry
r = 0.032
d = r / np.sqrt(2.0)

MIC_POS = np.array([
    [-d, -d],   # A: lower-left
    [+d, -d],   # B: lower-right
    [+d, +d],   # C: upper-right
    [-d, +d],   # D: upper-left
], dtype=float)

C_SOUND = 343.0

# Helper functions
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0

def ang_dist_deg(a, b):
    return np.abs(wrap180(a - b))

def circ_diff_deg(a, b):
    return wrap180(a - b)

def smooth_angle_deg(a_deg, kernel_size=7):
    a = a_deg.copy()
    valid = np.isfinite(a)
    if np.sum(valid) < kernel_size or kernel_size < 3:
        return a

    rad = np.deg2rad(a[valid])
    rad_unwrapped = np.unwrap(rad)
    rad_smoothed = medfilt(rad_unwrapped, kernel_size=kernel_size)
    a[valid] = wrap180(np.rad2deg(rad_smoothed))
    return a

def read_wav_multichannel(path):
    fs, x = wavfile.read(path)

    if x.ndim == 1:
        raise ValueError(f"{path}: the recording appears to be single-channel; multichannel audio is required.")

    orig_dtype = x.dtype
    x = x.astype(np.float32)

    if np.issubdtype(orig_dtype, np.integer):
        max_val = max(abs(np.iinfo(orig_dtype).min), np.iinfo(orig_dtype).max)
        x /= float(max_val)

    return fs, x

def select_mics(x, chs):
    if x.shape[1] < 6:
        raise ValueError(
            f"The WAV file has fewer channels than expected. Expected at least 6, found {x.shape[1]}."
        )

    if max(chs) >= x.shape[1]:
        raise ValueError(
            f"Channel index is out of range. WAV channels={x.shape[1]}, requested={chs}."
        )

    X = x[:, chs]
    if X.shape[1] != 4:
        raise ValueError("Could not select four channels for microphones A, B, C, and D.")
    return X

def bandpass_filter(X, fs, f_lo, f_hi, order=4):
    nyq = fs / 2.0
    f_hi = min(f_hi, nyq * 0.999)

    if not (0 < f_lo < f_hi < nyq):
        raise ValueError(f"Invalid band-pass limits: {f_lo}-{f_hi}, Nyquist={nyq}.")

    b, a = butter(order, [f_lo / nyq, f_hi / nyq], btype="band")
    return filtfilt(b, a, X, axis=0)

def spectrogram_1ch(x, fs):
    nperseg = max(256, int(0.032 * fs))
    noverlap = max(0, nperseg - int(0.010 * fs))
    f, t, Z = stft(
        x, fs=fs, window="hann",
        nperseg=nperseg, noverlap=noverlap,
        boundary=None, padded=False
    )
    S_db = 20.0 * np.log10(np.maximum(np.abs(Z), 1e-10))
    return f, t, S_db

def direction_unit_vector(angle_deg):
    """Return the unit vector for 0° down, 90° right, and 180° up."""
    theta = np.deg2rad(angle_deg)
    return np.array([np.sin(theta), -np.cos(theta)], dtype=float)


def steering_srp_phat(mic_xy, freqs, angles_deg, c=343.0):
    """
    Angle convention:
    0° = -Y (down), 90° = +X (right), 180° = +Y (up), -90° = -X (left).
    """
    A = len(angles_deg)
    M = mic_xy.shape[0]
    F = len(freqs)
    S = np.empty((A, M, F), dtype=np.complex64)

    for ai, ang in enumerate(angles_deg):
        u = direction_unit_vector(ang)
        tau = (mic_xy @ u) / c
        S[ai] = np.exp(-1j * 2.0 * np.pi * freqs[None, :] * tau[:, None]).astype(np.complex64)

    return S

def contiguous_runs(mask):
    runs = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif (not v) and (start is not None):
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs

def enforce_min_run(mask, min_run_len):
    if min_run_len <= 1:
        return mask.copy()

    out = mask.copy()
    for s, e in contiguous_runs(mask):
        if (e - s + 1) < min_run_len:
            out[s:e+1] = False
    return out

def normalize_01(x):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)

    valid = np.isfinite(x)
    if not np.any(valid):
        return out

    xmin = np.nanmin(x)
    xmax = np.nanmax(x)

    if xmax - xmin < 1e-12:
        out[valid] = 0.5
        return out

    out[valid] = (x[valid] - xmin) / (xmax - xmin)
    return out

def spectral_metrics_from_power(freqs, power):
    power = np.maximum(power, 1e-20)
    total = np.sum(power)

    if total <= 0:
        return np.nan, np.nan, np.nan, np.nan

    centroid = np.sum(freqs * power) / total
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total)
    dominant = freqs[int(np.argmax(power))]

    geom_mean = np.exp(np.mean(np.log(power)))
    arith_mean = np.mean(power)
    flatness = geom_mean / max(arith_mean, 1e-20)

    return dominant, centroid, bandwidth, float(flatness)

def estimate_snr_db(signal_energy, noise_energy):
    if noise_energy is None or noise_energy <= 0:
        return np.nan
    return 10.0 * np.log10(max(signal_energy / noise_energy, 1e-12))

def robust_nanpercentile(x, q, fallback=np.nan):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return fallback
    return float(np.percentile(x, q))

def safe_nanmean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.mean(x))

def compute_detection_level(keep_soft, keep_strict):
    level = np.zeros(len(keep_soft), dtype=np.int32)
    level[np.asarray(keep_soft).astype(bool)] = 1
    level[np.asarray(keep_strict).astype(bool)] = 2
    return level

def weighted_circular_mean_deg(angles_deg, weights):
    angles_deg = np.asarray(angles_deg, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = np.isfinite(angles_deg) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan

    ang = np.deg2rad(angles_deg[valid])
    w = weights[valid]

    x = np.sum(w * np.cos(ang))
    y = np.sum(w * np.sin(ang))
    return wrap180(np.rad2deg(np.arctan2(y, x)))

def circular_median_approx_deg(angles_deg):
    angles_deg = np.asarray(angles_deg, dtype=float)
    valid = np.isfinite(angles_deg)
    if not np.any(valid):
        return np.nan

    candidates = angles_deg[valid]
    best = candidates[0]
    best_cost = np.inf

    for a in candidates:
        cost = np.sum(np.abs(wrap180(candidates - a)))
        if cost < best_cost:
            best_cost = cost
            best = a
    return float(wrap180(best))

def compute_angular_velocity_deg_per_s(times, doa_deg):
    out = np.full_like(doa_deg, np.nan, dtype=float)
    valid = np.isfinite(doa_deg)
    idx = np.where(valid)[0]

    if len(idx) < 2:
        return out

    for i in range(1, len(idx)):
        i0 = idx[i - 1]
        i1 = idx[i]
        dt = times[i1] - times[i0]
        if dt <= 0:
            continue
        da = wrap180(doa_deg[i1] - doa_deg[i0])
        out[i1] = da / dt

    return out

# Noise reference
def compute_noise_floor_from_optional_file(noise_path, selection_band):
    if not os.path.exists(noise_path):
        print(f"[Info] Noise reference file not found: {noise_path}")
        print("[Info] Continuing without a noise reference.")
        return None

    print(f"[Info] Noise reference file found: {noise_path}")

    fs_n, x_n = read_wav_multichannel(noise_path)
    X_n = select_mics(x_n, CHANNELS_TO_USE)

    X_sel = bandpass_filter(X_n, fs_n, selection_band[0], selection_band[1], order=4)

    L = int(SNAPSHOT_SEC * fs_n)
    H = int(HOP_SEC * fs_n)

    if X_n.shape[0] < L:
        print("[Warning] The noise recording is shorter than one snapshot and will not be used.")
        return None

    win = get_window(WINDOW_TYPE, L).astype(np.float32)
    K = 1 + (X_n.shape[0] - L) // H

    sel_energies = []

    for k in range(K):
        s = k * H
        e = s + L
        frame_sel = X_sel[s:e, :] * win[:, None]
        sel_energies.append(float(np.mean(frame_sel ** 2)))

    return {
        "sel_noise_p50": float(np.percentile(sel_energies, 50)),
        "sel_noise_p95": float(np.percentile(sel_energies, 95)),
    }

# Beam analysis
def compute_band_srp(frame_bp, fs, band_lo, band_hi, steer_cache):
    L = frame_bp.shape[0]
    freqs_full = np.fft.rfftfreq(L, d=1.0 / fs)
    band_idx = np.where((freqs_full >= band_lo) & (freqs_full <= min(band_hi, fs / 2.0)))[0]

    if len(band_idx) < 2:
        return None

    band_key = (L, fs, band_lo, band_hi)
    if band_key not in steer_cache:
        band_freqs = freqs_full[band_idx].astype(np.float32)
        steer_cache[band_key] = (
            band_idx,
            band_freqs,
            steering_srp_phat(MIC_POS, band_freqs, ANGLES, c=C_SOUND)
        )

    band_idx, band_freqs, steer = steer_cache[band_key]

    Xf_full = np.fft.rfft(frame_bp, axis=0)
    Xf_band = Xf_full[band_idx, :].T

    Xf_use = Xf_band.copy()
    if USE_PHAT:
        Xf_use = Xf_use / (np.abs(Xf_use) ** PHAT_ALPHA + 1e-12)

    Y = np.mean(np.conj(steer) * Xf_use[None, :, :], axis=1)
    P = np.mean(np.abs(Y) ** 2, axis=1)

    i1 = int(np.argmax(P))
    p1 = float(P[i1])
    P2 = P.copy()
    P2[i1] = -np.inf
    p2 = float(np.max(P2))
    ratio = p1 / (p2 + 1e-12)
    doa = float(ANGLES[i1])

    return {
        "P": P.astype(np.float32),
        "doa": doa,
        "peak_ratio": ratio,
        "peak_power": p1
    }

# Main analysis
def analyze_recording(path, noise_floor=None):
    fs, x_raw = read_wav_multichannel(path)
    X_raw = select_mics(x_raw, CHANNELS_TO_USE)

    L = int(SNAPSHOT_SEC * fs)
    H = int(HOP_SEC * fs)

    if X_raw.shape[0] < L:
        raise ValueError(f"{path}: the recording is shorter than one snapshot.")

    X_main = bandpass_filter(X_raw, fs, MAIN_BAND[0], MAIN_BAND[1], order=4)
    X_sel = bandpass_filter(X_raw, fs, SELECTION_BAND[0], SELECTION_BAND[1], order=4)

    X_multi = []
    for b_lo, b_hi in MULTI_BANDS:
        X_multi.append(bandpass_filter(X_raw, fs, b_lo, b_hi, order=4))

    win = get_window(WINDOW_TYPE, L).astype(np.float32)
    K = 1 + (X_raw.shape[0] - L) // H
    times = (np.arange(K) * H + L / 2.0) / fs

    freqs_full = np.fft.rfftfreq(L, d=1.0 / fs)
    steer_cache = {}

    Pscan_main = np.zeros((K, len(ANGLES)), dtype=np.float32)

    doa_main_raw = np.full(K, np.nan, dtype=np.float32)
    doa_multi_raw = np.full(K, np.nan, dtype=np.float32)

    peak_ratio_main = np.full(K, np.nan, dtype=np.float32)
    peak_power_main = np.full(K, np.nan, dtype=np.float32)

    peak_ratio_multi = np.full(K, np.nan, dtype=np.float32)
    peak_power_multi = np.full(K, np.nan, dtype=np.float32)

    total_energy = np.zeros(K, dtype=np.float32)
    main_energy = np.zeros(K, dtype=np.float32)
    selection_energy = np.zeros(K, dtype=np.float32)

    dominant_freq = np.full(K, np.nan, dtype=np.float32)
    spectral_centroid = np.full(K, np.nan, dtype=np.float32)
    spectral_bandwidth = np.full(K, np.nan, dtype=np.float32)
    spectral_flatness = np.full(K, np.nan, dtype=np.float32)
    estimated_snr_db = np.full(K, np.nan, dtype=np.float32)

    prev_angle_main = None
    prev_angle_multi = None
    sel_noise_ref = None if noise_floor is None else noise_floor.get("sel_noise_p95", None)

    for k in range(K):
        s = k * H
        e = s + L

        frame_raw = X_raw[s:e, :] * win[:, None]
        frame_main = X_main[s:e, :] * win[:, None]
        frame_sel = X_sel[s:e, :] * win[:, None]

        total_energy[k] = float(np.mean(frame_raw ** 2))
        main_energy[k] = float(np.mean(frame_main ** 2))
        selection_energy[k] = float(np.mean(frame_sel ** 2))
        estimated_snr_db[k] = estimate_snr_db(selection_energy[k], sel_noise_ref)

        Xf_full = np.fft.rfft(frame_main, axis=0)
        avg_power = np.mean(np.abs(Xf_full) ** 2, axis=1)

        df, cent, bw, flat = spectral_metrics_from_power(freqs_full, avg_power)
        dominant_freq[k] = df
        spectral_centroid[k] = cent
        spectral_bandwidth[k] = bw
        spectral_flatness[k] = flat

        res_main = compute_band_srp(frame_main, fs, MAIN_BAND[0], MAIN_BAND[1], steer_cache)
        if res_main is not None:
            P = res_main["P"]
            Pscan_main[k, :] = P
            doa_cand = res_main["doa"]
            peak_ratio_main[k] = res_main["peak_ratio"]
            peak_power_main[k] = res_main["peak_power"]

            if USE_TRACKING and prev_angle_main is not None:
                penalty = TRACK_LAMBDA * (ang_dist_deg(ANGLES, prev_angle_main) ** 2)
                idx = int(np.argmax(P - penalty))
                doa_main_raw[k] = ANGLES[idx]
            else:
                doa_main_raw[k] = doa_cand
            prev_angle_main = doa_main_raw[k]

        band_doas = []
        band_weights = []
        band_peak_powers = []
        band_peak_ratios = []

        for (b_lo, b_hi), Xb in zip(MULTI_BANDS, X_multi):
            frame_b = Xb[s:e, :] * win[:, None]
            res_b = compute_band_srp(frame_b, fs, b_lo, b_hi, steer_cache)
            if res_b is None:
                continue

            band_doas.append(res_b["doa"])
            w = max(res_b["peak_power"], 1e-12) * max(res_b["peak_ratio"], 1e-12)
            band_weights.append(w)
            band_peak_powers.append(res_b["peak_power"])
            band_peak_ratios.append(res_b["peak_ratio"])

        if len(band_doas) > 0:
            fused = weighted_circular_mean_deg(band_doas, band_weights)
            peak_power_multi[k] = float(np.mean(band_peak_powers))
            peak_ratio_multi[k] = float(np.mean(band_peak_ratios))

            if USE_TRACKING and prev_angle_multi is not None:
                candidate_angles = np.array(band_doas + [fused], dtype=float)
                candidate_weights = np.array(band_weights + [np.sum(band_weights)], dtype=float)
                penalty = np.abs(wrap180(candidate_angles - prev_angle_multi))
                score = candidate_weights / (1.0 + 0.05 * penalty)
                doa_multi_raw[k] = candidate_angles[int(np.argmax(score))]
            else:
                doa_multi_raw[k] = fused

            prev_angle_multi = doa_multi_raw[k]

# ----------------------------
# ----------------------------
    valid_time_mask = times >= IGNORE_FIRST_SEC
    if not np.any(valid_time_mask):
        valid_time_mask = np.ones_like(times, dtype=bool)

    thr_strict = np.percentile(
        selection_energy[valid_time_mask],
        100 - STRICT_TOP_ENERGY_PERCENT
    )
    keep_strict = selection_energy >= thr_strict
    keep_strict &= valid_time_mask

    if USE_SPECTRAL_FLATNESS_FILTER:
        keep_strict &= (spectral_flatness <= MAX_FLATNESS_STRICT)

    if USE_PEAK_RATIO_FILTER:
        keep_strict &= (peak_ratio_main >= PEAK_RATIO_THR)

    if USE_SNR_FILTER:
        keep_strict &= np.isfinite(estimated_snr_db) & (estimated_snr_db >= MIN_SNR_DB)

    keep_strict = enforce_min_run(keep_strict, MIN_RUN_LENGTH_STRICT)

    if np.sum(keep_strict) < 5:
        fallback_thr = np.percentile(selection_energy[valid_time_mask], 80)
        keep_strict = (selection_energy >= fallback_thr) & valid_time_mask
        keep_strict = enforce_min_run(keep_strict, 1)

    if np.sum(keep_strict) == 0:
        idx = np.argmax(selection_energy * valid_time_mask.astype(float))
        keep_strict[idx] = True

    thr_soft = np.percentile(
        selection_energy[valid_time_mask],
        100 - SOFT_TOP_ENERGY_PERCENT
    )
    keep_soft = selection_energy >= thr_soft
    keep_soft &= valid_time_mask

    if USE_SPECTRAL_FLATNESS_FILTER:
        keep_soft &= (spectral_flatness <= MAX_FLATNESS_SOFT)

    keep_soft = enforce_min_run(keep_soft, MIN_RUN_LENGTH_SOFT)

    if np.sum(keep_soft) < np.sum(keep_strict):
        keep_soft = keep_strict.copy()

    doa_main_smooth = smooth_angle_deg(doa_main_raw, kernel_size=SMOOTH_KERNEL) if USE_SMOOTHING else doa_main_raw.copy()
    doa_multi_smooth = smooth_angle_deg(doa_multi_raw, kernel_size=SMOOTH_KERNEL) if USE_SMOOTHING else doa_multi_raw.copy()

    mean_scan_strict = np.nanmean(np.where(keep_strict[:, None], Pscan_main, np.nan), axis=0)
    doa_overall_main = ANGLES[int(np.nanargmax(mean_scan_strict))]

    doa_overall_multi = weighted_circular_mean_deg(
        doa_multi_smooth[keep_strict],
        np.maximum(peak_ratio_multi[keep_strict], 1e-12)
    )

    pr_norm = normalize_01(peak_ratio_main)
    pr_multi_norm = normalize_01(peak_ratio_multi)
    snr_norm = normalize_01(estimated_snr_db)
    flat_penalty = 1.0 - np.clip(spectral_flatness, 0.0, 1.0)

    confidence = np.full(K, np.nan, dtype=np.float32)
    for k in range(K):
        vals = []
        if np.isfinite(pr_norm[k]):
            vals.append(pr_norm[k])
        if np.isfinite(pr_multi_norm[k]):
            vals.append(pr_multi_norm[k])
        if np.isfinite(snr_norm[k]):
            vals.append(snr_norm[k])
        if np.isfinite(flat_penalty[k]):
            vals.append(flat_penalty[k])
        confidence[k] = np.mean(vals) if len(vals) > 0 else np.nan

    angular_velocity = compute_angular_velocity_deg_per_s(times, doa_multi_smooth)

    fsp, tsp, Ssp = spectrogram_1ch(X_raw[:, 0], fs)

    avg_spec_accum = []
    for k in np.where(keep_strict)[0]:
        s = k * H
        e = s + L
        frame = X_main[s:e, 0] * win
        Xf = np.fft.rfft(frame)
        avg_spec_accum.append(np.abs(Xf) ** 2)

    if len(avg_spec_accum) > 0:
        avg_spec = np.mean(np.array(avg_spec_accum), axis=0)
    else:
        avg_spec = np.zeros_like(freqs_full)

    peaks, _ = find_peaks(
        avg_spec,
        prominence=np.max(avg_spec) * 0.05 if np.max(avg_spec) > 0 else 1.0
    )
    peak_freqs = freqs_full[peaks]
    peak_powers = avg_spec[peaks]

    harmonic_info = []
    if len(peak_freqs) > 0:
        candidate_f0s = peak_freqs[(peak_freqs >= 80) & (peak_freqs <= 1200)]
        if len(candidate_f0s) > 0:
            f0 = float(candidate_f0s[0])
            for pf, pp in zip(peak_freqs, peak_powers):
                hn = int(np.round(pf / f0)) if f0 > 0 else 0
                if hn >= 1 and abs(pf - hn * f0) <= max(20.0, 0.05 * hn * f0):
                    harmonic_info.append((pf, pp, hn))
        else:
            f0 = np.nan
    else:
        f0 = np.nan

    strict_energy_thr = thr_strict
    soft_energy_thr = thr_soft
    confidence_soft_thr = robust_nanpercentile(confidence[keep_soft], 30, fallback=np.nan)
    confidence_strict_thr = robust_nanpercentile(confidence[keep_strict], 30, fallback=np.nan)

    snapshot_df = pd.DataFrame({
        "time": times,
        "doa_main_raw": doa_main_raw,
        "doa_main_smooth": doa_main_smooth,
        "doa_multi_raw": doa_multi_raw,
        "doa_multi_smooth": doa_multi_smooth,
        "keep_strict": keep_strict.astype(int),
        "keep_soft": keep_soft.astype(int),
        "detection_level": compute_detection_level(keep_soft, keep_strict),
        "total_energy": total_energy,
        "main_energy": main_energy,
        "selection_band_energy": selection_energy,
        "dominant_freq": dominant_freq,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_flatness": spectral_flatness,
        "estimated_snr_db": estimated_snr_db,
        "peak_ratio_main": peak_ratio_main,
        "peak_ratio_multi": peak_ratio_multi,
        "peak_power_main": peak_power_main,
        "peak_power_multi": peak_power_multi,
        "confidence": confidence,
        "angular_velocity_deg_s": angular_velocity
    })

    return {
        "path": path,
        "name": os.path.splitext(os.path.basename(path))[0],
        "fs": fs,
        "times": times,
        "snapshot_df": snapshot_df,
        "Pscan_main": Pscan_main,
        "mean_scan_strict": mean_scan_strict,
        "doa_overall_main": doa_overall_main,
        "doa_overall_multi": doa_overall_multi,
        "fsp": fsp,
        "tsp": tsp,
        "Ssp": Ssp,
        "freqs_full": freqs_full,
        "avg_spec": avg_spec,
        "peak_freqs": peak_freqs,
        "peak_powers": peak_powers,
        "harmonic_info": harmonic_info,
        "fundamental_estimate": f0,
        "strict_energy_thr": strict_energy_thr,
        "soft_energy_thr": soft_energy_thr,
        "confidence_soft_thr": confidence_soft_thr,
        "confidence_strict_thr": confidence_strict_thr
    }

# Per-second summary
def per_second_summary(snapshot_df):
    sec_idx = np.floor(snapshot_df["time"].values).astype(int)
    rows = []

    for sec in np.unique(sec_idx):
        block = snapshot_df.iloc[np.where(sec_idx == sec)[0]]

        soft_block = block[block["keep_soft"] == 1]
        strict_block = block[block["keep_strict"] == 1]

        doa_soft = soft_block["doa_multi_smooth"].values
        doa_strict = strict_block["doa_multi_smooth"].values

        dom_vals = strict_block["dominant_freq"].values
        cen_vals = strict_block["spectral_centroid"].values
        en_vals = strict_block["selection_band_energy"].values
        snr_vals = strict_block["estimated_snr_db"].values
        conf_vals = strict_block["confidence"].values
        vel_vals = strict_block["angular_velocity_deg_s"].values

        rows.append({
            "second_index": sec,
            "doa_soft_median": circular_median_approx_deg(doa_soft) if len(doa_soft) > 0 else np.nan,
            "doa_strict_median": circular_median_approx_deg(doa_strict) if len(doa_strict) > 0 else np.nan,
            "doa_soft_mean": weighted_circular_mean_deg(doa_soft, np.ones(len(doa_soft))) if len(doa_soft) > 0 else np.nan,
            "doa_strict_mean": weighted_circular_mean_deg(doa_strict, np.ones(len(doa_strict))) if len(doa_strict) > 0 else np.nan,
            "doa_variance_proxy": float(np.nanvar(doa_strict)) if len(doa_strict) > 0 else np.nan,
            "doa_confidence": float(np.nanmean(conf_vals)) if len(conf_vals) > 0 else np.nan,
            "snapshot_count": int(len(block)),
            "soft_snapshot_count": int(len(soft_block)),
            "strict_snapshot_count": int(len(strict_block)),
            "dominant_freq_median": float(np.nanmedian(dom_vals)) if len(dom_vals) > 0 else np.nan,
            "centroid_mean": float(np.nanmean(cen_vals)) if len(cen_vals) > 0 else np.nan,
            "energy_mean": float(np.nanmean(en_vals)) if len(en_vals) > 0 else np.nan,
            "snr_mean": float(np.nanmean(snr_vals)) if len(snr_vals) > 0 else np.nan,
            "angular_velocity_mean": float(np.nanmean(np.abs(vel_vals))) if len(vel_vals) > 0 else np.nan,
        })

    return pd.DataFrame(rows)

def save_or_show(fig, path=None):
    if SAVE_PLOTS and path is not None:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show(block=False)
    else:
        plt.close(fig)

def plot_doa_vs_time(result, out_dir):
    df = result["snapshot_df"]
    fig = plt.figure(figsize=(12, 5))
    plt.plot(df["time"], df["doa_main_raw"], alpha=0.20, linewidth=1.0, label="main raw")
    plt.plot(df["time"], df["doa_main_smooth"], linewidth=1.1, alpha=0.75, label="main smooth")
    plt.plot(df["time"], df["doa_multi_smooth"], linewidth=2.0, label="multi smooth")

    soft = df["keep_soft"].values.astype(bool)
    strict = df["keep_strict"].values.astype(bool)

    plt.scatter(df["time"][soft], df["doa_multi_smooth"][soft], s=10, label="soft keep")
    plt.scatter(df["time"][strict], df["doa_multi_smooth"][strict], s=18, label="strict keep")

    if np.isfinite(result["doa_overall_multi"]):
        plt.axhline(result["doa_overall_multi"], linestyle=":", linewidth=1.5, label=f"overall multi={result['doa_overall_multi']:.1f}°")

    plt.title(f'{result["name"]} - DOA vs Time ({MODE})')
    plt.xlabel("Time (s)")
    plt.ylabel("Azimuth (deg)")
    plt.ylim(-180, 180)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3)
    plt.tight_layout()
    save_or_show(fig, os.path.join(out_dir, f"{result['name']}_doa_vs_time.png"))

def plot_doa_heatmap(result, out_dir):
    df = result["snapshot_df"]
    Pmap = result["Pscan_main"].copy()

    soft = df["keep_soft"].values.astype(bool)
    Pmap[~soft, :] = np.nan
    Pmap_db = 10.0 * np.log10(np.maximum(Pmap, 1e-12)).T

    fig = plt.figure(figsize=(12, 5))
    finite = np.isfinite(Pmap_db)
    vmin = np.nanpercentile(Pmap_db, 20) if np.any(finite) else -80
    vmax = np.nanpercentile(Pmap_db, 99) if np.any(finite) else 0

    plt.pcolormesh(result["times"], ANGLES, Pmap_db, shading="auto", vmin=vmin, vmax=vmax)
    plt.plot(df["time"], df["doa_multi_smooth"], linewidth=1.5, label="multi smooth")
    plt.title(f'{result["name"]} - DOA Heatmap (soft keep)')
    plt.xlabel("Time (s)")
    plt.ylabel("Azimuth (deg)")
    plt.colorbar(label="Power (dB)")
    plt.legend()
    plt.tight_layout()
    save_or_show(fig, os.path.join(out_dir, f"{result['name']}_doa_heatmap.png"))

def plot_beam_power_scan(result, out_dir):
    fig = plt.figure(figsize=(10, 4.5))
    peak_ang = ANGLES[int(np.nanargmax(result["mean_scan_strict"]))]
    beam_db = 10.0 * np.log10(np.maximum(result["mean_scan_strict"], 1e-12))
    plt.plot(ANGLES, beam_db, linewidth=2.0, label="strict mean beam power")
    plt.axvline(peak_ang, linestyle="--", label=f"main peak={peak_ang:.1f}°")
    if np.isfinite(result["doa_overall_multi"]):
        plt.axvline(result["doa_overall_multi"], linestyle=":", label=f"multi overall={result['doa_overall_multi']:.1f}°")

    plt.title(f'{result["name"]} - Beamforming Power Scan (strict)')
    plt.xlabel("Angle (deg)")
    plt.ylabel("Mean beam power (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_or_show(fig, os.path.join(out_dir, f"{result['name']}_beam_power_scan.png"))

def plot_spectrogram(result, out_dir):
    fig = plt.figure(figsize=(12, 5))
    plt.pcolormesh(result["tsp"], result["fsp"], result["Ssp"], shading="auto")
    plt.ylim(0, min(DISPLAY_SPEC_MAX_HZ, result["fs"] / 2.0))
    plt.axhline(MAIN_BAND[0], linestyle="--", linewidth=1.0, label="main band")
    plt.axhline(MAIN_BAND[1], linestyle="--", linewidth=1.0)
    plt.axhline(SELECTION_BAND[0], linestyle=":", linewidth=1.2, label="selection band")
    plt.axhline(SELECTION_BAND[1], linestyle=":", linewidth=1.2)
    plt.title(f'{result["name"]} - Spectrogram (Mic A raw)')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="dB")
    plt.legend(loc="upper right")
    plt.tight_layout()
    save_or_show(fig, os.path.join(out_dir, f"{result['name']}_spectrogram.png"))

def plot_average_spectrum(result, out_dir):
    fig = plt.figure(figsize=(12, 5))
    avg_spec_db = 10.0 * np.log10(np.maximum(result["avg_spec"], 1e-12))
    plt.plot(result["freqs_full"], avg_spec_db, linewidth=1.8, label="strict snapshot average")
    plt.xlim(0, min(DISPLAY_SPEC_MAX_HZ, result["fs"] / 2.0))

    for pf, pp, hn in result["harmonic_info"]:
        y = 10.0 * np.log10(max(pp, 1e-12))
        plt.axvline(pf, linestyle="--", alpha=0.4)
        plt.text(pf, y, f"H{hn}", fontsize=8)

    if np.isfinite(result["fundamental_estimate"]):
        plt.axvline(result["fundamental_estimate"], linestyle=":", linewidth=2.0,
                    label=f"f0≈{result['fundamental_estimate']:.1f} Hz")

    plt.axvspan(MAIN_BAND[0], MAIN_BAND[1], alpha=0.08, label="main band")
    plt.title(f'{result["name"]} - Average Spectrum')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_or_show(fig, os.path.join(out_dir, f"{result['name']}_average_spectrum.png"))

def plot_snapshot_energy(result, out_dir):
    df = result["snapshot_df"]
    fig = plt.figure(figsize=(12, 5))
    plt.plot(df["time"], df["selection_band_energy"], linewidth=1.5, label="selection energy")
    soft = df["keep_soft"].values.astype(bool)
    strict = df["keep_strict"].values.astype(bool)

    plt.scatter(df["time"][soft], df["selection_band_energy"][soft], s=10, label="soft")
    plt.scatter(df["time"][strict], df["selection_band_energy"][strict], s=18, label="strict")

    plt.axhline(result["soft_energy_thr"], linestyle=":", linewidth=1.0, label=f"soft thr={result['soft_energy_thr']:.3e}")
    plt.axhline(result["strict_energy_thr"], linestyle="--", linewidth=1.2, label=f"strict thr={result['strict_energy_thr']:.3e}")
    plt.axvline(IGNORE_FIRST_SEC, linestyle="--", label=f"ignore first {IGNORE_FIRST_SEC}s")
    plt.title(f'{result["name"]} - Snapshot Energy')
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    save_or_show(fig, os.path.join(out_dir, f"{result['name']}_snapshot_energy.png"))

def plot_per_second_doa(result, sec_df, out_dir):
    fig = plt.figure(figsize=(12, 5))
    plt.plot(sec_df["second_index"], sec_df["doa_soft_median"], marker="o", linewidth=1.2, label="soft median")
    plt.plot(sec_df["second_index"], sec_df["doa_strict_median"], marker="o", linewidth=1.8, label="strict median")
    if np.isfinite(result["doa_overall_multi"]):
        plt.axhline(result["doa_overall_multi"], linestyle=":", linewidth=1.2, label="overall multi")
    plt.title(f'{result["name"]} - Per-Second DOA')
    plt.xlabel("Second")
    plt.ylabel("Azimuth (deg)")
    plt.ylim(-180, 180)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_or_show(fig, os.path.join(out_dir, f"{result['name']}_per_second_doa.png"))

def plot_doa_map_enhanced(result, sec_df, out_dir):
    df = result["snapshot_df"]
    angles_strict = df.loc[df["keep_strict"] == 1, "doa_multi_smooth"].to_numpy(dtype=float)
    conf_strict = df.loc[df["keep_strict"] == 1, "confidence"].to_numpy(dtype=float)
    angles_soft = df.loc[df["keep_soft"] == 1, "doa_multi_smooth"].to_numpy(dtype=float)

    bins_deg = np.arange(-180, 181, 10)
    centers_deg = (bins_deg[:-1] + bins_deg[1:]) / 2.0
    centers_rad = np.deg2rad(centers_deg)
    width_rad = np.deg2rad(np.diff(bins_deg))

    strict_counts, _ = np.histogram(angles_strict[np.isfinite(angles_strict)], bins=bins_deg)
    soft_counts, _ = np.histogram(angles_soft[np.isfinite(angles_soft)], bins=bins_deg)

    weighted_counts = np.zeros(len(centers_deg), dtype=float)
    if len(angles_strict) > 0:
        valid = np.isfinite(angles_strict)
        a = angles_strict[valid]
        w = conf_strict[valid] if len(conf_strict) == len(angles_strict) else np.ones(np.sum(valid), dtype=float)
        w = np.where(np.isfinite(w), np.maximum(w, 0), 0)
        idx = np.digitize(a, bins_deg) - 1
        for i, wi in zip(idx, w):
            if 0 <= i < len(weighted_counts):
                weighted_counts[i] += wi

    max_count = max(np.max(soft_counts) if len(soft_counts) else 0, np.max(strict_counts) if len(strict_counts) else 0, 1)
    max_weight = max(np.max(weighted_counts) if len(weighted_counts) else 0, 1e-6)

    fig = plt.figure(figsize=(13, 7))
    ax1 = plt.subplot(1, 2, 1, projection='polar')
    ax1.bar(centers_rad, soft_counts / max_count, width=width_rad, bottom=0.0, alpha=0.25, label='soft density')
    ax1.bar(centers_rad, strict_counts / max_count, width=width_rad, bottom=0.0, alpha=0.65, label='strict density')
    if np.isfinite(result['doa_overall_multi']):
        ax1.plot([np.deg2rad(result['doa_overall_multi'])]*2, [0, 1.05], linewidth=2.0, label=f"overall={result['doa_overall_multi']:.1f}°")
    ax1.set_theta_zero_location('S')
    ax1.set_theta_direction(1)
    ax1.set_ylim(0, 1.05)
    ax1.set_title('Polar DOA occupancy')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.30, 1.15))

    ax2 = plt.subplot(1, 2, 2)
    ax2.bar(centers_deg, weighted_counts / max_weight, width=np.diff(bins_deg), alpha=0.75, align='center', label='confidence-weighted strict density')
    if np.isfinite(result['doa_overall_multi']):
        ax2.axvline(result['doa_overall_multi'], linestyle='--', linewidth=1.7, label='overall multi')
    if np.isfinite(result['doa_overall_main']):
        ax2.axvline(result['doa_overall_main'], linestyle=':', linewidth=1.5, label='overall main')

    active_sec = sec_df[sec_df['strict_snapshot_count'] > 0]
    for _, row in active_sec.iterrows():
        if np.isfinite(row['doa_strict_median']):
            ax2.scatter(row['doa_strict_median'], 0.02, s=28)

    ax2.set_title('DOA map summary')
    ax2.set_xlabel('Azimuth (deg)')
    ax2.set_ylabel('Normalized angular density')
    ax2.set_xlim(-180, 180)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(f'{result["name"]} - Enhanced DOA Map', fontsize=14)
    plt.tight_layout()
    save_or_show(fig, os.path.join(out_dir, f"{result['name']}_doa_map_enhanced.png"))

def plot_detection_timeline(result, out_dir):
    df = result["snapshot_df"]
    t = df["time"].values
    level = df["detection_level"].values
    conf = df["confidence"].values

    fig = plt.figure(figsize=(12, 5))
    ax1 = plt.gca()
    ax1.step(t, level, where="mid", linewidth=2.0, label="detection level")
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(["none", "soft", "strict"])
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Detection state")
    ax1.grid(True, alpha=0.3)
    ax1.axvline(IGNORE_FIRST_SEC, linestyle="--", label=f"ignore first {IGNORE_FIRST_SEC}s")

    ax2 = ax1.twinx()
    ax2.plot(t, conf, linewidth=1.2, alpha=0.85, label="confidence")
    if np.isfinite(result["confidence_soft_thr"]):
        ax2.axhline(result["confidence_soft_thr"], linestyle=":", linewidth=1.0, label="soft conf ref")
    if np.isfinite(result["confidence_strict_thr"]):
        ax2.axhline(result["confidence_strict_thr"], linestyle="--", linewidth=1.0, label="strict conf ref")
    ax2.set_ylabel("Confidence")
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.title(f'{result["name"]} - Detection Timeline')
    plt.tight_layout()
    save_or_show(fig, os.path.join(out_dir, f"{result['name']}_detection_timeline.png"))

def plot_confidence_diagnostics(result, out_dir):
    df = result["snapshot_df"]
    fig = plt.figure(figsize=(12, 5))
    plt.plot(df["time"], df["confidence"], linewidth=1.5, label="confidence")
    plt.plot(df["time"], normalize_01(df["peak_ratio_main"].values), linewidth=1.0, alpha=0.8, label="norm peak ratio main")
    plt.plot(df["time"], normalize_01(df["peak_ratio_multi"].values), linewidth=1.0, alpha=0.8, label="norm peak ratio multi")
    flat_reliability = 1.0 - np.clip(df["spectral_flatness"].values, 0.0, 1.0)
    plt.plot(df["time"], flat_reliability, linewidth=1.0, alpha=0.8, label="1 - spectral flatness")
    if np.isfinite(result["confidence_soft_thr"]):
        plt.axhline(result["confidence_soft_thr"], linestyle=":", linewidth=1.0, label="soft conf ref")
    if np.isfinite(result["confidence_strict_thr"]):
        plt.axhline(result["confidence_strict_thr"], linestyle="--", linewidth=1.0, label="strict conf ref")
    plt.title(f'{result["name"]} - Reliability / Confidence Diagnostics')
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized value")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    save_or_show(fig, os.path.join(out_dir, f"{result['name']}_confidence_diagnostics.png"))

def plot_algorithm_flow_summary(result, out_dir):
    fig = plt.figure(figsize=(14, 8))
    ax = plt.gca()
    ax.axis("off")

    boxes = [
        (0.04, 0.78, 0.18, 0.12, "1) Input\nMultichannel WAV\nMic A-B-C-D selection"),
        (0.28, 0.78, 0.18, 0.12, "2) Preprocess\nBand-pass\nwindowing / snapshots"),
        (0.52, 0.78, 0.18, 0.12, "3) Feature Layer\nEnergy\npeak ratio\nflatness / SNR"),
        (0.76, 0.78, 0.18, 0.12, "4) SRP-PHAT\nMain band DOA\nMulti-band DOA"),
        (0.16, 0.50, 0.22, 0.14, f"5) Selection\nsoft={int(result['snapshot_df']['keep_soft'].sum())} snapshot\nstrict={int(result['snapshot_df']['keep_strict'].sum())} snapshot"),
        (0.42, 0.50, 0.22, 0.14, f"6) Stabilization\ntracking={USE_TRACKING}\nsmoothing={USE_SMOOTHING}\nmode={MODE}"),
        (0.68, 0.50, 0.22, 0.14, f"7) Final Outputs\nOverall DOA={result['doa_overall_multi']:.1f}°\nf0≈{result['fundamental_estimate']:.1f} Hz" if np.isfinite(result['fundamental_estimate']) and np.isfinite(result['doa_overall_multi']) else "7) Final Outputs\nDOA / frequency summary"),
        (0.28, 0.20, 0.18, 0.12, "8) What to explain in presentation\nHow snapshots are kept\nHow angle is chosen"),
        (0.54, 0.20, 0.18, 0.12, "9) Key visuals\nDOA-time\nheatmap\nspectrum\ndetection timeline"),
    ]

    for x, y, w, h, txt in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.6)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=11)

    arrows = [
        ((0.22, 0.84), (0.28, 0.84)),
        ((0.46, 0.84), (0.52, 0.84)),
        ((0.70, 0.84), (0.76, 0.84)),
        ((0.85, 0.78), (0.79, 0.64)),
        ((0.62, 0.78), (0.53, 0.64)),
        ((0.37, 0.78), (0.27, 0.64)),
        ((0.38, 0.57), (0.42, 0.57)),
        ((0.64, 0.57), (0.68, 0.57)),
        ((0.49, 0.50), (0.37, 0.32)),
        ((0.75, 0.50), (0.63, 0.32)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", linewidth=1.5))

    plt.title(f'{result["name"]} - Algorithm Flow Summary', fontsize=14)
    plt.tight_layout()
    save_or_show(fig, os.path.join(out_dir, f"{result['name']}_algorithm_flow_summary.png"))

def export_presentation_notes(result, sec_df, out_dir):
    if not EXPORT_PRESENTATION_NOTES:
        return None

    df = result["snapshot_df"]
    strict_count = int(df["keep_strict"].sum())
    soft_count = int(df["keep_soft"].sum())
    mean_conf = safe_nanmean(df["confidence"].values)
    median_dom = float(np.nanmedian(df.loc[df["keep_strict"] == 1, "dominant_freq"].values)) if strict_count > 0 else np.nan
    mean_ang_vel = float(np.nanmean(np.abs(df.loc[df["keep_strict"] == 1, "angular_velocity_deg_s"].values))) if strict_count > 0 else np.nan
    stable_sec = int(np.sum(sec_df["strict_snapshot_count"].values > 0))

    path = os.path.join(out_dir, "presentation_notes.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("BRIEF NOTES FOR AN ACADEMIC PRESENTATION\n")
        f.write("=" * 48 + "\n\n")
        f.write("1. What does the algorithm do?\n")
        f.write("- Divides a multichannel UAV recording into snapshots.\n")
        f.write("- Estimates energy, spectral features, and direction for each snapshot.\n")
        f.write("- Uses SRP-PHAT for primary-band and multiband direction estimates.\n")
        f.write("- Classifies high-energy, high-confidence snapshots as soft or strict.\n")
        f.write("- Produces a stable DOA track and frequency summary.\n\n")

        f.write("2. Which plots should be shown first?\n")
        f.write("- Snapshot Energy: Which time intervals were selected?\n")
        f.write("- DOA vs Time: How did the UAV direction change over time?\n")
        f.write("- DOA Heatmap: Why were these angles selected?\n")
        f.write("- Average Spectrum: What is the UAV acoustic signature?\n")
        f.write("- Detection Timeline: When did soft or strict detections occur?\n")
        f.write("- Algorithm Flow Summary: Block view of the complete process.\n\n")

        f.write("3. Automatic summary for this recording\n")
        f.write(f"- Overall DOA main  : {result['doa_overall_main']:.2f} deg\n")
        f.write(f"- Overall DOA multi : {result['doa_overall_multi']}\n")
        f.write(f"- Strict snapshot   : {strict_count}\n")
        f.write(f"- Soft snapshot     : {soft_count}\n")
        f.write(f"- Mean confidence   : {mean_conf:.3f}\n")
        f.write(f"- Median dom freq   : {median_dom:.2f} Hz\n")
        f.write(f"- Mean ang velocity : {mean_ang_vel:.2f} deg/s\n")
        f.write(f"- Active seconds    : {stable_sec}\n\n")

        f.write("4. Key statements for the review panel\n")
        f.write("- The algorithm evaluates individual snapshots and retains reliable segments instead of making one immediate decision.\n")
        f.write("- The multiband smoothed track is therefore preferred over the raw DOA.\n")
        f.write("- The heatmap shows the beamforming energy behind the selected angle.\n")
        f.write("- The spectrum and dominant frequencies show that detection depends on both direction and acoustic signature.\n")
        f.write("- The detection timeline translates technical metrics into application-level events.\n\n")

        f.write("5. Suggested slide sequence\n")
        f.write("- Problem definition\n")
        f.write("- Microphone array and data structure\n")
        f.write("- Algorithm flowchart\n")
        f.write("- Snapshot selection\n")
        f.write("- DOA output\n")
        f.write("- Frequency analysis\n")
        f.write("- Detection timeline\n")
        f.write("- Results and limitations\n")

    return path

# Program entry point
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyze multichannel UAV audio and estimate direction of arrival."
    )
    parser.add_argument(
        "--input",
        default=INPUT_PATH,
        help="Path to the multichannel WAV recording.",
    )
    parser.add_argument(
        "--noise-reference",
        default=OPTIONAL_NOISE_PATH,
        help="Optional WAV file used to estimate the noise floor.",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_ROOT,
        help="Directory where analysis artifacts will be written.",
    )
    parser.add_argument(
        "--mode",
        choices=("hover", "circle"),
        default=MODE,
        help="Parameter profile used for direction tracking.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Generate plots without opening interactive windows.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    global INPUT_PATH, OPTIONAL_NOISE_PATH, OUTPUT_ROOT, SHOW_PLOTS

    args = parse_args(argv)
    INPUT_PATH = os.path.abspath(args.input)
    OPTIONAL_NOISE_PATH = os.path.abspath(args.noise_reference)
    OUTPUT_ROOT = os.path.abspath(args.output)
    SHOW_PLOTS = not args.no_show
    configure_mode(args.mode)

    ensure_dir(OUTPUT_ROOT)

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    noise_floor = compute_noise_floor_from_optional_file(OPTIONAL_NOISE_PATH, SELECTION_BAND)
    if noise_floor is not None:
        print("\n=== Noise reference summary ===")
        print(f"sel_noise_p50 = {noise_floor['sel_noise_p50']:.3e}")
        print(f"sel_noise_p95 = {noise_floor['sel_noise_p95']:.3e}")

    print(f"\nAnalyzing: {INPUT_PATH}")
    print(f"Mode: {MODE}")

    rec_dir = os.path.join(OUTPUT_ROOT, os.path.splitext(os.path.basename(INPUT_PATH))[0])
    ensure_dir(rec_dir)

    result = analyze_recording(INPUT_PATH, noise_floor=noise_floor)
    sec_df = per_second_summary(result["snapshot_df"])

    snapshot_csv = os.path.join(rec_dir, "snapshot_summary.csv")
    per_second_csv = os.path.join(rec_dir, "per_second_summary.csv")

    result["snapshot_df"].to_csv(snapshot_csv, index=False)
    sec_df.to_csv(per_second_csv, index=False)

    harm_txt = os.path.join(rec_dir, "harmonic_summary.txt")
    with open(harm_txt, "w", encoding="utf-8") as f:
        f.write(f"File: {result['name']}\n")
        f.write(f"Mode: {MODE}\n")
        f.write(f"Overall DOA main: {result['doa_overall_main']:.2f} deg\n")
        f.write(f"Overall DOA multi: {result['doa_overall_multi']}\n")
        f.write(f"Estimated fundamental: {result['fundamental_estimate']}\n")
        f.write("\nDetected harmonics:\n")
        if len(result["harmonic_info"]) == 0:
            f.write("None\n")
        else:
            for pf, pp, hn in result["harmonic_info"]:
                f.write(f"H{hn}: {pf:.2f} Hz, power={pp:.4e}\n")

    plot_snapshot_energy(result, rec_dir)
    plot_doa_vs_time(result, rec_dir)
    plot_doa_heatmap(result, rec_dir)
    plot_doa_map_enhanced(result, sec_df, rec_dir)
    plot_beam_power_scan(result, rec_dir)
    plot_spectrogram(result, rec_dir)
    plot_average_spectrum(result, rec_dir)
    plot_per_second_doa(result, sec_df, rec_dir)
    plot_detection_timeline(result, rec_dir)
    plot_confidence_diagnostics(result, rec_dir)
    plot_algorithm_flow_summary(result, rec_dir)

    notes_path = export_presentation_notes(result, sec_df, rec_dir)

    df = result["snapshot_df"]
    strict_count = int(df["keep_strict"].sum())
    soft_count = int(df["keep_soft"].sum())
    total = len(df)

    mean_conf = float(np.nanmean(df["confidence"].values))
    median_dom = float(
        np.nanmedian(
            df.loc[df["keep_strict"] == 1, "dominant_freq"].values
        )
    ) if strict_count > 0 else np.nan

    mean_ang_vel = float(
        np.nanmean(
            np.abs(df.loc[df["keep_strict"] == 1, "angular_velocity_deg_s"].values)
        )
    ) if strict_count > 0 else np.nan

    print("\n=== ANALYSIS SUMMARY ===")
    print(f"Overall DOA main        : {result['doa_overall_main']:.2f} deg")
    print(f"Overall DOA multi       : {result['doa_overall_multi']}")
    print(f"Strict snapshots        : {strict_count}/{total}")
    print(f"Soft snapshots          : {soft_count}/{total}")
    print(f"Median dominant freq    : {median_dom:.2f} Hz")
    print(f"Mean confidence         : {mean_conf:.3f}")
    print(f"Mean angular velocity   : {mean_ang_vel:.2f} deg/s")
    print(f"Snapshot CSV            : {snapshot_csv}")
    print(f"Per-second CSV          : {per_second_csv}")
    print(f"Harmonic summary        : {harm_txt}")
    if notes_path is not None:
        print(f"Presentation notes      : {notes_path}")
    print(f"Output folder           : {rec_dir}")

    if SHOW_PLOTS:
        plt.show()

if __name__ == "__main__":
    main()
