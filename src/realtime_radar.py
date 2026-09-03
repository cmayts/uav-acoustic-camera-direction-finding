
# -*- coding: utf-8 -*-
"""
Drone Acoustic Radar UI - RealTime Fix v11
- Real-time WAV replay + live ReSpeaker input
- Fast acquire + stable track continuity
- No initial 4-second ignore

Notes:
- UI remains responsive by separating:
  1) fast preview updates (small hop, lightweight)
  2) confirmed offline-style window analysis (delayed, more reliable)
- Designed to work with either WAV replay or live microphone input
"""

import os
import sys
import math
import time
import queue
import threading
from collections import deque

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, get_window, stft, medfilt

from PyQt5.QtCore import Qt, QTimer, QPointF, QRectF
from PyQt5.QtGui import QColor, QPainter, QPen, QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout,
    QGridLayout, QFrame, QListWidget, QListWidgetItem, QSplitter, QPushButton,
    QComboBox, QLineEdit, QFileDialog
)

import pyqtgraph as pg


# =========================================================
# DEFAULTS
# =========================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SOURCE_MODE = "wav"   # wav / live
DEFAULT_INPUT_WAV_PATH = os.path.join(PROJECT_ROOT, "data", "input.wav")
DEFAULT_NOISE_WAV_PATH = os.path.join(PROJECT_ROOT, "data", "analiz.wav")
DEFAULT_DEVICE = None

FS = 16000
CHANNELS_TOTAL = 6
CHANNELS_TO_USE = [1, 2, 3, 4]

MODE = "circle"  # hover / circle
WINDOW_TYPE = "hann"
ANGLES = np.linspace(-180, 180, 361)
C_SOUND = 343.0

# preview branch
PREVIEW_WINDOW_SEC = 0.35
PREVIEW_HOP_SEC = 0.05

# confirmed branch
CONFIRM_WINDOW_SEC = 1.60
CONFIRM_DELAY_SEC = 0.10
CONFIRM_STEP_SEC = 0.10

DISPLAY_HISTORY_SEC = 25.0
RADAR_TRAIL_SECONDS = 8.0
UI_POLL_MS = 35
PLOT_REFRESH_MS = 70

INPUT_WAV_REPLAY_SPEED = 1.0

DISPLAY_SPEC_MAX_HZ = 2500.0

if MODE == "hover":
    SNAPSHOT_SEC = 0.10
    HOP_SEC = 0.05
    MAIN_BAND = (400.0, 1200.0)
    SELECTION_BAND = (500.0, 1500.0)
    MULTI_BANDS = [(400.0, 800.0), (800.0, 1400.0), (1400.0, 2200.0)]
    TRACK_LAMBDA = 0.003
    PHAT_ALPHA = 0.6
    SMOOTH_KERNEL = 7
    STRICT_TOP_ENERGY_PERCENT = 35
    SOFT_TOP_ENERGY_PERCENT = 60
    MIN_RUN_LENGTH_STRICT = 2
    MIN_RUN_LENGTH_SOFT = 1
    MAX_FLATNESS_STRICT = 0.75
    MAX_FLATNESS_SOFT = 0.90
else:
    SNAPSHOT_SEC = 0.05
    HOP_SEC = 0.025
    MAIN_BAND = (400.0, 1200.0)
    SELECTION_BAND = (500.0, 1700.0)
    MULTI_BANDS = [(400.0, 800.0), (800.0, 1400.0), (1400.0, 2200.0)]
    TRACK_LAMBDA = 0.0005
    PHAT_ALPHA = 0.5
    SMOOTH_KERNEL = 3
    STRICT_TOP_ENERGY_PERCENT = 45
    SOFT_TOP_ENERGY_PERCENT = 70
    MIN_RUN_LENGTH_STRICT = 1
    MIN_RUN_LENGTH_SOFT = 1
    MAX_FLATNESS_STRICT = 0.90
    MAX_FLATNESS_SOFT = 0.98

PREVIEW_CONF_THRESHOLD = 0.12
TRACK_CONF_THRESHOLD = 0.22
DETECT_CONF_THRESHOLD = 0.11

# Evidence / no-target gate (kept intentionally soft to avoid missing real targets)
MIN_STRICT_COUNT = 1
MIN_STRICT_RATIO = 0.05
MIN_PEAK_RATIO_MULTI = 1.015
MAX_MEDIAN_FLATNESS = 0.985
MIN_SELECTION_SNR_DB = -8.0

# Persistence memory
PERSIST_HISTORY_LEN = 8
PERSIST_MIN_POSITIVE = 2
PERSIST_TRACK_MIN_POSITIVE = 2
PERSIST_DECAY_KEEP_SEC = 3.00

# display smoothing
RADAR_DISPLAY_ALPHA = 0.45
TIMELINE_HIDE_WHEN_NO_TARGET = True

# motion plausibility filter
MAX_PREVIEW_ANGULAR_SPEED_DEG_S = 240.0
MAX_CONFIRMED_ANGULAR_SPEED_DEG_S = 360.0
PREVIEW_HOLD_SEC = 0.35
CONFIRMED_HOLD_SEC = 1.40
JITTER_STD_THRESHOLD_DEG = 95.0
JITTER_MEAN_STEP_THRESHOLD_DEG = 100.0
STABLE_STD_THRESHOLD_DEG = 45.0
STABLE_MEAN_STEP_THRESHOLD_DEG = 50.0

# Temporal angular clustering stabilizer
CLUSTER_WINDOW_SEC = 1.00
CLUSTER_BIN_DEG = 10.0
CLUSTER_RADIUS_DEG = 38.0
CLUSTER_MIN_VOTES_DETECTED = 1.10
CLUSTER_MIN_VOTES_TRACKING = 1.70
CLUSTER_MIN_DENSITY_DETECTED = 0.34
CLUSTER_MIN_DENSITY_TRACKING = 0.44
CLUSTER_HOLD_SEC = 2.80
CLUSTER_SMOOTH_ALPHA_ACQUIRE = 0.32
CLUSTER_SMOOTH_ALPHA_TRACK = 0.24
# While a drone is already locked, follow continuous motion instead of re-clustering every frame.
TRACK_CONTINUITY_RADIUS_DEG = 95.0
TRACK_CONTINUITY_ALPHA = 0.36
TRACK_CONTINUITY_MIN_GATE = 0.45

# Field alarm mode: target must pass these before it is shown as a real drone.
FIELD_SUSPECT_SCORE = 0.52
FIELD_CONFIRM_SCORE = 0.57
FIELD_ALARM_SCORE = 0.64
FIELD_CONFIRM_SEC = 0.45
FIELD_ALARM_SEC = 0.70
FIELD_LOST_GRACE_SEC = 2.80
ALARM_BEEP_INTERVAL_SEC = 0.75

r = 0.032
d = r / np.sqrt(2.0)
MIC_POS = np.array([
    [-d, -d],   # A: lower-left
    [+d, -d],   # B: lower-right
    [+d, +d],   # C: upper-right
    [-d, +d],   # D: upper-left
], dtype=float)


# =========================================================
# HELPERS
# =========================================================
def wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


def ang_dist_deg(a, b):
    return np.abs(wrap180(a - b))


def normalize_01(x):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    valid = np.isfinite(x)
    if not np.any(valid):
        return out
    xmin = np.nanmin(x[valid])
    xmax = np.nanmax(x[valid])
    if xmax - xmin < 1e-12:
        out[valid] = 0.5
        return out
    out[valid] = (x[valid] - xmin) / (xmax - xmin)
    return out


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


def smooth_angle_deg(a_deg, kernel_size=7):
    a = np.asarray(a_deg, dtype=float).copy()
    valid = np.isfinite(a)
    if np.sum(valid) < kernel_size or kernel_size < 3:
        return a
    rad = np.deg2rad(a[valid])
    rad_unwrapped = np.unwrap(rad)
    rad_smoothed = medfilt(rad_unwrapped, kernel_size=kernel_size)
    a[valid] = wrap180(np.rad2deg(rad_smoothed))
    return a


def bandpass_filter(X, fs, f_lo, f_hi, order=4):
    nyq = fs / 2.0
    f_hi = min(f_hi, nyq * 0.999)
    if not (0 < f_lo < f_hi < nyq):
        raise ValueError(f"Invalid bandpass: {f_lo}-{f_hi}, Nyquist={nyq}")
    b, a = butter(order, [f_lo / nyq, f_hi / nyq], btype="band")
    return filtfilt(b, a, X, axis=0)


def normalize_audio_array(x):
    orig_dtype = x.dtype
    x = x.astype(np.float32)
    if np.issubdtype(orig_dtype, np.integer):
        max_val = max(abs(np.iinfo(orig_dtype).min), np.iinfo(orig_dtype).max)
        x /= float(max_val)
    return x


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


def steering_srp_phat(mic_xy, freqs, angles_deg, c=343.0):
    A = len(angles_deg)
    M = mic_xy.shape[0]
    F = len(freqs)
    S = np.empty((A, M, F), dtype=np.complex64)
    for ai, ang in enumerate(angles_deg):
        th = np.deg2rad(ang)
        # 0° down, 90° right, 180° up, and -90°/270° left.
        u = np.array([np.sin(th), -np.cos(th)], dtype=float)
        tau = (mic_xy @ u) / c
        S[ai] = np.exp(-1j * 2.0 * np.pi * freqs[None, :] * tau[:, None]).astype(np.complex64)
    return S


def enforce_min_run(mask, min_run_len):
    if min_run_len <= 1:
        return mask.copy()
    out = mask.copy()
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif (not v) and (start is not None):
            if (i - start) < min_run_len:
                out[start:i] = False
            start = None
    if start is not None and (len(mask) - start) < min_run_len:
        out[start:] = False
    return out


def robust_angle_motion_metrics(times, angles_deg):
    times = np.asarray(times, dtype=float)
    angles_deg = np.asarray(angles_deg, dtype=float)
    valid = np.isfinite(times) & np.isfinite(angles_deg)
    times = times[valid]
    angles_deg = angles_deg[valid]
    if len(times) < 3:
        return {
            "step_mean": np.nan,
            "step_std": np.nan,
            "vel_mean_abs": np.nan,
            "vel_max_abs": np.nan,
        }

    steps = np.abs(wrap180(np.diff(angles_deg)))
    dts = np.diff(times)
    good = dts > 1e-6
    if not np.any(good):
        return {
            "step_mean": np.nan,
            "step_std": np.nan,
            "vel_mean_abs": np.nan,
            "vel_max_abs": np.nan,
        }

    steps = steps[good]
    dts = dts[good]
    vels = steps / dts
    return {
        "step_mean": float(np.nanmean(steps)),
        "step_std": float(np.nanstd(steps)),
        "vel_mean_abs": float(np.nanmean(np.abs(vels))),
        "vel_max_abs": float(np.nanmax(np.abs(vels))),
    }


def bounded_angle_update(prev_angle, candidate_angle, dt, max_speed_deg_s):
    if prev_angle is None or (not np.isfinite(prev_angle)) or (not np.isfinite(candidate_angle)):
        return candidate_angle, True
    if dt <= 1e-6:
        return prev_angle, False
    delta = wrap180(candidate_angle - prev_angle)
    max_step = max_speed_deg_s * dt
    if np.abs(delta) <= max_step:
        return candidate_angle, True
    return prev_angle, False

def smooth_display_angle(prev_angle, target_angle, alpha=0.22):
    if prev_angle is None or (not np.isfinite(prev_angle)):
        return target_angle
    if not np.isfinite(target_angle):
        return np.nan
    delta = wrap180(target_angle - prev_angle)
    return wrap180(prev_angle + alpha * delta)


def select_mics(x, chs):
    if x.shape[1] < 6:
        raise ValueError(f"Need >=6 channels, got {x.shape[1]}")
    return x[:, chs]


def read_wav_multichannel(path):
    fs, x = wavfile.read(path)
    if x.ndim == 1:
        raise ValueError("Single-channel WAV not supported")
    return fs, normalize_audio_array(x)


def compute_noise_floor_from_optional_file(noise_path):
    if not noise_path or not os.path.exists(noise_path):
        return None
    fs_n, x_n = read_wav_multichannel(noise_path)
    if fs_n != FS:
        raise ValueError(f"Noise WAV fs={fs_n}, expected {FS}")
    X_n = select_mics(x_n, CHANNELS_TO_USE)
    X_sel = bandpass_filter(X_n, fs_n, SELECTION_BAND[0], SELECTION_BAND[1], order=4)

    L = int(SNAPSHOT_SEC * fs_n)
    H = int(HOP_SEC * fs_n)
    if X_n.shape[0] < L:
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


def compute_band_srp(frame_bp, fs, band_lo, band_hi, steer_cache, use_tracking_prev=None):
    L = frame_bp.shape[0]
    freqs_full = np.fft.rfftfreq(L, d=1.0 / fs)
    band_idx = np.where((freqs_full >= band_lo) & (freqs_full <= min(band_hi, fs / 2.0)))[0]
    if len(band_idx) < 2:
        return None
    band_key = (L, fs, band_lo, band_hi)
    if band_key not in steer_cache:
        band_freqs = freqs_full[band_idx].astype(np.float32)
        steer_cache[band_key] = (band_idx, steering_srp_phat(MIC_POS, band_freqs, ANGLES, c=C_SOUND))
    band_idx, steer = steer_cache[band_key]
    Xf_full = np.fft.rfft(frame_bp, axis=0)
    Xf_band = Xf_full[band_idx, :].T
    Xf_use = Xf_band / (np.abs(Xf_band) ** PHAT_ALPHA + 1e-12)
    Y = np.mean(np.conj(steer) * Xf_use[None, :, :], axis=1)
    P = np.mean(np.abs(Y) ** 2, axis=1)
    if use_tracking_prev is not None:
        penalty = TRACK_LAMBDA * (ang_dist_deg(ANGLES, use_tracking_prev) ** 2)
        i1 = int(np.argmax(P - penalty))
    else:
        i1 = int(np.argmax(P))
    p1 = float(P[i1])
    P2 = P.copy()
    P2[i1] = -np.inf
    p2 = float(np.max(P2))
    return {
        "P": P.astype(np.float32),
        "doa": float(ANGLES[i1]),
        "peak_ratio": p1 / (p2 + 1e-12),
        "peak_power": p1,
    }


def analyze_array_offline_style(X_raw_full, fs, noise_floor=None):
    """
    Offline V1.12-style analysis on a short rolling window.
    Accepts full multichannel input and internally selects A/B/C/D channels.
    Returns a compact dict for UI.
    """
    if X_raw_full.shape[0] < int(SNAPSHOT_SEC * fs) * 2:
        return None

    L = int(SNAPSHOT_SEC * fs)
    H = int(HOP_SEC * fs)
    if X_raw_full.shape[0] < L:
        return None

    # Match offline V1.12 behavior: use only selected microphone channels.
    X_raw = select_mics(X_raw_full, CHANNELS_TO_USE).astype(np.float32)

    X_main = bandpass_filter(X_raw, fs, MAIN_BAND[0], MAIN_BAND[1], order=4)
    X_sel = bandpass_filter(X_raw, fs, SELECTION_BAND[0], SELECTION_BAND[1], order=4)
    X_multi = [bandpass_filter(X_raw, fs, b_lo, b_hi, order=4) for b_lo, b_hi in MULTI_BANDS]

    win = get_window(WINDOW_TYPE, L).astype(np.float32)
    K = 1 + (X_raw.shape[0] - L) // H
    times = (np.arange(K) * H + L / 2.0) / fs
    steer_cache = {}

    Pscan_main = np.zeros((K, len(ANGLES)), dtype=np.float32)
    doa_main_raw = np.full(K, np.nan, dtype=np.float32)
    doa_multi_raw = np.full(K, np.nan, dtype=np.float32)
    peak_ratio_main = np.full(K, np.nan, dtype=np.float32)
    peak_ratio_multi = np.full(K, np.nan, dtype=np.float32)
    peak_power_multi = np.full(K, np.nan, dtype=np.float32)
    selection_energy = np.zeros(K, dtype=np.float32)
    main_energy = np.zeros(K, dtype=np.float32)
    spectral_flatness = np.full(K, np.nan, dtype=np.float32)
    dominant_freq = np.full(K, np.nan, dtype=np.float32)
    estimated_snr_db = np.full(K, np.nan, dtype=np.float32)

    prev_angle_main = None
    prev_angle_multi = None
    sel_noise_ref = None if noise_floor is None else noise_floor.get("sel_noise_p95", None)
    freqs_full = np.fft.rfftfreq(L, d=1.0 / fs)

    for k in range(K):
        s = k * H
        e = s + L
        frame_main = X_main[s:e, :] * win[:, None]
        frame_sel = X_sel[s:e, :] * win[:, None]
        selection_energy[k] = float(np.mean(frame_sel ** 2))
        main_energy[k] = float(np.mean(frame_main ** 2))
        estimated_snr_db[k] = estimate_snr_db(selection_energy[k], sel_noise_ref)

        Xf_full = np.fft.rfft(frame_main, axis=0)
        avg_power = np.mean(np.abs(Xf_full) ** 2, axis=1)
        df, _, _, flat = spectral_metrics_from_power(freqs_full, avg_power)
        dominant_freq[k] = df
        spectral_flatness[k] = flat

        res_main = compute_band_srp(frame_main, fs, MAIN_BAND[0], MAIN_BAND[1], steer_cache, prev_angle_main)
        if res_main is not None:
            peak_ratio_main[k] = res_main["peak_ratio"]
            Pscan_main[k, :] = res_main["P"]
            doa_main_raw[k] = res_main["doa"]
            prev_angle_main = doa_main_raw[k]

        band_doas = []
        band_weights = []
        band_ratios = []
        band_powers = []
        for (b_lo, b_hi), Xb in zip(MULTI_BANDS, X_multi):
            frame_b = Xb[s:e, :] * win[:, None]
            res_b = compute_band_srp(frame_b, fs, b_lo, b_hi, steer_cache, prev_angle_multi)
            if res_b is None:
                continue
            band_doas.append(res_b["doa"])
            w = max(res_b["peak_power"], 1e-12) * max(res_b["peak_ratio"], 1e-12)
            band_weights.append(w)
            band_ratios.append(res_b["peak_ratio"])
            band_powers.append(res_b["peak_power"])

        if band_doas:
            fused = weighted_circular_mean_deg(band_doas, band_weights)
            peak_ratio_multi[k] = float(np.mean(band_ratios))
            peak_power_multi[k] = float(np.mean(band_powers))
            if prev_angle_multi is not None:
                cand = np.array(band_doas + [fused], dtype=float)
                cw = np.array(band_weights + [np.sum(band_weights)], dtype=float)
                penalty = np.abs(wrap180(cand - prev_angle_multi))
                score = cw / (1.0 + 0.05 * penalty)
                doa_multi_raw[k] = cand[int(np.argmax(score))]
            else:
                doa_multi_raw[k] = fused
            prev_angle_multi = doa_multi_raw[k]

    valid_time_mask = np.ones_like(times, dtype=bool)

    thr_strict = np.percentile(selection_energy[valid_time_mask], 100 - STRICT_TOP_ENERGY_PERCENT)
    keep_strict = (selection_energy >= thr_strict) & valid_time_mask
    keep_strict &= (spectral_flatness <= MAX_FLATNESS_STRICT)
    keep_strict = enforce_min_run(keep_strict, MIN_RUN_LENGTH_STRICT)

    if np.sum(keep_strict) < 3:
        fallback_thr = np.percentile(selection_energy[valid_time_mask], 80)
        keep_strict = (selection_energy >= fallback_thr) & valid_time_mask
        keep_strict = enforce_min_run(keep_strict, 1)

    if np.sum(keep_strict) == 0:
        idx = int(np.argmax(selection_energy))
        keep_strict[idx] = True

    thr_soft = np.percentile(selection_energy[valid_time_mask], 100 - SOFT_TOP_ENERGY_PERCENT)
    keep_soft = (selection_energy >= thr_soft) & valid_time_mask
    keep_soft &= (spectral_flatness <= MAX_FLATNESS_SOFT)
    keep_soft = enforce_min_run(keep_soft, MIN_RUN_LENGTH_SOFT)

    if np.sum(keep_soft) < np.sum(keep_strict):
        keep_soft = keep_strict.copy()

    doa_multi_smooth = smooth_angle_deg(doa_multi_raw, kernel_size=SMOOTH_KERNEL)

    mean_scan_strict = np.nanmean(np.where(keep_strict[:, None], Pscan_main, np.nan), axis=0)
    if not np.any(np.isfinite(mean_scan_strict)):
        doa_overall_main = np.nan
    else:
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
        confidence[k] = float(np.mean(vals)) if vals else np.nan

    final_idx = np.where(keep_strict)[0]
    if len(final_idx) == 0:
        final_idx = np.where(keep_soft)[0]
    if len(final_idx) == 0:
        final_idx = np.arange(K)

    final_conf = float(np.nanmean(confidence[final_idx])) if len(final_idx) else np.nan
    final_dom = float(np.nanmedian(dominant_freq[final_idx])) if len(final_idx) else np.nan
    final_snr = float(np.nanmean(estimated_snr_db[final_idx])) if len(final_idx) else np.nan
    final_pr = float(np.nanmean(peak_ratio_multi[final_idx])) if len(final_idx) else np.nan
    final_flat = float(np.nanmedian(spectral_flatness[final_idx])) if len(final_idx) else np.nan
    strict_ratio = float(np.sum(keep_strict) / max(np.sum(keep_soft), 1))

    motion_metrics = robust_angle_motion_metrics(times[final_idx] if len(final_idx) else times,
                                                 doa_multi_smooth[final_idx] if len(final_idx) else doa_multi_smooth)
    jittery = (
        np.isfinite(motion_metrics["step_std"]) and motion_metrics["step_std"] > JITTER_STD_THRESHOLD_DEG
    ) or (
        np.isfinite(motion_metrics["step_mean"]) and motion_metrics["step_mean"] > JITTER_MEAN_STEP_THRESHOLD_DEG
    )
    stable_track = (
        np.isfinite(motion_metrics["step_std"]) and motion_metrics["step_std"] <= STABLE_STD_THRESHOLD_DEG
        and np.isfinite(motion_metrics["step_mean"]) and motion_metrics["step_mean"] <= STABLE_MEAN_STEP_THRESHOLD_DEG
    )

    latest_angle_preview = float(doa_multi_raw[-1]) if np.isfinite(doa_multi_raw[-1]) else np.nan
    latest_conf_preview = float(confidence[-1]) if np.isfinite(confidence[-1]) else np.nan

    raw_mic = X_raw[:, 0].astype(np.float32)
    fsp, tsp, Z = stft(raw_mic, fs=fs, window="hann",
                       nperseg=max(256, int(0.032 * fs)),
                       noverlap=max(0, max(256, int(0.032 * fs)) - int(0.010 * fs)),
                       boundary=None, padded=False)
    Ssp = 20.0 * np.log10(np.maximum(np.abs(Z), 1e-10))

    # band energy summary
    band_energy = []
    for b_lo, b_hi in MULTI_BANDS:
        Xb = bandpass_filter(X_raw, fs, b_lo, b_hi, order=4)
        band_energy.append(float(np.mean(Xb ** 2)))

    return {
        "doa_confirmed": float(doa_overall_multi) if np.isfinite(doa_overall_multi) else np.nan,
        "doa_main": float(doa_overall_main) if np.isfinite(doa_overall_main) else np.nan,
        "confidence_confirmed": final_conf,
        "dominant_freq": final_dom,
        "snr_est": final_snr,
        "peak_ratio_multi": final_pr,
        "strict_count": int(np.sum(keep_strict)),
        "soft_count": int(np.sum(keep_soft)),
        "latest_preview_angle": latest_angle_preview,
        "latest_preview_conf": latest_conf_preview,
        "selection_energy": float(np.nanmean(selection_energy[final_idx])) if len(final_idx) else np.nan,
        "beam_scan": mean_scan_strict if np.any(np.isfinite(mean_scan_strict)) else None,
        "times": times,
        "doa_series": doa_multi_smooth,
        "keep_soft": keep_soft.astype(bool),
        "keep_strict": keep_strict.astype(bool),
        "confidence_series": confidence,
        "dominant_series": dominant_freq,
        "spectrogram_f": fsp,
        "spectrogram_t": tsp,
        "spectrogram_db": Ssp,
        "band_energy": band_energy,
        "raw_power_freqs": freqs_full,
        "raw_power_db": 10.0 * np.log10(np.maximum(np.mean(np.abs(np.fft.rfft(X_raw[:, 0] * get_window(WINDOW_TYPE, X_raw.shape[0]).astype(np.float32))) ** 2, axis=0) if X_raw.shape[0] > 0 else 1e-12, 1e-12)),
        "motion_step_mean": motion_metrics["step_mean"],
        "motion_step_std": motion_metrics["step_std"],
        "jittery": bool(jittery),
        "stable_track": bool(stable_track),
        "strict_ratio": strict_ratio,
        "median_flatness": final_flat,
    }


# =========================================================
# FAST PREVIEW
# =========================================================
class PreviewProcessor:
    def __init__(self, fs):
        self.fs = fs
        self.L = int(PREVIEW_WINDOW_SEC * fs)
        self.win = get_window(WINDOW_TYPE, self.L).astype(np.float32)
        self.prev_angle = None
        self.prev_output_angle = None
        self.prev_output_time = None
        self.last_valid_time = None
        self.steer_cache = {}

    def process(self, frame6, noise_floor=None, timestamp=None):
        if frame6.shape[0] < self.L:
            return None
        if timestamp is None:
            timestamp = 0.0
        X = frame6[-self.L:, CHANNELS_TO_USE].astype(np.float32)
        frame_main = bandpass_filter(X, self.fs, MAIN_BAND[0], MAIN_BAND[1], order=4) * self.win[:, None]
        res = compute_band_srp(frame_main, self.fs, MAIN_BAND[0], MAIN_BAND[1], self.steer_cache, self.prev_angle)
        if res is None:
            return None

        freqs_full = np.fft.rfftfreq(self.L, d=1.0 / self.fs)
        Xf_full = np.fft.rfft(frame_main, axis=0)
        avg_power = np.mean(np.abs(Xf_full) ** 2, axis=1)
        dom, _, _, flat = spectral_metrics_from_power(freqs_full, avg_power)

        X_sel = bandpass_filter(X, self.fs, SELECTION_BAND[0], SELECTION_BAND[1], order=4) * self.win[:, None]
        sel_energy = float(np.mean(X_sel ** 2))
        snr_est = estimate_snr_db(sel_energy, None if noise_floor is None else noise_floor.get("sel_noise_p95", None))
        pr = float(res["peak_ratio"])
        # lightweight preview confidence
        c_terms = [
            np.clip((pr - 1.0) / 0.25, 0.0, 1.0),
            1.0 - np.clip(flat, 0.0, 1.0),
        ]
        if np.isfinite(snr_est):
            c_terms.append(np.clip((snr_est + 5.0) / 15.0, 0.0, 1.0))
        conf = float(np.mean(c_terms))
        raw_angle = float(res["doa"])
        dt = PREVIEW_HOP_SEC if self.prev_output_time is None else max(1e-6, float(timestamp) - float(self.prev_output_time))
        filtered_angle, accepted = bounded_angle_update(
            self.prev_output_angle, raw_angle, dt, MAX_PREVIEW_ANGULAR_SPEED_DEG_S
        )
        if accepted:
            self.prev_output_angle = filtered_angle
            self.last_valid_time = float(timestamp)
        else:
            if self.last_valid_time is not None and (float(timestamp) - self.last_valid_time) > PREVIEW_HOLD_SEC:
                self.prev_output_angle = np.nan
                filtered_angle = np.nan
            else:
                filtered_angle = self.prev_output_angle if self.prev_output_angle is not None else np.nan

        self.prev_output_time = float(timestamp)
        self.prev_angle = raw_angle
        return {
            "angle": float(filtered_angle) if np.isfinite(filtered_angle) else np.nan,
            "raw_angle": raw_angle,
            "accepted": bool(accepted),
            "confidence": conf,
            "dominant_freq": float(dom) if np.isfinite(dom) else np.nan,
            "peak_ratio": pr,
            "flatness": float(flat) if np.isfinite(flat) else np.nan,
            "snr_est": float(snr_est) if np.isfinite(snr_est) else np.nan,
            "selection_energy": sel_energy,
        }


# =========================================================
# ENGINE
# =========================================================
class HybridEngine:
    def __init__(self, result_queue, event_callback=None):
        self.result_queue = result_queue
        self.event_callback = event_callback

        self.source_mode = DEFAULT_SOURCE_MODE
        self.input_wav_path = DEFAULT_INPUT_WAV_PATH
        self.noise_wav_path = DEFAULT_NOISE_WAV_PATH
        self.device = DEFAULT_DEVICE

        self.running = False
        self.stream = None
        self.wav_data = None
        self.wav_pos = 0
        self.io_thread = None
        self.analysis_thread = None

        self.sample_counter = 0
        self.last_preview_emit = 0
        self.last_confirm_emit = 0

        self.buffer_lock = threading.Lock()
        self.buffer = deque(maxlen=int(FS * (CONFIRM_WINDOW_SEC + CONFIRM_DELAY_SEC + 2.0)))
        self.latest_preview = None
        self.latest_confirmed = None

        self.preview_processor = PreviewProcessor(FS)
        self.noise_floor = None
        self.rebuild_after_settings()

    def log(self, text):
        if self.event_callback is not None:
            self.event_callback(text)

    def rebuild_after_settings(self):
        try:
            self.noise_floor = compute_noise_floor_from_optional_file(self.noise_wav_path)
            if self.noise_floor is not None:
                self.log(f"Noise loaded: {os.path.basename(self.noise_wav_path)}")
        except Exception as e:
            self.noise_floor = None
            self.log(f"Noise load failed: {e}")

    def apply_settings(self, source_mode, input_wav_path, noise_wav_path, device):
        self.source_mode = source_mode
        self.input_wav_path = input_wav_path
        self.noise_wav_path = noise_wav_path
        self.device = device
        self.rebuild_after_settings()

    def clear(self):
        with self.buffer_lock:
            self.buffer.clear()
        self.sample_counter = 0
        self.last_preview_emit = 0
        self.last_confirm_emit = 0
        self.latest_preview = None
        self.latest_confirmed = None

    def _push_samples(self, block):
        with self.buffer_lock:
            for row in block:
                self.buffer.append(row.copy())
                self.sample_counter += 1

            # fast preview cadence
            if len(self.buffer) >= int(PREVIEW_WINDOW_SEC * FS):
                while (self.sample_counter - self.last_preview_emit) >= int(PREVIEW_HOP_SEC * FS):
                    arr = np.array(self.buffer, dtype=np.float32)
                    preview = self.preview_processor.process(arr, noise_floor=self.noise_floor, timestamp=self.sample_counter / FS)
                    if preview is not None:
                        self.latest_preview = preview
                    self.last_preview_emit += int(PREVIEW_HOP_SEC * FS)

    def _analysis_loop(self):
        confirm_step_samples = int(CONFIRM_STEP_SEC * FS)
        while self.running:
            time.sleep(max(CONFIRM_STEP_SEC * 0.5, 0.05))
            with self.buffer_lock:
                enough = len(self.buffer) >= int((CONFIRM_WINDOW_SEC + CONFIRM_DELAY_SEC) * FS)
                if not enough:
                    continue
                if (self.sample_counter - self.last_confirm_emit) < confirm_step_samples:
                    continue
                arr = np.array(self.buffer, dtype=np.float32)
                delay_n = int(CONFIRM_DELAY_SEC * FS)
                win_n = int(CONFIRM_WINDOW_SEC * FS)
                if arr.shape[0] < (delay_n + win_n):
                    continue
                if delay_n > 0:
                    segment = arr[-(delay_n + win_n):-delay_n, :]
                else:
                    segment = arr[-win_n:, :]
                sample_counter_snapshot = self.sample_counter
                preview_snapshot = self.latest_preview

            try:
                confirmed = analyze_array_offline_style(
                    segment[:, :CHANNELS_TOTAL].astype(np.float32),
                    FS,
                    noise_floor=self.noise_floor
                )
                if confirmed is None:
                    continue
                payload = {
                    "t": sample_counter_snapshot / FS,
                    "preview": preview_snapshot,
                    "confirmed": confirmed,
                }
                self.result_queue.put(payload)
                self.last_confirm_emit = sample_counter_snapshot
            except Exception as e:
                self.log(f"Analysis error: {e}")

    def _wav_loop(self):
        if not os.path.exists(self.input_wav_path):
            self.log(f"Input WAV not found: {self.input_wav_path}")
            self.running = False
            return
        fs_wav, x = wavfile.read(self.input_wav_path)
        if fs_wav != FS:
            self.log(f"WAV fs={fs_wav}, expected {FS}")
            self.running = False
            return
        if x.ndim != 2 or x.shape[1] < CHANNELS_TOTAL:
            self.log("WAV must have at least 6 channels")
            self.running = False
            return

        self.wav_data = normalize_audio_array(x)
        self.wav_pos = 0
        duration_sec = len(self.wav_data) / float(fs_wav)
        self.log(f"WAV replay started REAL-TIME: {os.path.basename(self.input_wav_path)} | duration={duration_sec:.1f}s")

        # Real-time scheduler: do not let CPU/analysis speed determine playback speed.
        # A 90-second WAV must take approximately 90 seconds at INPUT_WAV_REPLAY_SPEED=1.0.
        block_size = max(1, int(0.025 * fs_wav))
        speed = max(float(INPUT_WAV_REPLAY_SPEED), 1e-6)
        t0 = time.perf_counter()
        pushed_samples = 0

        while self.running and self.wav_pos < len(self.wav_data):
            end = min(self.wav_pos + block_size, len(self.wav_data))
            block = self.wav_data[self.wav_pos:end, :]
            self._push_samples(block)
            self.wav_pos = end
            pushed_samples += len(block)

            target_elapsed = (pushed_samples / float(fs_wav)) / speed
            sleep_for = (t0 + target_elapsed) - time.perf_counter()
            if sleep_for > 0:
                time.sleep(min(sleep_for, 0.05))
            else:
                # If analysis is temporarily late, yield briefly but never fast-forward the file.
                time.sleep(0.001)

        self.log("WAV replay finished.")
        self.running = False

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            self.log(str(status))
        self._push_samples(indata.copy())

    def start(self):
        if self.running:
            return
        self.clear()
        self.running = True

        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()

        if self.source_mode == "live":
            try:
                dev = self.device
                if dev is None:
                    dev = sd.default.device[0]
                self.stream = sd.InputStream(
                    samplerate=FS,
                    channels=CHANNELS_TOTAL,
                    dtype="float32",
                    callback=self._audio_callback,
                    blocksize=max(1, int(PREVIEW_HOP_SEC * FS)),
                    device=dev,
                )
                self.stream.start()
                self.log(f"Live microphone started. device={dev}")
            except Exception as e:
                self.running = False
                self.log(f"Error opening InputStream: {e}")
        else:
            self.io_thread = threading.Thread(target=self._wav_loop, daemon=True)
            self.io_thread.start()

    def stop(self):
        self.running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.log("Engine stopped.")




# =========================================================
# TEMPORAL ANGULAR CLUSTERING
# =========================================================
class AngularClusterStabilizer:
    """
    Converts sensitive, jumpy raw candidates into a stable visual/track angle.

    Principle:
    - Raw live DOA may jump because every sliding window can have a different SRP peak.
    - A real drone tends to repeat around the same azimuth for a short period.
    - We therefore vote candidate angles into a rolling circular cluster before showing
      the confirmed radar cone.
    """
    def __init__(self):
        self.samples = deque()
        self.stable_angle = None
        self.last_cluster_time = None
        self.locked = False
        self.last_cluster_score = 0.0
        self.last_cluster_density = 0.0
        self.last_votes = 0.0

    def reset(self):
        self.samples.clear()
        self.stable_angle = None
        self.last_cluster_time = None
        self.locked = False
        self.last_cluster_score = 0.0
        self.last_cluster_density = 0.0
        self.last_votes = 0.0

    def _prune(self, now_t):
        while self.samples and (now_t - self.samples[0][0]) > CLUSTER_WINDOW_SEC:
            self.samples.popleft()

    def add_candidate(self, now_t, angle, confidence, gate_score=0.0, candidate=True):
        if not candidate or not np.isfinite(angle):
            self._prune(now_t)
            return
        conf = float(confidence) if np.isfinite(confidence) else 0.05
        conf = max(0.02, min(1.0, conf))
        gate_score = max(0.0, min(1.0, float(gate_score)))
        # Vote weight keeps weak acquisitions visible to the cluster, but high evidence dominates.
        weight = 0.40 + 0.60 * conf
        weight *= 0.60 + 0.40 * gate_score
        self.samples.append((float(now_t), float(wrap180(angle)), float(weight)))
        self._prune(now_t)

    def _best_cluster(self):
        if len(self.samples) == 0:
            return np.nan, 0.0, 0.0, 0.0
        angles = np.array([a for _, a, _ in self.samples], dtype=float)
        weights = np.array([w for _, _, w in self.samples], dtype=float)
        total_w = float(np.sum(weights))
        if total_w <= 1e-9:
            return np.nan, 0.0, 0.0, 0.0

        centers = np.arange(-180.0, 180.0, CLUSTER_BIN_DEG)
        best_score = -1.0
        best_center = np.nan
        best_mask = None
        for c in centers:
            dist = ang_dist_deg(angles, c)
            mask = dist <= CLUSTER_RADIUS_DEG
            # Triangular kernel: center votes more, edges vote less.
            local_w = weights[mask] * (1.0 - np.clip(dist[mask] / CLUSTER_RADIUS_DEG, 0.0, 1.0) * 0.45)
            score = float(np.sum(local_w))
            if score > best_score:
                best_score = score
                best_center = c
                best_mask = mask
        if best_mask is None or best_score <= 0:
            return np.nan, 0.0, 0.0, total_w
        cluster_angle = weighted_circular_mean_deg(angles[best_mask], weights[best_mask])
        density = float(best_score / max(total_w, 1e-9))
        return cluster_angle, best_score, density, total_w

    def step(self, now_t, raw_angle, raw_conf, gate_score=0.0, candidate=True):
        """
        Fast-track stabilizer for field use.

        Old behavior waited for a dense angular cluster every time. That was safe,
        but it fragmented continuous paths such as a figure-eight: when the source
        crossed quickly, the rolling cluster split and the displayed track disappeared.

        New behavior:
        1) Before lock: still requires a short angular cluster.
        2) After lock: uses track continuity, so a moving drone can be followed
           smoothly through turns without forcing a new cluster at every angle.
        3) If evidence weakens, it holds the last valid track briefly instead of
           chopping the line into pieces.
        """
        self.add_candidate(now_t, raw_angle, raw_conf, gate_score=gate_score, candidate=candidate)
        cluster_angle, votes, density, total_votes = self._best_cluster()
        self.last_votes = float(votes)
        self.last_cluster_density = float(density)
        self.last_cluster_score = float(votes * density)

        raw_valid = candidate and np.isfinite(raw_angle) and float(gate_score) >= TRACK_CONTINUITY_MIN_GATE

        # TRACK CONTINUITY MODE: once locked, do not break the track just because
        # the best cluster moves or becomes temporarily broad. Follow plausible raw
        # motion and keep the visual output continuous.
        if self.locked and self.stable_angle is not None and np.isfinite(self.stable_angle):
            if raw_valid:
                dist = float(ang_dist_deg(float(raw_angle), float(self.stable_angle)))
                # If motion is plausible or cluster is weak/broad, follow the raw candidate.
                # This is what prevents 8-shaped passes from fragmenting.
                if dist <= TRACK_CONTINUITY_RADIUS_DEG or density < CLUSTER_MIN_DENSITY_TRACKING:
                    self.stable_angle = smooth_display_angle(self.stable_angle, float(raw_angle), alpha=TRACK_CONTINUITY_ALPHA)
                    self.last_cluster_time = float(now_t)
                    return {
                        "state": "TRACKING",
                        "angle": self.stable_angle,
                        "cluster_angle": cluster_angle,
                        "cluster_votes": votes,
                        "cluster_density": density,
                        "cluster_score": self.last_cluster_score,
                        "locked": True,
                        "has_cluster": np.isfinite(cluster_angle),
                    }

        detected = np.isfinite(cluster_angle) and votes >= CLUSTER_MIN_VOTES_DETECTED and density >= CLUSTER_MIN_DENSITY_DETECTED
        tracking = np.isfinite(cluster_angle) and votes >= CLUSTER_MIN_VOTES_TRACKING and density >= CLUSTER_MIN_DENSITY_TRACKING

        if detected:
            alpha = CLUSTER_SMOOTH_ALPHA_TRACK if self.locked else CLUSTER_SMOOTH_ALPHA_ACQUIRE
            self.stable_angle = smooth_display_angle(self.stable_angle, cluster_angle, alpha=alpha)
            self.last_cluster_time = float(now_t)
            self.locked = bool(tracking or self.locked)
            state = "TRACKING" if tracking or self.locked else "DETECTED"
            return {
                "state": state,
                "angle": self.stable_angle,
                "cluster_angle": cluster_angle,
                "cluster_votes": votes,
                "cluster_density": density,
                "cluster_score": self.last_cluster_score,
                "locked": self.locked,
                "has_cluster": True,
            }

        # If we have a valid raw candidate but not enough cluster yet, show it as ACQUIRING
        # only internally. It can trigger early suspected state, but not hard alarm alone.
        if raw_valid and not self.locked:
            if self.stable_angle is None or not np.isfinite(self.stable_angle):
                self.stable_angle = float(raw_angle)
            else:
                self.stable_angle = smooth_display_angle(self.stable_angle, float(raw_angle), alpha=CLUSTER_SMOOTH_ALPHA_ACQUIRE)
            self.last_cluster_time = float(now_t)
            return {
                "state": "ACQUIRING",
                "angle": self.stable_angle,
                "cluster_angle": cluster_angle,
                "cluster_votes": votes,
                "cluster_density": density,
                "cluster_score": self.last_cluster_score,
                "locked": False,
                "has_cluster": False,
            }

        # Keep last stable angle after evidence weakens; prevents timeline/radar gaps.
        recent = self.last_cluster_time is not None and (now_t - self.last_cluster_time) <= CLUSTER_HOLD_SEC
        if recent and self.stable_angle is not None and np.isfinite(self.stable_angle):
            return {
                "state": "HOLD",
                "angle": self.stable_angle,
                "cluster_angle": cluster_angle,
                "cluster_votes": votes,
                "cluster_density": density,
                "cluster_score": self.last_cluster_score,
                "locked": self.locked,
                "has_cluster": False,
            }

        self.locked = False
        self.stable_angle = None
        return {
            "state": "NO TARGET" if not candidate else "ACQUIRING",
            "angle": np.nan,
            "cluster_angle": cluster_angle,
            "cluster_votes": votes,
            "cluster_density": density,
            "cluster_score": self.last_cluster_score,
            "locked": False,
            "has_cluster": False,
        }

# =========================================================
# FINAL STABLE GATE
# =========================================================
class StableRadarGate:
    """
    V6 acquisition-first gate.

    Main change vs V5:
    - During acquisition, DOA angle is NOT killed by angular-speed filtering.
      Live windows can jump before a target is locked; rejecting these early jumps
      was the main reason weak drone sounds were not appearing.
    - After TRACKING is reached, angular-speed filtering becomes active.
    - If tracking is lost for several windows, the gate unlocks and can re-acquire
      from any angle.
    """
    def __init__(self):
        self.history = deque(maxlen=PERSIST_HISTORY_LEN)
        self.last_angle = None
        self.last_time = None
        self.last_positive_time = None
        self.tracking_locked = False
        self.reject_count = 0

    def reset(self):
        self.history.clear()
        self.last_angle = None
        self.last_time = None
        self.last_positive_time = None
        self.tracking_locked = False
        self.reject_count = 0

    def step(self, now_t, angle_raw, conf_raw, evidence, preview_ok=False):
        strict_count = int(evidence["strict_count"])
        soft_count = int(evidence["soft_count"])
        strict_ratio = float(evidence["strict_ratio"])
        peak_ratio_multi = float(evidence["peak_ratio_multi"]) if np.isfinite(evidence["peak_ratio_multi"]) else np.nan
        median_flatness = float(evidence["median_flatness"]) if np.isfinite(evidence["median_flatness"]) else np.nan
        snr_est = float(evidence["snr_est"]) if np.isfinite(evidence["snr_est"]) else np.nan
        jittery = bool(evidence["jittery"])
        stable_track = bool(evidence["stable_track"])
        gate_score = float(evidence["gate_score"])
        force_noise_block = bool(evidence.get("force_noise_block", False))

        # Extremely soft hard-veto. False positives are handled by state/persistence,
        # not by blocking weak candidates before they are visible.
        hard_veto = bool(force_noise_block)
        if strict_count == 0 and (not preview_ok) and gate_score < 0.25:
            hard_veto = True
        if np.isfinite(snr_est) and np.isfinite(peak_ratio_multi):
            if snr_est < -14.0 and peak_ratio_multi < 1.0005 and gate_score < 0.35:
                hard_veto = True
            if snr_est < 0.5 and peak_ratio_multi < 1.03:
                hard_veto = True
            if snr_est < -2.0 and peak_ratio_multi < 1.06:
                hard_veto = True
        if np.isfinite(median_flatness) and median_flatness > 0.995 and strict_ratio < 0.03 and gate_score < 0.35:
            hard_veto = True

        # Acquisition-first candidate rule.
        candidate = (
            (not hard_veto)
            and np.isfinite(angle_raw)
            and (
                gate_score >= 0.50
                or (strict_count >= 2 and (not np.isfinite(snr_est) or snr_est >= 0.5))
                or (strict_ratio >= 0.12 and (not np.isfinite(peak_ratio_multi) or peak_ratio_multi >= 1.03))
                or (preview_ok and (not np.isfinite(snr_est) or snr_est >= 1.0))
            )
        )

        conf = float(conf_raw) if np.isfinite(conf_raw) else np.nan
        if np.isfinite(conf):
            # Keep weak candidates visible. Tracking still needs repeated positives.
            conf *= (0.70 + 0.30 * max(0.0, min(1.0, gate_score)))
            if hard_veto:
                conf *= 0.05
            elif jittery and self.tracking_locked:
                conf *= 0.75

        angle = angle_raw if np.isfinite(angle_raw) else np.nan
        accepted = False
        if candidate and np.isfinite(angle):
            dt = CONFIRM_STEP_SEC if self.last_time is None else max(1e-6, now_t - float(self.last_time))

            if not self.tracking_locked or self.last_angle is None or not np.isfinite(self.last_angle):
                # During acquisition, accept the candidate even if it jumps.
                accepted = True
            else:
                delta = wrap180(angle - self.last_angle)
                max_step = MAX_CONFIRMED_ANGULAR_SPEED_DEG_S * dt
                accepted = np.abs(delta) <= max_step

            if accepted:
                self.last_angle = float(angle)
                self.last_positive_time = float(now_t)
                self.reject_count = 0
            else:
                self.reject_count += 1
                if self.reject_count >= 3:
                    # Lost lock: re-open acquisition from any angle.
                    self.tracking_locked = False
                    self.last_angle = None
                    self.history.clear()
                    accepted = True
                    self.last_angle = float(angle)
                    self.last_positive_time = float(now_t)
                    self.reject_count = 0
                else:
                    recent = (self.last_positive_time is not None) and ((now_t - self.last_positive_time) <= PERSIST_DECAY_KEEP_SEC)
                    if recent and self.last_angle is not None:
                        angle = float(self.last_angle)
                    else:
                        angle = np.nan
        else:
            recent = (self.last_positive_time is not None) and ((now_t - self.last_positive_time) <= PERSIST_DECAY_KEEP_SEC)
            if recent and self.last_angle is not None:
                angle = float(self.last_angle)
            else:
                angle = np.nan

        self.last_time = float(now_t)
        self.history.append(1 if candidate and np.isfinite(angle) else 0)
        positive_count = int(np.sum(self.history))

        if hard_veto:
            state = "NO TARGET"
            angle = np.nan
            conf = 0.0 if np.isfinite(conf) else 0.0
        elif positive_count >= PERSIST_TRACK_MIN_POSITIVE and np.isfinite(conf) and conf >= TRACK_CONF_THRESHOLD and np.isfinite(angle):
            state = "TRACKING"
            self.tracking_locked = True
        elif positive_count >= PERSIST_MIN_POSITIVE and np.isfinite(conf) and conf >= DETECT_CONF_THRESHOLD and np.isfinite(angle):
            state = "DETECTED"
            self.tracking_locked = False
        elif candidate and np.isfinite(angle):
            state = "ACQUIRING"
            self.tracking_locked = False
        elif preview_ok:
            state = "PREVIEW"
            self.tracking_locked = False
        else:
            state = "NO TARGET"
            angle = np.nan
            conf = 0.0 if np.isfinite(conf) else 0.0
            self.tracking_locked = False

        return {
            "state": state,
            "angle": angle,
            "confidence": conf,
            "positive_count": positive_count,
            "candidate": candidate,
            "hard_veto": hard_veto,
        }

# =========================================================
# UI
# =========================================================
class RadarWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.preview_angle = None
        self.preview_conf = 0.0
        self.confirmed_angle = None
        self.confirmed_conf = 0.0
        self.confirmed_detected = False
        self.history = []
        self.setMinimumSize(420, 420)

    def clear(self):
        self.preview_angle = None
        self.confirmed_angle = None
        self.confirmed_conf = 0.0
        self.confirmed_detected = False
        self.history = []
        self.update()

    def update_target(self, t, preview_angle, preview_conf, confirmed_angle, confirmed_conf, confirmed_detected):
        self.preview_angle = preview_angle if np.isfinite(preview_angle) else None
        self.preview_conf = float(preview_conf) if np.isfinite(preview_conf) else 0.0
        self.confirmed_angle = confirmed_angle if np.isfinite(confirmed_angle) else None
        self.confirmed_conf = float(confirmed_conf) if np.isfinite(confirmed_conf) else 0.0
        self.confirmed_detected = bool(confirmed_detected)

        if self.confirmed_angle is not None and self.confirmed_detected:
            self.history.append((float(t), float(self.confirmed_angle), float(self.confirmed_conf)))

        tmin = float(t) - RADAR_TRAIL_SECONDS
        self.history = [h for h in self.history if h[0] >= tmin]
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        w = rect.width()
        h = rect.height()
        cx = w / 2
        cy = h / 2
        r0 = min(w, h) * 0.42

        p.fillRect(rect, QColor(8, 15, 24))

        grid_pen = QPen(QColor(55, 85, 105), 1)
        p.setPen(grid_pen)
        for scale in [0.25, 0.50, 0.75, 1.0]:
            rr = r0 * scale
            p.drawEllipse(QPointF(cx, cy), rr, rr)

        p.drawLine(QPointF(cx - r0, cy), QPointF(cx + r0, cy))
        p.drawLine(QPointF(cx, cy - r0), QPointF(cx, cy + r0))
        d0 = r0 / math.sqrt(2)
        p.drawLine(QPointF(cx - d0, cy - d0), QPointF(cx + d0, cy + d0))
        p.drawLine(QPointF(cx - d0, cy + d0), QPointF(cx + d0, cy - d0))

        # Angle labels use the SAME convention as the algorithm and the radar drawing:
        # 0° = top/front, +90° = right, -90° = left, ±180° = bottom/back.
        # Extra 30° labels make values such as -136° visually understandable.
        p.setFont(QFont("Arial", 9))
        for label_ang in [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180]:
            th_label = math.radians(label_ang)
            lx = cx + r0 * 1.10 * math.sin(th_label)
            ly = cy + r0 * 1.10 * math.cos(th_label)
            label = "180°" if abs(label_ang) == 180 else f"{label_ang}°"
            if label_ang in [0, 90, -90, 180]:
                p.setPen(QColor(170, 255, 210))
                p.setFont(QFont("Arial", 10, QFont.Bold))
            else:
                p.setPen(QColor(170, 195, 215))
                p.setFont(QFont("Arial", 8))
            p.drawText(QPointF(lx - 16, ly + 5), label)

        # preview overlay
        if self.preview_angle is not None and self.preview_conf >= PREVIEW_CONF_THRESHOLD:
            p.setPen(QPen(QColor(80, 160, 255, 140), 2, Qt.DashLine))
            th = math.radians(self.preview_angle)
            x = cx + r0 * 0.82 * math.sin(th)
            y = cy + r0 * 0.82 * math.cos(th)
            p.drawLine(QPointF(cx, cy), QPointF(x, y))

        # confirmed trail
        for i, (_, ang, conf) in enumerate(self.history):
            alpha = int(255 * (i + 1) / max(len(self.history), 1))
            alpha = max(25, min(alpha, 210))
            color = QColor(0, 255, 160, alpha)
            p.setPen(QPen(color, 2))
            th = math.radians(ang)
            x = cx + r0 * 0.90 * math.sin(th)
            y = cy + r0 * 0.90 * math.cos(th)
            p.drawLine(QPointF(cx, cy), QPointF(x, y))

        # confirmed cone
        if self.confirmed_angle is not None and self.confirmed_detected:
            conf = max(0.0, min(1.0, self.confirmed_conf))
            cone_width = max(8, 32 - 20 * conf)
            color = QColor(0, 255, 140, 150) if conf >= TRACK_CONF_THRESHOLD else QColor(255, 210, 0, 150)
            p.setPen(Qt.NoPen)
            p.setBrush(color)
            start_deg = -(self.confirmed_angle + cone_width / 2.0) + 90
            span_deg = cone_width
            radar_rect = QRectF(cx - r0, cy - r0, 2 * r0, 2 * r0)
            p.drawPie(radar_rect, int(start_deg * 16), int(span_deg * 16))

            p.setPen(QPen(QColor(255, 255, 255), 3))
            th = math.radians(self.confirmed_angle)
            x = cx + r0 * 0.96 * math.sin(th)
            y = cy + r0 * 0.96 * math.cos(th)
            p.drawLine(QPointF(cx, cy), QPointF(x, y))

        p.setPen(QPen(QColor(110, 150, 175), 2))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), r0, r0)
        p.end()


class MetricCard(QFrame):
    def __init__(self, title, value="--", parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame { background-color: #111b25; border: 1px solid #294055; border-radius: 12px; }
            QLabel { color: #d7ecff; }
        """)
        layout = QVBoxLayout(self)
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-size: 12px; color: #8fb4d8;")
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #ffffff;")
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, text):
        self.value_label.setText(text)


class AlertPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.title = QLabel("FIELD ALARM")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("font-size: 13px; color: #9fbdda; letter-spacing: 1px;")
        self.message = QLabel("ARMED - NO TARGET")
        self.message.setAlignment(Qt.AlignCenter)
        self.message.setStyleSheet("font-size: 34px; font-weight: 900; color: #6dff9a;")
        self.sub = QLabel("Drone signature required before alarm")
        self.sub.setAlignment(Qt.AlignCenter)
        self.sub.setStyleSheet("font-size: 13px; color: #b7cbe0;")
        self.layout.addWidget(self.title)
        self.layout.addWidget(self.message)
        self.layout.addWidget(self.sub)
        self.setMinimumHeight(96)
        self.set_mode("idle")

    def set_mode(self, mode, message=None, sub=None):
        if mode == "alarm":
            bg = "#5b0000"; border = "#ff2b2b"; fg = "#ffffff"
            msg = message or "⚠ DRONE DETECTED"
        elif mode == "confirmed":
            bg = "#302000"; border = "#ffcc00"; fg = "#ffd95a"
            msg = message or "DRONE CONFIRMED"
        elif mode == "suspect":
            bg = "#182438"; border = "#3d84ff"; fg = "#7fb7ff"
            msg = message or "DRONE SUSPECTED"
        elif mode == "disarmed":
            bg = "#161616"; border = "#666666"; fg = "#bbbbbb"
            msg = message or "DISARMED"
        else:
            bg = "#07131d"; border = "#27506e"; fg = "#6dff9a"
            msg = message or "ARMED - NO TARGET"
        self.setStyleSheet(f"QFrame {{ background-color: {bg}; border: 2px solid {border}; border-radius: 14px; }}")
        self.message.setText(msg)
        self.message.setStyleSheet(f"font-size: 34px; font-weight: 900; color: {fg};")
        if sub is not None:
            self.sub.setText(sub)


class RadarResearchWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Acoustic Radar - Field Fast Track Mode v10")
        self.resize(1800, 1040)

        self.pending_logs = []
        self.result_queue = queue.Queue()

        self.event_list = QListWidget()

        self.engine = HybridEngine(self.result_queue, event_callback=self.log_event)

        self.display_history = deque(maxlen=400)
        self.last_confirmed = None
        self.last_preview = None
        self.stable_gate = StableRadarGate()
        self.cluster_stabilizer = AngularClusterStabilizer()

        self.alarm_armed = True
        self.alarm_muted = False
        self.alarm_active = False
        self.field_positive_since = None
        self.field_last_positive_time = None
        self.last_beep_time = 0.0
        self.last_field_score = 0.0

        self._build_ui()
        self._setup_plots()
        self.populate_devices()
        self.apply_engine_settings_to_ui()

        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(self.poll_queue)
        self.poll_timer.start(UI_POLL_MS)

        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.refresh_plots)
        self.plot_timer.start(PLOT_REFRESH_MS)

    def log_event(self, text):
        if not hasattr(self, "event_list") or self.event_list is None:
            self.pending_logs.append(text)
            return
        if self.event_list.count() > 100:
            self.event_list.takeItem(self.event_list.count() - 1)
        self.event_list.insertItem(0, QListWidgetItem(text))

    def closeEvent(self, event):
        self.engine.stop()
        event.accept()

    def toggle_arm(self):
        self.alarm_armed = not self.alarm_armed
        if self.alarm_armed:
            self.arm_btn.setText("ARMED")
            self.arm_btn.setStyleSheet("background: #0f5f2a; color: white; border: 1px solid #38d46a; padding: 6px 10px; border-radius: 8px;")
            self.log_event("Alarm armed.")
        else:
            self.arm_btn.setText("DISARMED")
            self.arm_btn.setStyleSheet("background: #3a3a3a; color: white; border: 1px solid #888; padding: 6px 10px; border-radius: 8px;")
            self.alarm_active = False
            self.log_event("Alarm disarmed.")
        self.update_alert_panel("NO TARGET", np.nan, 0.0)

    def toggle_mute_alarm(self):
        self.alarm_muted = not self.alarm_muted
        self.mute_btn.setText("Unmute Alarm" if self.alarm_muted else "Mute Alarm")
        self.log_event("Alarm muted." if self.alarm_muted else "Alarm unmuted.")

    def update_alert_panel(self, status, angle, field_score):
        if not self.alarm_armed:
            self.alert_panel.set_mode("disarmed", sub="Alarm output disabled")
            return
        if status == "ALARM":
            angle_txt = f" | {angle:.1f}°" if np.isfinite(angle) else ""
            self.alert_panel.set_mode("alarm", message="⚠ DRONE DETECTED" + angle_txt, sub=f"field score={field_score:.2f} | audible alarm {'muted' if self.alarm_muted else 'active'}")
        elif status == "DRONE CONFIRMED":
            angle_txt = f" | {angle:.1f}°" if np.isfinite(angle) else ""
            self.alert_panel.set_mode("confirmed", message="DRONE CONFIRMED" + angle_txt, sub=f"waiting alarm persistence | score={field_score:.2f}")
        elif status == "DRONE SUSPECTED":
            self.alert_panel.set_mode("suspect", message="DRONE SUSPECTED", sub=f"not enough evidence for alarm | score={field_score:.2f}")
        elif status == "ACQUIRING":
            self.alert_panel.set_mode("suspect", message="ACQUIRING", sub=f"checking direction cluster and drone signature | score={field_score:.2f}")
        else:
            self.alert_panel.set_mode("idle", message="ARMED - NO TARGET", sub=f"background/noise blocked | score={field_score:.2f}")

    def compute_field_score(self, evidence, cluster_out, confirmed):
        gate_score = float(evidence.get("gate_score", 0.0))
        snr = float(evidence.get("snr_est", np.nan))
        pr = float(evidence.get("peak_ratio_multi", np.nan))
        flat = float(evidence.get("median_flatness", np.nan))
        strict_count = float(evidence.get("strict_count", 0))
        density = float(cluster_out.get("cluster_density", np.nan))
        votes = float(cluster_out.get("cluster_votes", np.nan))
        dom = float(confirmed.get("dominant_freq", np.nan)) if confirmed is not None else np.nan

        # Scores are deliberately conservative. Energy alone cannot trigger field alarm.
        gate_s = max(0.0, min(1.0, gate_score))
        cluster_s = 0.0 if not np.isfinite(density) else max(0.0, min(1.0, density))
        vote_s = 0.0 if not np.isfinite(votes) else max(0.0, min(1.0, votes / max(CLUSTER_MIN_VOTES_TRACKING, 1e-6)))
        snr_s = 0.5 if not np.isfinite(snr) else max(0.0, min(1.0, (snr + 3.0) / 14.0))
        pr_s = 0.0 if not np.isfinite(pr) else max(0.0, min(1.0, (pr - 1.00) / 0.12))
        flat_s = 0.5 if not np.isfinite(flat) else max(0.0, min(1.0, (0.99 - flat) / 0.16))
        strict_s = max(0.0, min(1.0, strict_count / 25.0))
        dom_s = 0.0
        if np.isfinite(dom):
            # Drone-useful range: reject very low rumble and very high-only random noise.
            if 350.0 <= dom <= 2200.0:
                dom_s = 1.0
            elif 250.0 <= dom <= 2600.0:
                dom_s = 0.5

        score = (
            0.22 * gate_s +
            0.20 * cluster_s +
            0.12 * vote_s +
            0.18 * snr_s +
            0.12 * pr_s +
            0.08 * flat_s +
            0.04 * strict_s +
            0.04 * dom_s
        )
        return max(0.0, min(1.0, float(score)))

    def input_and_noise_are_same_file(self):
        try:
            if self.source_combo.currentText() != "wav":
                return False
            a = os.path.abspath(self.input_path_edit.text().strip())
            b = os.path.abspath(self.noise_path_edit.text().strip())
            return bool(a and b and os.path.normcase(a) == os.path.normcase(b))
        except Exception:
            return False

    def evaluate_confirmed_evidence(self, confirmed):
        strict_count = int(confirmed.get("strict_count", 0))
        soft_count = int(confirmed.get("soft_count", 0))
        strict_ratio = float(confirmed.get("strict_ratio", 0.0))
        peak_ratio_multi = float(confirmed.get("peak_ratio_multi", np.nan))
        median_flatness = float(confirmed.get("median_flatness", np.nan))
        snr_est = float(confirmed.get("snr_est", np.nan))
        jittery = bool(confirmed.get("jittery", False))
        stable_track = bool(confirmed.get("stable_track", False))

        # Calibration sanity check:
        # If the same WAV is selected as both input and noise reference, it must be treated
        # as background/calibration sound, not as a drone target.
        if self.input_and_noise_are_same_file():
            return {
                "candidate": False,
                "gate_score": 0.0,
                "strict_count": strict_count,
                "soft_count": soft_count,
                "strict_ratio": strict_ratio,
                "peak_ratio_multi": peak_ratio_multi,
                "median_flatness": median_flatness,
                "snr_est": snr_est,
                "jittery": jittery,
                "stable_track": stable_track,
                "force_noise_block": True,
                "noise_reason": "INPUT=NOISE",
            }

        score = 0
        checks = 0

        checks += 1
        if strict_count >= MIN_STRICT_COUNT:
            score += 1

        checks += 1
        if strict_ratio >= MIN_STRICT_RATIO:
            score += 1

        checks += 1
        if np.isfinite(peak_ratio_multi) and peak_ratio_multi >= MIN_PEAK_RATIO_MULTI:
            score += 1

        checks += 1
        if np.isfinite(median_flatness) and median_flatness <= MAX_MEDIAN_FLATNESS:
            score += 1

        checks += 1
        if (not np.isfinite(snr_est)) or snr_est >= MIN_SELECTION_SNR_DB:
            score += 1

        checks += 1
        if stable_track and (not jittery):
            score += 1

        gate_score = score / max(checks, 1)

        # Noise-reference gate:
        # The old sensitive v7 allowed strict_count alone to pass. That is why a
        # calibration/non-drone file could still create an angle. Here, weak SNR +
        # weak beam peak is explicitly treated as background, even if energy exists.
        noise_like = False
        if np.isfinite(snr_est) and np.isfinite(peak_ratio_multi):
            if snr_est < 0.5 and peak_ratio_multi < 1.03:
                noise_like = True
            if snr_est < -2.0 and peak_ratio_multi < 1.06:
                noise_like = True
        if np.isfinite(median_flatness) and median_flatness > 0.985 and peak_ratio_multi < 1.05:
            noise_like = True

        # Acquisition gate: still sensitive, but background-like windows cannot become candidates.
        candidate = (gate_score >= 0.50) and (not noise_like)

        return {
            "candidate": candidate,
            "gate_score": gate_score,
            "strict_count": strict_count,
            "soft_count": soft_count,
            "strict_ratio": strict_ratio,
            "peak_ratio_multi": peak_ratio_multi,
            "median_flatness": median_flatness,
            "snr_est": snr_est,
            "jittery": jittery,
            "stable_track": stable_track,
            "force_noise_block": bool(noise_like),
            "noise_reason": "LOW_SNR_LOW_PEAK" if noise_like else "",
        }


    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # left
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        control_frame = QFrame()
        control_frame.setStyleSheet("QFrame { background: #0d141b; border: 1px solid #294055; border-radius: 10px; } QLabel { color: #d7ecff; } QLineEdit, QComboBox { background: #111b25; color: #e7f4ff; border: 1px solid #35506b; padding: 4px; } QPushButton { background: #173149; color: white; border: 1px solid #3d6488; padding: 6px 10px; border-radius: 8px; }")
        cf = QGridLayout(control_frame)

        self.source_combo = QComboBox()
        self.source_combo.addItems(["wav", "live"])

        self.device_combo = QComboBox()
        self.refresh_devices_btn = QPushButton("Refresh Devices")

        self.input_path_edit = QLineEdit(DEFAULT_INPUT_WAV_PATH)
        self.noise_path_edit = QLineEdit(DEFAULT_NOISE_WAV_PATH)
        self.browse_input_btn = QPushButton("Browse Input")
        self.browse_noise_btn = QPushButton("Browse Noise")

        self.apply_btn = QPushButton("Apply Settings")
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.clear_btn = QPushButton("Clear")
        self.arm_btn = QPushButton("ARMED")
        self.mute_btn = QPushButton("Mute Alarm")
        self.arm_btn.setStyleSheet("background: #0f5f2a; color: white; border: 1px solid #38d46a; padding: 6px 10px; border-radius: 8px;")
        self.mute_btn.setStyleSheet("background: #33210b; color: white; border: 1px solid #d89b34; padding: 6px 10px; border-radius: 8px;")

        cf.addWidget(QLabel("Source"), 0, 0)
        cf.addWidget(self.source_combo, 0, 1)
        cf.addWidget(QLabel("Input Device"), 0, 2)
        cf.addWidget(self.device_combo, 0, 3)
        cf.addWidget(self.refresh_devices_btn, 0, 4)

        cf.addWidget(QLabel("Input WAV"), 1, 0)
        cf.addWidget(self.input_path_edit, 1, 1, 1, 3)
        cf.addWidget(self.browse_input_btn, 1, 4)

        cf.addWidget(QLabel("Noise WAV"), 2, 0)
        cf.addWidget(self.noise_path_edit, 2, 1, 1, 3)
        cf.addWidget(self.browse_noise_btn, 2, 4)

        cf.addWidget(self.apply_btn, 3, 1)
        cf.addWidget(self.start_btn, 3, 2)
        cf.addWidget(self.stop_btn, 3, 3)
        cf.addWidget(self.clear_btn, 3, 4)
        cf.addWidget(self.arm_btn, 4, 1, 1, 2)
        cf.addWidget(self.mute_btn, 4, 3, 1, 2)

        left_layout.addWidget(control_frame, 0)

        self.radar = RadarWidget()
        left_layout.addWidget(self.radar, 4)

        self.timeline_plot = pg.PlotWidget()
        self.timeline_plot.setBackground("#0d141b")
        left_layout.addWidget(self.timeline_plot, 2)

        splitter.addWidget(left_widget)

        # right
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        metrics_grid = QGridLayout()
        self.status_card = MetricCard("Target Status", "NO TARGET")
        self.az_card = MetricCard("Confirmed Azimuth", "--")
        self.conf_card = MetricCard("Confirmed Confidence", "--")
        self.preview_conf_card = MetricCard("Preview Confidence", "--")
        self.freq_card = MetricCard("Dominant Frequency", "--")
        self.snr_card = MetricCard("SNR Estimate", "--")
        self.strict_card = MetricCard("Strict / Soft", "--")
        self.pr_card = MetricCard("Peak Ratio Multi", "--")
        self.noise_gate_card = MetricCard("Noise Gate", "--")
        self.motion_card = MetricCard("Motion Stability", "--")

        metrics_grid.addWidget(self.status_card, 0, 0)
        metrics_grid.addWidget(self.az_card, 0, 1)
        metrics_grid.addWidget(self.conf_card, 0, 2)
        metrics_grid.addWidget(self.preview_conf_card, 1, 0)
        metrics_grid.addWidget(self.freq_card, 1, 1)
        metrics_grid.addWidget(self.snr_card, 1, 2)
        metrics_grid.addWidget(self.strict_card, 2, 0)
        metrics_grid.addWidget(self.pr_card, 2, 1)
        metrics_grid.addWidget(self.noise_gate_card, 2, 2)
        metrics_grid.addWidget(self.motion_card, 3, 0, 1, 3)

        self.alert_panel = AlertPanel()
        right_layout.addWidget(self.alert_panel, 0)

        metrics_panel = QWidget()
        metrics_panel.setLayout(metrics_grid)
        right_layout.addWidget(metrics_panel)

        self.band_plot = pg.PlotWidget()
        self.band_plot.setBackground("#0d141b")
        self.band_plot.setTitle("Band Energy Timeline")
        self.band_plot.setLabel("left", "Energy")
        self.band_plot.setLabel("bottom", "Time (s)")
        right_layout.addWidget(self.band_plot, 2)

        self.beam_plot = pg.PlotWidget()
        self.beam_plot.setBackground("#0d141b")
        self.beam_plot.setTitle("Confirmed Beam Power Scan")
        self.beam_plot.setLabel("left", "Power")
        self.beam_plot.setLabel("bottom", "Angle (deg)")
        right_layout.addWidget(self.beam_plot, 2)

        self.spec_plot = pg.PlotWidget()
        self.spec_plot.setBackground("#0d141b")
        self.spec_plot.setTitle("Raw Spectrogram (Mic A)")
        self.spec_plot.setLabel("left", "Freq (Hz)")
        self.spec_plot.setLabel("bottom", "Time (s)")
        self.spec_img = pg.ImageItem()
        self.spec_plot.addItem(self.spec_img)
        right_layout.addWidget(self.spec_plot, 3)

        self.event_list.setStyleSheet("""
            QListWidget {
                background-color: #0d141b;
                color: #d7ecff;
                border: 1px solid #294055;
                border-radius: 8px;
                font-size: 12px;
            }
        """)
        self.event_list.setMaximumHeight(180)
        right_layout.addWidget(self.event_list, 1)

        splitter.addWidget(right_widget)
        splitter.setSizes([820, 980])

        # connect
        self.refresh_devices_btn.clicked.connect(self.populate_devices)
        self.browse_input_btn.clicked.connect(self.browse_input_wav)
        self.browse_noise_btn.clicked.connect(self.browse_noise_wav)
        self.apply_btn.clicked.connect(self.apply_settings)
        self.start_btn.clicked.connect(self.start_engine)
        self.stop_btn.clicked.connect(self.stop_engine)
        self.clear_btn.clicked.connect(self.clear_all)
        self.arm_btn.clicked.connect(self.toggle_arm)
        self.mute_btn.clicked.connect(self.toggle_mute_alarm)

        for msg in self.pending_logs:
            self.log_event(msg)
        self.pending_logs = []

    def _setup_plots(self):
        pg.setConfigOptions(antialias=False)
        self.timeline_plot.setTitle("Preview + Confirmed DOA Timeline")
        self.timeline_plot.setLabel("left", "Azimuth (deg)")
        self.timeline_plot.setLabel("bottom", "Time (s)")
        self.timeline_plot.showGrid(x=True, y=True, alpha=0.25)
        self.timeline_plot.setYRange(-180, 180)

        self.preview_curve = self.timeline_plot.plot(pen=pg.mkPen((80, 160, 255, 110), width=1, style=Qt.DashLine))
        self.confirm_curve = self.timeline_plot.plot(pen=pg.mkPen((0, 255, 140, 220), width=2))

        self.band_curves = [
            self.band_plot.plot(pen=pg.mkPen((255, 180, 0), width=2)),
            self.band_plot.plot(pen=pg.mkPen((0, 210, 255), width=2)),
            self.band_plot.plot(pen=pg.mkPen((170, 120, 255), width=2)),
        ]

        self.beam_curve = self.beam_plot.plot(pen=pg.mkPen((255, 255, 255), width=2))

    def populate_devices(self):
        self.device_combo.clear()
        self.device_combo.addItem("Default system device", None)
        try:
            for i, d in enumerate(sd.query_devices()):
                if int(d.get("max_input_channels", 0)) > 0:
                    label = f"{i}: {d['name']} ({int(d['max_input_channels'])} in)"
                    self.device_combo.addItem(label, i)
        except Exception as e:
            self.log_event(f"Device listing failed: {e}")

    def browse_input_wav(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Input WAV", self.input_path_edit.text(), "WAV Files (*.wav);;All Files (*)")
        if path:
            self.input_path_edit.setText(path)
            self.log_event(f"Input WAV selected: {path}")

    def browse_noise_wav(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Noise WAV", self.noise_path_edit.text(), "WAV Files (*.wav);;All Files (*)")
        if path:
            self.noise_path_edit.setText(path)
            self.log_event(f"Noise WAV selected: {path}")

    def apply_engine_settings_to_ui(self):
        self.source_combo.setCurrentText(self.engine.source_mode)
        self.input_path_edit.setText(self.engine.input_wav_path)
        self.noise_path_edit.setText(self.engine.noise_wav_path)

    def apply_settings(self):
        source_mode = self.source_combo.currentText()
        input_wav = self.input_path_edit.text().strip()
        noise_wav = self.noise_path_edit.text().strip()
        device = self.device_combo.currentData()
        self.engine.apply_settings(source_mode, input_wav, noise_wav, device)
        self.log_event(f"Settings applied. source={source_mode} | input={input_wav} | noise={noise_wav}")
        if source_mode == "wav":
            try:
                if os.path.normcase(os.path.abspath(input_wav)) == os.path.normcase(os.path.abspath(noise_wav)):
                    self.log_event("Calibration mode: input WAV and noise WAV are identical, target output will be blocked.")
            except Exception:
                pass

    def start_engine(self):
        self.apply_settings()
        self.engine.start()

    def stop_engine(self):
        self.engine.stop()

    def clear_all(self):
        self.engine.clear()
        self.display_history.clear()
        self.last_confirmed = None
        self.last_preview = None
        self.stable_gate.reset()
        self.cluster_stabilizer.reset()
        self._radar_display_angle = None
        self.radar.clear()
        self.preview_curve.setData([], [])
        self.confirm_curve.setData([], [])
        for c in self.band_curves:
            c.setData([], [])
        self.beam_curve.setData([], [])
        self.spec_img.clear()
        self.status_card.set_value("NO TARGET")
        self.az_card.set_value("--")
        self.conf_card.set_value("--")
        self.preview_conf_card.set_value("--")
        self.freq_card.set_value("--")
        self.snr_card.set_value("--")
        self.strict_card.set_value("--")
        self.pr_card.set_value("--")
        self.noise_gate_card.set_value("--")
        self.motion_card.set_value("--")
        self.alarm_active = False
        self.field_positive_since = None
        self.field_last_positive_time = None
        self.last_field_score = 0.0
        self.update_alert_panel("NO TARGET", np.nan, 0.0)
        self.event_list.clear()
        self.log_event("All live data and plots cleared.")

    def poll_queue(self):
        changed = False
        while not self.result_queue.empty():
            payload = self.result_queue.get_nowait()
            t = float(payload["t"])
            preview = payload.get("preview")
            confirmed = payload.get("confirmed")
            self.last_preview = preview
            self.last_confirmed = confirmed
            self.display_history.append({"t": t, "preview": preview, "confirmed": confirmed})
            changed = True

        if not changed or self.last_confirmed is None:
            return

        confirmed = self.last_confirmed
        preview = self.last_preview or {}
        now_t = float(self.display_history[-1]["t"])

        preview_conf = preview.get("confidence", np.nan)
        preview_angle = preview.get("angle", np.nan)
        preview_ok = np.isfinite(preview_conf) and preview_conf >= PREVIEW_CONF_THRESHOLD and np.isfinite(preview_angle)

        evidence = self.evaluate_confirmed_evidence(confirmed)
        if evidence.get("force_noise_block", False):
            self.cluster_stabilizer.reset()
        gate_out = self.stable_gate.step(
            now_t=now_t,
            angle_raw=confirmed.get("doa_confirmed", np.nan),
            conf_raw=confirmed.get("confidence_confirmed", np.nan),
            evidence=evidence,
            preview_ok=preview_ok,
        )

        # Dual-layer tracking: keep V6 sensitive acquisition internally, but only show
        # a confirmed radar angle after temporal angular clustering finds a repeated direction.
        cluster_out = self.cluster_stabilizer.step(
            now_t=now_t,
            raw_angle=gate_out.get("angle", np.nan),
            raw_conf=gate_out.get("confidence", np.nan),
            gate_score=evidence.get("gate_score", 0.0),
            candidate=gate_out.get("candidate", False) and not gate_out.get("hard_veto", False),
        )

        raw_status = gate_out["state"]
        cluster_status = cluster_out["state"]
        angle_candidate = cluster_out["angle"]
        conf_c = gate_out["confidence"] if np.isfinite(gate_out.get("confidence", np.nan)) else 0.0

        field_score = self.compute_field_score(evidence, cluster_out, confirmed)
        self.last_field_score = field_score

        has_cluster = cluster_status in ["TRACKING", "DETECTED", "HOLD"] and np.isfinite(angle_candidate)
        noise_blocked = evidence.get("force_noise_block", False) or gate_out.get("hard_veto", False)
        suspected = (not noise_blocked) and has_cluster and field_score >= FIELD_SUSPECT_SCORE
        field_positive = suspected and field_score >= FIELD_CONFIRM_SCORE

        if field_positive:
            if self.field_positive_since is None:
                self.field_positive_since = now_t
            self.field_last_positive_time = now_t
        else:
            recent_positive = self.field_last_positive_time is not None and (now_t - self.field_last_positive_time) <= FIELD_LOST_GRACE_SEC
            if not recent_positive:
                self.field_positive_since = None

        positive_duration = 0.0 if self.field_positive_since is None else max(0.0, now_t - self.field_positive_since)
        confirmed_field = field_positive and positive_duration >= FIELD_CONFIRM_SEC
        alarm_field = self.alarm_armed and field_positive and positive_duration >= FIELD_ALARM_SEC and field_score >= FIELD_ALARM_SCORE

        if noise_blocked:
            status = "NO TARGET"
        elif alarm_field:
            status = "ALARM"
        elif confirmed_field:
            status = "DRONE CONFIRMED"
        elif suspected:
            status = "DRONE SUSPECTED"
        elif raw_status in ["ACQUIRING", "DETECTED", "TRACKING", "PREVIEW"]:
            status = "ACQUIRING"
        else:
            status = "NO TARGET"

        angle_c = angle_candidate if status in ["DRONE SUSPECTED", "DRONE CONFIRMED", "ALARM"] else np.nan

        # Audible alarm: short system beep, repeated while alarm is active.
        self.alarm_active = bool(status == "ALARM")
        now_wall = time.time()
        if self.alarm_active and not self.alarm_muted and (now_wall - self.last_beep_time) >= ALARM_BEEP_INTERVAL_SEC:
            QApplication.beep()
            self.last_beep_time = now_wall

        self.status_card.set_value(status)
        self.update_alert_panel(status, angle_c, field_score)
        self.az_card.set_value(f"{angle_c:.1f}°" if np.isfinite(angle_c) else "--")
        self.conf_card.set_value(f"{conf_c:.2f}" if np.isfinite(conf_c) else "--")
        self.preview_conf_card.set_value(f"{preview_conf:.2f}" if np.isfinite(preview_conf) else "--")
        self.freq_card.set_value(f"{confirmed.get('dominant_freq', np.nan):.1f} Hz" if np.isfinite(confirmed.get('dominant_freq', np.nan)) else "--")
        self.snr_card.set_value(f"{evidence['snr_est']:.1f} dB" if np.isfinite(evidence['snr_est']) else "--")
        gate_txt = f"G {evidence['gate_score']:.2f}"
        veto_txt = " VETO" if gate_out["hard_veto"] else ""
        self.strict_card.set_value(f"{evidence['strict_count']} / {evidence['soft_count']}  {gate_txt}{veto_txt}")
        self.pr_card.set_value(f"{evidence['peak_ratio_multi']:.2f}" if np.isfinite(evidence['peak_ratio_multi']) else "--")
        if evidence.get("force_noise_block", False):
            self.noise_gate_card.set_value("BLOCKED " + str(evidence.get("noise_reason", "")))
        else:
            self.noise_gate_card.set_value("BLOCKED" if gate_out["hard_veto"] else ("PASS" if gate_out["candidate"] else "WAIT"))
        motion_txt = "Stable" if evidence["stable_track"] else ("Jitter" if evidence["jittery"] else "Checking")
        cluster_txt = f"C {cluster_out['cluster_density']:.2f}/{cluster_out['cluster_votes']:.1f}"
        field_txt = f"F {field_score:.2f}"
        self.motion_card.set_value(f"{motion_txt} | {cluster_txt} | {field_txt}" if np.isfinite(cluster_out['cluster_density']) else f"{motion_txt} | {field_txt}")

        # light visual smoothing only for radar cone/line
        if not hasattr(self, "_radar_display_angle"):
            self._radar_display_angle = None
        if np.isfinite(angle_c):
            self._radar_display_angle = smooth_display_angle(self._radar_display_angle, angle_c, alpha=RADAR_DISPLAY_ALPHA)
        else:
            self._radar_display_angle = None

        # save plotted confirmed angle for timeline
        self.display_history[-1]["confirmed_display_angle"] = angle_c if np.isfinite(angle_c) else np.nan
        self.display_history[-1]["status"] = status
        self.display_history[-1]["raw_status"] = raw_status
        self.display_history[-1]["cluster_density"] = cluster_out["cluster_density"]

        t = self.display_history[-1]["t"]
        show_confirmed = status in ["DRONE SUSPECTED", "DRONE CONFIRMED", "ALARM"]
        self.radar.update_target(t, preview_angle, preview_conf, self._radar_display_angle if self._radar_display_angle is not None else angle_c, conf_c, show_confirmed)

    def refresh_plots(self):
        if len(self.display_history) == 0:
            return

        hist = list(self.display_history)
        t_now = hist[-1]["t"]
        hist = [h for h in hist if h["t"] >= t_now - DISPLAY_HISTORY_SEC]

        x = np.array([h["t"] for h in hist], dtype=float)
        y_prev = np.array([np.nan if h["preview"] is None else h["preview"].get("angle", np.nan) for h in hist], dtype=float)

        y_conf = np.array([
            h.get("confirmed_display_angle",
                  np.nan if h.get("confirmed") is None else h["confirmed"].get("doa_confirmed", np.nan))
            for h in hist
        ], dtype=float)
        if TIMELINE_HIDE_WHEN_NO_TARGET:
            y_conf = np.array([
                y if h.get("status") in ["DRONE SUSPECTED", "DRONE CONFIRMED", "ALARM"] else np.nan
                for y, h in zip(y_conf, hist)
            ], dtype=float)

        self.preview_curve.setData(x, y_prev)
        self.confirm_curve.setData(x, y_conf)

        # band energies from confirmed results
        band_data = [[], [], []]
        band_x = []
        for h in hist:
            conf = h["confirmed"]
            if conf is None:
                continue
            be = conf.get("band_energy", [])
            if len(be) >= 3:
                band_x.append(h["t"])
                for i in range(3):
                    band_data[i].append(be[i])
        bx = np.array(band_x, dtype=float)
        if len(bx) > 0:
            for i in range(3):
                self.band_curves[i].setData(bx, np.array(band_data[i], dtype=float))

        # latest beam
        conf = self.last_confirmed
        if conf is not None and conf.get("beam_scan") is not None:
            self.beam_curve.setData(ANGLES, conf["beam_scan"])

        # latest spectrogram
        if conf is not None:
            S = conf.get("spectrogram_db")
            f = conf.get("spectrogram_f")
            t = conf.get("spectrogram_t")
            if S is not None and f is not None and t is not None and len(t) > 0 and len(f) > 0:
                valid_f = f <= DISPLAY_SPEC_MAX_HZ
                S2 = S[valid_f, :]
                f2 = f[valid_f]
                self.spec_img.setImage(S2.T, autoLevels=False)  # FIX: ImageItem expects x=time, y=freq; transpose prevents vertical scrolling
                self.spec_img.setLevels((np.nanpercentile(S2, 20), np.nanpercentile(S2, 99)))
                lut = pg.colormap.get("viridis").getLookupTable(0.0, 1.0, 256)
                self.spec_img.setLookupTable(lut)
                self.spec_img.setRect(QRectF(0, f2[0], t[-1], f2[-1] - f2[0]))
                self.spec_plot.setXRange(0, t[-1], padding=0)
                self.spec_plot.setYRange(f2[0], f2[-1], padding=0)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = RadarResearchWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
