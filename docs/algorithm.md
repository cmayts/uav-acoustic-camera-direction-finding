# Algorithm outline

## Processing stages

1. Read the multichannel WAV recording and normalize integer samples.
2. Select the four configured microphone channels.
3. Apply the configured band-pass filters.
4. Divide the signal into overlapping snapshots.
5. Estimate direction with SRP-PHAT in the primary and secondary frequency bands.
6. Calculate energy, spectral flatness, peak ratio, dominant frequency, and confidence metrics.
7. Retain snapshots using soft and strict selection thresholds.
8. Apply circular-angle smoothing and tracking penalties where enabled.
9. Aggregate reliable snapshots into an overall direction estimate.
10. Export numerical summaries, diagnostic plots, and presentation notes.

## Coordinate convention

Angles are represented in degrees and wrapped to the interval `[-180, 180)`. The configured physical convention is `0°` down, `90°` right, `180°` up, and `270°` (equivalent to `-90°`) left. Establish the hardware orientation experimentally before comparing estimates with ground truth.

## Microphone geometry

The default array uses four positions on a square: `A` lower-left, `B` lower-right, `C` upper-right, and `D` upper-left. See [microphone_layout.svg](microphone_layout.svg) for the complete geometry and angle reference. Reversing channels or rotating the device changes the reported angle convention.

## Validation guidance

Use controlled recordings with known source angles, distances, and background-noise conditions. Report angular error distributions rather than a single successful example. Keep synthetic plots clearly separated from measured results.
