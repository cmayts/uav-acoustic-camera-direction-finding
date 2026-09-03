# Recording protocol

This document summarizes the local experiment catalog without publishing raw audio, personal information, or location metadata.

## Dataset scope

- 85 cataloged recording entries
- Nominal duration: 10 seconds per measurement entry
- Target heights: 2.5 m and 4.0 m
- Fixed azimuths: 0°, 45°, 90°, 135°, 180°, 225°, 270°, and 315°
- Horizontal distances: 1.5 m, 2 m, 3 m, 4 m, and 6 m depending on angle and height
- Additional forward, forward/backward, semicircular, and circular motion paths
- Fan-noise measurements and an empty-room/no-UAV control recording

## Fixed-angle coverage

| Angle | Catalog entries |
|---:|---:|
| 0° | 13 |
| 45° | 8 |
| 90° | 8 |
| 135° | 10 |
| 180° | 14 |
| 225° | 8 |
| 270° | 8 |
| 315° | 8 |

The remaining entries describe moving paths or control conditions rather than a single fixed angle.

## Angle convention

The project convention is:

- 0°: down
- 90°: right
- 180°: up
- 270° (or −90°): left

Microphone positions are A lower-left, B lower-right, C upper-right, and D upper-left. See [microphone_layout.svg](microphone_layout.svg).

## Recommended evaluation fields

For each recording, retain a non-identifying identifier and record:

- Ground-truth azimuth in degrees
- Horizontal distance in metres
- Target height in metres
- Motion pattern
- Noise condition
- Duration in seconds
- Predicted azimuth
- Circular absolute error in degrees
- Detection confidence
- Valid/invalid quality flag with a reason

## Reproducibility and privacy

Raw recordings should remain outside the public repository unless publication rights, consent, and location/privacy risks have been reviewed. A future public dataset should use neutral identifiers, remove embedded metadata, document the microphone orientation, and include a clear data license. Summary statistics and selected plots should not be presented as validation until the angle convention and recording-to-catalog mapping have been independently checked.
