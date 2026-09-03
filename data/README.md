# Input data

Place local recordings in this directory. Audio files are intentionally excluded from version control to avoid publishing large or sensitive recordings.

Expected defaults:

- `input.wav`: multichannel recording to analyze.
- `analiz.wav`: optional noise-reference recording.

The analyzer expects at least six WAV channels and selects channels 1, 2, 3, and 4 using zero-based indexing. Confirm the channel map for your microphone hardware before running an analysis.

You can use different filenames through the command-line options:

```bash
python src/uav_acoustic_direction_finding.py --input path/to/recording.wav --noise-reference path/to/noise.wav
```

Do not commit recordings unless you have permission to publish them.
