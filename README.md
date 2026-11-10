#  Audio Steganography in  Images

A reproducible Python implementation of audio-file embedding in RGB cover images using parity along a serpentine grid-graph traversal. The embedded payload is the **original audio file bytes** (not encrypted or converted to waveform samples). At extraction, exact MP3 file bytes are recovered and verified by SHA-256.

## Repository contents

```text
run_experiment.py              Embedding, extraction, analysis and visualizations
requirements.txt              Python dependencies
assets/images/f16.jpeg        F-16 cover image (512 x 512)
assets/images/lady_diana.jpg  Lady Diana cover image (461 x 664)
assets/images/mona_lisa.jpg   Mona Lisa cover image (1024 x 1526)
assets/audio/bird_chirping.mp3 Original audio payload
results/                     Optional generated plots (Git-ignored)
```

## Setup

Python 3.10+ is recommended. Open a terminal in this project directory and run:

```bash
python -m venv .venv
# Windows CMD:
.venv\Scripts\activate.bat
# macOS / Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

If using Spyder, configure it to run with the same Python environment that has these packages installed. `imageio-ffmpeg` is used to decode MP3 for visualization only; `soundfile` is an additional fallback.

## Run

```bash
python run_experiment.py --image f16 --save-plots
python run_experiment.py --image lady_diana --save-plots
python run_experiment.py --image mona_lisa --save-plots
```

Use `--no-plots` for faster numeric experiments, or omit `--save-plots` for display-only mode. Graphs saved with `--save-plots` go into `results/<image>/` and include original vs. stego images, six RGB histograms, separate Red/Blue/Green cover-versus-stego H/D/V scatter figures, and original-vs-recovered audio waveforms. Use `--audio /path/to/audio.mp3` to select a different payload. No resizing, trimming, or rewriting of source files occurs.

### What the scatter plots show

For each RGB channel, horizontal, diagonal, and vertical adjacent-pixel intensity pairs are plotted for both the **cover** and **stego** images. The plots sample 15,000 pairs for legibility, while printed Pearson coefficients use the full image.

### Embedding capacity

Each RGB channel supports `height * width - 1` embedded bits; the total maximum is `3 * (height * width - 1)` bits. Reported **theoretical capacity** is the maximum divided by pixels and approaches 3 bits/pixel. **Actual capacity** is the payload size divided by pixels and depends on image dimensions.

### Recovery and limitations

Exact recovery is tested against original MP3 bytes, with SHA-256 of both files printed. Extraction currently requires the exact number of embedded bits, and the same channel order/traversal; these are supplied internally by the demonstration. This is **steganography, not encryption**; the audio contents are not encrypted. Lossy re-encoding, resizing or image recompression of the stego image can destroy embedded information. For persistence, use a lossless PNG, not a JPEG.

The bundled image/audio files are included for local research demonstration. **Confirm redistribution rights for each sample file before publishing this repository publicly.** The MIT license applies to the code, not automatically to third-party photographs, artwork or audio.

## Tests

```bash
python -m unittest discover -s tests
```

## Research use

The software reports channel-wise means, standard deviations, entropy and H/D/V Pearson correlations, plus combined-image MAE, MSE, PSNR, SSIM, embedding/extraction time, and SHA-256 verification. MAE and MSE can be identical here because the embedding modifies each affected pixel by exactly one intensity unit.
