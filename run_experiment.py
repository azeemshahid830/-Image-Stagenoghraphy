from pathlib import Path
from io import BytesIO
import hashlib
import time
import argparse

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# =====================================================
# INPUT FILES
# =====================================================

ROOT = Path(__file__).resolve().parent
IMAGE_CHOICES = {
    "f16": ROOT / "assets" / "images" / "f16.jpeg",
    "lady_diana": ROOT / "assets" / "images" / "lady_diana.jpg",
    "mona_lisa": ROOT / "assets" / "images" / "mona_lisa.jpg",
}
IMAGE_PATH = IMAGE_CHOICES["mona_lisa"]
AUDIO_PATH = ROOT / "assets" / "audio" / "bird_chirping.mp3"
SAVE_PLOTS = False
PLOT_DIR = None
PLOT_COUNT = 0


ALPHA = 1  # Must be a positive odd integer

# No automatic resizing.
# No audio trimming.
# Output plot files are optional (--save-plots).

CHANNELS = [
    (0, "Red", "red"),
    (1, "Green", "green"),
    (2, "Blue", "blue")
]

EMBED_ORDER = [
    (0, "Red"),
    (2, "Blue"),
    (1, "Green")
]


# =====================================================
# DISPLAY FIGURES DIRECTLY IN SPYDER
# =====================================================

def display_figure(fig):
    global PLOT_COUNT
    if SAVE_PLOTS:
        PLOT_COUNT += 1
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            PLOT_DIR / f"figure_{PLOT_COUNT:02d}.png",
            dpi=200, bbox_inches="tight"
        )

    # Render the figure in memory.
    # This also works when Matplotlib uses Agg.

    try:
        from IPython.display import display
        from IPython.display import Image as NotebookImage

        buffer = BytesIO()

        fig.savefig(
            buffer,
            format="png",
            dpi=130,
            bbox_inches="tight"
        )

        display(
            NotebookImage(data=buffer.getvalue())
        )

        buffer.close()

    except ImportError:
        plt.show()

    finally:
        plt.close(fig)


# =====================================================
# GRID GRAPH SPANNING PATH
# =====================================================

def forest_spanning_path(height, width):

    path = np.arange(
        height * width,
        dtype=np.intp
    ).reshape(height, width)

    path[1::2] = path[1::2, ::-1]

    return path.ravel().copy()


# =====================================================
# GRAPH-BASED AUDIO EMBEDDING
# =====================================================

def embed(plain, bits, path):

    h, w, _ = plain.shape

    stego = plain.copy()

    per_channel = h * w - 1

    position = 0
    usage = {}

    for channel, name in EMBED_ORDER:

        count = min(
            len(bits) - position,
            per_channel
        )

        usage[name] = max(count, 0)

        if count <= 0:
            continue

        pixels = stego[:, :, channel].ravel().copy()

        vertices = path[:count + 1]

        values = pixels[vertices]

        payload = bits[position:position + count]

        # For odd ALPHA:
        # (ALPHA * abs(parent-child)) % 2
        # equals XOR of pixel parities.

        desired_child_parity = (
            (int(values[0]) & 1)
            ^ np.bitwise_xor.accumulate(payload)
        )

        children = values[1:].copy()

        mismatches = (
            (children & 1) != desired_child_parity
        )

        plus = mismatches & (children < 255)
        minus = mismatches & (children == 255)

        children[plus] += np.uint8(1)
        children[minus] -= np.uint8(1)

        pixels[vertices[1:]] = children

        stego[:, :, channel] = pixels.reshape(h, w)

        position += count

    return stego, usage


# =====================================================
# GRAPH-BASED AUDIO EXTRACTION
# =====================================================

def extract(stego, number_of_bits, path):

    h, w, _ = stego.shape

    remaining = number_of_bits

    parts = []

    for channel, _ in EMBED_ORDER:

        count = min(
            remaining,
            h * w - 1
        )

        if count <= 0:
            break

        pixels = stego[:, :, channel].ravel()[
            path[:count + 1]
        ]

        recovered = (
            (pixels[:-1] ^ pixels[1:]) & 1
        ).astype(np.uint8)

        parts.append(recovered)

        remaining -= count

    if parts:
        return np.concatenate(parts)

    return np.empty(0, dtype=np.uint8)


# =====================================================
# PEARSON CORRELATION
# =====================================================

def pearson(x, y):

    x = np.asarray(
        x, dtype=np.float64
    ).ravel()

    y = np.asarray(
        y, dtype=np.float64
    ).ravel()

    sx = x.std()
    sy = y.std()

    if sx == 0 or sy == 0:
        return float("nan")

    return float(
        np.mean(
            (x - x.mean()) * (y - y.mean())
        ) / (sx * sy)
    )


# =====================================================
# HORIZONTAL, VERTICAL, DIAGONAL CORRELATION
# =====================================================

def direction_correlations(channel):

    return {

        "Horizontal": pearson(
            channel[:, :-1],
            channel[:, 1:]
        ),

        "Vertical": pearson(
            channel[:-1, :],
            channel[1:, :]
        ),

        "Diagonal": pearson(
            channel[:-1, :-1],
            channel[1:, 1:]
        )
    }


# =====================================================
# INDEPENDENT RGB IMAGE ANALYSIS
# =====================================================

def image_stats(image):

    result = {}

    for index, name, _ in CHANNELS:

        c = image[:, :, index]

        freq = np.bincount(
            c.ravel(),
            minlength=256
        )

        p = freq[freq > 0] / freq.sum()

        entropy = float(
            -np.sum(p * np.log2(p))
        )

        result[name] = {

            "Mean": float(c.mean()),

            "Std": float(c.std()),

            "Entropy": entropy,

            **direction_correlations(c)
        }

    return result


# =====================================================
# LOCAL MEAN FOR SSIM
# =====================================================

def local_average(a, size=7):

    pad = size // 2

    a = np.pad(
        np.asarray(a, dtype=np.float64),
        pad,
        mode="reflect"
    )

    integral = np.pad(
        a,
        ((1, 0), (1, 0)),
        mode="constant"
    )

    integral = np.cumsum(
        np.cumsum(integral, axis=0),
        axis=1
    )

    return (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    ) / (size * size)


# =====================================================
# SSIM CALCULATION
# =====================================================

def ssim_per_channel(x, y):

    x = x.astype(float)
    y = y.astype(float)

    mx = local_average(x)
    my = local_average(y)

    vx = np.maximum(
        local_average(x * x) - mx * mx,
        0
    )

    vy = np.maximum(
        local_average(y * y) - my * my,
        0
    )

    cov = (
        local_average(x * y) - mx * my
    )

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    ssim_map = (
        (2 * mx * my + c1)
        * (2 * cov + c2)
    ) / (
        (mx * mx + my * my + c1)
        * (vx + vy + c2)
    )

    return float(ssim_map.mean())


# =====================================================
# MAE, MSE, PSNR AND SSIM
# =====================================================

def comparison(reference, candidate):

    delta = (
        reference.astype(np.float64)
        - candidate.astype(np.float64)
    )

    mae = float(
        np.mean(np.abs(delta))
    )

    mse = float(
        np.mean(delta ** 2)
    )

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = float(
            10 * np.log10(255 ** 2 / mse)
        )

    ssim = float(
        np.mean([
            ssim_per_channel(
                reference[:, :, i],
                candidate[:, :, i]
            )
            for i in range(3)
        ])
    )

    return {
        "MAE": mae,
        "MSE": mse,
        "PSNR (dB)": psnr,
        "SSIM": ssim
    }


# =====================================================
# PRINT BOTH IMAGES' CORRELATION AND STATISTICS
# =====================================================

def print_independent_table(plain, stego):

    p_stats = image_stats(plain)
    s_stats = image_stats(stego)

    print("\n" + "=" * 85)

    print("PLAIN IMAGE AND STEGO IMAGE - RGB ANALYSIS")

    print("=" * 85)

    print(
        f"{'Channel':<10}"
        f"{'Measure':<20}"
        f"{'PLAIN IMAGE':>21}"
        f"{'STEGO IMAGE':>21}"
    )

    print("-" * 78)

    for channel in ["Red", "Green", "Blue"]:

        for name in [
            "Mean",
            "Std",
            "Entropy",
            "Horizontal",
            "Vertical",
            "Diagonal"
        ]:

            print(
                f"{channel:<10}"
                f"{name:<20}"
                f"{p_stats[channel][name]:>21.8f}"
                f"{s_stats[channel][name]:>21.8f}"
            )

        print("-" * 78)


# =====================================================
# PRINT MAE, MSE, PSNR AND SSIM COMPARISON
# =====================================================

def print_quality_table(plain, stego):

    baseline = comparison(plain, plain)
    embedded = comparison(plain, stego)

    print("\n" + "=" * 78)

    print("IMAGE QUALITY COMPARISON")

    print("=" * 78)

    print(
        f"{'Measure':<20}"
        f"{'PLAIN vs PLAIN':>25}"
        f"{'PLAIN vs STEGO':>25}"
    )

    for metric in [
        "MAE",
        "MSE",
        "PSNR (dB)",
        "SSIM"
    ]:

        print(
            f"{metric:<20}"
            f"{baseline[metric]:>25.8f}"
            f"{embedded[metric]:>25.8f}"
        )


# =====================================================
# SHOW BOTH IMAGES
# =====================================================

def plot_images(plain, stego):

    fig, axs = plt.subplots(
        1, 2,
        figsize=(11, 5)
    )

    axs[0].imshow(plain)
    axs[0].set_title("Original Plain Image")

    axs[1].imshow(stego)
    axs[1].set_title(
        "Stego Image (Audio Embedded)"
    )

    for ax in axs:
        ax.axis("off")

    fig.tight_layout()

    display_figure(fig)


# =====================================================
# SIX RGB FREQUENCY HISTOGRAMS
# =====================================================

def plot_histograms(plain, stego):

    fig, axs = plt.subplots(
        3, 2,
        figsize=(14, 12)
    )

    images = [
        ("PLAIN", plain),
        ("STEGO", stego)
    ]

    for row, (index, channel, color) in enumerate(CHANNELS):

        for col, (label, img) in enumerate(images):

            frequencies = np.bincount(
                img[:, :, index].ravel(),
                minlength=256
            )

            ax = axs[row, col]

            ax.bar(
                np.arange(256),
                frequencies,
                width=1,
                color=color,
                linewidth=0
            )

            ax.set_title(
                f"{label} IMAGE - {channel} Histogram"
            )

            ax.set_xlabel(
                "Pixel Intensity (0-255)"
            )

            ax.set_ylabel(
                "Pixel Frequency"
            )

            ax.set_xlim(-0.5, 255.5)

    fig.suptitle(
        "PLAIN vs STEGO: RGB FREQUENCY HISTOGRAMS",
        fontsize=15
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.97)
    )

    display_figure(fig)


# =====================================================
# CHANNEL-WISE HORIZONTAL / DIAGONAL / VERTICAL SCATTER
# =====================================================

def plot_channel_hdv(plain, stego, channel_index, channel_name, color):
    """One figure per channel: image plus cover/stego H, D, V scatters."""

    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(
        2, 4, width_ratios=[1.20, 1, 1, 1],
        wspace=0.38, hspace=0.42
    )

    ax_image = fig.add_subplot(gs[:, 0])
    ax_image.imshow(plain)
    ax_image.set_title("Original Cover Image")
    ax_image.axis("off")

    directions = (
        ("Horizontal", lambda x: (x[:, :-1], x[:, 1:])),
        ("Diagonal", lambda x: (x[:-1, :-1], x[1:, 1:])),
        ("Vertical", lambda x: (x[:-1, :], x[1:, :])),
    )
    rng = np.random.default_rng(42)

    for row, (label, image) in enumerate((('Cover', plain), ('Stego', stego))):
        channel = image[:, :, channel_index]
        for col, (direction, pair_fn) in enumerate(directions, start=1):
            a, b = pair_fn(channel)
            x = a.ravel()
            y = b.ravel()

            # Sample only for plot readability and plotting speed.
            # Does not modify data or the underlying correlation analysis.
            sample_size = min(15000, x.size)
            indices = rng.choice(x.size, size=sample_size, replace=False)

            ax = fig.add_subplot(gs[row, col])
            ax.scatter(
                x[indices], y[indices], s=2, alpha=0.18,
                c=color, linewidths=0, rasterized=True
            )
            ax.set_xlim(0, 255)
            ax.set_ylim(0, 255)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel('Pixel intensity', fontsize=8)
            ax.set_ylabel('Adjacent pixel intensity', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.set_title(f'{label} - {direction}', fontsize=10)

    fig.suptitle(f'{channel_name} Channel: Cover Image and H/D/V Pixel Correlations',
                 fontsize=13)
    fig.subplots_adjust(top=0.88)
    display_figure(fig)


# =====================================================
# AUDIO WAVEFORMS (DISPLAY ONLY; ORIGINAL BYTES PRESERVED)
# =====================================================

def decode_audio_for_plot(audio_bytes):
    """
    Decode MP3 bytes into real audio samples strictly for waveform display.
    Uses a bundled FFmpeg (imageio-ffmpeg), system FFmpeg, then soundfile.
    The original MP3 data used for embedding is never changed.
    """
    import shutil
    import subprocess

    ffmpeg_exe = None
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        ffmpeg_exe = shutil.which("ffmpeg")

    if ffmpeg_exe:
        # Decode to mono, 44,100 Hz, 32-bit floating-point PCM.
        # This avoids relying on an installed MP3 codec in soundfile.
        result = subprocess.run(
            [
                ffmpeg_exe, "-hide_banner", "-loglevel", "error",
                "-i", "pipe:0", "-vn",
                "-ac", "1", "-ar", "44100",
                "-f", "f32le", "pipe:1"
            ],
            input=audio_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
        if result.returncode == 0 and result.stdout:
            return np.frombuffer(
                result.stdout, dtype="<f4"
            ).copy(), 44100

    try:
        import soundfile as sf
        samples, sample_rate = sf.read(
            BytesIO(audio_bytes), dtype="float32", always_2d=True
        )
        return samples.mean(axis=1), sample_rate
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        raise RuntimeError(
            "Could not decode MP3 for waveform visualization. "
            "In Spyder's console, run: pip install imageio-ffmpeg "
            "and restart the Python kernel. This does not affect "
            "the original embedding or recovery."
        ) from exc


def plot_audio_waveforms(original_audio, recovered_audio):
    """Display original and recovered decoded waveforms for comparison."""
    original, fs_original = decode_audio_for_plot(original_audio)
    extracted, fs_extracted = decode_audio_for_plot(recovered_audio)

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, sharey=True
    )

    common_duration = max(
        len(original) / fs_original,
        len(extracted) / fs_extracted
    )
    for ax, (label, samples, fs) in zip(axes, (
        ("Original Audio (Before Embedding)", original, fs_original),
        ("Recovered Audio (After Extraction)", extracted, fs_extracted),
    )):
        # Keep all waveform sections visible, even for long files.
        stride = max(1, int(np.ceil(len(samples) / 100000)))
        times = np.arange(0, len(samples), stride) / fs
        ax.plot(times, samples[::stride], linewidth=0.65)
        ax.set_xlim(0, common_duration)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("Amplitude")
        ax.set_title(label)
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle("Audio Waveform Before Embedding and After Extraction")
    fig.tight_layout()
    display_figure(fig)


# =====================================================
# MAIN PROGRAM
# =====================================================

def main():
    global IMAGE_PATH, AUDIO_PATH, SAVE_PLOTS, PLOT_DIR

    parser = argparse.ArgumentParser(
        description="RGB grid-graph audio steganography experiment"
    )
    parser.add_argument(
        "--image", choices=IMAGE_CHOICES, default="mona_lisa",
        help="Cover image (default: mona_lisa)"
    )
    parser.add_argument(
        "--audio", type=Path, default=AUDIO_PATH,
        help="MP3 or other audio file to embed as bytes"
    )
    parser.add_argument(
        "--save-plots", action="store_true",
        help="Save comparison, histogram, HDV, and waveform figures to results/"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Perform analysis without producing figures"
    )
    args = parser.parse_args()
    IMAGE_PATH = IMAGE_CHOICES[args.image]
    AUDIO_PATH = args.audio.expanduser().resolve()
    SAVE_PLOTS = args.save_plots
    PLOT_DIR = ROOT / "results" / args.image

    print("PROGRAM STARTED", flush=True)

    if ALPHA <= 0 or ALPHA % 2 != 1:
        raise ValueError(
            "ALPHA must be a positive odd integer."
        )

    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    if not AUDIO_PATH.is_file():
        raise FileNotFoundError(
            f"Audio not found: {AUDIO_PATH}"
        )

    # -------------------------------------------------
    # LOAD ORIGINAL IMAGE - NO RESIZING
    # -------------------------------------------------

    with Image.open(IMAGE_PATH) as image:

        plain = np.asarray(
            image.convert("RGB"),
            dtype=np.uint8
        ).copy()

    h, w, _ = plain.shape

    if h < 2 or w < 2:
        raise ValueError(
            "Image must be at least 2x2 pixels."
        )

    # -------------------------------------------------
    # LOAD AUDIO AS ORIGINAL BYTES
    # -------------------------------------------------

    audio = AUDIO_PATH.read_bytes()

    bits = np.unpackbits(
        np.frombuffer(
            audio,
            dtype=np.uint8
        ),
        bitorder="big"
    )

    capacity = 3 * (h * w - 1)

    print(
        f"\nCover Image: {IMAGE_PATH.name}"
    )

    print(
        f"Image Dimensions: {w} x {h}"
    )

    print(
        f"Audio File: {AUDIO_PATH.name}"
    )

    print(
        f"Audio Payload: {bits.size:,} bits"
    )

    print(
        f"Image Embedding Capacity: {capacity:,} bits"
    )

    # =================================================
    # CRITICAL CAPACITY CHECK - NEVER REMOVE THIS
    # =================================================

    if bits.size > capacity:

        print("\n" + "=" * 70)

        print(
            "ERROR: AUDIO IS TOO LARGE FOR THIS IMAGE"
        )

        print("=" * 70)

        print(
            f"Required: {bits.size:,} bits"
        )

        print(
            f"Available: {capacity:,} bits"
        )

        print(
            f"Missing: {bits.size - capacity:,} bits"
        )

        print("\nPlease choose one option:")

        print("1. Use a larger image.")
        print("2. Use a shorter/smaller audio file.")
        print("3. Change the audio file.")

        print(
            "\nEmbedding cancelled."
        )

        print(
            "No image resizing or audio trimming performed."
        )

        return

    # =================================================
    # EMBEDDING
    # =================================================

    path = forest_spanning_path(h, w)

    start = time.perf_counter()

    stego, usage = embed(
        plain,
        bits,
        path
    )

    embedding_time = (
        time.perf_counter() - start
    )

    # =================================================
    # EXTRACTION
    # =================================================

    start = time.perf_counter()

    output_bits = extract(
        stego,
        bits.size,
        path
    )

    recovered = np.packbits(
        output_bits,
        bitorder="big"
    ).tobytes()

    extraction_time = (
        time.perf_counter() - start
    )

    exact_recovery = recovered == audio

    # Show actual decoded waveform immediately after extraction.
    print("\nDisplaying original and recovered audio waveforms...", flush=True)
    if not args.no_plots:
        try:
            plot_audio_waveforms(audio, recovered)
        except RuntimeError as exc:
            print(f"Waveform display skipped: {exc}")

    # =================================================
    # CAPACITY AND TIME ANALYSIS
    # =================================================

    print("\n" + "=" * 70)

    print("EMBEDDING CAPACITY AND TIME ANALYSIS")

    print("=" * 70)

    print(
        f"Maximum Capacity: {capacity:,} bits"
    )

    print(
        f"Theoretical EC: {capacity/(h*w):.8f} bpp"
    )

    print(
        f"Actual EC: {bits.size/(h*w):.8f} bpp"
    )

    print(
        f"Red Channel: {usage['Red']:,} bits"
    )

    print(
        f"Blue Channel: {usage['Blue']:,} bits"
    )

    print(
        f"Green Channel: {usage['Green']:,} bits"
    )

    print(
        f"Embedding Time: {embedding_time:.6f} seconds"
    )

    print(
        f"Extraction Time: {extraction_time:.6f} seconds"
    )

    print(
        f"Exact Audio Recovery: {exact_recovery}"
    )

    original_hash = hashlib.sha256(
        audio
    ).hexdigest()

    recovered_hash = hashlib.sha256(
        recovered
    ).hexdigest()

    print(
        "Original Audio SHA256:",
        original_hash
    )

    print(
        "Recovered Audio SHA256:",
        recovered_hash
    )

    if not exact_recovery:
        raise RuntimeError(
            "Recovered audio does not match original."
        )

    # =================================================
    # INDEPENDENT IMAGE CORRELATION ANALYSIS
    # =================================================

    print_independent_table(
        plain,
        stego
    )

    # =================================================
    # IMAGE QUALITY COMPARISON
    # =================================================

    print_quality_table(
        plain,
        stego
    )

    if not args.no_plots:
        # =================================================
        # DISPLAY ORIGINAL AND STEGO IMAGES
        # =================================================

        print(
            "\nDisplaying plain and stego images...",
            flush=True
        )

        plot_images(
            plain,
            stego
        )

        # =================================================
        # DISPLAY ALL SIX RGB HISTOGRAMS
        # =================================================

        print(
            "\nDisplaying six RGB histograms...",
            flush=True
        )

        plot_histograms(
            plain,
            stego
        )

        # =================================================
        # DISPLAY H/D/V SCATTER PLOTS: RED, BLUE, GREEN
        # =================================================

        for channel_index, channel_name, color in (
            (0, "Red", "red"),
            (2, "Blue", "blue"),
            (1, "Green", "green"),
        ):
            print(f"\nDisplaying {channel_name} channel H/D/V scatter plots...",
                  flush=True)
            plot_channel_hdv(plain, stego, channel_index, channel_name, color)

    print("\n" + "=" * 70)

    print("ANALYSIS COMPLETED SUCCESSFULLY")

    print("=" * 70)

    print(
        "Original image and audio inputs unchanged."
    )


if __name__ == "__main__":
    main()