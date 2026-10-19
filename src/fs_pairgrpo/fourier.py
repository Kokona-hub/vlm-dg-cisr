from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .schema import ImageSample, StylePair


def fourier_style_extrapolation(
    source: Image.Image,
    donor: Image.Image,
    alpha: float,
    low_frequency_ratio: float = 0.08,
) -> Image.Image:
    """Mix low-frequency log amplitudes while preserving the source phase."""
    if not 0.0 < low_frequency_ratio <= 0.5:
        raise ValueError("low_frequency_ratio must be in (0, 0.5].")
    source_rgb = source.convert("RGB")
    donor_rgb = donor.convert("RGB").resize(source_rgb.size, Image.Resampling.BICUBIC)
    source_array = np.asarray(source_rgb, dtype=np.float32) / 255.0
    donor_array = np.asarray(donor_rgb, dtype=np.float32) / 255.0

    source_fft = np.fft.fft2(source_array, axes=(0, 1))
    donor_fft = np.fft.fft2(donor_array, axes=(0, 1))
    source_amplitude = np.abs(source_fft)
    donor_amplitude = np.abs(donor_fft)
    source_phase = np.angle(source_fft)

    source_shifted = np.fft.fftshift(source_amplitude, axes=(0, 1))
    donor_shifted = np.fft.fftshift(donor_amplitude, axes=(0, 1))
    height, width = source_array.shape[:2]
    half_h = max(1, int(height * low_frequency_ratio / 2.0))
    half_w = max(1, int(width * low_frequency_ratio / 2.0))
    center_h, center_w = height // 2, width // 2
    h_slice = slice(max(0, center_h - half_h), min(height, center_h + half_h + 1))
    w_slice = slice(max(0, center_w - half_w), min(width, center_w + half_w + 1))

    epsilon = 1e-6
    source_log = np.log(source_shifted[h_slice, w_slice] + epsilon)
    donor_log = np.log(donor_shifted[h_slice, w_slice] + epsilon)
    mixed_log = source_log + alpha * (donor_log - source_log)
    mixed_shifted = source_shifted.copy()
    mixed_shifted[h_slice, w_slice] = np.exp(mixed_log)
    mixed_amplitude = np.fft.ifftshift(mixed_shifted, axes=(0, 1))

    reconstructed_fft = mixed_amplitude * np.exp(1j * source_phase)
    reconstructed = np.real(np.fft.ifft2(reconstructed_fft, axes=(0, 1)))
    reconstructed = np.clip(reconstructed, 0.0, 1.0)
    return Image.fromarray(np.round(reconstructed * 255.0).astype(np.uint8), mode="RGB")


def _pair_seed(global_seed: int, sample_id: str, view_index: int) -> int:
    digest = hashlib.sha1(f"{global_seed}|{sample_id}|{view_index}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def build_style_pairs(
    samples: list[ImageSample],
    output_dir: str | Path,
    views_per_sample: int,
    interpolation_range: tuple[float, float],
    extrapolation_range: tuple[float, float],
    extrapolation_probability: float,
    low_frequency_ratio: float,
    seed: int,
    materialize: bool = True,
    overwrite: bool = False,
) -> list[StylePair]:
    if views_per_sample < 1:
        raise ValueError("views_per_sample must be at least 1.")
    if not 0.0 <= extrapolation_probability <= 1.0:
        raise ValueError("extrapolation_probability must be in [0, 1].")
    output_dir = Path(output_dir).expanduser().resolve()
    donors: dict[tuple[int, str, str], list[ImageSample]] = defaultdict(list)
    domains_by_class: dict[tuple[int, str], set[str]] = defaultdict(set)
    for sample in samples:
        donors[(sample.label_id, sample.class_name, sample.domain)].append(sample)
        domains_by_class[(sample.label_id, sample.class_name)].add(sample.domain)

    pairs: list[StylePair] = []
    for sample in tqdm(samples, desc="Building Fourier pairs", unit="image"):
        other_domains = sorted(
            domain
            for domain in domains_by_class[(sample.label_id, sample.class_name)]
            if domain != sample.domain
        )
        if not other_domains:
            raise ValueError(
                f"No same-class donor from another source domain for {sample.class_name!r}. "
                "Use source domains with a shared label space."
            )
        for view_index in range(views_per_sample):
            pair_seed = _pair_seed(seed, sample.sample_id, view_index)
            rng = random.Random(pair_seed)
            donor_domain = rng.choice(other_domains)
            donor = rng.choice(donors[(sample.label_id, sample.class_name, donor_domain)])
            use_extrapolation = rng.random() < extrapolation_probability
            alpha_range = extrapolation_range if use_extrapolation else interpolation_range
            alpha = rng.uniform(*alpha_range)
            relative_output = Path(sample.domain) / sample.class_name.replace(" ", "_")
            styled_path = output_dir / relative_output / f"{sample.sample_id}_v{view_index}.jpg"
            pair = StylePair(
                pair_id=f"{sample.sample_id}-v{view_index}",
                source=sample,
                donor=donor,
                styled_path=styled_path,
                alpha=alpha,
                low_frequency_ratio=low_frequency_ratio,
                seed=pair_seed,
            )
            if materialize and (overwrite or not styled_path.exists()):
                styled_path.parent.mkdir(parents=True, exist_ok=True)
                with (
                    Image.open(sample.image_path) as source_image,
                    Image.open(donor.image_path) as donor_image,
                ):
                    styled = fourier_style_extrapolation(
                        source_image,
                        donor_image,
                        alpha=alpha,
                        low_frequency_ratio=low_frequency_ratio,
                    )
                    styled.save(styled_path, quality=95, subsampling=0)
            pairs.append(pair)
    return pairs
