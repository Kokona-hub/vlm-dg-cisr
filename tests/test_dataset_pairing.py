from pathlib import Path

from fs_pairgrpo.fourier import build_style_pairs
from fs_pairgrpo.schema import ImageSample


def sample(sample_id, domain, label_id, class_name, path):
    return ImageSample(sample_id, domain, "train", Path(path), label_id, class_name, (class_name,))


def test_pairs_use_same_class_and_other_domain(tmp_path):
    samples = [
        sample("a", "real", 0, "horse", "a.jpg"),
        sample("b", "clipart", 0, "horse", "b.jpg"),
        sample("c", "real", 1, "dog", "c.jpg"),
        sample("d", "clipart", 1, "dog", "d.jpg"),
    ]
    pairs = build_style_pairs(
        samples,
        output_dir=tmp_path,
        views_per_sample=1,
        interpolation_range=(0.3, 0.9),
        extrapolation_range=(1.0, 1.2),
        extrapolation_probability=0.5,
        low_frequency_ratio=0.08,
        seed=42,
        materialize=False,
    )
    assert len(pairs) == len(samples)
    assert all(pair.source.class_name == pair.donor.class_name for pair in pairs)
    assert all(pair.source.domain != pair.donor.domain for pair in pairs)
