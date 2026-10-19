from fs_pairgrpo.evidence import evidence_for_class


def test_missing_evidence_does_not_fall_back_to_the_label_when_disabled():
    assert evidence_for_class("airplane", {}, fallback_to_label=False) == ()
