"""
Unit tests for hdc_lang primitives.

These verify the math works the way Plate 1995 says it should.
If any of these fail, there is no point running the bAbI evaluation.
"""

import numpy as np

from hdc_lang import (
    Codebook,
    HDCMemory,
    bind,
    bundle,
    cosine,
    permute,
    random_bipolar,
    unbind,
)


def test_bind_is_self_inverse():
    """bind(bind(a, b), b) == a in bipolar space."""
    rng = np.random.default_rng(0)
    a = random_bipolar(1024, rng)
    b = random_bipolar(1024, rng)
    recovered = bind(bind(a, b), b)
    assert np.allclose(recovered, a), "bind is not self-inverse"


def test_unbind_retrieves_from_structure():
    """Given Bundle(Bind(a, role_a), Bind(b, role_b)), unbind by role_a gives a."""
    rng = np.random.default_rng(1)
    dim = 4096
    a = random_bipolar(dim, rng)
    b = random_bipolar(dim, rng)
    role_a = random_bipolar(dim, rng)
    role_b = random_bipolar(dim, rng)
    structure = bundle(bind(a, role_a), bind(b, role_b))
    recovered = unbind(structure, role_a)
    sim_a = cosine(recovered, a)
    sim_b = cosine(recovered, b)
    assert sim_a > sim_b, f"unbind failed to prefer a: {sim_a} vs {sim_b}"
    assert sim_a > 0.4, f"unbind signal too weak: {sim_a}"


def test_codebook_cleanup_resolves_noisy():
    """10% bit flip should still cleanup to the correct name."""
    cb = Codebook(dim=4096, seed=0)
    cb.add("cat")
    cb.add("dog")
    cb.add("mouse")
    rng = np.random.default_rng(123)
    cat_noisy = cb.get("cat").copy()
    flip = rng.random(4096) < 0.1
    cat_noisy[flip] *= -1
    name, sim = cb.cleanup(cat_noisy)
    assert name == "cat", f"cleanup returned {name}, expected cat (sim={sim})"
    assert sim > 0.7


def test_memory_single_item_recall():
    """Memory with one Bind(a, b) should return b almost perfectly via unbind(mem, a)."""
    rng = np.random.default_rng(2)
    dim = 4096
    mary = random_bipolar(dim, rng)
    kitchen = random_bipolar(dim, rng)
    mem = HDCMemory(dim, decay=1.0)
    mem.write(bind(mary, kitchen))
    recovered = mem.query(mary)
    sim = cosine(recovered, kitchen)
    assert sim > 0.99, f"single-item recall weak: {sim}"


def test_memory_decay_prefers_recent():
    """After two writes, decay<1 means the more recent one dominates cleanup."""
    rng = np.random.default_rng(3)
    dim = 4096
    mary = random_bipolar(dim, rng)
    kitchen = random_bipolar(dim, rng)
    bathroom = random_bipolar(dim, rng)
    mem = HDCMemory(dim, decay=0.5)
    mem.write(bind(mary, kitchen))
    mem.write(bind(mary, bathroom))
    recovered = mem.query(mary)
    sim_kitchen = cosine(recovered, kitchen)
    sim_bathroom = cosine(recovered, bathroom)
    assert sim_bathroom > sim_kitchen, (
        f"decay didn't prefer recent: kitchen={sim_kitchen}, bathroom={sim_bathroom}"
    )


def test_cross_term_noise_is_small_in_high_dim():
    """
    Noise from cross-terms (like mary*john*hallway) should be ~1/sqrt(D).
    For D=4096 that is ~0.016. Safe margin for cleanup.
    """
    rng = np.random.default_rng(4)
    dim = 4096
    mary = random_bipolar(dim, rng)
    john = random_bipolar(dim, rng)
    hallway = random_bipolar(dim, rng)
    kitchen = random_bipolar(dim, rng)
    noise = mary * john * hallway
    assert abs(cosine(noise, kitchen)) < 0.1
    assert abs(cosine(noise, hallway)) < 0.1


def test_permute_is_reversible():
    """permute(v, k) then permute back by -k returns v."""
    rng = np.random.default_rng(5)
    v = random_bipolar(1024, rng)
    assert np.allclose(permute(permute(v, 3), -3), v)


if __name__ == "__main__":
    tests = [
        test_bind_is_self_inverse,
        test_unbind_retrieves_from_structure,
        test_codebook_cleanup_resolves_noisy,
        test_memory_single_item_recall,
        test_memory_decay_prefers_recent,
        test_cross_term_noise_is_small_in_high_dim,
        test_permute_is_reversible,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {test.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {test.__name__}: {type(e).__name__}: {e}")
    print()
    print(f"{passed}/{len(tests)} tests passed")
    if passed < len(tests):
        raise SystemExit(1)
