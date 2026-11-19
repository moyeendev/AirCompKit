import importlib

import numpy as np
import pytest

import aircompkit
from aircompkit.cli import main
from aircompkit.core import registry
from aircompkit.core.rng import SeedTree

SUBPACKAGES = [
    "devices", "channel", "aircomp", "receiver", "adversary",
    "defense", "learning", "metrics", "reproduce", "plot",
]


def test_version():
    assert aircompkit.__version__.split(".")[0].isdigit()


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackages_import(name):
    importlib.import_module(f"aircompkit.{name}")


# ---------------------------------------------------------------- registry
def test_register_get_available():
    @registry.register("channel", "Dummy-Test")
    class Dummy:
        pass

    try:
        assert registry.get("channel", "dummy-test") is Dummy
        assert "dummy-test" in registry.available("channel")
        assert "channel" in registry.available()
    finally:
        registry.unregister("channel", "dummy-test")
    assert "dummy-test" not in registry.available("channel")


def test_duplicate_registration_rejected():
    registry.register("attack", "dup-test")(object)
    try:
        with pytest.raises(ValueError):
            registry.register("attack", "dup-test")(int)
        registry.register("attack", "dup-test", overwrite=True)(int)
        assert registry.get("attack", "dup-test") is int
    finally:
        registry.unregister("attack", "dup-test")


def test_unknown_kind_and_name():
    with pytest.raises(KeyError, match="unknown component kind"):
        registry.register("nope", "x")
    with pytest.raises(KeyError, match="available"):
        registry.get("defense", "does-not-exist")


# ---------------------------------------------------------------- seed tree
def test_same_seed_same_stream():
    a = SeedTree(7).generator("channel").standard_normal(8)
    b = SeedTree(7).generator("channel").standard_normal(8)
    np.testing.assert_array_equal(a, b)


def test_streams_independent_of_creation_order():
    t1 = SeedTree(3)
    t1.generator("adversary").standard_normal(100)  # extra stream drawn first
    h1 = t1.generator("channel").standard_normal(8)
    h2 = SeedTree(3).generator("channel").standard_normal(8)
    np.testing.assert_array_equal(h1, h2)


def test_different_names_seeds_and_paths_differ():
    t = SeedTree(0)
    x = t.generator("channel").standard_normal(8)
    assert not np.allclose(x, t.generator("noise").standard_normal(8))
    assert not np.allclose(x, SeedTree(1).generator("channel").standard_normal(8))
    assert not np.allclose(x, t.child("round").generator("channel").standard_normal(8))


def test_known_value_regression():
    # Guards against accidental changes to stream derivation (would break reproducibility).
    # If this fails after a deliberate change, bump the minor version and note it in CHANGELOG.
    assert SeedTree(0).generator("channel").integers(0, 2**31) == 1715899823


def test_bad_seed():
    with pytest.raises(ValueError):
        SeedTree(-1)


# ---------------------------------------------------------------- CLI
def test_cli_list(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "channel:" in out and "attack:" in out


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert aircompkit.__version__ in capsys.readouterr().out
