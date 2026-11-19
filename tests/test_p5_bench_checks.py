"""P5: benchmarks, reproduction checks and their CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aircompkit import bench
from aircompkit.cli import main
from aircompkit.reproduce import CHECKS, report, yang2020, zhu2020


def test_bench_quick_groups_and_markdown(tmp_path):
    res = bench.run(groups=("mse",), quick=True, repeat=1, out=tmp_path)
    assert res["machine"]["numpy"] and res["mse"][0]["trials_per_s"] > 0
    assert {r["K"] for r in res["mse"]} == {20, 200}
    md = (tmp_path / "bench.md").read_text()
    assert "## mse" in md and "## fl" not in md
    assert json.loads((tmp_path / "bench.json").read_text())["quick"] is True


def test_cli_bench(tmp_path, capsys):
    assert main(["bench", "mse", "--quick", "--repeat", "1", "--out", str(tmp_path)]) == 0
    assert "trials_per_s" in capsys.readouterr().out
    assert (tmp_path / "bench.md").exists()


def _yang_ok():
    g = yang2020.FIG4_GAMMA_DB.tolist()
    step = [0.0 if x < 0 else 1.0 for x in g]
    fig5 = {"n_rx": yang2020.FIG5_N.tolist(), "mean_abs_deviation": 0.05}
    for gg in yang2020.FIG5_GAMMA_DB:
        fig5[f"maxmin_{gg:g}dB"] = yang2020.PAPER_FIG5[gg]
    return {"fig4": {"gamma_db": g, "maxmin": step, "sdr": step, "eigen": step, "sdr_bound": step},
            "fig5": fig5}


def test_yang_checks_pass_and_fail():
    res = _yang_ok()
    assert all(ok for _, ok, _ in yang2020.check(res))
    res["fig4"]["eigen"] = [1.0] * len(res["fig4"]["gamma_db"])  # beats the SDR bound: impossible
    res["fig5"]["mean_abs_deviation"] = 0.3
    failed = [label for label, ok, _ in yang2020.check(res) if not ok]
    assert len(failed) == 2 and any("SDR bound" in f for f in failed)


def test_zhu_checks_without_and_with_fig7():
    z = zhu2020.FIG4_TRUNC.tolist()
    f4 = {"truncation_ratio": z, "closed_form_db": [1.0] * len(z), "sim_rho0_db": [1.0] * len(z),
          "sim_truncation": z, "sim_avg_power_over_budget": [1.0] * len(z)}
    assert len(zhu2020.check({"fig4": f4, "fig7": None})) == 3
    f7 = {"ratios": [0.1, 0.3, 0.5, 0.7, 0.9]}
    for s in ("iid", "noniid"):
        f7[s] = {"ota": [float("nan"), 0.80, 0.85, 0.86, 0.87],
                 "ideal": [float("nan"), 0.80, 0.85, 0.86, 0.90]}
    checks = zhu2020.check({"fig4": f4, "fig7": f7})
    assert len(checks) == 5 and [ok for _, ok, _ in checks] == [True, True, True, False, True]


def test_report_prints_and_registry_complete(capsys):
    assert set(CHECKS) == {"yang2020", "zhu2020"}
    assert report("yang2020", _yang_ok()) is True
    assert "[PASS]" in capsys.readouterr().out


def test_cli_reproduce_check_exit_code(monkeypatch, capsys):
    import aircompkit.core.registry as reg

    bad = _yang_ok()
    bad["fig5"]["mean_abs_deviation"] = 0.5
    monkeypatch.setitem(reg._REGISTRY["reproduction"], "yang2020", lambda **kw: bad)
    assert main(["reproduce", "yang2020", "--check"]) == 1
    assert "[FAIL]" in capsys.readouterr().out
    monkeypatch.setitem(reg._REGISTRY["reproduction"], "yang2020", lambda **kw: _yang_ok())
    assert main(["reproduce", "yang2020", "--check"]) == 0


def test_cli_reproduce_rejects_data_root_for_dataless():
    with pytest.raises(SystemExit, match="uses no dataset"):
        main(["reproduce", "yang2020", "--data-root", "x"])


def test_tutorial_notebooks_present_and_clean():
    nbs = sorted((Path(__file__).parents[1] / "docs" / "tutorials").glob("*.ipynb"))
    assert len(nbs) >= 5
    for p in nbs:
        nb = json.loads(p.read_text(encoding="utf-8"))
        outs = [o for c in nb["cells"] if c["cell_type"] == "code" for o in c.get("outputs", [])]
        assert outs, f"{p.name} has no stored outputs"
        assert not any(o["output_type"] == "error" for o in outs), p.name


def test_cli_reproduce_all_skips_missing_data_and_reports_failures(monkeypatch, tmp_path, capsys):
    import aircompkit.core.registry as reg

    seen = {}

    def needs_data(trials=None, quick=False, out=None, data_root=None):
        raise FileNotFoundError("mnist not found")

    def ok(trials=None, quick=False, out=None):
        seen["out"] = out
        return _yang_ok()

    bad = _yang_ok()
    bad["fig5"]["mean_abs_deviation"] = 0.5
    monkeypatch.setitem(reg._REGISTRY, "reproduction",
                        {"yang2020": ok, "needs_mnist": needs_data, "zhu2020": lambda **kw: None})
    monkeypatch.setitem(CHECKS, "zhu2020", lambda res: [("x", True, "")])
    assert main(["reproduce", "--all", "--quick", "--check", "--data-root", "d", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "2 run, 1 skipped (needs_mnist" in out and Path(seen["out"]) == tmp_path / "yang2020"
    monkeypatch.setitem(reg._REGISTRY["reproduction"], "yang2020", lambda **kw: bad)
    assert main(["reproduce", "--all", "--check"]) == 1
    assert "checks failed: yang2020" in capsys.readouterr().out
