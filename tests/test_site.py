"""Site hazard resolution: config validation, USGS payload mapping, the
Section 11.4.8 null handling, manual overrides, caching, and the SDC table.

No test touches the network: the USGS fetch is injected. The two payload
fixtures are real service responses (recorded 2026-07-26) for Houston, TX
(complete, SDC A) and Los Angeles, CA Site Class D (nulls per 11.4.8).
"""
import json

import pytest

from frame_optimizer.site import (SiteConfig, resolve_seismic,
                                  resolve_site_hazards, resolve_wind_speed,
                                  seismic_design_category)

HOUSTON = {
    "ss": 0.068, "s1": 0.039, "fa": 1.6, "fv": 2.4, "sms": 0.108,
    "sm1": 0.094, "sds": 0.072, "sd1": 0.062, "sdc": "A", "tl": 12,
}
LA_SITE_D_NULLS = {
    "ss": 1.97, "s1": 0.701, "fa": 1.0, "fv": None, "sms": 1.97,
    "sm1": None, "sds": 1.313, "sd1": None, "sdc": None,
    "fv_note": "See Section 11.4.8", "tl": 8,
}


def fetcher(payload, calls=None):
    def fetch(site):
        if calls is not None:
            calls.append(site)
        return dict(payload)
    return fetch


def no_fetch(site):
    raise AssertionError("network fetch should not have been attempted")


def site(**kw):
    base = dict(latitude=29.76, longitude=-95.37)
    base.update(kw)
    return SiteConfig(**base)


# ---------------------------------------------------------------- validation

@pytest.mark.parametrize("kw", [
    dict(latitude=91.0),
    dict(longitude=-181.0),
    dict(risk_category="V"),
    dict(site_class="F"),
    dict(exposure="A"),
    dict(basic_wind_speed_mph=-1.0),
    dict(sds_override=0.0),
])
def test_config_validation(kw):
    with pytest.raises(ValueError):
        site(**kw)


def test_importance_factor_by_risk_category():
    assert site(risk_category="II").Ie == 1.0
    assert site(risk_category="III").Ie == 1.25
    assert site(risk_category="IV").Ie == 1.5


# ------------------------------------------------------------- USGS mapping

def test_usgs_payload_maps_to_hazard():
    hazard = resolve_seismic(site(), fetch=fetcher(HOUSTON))
    assert hazard.sds == pytest.approx(0.072)
    assert hazard.sd1 == pytest.approx(0.062)
    assert hazard.s1 == pytest.approx(0.039)
    assert hazard.sdc == "A"
    assert hazard.tl_s == 12
    assert hazard.fa == pytest.approx(1.6)
    assert "USGS" in hazard.source


def test_null_sd1_raises_with_11_4_8_pointer():
    with pytest.raises(ValueError, match="11.4.8"):
        resolve_seismic(site(latitude=34.05, longitude=-118.25),
                        fetch=fetcher(LA_SITE_D_NULLS))


def test_override_fills_service_null_and_sdc_recomputes():
    hazard = resolve_seismic(
        site(latitude=34.05, longitude=-118.25, sd1_override=0.7),
        fetch=fetcher(LA_SITE_D_NULLS))
    assert hazard.sd1 == pytest.approx(0.7)
    assert hazard.sds == pytest.approx(1.313)   # service value kept
    assert hazard.sdc == "D"                    # computed: null from service


def test_fully_manual_site_never_fetches():
    hazard = resolve_seismic(
        site(sds_override=0.25, sd1_override=0.12, s1_override=0.08),
        fetch=no_fetch)
    assert hazard.sds == pytest.approx(0.25)
    assert hazard.sdc == "B"
    assert hazard.tl_s == pytest.approx(12.0)   # conservative default
    assert hazard.source == "manual"


# ------------------------------------------------------------------- caching

def test_cache_prevents_second_fetch(tmp_path):
    calls = []
    cache = tmp_path / "hazards.json"
    for _ in range(2):   # fresh SiteConfig each time, same site
        resolve_seismic(site(cache_path=cache), fetch=fetcher(HOUSTON, calls))
    assert len(calls) == 1
    assert cache.exists()
    # a different site misses the cache and fetches again
    resolve_seismic(site(latitude=30.0, cache_path=cache),
                    fetch=fetcher(HOUSTON, calls))
    assert len(calls) == 2


def test_corrupt_cache_refetches(tmp_path):
    cache = tmp_path / "hazards.json"
    cache.write_text("{not json", encoding="utf-8")
    calls = []
    hazard = resolve_seismic(site(cache_path=cache),
                             fetch=fetcher(HOUSTON, calls))
    assert len(calls) == 1 and hazard.sdc == "A"
    json.loads(cache.read_text(encoding="utf-8"))   # rewritten valid


# ------------------------------------------------------- SDC per Section 11.6

@pytest.mark.parametrize("sds, sd1, s1, risk, expected", [
    (0.10, 0.05, 0.02, "II", "A"),
    (0.20, 0.05, 0.10, "II", "B"),    # SDS table governs
    (0.40, 0.15, 0.30, "II", "C"),
    (0.55, 0.25, 0.30, "II", "D"),
    (0.40, 0.15, 0.30, "IV", "D"),    # risk IV escalation
    (0.10, 0.15, 0.30, "II", "C"),    # SD1 table governs
    (1.00, 0.50, 0.80, "II", "E"),    # S1 >= 0.75
    (1.00, 0.50, 0.80, "IV", "F"),
])
def test_seismic_design_category(sds, sd1, s1, risk, expected):
    assert seismic_design_category(sds, sd1, s1, risk) == expected


# ---------------------------------------------------------------------- wind

def test_wind_speed_is_required_manual_input():
    with pytest.raises(ValueError, match="ascehazardtool"):
        resolve_wind_speed(site())
    assert resolve_wind_speed(site(basic_wind_speed_mph=107.0)) == 107.0


def test_resolve_site_hazards_bundles_both():
    hazards = resolve_site_hazards(site(basic_wind_speed_mph=115.0),
                                   fetch=fetcher(HOUSTON))
    assert hazards.basic_wind_speed_mph == 115.0
    assert hazards.seismic.sdc == "A"
    assert hazards.site.exposure == "C"
