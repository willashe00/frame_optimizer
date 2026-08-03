"""Site hazard resolution: geographic location -> ASCE 7-16 design parameters.

Seismic: the USGS Design Maps web service — the authoritative public API
behind the ASCE Hazard Tool — returns the ASCE 7-16 Section 11.4 seismic
design parameters (Ss, S1, Fa, Fv, SMS, SM1, SDS, SD1, SDC, TL) for a
latitude/longitude + risk category + site class. Responses are cached to a
local JSON file when SiteConfig.cache_path is set, and every governing
parameter can be supplied manually instead, for offline runs or for sites
where the service declines to answer: at high-seismic Site Class D/E
locations ASCE 7-16 Section 11.4.8 requires a site-specific ground-motion
study, so the service returns null Fv/SM1/SD1/SDC. This module refuses to
guess in that case — supply sds/sd1/s1 overrides from the site-specific
study rather than letting a lookup fabricate them.

Wind: there is NO free official API for the ASCE 7 basic wind speed maps
(the ASCE Hazard Tool and ATC Hazards by Location are browser-only; the ATC
endpoint sits behind an interactive challenge and cannot be called from
code). basic_wind_speed_mph is therefore a required manual input for wind
design: look it up once for the site and risk category at
https://ascehazardtool.org/ (or https://hazards.atcouncil.org/) and record
it in the SiteConfig.

Units: spectral accelerations in g, wind speed in mph (3-s gust), TL in
seconds. Latitude/longitude in decimal degrees, west longitudes negative.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# ASCE 7-16 is the target edition throughout the lateral modules: its wind
# profile constants are implemented in lateral_loads.py and IBC 2021 adopts
# it. The USGS service also serves asce7-22; switching editions is future
# work and must change BOTH places together.
_EDITION = "asce7-16"
_USGS_URL = "https://earthquake.usgs.gov/ws/designmaps/{edition}.json"

RISK_CATEGORIES = ("I", "II", "III", "IV")
SITE_CLASSES = ("A", "B", "C", "D", "E")
EXPOSURES = ("B", "C", "D")

# ASCE 7-16 Table 1.5-2 seismic importance factor Ie by risk category
_IE = {"I": 1.0, "II": 1.0, "III": 1.25, "IV": 1.5}

# Longest TL on the ASCE 7-16 Figure 22-14..17 maps. Used only when a fully
# manual site does not state TL: the Cs cap switches from SD1/T to the
# steeper SD1*TL/T^2 only for T > TL, so overstating TL can never lower Cs.
_TL_DEFAULT_S = 12.0

_WIND_SPEED_HELP = (
    "basic_wind_speed_mph is required for wind design and has no free "
    "official API. Look up the ASCE 7-16 basic (3-s gust) wind speed for "
    "this site and risk category at https://ascehazardtool.org/ or "
    "https://hazards.atcouncil.org/ and set it on the SiteConfig."
)


@dataclass
class SiteConfig:
    """Where the building is and which code defaults apply.

    latitude/longitude are the only inputs without universal defaults;
    risk_category II, Site Class D (the ASCE 7-16 Section 11.4.3 default
    when soil properties are unknown), and Exposure C are the customary
    starting points for an industrial building on open terrain.
    """
    latitude: float
    longitude: float
    risk_category: str = "II"
    site_class: str = "D"          # seismic site class (11.4.3 default)
    exposure: str = "C"            # wind exposure category (26.7)
    basic_wind_speed_mph: float | None = None   # see _WIND_SPEED_HELP

    # --- manual seismic parameters (all three -> no network call; any one
    # --- fills a null returned by the USGS service) ---
    sds_override: float | None = None   # g
    sd1_override: float | None = None   # g
    s1_override: float | None = None    # g (mapped S1, for the Cs floor check)
    tl_override_s: float | None = None  # long-period transition period

    cache_path: str | Path | None = None   # JSON cache for USGS responses

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError(f"latitude must be in [-90, 90], got {self.latitude}.")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError(f"longitude must be in [-180, 180], got {self.longitude}.")
        if self.risk_category not in RISK_CATEGORIES:
            raise ValueError(
                f"risk_category must be one of {RISK_CATEGORIES}, "
                f"got {self.risk_category!r}.")
        if self.site_class not in SITE_CLASSES:
            raise ValueError(
                f"site_class must be one of {SITE_CLASSES}, "
                f"got {self.site_class!r}. (Site Class F always requires a "
                "site-specific study — supply sds/sd1/s1 overrides.)")
        if self.exposure not in EXPOSURES:
            raise ValueError(
                f"exposure must be one of {EXPOSURES}, got {self.exposure!r}.")
        if self.basic_wind_speed_mph is not None and self.basic_wind_speed_mph <= 0:
            raise ValueError("basic_wind_speed_mph must be positive when given.")
        for name in ("sds_override", "sd1_override", "s1_override", "tl_override_s"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when given.")

    @property
    def Ie(self) -> float:
        """Seismic importance factor, ASCE 7-16 Table 1.5-2."""
        return _IE[self.risk_category]

    @property
    def has_manual_seismic(self) -> bool:
        return (self.sds_override is not None
                and self.sd1_override is not None
                and self.s1_override is not None)


@dataclass(frozen=True)
class SeismicHazard:
    """ASCE 7-16 Section 11.4 seismic design parameters for one site."""
    sds: float                 # design spectral acceleration, short period (g)
    sd1: float                 # design spectral acceleration, 1 s (g)
    s1: float                  # mapped MCEr spectral acceleration, 1 s (g)
    sdc: str                   # seismic design category A..F
    tl_s: float                # long-period transition period
    ss: float | None = None    # mapped short-period MCEr (informational)
    fa: float | None = None    # site coefficients (informational)
    fv: float | None = None
    sms: float | None = None
    sm1: float | None = None
    source: str = "manual"     # 'manual' or the USGS service URL/edition


def seismic_design_category(sds: float, sd1: float, s1: float,
                            risk_category: str) -> str:
    """SDC per ASCE 7-16 Section 11.6 (Tables 11.6-1 and 11.6-2; the S1 >=
    0.75 escalation to E/F). Used when the USGS service cannot supply it —
    e.g. when sd1 came from a manual override."""
    if s1 >= 0.75:
        return "F" if risk_category == "IV" else "E"
    iv = risk_category == "IV"

    def category(value: float, cuts: tuple[float, float, float]) -> str:
        lo, mid, hi = cuts
        if value < lo:
            return "A"
        if value < mid:
            return "C" if iv else "B"
        if value < hi:
            return "D" if iv else "C"
        return "D"

    by_sds = category(sds, (0.167, 0.33, 0.50))    # Table 11.6-1
    by_sd1 = category(sd1, (0.067, 0.133, 0.20))   # Table 11.6-2
    return max(by_sds, by_sd1)   # letters order by severity


def fetch_usgs_seismic(site: SiteConfig, timeout_s: float = 30.0) -> dict:
    """One GET against the USGS Design Maps service; returns the raw
    response 'data' dict. Raises RuntimeError with an offline recovery hint
    on any network or service failure."""
    query = urlencode({
        "latitude": site.latitude,
        "longitude": site.longitude,
        "riskCategory": site.risk_category,
        "siteClass": site.site_class,
        "title": "frame_optimizer",
    })
    url = f"{_USGS_URL.format(edition=_EDITION)}?{query}"
    request = Request(url, headers={"User-Agent": "frame_optimizer/0.1",
                                    "Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout_s) as response:
            payload = json.load(response)
    except (URLError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"USGS seismic design map request failed ({exc}). If the "
            "machine is offline, set sds_override, sd1_override, and "
            "s1_override on the SiteConfig to run without the service. "
            f"URL tried: {url}"
        ) from exc
    status = payload.get("request", {}).get("status")
    if status != "success":
        raise RuntimeError(
            f"USGS seismic design map service returned status {status!r} "
            f"for {url}.")
    return payload["response"]["data"]


def _cache_key(site: SiteConfig) -> str:
    return (f"{_EDITION}|{site.latitude:.5f}|{site.longitude:.5f}"
            f"|{site.risk_category}|{site.site_class}")


def _fetch_with_cache(site: SiteConfig, fetch) -> dict:
    if site.cache_path is None:
        return fetch(site)
    path = Path(site.cache_path)
    cache: dict = {}
    if path.exists():
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cache = {}   # unreadable cache: refetch and rewrite
    key = _cache_key(site)
    if key not in cache:
        cache[key] = fetch(site)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
    return cache[key]


def resolve_seismic(site: SiteConfig, fetch=fetch_usgs_seismic) -> SeismicHazard:
    """Seismic design parameters for the site.

    Fully manual configs (sds/sd1/s1 overrides all set) never touch the
    network. Otherwise the USGS service is queried (through the JSON cache
    when SiteConfig.cache_path is set) and any individual override replaces
    the corresponding service value — including the nulls the service
    returns where ASCE 7-16 Section 11.4.8 demands a site-specific study.
    Remaining nulls raise instead of being guessed.
    """
    if site.has_manual_seismic:
        sds, sd1, s1 = site.sds_override, site.sd1_override, site.s1_override
        tl = site.tl_override_s if site.tl_override_s is not None else _TL_DEFAULT_S
        return SeismicHazard(
            sds=sds, sd1=sd1, s1=s1, tl_s=tl,
            sdc=seismic_design_category(sds, sd1, s1, site.risk_category),
            source="manual",
        )

    data = _fetch_with_cache(site, fetch)

    def merged(field: str, override: float | None) -> float | None:
        return data.get(field) if override is None else override

    sds = merged("sds", site.sds_override)
    sd1 = merged("sd1", site.sd1_override)
    s1 = merged("s1", site.s1_override)
    tl = merged("tl", site.tl_override_s)
    missing = [n for n, v in (("sds", sds), ("sd1", sd1), ("s1", s1)) if v is None]
    if missing:
        notes = "; ".join(
            str(data[k]) for k in ("fa_note", "fv_note") if data.get(k))
        raise ValueError(
            f"USGS returned null for {missing} at ({site.latitude}, "
            f"{site.longitude}), site class {site.site_class}"
            + (f" ({notes})" if notes else "") + ". ASCE 7-16 Section "
            "11.4.8 requires a site-specific ground-motion study here; set "
            "sds_override/sd1_override/s1_override from that study (or a "
            "different site_class if soil data supports it).")
    sdc = data.get("sdc") or seismic_design_category(
        sds, sd1, s1, site.risk_category)
    return SeismicHazard(
        sds=float(sds), sd1=float(sd1), s1=float(s1), sdc=sdc,
        tl_s=float(tl) if tl is not None else _TL_DEFAULT_S,
        ss=data.get("ss"), fa=data.get("fa"), fv=data.get("fv"),
        sms=data.get("sms"), sm1=data.get("sm1"),
        source=f"USGS {_EDITION} web service",
    )


def resolve_wind_speed(site: SiteConfig) -> float:
    """The basic wind speed, which must be a manual input (see module
    docstring for why no fetch is possible)."""
    if site.basic_wind_speed_mph is None:
        raise ValueError(_WIND_SPEED_HELP)
    return site.basic_wind_speed_mph


@dataclass(frozen=True)
class SiteHazards:
    """Everything the lateral-loads module needs to know about the site."""
    site: SiteConfig
    seismic: SeismicHazard
    basic_wind_speed_mph: float


def resolve_site_hazards(site: SiteConfig,
                         fetch=fetch_usgs_seismic) -> SiteHazards:
    """Resolve seismic (USGS/cache/manual) and wind (manual) hazards in one
    call — the Phase-1 entry point for lateral_design.py."""
    return SiteHazards(
        site=site,
        seismic=resolve_seismic(site, fetch=fetch),
        basic_wind_speed_mph=resolve_wind_speed(site),
    )
