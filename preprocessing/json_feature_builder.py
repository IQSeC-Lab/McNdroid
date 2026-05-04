#!/usr/bin/env python3
"""
Refined frozen feature-space builder for Android telemetry JSON datasets.

Key refinements
---------------
- Uses shared train/test split manifests in both initializer and adaptation.
- Removes the residual explicit-feature hashing mechanism completely.
- Freezes one explicit vocabulary from the initializer train split only.
- Ignores unseen explicit features after freezing.
- Normalizes component actions before adding them to the explicit vocabulary.
- Keeps only coarse, stable component-action semantics as explicit features.
- Pushes normalized open-vocabulary component/action payload evidence into hashed channels.
- Drops raw payload-heavy values from the explicit vocabulary.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
from scipy.sparse import hstack, save_npz
from sklearn.feature_extraction import DictVectorizer, FeatureHasher

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATASET_ROOT = Path("/data/mcndroid/all_json")

LEAKY_TOP_LEVEL_FIELDS = {
    "verdicts",
    "signature_matches",
    "verdict_confidence",
    "sigma_analysis_results",
    "tags",
    "mitre_attack_techniques",
    "attack_techniques",
}

SENSITIVE_PERMISSIONS = {
# DANGEROUS_PERMISSIONS = {
    "android.permission.accept_handover",
    "android.permission.access_background_location",
    "android.permission.access_coarse_location",
    "android.permission.access_fine_location",
    "android.permission.access_media_location",
    "android.permission.activity_recognition",
    "android.permission.answer_phone_calls",
    "android.permission.bluetooth_advertise",
    "android.permission.bluetooth_connect",
    "android.permission.bluetooth_scan",
    "android.permission.body_sensors",
    "android.permission.body_sensors_background",
    "android.permission.call_phone",
    "android.permission.camera",
    "android.permission.get_accounts",
    "android.permission.nearby_wifi_devices",
    "android.permission.post_notifications",
    "android.permission.process_outgoing_calls",
    "android.permission.ranging",
    "android.permission.read_calendar",
    "android.permission.read_call_log",
    "android.permission.read_contacts",
    "android.permission.read_external_storage",
    "android.permission.read_media_audio",
    "android.permission.read_media_images",
    "android.permission.read_media_video",
    "android.permission.read_media_visual_user_selected",
    "android.permission.read_phone_numbers",
    "android.permission.read_phone_state",
    "android.permission.read_sms",
    "android.permission.receive_mms",
    "android.permission.receive_sms",
    "android.permission.receive_wap_push",
    "android.permission.record_audio",
    "android.permission.send_sms",
    "android.permission.use_sip",
    "android.permission.uwb_ranging",
    "android.permission.write_calendar",
    "android.permission.write_call_log",
    "android.permission.write_contacts",
    "android.permission.write_external_storage",
}

PATH_PREFIXES = [
    "/data/data",
    "/data/user",
    "/data",
    "/system",
    "/mnt/sdcard",
    "/sdcard",
    "/proc",
    "/sys",
    "/dev",
    "APP_ASSETS",
]

ENV_HOSTS = {
    "connectivitycheck.gstatic.com",
    "clientservices.googleapis.com",
    "firebaseinstallations.googleapis.com",
    "www.gstatic.com",
}
ENV_REG_DOMAINS = {"gstatic.com", "googleapis.com", "googleusercontent.com"}

UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
HEX_RE = re.compile(r"\b[0-9a-fA-F]{24,}\b")
INT_RE = re.compile(r"\d+")
PKG_RE = re.compile(r"\b(?:[a-zA-Z_][a-zA-Z0-9_]*\.){2,}[A-Za-z_][A-Za-z0-9_]*\b")
ACTION_BASE_RE = re.compile(r"^android\.intent\.action\.[A-Z_]+$")
SHORT_CUSTOM_ACTION_RE = re.compile(r"^[A-Za-z0-9_.$:-]{1,64}$")
PKG_NUM_SUFFIX_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_$.]*?)(\d+)$")
URI_SCHEME_RE = re.compile(r"^(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*):")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def ensure_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def as_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(x)


def lower(x: Any) -> str:
    return as_str(x).strip().lower()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_data(sample: Dict[str, Any]) -> Dict[str, Any]:
    d = sample.get("data", sample)
    d = ensure_dict(d)
    return {k: v for k, v in d.items() if k not in LEAKY_TOP_LEVEL_FIELDS}


def normalize_permission(s: str) -> str:
    s = lower(s)
    if ":" in s:
        s = s.split(":", 1)[0]
    return s


def normalize_domain(host: str) -> str:
    s = lower(host).strip("[]")
    if ":" in s and s.count(":") == 1:
        lhs, rhs = s.rsplit(":", 1)
        if rhs.isdigit():
            s = lhs
    return s.strip(".")


def approx_reg_domain(host: str) -> str:
    host = normalize_domain(host)
    if not host:
        return ""
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    if parts[-2] in {"co", "com", "org", "net", "gov", "ac"} and len(parts) >= 3:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def normalize_path(path: str) -> str:
    s = as_str(path)
    s = UUID_RE.sub("<uuid>", s)
    s = HEX_RE.sub("<hex>", s)
    s = INT_RE.sub("<num>", s)
    s = PKG_RE.sub("<pkg>", s)
    return s.lower()


def path_bucket(path: str) -> str:
    s = as_str(path)
    for prefix in PATH_PREFIXES:
        if s.startswith(prefix):
            return prefix
    if s.startswith("/"):
        return "/other_abs"
    return "other_rel"


def canonical_component(value: str) -> Dict[str, str]:
    s = as_str(value).strip()
    out = {"kind": "unknown", "package": "", "class": "", "action": "", "raw": s}
    if s.startswith("#Intent;"):
        out["kind"] = "intent"
        for part in s.split(";"):
            if part.startswith("action="):
                out["action"] = part.split("=", 1)[1]
            elif part.startswith("component="):
                comp = part.split("=", 1)[1]
                if "/" in comp:
                    pkg, cls = comp.split("/", 1)
                    out["package"] = pkg
                    out["class"] = cls
        return out
    if "(" in s and ")" in s:
        left, right = s.rsplit("(", 1)
        out["kind"] = "component"
        out["package"] = right.rstrip(")").strip()
        out["class"] = left.strip()
        return out
    if s.startswith("android.intent.action."):
        out["kind"] = "action"
        out["action"] = s
        return out
    if "/" in s:
        pkg, cls = s.split("/", 1)
        out["kind"] = "component"
        out["package"] = pkg
        out["class"] = cls
        return out
    out["class"] = s
    return out


# def parse_http_entry(obj: Any) -> Dict[str, Any]:
#     d = ensure_dict(obj)
#     url = as_str(d.get("url", ""))
#     p = urlparse(url) if url else None
#     host = normalize_domain(p.hostname or "") if p else ""
#     return {
#         "scheme": (p.scheme or "").lower() if p else "",
#         "host": host,
#         "reg_domain": approx_reg_domain(host),
#         "path": p.path if p else "",
#         "query_keys": sorted(parse_qs(p.query).keys()) if p else [],
#         "method": lower(d.get("request_method", "")),
#         "status_code": d.get("response_status_code"),
#     }
def parse_http_entry(obj: Any) -> Dict[str, Any]:
    d = ensure_dict(obj)
    url = as_str(d.get("url", ""))

    try:
        p = urlparse(url) if url else None
        host = normalize_domain(p.hostname or "") if p else ""
        scheme = (p.scheme or "").lower() if p else ""
        path = p.path if p else ""
        query_keys = sorted(parse_qs(p.query).keys()) if p else []
    except Exception:
        return {
            "scheme": "",
            "host": "",
            "reg_domain": "",
            "path": "",
            "query_keys": [],
            "method": lower(d.get("request_method", "")),
            "status_code": d.get("response_status_code"),
        }

    return {
        "scheme": scheme,
        "host": host,
        "reg_domain": approx_reg_domain(host),
        "path": path,
        "query_keys": query_keys,
        "method": lower(d.get("request_method", "")),
        "status_code": d.get("response_status_code"),
    }

def save_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_meta_npz(path: Path, y: np.ndarray, hashes: List[str], paths: List[str]) -> None:
    np.savez_compressed(
        path,
        y=y,
        hashes=np.array(hashes, dtype=object),
        paths=np.array(paths, dtype=object),
    )


def sanitize_short_token(s: str) -> str:
    s = lower(s)
    s = UUID_RE.sub("<uuid>", s)
    s = HEX_RE.sub("<hex>", s)
    s = INT_RE.sub("<num>", s)
    s = PKG_RE.sub("<pkg>", s)
    return s


def uri_scheme_of(s: str) -> str:
    m = URI_SCHEME_RE.match(as_str(s).strip())
    return lower(m.group("scheme")) if m else ""


def looks_payload_heavy(s: str) -> bool:
    ss = as_str(s)
    noisy_markers = [
        "{", "}", "[", "]", "Intent {", "mailto:", "http://", "https://",
        "market://", "file://", "content://", "android.intent.extra.", "\n",
    ]
    return any(marker in ss for marker in noisy_markers)


def normalize_component_action_for_explicit(action: str) -> str:
    s = as_str(action).strip()
    if not s:
        return ""

    if ACTION_BASE_RE.fullmatch(s):
        return s

    s = PKG_NUM_SUFFIX_RE.sub(r"\1<num>", s)
    s_norm = sanitize_short_token(s)

    if looks_payload_heavy(s_norm):
        return ""

    if SHORT_CUSTOM_ACTION_RE.fullmatch(s_norm):
        return s_norm

    return ""


def normalize_component_action_for_hash(action: str) -> str:
    s = sanitize_short_token(action)
    s = PKG_NUM_SUFFIX_RE.sub(r"\1<num>", s)
    if len(s) > 256:
        s = s[:256]
    return s


def bucket_count(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 4:
        return "2_4"
    if n <= 8:
        return "5_8"
    return "9_plus"


# -----------------------------------------------------------------------------
# Extractor
# -----------------------------------------------------------------------------

@dataclass
class FrozenFeatureSpaceConfig:
    hash_dim_domains: int = 512
    hash_dim_paths: int = 512
    hash_dim_components: int = 512
    hash_dim_strings: int = 512
    drop_env_hosts: bool = True
    max_text_tokens: int = 32


class Frozen2013FeatureExtractor:
    def __init__(self, config: Optional[FrozenFeatureSpaceConfig] = None):
        self.config = config or FrozenFeatureSpaceConfig()
        self.vectorizer = DictVectorizer(sparse=True)
        self.domain_hasher = FeatureHasher(
            n_features=self.config.hash_dim_domains,
            input_type="string",
            alternate_sign=True,
        )
        self.path_hasher = FeatureHasher(
            n_features=self.config.hash_dim_paths,
            input_type="string",
            alternate_sign=True,
        )
        self.component_hasher = FeatureHasher(
            n_features=self.config.hash_dim_components,
            input_type="string",
            alternate_sign=True,
        )
        self.string_hasher = FeatureHasher(
            n_features=self.config.hash_dim_strings,
            input_type="string",
            alternate_sign=True,
        )
        self.fitted_ = False

    def fit(self, samples: List[Dict[str, Any]]) -> "Frozen2013FeatureExtractor":
        explicit_rows = []
        for i, sample in enumerate(samples):
            try:
                row = self._extract_one(sample)
                explicit_rows.append(row["explicit"])
            except Exception as e:
                logger.warning("fit: failed on sample %d: %s — skipping", i, e)
                explicit_rows.append({})
        self.vectorizer.fit(explicit_rows)
        self.fitted_ = True
        return self

    def transform(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.fitted_:
            raise RuntimeError("Extractor must be fit before transform.")

        explicit_rows = []
        domain_rows = []
        path_rows = []
        comp_rows = []
        string_rows = []

        for i, sample in enumerate(samples):
            try:
                row = self._extract_one(sample)
            except Exception as e:
                logger.warning("transform: failed on sample %d: %s — inserting empty row", i, e)
                row = {
                    "explicit": {},
                    "hash_domains": [],
                    "hash_paths": [],
                    "hash_components": [],
                    "hash_strings": [],
                }
            explicit_rows.append(row["explicit"])
            domain_rows.append(row["hash_domains"])
            path_rows.append(row["hash_paths"])
            comp_rows.append(row["hash_components"])
            string_rows.append(row["hash_strings"])

        X_explicit = self.vectorizer.transform(explicit_rows)
        X_domain = self.domain_hasher.transform(domain_rows)
        X_path = self.path_hasher.transform(path_rows)
        X_comp = self.component_hasher.transform(comp_rows)
        X_string = self.string_hasher.transform(string_rows)

        X_full = hstack([X_explicit, X_domain, X_path, X_comp, X_string], format="csr")

        feature_names = list(self.vectorizer.get_feature_names_out())
        feature_names += [f"hash_domain_{i}" for i in range(self.config.hash_dim_domains)]
        feature_names += [f"hash_path_{i}" for i in range(self.config.hash_dim_paths)]
        feature_names += [f"hash_component_{i}" for i in range(self.config.hash_dim_components)]
        feature_names += [f"hash_string_{i}" for i in range(self.config.hash_dim_strings)]

        return {
            "X_full": X_full,
            "feature_names": feature_names,
        }

    def _extract_one(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        data = get_data(sample)
        explicit: Dict[str, float] = defaultdict(float)
        hash_domains: List[str] = []
        hash_paths: List[str] = []
        hash_components: List[str] = []
        hash_strings: List[str] = []

        permissions_requested = [normalize_permission(x) for x in ensure_list(data.get("permissions_requested")) if as_str(x)]
        permissions_checked = [
            normalize_permission(ensure_dict(x).get("permission", ""))
            for x in ensure_list(data.get("permissions_checked"))
        ]
        http_entries = [parse_http_entry(x) for x in ensure_list(data.get("http_conversations"))]
        dns_entries = [ensure_dict(x) for x in ensure_list(data.get("dns_lookups"))]
        tls_entries = [ensure_dict(x) for x in ensure_list(data.get("tls"))]
        ip_entries = [ensure_dict(x) for x in ensure_list(data.get("ip_traffic"))]
        files_opened = [as_str(x) for x in ensure_list(data.get("files_opened")) if as_str(x)]
        files_written = [as_str(x) for x in ensure_list(data.get("files_written")) if as_str(x)]
        files_deleted = [as_str(x) for x in ensure_list(data.get("files_deleted")) if as_str(x)]
        services_started = [canonical_component(x) for x in ensure_list(data.get("services_started")) if as_str(x)]
        services_opened = [canonical_component(x) for x in ensure_list(data.get("services_opened")) if as_str(x)]
        services_bound = [canonical_component(x) for x in ensure_list(data.get("services_bound")) if as_str(x)]
        services_stopped = [canonical_component(x) for x in ensure_list(data.get("services_stopped")) if as_str(x)]
        activities_started = [canonical_component(x) for x in ensure_list(data.get("activities_started")) if as_str(x)]
        calls_highlighted = [as_str(x) for x in ensure_list(data.get("calls_highlighted")) if as_str(x)]
        invokes = [as_str(x) for x in ensure_list(data.get("invokes")) if as_str(x)]
        commands = [as_str(x) for x in ensure_list(data.get("command_executions")) if as_str(x)]
        system_props = [lower(x) for x in ensure_list(data.get("system_property_lookups")) if as_str(x)]
        signals_hooked = [as_str(x) for x in ensure_list(data.get("signals_hooked")) if as_str(x)]
        signals_observed = [as_str(x) for x in ensure_list(data.get("signals_observed")) if as_str(x)]
        memory_domains = [normalize_domain(x) for x in ensure_list(data.get("memory_pattern_domains")) if as_str(x)]
        memory_urls = [as_str(x) for x in ensure_list(data.get("memory_pattern_urls")) if as_str(x)]
        shared_prefs = [ensure_dict(x) for x in ensure_list(data.get("shared_preferences_sets"))]
        content_sets = [ensure_dict(x) for x in ensure_list(data.get("content_model_sets"))]
        text_highlighted = [as_str(x) for x in ensure_list(data.get("text_highlighted")) if as_str(x)]
        text_decoded = [as_str(x) for x in ensure_list(data.get("text_decoded")) if as_str(x)]
        modules_loaded = [as_str(x) for x in ensure_list(data.get("modules_loaded")) if as_str(x)]
        mutexes = [as_str(x) for x in ensure_list(data.get("mutexes_created")) if as_str(x)]
        proc_created = [as_str(x) for x in ensure_list(data.get("processes_created")) if as_str(x)]
        proc_killed = [as_str(x) for x in ensure_list(data.get("processes_killed")) if as_str(x)]
        reg_keys = [as_str(x) for x in ensure_list(data.get("registry_keys_set")) if as_str(x)]
        files_dropped = [as_str(x) for x in ensure_list(data.get("files_dropped")) if as_str(x)]
        files_copied = [ensure_dict(x) for x in ensure_list(data.get("files_copied"))]
        files_attr = [ensure_dict(x) for x in ensure_list(data.get("files_attribute_changed"))]
        db_opened = [as_str(x) for x in ensure_list(data.get("databases_opened")) if as_str(x)]

        explicit["count.permissions_requested"] = len(permissions_requested)
        explicit["count.permissions_checked"] = len([p for p in permissions_checked if p])
        explicit["count.http"] = len(http_entries)
        explicit["count.dns"] = len(dns_entries)
        explicit["count.tls"] = len(tls_entries)
        explicit["count.ip"] = len(ip_entries)
        explicit["count.fs_open"] = len(files_opened)
        explicit["count.fs_write"] = len(files_written)
        explicit["count.fs_delete"] = len(files_deleted)
        explicit["count.fs_drop"] = len(files_dropped)
        explicit["count.fs_db_open"] = len(db_opened)
        explicit["count.services_started"] = len(services_started)
        explicit["count.services_opened"] = len(services_opened)
        explicit["count.services_bound"] = len(services_bound)
        explicit["count.services_stopped"] = len(services_stopped)
        explicit["count.activities_started"] = len(activities_started)
        explicit["count.calls_highlighted"] = len(calls_highlighted)
        explicit["count.invokes"] = len(invokes)
        explicit["count.commands"] = len(commands)
        explicit["count.system_props"] = len(system_props)
        explicit["count.signals_hooked"] = len(signals_hooked)
        explicit["count.signals_observed"] = len(signals_observed)
        explicit["count.mutexes"] = len(mutexes)
        explicit["count.processes_created"] = len(proc_created)
        explicit["count.processes_killed"] = len(proc_killed)
        explicit["count.registry_keys"] = len(reg_keys)
        explicit["count.modules_loaded"] = len(modules_loaded)
        explicit["count.shared_prefs"] = len(shared_prefs)
        explicit["count.content_sets"] = len(content_sets)

        for p in sorted(set(permissions_requested + [p for p in permissions_checked if p])):
            explicit[f"permission::{p}"] = 1.0
            if p in SENSITIVE_PERMISSIONS:
                explicit[f"permission_sensitive::{p.rsplit('.', 1)[-1]}"] = 1.0

        for sig in sorted(set(signals_hooked)):
            explicit[f"signal_hooked::{sig}"] = 1.0
        for sig in sorted(set(signals_observed)):
            explicit[f"signal_observed::{sig}"] = 1.0
        for sp in sorted(set(system_props)):
            explicit[f"sysprop::{sp}"] = 1.0

        telephony_sinks = {
            "getdeviceid", "getsubscriberid", "getline1number",
            "getimei", "getmeid", "getsimserial",
        }

        explicit["behavior.has_location_permission"] = float(any("location" in p for p in permissions_requested))
        explicit["behavior.has_phone_state_permission"] = float("android.permission.read_phone_state" in permissions_requested)
        explicit["behavior.has_accounts_permission"] = float("android.permission.get_accounts" in permissions_requested)
        explicit["behavior.has_sms_permission"] = float(any("sms" in p for p in permissions_requested))
        explicit["behavior.has_boot_permission"] = float("android.permission.receive_boot_completed" in permissions_requested)
        explicit["behavior.telephony_access"] = float(
            any(
                any(sink in lower(x) for sink in telephony_sinks) or "telephonymanager" in lower(x)
                for x in calls_highlighted + invokes
            )
        )
        explicit["behavior.location_access"] = float(
            any("locationmanager" in lower(x) or "getlastknownlocation" in lower(x) for x in calls_highlighted + invokes)
        )
        explicit["behavior.account_access"] = float(
            any("accountmanager" in lower(x) or "getaccounts" in lower(x) for x in calls_highlighted + invokes)
        )
        explicit["behavior.reflection"] = float(
            any("reflect" in lower(x) or "method.invoke" in lower(x) or "field.get" in lower(x) for x in calls_highlighted + invokes)
        )
        explicit["behavior.root_activity"] = float(
            any(re.search(r"\bsu\b", lower(x)) is not None or "busybox" in lower(x) for x in commands + files_opened + files_written)
        )
        explicit["behavior.boot_persistence"] = float(
            any("boot_completed" in lower(c.get("action", "") + " " + c.get("raw", "")) for c in services_started + services_bound)
            or any("BOOT_COMPLETED" in s for s in signals_hooked)
        )
        explicit["behavior.proc_access"] = float(any("/proc" in p for p in files_opened + files_written + files_deleted))
        explicit["behavior.system_access"] = float(any("/system" in p for p in files_opened + files_written + files_deleted))
        explicit["behavior.network_present"] = float(len(http_entries) + len(dns_entries) + len(tls_entries) + len(ip_entries) > 0)
        explicit["behavior.network_and_id"] = float(explicit["behavior.network_present"] and explicit["behavior.telephony_access"])

        suspicious_ports = {1025, 1030, 1042, 1075, 1080, 1170, 1194, 1234, 1235, 1241, 1243, 1337, 1443, 1541, 1604, 1717, 1900, 1981, 1999, 2001, 2222, 2766, 2773, 2989, 3000, 3001, 3024, 3030, 3128, 3129, 3200, 3333, 3389, 3410, 3478, 3489, 3567, 3790, 4000, 4041, 4043, 4051, 4092, 4222, 4433, 4434, 4435, 4436, 4437, 4438, 4439, 4440, 4441, 4442, 4443, 4444, 4567, 4590, 4747, 4782, 5000, 5001, 5002, 5096, 5222, 5321, 5400, 5405, 5421, 5500, 5555, 5556, 5631, 5632, 5646, 5647, 5650, 5651, 5655, 5665, 5721, 5800, 5900, 5901, 5902, 5903, 5904, 5905, 5906, 5907, 5908, 5909, 5910, 5938, 5939, 5950, 5985, 5986, 6030, 6040, 6129, 6130, 6132, 6133, 6568, 6666, 6667, 6722, 6783, 6784, 6785, 6881, 6882, 6883, 6884, 6885, 6886, 6887, 6888, 6889, 6890, 6891, 6892, 6893, 6894, 6895, 6896, 6897, 6898, 6899, 7070, 7096, 7443, 7444, 7448, 7474, 7681, 7682, 7687, 7712, 7777, 7844, 8022, 8040, 8041, 8080, 8081, 8111, 8118, 8200, 8384, 8443, 8531, 8848, 8888, 8936, 8999, 9001, 9050, 9051, 9090, 9200, 9235, 9631, 9931, 9936, 9966, 9984, 9988, 9999, 10002, 10110, 10426, 10666, 12122, 12345, 12346, 13333, 15000, 15555, 16990, 16991, 16992, 16993, 17300, 19999, 20034, 20778, 21115, 21116, 21802, 22000, 22001, 22543, 25405, 27374, 30662, 31335, 31337, 31338, 31785, 31789, 32226, 32227, 32233, 32234, 35000, 48101, 50001, 50002, 50003, 50050, 50501, 51820, 52935, 53531, 54320, 55553, 57230, 59074, 59076, 61466, 65000}
        for ip_entry in ip_entries:
            dst_port = ip_entry.get("destination_port") or ip_entry.get("dst_port")
            protocol = lower(ip_entry.get("protocol", ""))
            if dst_port is not None:
                try:
                    port = int(dst_port)
                    if port in suspicious_ports:
                        explicit[f"ip_suspicious_port::{port}"] += 1.0
                    if port < 1024:
                        explicit["ip_well_known_port"] += 1.0
                    else:
                        explicit["ip_high_port"] += 1.0
                except (ValueError, TypeError):
                    pass
            if protocol:
                explicit[f"ip_protocol::{protocol}"] += 1.0

        for h in http_entries:
            host = h["host"]
            reg = h["reg_domain"]
            if host and not (self.config.drop_env_hosts and (host in ENV_HOSTS or reg in ENV_REG_DOMAINS)):
                hash_domains.append(f"http_host::{host}")
                hash_domains.append(f"http_regdom::{reg}")
            if h["path"]:
                for seg in [x for x in h["path"].split("/") if x][:4]:
                    hash_strings.append(f"urlseg::{INT_RE.sub('<num>', seg.lower())}")
            for qk in h["query_keys"][:12]:
                hash_strings.append(f"qk::{lower(qk)}")
            if h["method"]:
                explicit[f"http_method::{h['method']}"] += 1.0
            if h["scheme"]:
                explicit[f"http_scheme::{h['scheme']}"] += 1.0
            if h["status_code"] is not None:
                try:
                    sc = int(h["status_code"])
                except (ValueError, TypeError):
                    sc = -1
                if 100 <= sc < 200:
                    explicit["http_status::1xx"] += 1.0
                elif 200 <= sc < 300:
                    explicit["http_status::2xx"] += 1.0
                elif 300 <= sc < 400:
                    explicit["http_status::3xx"] += 1.0
                elif 400 <= sc < 500:
                    explicit["http_status::4xx"] += 1.0
                elif 500 <= sc < 600:
                    explicit["http_status::5xx"] += 1.0

        for d in dns_entries:
            host = normalize_domain(d.get("hostname", ""))
            reg = approx_reg_domain(host)
            if host and not (self.config.drop_env_hosts and (host in ENV_HOSTS or reg in ENV_REG_DOMAINS)):
                hash_domains.append(f"dns_host::{host}")
                hash_domains.append(f"dns_regdom::{reg}")

        for t in tls_entries:
            sni = normalize_domain(t.get("sni", ""))
            reg = approx_reg_domain(sni)
            if sni and not (self.config.drop_env_hosts and (sni in ENV_HOSTS or reg in ENV_REG_DOMAINS)):
                hash_domains.append(f"tls_sni::{sni}")
                hash_domains.append(f"tls_regdom::{reg}")
            ver = lower(t.get("version", ""))
            if ver:
                explicit[f"tls_version::{ver}"] += 1.0
            ja3 = lower(t.get("ja3", ""))
            if ja3:
                hash_strings.append(f"ja3::{ja3}")

        for j in ensure_list(data.get("ja3_digests")):
            jj = lower(j)
            if jj:
                hash_strings.append(f"ja3digest::{jj}")

        for prefix, paths in {
            "open": files_opened,
            "write": files_written,
            "delete": files_deleted,
            "drop": files_dropped,
            "dbopen": db_opened,
        }.items():
            for p in paths:
                hash_paths.append(f"{prefix}_bucket::{path_bucket(p)}")
                hash_paths.append(f"{prefix}_path::{normalize_path(p)}")

        for item in files_copied:
            src = as_str(item.get("source", ""))
            dst = as_str(item.get("destination", ""))
            if src:
                hash_paths.append(f"copy_src::{path_bucket(src)}")
            if dst:
                hash_paths.append(f"copy_dst::{path_bucket(dst)}")

        for item in files_attr:
            p = as_str(item.get("path", ""))
            if p:
                hash_paths.append(f"attr::{path_bucket(p)}")

        for family, comps in {
            "service_started": services_started,
            "service_opened": services_opened,
            "service_bound": services_bound,
            "service_stopped": services_stopped,
            "activity_started": activities_started,
        }.items():
            for c in comps:
                if c["kind"]:
                    explicit[f"component_kind::{family}::{c['kind']}"] += 1.0
                if c["package"]:
                    hash_components.append(f"{family}_pkg::{sanitize_short_token(c['package'])}")
                if c["class"]:
                    cls_tail = sanitize_short_token(c["class"].split(".")[-1])
                    hash_components.append(f"{family}_cls::{cls_tail}")

                action_raw = c.get("action", "")
                if action_raw:
                    action_explicit = normalize_component_action_for_explicit(action_raw)
                    action_hash = normalize_component_action_for_hash(action_raw)

                    if action_explicit:
                        explicit[f"component_action_base::{family}::{action_explicit}"] += 1.0
                    else:
                        explicit[f"component_action_base::{family}::other"] += 1.0

                    hash_components.append(f"{family}_action_raw::{action_hash}")

                    if looks_payload_heavy(action_raw):
                        explicit[f"component_action_has_payload::{family}"] += 1.0

                    extra_count = action_raw.count("android.intent.extra.")
                    if extra_count > 0:
                        explicit[f"component_extra_count_bucket::{family}::{bucket_count(extra_count)}"] += 1.0

                    for extra_key in sorted(set(re.findall(r"android\.intent\.extra\.[A-Za-z0-9_.]+", action_raw))):
                        hash_components.append(f"{family}_extra_key::{lower(extra_key)}")

                    for scheme in sorted(set(re.findall(r"\b([a-zA-Z][a-zA-Z0-9+.-]*)://", action_raw))):
                        explicit[f"component_uri_scheme::{family}::{lower(scheme)}"] += 1.0

                    if "mailto:" in lower(action_raw):
                        explicit[f"component_has_mailto::{family}"] += 1.0
                    if "content://" in lower(action_raw):
                        explicit[f"component_has_content_uri::{family}"] += 1.0
                    if "file://" in lower(action_raw):
                        explicit[f"component_has_file_uri::{family}"] += 1.0
                    if "http://" in lower(action_raw) or "https://" in lower(action_raw):
                        explicit[f"component_has_http_uri::{family}"] += 1.0
                    if "market://" in lower(action_raw):
                        explicit[f"component_has_market_uri::{family}"] += 1.0
                    if "android.intent.action.send" in lower(action_raw):
                        explicit[f"component_has_send_payload::{family}"] += 1.0

                    for domain in sorted(set(re.findall(r"https?://([^/\s\]\}]+)", action_raw, flags=re.IGNORECASE))):
                        host = normalize_domain(domain)
                        reg = approx_reg_domain(host)
                        if host and not (self.config.drop_env_hosts and (host in ENV_HOSTS or reg in ENV_REG_DOMAINS)):
                            hash_domains.append(f"component_host::{host}")
                            hash_domains.append(f"component_regdom::{reg}")

        for s in calls_highlighted:
            hash_strings.append(f"api::{s}")
        for s in invokes:
            hash_strings.append(f"invoke::{s}")
        for s in commands:
            hash_strings.append(f"cmd::{lower(s)}")
        for s in memory_domains:
            reg = approx_reg_domain(s)
            hash_domains.append(f"mem_host::{s}")
            hash_domains.append(f"mem_regdom::{reg}")
        for s in memory_urls:
            parsed = parse_http_entry({"url": s})
            if parsed["host"]:
                hash_domains.append(f"mem_url_host::{parsed['host']}")
                hash_domains.append(f"mem_url_regdom::{parsed['reg_domain']}")
        for x in shared_prefs:
            k = lower(x.get("key", ""))
            if k:
                hash_strings.append(f"pref_key::{k}")
        for x in content_sets:
            k = lower(x.get("key", ""))
            if k:
                hash_strings.append(f"content_key::{k}")
        for x in modules_loaded:
            hash_strings.append(f"module::{as_str(x).split('/')[-1]}")
        for x in mutexes:
            hash_strings.append(f"mutex::{lower(x)}")
        for x in proc_created:
            hash_strings.append(f"proc_create::{lower(x)}")
        for x in proc_killed:
            hash_strings.append(f"proc_kill::{lower(x)}")
        for x in reg_keys:
            hash_strings.append(f"reg::{lower(x)}")

        for text in (text_highlighted + text_decoded)[:64]:
            s = lower(text)
            s = UUID_RE.sub("<uuid>", s)
            s = HEX_RE.sub("<hex>", s)
            s = INT_RE.sub("<num>", s)
            s = PKG_RE.sub("<pkg>", s)
            toks = re.split(r"[^a-z0-9_./:-]+", s)
            kept = 0
            for tok in toks:
                if 2 <= len(tok) <= 32:
                    hash_strings.append(f"txt::{tok}")
                    kept += 1
                    if kept >= self.config.max_text_tokens:
                        break

        return {
            "explicit": dict(explicit),
            "hash_domains": hash_domains,
            "hash_paths": hash_paths,
            "hash_components": hash_components,
            "hash_strings": hash_strings,
        }


# -----------------------------------------------------------------------------
# Dataset I/O and split manifest
# -----------------------------------------------------------------------------

def collect_year_files(dataset_root: Path, year: int) -> Tuple[List[Path], np.ndarray]:
    paths: List[Path] = []
    labels: List[int] = []

    for label_name, y in [("1", 1), ("0", 0)]:
        folder = dataset_root / str(year) / label_name
        files = sorted(p for p in folder.rglob("*.json") if p.is_file())
        paths.extend(files)
        labels.extend([y] * len(files))

    return paths, np.array(labels, dtype=np.int64)


def load_samples(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in paths:
        try:
            out.append(load_json(p))
        except Exception as e:
            logger.warning("Failed to load %s: %s — inserting empty sample", p, e)
            out.append({})
    return out


def get_hashes_from_paths(paths: Sequence[Path]) -> List[str]:
    return [p.stem for p in paths]


def get_paths_as_str(paths: Sequence[Path]) -> List[str]:
    return [str(p) for p in paths]


def load_split_manifest(split_manifest_path: Path):
    manifest = load_json(split_manifest_path)

    train_items = manifest.get("train", [])
    test_items = manifest.get("test", [])
    if not train_items or not test_items:
        raise ValueError("Split manifest must contain non-empty 'train' and 'test' lists.")

    train_hashes = [item["hash"] for item in train_items]
    test_hashes = [item["hash"] for item in test_items]
    label_by_hash = {}

    for item in train_items + test_items:
        h = item["hash"]
        y = int(item["y"])
        old = label_by_hash.get(h)
        if old is not None and old != y:
            raise ValueError(f"Conflicting labels in split manifest for hash={h}: {old} vs {y}")
        label_by_hash[h] = y

    if len(set(train_hashes) & set(test_hashes)) != 0:
        raise ValueError("Split manifest is invalid: overlap detected between train and test hashes.")

    return train_hashes, test_hashes, label_by_hash


def split_year_from_manifest(year_paths: Sequence[Path], y: np.ndarray, split_manifest_path: Path):
    hashes = get_hashes_from_paths(year_paths)
    idx_by_hash = {}
    for i, h in enumerate(hashes):
        if h in idx_by_hash:
            raise ValueError(f"Duplicate hash found in dataset: {h}")
        idx_by_hash[h] = i

    train_hashes, test_hashes, label_by_hash = load_split_manifest(split_manifest_path)

    missing_train = [h for h in train_hashes if h not in idx_by_hash]
    missing_test = [h for h in test_hashes if h not in idx_by_hash]
    if missing_train or missing_test:
        msg = (
            f"Split manifest contains hashes missing from this dataset. "
            f"missing_train={len(missing_train)}, missing_test={len(missing_test)}"
        )
        if missing_train:
            msg += f"\nFirst missing train hashes: {missing_train[:10]}"
        if missing_test:
            msg += f"\nFirst missing test hashes: {missing_test[:10]}"
        raise ValueError(msg)

    train_idx = np.asarray([idx_by_hash[h] for h in train_hashes], dtype=np.int64)
    test_idx = np.asarray([idx_by_hash[h] for h in test_hashes], dtype=np.int64)

    for split_name, split_hashes, split_idx in (("train", train_hashes, train_idx), ("test", test_hashes, test_idx)):
        for expected_hash, idx in zip(split_hashes, split_idx):
            actual_hash = hashes[idx]
            actual_label = int(y[idx])
            expected_label = int(label_by_hash[expected_hash])
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Row ordering error for {split_name}: expected hash {expected_hash}, got {actual_hash}"
                )
            if actual_label != expected_label:
                raise ValueError(
                    f"Label mismatch for hash={expected_hash}: dataset={actual_label}, manifest={expected_label}"
                )

    train_paths = [year_paths[i] for i in train_idx]
    test_paths = [year_paths[i] for i in test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    return train_paths, y_train, test_paths, y_test


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------

def save_feature_space(
    feature_dir: Path,
    extractor: Frozen2013FeatureExtractor,
    feature_names: List[str],
    fit_year: int,
    split_manifest_path: Path,
) -> None:
    feature_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(extractor, feature_dir / "extractor_2013.joblib")
    save_json(feature_dir / "feature_names.json", feature_names)
    save_json(
        feature_dir / "meta.json",
        {
            "fit_year": fit_year,
            "dataset_root": str(DATASET_ROOT),
            "n_features": len(feature_names),
            "split_manifest_path": str(split_manifest_path),
            "explicit_feature_count": len(extractor.vectorizer.get_feature_names_out()),
            "residual_explicit_hashing": False,
            "component_action_strategy": "coarse_explicit_plus_normalized_hash",
        },
    )


def copy_feature_space(init_dir: Path, out_dir: Path) -> None:
    src = init_dir / "feature_space"
    dst = out_dir / "feature_space"
    if not src.exists():
        raise FileNotFoundError(f"Missing feature_space in init-dir: {src}")
    if not dst.exists():
        shutil.copytree(src, dst)


def run_initializer(year: int, out_dir: Path, split_manifest_path: Path, workers: int) -> None:
    if workers != 1:
        logger.warning("--workers=%d is accepted for compatibility but unused.", workers)

    year_paths, y = collect_year_files(DATASET_ROOT, year)
    if not year_paths:
        raise RuntimeError(f"No JSON files found for year {year} under {DATASET_ROOT}")

    train_paths, y_train, test_paths, y_test = split_year_from_manifest(year_paths, y, split_manifest_path)
    train_samples = load_samples(train_paths)
    test_samples = load_samples(test_paths)

    extractor = Frozen2013FeatureExtractor()
    extractor.fit(train_samples)
    train_transformed = extractor.transform(train_samples)
    test_transformed = extractor.transform(test_samples)

    X_train = train_transformed["X_full"]
    X_test = test_transformed["X_full"]
    feature_names = train_transformed["feature_names"]

    train_hashes = get_hashes_from_paths(train_paths)
    test_hashes = get_hashes_from_paths(test_paths)
    train_json_paths = get_paths_as_str(train_paths)
    test_json_paths = get_paths_as_str(test_paths)

    feature_dir = out_dir / "feature_space"
    year_dir = out_dir / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    save_feature_space(feature_dir, extractor, feature_names, fit_year=year, split_manifest_path=split_manifest_path)

    save_npz(year_dir / "train_X.npz", X_train)
    save_meta_npz(year_dir / "train_meta.npz", y=y_train, hashes=train_hashes, paths=train_json_paths)
    save_npz(year_dir / "test_X.npz", X_test)
    save_meta_npz(year_dir / "test_meta.npz", y=y_test, hashes=test_hashes, paths=test_json_paths)

    logger.info(
        "Initializer complete: year=%d, train_samples=%d, test_samples=%d, features=%d",
        year,
        X_train.shape[0],
        X_test.shape[0],
        X_train.shape[1],
    )


def run_adaptation(year: int, init_dir: Path, out_dir: Path, split_manifest_path: Path, workers: int) -> None:
    if workers != 1:
        logger.warning("--workers=%d is accepted for compatibility but unused.", workers)

    extractor_path = init_dir / "feature_space" / "extractor_2013.joblib"
    if not extractor_path.exists():
        raise FileNotFoundError(f"Missing extractor: {extractor_path}")

    year_paths, y = collect_year_files(DATASET_ROOT, year)
    if not year_paths:
        raise RuntimeError(f"No JSON files found for year {year} under {DATASET_ROOT}")

    train_paths, y_train, test_paths, y_test = split_year_from_manifest(year_paths, y, split_manifest_path)
    train_samples = load_samples(train_paths)
    test_samples = load_samples(test_paths)

    extractor: Frozen2013FeatureExtractor = joblib.load(extractor_path)
    train_transformed = extractor.transform(train_samples)
    test_transformed = extractor.transform(test_samples)

    X_train = train_transformed["X_full"]
    X_test = test_transformed["X_full"]
    train_hashes = get_hashes_from_paths(train_paths)
    test_hashes = get_hashes_from_paths(test_paths)
    train_json_paths = get_paths_as_str(train_paths)
    test_json_paths = get_paths_as_str(test_paths)

    copy_feature_space(init_dir, out_dir)

    year_dir = out_dir / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    save_npz(year_dir / "train_X.npz", X_train)
    save_meta_npz(year_dir / "train_meta.npz", y=y_train, hashes=train_hashes, paths=train_json_paths)
    save_npz(year_dir / "test_X.npz", X_test)
    save_meta_npz(year_dir / "test_meta.npz", y=y_test, hashes=test_hashes, paths=test_json_paths)

    logger.info(
        "Adaptation complete: year=%d, train_samples=%d, test_samples=%d, features=%d",
        year,
        X_train.shape[0],
        X_test.shape[0],
        X_train.shape[1],
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Refined frozen feature-space data builder")
    sub = ap.add_subparsers(dest="command", required=True)

    ap_init = sub.add_parser("initializer", help="Fit extractor and write base-year train/test data")
    ap_init.add_argument("--year", type=int, required=True)
    ap_init.add_argument("--out-dir", required=True)
    ap_init.add_argument("--split-manifest-path", type=Path, required=True)
    ap_init.add_argument("--workers", type=int, default=1)

    ap_adapt = sub.add_parser("adaptation", help="Load frozen extractor and write train/test data for a later year")
    ap_adapt.add_argument("--year", type=int, required=True)
    ap_adapt.add_argument("--init-dir", required=True)
    ap_adapt.add_argument("--out-dir", required=True)
    ap_adapt.add_argument("--split-manifest-path", type=Path, required=True)
    ap_adapt.add_argument("--workers", type=int, default=1)

    args = ap.parse_args()

    if args.command == "initializer":
        run_initializer(
            year=args.year,
            out_dir=Path(args.out_dir),
            split_manifest_path=args.split_manifest_path,
            workers=args.workers,
        )
    elif args.command == "adaptation":
        run_adaptation(
            year=args.year,
            init_dir=Path(args.init_dir),
            out_dir=Path(args.out_dir),
            split_manifest_path=args.split_manifest_path,
            workers=args.workers,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
