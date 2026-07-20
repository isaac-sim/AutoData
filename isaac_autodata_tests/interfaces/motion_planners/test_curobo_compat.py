# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from isaac_autodata_interfaces.motion_planners.curobo.compat import (
    CUROBO_V1_MARKERS,
    CUROBO_V2_MARKERS,
    SCHEDULESTREAM_V1_MARKERS,
    SCHEDULESTREAM_V2_MARKERS,
    BackendCompatibilityError,
    CuroboApiGeneration,
    DistributionIdentity,
    _sanitize_source_url,
    detect_curobo_runtime,
    select_schedulestream_backend,
)


def _detect(available: set[str]):
    return detect_curobo_runtime(
        module_available=lambda name: name in available,
        distribution_reader=lambda names: DistributionIdentity(names[0], "test"),
    )


def test_detects_curobo_v1_and_custream():
    capabilities = _detect({"curobo", "schedulestream", *CUROBO_V1_MARKERS, *SCHEDULESTREAM_V1_MARKERS})

    assert capabilities.api_generation is CuroboApiGeneration.V1
    selection = select_schedulestream_backend("auto", capabilities)
    assert selection.motion_backend == "curobo_v1"
    assert selection.schedulestream_application == "custream"


def test_detects_curobo_v2_and_custream2():
    capabilities = _detect({"curobo", "schedulestream", *CUROBO_V2_MARKERS, *SCHEDULESTREAM_V2_MARKERS})

    assert capabilities.api_generation is CuroboApiGeneration.V2
    selection = select_schedulestream_backend("curobo_v2", capabilities)
    assert selection.schedulestream_application == "custream2"


def test_v2_markers_win_when_compatibility_v1_modules_are_also_present():
    capabilities = _detect({
        "curobo",
        "schedulestream",
        *CUROBO_V1_MARKERS,
        *CUROBO_V2_MARKERS,
        *SCHEDULESTREAM_V1_MARKERS,
        *SCHEDULESTREAM_V2_MARKERS,
    })

    assert capabilities.api_generation is CuroboApiGeneration.V2
    assert capabilities.schedulestream_application == "custream2"


def test_rejects_requested_api_that_does_not_match_runtime():
    capabilities = _detect({"curobo", "schedulestream", *CUROBO_V1_MARKERS, *SCHEDULESTREAM_V1_MARKERS})

    with pytest.raises(BackendCompatibilityError, match="provides curobo_v1"):
        select_schedulestream_backend("curobo_v2", capabilities)


def test_reports_missing_matching_schedulestream_application():
    capabilities = _detect({"curobo", *CUROBO_V1_MARKERS})

    with pytest.raises(BackendCompatibilityError, match="custream"):
        select_schedulestream_backend("auto", capabilities)


def test_reports_unavailable_curobo_without_importing_it():
    capabilities = _detect(set())

    assert capabilities.api_generation is CuroboApiGeneration.UNAVAILABLE
    with pytest.raises(BackendCompatibilityError, match="No supported cuRobo API"):
        select_schedulestream_backend("auto", capabilities)


def test_current_host_probe_is_serializable():
    # This is deliberately not an acceptance assertion about which backend the developer has.
    # It verifies that real filesystem/metadata probing is safe and produces runtime identity data.
    capabilities = detect_curobo_runtime()
    payload = capabilities.to_dict()

    assert payload["api_generation"] in {item.value for item in CuroboApiGeneration}
    assert "missing_markers" in payload


@pytest.mark.parametrize(
    ("raw_url", "expected"),
    [
        (
            "https://robot:secret@example.com:8443/org/repo.git?token=private#fragment",
            "https://example.com:8443/org/repo.git",
        ),
        ("file:///home/engineer/private/schedulestream", "file:///<local-source>"),
        ("/home/engineer/private/schedulestream", None),
        ("https://example.com/repo.git\nAuthorization: secret", None),
    ],
)
def test_direct_install_url_is_sanitized_before_runtime_identity(raw_url: str, expected: str | None) -> None:
    assert _sanitize_source_url(raw_url) == expected
