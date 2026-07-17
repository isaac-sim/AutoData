#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac AutoData Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Print a human-readable summary and table of a JUnit XML test report.

Used by the nightly workflow to surface pass/fail counts and failed-test details
directly in the run log (its own step, so it is easy to find), without needing to
download the XML artifact. Always exits 0: the test step already reflects the
overall pass/fail status; this output is informational.

Usage: summarize_test_results.py [path-to-junit.xml]
"""

import math
import sys
import xml.etree.ElementTree as ET

TEST_COL = 70
MSG_COL = 60


def first_line(text: str | None) -> str:
    """Return the first non-blank line of a string.

    Args:
        text: The text to read, or None.

    Returns:
        The first non-blank line, or an empty string if there is none.
    """
    if not text:
        return ""
    stripped = text.strip()
    return stripped.splitlines()[0] if stripped else ""


def truncate(text: str, width: int) -> str:
    """Clip text to a maximum width.

    Args:
        text: The text to clip.
        width: The maximum number of characters to keep.

    Returns:
        The text unchanged if within width, otherwise clipped to width
        characters with a trailing ellipsis.
    """
    assert width >= 0, "width must be non-negative"
    if width < 3:
        return text[:width]
    return text if len(text) <= width else text[: width - 3] + "..."


def parse_time(value: str | None) -> float:
    """Parse a testcase duration, tolerating missing or malformed values.

    Args:
        value: The ``time`` attribute of a testcase, or None.

    Returns:
        The duration in seconds, or 0.0 when it is missing, non-numeric,
        non-finite, or negative.
    """
    try:
        seconds = float(value) if value else 0.0
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(seconds) or seconds < 0:
        return 0.0
    return seconds


def main() -> int:
    """Print a summary and per-test table for a JUnit XML report.

    Returns:
        0 always; the output is informational and never fails the job.
    """
    path = sys.argv[1] if len(sys.argv) > 1 else "test-results/junit.xml"

    try:
        root = ET.parse(path).getroot()
    except FileNotFoundError:
        print(f">>> No test report found at {path}; nothing to summarize.")
        return 0
    except OSError as exc:
        print(f">>> Could not read test report {path}: {exc}")
        return 0
    except ET.ParseError as exc:
        print(f">>> Could not parse test report {path}: {exc}")
        return 0

    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")

    rows = []
    passed = failed = errors = skipped = 0
    total_time = 0.0

    for suite in suites:
        for case in suite.findall("testcase"):
            name = case.get("name", "")
            classname = case.get("classname", "")
            full = f"{classname}::{name}" if classname else name
            duration = parse_time(case.get("time"))
            total_time += duration

            failure = case.find("failure")
            error = case.find("error")
            skip = case.find("skipped")

            if error is not None:
                status = "ERROR"
                message = error.get("message") or first_line(error.text)
                errors += 1
            elif failure is not None:
                status = "FAIL"
                message = failure.get("message") or first_line(failure.text)
                failed += 1
            elif skip is not None:
                status = "SKIP"
                message = skip.get("message") or ""
                skipped += 1
            else:
                status = "PASS"
                message = ""
                passed += 1

            rows.append((status, full, duration, first_line(message)))

    total = passed + failed + errors + skipped

    print("=" * 78)
    print("Nightly test summary")
    print("=" * 78)
    print(
        f"Total {total}  |  Passed {passed}  |  Failed {failed}  |  "
        f"Errors {errors}  |  Skipped {skipped}  |  Time {total_time:.1f}s"
    )
    print()
    print(f"| {'Result':<6} | {'Test':<{TEST_COL}} | {'Time(s)':>8} | Message |")
    print(f"| {'-' * 6} | {'-' * TEST_COL} | {'-' * 8} | {'-' * MSG_COL} |")
    for status, full, duration, message in rows:
        print(
            f"| {status:<6} | {truncate(full, TEST_COL):<{TEST_COL}} "
            f"| {duration:>8.1f} | {truncate(message, MSG_COL)} |"
        )

    if failed or errors:
        print()
        print(">>> See the run log above or the nightly-test-results artifact for full tracebacks.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
