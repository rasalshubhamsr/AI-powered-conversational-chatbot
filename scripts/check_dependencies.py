#!/usr/bin/env python
"""
Quick dependency check script.
- Compares installed package versions with constraints in backend/requirements.txt
- Reports: OK, MISSING, or VERSION MISMATCH
- Exits with nonzero code if any issues are found

Usage:
  python scripts/check_dependencies.py
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from importlib.metadata import version, PackageNotFoundError
except Exception:  # pragma: no cover
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

try:
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version
except Exception:
    print("Error: packaging module is required. Install with: pip install packaging", file=sys.stderr)
    sys.exit(2)

REQ_FILE = os.path.join("backend", "requirements.txt")

@dataclass
class Requirement:
    name: str
    spec: Optional[SpecifierSet]


def parse_requirement_line(line: str) -> Optional[Requirement]:
    # Remove comments and whitespace
    line = line.strip()
    if not line or line.startswith('#'):
        return None

    # Skip direct URLs or local paths for simplicity
    if line.startswith(('git+', 'http:', 'https:', 'file:', '-e ')):
        return None

    # Remove extras like [standard]
    name_part = re.split(r"[<>=!~]", line, maxsplit=1)[0]
    name_part = name_part.split("[")[0]
    name_part = name_part.strip()

    # Extract specifier if present
    spec_str = None
    m = re.search(r"([<>=!~].*)", line)
    if m:
        spec_str = m.group(1).strip()

    spec = SpecifierSet(spec_str) if spec_str else None
    return Requirement(name=name_part, spec=spec)


def load_requirements(path: str) -> List[Requirement]:
    reqs: List[Requirement] = []
    if not os.path.exists(path):
        print(f"Requirements file not found: {path}", file=sys.stderr)
        return reqs
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            req = parse_requirement_line(line)
            if req:
                reqs.append(req)
    return reqs


def check_requirements(reqs: List[Requirement]) -> int:
    problems = 0
    print(f"Checking dependencies against {REQ_FILE}\n")
    for req in reqs:
        try:
            installed_ver = Version(version(req.name))
        except PackageNotFoundError:
            print(f"MISSING          {req.name}")
            problems += 1
            continue
        except Exception as e:
            print(f"ERROR            {req.name}: {e}")
            problems += 1
            continue

        status = "OK"
        if req.spec and installed_ver not in req.spec:
            status = "VERSION MISMATCH"
            problems += 1

        spec_str = f" {req.spec}" if req.spec else ""
        print(f"{status:<16} {req.name}=={installed_ver}{spec_str}")

    return problems


def main() -> int:
    reqs = load_requirements(REQ_FILE)
    if not reqs:
        print("No requirements parsed or file missing.")
        return 1
    problems = check_requirements(reqs)
    if problems:
        print(f"\nFound {problems} issue(s). Consider running: pip install --upgrade -r {REQ_FILE}")
        return 1
    print("\nAll dependencies satisfy specified constraints.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
