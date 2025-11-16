#!/usr/bin/env python3
"""Check latest versions of all packages in requirements.txt"""
import subprocess
import re
import sys

packages = [
    "fastapi",
    "uvicorn",
    "python-multipart",
    "neo4j",
    "asyncpg",
    "sqlalchemy",
    "redis",
    "python-dotenv",
    "pydantic-settings",
    "sentence-transformers",
    "litellm",
    "arq",
    "python-jose",
    "passlib",
    "numpy",
    "pytest",
    "pytest-asyncio",
    "httpx",
]

print("Checking latest versions...")
print("=" * 60)

latest_versions = {}

for package in packages:
    try:
        # Use pip index to get latest version
        result = subprocess.run(
            ["pip", "index", "versions", package],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # Extract version from output (format: "package (version)")
            match = re.search(rf"{re.escape(package)}\s+\(([\d.]+)\)", result.stdout)
            if match:
                latest_versions[package] = match.group(1)
                print(f"{package:25} {match.group(1)}")
            else:
                print(f"{package:25} Could not parse")
        else:
            print(f"{package:25} Error checking")
    except Exception as e:
        print(f"{package:25} Error: {e}")

print("\n" + "=" * 60)
print("\nLatest versions found:")
for pkg, ver in latest_versions.items():
    print(f"{pkg}=={ver}")
