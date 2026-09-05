"""Small, typed readers for environment-backed configuration."""

from __future__ import annotations

import os


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def env_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    return None if not value else int(value)


def env_optional_float(name: str) -> float | None:
    value = os.getenv(name)
    return None if not value else float(value)


def env_optional_str(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    return value.strip() or None


def env_tuple(name: str) -> tuple[str, ...]:
    value = os.getenv(name)
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def env_int_tuple(name: str) -> tuple[int, ...]:
    return tuple(int(value) for value in env_tuple(name))
