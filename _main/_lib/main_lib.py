from importlib.metadata import version, PackageNotFoundError

def _get_version() -> str:
    try:
        return version("numfast")
    except PackageNotFoundError:
        return "0.0.0-dev"

def _get_status() -> dict:
    return {
        "name": "numfast",
        "version": _get_version(),
        "extensions": [
            "_main",
            "Series",
            "Tables",
            "Memory",
            "Mods",
            "Proxy",
        ],
    }
