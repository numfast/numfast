from _main._lib.main_lib import _get_version, _get_status

def info() -> str:
    version = _get_version()
    return f"NumFast v{version}"

def status() -> dict:
    ext_status = _get_status()
    return ext_status
