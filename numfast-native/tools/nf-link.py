"""nf-link: cargo/rustc -> zig cc (windows-gnu). New shim (correct paths).

Drops `-Wl,*.def` file args (zig 0.16 cc rejects them as `-Wl,` input);
GNU ld default exports all globals from shared objects, so cdylib
symbols stay exported. Drops `-l:libpthread.a` colon form (segfaults
zig 0.16) plus `-nodefaultlibs -Wl,-Bstatic -Wl,-Bdynamic -lmsvcrt`
(proven single-threaded link set; MT uses std threads via system libs
kept below). Everything else passes through untouched.
"""
import subprocess
import sys

ZIG = r"C:\Users\Mikech\AppData\Local\Temp\nfzig\zigfull\zig-x86_64-windows-0.16.0\zig.exe"
TARGET = "x86_64-windows-gnu"


def _filtered(argv):
    fwd = []
    dropped = []
    for a in argv:
        if a.startswith("-Wl,") and a[4:].endswith(".def"):
            dropped.append(a)
            continue
        if a in ("-nodefaultlibs", "-Wl,-Bstatic", "-Wl,-Bdynamic", "-lmsvcrt",
                 "-l:libpthread.a"):
            dropped.append(a)
            continue
        fwd.append(a)
    return fwd, dropped


def main() -> int:
    fwd, dropped = _filtered(sys.argv[1:])
    cmd = [ZIG, "cc", "-target", TARGET] + fwd
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if dropped:
        sys.stderr.write("[nf-link] dropped: %s\n" % dropped)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
