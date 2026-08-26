# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.0.0a1 | :white_check_mark: |

## Reporting a Vulnerability

Please do **not** report security vulnerabilities through public GitHub issues.

Instead, report them privately:

- **GitHub:** use [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability) on this repository
- **Email:** security@numfast.example (placeholder — replace before release)

Include as much of the following as you can:

- Type of issue (e.g. buffer overflow, injection, race condition)
- Full paths of source file(s) related to the manifestation of the issue
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

You will receive a response within 7 days. If the issue is confirmed, we will
release a patch as soon as possible depending on complexity.

## Scope Notes

NumFast executes user-supplied WGSL compute shaders and loads `.csv.zst` /
`.nfa` data files locally. Reports concerning local file parsing, shader
compilation, and GPU driver interaction are especially welcome.
