# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.2.1 | yes |
| older  | no |

## Reporting a vulnerability

Please do **not** report a security vulnerability through a public GitHub issue.

- **Preferred:** use [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
  on this repository. It reaches the maintainers without publishing anything,
  and it needs no mailbox to be configured first.

  No email address is published for this repository. That is deliberate: a
  stale or placeholder address in a security policy is worse than none, because
  a reporter who trusts it waits for a reply that never comes. Use the GitHub
  route above.

Please include as much of the following as you can:

- the type of issue (for example: buffer overflow, out-of-bounds read, integer
  overflow, race condition, path traversal);
- the full paths of the source files involved;
- where in the code the issue lives — tag, branch, commit, or a direct URL;
- step-by-step reproduction;
- proof-of-concept or exploit code, if you have it;
- the impact, and how an attacker might use it.

You can expect an acknowledgement within 7 days. If a report is confirmed, a fix
will be released as soon as its complexity allows, and the reporter will be named
in the release notes unless they ask not to be.

## Scope notes

What an attacker can actually reach in this project:

- **The Rust FFI boundary has no bounds checking.** `numfast-native/src/core/buffers.rs`
  builds slices with `std::slice::from_raw_parts{,_mut}` from caller-supplied
  pointers and lengths. Every length, capacity and range is enforced by the
  caller, not by the kernel. A wrong length is undefined behaviour in the native
  build. This is recorded as hazard H3 in
  [`numfast-native/abi/census.json`](numfast-native/abi/census.json), with the
  measured consequence on wasm32 (a trap, or silent corruption when the address
  stays inside linear memory).
- **The WebAssembly build has no bounds check either**, and eleven of its
  exports call `std::thread::scope`, which `wasm32-unknown-unknown` does not
  support. Those eleven trap on every non-empty call.
- **NumFast parses columnar data.** `Table`, `open_stream` and the NFS reader
  consume files you point them at. Reports about local file parsing, the
  dictionary coder and the storage encodings are especially welcome.
- **The GPU path compiles WGSL at run time** through `wgpu-py`. Reports about
  shader compilation and driver interaction are in scope.

What is **out of scope**: NumFast is a library, not a network service. It opens no
listening socket, and it does not fetch anything on your behalf. A report that
depends on an attacker-controlled server is not a NumFast vulnerability.

## Disclosure

There is no bug-bounty programme and no paid support contract attached to
vulnerability reporting. Commercial support and consulting are separate from it
and are described in [COMMERCIAL-LICENCE.md](COMMERCIAL-LICENCE.md).