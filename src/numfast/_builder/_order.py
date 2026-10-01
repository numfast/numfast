# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Resolver: deterministic topological order (Kahn, alphabetical tiebreak)."""


def resolve_order(extensions):
    if not extensions:
        return []
    name_to_ext = {}
    deps_map = {}
    for ext in extensions:
        name = ext.get("name", "")
        if not name:
            continue
        name_to_ext[name] = ext
        deps_map[name] = list(ext.get("depends", []))
    all_names = set(deps_map)
    for name, deps in deps_map.items():
        for dep in deps:
            if dep not in all_names:
                raise RuntimeError(
                    f"Extension '{name}' depends on '{dep}', "
                    "but '{dep}' is not in the extension list"
                )
    in_degree = {n: len(d) for n, d in deps_map.items()}
    dependents = {n: [] for n in deps_map}
    for name, deps in deps_map.items():
        for dep in deps:
            dependents[dep].append(name)
    queue = sorted(n for n, deg in in_degree.items() if deg == 0)
    ordered = []
    while queue:
        name = queue.pop(0)
        ordered.append(name)
        for dep in dependents[name]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)
        queue.sort()
    if len(ordered) != len(deps_map):
        raise RuntimeError(
            f"Circular dependency detected: {set(deps_map) - set(ordered)}"
        )
    return [name_to_ext[n] for n in ordered]
