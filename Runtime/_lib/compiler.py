"""Compiler — jobs → tasks.

Фаза Compilation:
1. Для каждого job найти kernel по alias в kernel_table
2. Развернуть массивы params → cartesian product
3. Для каждой комбинации спросить describe() → ExecutionPlan
4. Сгенерировать имена выходов через шаблоны
5. Построить временную SymbolTable
6. Преобразовать строки inputs → ResourceRef
7. Заполнить task.workspace из плана
8. Вернуть tasks (ExecutionPlan и SymbolTable уничтожены)
"""

from typing import Any
from .task import Task, ResourceRef
from .mod_iface import ExecutionPlan


def _cartesian_product(params: dict) -> list[dict]:
    """Развернуть params с массивами в список конкретных комбинаций.
    
    Пример:
        {"period": [5, 20], "fast": 12}
        → [{"period": 5, "fast": 12}, {"period": 20, "fast": 12}]
    
    Если ни один параметр не массив — вернуть [params].
    """
    import itertools

    array_keys = []
    array_values = []
    scalar_params = {}

    for k, v in params.items():
        if isinstance(v, (list, tuple)):
            array_keys.append(k)
            array_values.append(v)
        else:
            scalar_params[k] = v

    if not array_keys:
        return [dict(params)]

    combos = []
    for combo_values in itertools.product(*array_values):
        combo = dict(scalar_params)
        for i, k in enumerate(array_keys):
            combo[k] = combo_values[i]
        combos.append(combo)

    return combos


def _apply_template(template: str, params: dict) -> str:
    """Применить шаблон с подстановкой параметров.
    
    Пример:
        _apply_template("ema_{period}", {"period": 20})
        → "ema_20"
    """
    return template.format(**params)


def compile(jobs: list[dict], kernel_table: dict) -> list[Task]:
    """Преобразовать список Job (dict) в список Task.
    
    Args:
        jobs: список Job в формате v0.7
        kernel_table: dict[alias] → {"describe": fn, "cpu": fn, ...}
    
    Returns:
        list[Task] — готовые к выполнению задачи, без строк зависимостей
    """
    tasks: list[Task] = []
    symbol_table: dict[str, tuple[int, int]] = {}
    next_id = 0

    for job_idx, job in enumerate(jobs):
        op = job.get("op", "")
        raw_params = job.get("params", {})
        raw_inputs = job.get("inputs", [])
        resource = job.get("resource", "@tmp")
        out_override = job.get("out", None)

        if not op:
            raise ValueError(f"Job {job_idx}: missing 'op'")
        if op not in kernel_table:
            raise KeyError(f"Kernel '{op}' not registered. Available: {list(kernel_table.keys())}")

        kernel = kernel_table[op]

        # Развернуть массивы params → конкретные комбинации
        param_combos = _cartesian_product(raw_params)

        for combo in param_combos:
            # Получить план выполнения
            plan: ExecutionPlan = kernel["describe"](combo)

            # Сгенерировать имена выходов
            names = []
            if out_override is not None:
                if isinstance(out_override, str):
                    names = [_apply_template(out_override, combo)]
                elif isinstance(out_override, (list, tuple)):
                    names = list(out_override)
                else:
                    names = [str(out_override)]
            else:
                for slot in plan.outputs:
                    names.append(_apply_template(slot.template, combo))

            # Разрешить inputs → ResourceRef
            input_refs: list[ResourceRef] = []
            for inp_str in raw_inputs:
                if inp_str in symbol_table:
                    ref_task_id, ref_out_idx = symbol_table[inp_str]
                    input_refs.append(ResourceRef(
                        type="task",
                        task_id=ref_task_id,
                        output_idx=ref_out_idx,
                    ))
                else:
                    input_refs.append(ResourceRef(
                        type="input",
                        column=inp_str,
                    ))

            # Создать Task (ExecutionPlan уничтожен — все данные в Task)
            task = Task(
                id=next_id,
                op=op,
                params=combo,
                inputs=input_refs,
                resource=resource,
                num_outputs=len(plan.outputs),
                out_names=names,
                workspace=list(plan.workspace),
                uniforms=dict(plan.uniforms),
                output_size_fn=plan.output_size_fn,
                dispatch=plan.dispatch,
            )
            tasks.append(task)

            # Зарегистрировать в SymbolTable
            for i, name in enumerate(names):
                symbol_table[name] = (next_id, i)

            next_id += 1

    # SymbolTable уничтожена (вышла из области видимости)
    return tasks

