"""Dispatcher — выполняет задачи Pipeline последовательно.

Dispatcher читает источник, запускает каждую задачу,
собирает результаты в одну Table.
"""

from ..._core.series_proxy import SeriesProxy
from ...Storage import Table, Column, ColumnType


class Dispatcher:
    """Выполняет список задач.
    
    Проходит по задачам по порядку.
    Каждая задача читает данные из source, вычисляет результат,
    сохраняет во временный словарь.
    В конце собирает одну Table.
    """
    
    def execute(self, tasks, source):
        """Выполнить задачи.
        
        Args:
            tasks: list of Task
            source: SeriesProxy, Table, или dict
        
        Returns:
            Table со всеми результатами
        """
        # Определяем источник
        if isinstance(source, SeriesProxy):
            source_table = source._table
            source_column = source._column
            num_rows = source.num_rows
        elif isinstance(source, Table):
            source_table = source
            source_column = source.column_list[0].name
            num_rows = source.num_rows
        elif isinstance(source, dict):
            first_key = list(source.keys())[0]
            cols = [Column(first_key, ColumnType.SCALED, scale=0.01)]
            source_table = Table(cols, source)
            source_column = first_key
            num_rows = len(source[first_key])
        else:
            raise TypeError(f"Expected SeriesProxy, Table, or dict, got {type(source)}")
        
        # Читаем source данные один раз
        source_data = [
            source_table.get(source_column, i)
            for i in range(num_rows)
        ]
        
        # Выполняем задачи по порядку
        columns = []
        data = {}
        
        for task in tasks:
            args = task.params.get("args", ())
            kwargs = task.params.get("kwargs", {})
            
            result = task.op.compute(source_data, *args, **kwargs)
            
            col = Column(task.name, ColumnType.SCALED, scale=0.01)
            columns.append(col)
            data[task.name] = result
        
        return Table(columns, data)
