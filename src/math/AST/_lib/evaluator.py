"""AST Evaluator for NumFast.

Provides runtime evaluation of AST nodes with support for trading contexts.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from .ast_nodes import (
    ASTNode,
    Identifier,
    Literal,
    BinaryOp,
    UnaryOp,
    Call,
    MemberAccess,
)
from .parser import BinaryOpType, UnaryOpType


class ASTEvaluator:
    """Evaluates AST nodes to produce numerical results."""

    def __init__(self, context=None, scope: Optional[Dict[str, np.ndarray]] = None):
        self.context = context
        self.scope = scope or {}

    def evaluate(self, node: ASTNode, context=None) -> np.ndarray:
        """Evaluate an AST node.

        Args:
            node: AST node to evaluate
            context: Runtime context (overrides self.context)

        Returns:
            numpy array with evaluation result
        """
        if context is not None:
            self.context = context

        if isinstance(node, Identifier):
            return self._evaluate_identifier(node)
        elif isinstance(node, Literal):
            return self._evaluate_literal(node)
        elif isinstance(node, BinaryOp):
            return self._evaluate_binary_op(node)
        elif isinstance(node, UnaryOp):
            return self._evaluate_unary_op(node)
        elif isinstance(node, Call):
            return self._evaluate_call(node)
        elif isinstance(node, MemberAccess):
            return self._evaluate_member_access(node)
        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")

    def _evaluate_identifier(self, node: Identifier) -> np.ndarray:
        """Evaluate identifier lookup."""
        if node.name in self.scope:
            return self.scope[node.name]

        if self.context:
            if hasattr(self.context, '_lookup_var'):
                return self.context._lookup_var(node.name)

        raise NameError(f"Identifier '{node.name}' not found in scope or context")

    def _evaluate_literal(self, node: Literal) -> np.ndarray:
        """Evaluate literal value."""
        if isinstance(node.value, (int, float)):
            return np.array([float(node.value)], dtype=np.float64)
        elif isinstance(node.value, bool):
            return np.array([1.0 if node.value else 0.0], dtype=np.float64)
        else:
            return np.array([node.value], dtype=np.float64)

    def _evaluate_binary_op(self, node: BinaryOp) -> np.ndarray:
        """Evaluate binary operation."""
        left = self.evaluate(node.left)
        right = self.evaluate(node.right)

        if len(left.shape) == 0 or len(right.shape) == 0:
            left = np.array([left], dtype=np.float64)
            right = np.array([right], dtype=np.float64)

        op_type = node.op_type

        if op_type == BinaryOpType.PLUS:
            result = np.add(left, right)
        elif op_type == BinaryOpType.MINUS:
            result = np.subtract(left, right)
        elif op_type == BinaryOpType.MULTIPLY:
            result = np.multiply(left, right)
        elif op_type == BinaryOpType.DIVIDE:
            out_shape = np.broadcast(left, right).shape
            result = np.divide(left, right, where=right != 0, out=np.full(out_shape, np.inf))
        elif op_type == BinaryOpType.MODULO:
            out_shape = np.broadcast(left, right).shape
            result = np.mod(left, right, where=right != 0, out=np.full(out_shape, 0))
        elif op_type == BinaryOpType.POWER:
            result = np.power(left, right)
        else:
            raise ValueError(f"Unsupported binary operator: {op_type}")

        return result

    def _evaluate_unary_op(self, node: UnaryOp) -> np.ndarray:
        """Evaluate unary operation."""
        operand = self.evaluate(node.operand)
        op_type = node.op_type

        if op_type == UnaryOpType.NEGATIVE:
            result = np.negative(operand)
        elif op_type == UnaryOpType.NOT:
            result = 1.0 - np.minimum(1.0, np.maximum(0.0, operand))
        else:
            raise ValueError(f"Unsupported unary operator: {op_type}")

        return result

    def _evaluate_call(self, node: Call) -> np.ndarray:
        """Evaluate function call."""
        func_name = node.func_name

        if self.context and hasattr(self.context, func_name):
            if func_name in ['SMA', 'EMA', 'ATR', 'ROC', 'STOCH', 'CCI']:
                return self._evaluate_trading_function(func_name, node)

            context_method = getattr(self.context, func_name)
            return self._call_context_method(context_method, node)

        if func_name == 'SMA':
            return self._evaluate_sma(node)
        elif func_name == 'EMA':
            return self._evaluate_ema(node)
        elif func_name == 'ATR':
            return self._evaluate_atr(node)
        else:
            raise NameError(f"Unknown function: {func_name}")

    def _evaluate_trading_function(self, func_name: str, node: Call) -> np.ndarray:
        """Evaluate trading function using context."""
        args = []
        kwargs = {}

        for arg in node.args:
            val = self.evaluate(arg)
            if len(val) == 1:
                val = val[0] if isinstance(val, np.ndarray) else val
            args.append(val)

        if node.kwargs:
            for key, value in node.kwargs.items():
                val = self.evaluate(value)
                if len(val) == 1:
                    val = val[0] if isinstance(val, np.ndarray) else val
                kwargs[key] = val

        if self.context:
            if func_name == 'SMA':
                close = args[0]
                period = kwargs.get('period', args[1] if len(args) > 1 else 14)
                return np.array(self.context.sma(close, period), dtype=np.float64)
            elif func_name == 'EMA':
                close = args[0]
                period = kwargs.get('period', args[1] if len(args) > 1 else 20)
                return np.array(self.context.ema(close, period), dtype=np.float64)
            elif func_name == 'ATR':
                high = args[0]
                low = args[1]
                close = args[2]
                period = kwargs.get('period', args[3] if len(args) > 3 else 14)
                return np.array(self.context.atr(high, low, close, period), dtype=np.float64)
            elif func_name == 'ROC':
                data = args[0]
                periods = kwargs.get('periods', args[1] if len(args) > 1 else 10)
                return np.array(self.context.roc(data, periods), dtype=np.float64)
            elif func_name == 'STOCH':
                high = args[0]
                low = args[1]
                close = args[2]
                windows = kwargs.get('windows', args[3] if len(args) > 3 else 14)
                return np.array(self.context.stoch(high, low, close, windows), dtype=np.float64)
            elif func_name == 'CCI':
                high = args[0]
                low = args[1]
                close = args[2]
                windows = kwargs.get('windows', args[3] if len(args) > 3 else 20)
                return np.array(self.context.cci(high, low, close, windows), dtype=np.float64)

        raise RuntimeError(f"Function {func_name} requires a RuntimeContext")

    def _call_context_method(self, method, node: Call) -> np.ndarray:
        """Call a context method with evaluated arguments."""
        args = []
        for arg in node.args:
            args.append(self.evaluate(arg))

        if node.kwargs:
            kwargs = {}
            for key, value in node.kwargs.items():
                kwargs[key] = self.evaluate(value)
        else:
            kwargs = {}

        return method(*args, **kwargs)

    def _evaluate_sma(self, node: Call) -> np.ndarray:
        """Evaluate SMA without context."""
        args = []
        for arg in node.args:
            args.append(self.evaluate(arg)[0] if isinstance(self.evaluate(arg), np.ndarray) and len(self.evaluate(arg)) == 1 else self.evaluate(arg))

        kwargs = {}
        if node.kwargs:
            for key, value in node.kwargs.items():
                kwargs[key] = self.evaluate(value)[0] if isinstance(self.evaluate(value), np.ndarray) and len(self.evaluate(value)) == 1 else self.evaluate(value)

        close = args[0]
        period = int(kwargs.get('period', args[1] if len(args) > 1 else 14))

        n = len(close)
        result = np.full(n, np.nan, dtype=np.float64)

        for i in range(period - 1, n):
            window = close[i-period+1:i+1]
            result[i] = np.sum(window) / period

        return result

    def _evaluate_ema(self, node: Call) -> np.ndarray:
        """Evaluate EMA without context."""
        args = []
        for arg in node.args:
            args.append(self.evaluate(arg)[0] if isinstance(self.evaluate(arg), np.ndarray) and len(self.evaluate(arg)) == 1 else self.evaluate(arg))

        kwargs = {}
        if node.kwargs:
            for key, value in node.kwargs.items():
                kwargs[key] = self.evaluate(value)[0] if isinstance(self.evaluate(value), np.ndarray) and len(self.evaluate(value)) == 1 else self.evaluate(value)

        close = args[0]
        period = int(kwargs.get('period', args[1] if len(args) > 1 else 20))

        n = len(close)
        result = np.full(n, np.nan, dtype=np.float64)
        result[0] = close[0]

        alpha = 2.0 / (period + 1.0)

        for i in range(1, n):
            result[i] = alpha * close[i] + (1 - alpha) * result[i-1]

        return result

    def _evaluate_atr(self, node: Call) -> np.ndarray:
        """Evaluate ATR without context."""
        args = []
        for arg in node.args:
            args.append(self.evaluate(arg)[0] if isinstance(self.evaluate(arg), np.ndarray) and len(self.evaluate(arg)) == 1 else self.evaluate(arg))

        kwargs = {}
        if node.kwargs:
            for key, value in node.kwargs.items():
                kwargs[key] = self.evaluate(value)[0] if isinstance(self.evaluate(value), np.ndarray) and len(self.evaluate(value)) == 1 else self.evaluate(value)

        high = args[0]
        low = args[1]
        close = args[2]
        period = int(kwargs.get('period', args[3] if len(args) > 3 else 14))

        n = len(close)
        result = np.full(n, np.nan, dtype=np.float64)

        for i in range(period - 1, n):
            window_tr = np.full(period, np.nan, dtype=np.float64)

            for j in range(period):
                idx = i - period + 1 + j
                tr1 = high[idx] - low[idx]
                tr2 = abs(high[idx] - close[idx-1]) if idx > 0 else 0
                tr3 = abs(low[idx] - close[idx-1]) if idx > 0 else 0
                window_tr[j] = max(tr1, tr2, tr3)

            result[i] = np.sum(window_tr) / period

        return result

    def _evaluate_member_access(self, node: MemberAccess) -> np.ndarray:
        """Evaluate member access (e.g., context.close)."""
        object_result = self.evaluate(node.object)

        if self.context and node.attr_name == 'close':
            if hasattr(self.context, 'close_prices'):
                return self.context.close_prices
        elif self.context and node.attr_name == 'open':
            if hasattr(self.context, 'open_prices'):
                return self.context.open_prices
        elif self.context and node.attr_name == 'high':
            if hasattr(self.context, 'high_prices'):
                return self.context.high_prices
        elif self.context and node.attr_name == 'low':
            if hasattr(self.context, 'low_prices'):
                return self.context.low_prices

        if len(object_result) == 1 and isinstance(object_result[0], np.ndarray):
            obj_result = object_result[0]
            if node.attr_name == 'shape':
                return np.array(obj_result.shape)
            elif node.attr_name == 'dtype':
                return np.array([str(obj_result.dtype)])

        raise AttributeError(f"Member '{node.attr_name}' not available on object {object_result}")


def evaluate(node: ASTNode, context=None, scope: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """Evaluate an AST node.

    Args:
        node: AST node to evaluate
        context: Runtime context
        scope: Variable scope for evaluation

    Returns:
        numpy array with evaluation result
    """
    evaluator = ASTEvaluator(context, scope)
    return evaluator.evaluate(node, context)


if __name__ == "__main__":
    from .parser import parse

    test_exprs = [
        "SMA(close, 14)",
        "close + open",
        "price * 1.5",
        "-(close - high)",
        "EMA(high, period=20)",
    ]

    for expr in test_exprs:
        print(f"\nEvaluating: {expr}")
        try:
            ast = parse(expr)
            result = evaluate(ast)
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  Error: {e}")
