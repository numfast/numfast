"""Recursive descent parser: string -> AST.

Grammar:
  expr     -> term (('+' | '-') term)*
  term     -> unary (('*' | '/') unary)*
  unary    -> ('-' | '!') unary | call
  call     -> FUNC '(' expr ')' | atom
  atom     -> NUMBER | FIELD | '(' expr ')'

Tokens: NUMBER (int or float), FIELD ([a-zA-Z_]\w*),
        FUNC (same as FIELD but known function name),
        operators + - * /, parentheses ().

Functions: abs, max, min, sqrt, sign, floor, ceil, round, exp, log, pow
"""

from .ast_nodes import (
    AstNode, Field, Const, Unary, UnaryOp, Binary, BinaryOp, Call, extract_physicals,
)
from typing import Optional


# Known functions (lowercase)
_FUNCTIONS = {
    'abs', 'max', 'min', 'sqrt', 'sign',
    'floor', 'ceil', 'round', 'exp', 'log', 'pow',
}


class ParseError(SyntaxError):
    pass


def parse_expr(text: str) -> AstNode:
    """Parse a formula string into an AST node.

    Args:
        text: Formula string, e.g. "(high + low) * 0.5"

    Returns:
        Root AstNode

    Raises:
        ParseError: on syntax error
    """
    tokens = _tokenize(text)
    parser = _Parser(tokens)
    result = parser.parse_expr()
    if parser.pos < len(parser.tokens):
        raise ParseError(f"Unexpected token after expression: {parser.peek()}")
    return result


# --- Tokenizer ---

class _Token:
    __slots__ = ('kind', 'value')
    def __init__(self, kind: str, value: str):
        self.kind = kind
        self.value = value
    def __repr__(self):
        return f'Token({self.kind}, {self.value!r})'


def _tokenize(text: str) -> list[_Token]:
    """Convert formula string to token list."""
    tokens: list[_Token] = []
    i = 0
    while i < len(text):
        c = text[i]

        # Whitespace
        if c in ' \t\n\r':
            i += 1
            continue

        # Number
        if c.isdigit() or (c == '.' and i + 1 < len(text) and text[i + 1].isdigit()):
            start = i
            i += 1
            has_dot = (c == '.')
            while i < len(text) and (text[i].isdigit() or (text[i] == '.' and not has_dot)):
                if text[i] == '.':
                    has_dot = True
                i += 1
            tokens.append(_Token('NUMBER', text[start:i]))
            continue

        # Identifier or keyword
        if c.isalpha() or c == '_':
            start = i
            while i < len(text) and (text[i].isalnum() or text[i] == '_'):
                i += 1
            word = text[start:i]
            if word.lower() in _FUNCTIONS:
                tokens.append(_Token('FUNC', word.lower()))
            else:
                tokens.append(_Token('FIELD', word))
            continue

        # Multi-character operators
        if c in '<>!=' and i + 1 < len(text) and text[i + 1] == '=':
            tokens.append(_Token('OP', text[i:i + 2]))
            i += 2
            continue
        if c == '&' and i + 1 < len(text) and text[i + 1] == '&':
            tokens.append(_Token('OP', '&&'))
            i += 2
            continue
        if c == '|' and i + 1 < len(text) and text[i + 1] == '|':
            tokens.append(_Token('OP', '||'))
            i += 2
            continue

        # Single-character operators and punctuation
        if c in '+-*/()!<>,':
            if c in '+-*/<>!,':
                tokens.append(_Token('OP', c))
            else:
                tokens.append(_Token('PAREN', c))
            i += 1
            continue

        raise ParseError(f"Unexpected character: {c!r} at position {i}")

    return tokens


# --- Parser ---

class _Parser:
    def __init__(self, tokens: list[_Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[_Token]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, kind: Optional[str] = None, value: Optional[str] = None) -> _Token:
        tok = self.peek()
        if tok is None:
            raise ParseError("Unexpected end of expression")
        if kind is not None and tok.kind != kind:
            raise ParseError(f"Expected {kind}, got {tok.kind}({tok.value!r})")
        if value is not None and tok.value != value:
            raise ParseError(f"Expected {value!r}, got {tok.value!r}")
        self.pos += 1
        return tok

    def parse_expr(self) -> AstNode:
        """expr -> term (('+' | '-') term)*"""
        left = self.parse_term()
        while True:
            tok = self.peek()
            if tok is None or tok.kind != 'OP' or tok.value not in ('+', '-'):
                break
            self.consume()
            right = self.parse_term()
            left = Binary(BinaryOp.from_symbol(tok.value), left, right)
        return left

    def parse_term(self) -> AstNode:
        """term -> unary (('*' | '/') unary)*"""
        left = self.parse_unary()
        while True:
            tok = self.peek()
            if tok is None or tok.kind != 'OP' or tok.value not in ('*', '/'):
                break
            self.consume()
            right = self.parse_unary()
            left = Binary(BinaryOp.from_symbol(tok.value), left, right)
        return left

    def parse_unary(self) -> AstNode:
        """unary -> ('-' | '!') unary | call"""
        tok = self.peek()
        if tok is not None and tok.kind == 'OP' and tok.value in ('-', '!'):
            self.consume()
            expr = self.parse_unary()
            return Unary(UnaryOp.from_symbol(tok.value), expr)
        return self.parse_call()

    def parse_call(self) -> AstNode:
        """call -> FUNC '(' expr ')' | atom"""
        tok = self.peek()
        if tok is not None and tok.kind == 'FUNC':
            self.consume()
            self.consume('PAREN', '(')
            arg = self.parse_expr()
            # Check for multiple args (comma-separated)
            args = [arg]
            while self.peek() is not None and self.peek().kind == 'OP' and self.peek().value == ',':
                self.consume()
                args.append(self.parse_expr())
            self.consume('PAREN', ')')
            return Call(tok.value, args)
        return self.parse_atom()

    def parse_atom(self) -> AstNode:
        """atom -> NUMBER | FIELD | '(' expr ')'"""
        tok = self.peek()
        if tok is None:
            raise ParseError("Unexpected end of expression")

        if tok.kind == 'NUMBER':
            self.consume()
            return Const(float(tok.value))

        if tok.kind == 'FIELD':
            self.consume()
            return Field(tok.value)

        if tok.kind == 'PAREN' and tok.value == '(':
            self.consume()
            node = self.parse_expr()
            self.consume('PAREN', ')')
            return node

        raise ParseError(f"Unexpected token: {tok.kind}({tok.value!r})")
