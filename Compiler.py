from collections import defaultdict

class LexicalAnalyzer:
    keywords = {
        'bool': 'T_Bool',
        'break': 'T_Break',
        'char': 'T_Char',
        'continue': 'T_Continue',
        'else': 'T_Else',
        'false': 'T_False',
        'for': 'T_For',
        'if': 'T_If',
        'int': 'T_Int',
        'print': 'T_Print',
        'return': 'T_Return',
        'true': 'T_True'
    }

    whitespace_chars = {' ', '\t', '\n'}

    operators = {
        '+': 'T_AOp_PL',
        '-': 'T_AOp_MN',
        '*': 'T_AOp_ML',
        '/': 'T_AOp_DV',
        '%': 'T_AOp_RM',
        '<': 'T_ROp_L',
        '>': 'T_ROp_G',
        '<=': 'T_ROp_LE',
        '>=': 'T_ROp_GE',
        '!=': 'T_ROp_NE',
        '==': 'T_ROp_E',
        '&&': 'T_LOp_AND',
        '||': 'T_LOp_OR',
        '!': 'T_LOp_NOT',
        '=': 'T_Assign',
        '|': 'T_Unknown',
        '&': 'T_Unknown'
    }

    single_char_tokens = {
        '(': 'T_LP',
        ')': 'T_RP',
        '{': 'T_LC',
        '}': 'T_RC',
        '[': 'T_LB',
        ']': 'T_RB',
        ';': 'T_Semicolon',
        ',': 'T_Comma'
    }

    def __init__(self, source_code):
        self.source_code = source_code
        self.tokens = []

    def tokenize(self):
        self.tokens = []
        i = 0
        while i < len(self.source_code):
            char = self.source_code[i]
            if char in self.whitespace_chars:
                # حذف توکن‌های فضای خالی
                i += 1
            elif char in self.single_char_tokens:
                self.tokens.append((self.single_char_tokens[char], char))
                i += 1
            elif char in self.operators:
                if self.source_code[i:i + 2] in self.operators:
                    self.tokens.append((self.operators[self.source_code[i:i + 2]], self.source_code[i:i + 2]))
                    i += 2
                else:
                    if self.source_code[i] == '/' and i + 1 < len(self.source_code) and self.source_code[i + 1] == '/':
                        comment = self.source_code[i:].strip()
                        self.tokens.append(('T_Comment', comment))
                        break
                    else:
                        self.tokens.append((self.operators[char], char))
                        i += 1
            elif char.isalpha() or char == '_':
                lexeme = ''
                while i < len(self.source_code) and (self.source_code[i].isalnum() or self.source_code[i] == '_'):
                    lexeme += self.source_code[i]
                    i += 1
                if lexeme in self.keywords:
                    self.tokens.append((self.keywords[lexeme], lexeme))
                else:
                    self.tokens.append(('T_Id', lexeme))
            elif char.isdigit():
                lexeme = ''
                while i < len(self.source_code) and (
                        self.source_code[i].isdigit() or self.source_code[i].lower() in 'abcdefx'):
                    lexeme += self.source_code[i]
                    i += 1
                if lexeme.startswith('0x') or lexeme.startswith('0X'):
                    self.tokens.append(('T_Hexadecimal', lexeme))
                else:
                    self.tokens.append(('T_Decimal', lexeme))
            elif char == '"':
                lexeme = char
                i += 1
                while i < len(self.source_code) and self.source_code[i] != '"':
                    lexeme += self.source_code[i]
                    i += 1
                lexeme += '"'
                self.tokens.append(('T_String', lexeme))
                i += 1
            elif char == "'":
                lexeme = char
                i += 1
                if i < len(self.source_code):
                    lexeme += self.source_code[i]
                    i += 1
                if i < len(self.source_code) and self.source_code[i] == "'":
                    lexeme += self.source_code[i]
                    i += 1
                    lexeme += "'"
                    self.tokens.append(('T_Char', lexeme))
                    i += 1
                else:
                    raise ValueError(f"Error: Invalid character constant at position {i}")
            else:
                raise ValueError(f"Error: Invalid character '{char}' at position {i}")
        return self.tokens


# Define grammar
grammar = {
    'Program': [['DeclarationList']],
    'DeclarationList': [['Declaration', 'DeclarationList'], []],
    'Declaration': [['TypeSpecifier', 'Identifier', 'DeclarationPrime']],
    'DeclarationPrime': [['T_Semicolon'], ['T_LP', 'ParameterList', 'T_RP', 'Block']],
    'TypeSpecifier': [['T_Int'], ['T_Bool'], ['T_Char']],
    'ParameterList': [['TypeSpecifier', 'Identifier', 'ParameterListPrime'], []],
    'ParameterListPrime': [['T_Comma', 'TypeSpecifier', 'Identifier', 'ParameterListPrime'], []],
    'Block': [['T_LC', 'StatementList', 'T_RC']],
    'StatementList': [['Statement', 'StatementList'], []],
    'Statement': [['ExpressionStatement'], ['IfStatement'], ['PrintStatement'], ['ReturnStatement'], ['Block'],['Declaration']],
    'ExpressionStatement': [['Expression', 'T_Semicolon']],
    'IfStatement': [['T_If', 'T_LP', 'Expression', 'T_RP', 'Statement']],
    'PrintStatement': [['T_Print', 'T_LP', 'Expression', 'T_RP', 'T_Semicolon']],
    'ReturnStatement': [['T_Return', 'Expression', 'T_Semicolon']],
    'Expression': [['Identifier', 'T_Assign', 'Expression'], ['SimpleExpression']],
    'SimpleExpression': [['AdditiveExpression', 'SimpleExpressionPrime']],
    'SimpleExpressionPrime': [['RelationalOperator', 'AdditiveExpression'], []],
    'AdditiveExpression': [['Term', 'AdditiveExpressionPrime']],
    'AdditiveExpressionPrime': [['AddOperator', 'Term', 'AdditiveExpressionPrime'], []],
    'Term': [['Factor', 'TermPrime']],
    'TermPrime': [['MulOperator', 'Factor', 'TermPrime'], []],
    'Factor': [['T_LP', 'Expression', 'T_RP'], ['Identifier'], ['Integer'], ['String'], ['Char']],
    'RelationalOperator': [['T_ROp_L'], ['T_ROp_G'], ['T_ROp_LE'], ['T_ROp_GE'], ['T_ROp_NE'], ['T_ROp_E']],
    'AddOperator': [['T_AOp_PL'], ['T_AOp_MN']],
    'MulOperator': [['T_AOp_ML'], ['T_AOp_DV'], ['T_AOp_RM']],
    'Identifier': [['T_Id']],
    'Integer': [['T_Decimal'], ['T_Hexadecimal']],
    'String': [['T_String']],
    'Char': [['T_Char']],
}


# Compute FIRST sets
def compute_first_sets(grammar):
    first = defaultdict(set)
    def first_of(symbol):
        if symbol in first:
            return first[symbol]
        if symbol not in grammar:  # terminal
            return {symbol}
        result = set()
        for production in grammar[symbol]:
            if not production:  # Skip empty productions
                continue
            for s in production:
                result |= first_of(s)
                if 'ε' not in first[s]:  # If ε is not in the FIRST set, break
                    break
            else:
                result.add('ε')  # If ε is in the FIRST set of all symbols in production
        first[symbol] = result
        return result

    for non_terminal in grammar:
        first_of(non_terminal)
    return first


# Compute FOLLOW sets
def compute_follow_sets(grammar, first):

    follow = defaultdict(set)
    follow['Program'].add('$')  # Start symbol

    while True:
        updated = False
        for head, productions in grammar.items():
            for production in productions:
                trailer = follow[head]
                for symbol in reversed(production):
                    if symbol in grammar:  # non-terminal
                        if follow[symbol] != follow[symbol] | trailer:
                            follow[symbol] |= trailer
                            updated = True
                        if 'ε' in first[symbol]:
                            trailer |= first[symbol] - {'ε'}
                        else:
                            trailer = first[symbol]
                    else:
                        trailer = {symbol}
        if not updated:
            break
    return follow


def construct_parsing_table(grammar, first, follow):
    parsing_table = defaultdict(dict)
    for head, productions in grammar.items():
        for production in productions:
            if not production:
                continue
            if production == ['ε']:
                for terminal in follow[head]:
                    parsing_table[head][terminal] = production
            else:
                first_set = first[production[0]]
                for terminal in first_set:
                    if terminal != 'ε':
                        parsing_table[head][terminal] = production
                if 'ε' in first_set:
                    for terminal in follow[head]:
                        if terminal not in parsing_table[head]:
                            parsing_table[head][terminal] = production
    return parsing_table


def parse_source_code(tokens, parsing_table):
    stack = ['Program', '$']
    tokens.append(('$', '$'))
    tokens.reverse()
    output = []
    while len(stack) > 0:
        top = stack.pop()
        current_token = tokens[-1][0]
        if top == current_token:
            tokens.pop()
        elif top in parsing_table and current_token in parsing_table[top]:
            production = parsing_table[top][current_token]
            if production != ['ε']:
                stack.extend(reversed(production))
            output.append(production)
        else:
            raise SyntaxError("Parsing Error")
    return output

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.current_token_index = 0
        self.sync_tokens = ['T_Semicolon', 'T_RC', 'T_RP']

    def match(self, token_type):
        if self.current_token_index < len(self.tokens) and self.tokens[self.current_token_index][0] == token_type:
            self.current_token_index += 1
        else:
            self.error_recovery(token_type)

    def error_recovery(self, expected_token):
        print(f"Syntax Error: Expected token type '{expected_token}', found '{self.tokens[self.current_token_index][0]}'")
        while self.current_token_index < len(self.tokens) and self.tokens[self.current_token_index][0] not in self.sync_tokens:
            self.current_token_index += 1
        if self.current_token_index < len(self.tokens) and self.tokens[self.current_token_index][0] in self.sync_tokens:
            self.current_token_index += 1

    def parse(self):
        declarations = []
        while self.current_token_index < len(self.tokens):
            if self.tokens[self.current_token_index][0] in ['T_Int', 'T_Void']:
                declarations.append(self.parse_declaration())
            else:
                self.error_recovery('Type Specifier')
        return Program(declarations)
    
    def parse_if_statement(self):
        self.match('T_If')
        self.match('T_LP')
        condition = self.parse_expression()
        self.match('T_RP')
        then_branch = self.parse_statement()
        else_branch = None
        if self.tokens[self.current_token_index][0] == 'T_Else':
            self.match('T_Else')
            else_branch = self.parse_statement()
        return IfStmt(condition, then_branch, else_branch)
    
    def parse_print_statement(self):
        self.match('T_Print')
        self.match('T_LP')
        expression = self.parse_expression()
        self.match('T_RP')
        self.match('T_Semicolon')
        return PrintStmt(expression)
    
    def parse_return_statement(self):
        self.match('T_Return')
        expression = None
        if self.tokens[self.current_token_index][0] != 'T_Semicolon':
            expression = self.parse_expression()
        self.match('T_Semicolon')
        return ReturnStmt(expression)

    def parse_integer(self):
        integer = Integer(self.tokens[self.current_token_index][1])
        self.match('T_Decimal')
        return integer

    def parse_string(self):
        string = String(self.tokens[self.current_token_index][1])
        self.match('T_String')
        return string

    def parse_parameter_list(self):
        parameters = []
        if self.tokens[self.current_token_index][0] in ['T_Int', 'T_Bool', 'T_Char']:
            type_specifier = self.parse_type_specifier()
            identifier = self.parse_identifier()
            parameters.append((type_specifier, identifier))
            while self.tokens[self.current_token_index][0] == 'T_Comma':
                self.match('T_Comma')
                type_specifier = self.parse_type_specifier()
                identifier = self.parse_identifier()
                parameters.append((type_specifier, identifier))
        return parameters

    def parse_program(self):
        declarations = []
        while self.current_token_index < len(self.tokens):
            if self.tokens[self.current_token_index][0] in ['T_Int', 'T_Void']:
                declarations.append(self.parse_declaration())
            else:
                self.error_recovery('Type Specifier')
        return Program(declarations)
   
    def parse_declaration(self):
        type_specifier = self.parse_type_specifier()
        identifier = self.parse_identifier()

        if self.current_token_index < len(self.tokens) and self.tokens[self.current_token_index][0] == 'T_LP':
            return self.parse_function_declaration(type_specifier, identifier)
        else:
            return self.parse_variable_declaration(type_specifier, identifier)

    def parse_type_specifier(self):
        if self.tokens[self.current_token_index][0] in ['T_Int', 'T_Bool', 'T_Char']:
            token_type = self.tokens[self.current_token_index][0]
            self.match(token_type)
            return token_type
        else:
            self.error_recovery('Type Specifier')
            return None

    def parse_function_declaration(self, type_specifier, identifier):
        self.match('T_LP')
        parameters = self.parse_parameter_list()
        self.match('T_RP')
        body = self.parse_block()
        return FunDecl(type_specifier, identifier, parameters, body)

    def parse_variable_declaration(self, type_specifier, identifier):
        expression = None
        if self.current_token_index < len(self.tokens) and self.tokens[self.current_token_index][0] == 'T_Assign':
            self.match('T_Assign')
            expression = self.parse_expression()
        self.match('T_Semicolon')
        return VarDecl(type_specifier, identifier, expression)

    def parse_block(self):
        self.match('T_LC')
        statements = []
        while self.current_token_index < len(self.tokens) and self.tokens[self.current_token_index][0] != 'T_RC':
            statements.append(self.parse_statement())
        self.match('T_RC')
        return Block(statements)

    def parse_statement(self):
        if self.current_token_index >= len(self.tokens):
            return None

        current_token = self.tokens[self.current_token_index][0]

        if current_token == 'T_Id':
            return self.parse_expression_statement()
        elif current_token == 'T_If':
            return self.parse_if_statement()
        elif current_token == 'T_Print':
            return self.parse_print_statement()
        elif current_token == 'T_Return':
            return self.parse_return_statement()
        elif current_token == 'T_LC':
            return self.parse_block()
        elif current_token in ['T_Int', 'T_Bool', 'T_Char']:
            return self.parse_declaration()
        else:
            self.current_token_index += 1
            return None

    def parse_expression_statement(self):
        expression = self.parse_expression()
        self.match('T_Semicolon')
        return ExprStmt(expression)

    def parse_expression(self):
        left = self.parse_simple_expression()
        if self.current_token_index < len(self.tokens):
            if self.tokens[self.current_token_index][0] == 'T_Assign':
                self.match('T_Assign')
                right = self.parse_expression()
                return BinaryOp('=', left, right)
            elif self.tokens[self.current_token_index][0] == 'T_LBracket':
                self.match('T_LBracket')
                index = self.parse_expression()
                self.match('T_RBracket')
                return ArrayIndex(left, index)
        return left

    def parse_simple_expression(self):
        left = self.parse_additive_expression()
        if self.current_token_index < len(self.tokens) and self.tokens[self.current_token_index][0] in ['T_ROp_L', 'T_ROp_G', 'T_ROp_LE', 'T_ROp_GE', 'T_ROp_NE', 'T_ROp_E']:
            operator = self.tokens[self.current_token_index][0]
            self.match(operator)
            right = self.parse_additive_expression()
            return BinaryOp(operator, left, right)
        return left

    def parse_additive_expression(self):
        left = self.parse_term()
        while self.current_token_index < len(self.tokens) and self.tokens[self.current_token_index][0] in ['T_AOp_PL', 'T_AOp_MN']:
            operator = self.tokens[self.current_token_index][0]
            self.match(operator)
            right = self.parse_term()
            left = BinaryOp(operator, left, right)
        return left

    def parse_term(self):
        left = self.parse_factor()
        while self.current_token_index < len(self.tokens) and self.tokens[self.current_token_index][0] in ['T_AOp_ML', 'T_AOp_DV']:
            operator = self.tokens[self.current_token_index][0]
            self.match(operator)
            right = self.parse_factor()
            left = BinaryOp(operator, left, right)
        return left

    def parse_factor(self):
        if self.tokens[self.current_token_index][0] == 'T_LP':
            self.match('T_LP')
            expression = self.parse_expression()
            self.match('T_RP')
            return expression
        elif self.tokens[self.current_token_index][0] == 'T_Id':
            identifier = self.parse_identifier()
            if self.current_token_index < len(self.tokens) and self.tokens[self.current_token_index][0] == 'T_LP':
                return self.parse_function_call(identifier)
            elif self.current_token_index < len(self.tokens) and self.tokens[self.current_token_index][0] == 'T_LB':
                return self.parse_array_index(identifier)
            else:
                return identifier
        elif self.tokens[self.current_token_index][0] == 'T_Decimal':
            return self.parse_integer()
        elif self.tokens[self.current_token_index][0] == 'T_String':
            return self.parse_string()

    def parse_identifier(self):
        if self.tokens[self.current_token_index][0] == 'T_Id':
            identifier = Identifier(self.tokens[self.current_token_index][1])
            self.match('T_Id')
            return identifier
        else:
            self.error_recovery('Identifier')
            return None

    def parse_function_call(self, identifier):
        self.match('T_LP')
        arguments = []
        if self.tokens[self.current_token_index][0] != 'T_RP':
            arguments.append(self.parse_expression())
            while self.tokens[self.current_token_index][0] == 'T_Comma':
                self.match('T_Comma')
                arguments.append(self.parse_expression())
        self.match('T_RP')
        return FunctionCall(identifier, arguments)

    def parse_array_index(self, identifier):
        self.match('T_LB')
        index = self.parse_expression()
        self.match('T_RB')
        return ArrayIndex(identifier, index)



# AST Code
class Program:
    def __init__(self, declarations):
        self.declarations = declarations

    def __str__(self):
        return f"Program({self.declarations})"

class VarDecl:
    def __init__(self, type, identifier, expression=None):
        self.type = type
        self.identifier = identifier
        self.expression = expression  # Optional initial expression

    def __str__(self):
        return f"VarDecl(type={self.type}, identifier={self.identifier}, expression={self.expression})"

class AssignNode:
    def __init__(self, identifier, expression, line_number):
        self.identifier = identifier
        self.expression = expression
        self.line_number = line_number

    def __str__(self):
        return f"AssignNode(identifier={self.identifier}, expression={self.expression}, line_number={self.line_number})"
    
class FunDecl:
    def __init__(self, type, identifier, parameters, body):
        self.type = type
        self.identifier = identifier
        self.parameters = parameters
        self.body = body

    def __str__(self):
        params_str = ", ".join([f"{param[0]} {param[1]}" for param in self.parameters])
        return f"FunDecl(type={self.type}, identifier={self.identifier}, parameters=[{params_str}], body={self.body})"

class Block:
    def __init__(self, statements):
        self.statements = statements

    def __str__(self):
        return f"Block(statements={self.statements})"

class ExprStmt:
    def __init__(self, expression):
        self.expression = expression

    def __str__(self):
        return f"ExprStmt(expression={self.expression})"

class IfStmt:
    def __init__(self, condition, then_branch, else_branch=None):
        self.condition = condition
        self.then_branch = then_branch
        self.else_branch = else_branch

    def __str__(self):
        return f"IfStmt(condition={self.condition}, then_branch={self.then_branch}, else_branch={self.else_branch})"

class PrintStmt:
    def __init__(self, expression):
        self.expression = expression

    def __str__(self):
        return f"PrintStmt(expression={self.expression})"

class ReturnStmt:
    def __init__(self, expression):
        self.expression = expression

    def __str__(self):
        return f"ReturnStmt(expression={self.expression})"

class BinaryOp:
    def __init__(self, operator, left, right):
        self.operator = operator
        self.left = left
        self.right = right

    def __str__(self):
        return f"BinaryOp(operator={self.operator}, left={self.left}, right={self.right})"

class Integer:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Integer(value={self.value})"

class String:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"String(value={self.value})"

class Identifier:
    def __init__(self, name):
        self.name = name
        
    def __str__(self):
        return f"Identifier(name={self.name})"
        
class ArrayIndex:
    def __init__(self, index):
        self.index = index

    def __str__(self):
        return f"ArrayIndex(index={self.index})"

class FunctionCall:
    def __init__(self, identifier, arguments):
        self.identifier = identifier
        self.arguments = arguments

    def __str__(self):
        args_str = ", ".join(map(str, self.arguments))
        return f"FunctionCall(identifier={self.identifier}, arguments=[{args_str}])"
    

def print_ast(node, indent=""):
    if isinstance(node, Program):
        print(f"{indent}Program")
        for decl in node.declarations:
            print_ast(decl, indent + "  ")
    elif isinstance(node, VarDecl):
        print(f"{indent}VarDecl(type={node.type}, identifier={node.identifier})")
        if node.expression:
            print_ast(node.expression, indent + "  ")
    elif isinstance(node, FunDecl):
        print(f"{indent}FunDecl(type={node.type}, identifier={node.identifier})")
        for param in node.parameters:
            print(f"{indent}  Param(type={param[0]}, identifier={param[1]})")
        print_ast(node.body, indent + "  ")
    elif isinstance(node, Block):
        print(f"{indent}Block")
        for stmt in node.statements:
            print_ast(stmt, indent + "  ")
    elif isinstance(node, ExprStmt):
        print(f"{indent}ExprStmt")
        print_ast(node.expression, indent + "  ")
    elif isinstance(node, IfStmt):
        print(f"{indent}IfStmt")
        print_ast(node.condition, indent + "  ")
        print_ast(node.then_branch, indent + "  ")
        if node.else_branch:
            print_ast(node.else_branch, indent + "  ")
    elif isinstance(node, PrintStmt):
        print(f"{indent}PrintStmt")
        print_ast(node.expression, indent + "  ")
    elif isinstance(node, ReturnStmt):
        print(f"{indent}ReturnStmt")
        print_ast(node.expression, indent + "  ")
    elif isinstance(node, BinaryOp):
        print(f"{indent}BinaryOp(operator={node.operator})")
        print_ast(node.left, indent + "  ")
        print_ast(node.right, indent + "  ")
    elif isinstance(node, Identifier):
        print(f"{indent}Identifier(name={node.name})")
    elif isinstance(node, Integer):
        print(f"{indent}Integer(value={node.value})")
    elif isinstance(node, String):
        print(f"{indent}String(value={node.value})")
    elif isinstance(node, ArrayIndex):
        print(f"{indent}ArrayIndex(name={node.array})")
        print_ast(node.index, indent + "  ")
    elif isinstance(node, FunctionCall):
        print(f"{indent}FunctionCall(identifier={node.identifier})")
        for arg in node.arguments:
            print_ast(arg, indent + "  ")


class SemanticAnalyzer:
    def __init__(self):
        self.symbol_table = {}
        self.current_scope = [{}]
        self.current_function = None
        self.main_found = False

    def analyze_array_index(self, array_index):
        self.analyze(array_index.index)
        index_type = self.get_expression_type(array_index.index)
        if index_type != 'T_Int':
            raise Exception("Array index must be of type 'T_Int'.")
        if isinstance(array_index.index, Integer) and array_index.index.value <= 0:
            raise Exception("Array index must be greater than zero.")
        
    def visit_ArrayIndex(self, node):
        index = self.visit(node.index)
        if isinstance(index, int):
            if index < 0:
                print(f"Semantic Error: Array index cannot be negative: {index}")
        else:
            print(f"Semantic Error: Array index must be an integer: {index}")
            

    def visit(self, node):
        if isinstance(node, Program):
            for decl in node.declarations:
                self.visit(decl)
        elif isinstance(node, FunDecl):
            self.visit(node.body)
        elif isinstance(node, Block):
            for stmt in node.statements:
                self.visit(stmt)
        elif isinstance(node, VarDecl):
            if node.expression:
                self.visit(node.expression)
        elif isinstance(node, ExprStmt):
            self.visit(node.expression)
        elif isinstance(node, IfStmt):
            self.visit(node.condition)
            self.visit(node.then_branch)
            if node.else_branch:
                self.visit(node.else_branch)
        elif isinstance(node, ReturnStmt):
            self.visit(node.expression)
        elif isinstance(node, BinaryOp):
            self.visit(node.left)
            self.visit(node.right)
        elif isinstance(node, ArrayIndex):
            self.visit_ArrayIndex(node)
        elif isinstance(node, FunctionCall):
            for arg in node.arguments:
                self.visit(arg)
        elif isinstance(node, Identifier):
            pass
        elif isinstance(node, Integer):
            return int(node.value)
        elif isinstance(node, String):
            pass


    def analyze_function_call(self, func_call):
        func_info = self.symbol_table.get(func_call.identifier.name)
        if not func_info:
            raise Exception(f"Function '{func_call.identifier.name}' is not defined.")
        if len(func_call.arguments) != len(func_info['parameters']):
            raise Exception(f"Function '{func_call.identifier.name}' called with incorrect number of arguments.")
        for arg, param in zip(func_call.arguments, func_info['parameters']):
            self.analyze(arg)
            arg_type = self.get_expression_type(arg)
            if arg_type != param[0]:
                raise Exception(f"Type mismatch in function call: Expected '{param[0]}' but got '{arg_type}'.")

    def analyze(self, node):
        if isinstance(node, Program):
            self.analyze_program(node)
        elif isinstance(node, FunDecl):
            self.analyze_function_declaration(node)
        elif isinstance(node, VarDecl):
            self.analyze_variable_declaration(node)
        elif isinstance(node, Block):
            self.analyze_block(node)
        elif isinstance(node, IfStmt):
            self.analyze_if_statement(node)
        elif isinstance(node, PrintStmt):
            self.analyze_print_statement(node)
        elif isinstance(node, ReturnStmt):
            self.analyze_return_statement(node)
        elif isinstance(node, BinaryOp):
            self.analyze_binary_op(node)
        elif isinstance(node, Integer):
            self.analyze_integer(node)
        elif isinstance(node, String):
            self.analyze_string(node)
        elif isinstance(node, Identifier):
            self.analyze_identifier(node)
        elif isinstance(node, ArrayIndex):  
            self.analyze_array_index(node)
        elif isinstance(node, FunctionCall):
            self.analyze_function_call(node)
        else:
            raise Exception(f"Unsupported node type: {type(node)}")

    #قانون تابع main
    def analyze_program(self, program):
        for decl in program.declarations:
            if isinstance(decl, FunDecl):
                if decl.identifier.name == 'main':
                    if decl.type == 'T_Int' and not decl.parameters:
                        self.main_found = True
                    else:
                        raise Exception("Main function signature is incorrect.")

                # Check for duplicate function definitions
                if decl.identifier.name in self.symbol_table:
                    raise Exception(f"Function '{decl.identifier.name}' is already defined.")

            # Analyze each declaration (variable or function)
            self.analyze(decl)
        
        if not self.main_found:
            raise Exception("Main function not found.")

    def analyze_function_declaration(self, func_decl):
        if func_decl.identifier.name in self.symbol_table:
            raise Exception(f"Function '{func_decl.identifier.name}' is already defined.")

        self.symbol_table[func_decl.identifier.name] = {
            'type': func_decl.type,
            'parameters': func_decl.parameters,
            'scope_level': 0
        }

        self.current_function = func_decl.identifier.name
        self.current_scope.append({})

        # Check parameter types and count
        for param in func_decl.parameters:
            self.analyze_variable_declaration(VarDecl(param[0], param[1]))

        self.analyze(func_decl.body)
        self.current_scope.pop()
        self.current_function = None


    #قانون حوزه‌های تعریف و تو در تو بودن آن‌ها
    def analyze_block(self, block):
        self.current_scope.append({})
        for stmt in block.statements:
            self.analyze(stmt)
        self.current_scope.pop()

    #قانون تطابق نوع در عبارات انتساب    
    def analyze_variable_declaration(self, var_decl):
        current_scope_level = len(self.current_scope) - 1
        current_scope_names = self.current_scope[current_scope_level]

        if var_decl.identifier.name in current_scope_names:
            raise Exception(f"Variable '{var_decl.identifier.name}' is already defined in the current scope.")

        current_scope_names[var_decl.identifier.name] = var_decl.type

        if isinstance(var_decl.expression, ArrayIndex):
            self.analyze_array_declaration(var_decl.identifier.name, var_decl.expression)
        elif var_decl.expression:
            self.analyze(var_decl.expression)
            expr_type = self.get_expression_type(var_decl.expression)
            if var_decl.type != expr_type:
                raise Exception(f"Type mismatch in variable declaration: Expected '{var_decl.type}' but got '{expr_type}'.")
            
    def analyze_array_declaration(self, array_name, array_index):
        # Check if array index type is 'T_Int'
        index_type = self.get_expression_type(array_index)
        if index_type != 'T_Int':
            raise Exception(f"Array index must have type 'T_Int' but got '{index_type}' for array '{array_name}'.")
            
    #قانون نوع شرط دستور if
    def analyze_if_statement(self, if_stmt):
        self.analyze(if_stmt.condition)
        cond_type = self.get_expression_type(if_stmt.condition)
        if cond_type != 'T_Bool':
            raise Exception("Condition in if statement must be of type 'T_Bool'.")
        self.analyze(if_stmt.then_branch)
        if if_stmt.else_branch:
            self.analyze(if_stmt.else_branch)

    #قانون تطابق نوع تابع و عبارت return
    def analyze_return_statement(self, return_stmt):
        if self.current_function is None:
            raise Exception("Return statement outside of function.")

        current_function_type = self.symbol_table[self.current_function]['type']

        if return_stmt.expression:
            self.analyze(return_stmt.expression)
            return_expr_type = self.get_expression_type(return_stmt.expression)
            if return_expr_type != current_function_type:
                raise Exception(f"Return type mismatch: Expected '{current_function_type}' but got '{return_expr_type}'.")
        elif current_function_type != 'T_Void':
            raise Exception(f"Return statement missing for function '{self.current_function}' returning '{current_function_type}'.")

    def analyze_print_statement(self, print_stmt):
        self.analyze(print_stmt.expression)

    #قانون نوع عملوندهای عملگرهای محاسباتی و منطقی
    def analyze_binary_op(self, binary_op):
        self.analyze(binary_op.left)
        self.analyze(binary_op.right)

        left_type = self.get_expression_type(binary_op.left)
        right_type = self.get_expression_type(binary_op.right)

        if binary_op.operator == 'T_AOp_IDX':
            if left_type != 'T_Int':
                raise Exception("Array index must be of type 'T_Int'.")
        elif binary_op.operator in ['=', 'T_AOp_PL', 'T_AOp_MN', 'T_AOp_ML', 'T_AOp_DV', 'T_AOp_RM']:
            if left_type != 'T_Int' or right_type != 'T_Int':
                raise Exception("Type mismatch: Expected 'T_Int' operands for arithmetic operation.")
        elif binary_op.operator in ['T_ROp_L', 'T_ROp_G', 'T_ROp_LE', 'T_ROp_GE', 'T_ROp_NE', 'T_ROp_E']:
            if left_type != 'T_Int' or right_type != 'T_Int':
                raise Exception("Type mismatch: Expected 'T_Int' operands for relational operation.")
        elif binary_op.operator in ['T_LOp_AND', 'T_LOp_OR']:
            if left_type != 'T_Bool' or right_type != 'T_Bool':
                raise Exception("Type mismatch: Expected 'T_Bool' operands for logical operation.")

    #قانون تعریف شناسه قبل از استفاده
    def analyze_identifier(self, identifier):
        if not self.get_identifier_type(identifier):
            raise Exception(f"Semantic Error: Identifier '{identifier.name}' is not defined.")

    def analyze_integer(self, integer):
        pass

    def analyze_string(self, string):
        pass

    def get_identifier_type(self, identifier):
        for scope in reversed(self.current_scope):
            if identifier.name in scope:
                return scope[identifier.name]
        if identifier.name in self.symbol_table:
            return self.symbol_table[identifier.name]['type']
        return None

    def get_expression_type(self, expression):
        if isinstance(expression, Identifier):
            return self.get_identifier_type(expression)
        elif isinstance(expression, Integer):
            return 'T_Int'
        elif isinstance(expression, String):
            return 'T_String'
        elif isinstance(expression, BinaryOp):
            left_type = self.get_expression_type(expression.left)
            right_type = self.get_expression_type(expression.right)
            if expression.operator in ['=', 'T_AOp_PL', 'T_AOp_MN', 'T_AOp_ML', 'T_AOp_DV', 'T_AOp_RM']:
                return 'T_Int'
            elif expression.operator in ['T_ROp_L', 'T_ROp_G', 'T_ROp_LE', 'T_ROp_GE', 'T_ROp_NE', 'T_ROp_E']:
                return 'T_Bool'
            elif expression.operator in ['T_LOp_AND', 'T_LOp_OR']:
                return 'T_Bool'
        else:
            raise Exception(f"Unsupported expression type: {type(expression)}")


# Example usage:
def run_compiler(source_code):
    lexer = LexicalAnalyzer(source_code)
    tokens = lexer.tokenize()

    parser = Parser(tokens)
    try:
        ast = parser.parse_program()
        print("Parsing Successful")
        print_ast(ast)

        # Semantic analysis
        semantic_analyzer = SemanticAnalyzer()
        semantic_analyzer.analyze(ast)
        print("Semantic Analysis Successful")

        return ast
    except SyntaxError as e:
        print(f"Syntax Error: {e}")
    except Exception as e:
        print(f"Semantic Error: {e}")


source_code = """

int main() {
    int b = 10;
    bool f = false ;
    int a = 5;
    int arr[0];
    bool condition1 = a < b; 
    bool condition2 = a == b;
    bool result = condition1 && condition2; 
    if (result) {
        int a ;
        print("Both conditions are true");
    } else {
        print("At least one condition is false");
    }

    return 0;
}


"""

tree = run_compiler(source_code)
