import os
import func_timeout
import re
import random
from collections import defaultdict

class CSP_Program:
    def __init__(self, logic_program:str, dataset_name:str) -> None:
        self.logic_program = logic_program
        self.flag = self.parse_logic_program()
        self.dataset_name = dataset_name
        self.timeout = 20

    def parse_logic_program(self):
        keywords = ['Query:', 'Constraints:', 'Variables:', 'Domain:']
        program_str = self.logic_program
        for keyword in keywords:
            try:
                program_str, segment_list = self._parse_segment(program_str, keyword)
                setattr(self, keyword[:-1], segment_list)
            except:
                setattr(self, keyword[:-1], None)
        
        # 检查必需的段落是否存在
        if self.Query is None or self.Constraints is None or self.Variables is None or self.Domain is None:
            return False
        
        # 验证变量格式：每个变量必须包含[IN]分隔符
        if hasattr(self, 'Variables') and self.Variables:
            for variable in self.Variables:
                if '[IN]' not in variable:
                    return False  # 格式不符合要求
        
        return True
    
    def _parse_segment(self, program_str, key_phrase):
        # 检查关键词是否存在
        if key_phrase not in program_str:
            return program_str, []  # 返回原字符串和空列表
        
        # 正常解析，限制分割次数为1
        remain_program_str, segment = program_str.split(key_phrase, 1)
        segment_list = [line.strip() for line in segment.strip().split('\n') if line.strip()]
        for i in range(len(segment_list)):
            segment_list[i] = segment_list[i].split(':::')[0].strip()
        return remain_program_str, segment_list

    def safe_execute(self, code_string: str, keys = None, debug_mode = False):
        def execute(x):
            try:
                exec(x)
                locals_ = locals()
                if keys is None:
                    return locals_.get('ans', None), ""
                else:
                    return [locals_.get(k, None) for k in keys], ""
            except Exception as e:
                if debug_mode:
                    print(e)
                return None, e
        try:
            ans, error_msg = func_timeout.func_timeout(self.timeout, execute, args=(code_string,))
        except func_timeout.FunctionTimedOut:
            ans = None
            error_msg = "timeout"

        return ans, error_msg

    # comparison (>, <), fixed value (==, !=), etc
    def parse_numeric_constraint(self, constraint):
        # get all the variables in the rule from left to right
        pattern = r'\b[a-zA-Z_]+\b'  # Matches word characters (letters and underscores)
        variables_in_rule = re.findall(pattern, constraint)
        unique_list = []
        for item in variables_in_rule:
            if item not in unique_list:
                unique_list.append(item)
        str_variables_in_rule = ', '.join(unique_list)
        str_variables_in_rule_with_quotes = ', '.join([f'"{v}"' for v in unique_list]) + ','
        parsed_constraint = f"lambda {str_variables_in_rule}: {constraint}, ({str_variables_in_rule_with_quotes})"
        return parsed_constraint
    
    # handle complex logical constraints with and/or/not
    def parse_logical_constraint(self, constraint):
        # Extract variables from the constraint, excluding Python keywords
        python_keywords = {'and', 'or', 'not', 'True', 'False', 'None', 'in', 'is'}
        pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'  # Matches variable names
        all_matches = re.findall(pattern, constraint)
        
        # Filter out Python keywords and get unique variables
        variables_in_rule = []
        for item in all_matches:
            if item not in python_keywords and item not in variables_in_rule:
                variables_in_rule.append(item)
        
        str_variables_in_rule = ', '.join(variables_in_rule)
        str_variables_in_rule_with_quotes = ', '.join([f'"{v}"' for v in variables_in_rule]) + ','
        parsed_constraint = f"lambda {str_variables_in_rule}: {constraint}, ({str_variables_in_rule_with_quotes})"
        return parsed_constraint
    
    # check if constraint contains logical operators
    def is_logical_constraint(self, constraint):
        logical_operators = [' or ', ' and ', ' not ', '(', ')']
        return any(op in constraint for op in logical_operators)
    
    # all different constraint
    def parse_all_different_constraint(self, constraint):
        pattern = r'AllDifferentConstraint\(\[(.*?)\]\)'
        # Extract the content inside the parentheses
        result = re.search(pattern, constraint)
        if result:
            values_str = result.group(1)
            values = [value.strip() for value in values_str.split(',')]
        else:
            return None
        parsed_constraint = f"AllDifferentConstraint(), {str(values)}"
        return parsed_constraint

    def execute_program(self, debug_mode = False):
        # 直接导入tracer模块
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        try:
            import tracer
            tracer.enable_tracing()
            tracer.clear_constraint_rules()
        except ImportError:
            # 如果无法导入tracer，返回错误
            return None, "Failed to import tracer module", ""
        
        # parse the logic program into CSP python program
        python_program_list = [
            'from constraint import *', 
            'problem = Problem()'
        ]
        
        # 将原始约束规则传递给tracer
        constraint_rules = {}
        constraint_index = 0
        
        # add variables
        for variable in self.Variables:
            variable_name, variable_domain = variable.split('[IN]')
            variable_name, variable_domain = variable_name.strip(), variable_domain.strip()
            # variable_domain = ast.literal_eval(variable_domain)
            python_program_list.append(f'problem.addVariable("{variable_name}", {variable_domain})')
        
        # add constraints
        for rule in self.Constraints:
            rule = rule.strip()
            parsed_constraint = None
            if rule.startswith('AllDifferentConstraint'):
                parsed_constraint = self.parse_all_different_constraint(rule)
            elif self.is_logical_constraint(rule):
                parsed_constraint = self.parse_logical_constraint(rule)
            else:
                parsed_constraint = self.parse_numeric_constraint(rule)
            
            # 存储原始规则与约束的映射
            constraint_rules[constraint_index] = rule
            tracer.set_constraint_rule(constraint_index, rule)
            
            # create the constraint with index
            python_program_list.append(f'problem.addConstraint({parsed_constraint})')
            constraint_index += 1
        
        # solve the problem
        python_program_list.append(f'ans = problem.getSolutions()')
        # execute the python program
        py_program_str = '\n'.join(python_program_list)
        if debug_mode:
            print(py_program_str)
        
        result, err_msg = self.safe_execute(py_program_str, keys=["ans"], debug_mode=debug_mode)
        if result is None:
            return None, err_msg, ""
        
        ans = result[0] if isinstance(result, list) else result
        reasoning = tracer.get_trace()
        reasoning = tracer.trace_to_text(reasoning)
        return ans, err_msg, reasoning
    
    def answer_mapping(self, answer):
        """
        answer是一个解的列表，每个解是一个字典，字典的key是变量名，value是变量的值。
        来自于problem.getSolutions()的返回值：
        answer = [
            {'variable1': value1, 'variable2': value2, ...},  # 解1
            {'variable1': value1, 'variable2': value2, ...},  # 解2
            ...
        ]

        """
        # Use LogicalDeduction logic for all datasets
        self.option_pattern = r'^\w+\)' # 用于匹配选项标识符: "A)", "B)", "C)" 等
        self.expression_pattern = r'\w+ == \d+'   # 用于匹配变量等式: "T == 1", "T == 0" 等

        variable_ans_map = defaultdict(set)
        if answer is not None:
            for result in answer:
                for variable, value in result.items():
                    variable_ans_map[variable].add(value)

        # Check each query option
        for option_str in self.Query:
            # Extract the option using regex
            option_match = re.match(self.option_pattern, option_str)
            if option_match:
                option = option_match.group().replace(')', '')
                # Extract the expression using regex
                expression_match = re.search(self.expression_pattern, option_str)
                if expression_match:
                    expression_str = expression_match.group()
                    # Extract the variable and its value
                    variable, value = expression_str.split('==')
                    variable, value = variable.strip(), int(value.strip())
                    # Check if the variable is in the execution result
                    if len(variable_ans_map[variable]) == 1 and value in variable_ans_map[variable]:
                        return option

        # Dataset-specific fallback behavior when no match is found
        if self.dataset_name == 'ProofWriter':
            return 'C'  # Unknown
        elif self.dataset_name == 'ProntoQA':
            return random.choice(['A', 'B'])  # Random between A and B
        else:
            return None  # LogicalDeduction returns None
    
if __name__ == "__main__":
    logic_program = "Domain:\n1: leftmost\n5: rightmost\nVariables:\ngreen_book [IN] [1, 2, 3, 4, 5]\nblue_book [IN] [1, 2, 3, 4, 5]\nwhite_book [IN] [1, 2, 3, 4, 5]\npurple_book [IN] [1, 2, 3, 4, 5]\nyellow_book [IN] [1, 2, 3, 4, 5]\nConstraints:\nblue_book > yellow_book ::: The blue book is to the right of the yellow book.\nwhite_book < yellow_book ::: The white book is to the left of the yellow book.\nblue_book == 4 ::: The blue book is the second from the right.\npurple_book == 2 ::: The purple book is the second from the left.\nAllDifferentConstraint([green_book, blue_book, white_book, purple_book, yellow_book]) ::: All books have different values.\nQuery:\nA) green_book == 2 ::: The green book is the second from the left.\nB) blue_book == 2 ::: The blue book is the second from the left.\nC) white_book == 2 ::: The white book is the second from the left.\nD) purple_book == 2 ::: The purple book is the second from the left.\nE) yellow_book == 2 ::: The yellow book is the second from the left."
    proofwriter = """Domain:
[0, 1]
Variables:
anne_round  [IN] [0, 1]
anne_red    [IN] [0, 1]
anne_smart  [IN] [0, 1]
anne_furry  [IN] [0, 1]
anne_rough  [IN] [0, 1]
anne_big    [IN] [0, 1]
anne_white  [IN] [0, 1]
bob_round   [IN] [0, 1]
bob_red     [IN] [0, 1]
bob_smart   [IN] [0, 1]
bob_furry   [IN] [0, 1]
bob_rough   [IN] [0, 1]
bob_big     [IN] [0, 1]
bob_white   [IN] [0, 1]
erin_round  [IN] [0, 1]
erin_red    [IN] [0, 1]
erin_smart  [IN] [0, 1]
erin_furry  [IN] [0, 1]
erin_rough  [IN] [0, 1]
erin_big    [IN] [0, 1]
erin_white  [IN] [0, 1]
fiona_round [IN] [0, 1]
fiona_red   [IN] [0, 1]
fiona_smart [IN] [0, 1]
fiona_furry [IN] [0, 1]
fiona_rough [IN] [0, 1]
fiona_big   [IN] [0, 1]
fiona_white [IN] [0, 1]
Constraints:
anne_round == 1
bob_red   == 1
bob_smart == 1
erin_furry == 1
erin_red   == 1
erin_rough == 1
erin_smart == 1
fiona_big   == 1
fiona_furry == 1
fiona_smart == 1
(anne_smart == 0) or (anne_furry == 1)
(bob_smart  == 0) or (bob_furry  == 1)
(erin_smart == 0) or (erin_furry == 1)
(fiona_smart== 0) or (fiona_furry== 1)
(anne_furry == 0) or (anne_red == 1)
(bob_furry  == 0) or (bob_red  == 1)
(erin_furry == 0) or (erin_red == 1)
(fiona_furry== 0) or (fiona_red == 1)
(anne_round == 0) or (anne_rough == 1)
(bob_round  == 0) or (bob_rough  == 1)
(erin_round == 0) or (erin_rough == 1)
(fiona_round== 0) or (fiona_rough== 1)
(anne_red == 0) or (anne_rough == 0) or (anne_big == 1)
(bob_red  == 0) or (bob_rough  == 0) or (bob_big == 1)
(erin_red == 0) or (erin_rough == 0) or (erin_big == 1)
(fiona_red== 0) or (fiona_rough == 0) or (fiona_big == 1)
(anne_rough == 0) or (anne_smart == 1)
(bob_rough  == 0) or (bob_smart  == 1)
(erin_rough == 0) or (erin_smart == 1)
(fiona_rough== 0) or (fiona_smart == 1)
(bob_white == 0) or (bob_furry == 1)
(bob_round == 0) or (bob_big == 0) or (bob_furry == 1)
(fiona_furry == 0) or (fiona_red == 1)
(fiona_red == 0) or (fiona_white == 0) or (fiona_smart == 1)
Query:
A) bob_white == 0   ::: Bob is not white."""
    csp_program = CSP_Program(proofwriter, 'ProofWriter')
    ans, err_msg, reasoning = csp_program.execute_program(debug_mode=False)
    print("Answer:", ans)
    print("Error:", err_msg)
    print("Final answer:", csp_program.answer_mapping(ans))
    
    # 展示推理过程
    if reasoning:
        print("\nReasoning trace:")
        print(reasoning)


    