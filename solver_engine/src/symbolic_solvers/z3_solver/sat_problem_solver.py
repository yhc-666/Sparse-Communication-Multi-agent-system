from collections import OrderedDict
from .code_translator import *
import subprocess
from subprocess import check_output
from os.path import join
import os

class LSAT_Z3_Program:
    def __init__(self, logic_program:str, dataset_name:str) -> None:
        self.logic_program = logic_program
        self.dataset_name = dataset_name
        
        # create the folder to save the Pyke program (移到前面确保总是被设置)
        cache_dir = os.path.join(os.path.dirname(__file__), '.cache_program')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir
        
        try:
            self.parse_logic_program()
            self.standard_code = self.to_standard_code()
            self.flag = True
        except Exception as e:
            self.standard_code = None
            self.flag = False
            return

    def parse_logic_program(self):
        # 首先移除所有 ::: 注释
        cleaned_lines = []
        for line in self.logic_program.splitlines():
            if ':::' in line:
                cleaned_line = line.split(':::')[0].strip()
            else:
                cleaned_line = line.strip()
            if cleaned_line:  # 只保留非空行
                cleaned_lines.append(cleaned_line)
        
        # 重新组合成清理后的logic program
        cleaned_logic_program = '\n'.join(cleaned_lines)
        
        # split the logic program into different parts
        lines = [x for x in cleaned_logic_program.splitlines() if not x.strip() == ""]

        decleration_start_index = lines.index("# Declarations")
        constraint_start_index = lines.index("# Constraints")
        option_start_index = lines.index("# Options")
 
        declaration_statements = lines[decleration_start_index + 1:constraint_start_index]
        constraint_statements = lines[constraint_start_index + 1:option_start_index]
        option_statements = lines[option_start_index + 1:]

        try:
            (self.declared_enum_sorts, self.declared_int_sorts, self.declared_lists, self.declared_functions, self.variable_constrants) = self.parse_declaration_statements(declaration_statements)

            self.constraints = [x.strip() for x in constraint_statements]
            self.options = [x.strip() for x in option_statements if not x.startswith("Question")]
        except Exception as e:
            return False
        
        return True

    def __repr__(self):
        return f"LSATSatProblem:\n\tDeclared Enum Sorts: {self.declared_enum_sorts}\n\tDeclared Lists: {self.declared_lists}\n\tDeclared Functions: {self.declared_functions}\n\tConstraints: {self.constraints}\n\tOptions: {self.options}"

    def parse_declaration_statements(self, declaration_statements):
        enum_sort_declarations = OrderedDict()
        int_sort_declarations = OrderedDict()
        function_declarations = OrderedDict()
        pure_declaration_statements = [x for x in declaration_statements if "Sort" in x or "Function" in x]
        variable_constrant_statements = [x for x in declaration_statements if not "Sort" in x and not "Function" in x]
        for s in pure_declaration_statements:
            if "Function" in s:
                function_name = s.split("=")[0].strip()
                if "->" in s and "[" not in s:
                    function_args_str = s.split("=")[1].strip()[len("Function("):]
                    function_args_str = function_args_str.replace("->", ",").replace("(", "").replace(")", "")
                    function_args = [x.strip() for x in function_args_str.split(",")]
                    function_declarations[function_name] = function_args
                elif "->" in s and "[" in s:
                    function_args_str = s.split("=")[1].strip()[len("Function("):-1]
                    function_args_str = function_args_str.replace("->", ",").replace("[", "").replace("]", "")
                    function_args = [x.strip() for x in function_args_str.split(",")]
                    function_declarations[function_name] = function_args
                else:
                    # legacy way
                    function_args_str = s.split("=")[1].strip()[len("Function("):-1]
                    function_args = [x.strip() for x in function_args_str.split(",")]
                    function_declarations[function_name] = function_args
            elif "EnumSort" in s:
                sort_name = s.split("=")[0].strip()
                sort_member_str = s.split("=")[1].strip()[len("EnumSort("):-1]
                sort_members = [x.strip() for x in sort_member_str[1:-1].split(",")]
                enum_sort_declarations[sort_name] = sort_members
            elif "IntSort" in s:
                sort_name = s.split("=")[0].strip()
                sort_member_str = s.split("=")[1].strip()[len("IntSort("):-1]
                sort_members = [x.strip() for x in sort_member_str[1:-1].split(",")]
                int_sort_declarations[sort_name] = sort_members
            else:
                raise RuntimeError("Unknown declaration statement: {}".format(s))

        declared_enum_sorts = OrderedDict()
        declared_lists = OrderedDict()
        self.declared_int_lists = OrderedDict()

        declared_functions = function_declarations
        already_declared = set()
        for name, members in enum_sort_declarations.items():
            # all contained by other enum sorts
            if all([x not in already_declared for x in members]):
                declared_enum_sorts[name] = members
                already_declared.update(members)
            declared_lists[name] = members

        for name, members in int_sort_declarations.items():
            self.declared_int_lists[name] = members
            # declared_lists[name] = members

        return declared_enum_sorts, int_sort_declarations, declared_lists, declared_functions, variable_constrant_statements
    
    def to_standard_code(self):
        declaration_lines = []
        # translate enum sorts
        for name, members in self.declared_enum_sorts.items():
            declaration_lines += CodeTranslator.translate_enum_sort_declaration(name, members)

        # translate int sorts
        for name, members in self.declared_int_sorts.items():
            declaration_lines += CodeTranslator.translate_int_sort_declaration(name, members)

        # translate lists
        for name, members in self.declared_lists.items():
            declaration_lines += CodeTranslator.translate_list_declaration(name, members)

        scoped_list_to_type = {}
        for name, members in self.declared_lists.items():
            if all(x.isdigit() for x in members):
                scoped_list_to_type[name] = CodeTranslator.ListValType.INT
            else:
                scoped_list_to_type[name] = CodeTranslator.ListValType.ENUM

        for name, members in self.declared_int_lists.items():
            scoped_list_to_type[name] = CodeTranslator.ListValType.INT
        
        # translate functions
        for name, args in self.declared_functions.items():
            declaration_lines += CodeTranslator.translate_function_declaration(name, args)

        pre_condidtion_lines = []

        for constraint in self.constraints:
            pre_condidtion_lines += CodeTranslator.translate_constraint(constraint, scoped_list_to_type)

        # additional function scope control
        for name, args in self.declared_functions.items():
            if args[-1] in scoped_list_to_type and scoped_list_to_type[args[-1]] == CodeTranslator.ListValType.INT:
                # Get the list range
                if args[-1] in self.declared_int_lists:
                    list_range = self.declared_int_lists[args[-1]]
                else:
                    list_range = [int(x) for x in self.declared_lists[args[-1]]]
                
                list_range = [int(x) for x in list_range]
                assert list_range[-1] - list_range[0] == len(list_range) - 1
                scoped_vars = [x[0] + str(i) for i, x in enumerate(args[:-1])]
                func_call = f"{name}({', '.join(scoped_vars)})"

                additional_cons = "ForAll([{}], And({} <= {}, {} <= {}))".format(
                    ", ".join([f"{a}:{b}" for a, b in zip(scoped_vars, args[:-1])]),
                    list_range[0], func_call, func_call, list_range[-1]
                )
                pre_condidtion_lines += CodeTranslator.translate_constraint(additional_cons, scoped_list_to_type)

        # each block should express one option
        option_blocks = [CodeTranslator.translate_constraint(option, scoped_list_to_type) for option in self.options]

        return CodeTranslator.assemble_standard_code(declaration_lines, pre_condidtion_lines, option_blocks, len(self.options))
    
    def execute_program(self):
        # 检查解析是否成功，如果失败则返回错误信息
        if not self.flag or self.standard_code is None:
            return None, 'Logic program parsing failed', ""
        reasoning = "N/A"
            
        filename = join(self.cache_dir, f'tmp.py')
        with open(filename, "w") as f:
            f.write(self.standard_code)
        try:
            output = check_output(["python", filename], stderr=subprocess.STDOUT, timeout=1.0)
        except subprocess.CalledProcessError as e:
            outputs = e.output.decode("utf-8").strip().splitlines()[-1]
            return None, outputs, reasoning
        except subprocess.TimeoutExpired:
            return None, 'TimeoutError', reasoning
        output = output.decode("utf-8").strip()
        result = output.splitlines()
        if len(result) == 0:
            return None, 'No Output', reasoning
        
        return result, "", reasoning
    
    def answer_mapping(self, answer):
        cleaned_answer = answer[0].strip()
        
        if self.dataset_name == 'LogicalDeduction':
            # Dynamic mapping for LogicalDeduction based on option count
            num_options = len(self.options)
            mapping = {}
            for i in range(num_options):
                letter = chr(ord('A') + i)
                mapping[f'({letter})'] = letter
                mapping[letter] = letter
            return mapping[cleaned_answer]
        
        elif self.dataset_name in ['FOLIO', 'ProofWriter']:
            # If output is (A) or (B), map to A or B; otherwise map to C
            if cleaned_answer in ['(A)', 'A']:
                return 'A'
            elif cleaned_answer in ['(B)', 'B']:
                return 'B'
            else:
                return 'C'
        
        elif self.dataset_name == 'ProntoQA':
            # If output is (A) or (B), map to A or B
            if cleaned_answer in ['(A)', 'A']:
                return 'A'
            elif cleaned_answer in ['(B)', 'B']:
                return 'B'
            else:
                raise ValueError(f"Unexpected answer format for ProntoQA: {cleaned_answer}")
        
        else:
            # Fallback for unknown datasets - use original logic
            mapping = {'(A)': 'A', '(B)': 'B', '(C)': 'C', '(D)': 'D', '(E)': 'E',
                       'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E'}
            return mapping.get(cleaned_answer, cleaned_answer)

if __name__=="__main__":
    logic_program = '''# Declarations\nobjects = EnumSort([Bob, Charlie, Dave, Fiona])\nattributes = EnumSort([cold, quiet, red, smart, kind, rough, round])\nhas_attribute = Function([objects, attributes] -> [bool])\n# Constraints\nhas_attribute(Bob, cold) == True\nhas_attribute(Bob, quiet) == True\nhas_attribute(Bob, red) == True\nhas_attribute(Bob, smart) == True\nhas_attribute(Charlie, kind) == True\nhas_attribute(Charlie, quiet) == True\nhas_attribute(Charlie, red) == True\nhas_attribute(Charlie, rough) == True\nhas_attribute(Dave, cold) == True\nhas_attribute(Dave, kind) == True\nhas_attribute(Dave, smart) == True\nhas_attribute(Fiona, quiet) == True\nForAll([x:objects], Implies(And(has_attribute(x, quiet) == True, has_attribute(x, rough) == True))\nForAll([x:objects], Implies(And(has_attribute(x, quiet) == True, has_attribute(x, rough) == True))\nForAll([x:objects], Implies(And(has_attribute(x, red) == True, has_attribute(x, cold) == True), has_attribute(x, round) == True))\nForAll([x:objects], Implies(And(has_attribute(x, kind) == True, has_attribute(x, rough) == True), has_attribute(x, red) == True))\nForAll([x:objects], Implies(has_attribute(x, quiet) == True, has_attribute(x, rough) == True))\nForAll([x:objects], Implies(And(has_attribute(x, cold) == True, has_attribute(x, smart) == True), has_attribute(x, red) == True))\nForAll([x:objects], Implies(has_attribute(x, rough) == True, has_attribute(x, cold) == True))\nForAll([x:objects], Implies(has_attribute(x, red) == True, has_attribute(x, rough) == True))\nImplies(And(has_attribute(Dave, smart) == True, has_attribute(Dave, kind) == True), has_attribute(Dave, quiet) == True)\n# Options\nis_valid(has_attribute(Charlie, kind) == True)\nis_unsat(has_attribute(Charlie, kind) == True)'''


    sampel_proofwriter = """
Context："Bob is cold. Bob is quiet. Bob is red. Bob is smart. Charlie is kind. Charlie is quiet. Charlie is red. Charlie is rough. Dave is cold. Dave is kind. Dave is smart. Fiona is quiet. If something is quiet and cold then it is smart. Red, cold things are round. If something is kind and rough then it is red. All quiet things are rough. Cold, smart things are red. If something is rough then it is cold. All red things are rough. If Dave is smart and Dave is kind then Dave is quiet.",
Question: True or false: Charlie is kind.
"""

    proofwriter_program = '''# Declarations
objects = EnumSort([Bob, Charlie, Dave, Fiona])
attributes = EnumSort([cold, quiet, red, smart, kind, rough, round])
has_attribute = Function([objects, attributes] -> [bool])  
# Constraints
has_attribute(Bob, cold) == True ::: Bob is cold.
has_attribute(Bob, quiet) == True ::: Bob is quiet.
has_attribute(Bob, red) == True ::: Bob is red.
has_attribute(Bob, smart) == True ::: Bob is smart.
has_attribute(Charlie, kind) == True ::: Charlie is kind.
has_attribute(Charlie, quiet) == True ::: Charlie is quiet.
has_attribute(Charlie, red) == True ::: Charlie is red.
has_attribute(Charlie, rough) == True ::: Charlie is rough.
has_attribute(Dave, cold) == True ::: Dave is cold.
has_attribute(Dave, kind) == True ::: Dave is kind.
has_attribute(Dave, smart) == True ::: Dave is smart.
has_attribute(Fiona, quiet) == True ::: Fiona is quiet.
ForAll([x:objects], Implies(And(has_attribute(x, quiet) == True, has_attribute(x, cold) == True), has_attribute(x, smart) == True)) ::: If something is quiet and cold then it is smart.
ForAll([x:objects], Implies(And(has_attribute(x, red) == True, has_attribute(x, cold) == True), has_attribute(x, round) == True)) ::: Red, cold things are round.
ForAll([x:objects], Implies(And(has_attribute(x, kind) == True, has_attribute(x, rough) == True), has_attribute(x, red) == True)) ::: If something is kind and rough then it is red.
ForAll([x:objects], Implies(has_attribute(x, quiet) == True, has_attribute(x, rough) == True)) ::: All quiet things are rough.
ForAll([x:objects], Implies(And(has_attribute(x, cold) == True, has_attribute(x, smart) == True), has_attribute(x, red) == True)) ::: Cold, smart things are red.
ForAll([x:objects], Implies(has_attribute(x, rough) == True, has_attribute(x, cold) == True)) ::: If something is rough then it is cold.
ForAll([x:objects], Implies(has_attribute(x, red) == True, has_attribute(x, rough) == True)) ::: All red things are rough.
Implies(And(has_attribute(Dave, smart) == True, has_attribute(Dave, kind) == True), has_attribute(Dave, quiet) == True) ::: If Dave is smart and Dave is kind then Dave is quiet.
# Options
is_valid(has_attribute(Charlie, kind) == True) ::: Charlie is kind is True (A).
is_unsat(has_attribute(Charlie, kind) == True) ::: Charlie is kind is False (B).
'''


    sample_folio = """
    {
    "id": "FOLIO_dev_65",
    "context": "All cows are bovines. Some pets are cows. If something is a bovine, then it is domesticated. No domesticated animals are alligators. Ted is an aligator.",
    "question": "Based on the above information, is the following statement true, false, or uncertain? If Ted is a cow, then Ted is not a pet.",
    "options": [
      "A) True",
      "B) False",
      "C) Uncertain"
    ],
    "answer": "A"
  }
    """

    logic_program_folio = '''# Declarations
objects = EnumSort([Ted, W]) ::: domain of discourse (Ted and an existential witness W)
Cow = Function([objects] -> [bool]) ::: x is a cow
Bovine = Function([objects] -> [bool]) ::: x is a bovine
Pet = Function([objects] -> [bool]) ::: x is a pet
Domesticated = Function([objects] -> [bool]) ::: x is domesticated
Alligator = Function([objects] -> [bool]) ::: x is an alligator
# Constraints
ForAll([x:objects], Implies(Cow(x), Bovine(x))) ::: All cows are bovines
Exists([x:objects], And(Pet(x), Cow(x))) ::: Some pets are cows
ForAll([x:objects], Implies(Bovine(x), Domesticated(x))) ::: Bovines are domesticated
ForAll([x:objects], Implies(Domesticated(x), Not(Alligator(x)))) ::: No domesticated animals are alligators
Alligator(Ted) ::: Ted is an alligator
# Options
is_valid(Implies(Cow(Ted), Not(Pet(Ted)))) ::: A) True
is_valid(Not(Implies(Cow(Ted), Not(Pet(Ted))))) ::: B) False
'''


    sample_logic_deduction = """"id": "logical_deduction_35",
        "context": "The following paragraphs each describe a set of five objects arranged in a fixed order. The statements are logically consistent within each paragraph.\n\nOn a shelf, there are five books: a white book, an orange book, a yellow book, a blue book, and a red book. The yellow book is to the left of the white book. The red book is to the right of the blue book. The yellow book is to the right of the orange book. The blue book is to the right of the white book.",
        "question": "Which of the following is true?",
        "options": [
            "A) The white book is the second from the right.",
            "B) The orange book is the second from the right.",
            "C) The yellow book is the second from the right.",
            "D) The blue book is the second from the right.",
            "E) The red book is the second from the right."
        ],
        "answer": "D" 
    """


    logic_program_logic_deduction = """
# Declarations
objects = EnumSort([White, Orange, Yellow, Blue, Red])
positions = IntSort([1, 2, 3, 4, 5])
pos = Function([objects] -> [positions])
# Constraints
Distinct([b:objects], pos(b)) ::: Each book occupies a unique position
pos(Yellow) < pos(White) ::: The yellow book is to the left of the white book.
pos(Red) > pos(Blue) ::: The red book is to the right of the blue book.
pos(Yellow) > pos(Orange) ::: The yellow book is to the right of the orange book.
pos(Blue) > pos(White) ::: The blue book is to the right of the white book.
# Options
is_valid(pos(White) == 4)  ::: A) The white book is the second from the right.
is_valid(pos(Orange) == 4)  ::: B) The orange book is the second from the right.
is_valid(pos(Yellow) == 4)  ::: C) The yellow book is the second from the right.
is_valid(pos(Blue) == 4)  ::: D) The blue book is the second from the right.
is_valid(pos(Red) == 4)  ::: E) The red book is the second from the right.
    """


    test = "# Declarations\nfruits = EnumSort([Watermelons, Plums, Apples, Peaches, Kiwis])\nprice = Function([fruits] -> [IntSort()])\n# Constraints\nDistinct([f:fruits], price(f)) ::: Each fruit has a unique price\nprice(Apples) < price(Peaches) ::: The apples are less expensive than the peaches.\nprice(Plums) == 1 ::: The plums are the cheapest.\nprice(Kiwis) == 2 ::: The kiwis are the second-cheapest.\nprice(Peaches) < price(Watermelons) ::: The watermelons are more expensive than the peaches.\n# Options\nis_valid(price(Watermelons) == 5) ::: A) The watermelons are the most expensive.\nis_valid(price(Plums) == 5) ::: B) The plums are the most expensive.\nis_valid(price(Apples) == 5) ::: C) The apples are the most expensive.\nis_valid(price(Peaches) == 5) ::: D) The peaches are the most expensive.\nis_valid(price(Kiwis) == 5) ::: E) The kiwis are the most expensive."


    z3_program = LSAT_Z3_Program(test, 'LogicalDeduction')
    print(z3_program.standard_code)

    output, error_message, reasoning = z3_program.execute_program()
    print("Output:", output)
    print("Error message:", error_message)
    
    if output is not None:
        print("Answer mapping:", z3_program.answer_mapping(output))
    else:
        print("No valid output to map")