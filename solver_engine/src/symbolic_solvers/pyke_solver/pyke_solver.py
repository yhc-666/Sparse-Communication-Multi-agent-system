import os
import sys
import time
import random
# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pyke import knowledge_engine
import re
from symbolic_solvers.pyke_solver.pyke_trace import patch_pyke, unpatch_pyke, tracer

class Pyke_Program:
    def __init__(self, logic_program: str, dataset_name='ProntoQA') -> None:
        """
        Args:
            logic_program (str): SL, including Predicates, Facts, Rules, Query
            dataset_name (str): dataset name, support 'ProntoQA' and 'ProofWriter'
        """
        self.logic_program = logic_program
        self.flag = self.parse_logic_program()  # parse SL, return whether success
        self.dataset_name = dataset_name
        
        # create the folder to save the Pyke program
        cache_dir = os.path.join(os.path.dirname(__file__), 'cache_program')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir

        try:
            self.create_fact_file(self.Facts)
            self.create_rule_file(self.Rules)
            self.flag = True
        except:
            self.flag = False

        # answer mapping function for different datasets
        self.answer_map = {'ProntoQA': self.answer_map_prontoqa, 
                           'ProofWriter': self.answer_map_proofwriter,
                           'FOLIO': self.answer_map_folio}

    def parse_logic_program(self):
        """
        parse SL, extract each component
        
        SL format example:
        Predicates: ... 
        Facts: ... 
        Rules: ... 
        Query: ... 
        
        Returns:
            bool: 解析是否成功
        """
        keywords = ['Query:', 'Rules:', 'Facts:', 'Predicates:']
        program_str = self.logic_program

        for keyword in keywords:
            try:
                program_str, segment_list = self._parse_segment(program_str, keyword)
                cleaned = [line.split(':::')[0].strip() for line in segment_list]
                setattr(self, keyword[:-1], cleaned)
                setattr(self, keyword[:-1] + '_full', segment_list)
            except:
                setattr(self, keyword[:-1], None)
                setattr(self, keyword[:-1] + '_full', None)

        return self.validate_program()

    def _parse_segment(self, program_str, key_phrase):
        """
        parse specific segment in program
        
        Args:
            program_str (str): program string
            key_phrase (str): key phrase (e.g., 'Facts:')
            
        Returns:
            tuple: (remaining program string, parsed segment list)
        """
        remain_program_str, segment = program_str.split(key_phrase)
        segment_list = segment.strip().split('\n')
        return remain_program_str, segment_list

    def validate_program(self):
        if not self.Rules is None and not self.Facts is None:
            if not self.Rules[0] == '' and not self.Facts[0] == '':
                return True
                
        tmp_rules = []
        tmp_facts = []
        statements = self.Facts if self.Facts is not None else self.Rules
        if statements is None:
            return False
        
        for fact in statements:
            if fact.find('>>>') >= 0:
                tmp_rules.append(fact)
            else:
                tmp_facts.append(fact)
        self.Rules = tmp_rules
        self.Facts = tmp_facts
        return False
    
    def create_fact_file(self, facts):
        with open(os.path.join(self.cache_dir, 'facts.kfb'), 'w') as f:
            for fact in facts:
                # 过滤掉包含变量$x的无效事实
                if not fact.find('$x') >= 0:
                    f.write(fact + '\n')

    def create_rule_file(self, rules):
        pyke_rules = []
        for idx, rule in enumerate(rules):
            pyke_rules.append(self.parse_forward_rule(idx + 1, rule))

        with open(os.path.join(self.cache_dir, 'rules.krb'), 'w') as f:
            f.write('\n\n'.join(pyke_rules))

    def parse_forward_rule(self, f_index, rule):
        """
        parse forward reasoning rule, convert to Pyke format
        
        example rule: Furry($x, True) && Quite($x, True) >>> White($x, True)
        convert to Pyke's foreach-assert format

        Args:
            f_index (int): rule index
            rule (str): original rule string
            
        Returns:
            str: Pyke format rule
        """
        premise, conclusion = rule.split('>>>')
        premise = premise.strip()
        # 移除前提两端的括号（如果存在）
        if premise.startswith('(') and premise.endswith(')'):
            premise = premise[1:-1].strip()
        # 将前提条件按'&&'分割成多个条件
        premise = premise.split('&&')
        premise_list = [p.strip() for p in premise]

        conclusion = conclusion.strip()
        # 移除结论两端的括号（如果存在）
        if conclusion.startswith('(') and conclusion.endswith(')'):
            conclusion = conclusion[1:-1].strip()
        # 将结论按'&&'分割成多个结论
        conclusion = conclusion.split('&&')
        conclusion_list = [c.strip() for c in conclusion]

        # 创建Pyke规则格式
        pyke_rule = f'''rule{f_index}\n\tforeach'''
        # 添加前提条件
        for p in premise_list:
            pyke_rule += f'''\n\t\tfacts.{p}'''
        pyke_rule += f'''\n\tassert'''
        # 添加结论
        for c in conclusion_list:
            pyke_rule += f'''\n\t\tfacts.{c}'''
        return pyke_rule
    
    def check_specific_predicate(self, subject_name, predicate_name, engine):
        """
        check specific predicate value of specific subject
        
        example: check if Marvin is from Mars
        Query: FromMars(Marvin, $label)
        
        Args:
            subject_name
            predicate_name
            engine
            
        Returns:
            reasoning result (True/False/None)
        """
        results = []
        
        # 在事实库中查找
        with engine.prove_goal(f'facts.{predicate_name}({subject_name}, $label)') as gen:
            for vars, plan in gen:
                results.append(vars['label'])

        # 在规则库中查找
        with engine.prove_goal(f'rules.{predicate_name}({subject_name}, $label)') as gen:
            for vars, plan in gen:
                results.append(vars['label'])

        # 处理结果
        if len(results) == 1:
            return results[0]
        elif len(results) == 2:
            return results[0] and results[1]
        elif len(results) == 0:
            return None

    def parse_query(self, query):
        """
        解析查询语句
        
        输入示例: Metallic(Wren, False)
        
        Args:
            query (str): 查询字符串
            
        Returns:
            tuple: (函数名, 参数1, 参数2的布尔值)
        """
        pattern = r'(\w+)\(([^,]+),\s*([^)]+)\)'
        match = re.match(pattern, query)
        if match:
            function_name = match.group(1)
            arg1 = match.group(2)
            arg2 = match.group(3)
            arg2 = True if arg2 == 'True' else False
            return function_name, arg1, arg2
        else:
            raise ValueError(f'Invalid query: {query}')

    def execute_program_wo_reasoning(self, cleanup=True):
        """
        执行逻辑程序，进行推理
        
        Args:
            cleanup (bool): Whether to cleanup directories after execution
        
        Returns:
            tuple: (答案, 错误信息)
        """
        answer = None
        try:
            # 初始化Pyke推理引擎
            engine = knowledge_engine.engine(self.cache_dir)
            engine.reset()
            engine.activate('rules')  # 激活规则
            engine.get_kb('facts')    # 加载事实

            # 根据数据集类型处理查询
            if self.dataset_name == 'LogicalDeduction':
                # LogicalDeduction: 处理多个查询，找出所有True的选项
                true_options = []
                # Generate option letters dynamically based on number of queries
                option_letters = [chr(ord('A') + i) for i in range(len(self.Query))]
                
                for i, query in enumerate(self.Query):
                    predicate, subject, value_to_check = self.parse_query(query)
                    result = self.check_specific_predicate(subject, predicate, engine)
                    
                    # 如果结果匹配要检查的值，则该选项为True
                    if result == value_to_check:
                        true_options.append(option_letters[i])
                
                # 从True选项中随机选择，如果没有True选项则随机选择
                if true_options:
                    answer = random.choice(true_options)
                else:
                    answer = random.choice(option_letters)
            else:
                # 其他数据集: 处理单个查询
                predicate, subject, value_to_check = self.parse_query(self.Query[0])
                result = self.check_specific_predicate(subject, predicate, engine)
                
                # 根据数据集类型映射答案
                answer = self.answer_map[self.dataset_name](result, value_to_check)
            
            return answer, ""
        except Exception as e:
            return None, e
        finally:
            # 只在需要时清理
            if cleanup:
                self.cleanup_directories()

    def answer_mapping(self, answer):
        """答案映射函数（基础版本）"""
        return answer
        
    def answer_map_prontoqa(self, result, value_to_check):
        """
        ProntoQA数据集的答案映射
        
        Args:
            result: 推理结果
            value_to_check: 要检查的值
            
        Returns:
            str: 'A'（正确）或'B'（错误）
        """
        if result == value_to_check:
            return 'A'
        else:
            return 'B'

    def answer_map_proofwriter(self, result, value_to_check):
        """
        ProofWriter数据集的答案映射
        
        Args:
            result: 推理结果  
            value_to_check: 要检查的值
            
        Returns:
            str: 'A'（正确）、'B'（错误）或'C'（未知）
        """
        if result is None:
            return 'C'  # 未知
        elif result == value_to_check:
            return 'A'  # 正确
        else:
            return 'B'  # 错误

    def answer_map_folio(self, result, value_to_check):
        """
        FOLIO数据集的答案映射
        
        Args:
            result: 推理结果  
            value_to_check: 要检查的值
            
        Returns:
            str: 'A'（正确）、'B'（错误）或'C'（未知）
        """
        if result is None:
            return 'C'  # 未知
        elif result == value_to_check:
            return 'A'  # 正确
        else:
            return 'B'  # 错误

    def cleanup_directories(self):
        """
        清理缓存和编译目录
        """
        # 清理 compiled_krb 目录
        compiled_krb_dir = 'solver_engine/src/compiled_krb'
        if os.path.exists(compiled_krb_dir):
            print('removing compiled_krb')
            for file in os.listdir(compiled_krb_dir):
                if file != '__init__.py':
                    file_path = os.path.join(compiled_krb_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        os.system(f'rm -rf {file_path}')
        
        # 清理 cache 目录
        if os.path.exists(self.cache_dir):
            print('removing cache_program')
            os.system(f'rm -rf {self.cache_dir}/*')


    # ------------------------------------------------------------------
    # Functions for revealing solver reasoning process using Pyke tracing

    def execute_program(self):
        rule_map = {f"rule{i+1}": r.split(':::')[0].strip() for i, r in enumerate(self.Rules_full)}
        
        if self.dataset_name == 'LogicalDeduction':
            # For LogicalDeduction, first determine the chosen option without tracing
            answer, msg = self.execute_program_wo_reasoning(cleanup=False)
            
            # If answer is None, randomly choose an option
            option_letters = [chr(ord('A') + i) for i in range(len(self.Query))]
            if answer is None or answer not in option_letters:
                answer = random.choice(option_letters)
            
            # Then generate reasoning only for the chosen option
            patch_pyke(rule_map)
            
            # Find which query corresponds to the chosen option
            chosen_index = option_letters.index(answer)
            chosen_query = self.Query[chosen_index]
            
            # Generate reasoning for only the chosen query
            try:
                # Clear directories before reasoning
                self.cleanup_directories()
                
                # Recreate files for reasoning
                self.create_fact_file(self.Facts)
                self.create_rule_file(self.Rules)
                
                # 初始化Pyke推理引擎
                engine = knowledge_engine.engine(self.cache_dir)
                engine.reset()
                engine.activate('rules')  # 激活规则
                engine.get_kb('facts')    # 加载事实
                
                # 只处理选中的查询
                predicate, subject, value_to_check = self.parse_query(chosen_query)
                result = self.check_specific_predicate(subject, predicate, engine)
                
                reasoning = tracer.events
                if tracer.new_facts:
                    reasoning.append("All newly implied Facts: " + ', '.join(sorted(tracer.new_facts)))
                else:
                    reasoning.append("All newly implied Facts: None")
                self.reasoning_process = reasoning
                
                return answer, msg, self.build_reasoning_string(reasoning)
            except Exception as e:
                return answer, str(e), ""
            finally:
                unpatch_pyke()
                self.cleanup_directories()
        else:
            # For other datasets, use the original logic
            try:
                patch_pyke(rule_map)
                answer, msg = self.execute_program_wo_reasoning()
                reasoning = tracer.events
                if tracer.new_facts:
                    reasoning.append("All newly implied Facts: " + ', '.join(sorted(tracer.new_facts)))
                else:
                    reasoning.append("All newly implied Facts: None")
                self.reasoning_process = reasoning
                return answer, msg, self.build_reasoning_string(reasoning)
            finally:
                unpatch_pyke()

    def build_reasoning_string(self, reasoning):
        lines = []
        lines.append("We first define following predicates and corresponding natural language explanations:")
        for p in self.Predicates_full:
            lines.append(f"  {p}")
        lines.append("We have following known facts from the context:")
        for f in self.Facts_full:
            lines.append(f"  {f}")
        lines.append("We have following known rules from the context:")
        for i, r in enumerate(self.Rules_full):
            rule_txt = r.split(':::')[0].strip()
            lines.append(f"  rule{i+1}: {rule_txt}")
        lines.append("Now begin reasoning to obtain all implied facts:")
        lines.extend(reasoning)
        lines.append("Finish reasoning")
        return '\n'.join(lines)


if __name__ == "__main__":
    # test pyke solver

    logic_program = """Predicates:
    Round($x, bool) ::: Is x round?
    Red($x, bool) ::: Is x red?
    Smart($x, bool) ::: Is x smart?
    Furry($x, bool) ::: Is x furry?
    Rough($x, bool) ::: Is x rough?
    Big($x, bool) ::: Is x big?
    White($x, bool) ::: Is x white?
    
    Facts:
    Round(Anne, True) ::: Anne is round.
    Red(Bob, True) ::: Bob is red.
    Smart(Bob, True) ::: Bob is smart.
    Furry(Erin, True) ::: Erin is furry.
    Red(Erin, True) ::: Erin is red.
    Rough(Erin, True) ::: Erin is rough.
    Smart(Erin, True) ::: Erin is smart.
    Big(Fiona, True) ::: Fiona is big.
    Furry(Fiona, True) ::: Fiona is furry.
    Smart(Fiona, True) ::: Fiona is smart.
    
    Rules:
    Smart($x, True) >>> Furry($x, True) ::: All smart things are furry.
    Furry($x, True) >>> Red($x, True) ::: All furry things are red.
    Round($x, True) >>> Rough($x, True) ::: All round things are rough.
    White(Bob, True) >>> Furry(Bob, True) ::: If Bob is white then Bob is furry.
    Red($x, True) && Rough($x, True) >>> Big($x, True) ::: All red, rough things are big.
    Rough($x, True) >>> Smart($x, True) ::: All rough things are smart.
    Furry(Fiona, True) >>> Red(Fiona, True) ::: If Fiona is furry then Fiona is red.
    Round(Bob, True) && Big(Bob, True) >>> Furry(Bob, True) ::: If Bob is round and Bob is big then Bob is furry.
    Red(Fiona, True) && White(Fiona, True) >>> Smart(Fiona, True) ::: If Fiona is red and Fiona is white then Fiona is smart.
    
    Query:
    White(Bob, False) ::: Bob is not white."""

    # Answer: A
    logic_program1 = "Predicates:\nCold($x, bool) ::: Is x cold?\nQuiet($x, bool) ::: Is x quiet?\nRed($x, bool) ::: Is x red?\nSmart($x, bool) ::: Is x smart?\nKind($x, bool) ::: Is x kind?\nRough($x, bool) ::: Is x rough?\nRound($x, bool) ::: Is x round?\n\nFacts:\nCold(Bob, True) ::: Bob is cold.\nQuiet(Bob, True) ::: Bob is quiet.\nRed(Bob, True) ::: Bob is red.\nSmart(Bob, True) ::: Bob is smart.\nKind(Charlie, True) ::: Charlie is kind.\nQuiet(Charlie, True) ::: Charlie is quiet.\nRed(Charlie, True) ::: Charlie is red.\nRough(Charlie, True) ::: Charlie is rough.\nCold(Dave, True) ::: Dave is cold.\nKind(Dave, True) ::: Dave is kind.\nSmart(Dave, True) ::: Dave is smart.\nQuiet(Fiona, True) ::: Fiona is quiet.\n\nRules:\nQuiet($x, True) && Cold($x, True) >>> Smart($x, True) ::: If something is quiet and cold then it is smart.\nRed($x, True) && Cold($x, True) >>> Round($x, True) ::: Red, cold things are round.\nKind($x, True) && Rough($x, True) >>> Red($x, True) ::: If something is kind and rough then it is red.\nQuiet($x, True) >>> Rough($x, True) ::: All quiet things are rough.\nCold($x, True) && Smart($x, True) >>> Red($x, True) ::: Cold, smart things are red.\nRough($x, True) >>> Cold($x, True) ::: If something is rough then it is cold.\nRed($x, True) >>> Rough($x, True) ::: All red things are rough.\nSmart(Dave, True) && Kind(Dave, True) >>> Quiet(Dave, True) ::: If Dave is smart and Dave is kind then Dave is quiet.\n\nQuery:\nKind(Charlie, True) ::: Charlie is kind."

    # Answer: B
    logic_program2 = "Predicates:\nFurry($x, bool) ::: Is x furry?\nNice($x, bool) ::: Is x nice?\nSmart($x, bool) ::: Is x smart?\nYoung($x, bool) ::: Is x young?\nGreen($x, bool) ::: Is x green?\nBig($x, bool) ::: Is x big?\nRound($x, bool) ::: Is x round?\n\nFacts:\nFurry(Anne, True) ::: Anne is furry.\nNice(Anne, True) ::: Anne is nice.\nSmart(Anne, True) ::: Anne is smart.\nYoung(Bob, True) ::: Bob is young.\nNice(Erin, True) ::: Erin is nice.\nSmart(Harry, True) ::: Harry is smart.\nYoung(Harry, True) ::: Harry is young.\n\nRules:\nYoung($x, True) >>> Furry($x, True) ::: Young things are furry.\nNice($x, True) && Furry($x, True) >>> Green($x, True) ::: Nice, furry things are green.\nGreen($x, True) >>> Nice($x, True) ::: All green things are nice.\nNice($x, True) && Green($x, True) >>> Big($x, True) ::: Nice, green things are big.\nGreen($x, True) >>> Smart($x, True) ::: All green things are smart.\nBig($x, True) && Young($x, True) >>> Round($x, True) ::: If something is big and young then it is round.\nGreen($x, True) >>> Big($x, True) ::: All green things are big.\nYoung(Harry, True) >>> Furry(Harry, True) ::: If Harry is young then Harry is furry.\nFurry($x, True) && Smart($x, True) >>> Nice($x, True) ::: Furry, smart things are nice.\n\nQuery:\nGreen(Harry, False) ::: Harry is not green."

    # Answer: C
    logic_program3 = "Predicates:\nChases($x, $y, bool) ::: Does x chase y?\nRough($x, bool) ::: Is x rough?\nYoung($x, bool) ::: Is x young?\nNeeds($x, $y, bool) ::: Does x need y?\nGreen($x, bool) ::: Is x green?\nLikes($x, $y, bool) ::: Does x like y?\nBlue($x, bool) ::: Is x blue?\nRound($x, bool) ::: Is x round?\n\nFacts:\nChases(Cat, Lion, True) ::: The cat chases the lion.\nRough(Cat, True) ::: The cat is rough.\nYoung(Cat, True) ::: The cat is young.\nNeeds(Cat, Lion, True) ::: The cat needs the lion.\nNeeds(Cat, Rabbit, True) ::: The cat needs the rabbit.\nGreen(Dog, True) ::: The dog is green.\nYoung(Dog, True) ::: The dog is young.\nLikes(Dog, Cat, True) ::: The dog likes the cat.\nBlue(Lion, True) ::: The lion is blue.\nGreen(Lion, True) ::: The lion is green.\nChases(Rabbit, Lion, True) ::: The rabbit chases the lion.\nBlue(Rabbit, True) ::: The rabbit is blue.\nRough(Rabbit, True) ::: The rabbit is rough.\nLikes(Rabbit, Dog, True) ::: The rabbit likes the dog.\nNeeds(Rabbit, Dog, True) ::: The rabbit needs the dog.\nNeeds(Rabbit, Lion, True) ::: The rabbit needs the lion.\n\nRules:\nChases($x, Lion, True) >>> Round($x, True) ::: If someone chases the lion then they are round.\nNeeds(Lion, Rabbit, True) && Chases(Rabbit, Dog, True) >>> Likes(Lion, Dog, True) ::: If the lion needs the rabbit and the rabbit chases the dog then the lion likes the dog.\nRound($x, True) && Chases($x, Lion, True) >>> Needs($x, Cat, True) ::: If someone is round and they chase the lion then they need the cat.\nNeeds($x, Cat, True) && Chases($x, Dog, True) >>> Likes($x, Rabbit, True) ::: If someone needs the cat and they chase the dog then they like the rabbit.\nChases($x, Lion, True) && Blue(Lion, True) >>> Round(Lion, True) ::: If someone chases the lion and the lion is blue then the lion is round.\nChases($x, Rabbit, True) >>> Rough($x, True) ::: If someone chases the rabbit then they are rough.\nRough($x, True) && Likes($x, Rabbit, True) >>> Young(Rabbit, True) ::: If someone is rough and they like the rabbit then the rabbit is young.\nChases(Rabbit, Cat, True) && Needs(Cat, Lion, True) >>> Young(Rabbit, True) ::: If the rabbit chases the cat and the cat needs the lion then the rabbit is young.\nRound($x, True) && Needs($x, Cat, True) >>> Chases($x, Dog, True) ::: If someone is round and they need the cat then they chase the dog.\n\nQuery:\nLikes(Lion, Cat, False) ::: The lion does not like the cat."

    # Answer: A
    logic_program4 = "Predicates:\nFurry($x, bool) ::: Is x furry?\nNice($x, bool) ::: Is x nice?\n\nFacts:\nFurry(Anne, True) ::: Anne is furry.\n\nRules:\nFurry($x, True) >>> Nice($x, True) ::: All furry things are nice.\n\nQuery:\nNice(Anne, True) ::: Anne is nice."

    # Answer: B
    logic_program5 = "Predicates:\nFurry($x, bool) ::: Is x furry?\nNice($x, bool) ::: Is x nice?\n\nFacts:\nFurry(Anne, True) ::: Anne is furry.\n\nRules:\nFurry($x, True) >>> Nice($x, True) ::: All furry things are nice.\n\nQuery:\nNice(Anne, False) ::: Anne is not nice."

    # Answer: C
    logic_program6 = "Predicates:\nFurry($x, bool) ::: Is x furry?\nNice($x, bool) ::: Is x nice?\n\nFacts:\nFurry(Anne, True) ::: Anne is furry.\n\nRules:\nFurry($x, False) >>> Nice($x, True) ::: All non-furry things are nice.\n\nQuery:\nNice(Anne, True) ::: Anne is nice."

    # Answer: B
    logic_program7 = """Predicates:
Furry($x, bool) ::: Is x furry?
Nice($x, bool) ::: Is x nice?
Smart($x, bool) ::: Is x smart?
Young($x, bool) ::: Is x young?
Green($x, bool) ::: Is x green?
Big($x, bool) ::: Is x big?
Round($x, bool) ::: Is x round?

Facts:
Furry(Anne, True) ::: Anne is furry.
Nice(Anne, True) ::: Anne is nice.
Smart(Anne, True) ::: Anne is smart.
Young(Bob, True) ::: Bob is young.
Nice(Erin, True) ::: Erin is nice.
Smart(Harry, True) ::: Harry is smart.
Young(Harry, True) ::: Harry is young.

Rules:
Young($x, True) >>> Furry($x, True) ::: Young things are furry.
Nice($x, True) && Furry($x, True) >>> Green($x, True) ::: Nice, furry things are green.
Green($x, True) >>> Nice($x, True) ::: All green things are nice.
Nice($x, True) && Green($x, True) >>> Big($x, True) ::: Nice, green things are big.
Green($x, True) >>> Smart($x, True) ::: All green things are smart.
Big($x, True) && Young($x, True) >>> Round($x, True) ::: If something is big and young then it is round.
Green($x, True) >>> Big($x, True) ::: All green things are big.
Young(Harry, True) >>> Furry(Harry, True) ::: If Harry is young then Harry is furry.
Furry($x, True) && Smart($x, True) >>> Nice($x, True) ::: Furry, smart things are nice.

Query:
Green(Harry, False) ::: Harry is not green."""

    sample_logic_deduction = """ "id": "logical_deduction_35",
        "context": "The following paragraphs each describe a set of five objects arranged in a fixed order. The statements are logically consistent within each paragraph.\n\nOn a shelf, there are five books: a white book, an orange book, a yellow book, a blue book, and a red book. The yellow book is to the left of the white book. The red book is to the right of the blue book. The yellow book is to the right of the orange book. The blue book is to the right of the white book.",
        "question": "Which of the following is true?",
        "options": [
            "A) The white book is the second from the right.",
            "B) The orange book is the second from the right.",
            "C) The yellow book is the second from the right.",
            "D) The blue book is the second from the right.",
            "E) The red book is the second from the right."
        ],
        "answer": "D" """

    logic_program_logic_deduction = """
Predicates:
Book($x, bool)                  ::: $x is one of the five books.
LeftOf($x, $y, bool)            ::: Book $x is strictly to the left of book $y.
RightOf($x, $y, bool)           ::: Book $x is strictly to the right of book $y.
RightMost($x, bool)             ::: Book $x is the right‑most book on the shelf.
SecondFromRight($x, bool)       ::: Book $x is the second book from the right.
Facts:
Book(white,  True)              ::: The white book.
Book(orange, True)              ::: The orange book.
Book(yellow, True)              ::: The yellow book.
Book(blue,   True)              ::: The blue book.
Book(red,    True)              ::: The red book.
LeftOf(yellow, white,  True)    ::: The yellow book is to the left of the white book.
RightOf(red,   blue,   True)    ::: The red book is to the right of the blue book.
RightOf(yellow, orange, True)   ::: The yellow book is to the right of the orange book.
RightOf(blue,  white,  True)    ::: The blue book is to the right of the white book.
Rules:
LeftOf($a, $b, True) >>> RightOf($b, $a, True) ::: If $a is left of $b, then $b is right of $a.
RightOf($a, $b, True) >>> LeftOf($b, $a, True) ::: If $a is right of $b, then $b is left of $a.
RightOf($a, $b, True) && RightOf($b, $c, True) >>> RightOf($a, $c, True) ::: Right‑of is transitive.
RightOf($b, white,  True) && RightOf($b, orange, True) && RightOf($b, yellow, True) && RightOf($b, blue,   True)  >>> RightMost($b, True) ::: A book that is to the right of all the other four is the right‑most book.
RightMost($rm, True) && RightOf($rm, $s, True) && RightOf($s, white,  True) && RightOf($s, orange, True) && RightOf($s, yellow, True) >>> SecondFromRight($s, True) ::: The book immediately left of the right‑most—and still right of the remaining three—is second from the right.
Query:
SecondFromRight(white,  True)  ::: Option A
SecondFromRight(orange, True)  ::: Option B
SecondFromRight(yellow, True)  ::: Option C
SecondFromRight(blue,   True)  ::: Option D
SecondFromRight(red,    True)  ::: Option E
    """

    # SecondFromRight(white,  True)  ::: Option A
    # SecondFromRight(orange, True)  ::: Option B
    # SecondFromRight(yellow, True)  ::: Option C
    # SecondFromRight(blue,   True)  ::: Option D
    # SecondFromRight(red,    True)  ::: Option E

    logic_program_l = "Predicates:\nBook($x, bool)                  ::: $x is one of the five books.\nLeftOf($x, $y, bool)            ::: Book $x is strictly to the left of book $y.\nRightOf($x, $y, bool)           ::: Book $x is strictly to the right of book $y.\nSecondFromRight($x, bool)       ::: Book $x is the second book from the right.\nSecondFromLeft($x, bool)        ::: Book $x is the second book from the left.\nFacts:\nBook(green,  True)              ::: The green book.\nBook(blue,   True)              ::: The blue book.\nBook(white,  True)              ::: The white book.\nBook(purple, True)              ::: The purple book.\nBook(yellow, True)              ::: The yellow book.\nRightOf(blue, yellow, True)     ::: The blue book is to the right of the yellow book.\nLeftOf(white, yellow, True)     ::: The white book is to the left of the yellow book.\nSecondFromRight(blue, True)     ::: The blue book is the second from the right.\nSecondFromLeft(purple, True)    ::: The purple book is the second from the left.\nRules:\nLeftOf($a, $b, True) >>> RightOf($b, $a, True) ::: If $a is left of $b, then $b is right of $a.\nRightOf($a, $b, True) >>> LeftOf($b, $a, True) ::: If $a is right of $b, then $b is left of $a.\nRightOf($a, $b, True) && RightOf($b, $c, True) >>> RightOf($a, $c, True) ::: Right\u2011of is transitive.\n\nQuery:\nSecondFromLeft(green,  True)  ::: Option A\nSecondFromLeft(blue,   True)  ::: Option B\nSecondFromLeft(white,  True)  ::: Option C\nSecondFromLeft(purple, True)  ::: Option D\nSecondFromLeft(yellow, True)  ::: Option E"

    #tests = [logic_program1, logic_program2, logic_program3, logic_program4, logic_program5, logic_program6, logic_program7]
    tests = [logic_program_logic_deduction, logic_program_l]
   

    for test in tests:
        pyke_program = Pyke_Program(test, 'LogicalDeduction')
        result, error, reasoning = pyke_program.execute_program()
        print(f"Error: {error}")
        print(f"Result: {result}")
        #print(f"Reasoning: {reasoning}")

    compiled_krb_dir = os.path.join(os.path.dirname(__file__), 'compiled_krb')
    if os.path.exists(compiled_krb_dir):
        print('removing compiled_krb')
        os.system(f'rm -rf {compiled_krb_dir}')


