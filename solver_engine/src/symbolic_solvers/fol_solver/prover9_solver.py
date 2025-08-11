import os
import re
import sys
import random
from nltk.inference.prover9 import *
from nltk.sem.logic import NegatedExpression
import subprocess, shutil
import tempfile, textwrap, itertools as it

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..', '..')
sys.path.insert(0, project_root)

from src.symbolic_solvers.fol_solver.fol_prover9_parser import Prover9_FOL_Formula
from src.symbolic_solvers.fol_solver.Formula import FOL_Formula

# set the path to the prover9 executable
PROVER9_PATH = os.path.join(os.path.dirname(__file__), '..', 'Prover9', 'bin')
os.environ['PROVER9'] = PROVER9_PATH # Linux version
# os.environ['PROVER9'] = '/opt/homebrew/bin'  # macOS version installed via Homebrew


# --- helper utilities for raw prover9 interaction starts (for result with unkown)---
def _build_p9_input(assumptions: list[str], goal: str, max_seconds: int = 10) -> str:
    """Build a prover9 input string using NLTK's conversion utilities."""
    from nltk.inference.prover9 import Expression, convert_to_prover9

    ass_exprs = [Expression.fromstring(a) for a in assumptions]
    goal_expr = Expression.fromstring(goal)
    ass_strs = convert_to_prover9(ass_exprs)
    goal_str = convert_to_prover9(goal_expr)

    ass_block = "\n".join(a + "." for a in ass_strs)
    return textwrap.dedent(
        f"""
        assign(max_seconds,{max_seconds}).
        clear(auto_denials).

        formulas(assumptions).
        {ass_block}
        end_of_list.

        formulas(goals).
        {goal_str}.
        end_of_list.
    """
    )


def _run_prover9_raw(p9_input: str, timeout: int = 12) -> str:
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tf:
        tf.write(p9_input)
        tf.flush()
        cmd = [os.path.join(PROVER9_PATH, "prover9"), "-f", tf.name]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    os.unlink(tf.name)
    return proc.stdout


_LINE_PAT = re.compile(r"^(Derived:|kept:|given\s+#\d+|-\w|\w).*?\[.*\]")


def _clean_prefix(line: str) -> str:
    """Strip technical prefixes so duplicates are detected."""
    line = re.sub(r"^(Derived:|kept:|given\s+#\d+)\s*", "", line.strip())
    line = re.sub(r"^\d+\s+", "", line)
    return line


def _summarise_log(log: str, max_lines: int | None = None) -> str:
    seen, text_seen, selected = set(), set(), []
    mapping: dict[int, int] = {}
    key_to_index: dict[str, int] = {}
    for ln in log.splitlines():
        if ln.startswith(("Predicate symbol precedence",
                          "Function symbol precedence",
                          "given #")):
            continue
        if _LINE_PAT.match(ln):
            m = re.match(r"\s*(\d+)\s+(.*)", ln.strip())
            if m:
                orig_no, raw_output = int(m.group(1)), m.group(2)
            else:
                orig_no, raw_output = None, re.sub(r"^\d+\s+", "", ln.strip())
            if raw_output.startswith("kept:"):
                raw_output = re.sub(r"^kept:\s*\d+\s*", "kept: ", raw_output)
            dedup_key = " ".join(_clean_prefix(raw_output).split())
            text_key = _clean_prefix(raw_output).split("[", 1)[0].strip()
            if raw_output.startswith("kept:") and text_key in text_seen:
                continue
            if dedup_key not in seen:
                selected.append((orig_no, raw_output))
                idx = len(selected)
                if orig_no is not None:
                    mapping[orig_no] = idx
                seen.add(dedup_key)
                key_to_index[dedup_key] = idx
                text_seen.add(text_key)
            else:
                # duplicate line, map its original number to existing index
                if orig_no is not None and dedup_key in key_to_index:
                    mapping[orig_no] = key_to_index[dedup_key]
    if max_lines:
        selected = selected[:max_lines]
        # rebuild mapping for truncated output
        mapping = {orig: idx + 1 for idx, (orig, _) in enumerate(selected) if orig is not None}
    out = []
    for idx, (orig_no, ln) in enumerate(selected, 1):
        clause, label = ln.rsplit("[", 1)
        clause_part = re.sub(r"^\d+\s+", "", clause)
        label = re.sub(r'(?<=\(|,)\d+(?=\)|,)', lambda m: str(mapping.get(int(m.group(0)), int(m.group(0)))), '[' + label)
        out.append(f"{idx} {clause_part.strip()} {label}")
    reason = "-- Search terminated, no contradiction found --" if "sos_empty" in log else \
             "-- Timeout terminated, no contradiction found --" if "max_seconds" in log else \
             "-- Search terminated, no contradiction found --"
    return "\n".join(out) + f"\n{reason}"

# --- helper utilities for raw prover9 interaction ends ---



class FOL_Prover9_Program:
    def __init__(self, logic_program:str, dataset_name = 'FOLIO') -> None:
        self.logic_program = logic_program
        self.dataset_name = dataset_name
        self.flag = self.parse_logic_program()

    def parse_logic_program(self):
        try:        
            # Handle LogicalDeduction dataset separately
            if self.dataset_name == 'LogicalDeduction':
                return self._parse_logical_deduction_program()
            
            # Original parsing logic for other datasets
            # Split the string into premises and conclusion
            premises_string = self.logic_program.split("Conclusion:")[0].split("Premises:")[1].strip()
            conclusion_string = self.logic_program.split("Conclusion:")[1].strip()

            # Extract each premise and the conclusion using regex
            premises = premises_string.strip().split('\n')
            conclusion = conclusion_string.strip().split('\n')

            self.logic_premises = [premise.split(':::')[0].strip() for premise in premises]
            self.logic_conclusion = conclusion[0].split(':::')[0].strip()

            # convert to prover9 format
            self.prover9_premises = []
            for premise in self.logic_premises:
                fol_rule = FOL_Formula(premise)
                if fol_rule.is_valid == False:
                    return False
                prover9_rule = Prover9_FOL_Formula(fol_rule)
                self.prover9_premises.append(prover9_rule.formula)

            fol_conclusion = FOL_Formula(self.logic_conclusion)
            if fol_conclusion.is_valid == False:
                return False
            self.prover9_conclusion = Prover9_FOL_Formula(fol_conclusion).formula
            return True
        except:
            return False

    def _parse_logical_deduction_program(self):
        """Parse LogicalDeduction multi-choice format separately"""
        try:
            # Split the string into premises and conclusion
            premises_string = self.logic_program.split("Conclusion:")[0].split("Premises:")[1].strip()
            conclusion_string = self.logic_program.split("Conclusion:")[1].strip()

            # Extract each premise
            premises = premises_string.strip().split('\n')
            self.logic_premises = [premise.split(':::')[0].strip() for premise in premises]

            # Extract multiple conclusions for LogicalDeduction
            conclusion_lines = conclusion_string.strip().split('\n')
            self.multiple_conclusions = {}
            
            # Look for flexible option patterns: option/Option, options/Options, with optional period
            # Supports: "Option A", "option A", "Options A", "options A", "Option A.", "option A.", etc.
            option_pattern = re.compile(r'.*:::\s*options?\s+([A-Z])\.?\s*$', re.IGNORECASE)
            
            for line in conclusion_lines:
                if ':::' in line:
                    match = option_pattern.match(line)
                    if match:
                        option_letter = match.group(1).upper()  # Ensure uppercase for consistency
                        conclusion_formula = line.split(':::')[0].strip()
                        self.multiple_conclusions[option_letter] = conclusion_formula

            # convert premises to prover9 format
            self.prover9_premises = []
            for premise in self.logic_premises:
                fol_rule = FOL_Formula(premise)
                if fol_rule.is_valid == False:
                    return False
                prover9_rule = Prover9_FOL_Formula(fol_rule)
                self.prover9_premises.append(prover9_rule.formula)

            # Convert multiple conclusions to prover9 format
            self.prover9_multiple_conclusions = {}
            for option_letter, conclusion_formula in self.multiple_conclusions.items():
                fol_conclusion = FOL_Formula(conclusion_formula)
                if fol_conclusion.is_valid == False:
                    return False
                prover9_rule = Prover9_FOL_Formula(fol_conclusion)
                self.prover9_multiple_conclusions[option_letter] = prover9_rule.formula

            return True
        except Exception:
            return False

    def execute_program(self):
        # Check if logic program parsing was successful
        if not self.flag:
            return None, "Logic program parsing failed", ''
        
        # Handle LogicalDeduction dataset separately
        if self.dataset_name == 'LogicalDeduction':
            return self._execute_logical_deduction_program()
            
        try:
            goal = Expression.fromstring(self.prover9_conclusion)
            assumptions = [Expression.fromstring(a) for a in self.prover9_premises]
            timeout = 10
            #prover = Prover9()
            #result = prover.prove(goal, assumptions)
            
            prover = Prover9Command(goal, assumptions, timeout=timeout)
            result = prover.prove()
            # print(prover.proof())

            proof_trace = ''

            if result:
                # 证明成功：记录原结论的推导路径
                proof_core = self._extract_proof_steps_ture_false(prover.proof(simplify=True))
                proof_trace = 'prove original conclusion:\n' + proof_core
                return 'True', '', proof_trace
            else:
                # 证明失败，尝试证明结论的否定
                proof_trace += 'prove original conclusion:\n' + prover.proof(simplify=False) + '\n'

                negated_goal = NegatedExpression(goal)
                prover_neg = Prover9Command(negated_goal, assumptions, timeout=timeout)
                negation_result = prover_neg.prove()

                if negation_result:
                    # 证明否定成功 => 原结论为 False，只输出成功证明路径
                    proof_core = self._extract_proof_steps_ture_false(prover_neg.proof(simplify=True))
                    proof_trace = 'prove negation of original conclusion:\n' + proof_core
                    return 'False', '', proof_trace
                else:
                    # 两次证明都失败，结论未知 → 调命令行版抓完整日志
                    orig_in  = _build_p9_input(self.prover9_premises, self.prover9_conclusion)
                    orig_log = _run_prover9_raw(orig_in, timeout=timeout+2)
                    orig_tr  = _summarise_log(orig_log)

                    neg_goal = f"-({self.prover9_conclusion})"
                    neg_in   = _build_p9_input(self.prover9_premises, neg_goal)
                    neg_log  = _run_prover9_raw(neg_in, timeout=timeout+2)
                    neg_tr   = _summarise_log(neg_log)

                    proof_trace = (f"trying to prove original conclusion:\n{orig_tr}\n\n"
                                   f"trying to prove negation of original conclusion:\n{neg_tr}\n\n"
                                   f"So: Unknown")
                    return 'Unknown', '', proof_trace
        except Exception as e:
            return None, str(e), ''
    
    def _execute_logical_deduction_program(self):
        """Execute program for LogicalDeduction multi-choice format"""
        try:
            assumptions = [Expression.fromstring(a) for a in self.prover9_premises]
            timeout = 10
            
            # Try to prove each option
            proven_options = []
            available_options = list(self.prover9_multiple_conclusions.keys())
            
            for option_letter in available_options:
                conclusion_formula = self.prover9_multiple_conclusions[option_letter]
                result, reasoning = self._prove_single_conclusion(conclusion_formula, assumptions, timeout)
                if result == 'True':
                    proven_options.append((option_letter, reasoning))
            
            # If any option is proven true, return the first one
            if proven_options:
                chosen_option, reasoning = proven_options[0]
                return chosen_option, '', reasoning
            
            # If no option is proven true, randomly choose one
            chosen_option = random.choice(available_options)
            return chosen_option, '', ''
            
        except Exception as e:
            return None, str(e), ''
    
    def _prove_single_conclusion(self, conclusion_formula, assumptions, timeout):
        """
        Prove a single conclusion and return result and reasoning.
        Returns ('True', reasoning) if proven, ('False', '') otherwise.
        """
        try:
            goal = Expression.fromstring(conclusion_formula)
            prover = Prover9Command(goal, assumptions, timeout=timeout)
            result = prover.prove()
            
            if result:
                proof_core = self._extract_proof_steps_ture_false(prover.proof(simplify=True))
                reasoning = f'prove option conclusion:\n{proof_core}'
                return 'True', reasoning
            else:
                return 'False', ''
                
        except Exception:
            return 'False', ''
        
    def answer_mapping(self, answer):
        """
        Map the prover9 output to the appropriate dataset answer format.
        
        Args:
            answer: The prover9 output ('True', 'False', 'Unknown')
            
        Returns:
            str: The mapped answer for the specific dataset
        """
        if self.dataset_name == 'ProntoQA':
            # ProntoQA only has A/B options, no Unknown
            if answer == 'True':
                return 'A'
            elif answer == 'False':
                return 'B'
            elif answer == 'Unknown':
                # For ProntoQA, Unknown is randomly mapped to A or B
                return random.choice(['A', 'B'])
        elif self.dataset_name == 'ProofWriter':
            # ProofWriter supports A/B/C (True/False/Unknown)
            if answer == 'True':
                return 'A'
            elif answer == 'False':
                return 'B'
            elif answer == 'Unknown':
                return 'C'
        elif self.dataset_name == 'FOLIO':
            # FOLIO supports A/B/C (True/False/Unknown) - keep original logic
            if answer == 'True':
                return 'A'
            elif answer == 'False':
                return 'B'
            elif answer == 'Unknown':
                return 'C'
        elif self.dataset_name == 'LogicalDeduction':
            # LogicalDeduction returns the option letter directly (A, B, C, D, E)
            return answer
        else:
            raise ValueError(f'Unsupported dataset: {self.dataset_name}')
        
        # Fallback for unrecognized answers
        raise ValueError(f'Answer "{answer}" not recognized for dataset "{self.dataset_name}"')
        
    @staticmethod
    def _extract_proof_steps_ture_false(proof_str: str) -> str:
        """Extract only the numbered step lines from a Prover9 proof output.

        Prover9 proof outputs often contain headers, footers, and comments in
        addition to the essential step lines that begin with an integer index.
        This helper keeps only lines that start with digits (optionally
        preceded by whitespace), which correspond to the step annotations we
        are interested in displaying.
        """
        step_lines = []
        for line in proof_str.splitlines():
            if re.match(r"^\s*\d+", line):
                step_lines.append(line)
        return "\n".join(step_lines)

if __name__ == "__main__":
    ## ¬∀x (Movie(x) → HappyEnding(x))
    ## ∃x (Movie(x) → ¬HappyEnding(x))
    # ground-truth: True
    logic_program_t = """Premises:
    ¬∀x (Movie(x) → HappyEnding(x)) ::: Not all movie has a happy ending.
    Movie(titanic) ::: Titanic is a movie.
    ¬HappyEnding(titanic) ::: Titanic does not have a happy ending.
    Movie(lionKing) ::: Lion King is a movie.
    HappyEnding(lionKing) ::: Lion King has a happy ending.
    Conclusion:
    ∃x (Movie(x) ∧ ¬HappyEnding(x)) ::: Some movie does not have a happy ending.
    """

    # ground-truth: True
    logic_program = """Premises:
    ∀x (Drinks(x) → Dependent(x)) ::: All people who regularly drink coffee are dependent on caffeine.
    ∀x (Drinks(x) ⊕ Jokes(x)) ::: People either regularly drink coffee or joke about being addicted to caffeine.
    ∀x (Jokes(x) → ¬Unaware(x)) ::: No one who jokes about being addicted to caffeine is unaware that caffeine is a drug. 
    (Student(rina) ∧ Unaware(rina)) ⊕ ¬(Student(rina) ∨ Unaware(rina)) ::: Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug. 
    ¬(Dependent(rina) ∧ Student(rina)) → (Dependent(rina) ∧ Student(rina)) ⊕ ¬(Dependent(rina) ∨ Student(rina)) ::: If Rina is not a person dependent on caffeine and a student, then Rina is either a person dependent on caffeine and a student, or neither a person dependent on caffeine nor a student.
    Conclusion:
    Jokes(rina) ⊕ Unaware(rina) ::: Rina is either a person who jokes about being addicted to caffeine or is unaware that caffeine is a drug.
    """

    # ground-truth: True
    logic_program = """Premises:
    ∀x (Drinks(x) → Dependent(x)) ::: All people who regularly drink coffee are dependent on caffeine.
    ∀x (Drinks(x) ⊕ Jokes(x)) ::: People either regularly drink coffee or joke about being addicted to caffeine.
    ∀x (Jokes(x) → ¬Unaware(x)) ::: No one who jokes about being addicted to caffeine is unaware that caffeine is a drug. 
    (Student(rina) ∧ Unaware(rina)) ⊕ ¬(Student(rina) ∨ Unaware(rina)) ::: Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug. 
    ¬(Dependent(rina) ∧ Student(rina)) → (Dependent(rina) ∧ Student(rina)) ⊕ ¬(Dependent(rina) ∨ Student(rina)) ::: If Rina is not a person dependent on caffeine and a student, then Rina is either a person dependent on caffeine and a student, or neither a person dependent on caffeine nor a student.
    Conclusion:
    ((Jokes(rina) ∧ Unaware(rina)) ⊕ ¬(Jokes(rina) ∨ Unaware(rina))) → (Jokes(rina) ∧ Drinks(rina)) ::: If Rina is either a person who jokes about being addicted to caffeine and a person who is unaware that caffeine is a drug, or neither a person who jokes about being addicted to caffeine nor a person who is unaware that caffeine is a drug, then Rina jokes about being addicted to caffeine and regularly drinks coffee.
    """

    # ground-truth: Unknown
    logic_program = """Premises:
    Czech(miroslav) ∧ ChoralConductor(miroslav) ∧ Specialize(miroslav, renaissance) ∧ Specialize(miroslav, baroque) ::: Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
    ∀x (ChoralConductor(x) → Musician(x)) ::: Any choral conductor is a musician.
    ∃x (Musician(x) ∧ Love(x, music)) ::: Some musicians love music.
    Book(methodOfStudyingGregorianChant) ∧ Author(miroslav, methodOfStudyingGregorianChant) ∧ Publish(methodOfStudyingGregorianChant, year1946) ::: Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.
    Conclusion:
    Love(miroslav, music) ::: Miroslav Venhoda loved music.
    """

    # ground-truth: True
    logic_program = """Premises:
    Czech(miroslav) ∧ ChoralConductor(miroslav) ∧ Specialize(miroslav, renaissance) ∧ Specialize(miroslav, baroque) ::: Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
    ∀x (ChoralConductor(x) → Musician(x)) ::: Any choral conductor is a musician.
    ∃x (Musician(x) ∧ Love(x, music)) ::: Some musicians love music.
    Book(methodOfStudyingGregorianChant) ∧ Author(miroslav, methodOfStudyingGregorianChant) ∧ Publish(methodOfStudyingGregorianChant, year1946) ::: Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.
    Conclusion:
    ∃y ∃x (Czech(x) ∧ Author(x, y) ∧ Book(y) ∧ Publish(y, year1946)) ::: A Czech person wrote a book in 1946.
    """

    # ground-truth: False
    logic_program_f = """Premises:
    Czech(miroslav) ∧ ChoralConductor(miroslav) ∧ Specialize(miroslav, renaissance) ∧ Specialize(miroslav, baroque) ::: Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
    ∀x (ChoralConductor(x) → Musician(x)) ::: Any choral conductor is a musician.
    ∃x (Musician(x) ∧ Love(x, music)) ::: Some musicians love music.
    Book(methodOfStudyingGregorianChant) ∧ Author(miroslav, methodOfStudyingGregorianChant) ∧ Publish(methodOfStudyingGregorianChant, year1946) ::: Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.
    Conclusion:
    ¬∃x (ChoralConductor(x) ∧ Specialize(x, renaissance)) ::: No choral conductor specialized in the performance of Renaissance.
    """

    # ground-truth: Unknown
    # Premises:\nall x.(perform_in_school_talent_shows_often(x) -> (attend_school_events(x) & very_engaged_with_school_events(x))) ::: If people perform in school talent shows often, then they attend and are very engaged with school events.\nall x.(perform_in_school_talent_shows_often(x) ^ (inactive_member(x) & disinterested_member(x))) ::: People either perform in school talent shows often or are inactive and disinterested members of their community.\nall x.(chaperone_high_school_dances(x) -> not student_attend_school(x)) ::: If people chaperone high school dances, then they are not students who attend the school.\nall x.((inactive_member(x) & disinterested_member(x)) -> chaperone_high_school_dances(x)) ::: All people who are inactive and disinterested members of their community chaperone high school dances.\nall x.((young_child(x) | teenager(x)) & wish_to_further_academic_careers(x) & wish_to_further_educational_opportunities(x) -> student_attend_school(x)) ::: All young children and teenagers who wish to further their academic careers and educational opportunities are students who attend the school.\n(attend_school_events(bonnie) & very_engaged_with_school_events(bonnie) & student_attend_school(bonnie)) ^ (not attend_school_events(bonnie) & not very_engaged_with_school_events(bonnie) & not student_attend_school(bonnie)) ::: Bonnie either both attends and is very engaged with school events and is a student who attends the school, or she neither attends and is very engaged with school events nor is a student who attends the school.\nConclusion:\nperform_in_school_talent_shows_often(bonnie) ::: Bonnie performs in school talent shows often."
    logic_program = """Premises:
    ∀x (TalentShows(x) → Engaged(x)) ::: If people perform in school talent shows often, then they attend and are very engaged with school events.
    ∀x (TalentShows(x) ∨ Inactive(x)) ::: People either perform in school talent shows often or are inactive and disinterested members of their community.
    ∀x (Chaperone(x) → ¬Students(x)) ::: If people chaperone high school dances, then they are not students who attend the school.
    ∀x (Inactive(x) → Chaperone(x)) ::: All people who are inactive and disinterested members of their community chaperone high school dances.
    ∀x (AcademicCareer(x) → Students(x)) ::: All young children and teenagers who wish to further their academic careers and educational opportunities are students who attend the school.
    Conclusion:
    TalentShows(bonnie) ::: Bonnie performs in school talent shows often.
    """

    # ground-truth: False
    logic_program_u = """Premises:
    MusicPiece(symphonyNo9) ::: Symphony No. 9 is a music piece.
    ∀x ∃z (¬Composer(x) ∨ (Write(x,z) ∧ MusicPiece(z))) ::: Composers write music pieces.
    Write(beethoven, symphonyNo9) ::: Beethoven wrote Symphony No. 9.
    Lead(beethoven, viennaMusicSociety) ∧ Orchestra(viennaMusicSociety) ::: Vienna Music Society is an orchestra and Beethoven leads the Vienna Music Society.
    ∀x ∃z (¬Orchestra(x) ∨ (Lead(z,x) ∧ Conductor(z))) ::: Orchestras are led by conductors.
    Conclusion:
    ¬Conductor(beethoven) ::: Beethoven is not a conductor."""

    # ground-truth: True
    logic_program = """Predicates:
    JapaneseCompany(x) ::: x is a Japanese game company.
    Create(x, y) ::: x created the game y.
    Top10(x) ::: x is in the Top 10 list.
    Sell(x, y) ::: x sold more than y copies.
    Premises:
    ∃x (JapaneseCompany(x) ∧ Create(x, legendOfZelda)) ::: A Japanese game company created the game the Legend of Zelda.
    ∀x ∃z (¬Top10(x) ∨ (JapaneseCompany(z) ∧ Create(z,x))) ::: All games in the Top 10 list are made by Japanese game companies.
    ∀x (Sell(x, oneMillion) → Top10(x)) ::: If a game sells more than one million copies, then it will be selected into the Top 10 list.
    Sell(legendOfZelda, oneMillion) ::: The Legend of Zelda sold more than one million copies.
    Conclusion:
    Top10(legendOfZelda) ::: The Legend of Zelda is in the Top 10 list."""

    logic_program = """Premises:
    ∀x (Listed(x) → ¬NegativeReviews(x)) ::: If the restaurant is listed in Yelp's recommendations, then the restaurant does not receive many negative reviews.
    ∀x (GreaterThanNine(x) → Listed(x)) ::: All restaurants with a rating greater than 9 are listed in Yelp's recommendations.
    ∃x (¬TakeOut(x) ∧ NegativeReviews(x)) ::: Some restaurants that do not provide take-out service receive many negative reviews.
    ∀x (Popular(x) → GreaterThanNine(x)) ::: All restaurants that are popular among local residents have ratings greater than 9.
    GreaterThanNine(subway) ∨ Popular(subway) ::: Subway has a rating greater than 9 or is popular among local residents.
    Conclusion:
    TakeOut(subway) ∧ ¬NegativeReviews(subway) ::: Subway provides take-out service and does not receive many negative reviews."""
    
    logic_program_byfx = "Predicates:\nBlue(x) ::: x is blue\nRound(x) ::: x is round\nLikes(x, y) ::: x likes y\nVisits(x, y) ::: x visits y\nCold(x) ::: x is cold\nNice(x) ::: x is nice\nSees(x, y) ::: x sees y\nYoung(x) ::: x is young\nPremises:\nBlue(cow) ::: The cow is blue\nRound(cow) ::: The cow is round\nLikes(cow, lion) ::: The cow likes the lion\nVisits(cow, tiger) ::: The cow visits the tiger\nCold(lion) ::: The lion is cold\nNice(lion) ::: The lion is nice\nLikes(lion, squirrel) ::: The lion likes the squirrel\nRound(squirrel) ::: The squirrel is round\nSees(squirrel, lion) ::: The squirrel sees the lion\nVisits(squirrel, cow) ::: The squirrel visits the cow\nLikes(tiger, cow) ::: The tiger likes the cow\nLikes(tiger, squirrel) ::: The tiger likes the squirrel\n\u2200x (Cold(x) \u2192 Visits(x, tiger)) ::: If something is cold then it visits the tiger\n\u2200x (Visits(x, tiger) \u2192 Nice(x)) ::: If something visits the tiger then it is nice\n\u2200x (Nice(x) \u2192 Sees(x, tiger)) ::: If something is nice then it sees the tiger\n\u2200x (Nice(x) \u2227 Sees(x, tiger) \u2192 Young(x)) ::: If something is nice and it sees the tiger then it is young\n\u2200x (Sees(x, tiger) \u2227 Young(x) \u2192 Blue(x)) ::: If something sees the tiger and it is young then it is blue\n\u2200x (Likes(x, squirrel) \u2227 Likes(x, cow) \u2192 Visits(x, tiger)) ::: If something likes the squirrel and it likes the cow then it visits the tiger\nCold(cow) \u2227 Visits(cow, lion) \u2192 Sees(lion, squirrel) ::: If the cow is cold and the cow visits the lion then the lion sees the squirrel\nConclusion:\n\u00acYoung(tiger) ::: The tiger is not young"
    
    
    logic_program_ALSAT = """Predicates:
    Tour(d, division) ::: On day d (mon, tue, wed, thu, fri), the company tours division division (ops, prod, sales).
    Next(d1, d2) ::: Day d2 immediately follows day d1 (mon→tue→wed→thu→fri).
    Premises:
    Next(mon, tue) ::: Tuesday follows Monday.
    Next(tue, wed) ::: Wednesday follows Tuesday.
    Next(wed, thu) ::: Thursday follows Wednesday.
    Next(thu, fri) ::: Friday follows Thursday.
    ∀d (Tour(d, ops) ∨ Tour(d, prod) ∨ Tour(d, sales)) ::: Exactly one division is toured each day.
    ∀d ∀x ∀y ((Tour(d, x) ∧ Tour(d, y)) → x = y) ::: No day has more than one division (uniqueness).
    ∀div (Tour(mon, div) ∨ Tour(tue, div) ∨ Tour(wed, div) ∨ Tour(thu, div) ∨ Tour(fri, div)) ::: Each division is toured at least once during the week.
    ¬Tour(mon, ops) ::: Operations is not toured on Monday.
    ¬Tour(wed, prod) ::: Production is not toured on Wednesday.
    ∃d1 ∃d2 (Next(d1, d2) ∧ Tour(d1, sales) ∧ Tour(d2, sales) ∧ ∀d (Tour(d, sales) → (d = d1 ∨ d = d2))) ::: Sales is toured on exactly two consecutive days and on no other days.
    (Tour(thu, ops) → Tour(fri, prod)) ::: If Operations is toured on Thursday, then Production is toured on Friday.
    Conclusion:
    ∃div (Tour(tue, div) ∧ Tour(thu, div)) ::: The division toured on Tuesday is also toured on Thursday.   (Option C)
"""
    





    sample_logic_deduction = """"id": "logical_deduction_14",
        "context": "The following paragraphs each describe a set of five objects arranged in a fixed order. The statements are logically consistent within each paragraph.\n\nA fruit stand sells five fruits: mangoes, kiwis, plums, pears, and watermelons. The kiwis are less expensive than the plums. The pears are the third-most expensive. The kiwis are the second-cheapest. The watermelons are the most expensive.",
        "question": "Which of the following is true?",
        "options": [
            "A) The mangoes are the third-most expensive.",
            "B) The kiwis are the third-most expensive.",
            "C) The plums are the third-most expensive.",
            "D) The pears are the third-most expensive.",
            "E) The watermelons are the third-most expensive."
        ],
        "answer": "D" 
    """
    logic_program_logic_deduction = """Predicates:
Rank(fruit, pos) ::: fruit has price position pos, where pos ∈ {one,two,three,four,five}; one = most expensive, five = cheapest.
Cheaper(x, y) ::: x is cheaper (less expensive) than y.
Premises:
Rank(watermelon, one) :::Watermelons are the most expensive
Rank(pears, three) ::: Pears are the third‑most expensive
Rank(kiwis, four) ::: Kiwis are the second‑cheapest
Cheaper(kiwis, plums) ::: Kiwis are cheaper than plums
∀F ∀P ∀Q ((Rank(F,P) ∧ Rank(F,Q)) → (P = Q)) ::: One rank per fruit
∀P ∀F ∀G ((Rank(F,P) ∧ Rank(G,P)) → (F = G)) ::: One fruit per rank
Rank(mangoes, one) ∨ Rank(mangoes, two) ∨ Rank(mangoes, three) ∨ Rank(mangoes, four) ∨ Rank(mangoes, five) ::: each still‑unknown fruit occupies some rank 
Rank(plums, one) ∨ Rank(plums, two) ∨ Rank(plums, three) ∨ Rank(plums, four) ∨ Rank(plums, five) ::: each still‑unknown fruit occupies some rank 
∀X ∀Y (Rank(X, one) ∧ Rank(Y, two) → Cheaper(Y, X)) ::: “higher rank → more expensive” (10 ordered pairs) 
∀X ∀Y (Rank(X, one) ∧ Rank(Y, three) → Cheaper(Y, X)) ::: “higher rank → more expensive” (10 ordered pairs) 
∀X ∀Y (Rank(X, one) ∧ Rank(Y, four) → Cheaper(Y, X)) ::: “higher rank → more expensive” (10 ordered pairs) 
∀X ∀Y (Rank(X, one) ∧ Rank(Y, five) → Cheaper(Y, X)) ::: “higher rank → more expensive” (10 ordered pairs) 
∀X ∀Y (Rank(X, two) ∧ Rank(Y, three) → Cheaper(Y, X)) ::: “higher rank → more expensive” (10 ordered pairs) 
∀X ∀Y (Rank(X, two) ∧ Rank(Y, four) → Cheaper(Y, X)) ::: “higher rank → more expensive” (10 ordered pairs) 
∀X ∀Y (Rank(X, two) ∧ Rank(Y, five) → Cheaper(Y, X)) ::: “higher rank → more expensive” (10 ordered pairs) 
∀X ∀Y (Rank(X, three) ∧ Rank(Y, four) → Cheaper(Y, X)) ::: “higher rank → more expensive” (10 ordered pairs) 
∀X ∀Y (Rank(X, three) ∧ Rank(Y, five) → Cheaper(Y, X)) ::: “higher rank → more expensive” (10 ordered pairs) 
∀X ∀Y (Rank(X, four) ∧ Rank(Y, five) → Cheaper(Y, X)) ::: “higher rank → more expensive” (10 ordered pairs) 
∀X ∀Y (Cheaper(X, Y) → ¬Cheaper(Y, X)) ::: “cheaper” is asymmetric
Conclusion:
Rank(mangoes, three) ::: Option A
Rank(kiwis, three) ::: Option B
Rank(plums, three) ::: Option C
Rank(pears, three) ::: Option D
Rank(watermelon, three) ::: Option E
"""
    
    
    demo_logic_program = "Predicates:\nBird($x) ::: $x is one of the five birds.\nLeftOf($x, $y) ::: Bird $x is strictly to the left of bird $y.\nPosition($x, $n) ::: Bird $x is at position $n from the left (1-based index).\nPremises:\nBird(crow) ::: The crow.\nBird(robin) ::: The robin.\nBird(quail) ::: The quail.\nBird(blue_jay) ::: The blue jay.\nBird(falcon) ::: The falcon.\nLeftOf(robin, quail) ::: The robin is to the left of the quail.\nPosition(falcon, 3) ::: The falcon is the third from the left.\nLeftOf(crow, falcon) ::: The crow is to the left of the falcon.\nPosition(blue_jay, 1) ::: The blue jay is the leftmost.\n\u2200x \u2200y (LeftOf(x, y) \u2192 \u00acLeftOf(y, x)) ::: Left-of is asymmetric.\n\u2200x \u2200y \u2200z (LeftOf(x, y) \u2227 LeftOf(y, z) \u2192 LeftOf(x, z)) ::: Left-of is transitive.\n\u2200x \u2200n \u2200m (Position(x, n) \u2227 Position(x, m) \u2192 n = m) ::: One position per bird.\n\u2200n \u2200x \u2200y (Position(x, n) \u2227 Position(y, n) \u2192 x = y) ::: One bird per position.\n\u2200x \u2200y (LeftOf(x, y) \u2194 \u2203n \u2203m (Position(x, n) \u2227 Position(y, m) \u2227 n < m)) ::: Left-of corresponds to position ordering.\nConclusion:\nPosition(crow, 3) ::: Option A\nPosition(robin, 3) ::: Option B\nPosition(quail, 3) ::: Option C\nPosition(blue_jay, 3) ::: Option D\nPosition(falcon, 3) ::: Option E"
    
    # Test LogicalDeduction functionality
    prover9_program = FOL_Prover9_Program(logic_program_logic_deduction, dataset_name='LogicalDeduction')
    result, error_message, reasoning = prover9_program.execute_program()
    print('LogicalDeduction Test Results:')
    print('result:', result)
    print('error_message:', error_message)
    if reasoning:
        print('reasoning:', reasoning)
    
    # Test answer mapping for LogicalDeduction
    if result:
        mapped_answer = prover9_program.answer_mapping(result)
        print('mapped_answer:', mapped_answer)
    