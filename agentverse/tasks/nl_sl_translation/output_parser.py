from __future__ import annotations

import re
from typing import Union, Optional

from agentverse.parser import OutputParser, LLMResult
from pydantic import Field

from agentverse.utils import AgentAction, AgentFinish

from agentverse.parser import OutputParserError, output_parser_registry

# Simplified solver validation logic (for demonstration purposes)
def validate_logic_program(agent_name: str, logic_program: str) -> tuple[bool, str]:
    """
    Simplified logic program validation
    Returns (is_valid, error_message)
    """
    if not logic_program.strip():
        return False, "Empty logic program"
    
    # Simple validation based on agent type
    if agent_name == 'LP translator':
        # Check for basic LP structure
        if not any(keyword in logic_program for keyword in ['Facts:', 'Rules:', 'Query:']):
            return False, "Missing required LP sections (Facts, Rules, Query)"
        if 'Query:' not in logic_program:
            return False, "Missing Query section"
    elif agent_name == 'FOL translator':
        # Check for basic FOL structure
        if not any(keyword in logic_program for keyword in ['Predicates:', 'Premises:', 'Conclusion:']):
            return False, "Missing required FOL sections (Predicates, Premises, Conclusion)"
        if 'Conclusion:' not in logic_program:
            return False, "Missing Conclusion section"
    elif agent_name == 'SAT translator':
        # Check for basic SAT structure
        if not any(keyword in logic_program for keyword in ['# Declarations', '# Constraints', '# Options']):
            return False, "Missing required SAT sections (# Declarations, # Constraints, # Options)"
        if '# Declarations' not in logic_program:
            return False, "Missing Declarations section"
    
    return True, ""

# Use simplified validation instead of full solver import
# Try to import actual solvers first
from solver_engine.src.symbolic_solvers.pyke_solver.pyke_solver import Pyke_Program
from solver_engine.src.symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
from solver_engine.src.symbolic_solvers.z3_solver.sat_problem_solver import LSAT_Z3_Program
SOLVER_AVAILABLE = True
USE_ACTUAL_SOLVERS = True
# Agent name to solver mapping (based on logic_inference.py PROGRAM_CLASS)
AGENT_SOLVER_MAP = {
    'LP translator': ('LP', Pyke_Program),
    'FOL translator': ('FOL', FOL_Prover9_Program),
    'SAT translator': ('SAT', LSAT_Z3_Program),
}


@output_parser_registry.register("translate")
class TranslateParser(OutputParser):
    dataset_name: str = Field(default="ProofWriter")
    
    def parse(self, output: LLMResult, cnt_turn: int, max_turns: int, agent_nums: int, agent_name: Optional[str] = None) -> Union[AgentAction, AgentFinish]:
        text = output.content
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        cleaned_output = re.sub(r"\n-", "\n", cleaned_output)  # Remove dash after newline
        cleaned_output = cleaned_output.replace("**", "")

        # Check if it's the last turn and validate with solver
        if cnt_turn >= max_turns - agent_nums and agent_name and SOLVER_AVAILABLE:
            if agent_name in AGENT_SOLVER_MAP:
                solver_key, solver_class = AGENT_SOLVER_MAP[agent_name]
                
                if USE_ACTUAL_SOLVERS and solver_class is not None:
                    try:
                        # Create solver instance and validate (mimicking safe_execute_program)
                        program = solver_class(cleaned_output, self.dataset_name)
                        
                        # Check parsing flag (similar to safe_execute_program)
                        if not getattr(program, 'flag', True):
                            # Parsing error - raise to trigger retry
                            raise OutputParserError(f"Solver parsing error for {agent_name}: Failed to parse logic program")
                        
                        # Try to execute the program
                        answer, err, reasoning = program.execute_program()
                        if answer is None:
                            # Execution error - raise to trigger retry
                            raise OutputParserError(f"Solver execution error for {agent_name}: {err}")
                            
                    except OutputParserError:
                        # Re-raise OutputParserError directly
                        raise
                    except Exception as e:
                        # Any other error during solver validation
                        raise OutputParserError(f"Solver validation failed for {agent_name}: {str(e)}")
                else:
                    # Use simplified validation when actual solvers are not available
                    is_valid, error_message = validate_logic_program(agent_name, cleaned_output)
                    if not is_valid:
                        raise OutputParserError(f"Logic program validation failed for {agent_name}: {error_message}")
                    


        return AgentFinish({"output": cleaned_output}, cleaned_output)
