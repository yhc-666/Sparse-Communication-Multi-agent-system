# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Installation and Setup
```bash
pip install -r requirements.txt
# For users in China: python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
chmod +x solver_engine/src/symbolic_solvers/fol_solver/../Prover9/bin/*
```

### Running the Multi-Agent Debate System
The system operates in a pipeline with these main scripts:

1. **NL to SL Translation**: `python run_translate.py`
2. **Symbolic Solver**: `python run_symbolicsolver.py` 
3. **LLM as Solver**: `python run_llmsolver.py`
4. **Final Debate**: `python run_final_debate.py`
5. **Evaluation**: `python run_evaluation.py`

Each script requires configuring the corresponding .yaml file before running.

### API Configuration
Set API credentials in the run scripts:
```python
OPENAI_API_KEY="your_api_key_here"
OPENAI_BASE_URL="your_api_base_url"
```

## Architecture Overview

### Core Components

**AgentVerse Framework** (`agentverse/`):
- `agents/`: Multi-agent implementations including debate agents, LLM evaluators, and solver agents
- `environments/`: Environment orchestration with rule-based agent interaction systems
- `llms/`: LLM integrations (OpenAI API, local models)
- `tasks/`: Task-specific configurations and output parsers for different debate scenarios

**Solver Engine** (`solver_engine/`):
- `symbolic_solvers/`: Multiple reasoning approaches including FOL (with Prover9), Logic Programming (Pyke), and SAT (Z3)
- `logic_inference.py`: Main inference orchestration
- Integration with external theorem provers and constraint solvers

### Multi-Agent Debate System
The system implements structured debates between AI agents with different reasoning philosophies:
- **LP supporter**: Logic Programming approach using predicates and rules
- **FOL supporter**: First-Order Logic with quantifiers and formal reasoning
- **LLM supporters**: Various LLM-based reasoning approaches (forward reasoning, contradiction-based, random)

### Data Flow
1. Natural language problems are translated to symbolic logic
2. Multiple symbolic solvers generate candidate solutions
3. LLM solvers provide alternative reasoning paths
4. Agents debate the merits of different approaches
5. Final consensus is reached through structured debate

### Configuration System
All components are configured via YAML files in `agentverse/tasks/` and `solver_engine/`:
- `final_debate_config.yaml`: Multi-agent debate setup and agent roles
- `translate_config.yaml`: NL to SL translation parameters
- `llm_solver_config.yaml`: LLM-based solver configuration
- `symbolic_solver_config.yaml`: Symbolic reasoning engine settings

### Output Structure
Results are organized in `outputs/` by model and task type, containing:
- JSON result files with answers and reasoning
- LLM interaction logs
- Argument files with execution parameters

## Development Notes

### Cursor Rules Integration
The project includes specific development rules in `.cursor/rules/my-rules.mdc`:
- Operates in Plan/Act modes for structured development
- Emphasizes minimal, contained changes
- Focuses on isolated logic that doesn't break existing flows

### Key Dependencies
- `scitools-pyke`: Logic Programming solver
- `z3-solver`: SAT solver
  
- `openai`: LLM API integration
- `langchain`: LLM framework components