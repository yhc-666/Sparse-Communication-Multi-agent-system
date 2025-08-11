# Multi-Agent Debate System

A multi-agent debate system built on the AgentVerse framework, designed to facilitate structured debates between AI agents with different roles and perspectives.

## Overview

This system enables multiple AI agents to engage in structured debates on various topics. The agents are assigned different roles and personas, allowing them to present diverse viewpoints and engage in meaningful discussions. The system is particularly useful for exploring complex questions and analyzing problems from multiple angles.


## Quick Start

### Prerequisites

- Python 3.8+
- Required packages listed in `requirements.txt`

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
境内需通过 `python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt` 安装

3. Set up your API credentials inside `run_translate.py` and  `run_final_debate.py` and `run_llmsolver.py` (if using API mode):
```bash
OPENAI_API_KEY="your_api_key_here"
OPENAI_BASE_URL="your_api_base_url"
```
4. Grant Permission for Prover9 SOlver: 
```bash
chmod +x solver_engine/src/symbolic_solvers/fol_solver/../Prover9/bin/*
```

### Basic Usage

Note: config each session's .yaml accordingly

1. Run a debate session for NL->SL translation:

    Configure in `agentverse/tasks/nl_sl_translation/translate_config.yaml`

```bash
python run_translate.py
```

2. Run LLM as Solver session to directly generate (answer, reasoning_path) and BACKUP ANSWER from NL

    Configure in `agentverse/tasks/llm_as_solver/llm_solver_config.yaml` 

```bash
python run_llmsolver.py
```


3. Run solver engine to generate (answer, reasoning_path) from SL

    Configure in `solver_engine/symbolic_solver_config.yaml` 

```bash
python run_symbolicsolver.py 
```


4. Run a debate session for final result

    Configure in `agentverse/tasks/final_debate/final_debate_config.yaml`

```bash
python run_final_debate.py
```

5. Evaluation


```bash
python run_evaluation.py
```

## ⚙️ Configuration

The system is configured through YAML files.

### Key Configuration Sections

