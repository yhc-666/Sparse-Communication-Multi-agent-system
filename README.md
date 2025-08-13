# Sparse Multi-Agent Debate System

A pipeline for logical reasoning through multi-agent debate with sparse communication.

## Prerequisites

- Python 3.8+
- Install dependencies:
```bash
pip install -r requirements.txt
```

## Pipeline Stages

### 1. Translation (Natural Language → Symbolic Logic)

Configure API credentials in `agentverse/tasks/nl_sl_translation/translate_config.yaml`:
```python
OPENAI_API_KEY="your_api_key_here"
OPENAI_BASE_URL="your_api_base_url"
```

Run translation:
```bash
python run_translate.py
```

### 2. Multi-Agent Debate

Configure API credentials in `agentverse/tasks/final_debate/sparse_debate_config.yaml`:
```python
OPENAI_API_KEY="your_api_key_here"
OPENAI_BASE_URL="your_api_base_url"
```

#### 2a. Debate WITHOUT Sparse Communication (Full Visibility)

Edit `agentverse/tasks/final_debate/sparse_debate_config.yaml`:
```yaml
environment:
  rule:
    visibility:
      type: all        # Change from "sparse" to "all"
    updater:
      type: basic      # Change from "sparse" to "basic"
```

Run debate:
```bash
python run_sparse_debate.py
```

#### 2b. Debate WITH Sparse Communication (Selective Visibility)

Edit `agentverse/tasks/final_debate/sparse_debate_config.yaml`:
```yaml
environment:
  rule:
    visibility:
      type: sparse     # Use "sparse" for selective visibility
      bert_model: "prajjwal1/bert-tiny"
      lambda_param: 0.5
    updater:
      type: sparse     # Use "sparse" updater
```

Run debate:
```bash
python run_sparse_debate.py
```

### 3. Final Evaluation

Evaluate the debate results:
```bash
python run_evaluation.py --input_path outputs/sparse_debate/ProofWriter/sparse_debate_results.json --aggregation_strategy majority_vote
```

## Output

Results are saved in:
- Translation: `outputs/translation/ProofWriter/`
- Debate: `outputs/sparse_debate/ProofWriter/`
- Evaluation: Console output with accuracy metrics