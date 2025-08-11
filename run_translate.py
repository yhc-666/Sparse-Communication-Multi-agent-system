import os

# 移除硬编码的环境变量设置，现在通过YAML配置文件处理
# os.environ["OPENAI_API_KEY"] = "sk-733e47bc35da4b49b0bc7ca99ede48f8"
# os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"

# os.environ["OPENAI_API_KEY"] = "***"
# os.environ["OPENAI_BASE_URL"] = "***"

# always remember to put these lines at the top of your code if you are using clash
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"
# os.environ["all_proxy"] = "socks5://127.0.0.1:7890"


import json
from tqdm import tqdm
from eval_helper.get_evaluation import get_evaluation
from translation_output_helper.get_translation_output import get_translation_output

from agentverse.agentverse import AgentVerse
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--config", type=str, default="agentverse/tasks/nl_sl_translation/logideduct_translate_config.yaml")
parser.add_argument("--reverse_input", default=False, action="store_true")


args = parser.parse_args()

agentverse, args_data_path, args_output_dir = AgentVerse.from_task(args.config)

print(args)

os.makedirs(args_output_dir, exist_ok=True)
with open(os.path.join(args_output_dir, "args.txt"), "w") as f:
    f.writelines(str(args))

# uncomment this line if you don't want to overwrite your output_dir
# if os.path.exists(args_output_dir) and len(os.listdir(args_output_dir)) > 1 :
#
#     raise ValueError("the output_dir is not empty, check if is expected.")

with open(args_data_path) as f:
    data = json.load(f)

if "ruozhiba" in args_data_path:
    pair_comparison_output = []

    for num, ins in enumerate(tqdm(data, desc="Processing ruozhiba", unit="instance")):

        print(f"================================instance {num}====================================")

        # reassign the text to agents, and set final_prompt to null for debate at first round
        for agent_id in range(len(agentverse.agents)):
            agentverse.agents[agent_id].question = ins["instruction"]

            # if args.reverse_input:
            #     agentverse.agents[agent_id].compared_text_one = ins["response"]["vicuna"]
            #     agentverse.agents[agent_id].compared_text_two = ins["response"]["gpt35"]
            # else:
            #     agentverse.agents[agent_id].compared_text_one = ins["response"]["gpt35"]
            #     agentverse.agents[agent_id].compared_text_two = ins["response"]["vicuna"]

            agentverse.agents[agent_id].final_prompt = ""
        agentverse.run()

        evaluation = get_evaluation(setting="every_agent", messages=agentverse.agents[0].memory.messages, agent_nums=len(agentverse.agents))

        pair_comparison_output.append({"question": ins["instruction"],
                                       "evaluation": evaluation})

        os.makedirs(args_output_dir, exist_ok=True)
        with open(os.path.join(args_output_dir, "pair_comparison_results.json"), "w") as f:
            json.dump(pair_comparison_output, f, indent=4)
    # with open(os.path.join(args_output_dir, "gt_origin_results.json"), "w") as f:
    #     json.dump(gt_origin_output, f, indent=4)

elif "ProofWriter" in args_data_path:
    # 处理ProofWriter数据集
    proof_writer_output = []

    for num, ins in enumerate(tqdm(data[:30], desc="Processing ProofWriter", unit="instance")):
        print(f"================================instance {num}====================================")

        for agent_id in range(len(agentverse.agents)):
            agentverse.agents[agent_id].context = ins["context"]  # 赋值prompt template中的${context}字段
            agentverse.agents[agent_id].question = ins["question"].strip()  # 赋值prompt template中的${question}字段
            agentverse.agents[agent_id].final_prompt = ""

        agentverse.run()

        chat_history, translations = get_translation_output(setting="every_agent", messages=agentverse.agents[0].memory.messages,
                                    agent_nums=len(agentverse.agents))

        proof_writer_output.append({
            "id": ins["id"],
            "context": ins["context"],
            "question": ins["question"],
            "options": ins["options"],
            "answer": ins["answer"],
            "chat_history": chat_history,
            "translation": translations
        })

        os.makedirs(args_output_dir, exist_ok=True)
        with open(os.path.join(args_output_dir, "translation_results.json"), "w") as f:
            json.dump(proof_writer_output, f, indent=4)

elif "FOLIO" in args_data_path:
# 处理ProofWriter数据集
    smoketest_output = []

    for num, ins in enumerate(tqdm(data, desc="Processing FOLIO", unit="instance")):
        print(f"================================instance {num}====================================")

        for agent_id in range(len(agentverse.agents)):
            agentverse.agents[agent_id].context = ins["context"]  # 赋值prompt template中的${context}字段
            agentverse.agents[agent_id].question = ins["question"].strip()  # 赋值prompt template中的${question}字段
            agentverse.agents[agent_id].final_prompt = ""

        agentverse.run()

        chat_history, translations = get_translation_output(setting="every_agent", messages=agentverse.agents[0].memory.messages,
                                    agent_nums=len(agentverse.agents))

        smoketest_output.append({
            "id": ins["id"],
            "context": ins["context"],
            "question": ins["question"],
            "options": ins["options"],
            "answer": ins["answer"],
            "chat_history": chat_history,
            "translation": translations
        })

        os.makedirs(args_output_dir, exist_ok=True)
        with open(os.path.join(args_output_dir, "translation_results.json"), "w") as f:
            json.dump(smoketest_output, f, indent=4)

elif "LogicalDeduction" in args_data_path:
    # 处理LogicalDeduction数据集
    logical_deduction_output = []

    for num, ins in enumerate(tqdm(data, desc="Processing LogicalDeduction", unit="instance")):
        print(f"================================instance {num}====================================")

        # 将options数组转换为格式化的字符串
        options_str = "\n".join(ins["options"])

        for agent_id in range(len(agentverse.agents)):
            agentverse.agents[agent_id].context = ins["context"]  # 赋值prompt template中的${context}字段
            agentverse.agents[agent_id].question = ins["question"].strip()  # 赋值prompt template中的${question}字段
            agentverse.agents[agent_id].options = options_str  # 赋值prompt template中的${options}字段
            agentverse.agents[agent_id].final_prompt = ""

        agentverse.run()

        chat_history, translations = get_translation_output(setting="every_agent", messages=agentverse.agents[0].memory.messages,
                                    agent_nums=len(agentverse.agents))

        logical_deduction_output.append({
            "id": ins["id"],
            "context": ins["context"],
            "question": ins["question"],
            "options": ins["options"],
            "answer": ins["answer"],
            "chat_history": chat_history,
            "translation": translations
        })

        os.makedirs(args_output_dir, exist_ok=True)
        with open(os.path.join(args_output_dir, "translation_results.json"), "w") as f:
            json.dump(logical_deduction_output, f, indent=4)

else:
    raise ValueError(f"Unsupported dataset in run_translate.py: {args_data_path}")

    