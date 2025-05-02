import pandas as pd
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from collections import Counter

def load_model():
    model_name = "Qwen/Qwen3-30B-A3B"
    #model_name = "Qwen/Qwen3-0.6B"
    
    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    return model, tokenizer

def format_question(row):
    # 根据实际数据集格式提取问题和选项
    columns = row.values
    question = columns[0]
    choices = columns[1:5]  # 四个选项
    correct_answer = columns[5]  # 正确答案
    
    prompt = f"问题：{question}\n"
    prompt += "选项：\n"
    for i, choice in enumerate(['A', 'B', 'C', 'D']):
        prompt += f"{choice}. {choices[i]}\n"
    prompt += "\n请只回答选项字母（A、B、C或D）。"
    return prompt, correct_answer

def get_model_answer(model, tokenizer, prompt):
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 生成回答
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=128,  # 由于只需要选项答案，可以限制生成长度
        temperature=0.1  # 降低随机性
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    
    # 解析thinking content和答案
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    # 提取第一个出现的选项字母
    for char in content:
        if char in ['A', 'B', 'C', 'D']:
            return char
    return None

def get_majority_answer(model, tokenizer, prompt, n=5):
    answers = []
    for _ in range(n):
        ans = get_model_answer(model, tokenizer, prompt)
        if ans is not None:
            answers.append(ans)
    if not answers:
        return None
    most_common = Counter(answers).most_common(1)[0][0]
    return most_common

def evaluate_arc_easy():
    # 加载模型
    print("正在加载模型...")
    model, tokenizer = load_model()
    
    # 读取数据集
    print("正在读取数据集...")
    # 读取CSV时不使用列名
    df = pd.read_csv('/home/hljiang/LLMuncertain/benchmark/test/college_computer_science_test.csv', header=None)
    
    correct = 0
    total = 0
    
    print("开始评测...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt, correct_answer = format_question(row)
        model_answer = get_majority_answer(model, tokenizer, prompt, n=1)
        
        if model_answer == correct_answer:
            correct += 1
        total += 1
        
        if total % 10 == 0:
            print(f"当前准确率: {correct/total*100:.2f}% ({correct}/{total})")
    
    final_accuracy = correct/total*100
    print(f"\n最终准确率: {final_accuracy:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    evaluate_arc_easy()
