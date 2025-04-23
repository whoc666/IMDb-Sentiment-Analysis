# scripts/clean_text.py
import pandas as pd
import os
import re

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, "data")

# 定义文本清洗函数
def clean_text(text):
    text = re.sub(r"<br />", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower()
    return text.strip()

# 清洗并保存数据
def process_and_save(input_filename, output_filename):
    input_path = os.path.join(data_dir, input_filename)
    output_path = os.path.join(data_dir, output_filename)
    
    df = pd.read_csv(input_path)
    df["text"] = df["text"].apply(clean_text)
    df.to_csv(output_path, index=False)
    print(f"已保存清洗后的数据：{output_path}")

# 执行主流程
process_and_save("imdb_train.csv", "imdb_train_clean.csv")
