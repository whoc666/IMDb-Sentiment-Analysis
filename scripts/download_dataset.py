# scripts/download_dataset.py
from datasets import load_dataset
import os

# 获取当前脚本所在目录的上一级，也就是项目根目录 IMDb/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 创建 data 目录路径
data_dir = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True)

# 加载 IMDb 数据集
dataset = load_dataset("imdb")

# 保存为 CSV
train_path = os.path.join(data_dir, "imdb_train.csv")
test_path = os.path.join(data_dir, "imdb_test.csv")
dataset["train"].to_csv(train_path, index=False)
dataset["test"].to_csv(test_path, index=False)

print(f"数据集已保存至：\n{train_path}\n{test_path}")
