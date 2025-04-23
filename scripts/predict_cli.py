import os
import argparse
import joblib
import re
import string

# 获取项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 加载模型与向量器
model_path = os.path.join(project_root, "models", "logistic_model.pkl")
vectorizer_path = os.path.join(project_root, "models", "tfidf_vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# 文本预处理函数（与之前一样）
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 情感预测函数
def predict_sentiment(text):
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    prediction = model.predict(vec)[0]
    return "正面情绪" if prediction == 1 else "负面情绪"

# 主函数：命令行接口部分
def main():
    parser = argparse.ArgumentParser(description="IMDb情感分析模型")
    parser.add_argument("text", type=str, help="待预测的文本")
    args = parser.parse_args()

    # 获取用户输入的文本并进行预测
    sentiment = predict_sentiment(args.text)
    print(f"【预测结果】{sentiment}")

# 执行入口
if __name__ == "__main__":
    main()
