# scripts/train_model.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. 获取数据路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, "data", "imdb_train_clean.csv")

# 2. 加载数据
df = pd.read_csv(data_path)
texts = df["text"].astype(str).tolist()
labels = df["label"].tolist()

# 3. 划分训练和验证集
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 4. 文本特征提取：TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# 5. 模型训练：逻辑回归
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

# 6. 模型评估
y_pred = model.predict(X_val_tfidf)
accuracy = accuracy_score(y_val, y_pred)
print(f"\n验证集准确率: {accuracy:.4f}\n")
print("详细分类报告：")
print(classification_report(y_val, y_pred))

import joblib
import os

# 创建 models 目录（如不存在）
model_dir = os.path.join(project_root, "models")
os.makedirs(model_dir, exist_ok=True)

# 保存模型
model_path = os.path.join(model_dir, "logistic_model.pkl")
vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"\n模型已保存至: {model_path}")
print(f"向量器已保存至: {vectorizer_path}")
