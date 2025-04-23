```markdown
# 🎬 IMDb 电影评论情感分类系统

本项目基于 IMDb 电影评论数据集，构建了一个完整的情感分类系统，能够识别评论是 “正面” 还是 “负面”。该系统结合了传统机器学习方法与现代 Web 接口，适合初学者学习自然语言处理（NLP）项目全流程。

---

## ✨ 项目功能亮点

- **自动数据预处理**：清洗文本内容，规范大小写，去除噪声词。
- **模型训练与评估**：基于 TF-IDF 向量化 + 逻辑回归模型，准确率达 88.6%。
- **命令行接口**：输入任意文本即可获取情感预测结果。
- **图形化 Web 界面（Gradio）**：
  - ✅ 单条影评预测  
  - ✅ 支持上传 CSV 文件进行批量预测  
  - ✅ 可视化情感分布图（Plotly）
- **可部署性强**：模型与向量器已持久化保存，适用于后续部署或迁移。

---

## 🗂️ 项目结构

```
IMDb/
├── data/                    # 原始和清洗后的数据  
│   └── imdb_train_clean.csv  
├── models/                  # 保存的模型和向量器  
│   ├── logistic_model.pkl  
│   └── tfidf_vectorizer.pkl  
├── scripts/                 # 各功能脚本  
│   ├── clean_text.py        # 数据清洗  
│   ├── train_model.py       # 模型训练  
│   ├── predict_cli.py       # 命令行预测  
│   └── predict_gradio.py    # 图形化 Web 界面  
├── predictions.csv          # 批量预测输出文件  
└── README.md                # 项目说明文件
```

---

## 📦 安装依赖

建议使用 [Anaconda](https://www.anaconda.com/) 或 `venv` 创建虚拟环境：

```bash
# 推荐
pip install -r requirements.txt

# 如果没有 requirements.txt 文件，可手动安装：
pip install pandas scikit-learn gradio plotly joblib
```

---

## 🚀 使用方法

```bash
# 1️⃣ 数据清洗
python scripts/clean_text.py

# 2️⃣ 训练模型
python scripts/train_model.py

# 3️⃣ 命令行预测
python scripts/predict_cli.py --text "This movie is fantastic!"

# 4️⃣ 启动 Web 界面
python scripts/predict_gradio.py
```

---

## 📊 示例截图（点击放大）

> 以下截图请替换为你本地或图床的实际路径

### ▶️ 单条影评情感预测  
[image](https://github.com/user-attachments/assets/5a7b9596-911c-4c00-9b0e-fb834bf5567a)

### 📁 批量影评预测（上传 CSV） 
[image](https://github.com/user-attachments/assets/325f57a0-72b8-43a5-b3d2-a38ef593decd)

### 📈 情感分布图表  
[image](https://github.com/user-attachments/assets/a6d7f91a-d4a6-4b9d-9d2e-29d24223d5cb)

---

## 💡 项目进阶建议

- 尝试引入深度学习模型（如 BERT）提升效果
- 增加 REST API 支持（如 FastAPI + Docker）
- 增加多语言支持或用户登录功能

---

## 📚 数据来源

> IMDb 数据集（[Hugging Face Datasets](https://huggingface.co/datasets/imdb)）

---

## 📄 许可证

本项目仅用于学习与研究目的，未经许可请勿用于商业用途。
