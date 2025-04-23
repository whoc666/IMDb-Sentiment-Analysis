import gradio as gr
import pandas as pd
import joblib
import plotly.express as px

# 加载模型和向量器
model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# 单句预测函数
def predict_sentiment(text):
    vect_text = vectorizer.transform([text])
    pred = model.predict(vect_text)[0]
    return "正面 😊" if pred == 1 else "负面 😞"

# 批量预测函数（上传CSV）
def batch_predict(file):
    df = pd.read_csv(file.name)
    if "text" not in df.columns:
        return "CSV中必须包含名为 'text' 的列", None
    vect_texts = vectorizer.transform(df["text"])
    df["prediction"] = model.predict(vect_texts)
    df["prediction"] = df["prediction"].map({1: "正面", 0: "负面"})
    output_file = "predictions.csv"
    df.to_csv(output_file, index=False)
    return "预测完成 ✅", output_file

# 可视化函数
def visualize_sentiment(text):
    vect = vectorizer.transform([text])
    probas = model.predict_proba(vect)[0]
    df = pd.DataFrame({
        "情感": ["负面", "正面"],
        "概率": [probas[0], probas[1]]
    })
    fig = px.bar(df, x="情感", y="概率", color="情感", title="情感分布预测")
    return fig

# 单句预测界面
single_input = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="请输入影评文本..."),
    outputs="text",
    title="🎬 IMDb情感预测（单条）"
)

# 批量预测界面
batch_input = gr.Interface(
    fn=batch_predict,
    inputs=gr.File(label="上传包含 text 列的 CSV 文件"),
    outputs=["text", "file"],
    title="📦 批量情感预测（上传CSV）"
)

# 可视化界面
visual_input = gr.Interface(
    fn=visualize_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="请输入影评文本..."),
    outputs="plot",
    title="📊 情感概率分布图"
)

# 多功能组合页面
demo = gr.TabbedInterface(
    interface_list=[single_input, batch_input, visual_input],
    tab_names=["单条预测", "批量预测", "情感可视化"]
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
