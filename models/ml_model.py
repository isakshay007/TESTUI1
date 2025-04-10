"""
ml_model.py

此脚本用来：
1. 读取并预处理 StackOverflow 问题数据（文本 + 代码 + 标签）。
2. 训练多标签分类模型，预测问题所属的标签。
3. 将训练好的模型保存到本地文件，供推断时使用。
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, hamming_loss
import joblib

def load_and_preprocess_data(data_path):
    """
    读取CSV文件并做必要的预处理:
    1. 合并(title, description, code)作为一个整体的文本输入。
    2. 解析多标签列（tags）。
    """

    # 读取数据
    df = pd.read_csv(data_path)

    # 如果实际列名不一致，请修改下方的列名引用
    # 将 title、description、code 合并成为一个文本特征，方便向量化
    # 若某些字段没有，也可以省略
    df['combined_text'] = df['title'].fillna('') + ' ' \
                          + df['description'].fillna('') + ' ' \
                          + df['code'].fillna('')

    # 将tags列转换成list，如 "python;machine-learning" -> ["python","machine-learning"]
    # 请注意如果你的标签分隔符不是 ';'，需要自行修改
    df['tag_list'] = df['tags'].apply(lambda x: x.split(';'))

    return df

def build_and_train_model(df):
    """
    使用 OneVsRestClassifier + LogisticRegression 搭配 TfidfVectorizer
    构建多标签分类模型，并进行训练和简单评估。
    """
    # 准备特征和标签
    X = df['combined_text']
    y_raw = df['tag_list']

    # 将多标签转为多列二进制形式（MLB：MultiLabelBinarizer）
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y_raw)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2,   # 可根据数据规模调整测试集占比
        random_state=42
    )

    # 构建一个Pipeline:
    # 1. TfidfVectorizer 用于文本转向量
    # 2. OneVsRestClassifier(LogisticRegression()) 用于多标签分类
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),  
        ('clf', OneVsRestClassifier(LogisticRegression(max_iter=1000)))
    ])

    # 训练模型
    pipeline.fit(X_train, y_train)

    # 在测试集上评估模型
    y_pred = pipeline.predict(X_test)
    # 这里给出 F1-score 和 Hamming loss 作为示例指标，可根据需要增加/更换
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    h_loss = hamming_loss(y_test, y_pred)
    print(f"Micro F1-score: {micro_f1:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Hamming Loss:   {h_loss:.4f}")

    # 返回训练好的模型和 MultiLabelBinarizer
    return pipeline, mlb

if __name__ == "__main__":
    # 1. 读取 & 预处理数据
    data_path = "data/stackoverflow_data.csv"  # 根据实际数据文件路径调整
    df = load_and_preprocess_data(data_path)

    # 2. 训练模型
    model, mlb = build_and_train_model(df)

    # 3. 保存模型 & MultiLabelBinarizer
    #   说明：模型中已经包含了TfidfVectorizer，所以推断时不需要单独加载向量器
    model_filename = "tagging_model.pkl"
    mlb_filename = "tagging_mlb.pkl"
    joblib.dump(model, model_filename)
    joblib.dump(mlb, mlb_filename)
    print(f"模型已保存到: {model_filename}")
    print(f"MultiLabelBinarizer已保存到: {mlb_filename}")
