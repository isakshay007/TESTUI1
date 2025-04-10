"""
ml_classifier.py

此脚本用来加载训练好的模型，并对新输入文本（标题、描述）进行标签预测，
并输出每个标签的概率分数（方便分析模型行为）。
"""

import joblib
import sys

def load_model_and_mlb(model_path="tagging_model.pkl", mlb_path="tagging_mlb.pkl"):
    model = joblib.load(model_path)
    mlb = joblib.load(mlb_path)
    return model, mlb

def predict_tags(model, mlb, title, description, threshold=0.08):
    # 只使用 title + description（不再拼接 code bc tanishka doesnt use code for tests）
    combined_text = title + " " + description
    probs = model.predict_proba([combined_text])[0]  # 获取每个标签的概率分数

    # 输出每个标签及其分数（降序）output each tag + score
    prob_dict = dict(zip(mlb.classes_, probs))
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

    # 选择概率超过阈值的标签作为最终预测
    predicted_labels = [label for label, score in sorted_probs if score >= threshold]

    return predicted_labels, sorted_probs

if __name__ == "__main__":
    # 模型路径参数（默认）
    model_path = sys.argv[1] if len(sys.argv) > 1 else "tagging_model.pkl"
    mlb_path = sys.argv[2] if len(sys.argv) > 2 else "tagging_mlb.pkl"

    model, mlb = load_model_and_mlb(model_path, mlb_path)

    test_cases = [
    {
        "title": "Using logistic regression for multi-label classification in scikit-learn",
        "desc": "I am building a multi-label text classifier for StackOverflow questions. Each question can have several tags. I plan to use scikit-learn’s OneVsRestClassifier combined with LogisticRegression. I vectorize my text data using TF-IDF. I wonder if logistic regression is suitable for sparse high-dimensional features, or if I should consider random forest or XGBoost."
    },
    {
        "title": "How to split and join strings in Python",
        "desc": "I want to manipulate strings in Python. For example, I have a sentence and I want to split it into words using whitespace and then join those words using a hyphen. What’s the most Pythonic way to do this? Also, are there any built-in functions or methods I should know about?"
    },
    {
        "title": "What causes NullPointerException in Java and how to fix it",
        "desc": "I'm writing a Java application and encountering a NullPointerException. It happens when I try to call a method on an object that may not be initialized. How do I properly handle such cases? Should I add null checks, or use Optional? What are common patterns in Java to avoid this?"
    },
    {
        "title": "CSS Flexbox: How to center content both vertically and horizontally",
        "desc": "I'm trying to use CSS Flexbox to center a div in the middle of a webpage. I’ve tried using justify-content and align-items, but the content is still not perfectly centered. What’s the correct way to center elements using Flexbox? Does the parent element need to have specific height or display settings?"
    },
    {
        "title": "Merging dictionaries in Python efficiently and cleanly",
        "desc": "I have two Python dictionaries and I want to merge them into one. I know about the unpacking syntax like {**dict1, **dict2}, but I’m curious about performance, edge cases, and best practices. Are there other ways to merge dictionaries in Python 3.9 or 3.10 that are more readable or efficient?"
    },
]


    # 执行测试并输出结果
    for i, case in enumerate(test_cases, 1):
        tags, scores = predict_tags(model, mlb, case["title"], case["desc"])

        print(f"\n【Test Case {i}】")
        print(f"title: {case['title']}")
        print(f"predicted tags: {tags}")
        print("Top tag probabilities:")
        for tag, score in scores[:5]:  # 显示前5个分数
            print(f"  {tag}: {score:.3f}")
