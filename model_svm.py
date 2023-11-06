
import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn import svm
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

# Quy định chiều ngang và chiều dọc của màn hình
st.set_page_config(layout="wide", initial_sidebar_state="expanded")


mushrooms = pd.read_csv('mushrooms_clean.csv')
st.write('''
# <h1 style="font-size:65px; color: blue">:mushroom: Đánh giá mô hình dự đoán với thuật toán SVM :mushroom: </h1> <h1 style="font-size:45px; color: blue">:hibiscus: Dự đoán độ chính xác của thuật toán KNN trên toàn tập dữ liệu.</h1> <h2 style="font-size:35px; color: blue"> Lựa chọn hàm Kernel và siêu tham số C</h1>
''', unsafe_allow_html=True)

# Xáo trộn dữ liệu
mushrooms = mushrooms.sample(frac=1, random_state=42).reset_index(drop=True)


df = mushrooms.copy()
target = "class"
encode = list(df.loc[:, df.columns != "class"].columns)

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

target_mapper = {"Edible": 0, "Poisonous": 1}

def target_encode(val):
    return target_mapper[val]

df["class"] = df["class"].apply(target_encode)


X = df.drop("class", axis=1)
Y = df["class"]

# Chia lại dữ liệu thành tập huấn luyện và tập kiểm tra với tỷ lệ 6:4
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

# Lựa chọn kernel và siêu tham số C
st.markdown("<h2 style='font-size:35px'>Kernel:</h2>", unsafe_allow_html=True)
kernel = st.selectbox("Chọn: " ,["linear", "rbf", "poly", "sigmoid"])
st.markdown("<h2 style='font-size:35px'>Điều chỉnh tham số C:</h2>", unsafe_allow_html=True)
C = st.slider("Điều chỉnh:", 0.01, 10.0, 1.0)


if kernel == "linear":
    model_svc = svm.LinearSVC(C=C)
else:
    model_svc = svm.SVC(kernel=kernel, C=C)

model = CalibratedClassifierCV(model_svc)
model.fit(X_train, Y_train)

# Đánh giá mô hình trên tập kiểm tra
accuracy = model.score(X_test, Y_test)
st.markdown("<h2 style='font-size:28px'>Kết quả dự đoán: </h2>", unsafe_allow_html=True)
st.write("Accuracy: ", accuracy)

# Lưu mô hình
import pickle
pickle.dump(model, open("mushrooms_svm.pkl", "wb"))
