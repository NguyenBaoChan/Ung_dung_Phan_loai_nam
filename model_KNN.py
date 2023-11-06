
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# Quy định chiều ngang và chiều dọc của màn hình
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Đọc dữ liệu từ tệp CSV
mushrooms = pd.read_csv('mushrooms_clean.csv')
st.write('''
# <h1 style="font-size:65px; color: blue">:mushroom: Mô hình dự đoán với thuật toán KNN :mushroom: </h1><h1 style="font-size:45px; color: blue">:hibiscus: Dự đoán độ chính xác của thuật toán KNN trên toàn tập dữ liệu.</h1><h2 style="font-size:35px; color: blue"> Lựa chọn siêu tham số K (số láng giềng)</h2>
''', unsafe_allow_html=True)

# Xáo trộn dữ liệu
mushrooms = mushrooms.sample(frac=1, random_state=42).reset_index(drop=True)

#ma hóa đặt trưng
df = mushrooms.copy()
target = "class"
encode = list(df.loc[:, df.columns != "class"].columns)

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
#gan nhan nấm ăn đc và kh ăn đc
target_mapper = {"Edible": 0, "Poisonous": 1}

def target_encode(val):
    return target_mapper[val]

df["class"] = df["class"].apply(target_encode)


X = df.iloc[:, 1:]
Y = df.iloc[:, 0]

# Chia lại dữ liệu thành tập huấn luyện và tập kiểm tra với tỷ lệ 6:4
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

# Lựa chọn siêu tham số K (số láng giềng)
st.markdown("<h2 style='font-size:35px'>Số láng giềng (K):</h2>", unsafe_allow_html=True)
k = st.number_input("Nhập số láng giềng (K):", min_value=1, value=20, step=1)

# Xây dựng mô hình KNN với k được lựa chọn
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, Y_train)

# Dự đoán trên tập kiểm tra
y_pred = knn.predict(X_test)

# Tính chỉ số chính xác, hiện thị chỉ số chanh ac
accuracy = accuracy_score(Y_test, y_pred)
st.markdown("<h2 style='font-size:28px'>Kết quả dự đoán: </h2>", unsafe_allow_html=True)
st.write("Accuracy: ", accuracy)

# Lưu mô hình
with open("mushrooms_knn.pkl", "wb") as file:
    pickle.dump(knn, file)
