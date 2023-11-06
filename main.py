import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
import pickle
from PIL import Image

img = Image.open('mshrmparts.jpeg')

st.write('''
# :mushroom: Ăn Hay Không Ăn? :mushroom:
Ứng dụng này dự đoán liệu một loại nấm ăn được hay độc.
''')

st.sidebar.header("Vui lòng nhập các tính năng ở đây:")


def user_input_features():
    cap_shape = st.sidebar.selectbox("Cap Shape", ("Bell", "Conical", "Convex", "Flat", "Knobbed", "Sunken"))
    cap_surface = st.sidebar.selectbox("Cap Surface", ("Fibrous", "Grooves", "Scaly", "Smooth"))
    cap_color = st.sidebar.selectbox("Cap Colour", (
    "Brown", "Buff", "Cinnamon", "Gray", "Green", "Pink", "Purple", "Red", "White", "Yellow"))
    bruises = st.sidebar.selectbox("Are bruises present?", ("Yes", "No"))
    odor = st.sidebar.selectbox("Odor",
                                ("Almond", "Anise", "Creosote", "Fishy", "Foul", "Musty", "Pungent", "Spicy", "None"))
    gill_attachment = st.sidebar.selectbox("Gill Attachment", ("Attached", "Free"))
    gill_spacing = st.sidebar.selectbox("Gill Spacing", ("Close", "Crowded"))
    gill_size = st.sidebar.selectbox("Gill Size", ("Broad", "Narrow"))
    gill_color = st.sidebar.selectbox("Gill Colour", (
    "Black", "Brown", "Buff", "Chocolate", "Gray", "Green", "Orange", "Pink", "Purple", "Red", "White", "Yellow"))
    stalk_shape = st.sidebar.selectbox("Stalk Shape", ("Enlarging", "Tapering"))
    stalk_root = st.sidebar.selectbox("Stalk Root", ("Bulbous", "Club", "Equal", "Rooted", "Missing"))
    stalk_surface_above_ring = st.sidebar.selectbox("Stalk Surface Above Ring", ("Fibrous", "Scaly", "Silky", "Smooth"))
    stalk_surface_below_ring = st.sidebar.selectbox("Stalk Surface Below Ring", ("Fibrous", "Scaly", "Silky", "Smooth"))
    stalk_color_above_ring = st.sidebar.selectbox("Stalk Colour Above Ring", (
    "Brown", "Buff", "Cinnamon", "Gray", "Orange", "Pink", "Red", "White", "Yellow"))
    stalk_color_below_ring = st.sidebar.selectbox("Stalk Colour Below Ring", (
    "Brown", "Buff", "Cinnamon", "Gray", "Orange", "Pink", "Red", "White", "Yellow"))
    # veil_type=st.sidebar.selectbox("Veil Type",("Partial"))
    veil_color = st.sidebar.selectbox("Veil Colour", ("Brown", "Orange", "White", "Yellow"))
    ring_number = st.sidebar.selectbox("Ring Number", ("None", "One", "Two"))
    ring_type = st.sidebar.selectbox("Ring Type", ("Evanescent", "Flaring", "Large", "Pendant", "None"))
    spore_print_color = st.sidebar.selectbox("Spore Print Colour", (
    "Black", "Brown", "Buff", "Chocolate", "Green", "Orange", "Purple", "White", "Yellow"))
    population = st.sidebar.selectbox("Population",
                                      ("Abundant", "Clustered", "Numerous", "Scattered", "Several", "Solitary"))
    habitat = st.sidebar.selectbox("Habitat", ("Grasses", "Leaves", "Meadows", "Paths", "Urban", "Waste", "Woods"))

    data = {'cap_shape': cap_shape,
            'cap_surface': cap_surface,
            'cap_color': cap_color,
            'bruises': bruises,
            'odor': odor,
            'gill_attachment': gill_attachment,
            'gill_spacing': gill_spacing,
            'gill_size': gill_size,
            'gill_color': gill_color,
            'stalk_shape': stalk_shape,
            'stalk_root': stalk_root,
            'stalk_surface_above_ring': stalk_surface_above_ring,
            'stalk_surface_below_ring': stalk_surface_below_ring,
            'stalk_color_above_ring': stalk_color_above_ring,
            'stalk_color_below_ring': stalk_color_below_ring,
            'veil_type': "Partial",
            'veil_color': veil_color,
            'ring_number': ring_number,
            'ring_type': ring_type,
            'spore_print_color': spore_print_color,
            'population': population,
            'habitat': habitat}

    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()


mushrooms_svm = pd.read_csv('mushrooms_clean.csv')
mushrooms = mushrooms_svm.drop(columns=["class"])
df = pd.concat([input_df, mushrooms], axis=0)

mushrooms_knn = pd.read_csv('mushrooms_clean.csv')
mushrooms = mushrooms_knn.drop(columns=["class"])
dp= pd.concat([input_df, mushrooms], axis=0)


encode = list(df.columns)
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
    ###################
encode = list(dp.columns)
for col1 in encode:
    dummy1 = pd.get_dummies(dp[col1], prefix=col1)
    dp = pd.concat([dp, dummy1], axis=1)
    del dp[col1]

df = df[:1]
dp = dp[:1]
load_clf = pickle.load(open("mushrooms_svm.pkl", "rb"))
load_clf1 = pickle.load(open("mushrooms_KNN.pkl", "rb"))
#áp dụng mô hình để dự báo
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

prediction1 = load_clf1.predict(dp)
prediction1_proba = load_clf1.predict_proba(dp)


if prediction == 0:
    answer = "Edible"
else:
    answer = "Poisonous"
    ###########
if prediction1 == 0:
        answer1 = "Edible"
else:
        answer1= "Poisonous"

col1, col2 = st.columns(2)

with col1:
    st.subheader("Dự đoán kết quả SVM")
    mushrooms_class = np.array(["Edible", "Poisonous"])

    if prediction == 0:
        st.success("your mushrooms can be eaten  :relaxed:")
        st.balloons()
    else:
        st.error("your mushrooms are not edible :cry:")

    st.subheader("Xác suất dự đoán SVM")
    st.write("Lưu ý: 0 là xác suất ăn được, 1 là xác suất là độc")
    st.write(prediction_proba)

    st.subheader("Dự đoán kết quả KNN")
    mushrooms1_class = np.array(["Edible", "Poisonous"])

    if prediction1 == 0:
        st.success("your mushrooms can be eaten  :relaxed:")
        st.balloons()
    else:
        st.error("your mushrooms are not edible :cry:")

    st.subheader("Xác suất dự đoán KNN")
    st.write("Lưu ý: 0 là xác suất ăn được, 1 là xác suất là độc")
    st.write(prediction1_proba)

with col2:
    st.image(img, caption="Source:https://grocycle.com/parts-of-a-mushroom/", width=350)

