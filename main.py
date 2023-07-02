from sklearn.preprocessing import StandardScaler
from json import load
import streamlit as st
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

rfc_feature_df = pd.read_csv('data/rfc_removed_feature_ncr.csv')

new_rfc_feature_df = pd.read_csv('data/new_rfc_removed_feature_ncr.csv')
new_rfc_label_df = pd.read_csv('data/new_rfc_removed_label_ncr.csv')

after_resampling_df = new_rfc_feature_df.assign(HeartDisease=new_rfc_label_df)

# after_resampling_df.loc[(after_resampling_df.BMI > 40.79), 'BMI'] = 26.57

# after_resampling_df.loc[(after_resampling_df.BMI < 12.91), 'BMI'] = 26.57

x = after_resampling_df.drop(columns=['HeartDisease'], axis=1)
y = after_resampling_df['HeartDisease']

np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)


def transform_diabetes(data):
    result = 3
    if(data == 'Tidak'):
        result = 0
    elif(data == 'Prediabetes'):
        result = 1
    elif(data == 'Ya'):
        result = 2
    return (result)


def transform_genhealth(data):
    result = 4
    if(data == 'Buruk'):
        result = 0
    elif(data == 'Cukup'):
        result = 1
    elif(data == 'Baik'):
        result = 2
    elif(data == 'Sangat Baik'):
        result = 3
    return (result)


def load_model():
    with open('heart_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()

logistic = data


st.title("Prediksi Penyakit Jantung")

# st.write("#### Tolong Masukkan Beberapa Informasi untuk Model memprediksi apakah mungkin anda memiliki penyakit jantung atau tidak")
st.subheader('Tolong Masukkan Beberapa Informasi untuk Model memprediksi apakah mungkin anda memiliki penyakit jantung atau tidak')


st.write("###### Cari tahu Body Mass Index anda dengan memasukkan Berat Badan dan Tinggi Badan anda")

weight = st.number_input("Masukkan Berat Badan anda dalam (Kilogram)")
height = st.number_input("Masukkan Tinggi Badan anda dalam (Centimeter)")

if height > 0 and weight > 0:
    body_mass = weight/(height**2)*10000
    st.write("BMI anda : {:.2f}".format(body_mass))
else:
    st.write("Isi berat dan tinggi badan anda terlebih dahulu")

bmi = st.number_input("Masukkan Body Mass Index anda")

isSmoker = st.selectbox(
    "Apakah Anda Telah Merokok Paling Sedikit 100 Rokok sepanjang hidup ada?", ["Ya", "Tidak"])

alcoholic = st.selectbox(
    "Apa anda seorang pria peminum alkohol lebih dari 14 minuman setiap minggu, atau apa anda wanita peminum alkohol lebih dari 7 minuman setiap minggu?", ["Ya", "Tidak"])

stroke = st.selectbox(
    "Pernah dibilang punya atau tahu ada penyakit Stroke?", ["Ya", "Tidak"])

physicalhealth = st.slider(
    "Tentang kesehatah fisik anda, termasuk penyakit fisik atau luka, sudah berapa hari semenjak 30 hari terakhir kesehatan fisik anda tidak baik? (0-30 hari)", 0, 30, 0)

diffwalking = st.selectbox(
    "Apa anda mempunyai kesulitan serius dalam berjalan atau menaiki tangga?", ["Ya", "Tidak"])

gender = st.selectbox("Jenis Kelamin Anda", ["Pria", "Wanita"])

age = st.number_input("Masukkan umur anda")

diabetic = st.selectbox("Pernah dibilang punya atau tahu ada penyakit Diabetes?", [
                        "Ya", "Tidak", "Prediabetes", "Ya (Saat Hamil)"])

physactivity = st.selectbox(
    "Pernah melakukan aktivitas fisik atau olahraga dalam 30 hari terakhir selain pekerjaan biasa anda?", ["Ya", "Tidak"])

genhealth = st.selectbox(
    "Katakan seberapa baik secara umum kesehatan anda? (Buruk - Baik Sekali)", ["Buruk", "Cukup", "Baik", "Sangat Baik", "Baik Sekali"])

asthma = st.selectbox(
    "Pernah dibilang punya atau tahu ada penyakit Asthma?", ["Ya", "Tidak"])

kidney = st.selectbox(
    "Pernah dibilang punya atau tahu ada penyakit Ginjal?", ["Ya", "Tidak"])

skincancer = st.selectbox(
    "Pernah dibilang punya atau tahu ada penyakit Kanker Kulit?", ["Ya", "Tidak"])

ok = st.button('Prediksi')


if ok:
    df_pred = pd.DataFrame([[bmi, isSmoker, alcoholic, stroke, physicalhealth, diffwalking, gender, age, diabetic, physactivity, genhealth, asthma, kidney, skincancer]],
                           columns=['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer'])

    df_pred['Smoking'] = df_pred['Smoking'].apply(
        lambda x: 1 if x == 'Ya' else 0)

    df_pred['AlcoholDrinking'] = df_pred['AlcoholDrinking'].apply(
        lambda x: 1 if x == 'Ya' else 0)

    df_pred['Stroke'] = df_pred['Stroke'].apply(
        lambda x: 1 if x == 'Ya' else 0)

    df_pred['DiffWalking'] = df_pred['DiffWalking'].apply(
        lambda x: 1 if x == 'Ya' else 0)

    df_pred['Sex'] = df_pred['Sex'].apply(
        lambda x: 1 if x == 'Pria' else 0)

    df_pred['Diabetic'] = df_pred['Diabetic'].apply(transform_diabetes)

    df_pred['PhysicalActivity'] = df_pred['PhysicalActivity'].apply(
        lambda x: 1 if x == 'Ya' else 0)

    df_pred['GenHealth'] = df_pred['GenHealth'].apply(transform_genhealth)

    df_pred['Asthma'] = df_pred['Asthma'].apply(
        lambda x: 1 if x == 'Ya' else 0)

    df_pred['KidneyDisease'] = df_pred['KidneyDisease'].apply(
        lambda x: 1 if x == 'Ya' else 0)

    df_pred['SkinCancer'] = df_pred['SkinCancer'].apply(
        lambda x: 1 if x == 'Ya' else 0)

    # st.write(f'{df_pred}')
    st.table(df_pred)
    sc = StandardScaler()
    # rfc_feature = sc.fit_transform(rfc_feature_df)
    X_train = sc.fit_transform(X_train)
    predict_sc = sc.transform(df_pred)
    # st.write(f'{predict_sc}')
    predict_proba = logistic.predict_proba(predict_sc)
    predict = logistic.predict(predict_sc)
    st.write(f'{predict_proba}')
    st.write(f'{predict}')
    if(predict[0] == 0):
        st.write(f'<p class="big-font">Logistic Regression : Anda Mempunyai Kemungkinan {predict_proba[0][0] * 100:.2f}% Tidak Memiliki Penyakit Jantung.</p>',
                 unsafe_allow_html=True)

    else:
        st.write(f'<p class="big-font">Logistic Regression : Anda Mempunyai Kemungkinan {predict_proba[0][1] * 100:.2f}% Memiliki Penyakit Jantung.</p>',
                 unsafe_allow_html=True)

    # st.write(f"Look : ${predict_sc}")
