from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate, validation_curve, train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
import warnings
import pydotplus
from skompiler import skompile
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
#from catboost import CatBoostClassifier
#from my_functions import *
import streamlit as st
import warnings
from datetime import datetime as date_time
import numpy as np
#import my_functions
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
warnings.filterwarnings("ignore")
df = pd.read_csv("german_credit_data.csv", sep=';')
df = df.drop(columns=["Unnamed: 0"])
#df = [col.upper() for col in df.columns]

st.title("Fakir Demeyelim de Bad Risk Diyelim..")
st.header("Credit Risk Prediction Model")

st.sidebar.markdown("<h1 style='text-align: center;'> KREDİ RİSKİ PARAMETRELERİ </h1>", unsafe_allow_html=True)
#image_path = "Telco_Streamlit/VG3.jpeg"
#st.sidebar.image(image_path, use_column_width=True)
# MENÜ
menu_options = ["Giriş", "Veri Seti Hakkında", "Keşifçi Veri Analizi", "Tahminleme"]
st.sidebar.markdown("<h2>SAYFA MENUSU</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("", menu_options)
# Bugünün Tarihi
current_time = date_time.now().strftime("%d-%m-%Y")
st.sidebar.info("Bugünün Tarihi: {}".format(current_time))



    # GİRİŞ SAYFASININ İÇERİĞİ
if page == "Giriş":

    st.markdown("<h2 style='text-align:center;'> GİRİŞ </h2>", unsafe_allow_html=True)

    st.write(
        """Bu Site 4 sayfadan oluşmaktadır. Sayfalara sol menüden erişebilirsiniz.""")
    st.write("""1. Giriş""")
    st.write("""2. Veri Seti Hakkında""")
    st.write("""3. Keşifçi Veri Analizi""")
    st.write("""4. Tahminleme""")

    st.write(
        """Veri Seti Hakkında sayfasında German Credit Risk Prediction Projesi üzerinden çalışma yapılmıştır.""")
    st.write(
        """Keşifçi Veri Analizi sayfasında veriyi ve grafikleri inceleyebilirsiniz.""")
    st.write(
        """Tahminleme sayfasında eğitim-test verilerinin skorlarını görebilir ve kendiniz veya müşteriler için tahmin yapabilirsiniz.""")




elif page == "Veri Seti Hakkında":
    st.dataframe(df)
    #image = "Telco_Streamlit/Telecom-industry.jpg"
    #st.image(image, use_column_width=True)

    st.markdown("<h2 style='text-align:center;'> - VERİ SETİ HAKKINDA - </h2>", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align:left;'> PROBLEM </h3>", unsafe_allow_html=True)
    st.write(""" - Kredi talebi olan müşterilerin risk kategorisini tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir.
    Modeli geliştirmeden önce veri analizi ve özellik mühendisliği adımları gerçekleştirilecektir.""")

    st.markdown("<h3 style='text-align:left;'> İÇERİK </h3>", unsafe_allow_html=True)
    st.write("""- Kredi riski, borçlunun krediyi geri ödeyememe veya sözleşmelere uyamama olasılığını ifade eder. Bir banka, müşterilerine kredi verdiğinde, müşterilerinin  ödemelerini yapamama riski vardır. Bu durumda, şirket finansal kayıplarla karşılaşabilir.""")
    st.write("""- Good Risk: Bir kişinin veya şirketin karlı duruma getirme olasılığının yüksek olduğuna inanılan bir yatırım olarak değerlendirilmesini ifade eder. Bu terim genellikle krediye uygun bir kişiye veya şirkete verilen bir kredi için kullanılır. İyi risklerin geri ödeme olasılığının son derece yüksek olduğu düşünülür.""")
    st.write("""- Bad Risk: Kötü bir kredi geçmişi, yetersiz gelir veya başka bir nedenle geri ödeme olasılığının düşük olduğu bir krediyi ifade eder. Kötü risk, kredi verenin riskini artırır ve borçlu tarafından ödeme yapmama olasılığını yükseltir.""")
    st.markdown("<h3 style='text-align:left;'> DEĞİŞKENLER </h3>", unsafe_allow_html=True)
    st.write("**-Age   :** Müşteri Yaşı")
    st.write("**-Sex   :** Cinsiyet")
    st.write("**-Job    :** Müşterinin iş niteliği (0-unskilled and non-resident, 1-unskilled and resident, 2-skilled, 3-highly skilled)")
    st.write("**-Housing:** Müşterinin bir evi olup olmadığı (Own, Rent, or free)")
    st.write("**-Savings Account   :** little,moderate,quite rich,rich")
    st.write("**-Checking Account  :** little,moderate,rich")
    st.write("**-Credit Amount     :** Müşterinin talep ettiği kredi tutarı")
    st.write("**-Duration(numeric, in month):** ")
    st.write("**-Purpose:** car, furniture/equipment, radio/TV,domestic appliances, repairs, education, business, vacation/others")
    st.write("**-Risk:** Müşterinin risk grubu (Good or Bad)")


    # KEŞİFSEL VERİ ANALİZİ SAYFASININ İÇERİĞİ
elif page == "Keşifçi Veri Analizi":

    st.markdown("<h2 style='text-align:center;'> - KEŞİFÇİ VERİ ANALİZİ (EDA)- </h2>", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align:left;'>Boyut Bilgileri</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([7.5, 7.5, 5])
    col1.metric("Verinin Şekli:", str(df.shape))
    col2.metric('Gözlem Sayısı:', df.shape[0])
    col3.metric("Değişken Sayısı:", df.shape[1])

    ##################################################################################################################
    def grab_col_names(df, cat_th=10, car_th=20):
        cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
        num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]
        cat_but_car = [col for col in df.columns if
                       df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
        num_cols = [col for col in num_cols if col not in cat_cols]

        print(f"Observations: {df.shape[0]}")
        print(f"Variables:{df.shape[1]}")
        print(f'cat_cols:{len(cat_cols)}')
        print(f'num_cols:{len(num_cols)}')
        print(f'cat_but_car:{len(cat_but_car)}')
        print(f'num_but_cat:{len(num_but_cat)}')
        return cat_cols, num_cols, cat_but_car
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    ################################################################################################
    st.markdown("<h3 style='text-align:left;'>Değişken Bilgileri</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns([5, 5])
    col1.metric("Kategorik:", len(cat_cols))
    col2.metric('Numerik:', len(num_cols))

    st.markdown("<h3 style='text-align:left;'>Değişkenler</h3>", unsafe_allow_html=True)
    st.write("**Kategorik Değişkenler:**")
    st.write(str(cat_cols))
    st.write("**Numerik Değişkenler:**")
    st.write(str(num_cols))

    # Değişkenlerin içeriği
    st.markdown("<h3 style='text-align:left;'> Değişkenlerin İçerikleri (Unique Değerleri) </h3>", unsafe_allow_html=True)
    for column in cat_cols:
        if df[column].dtypes == "object":
            st.write(f'**{column}**: {df[column].nunique()} Adet -> {df[column].unique()}')

    # Gözlem Sayısı (Head, Tail)
    st.markdown("<h3 style='text-align:left;'>Veriye Genel Bakış (Gözlem Sayısı)</h3>", unsafe_allow_html=True)
    n = st.number_input("Baş ve Son Olarak Kaç Adet Gözlemi Görmek İstediğinizi Seçin (1-10):", min_value=1,
                        max_value=10, value=5, step=1)
    #n = st.slider("Baş ve Son Olarak Kaç Adet Gözlemi Görmek İstediğinizi Seçin", 1, 10)
    st.write("Baş :", df.head(n))
    st.write("Son :", df.tail(n))


    # Numerik Değişken Analizi
    st.markdown("<h3 style='text-align:left;'>- Numerik Değişkenlere göre Hedef Değişken Analizi -</h3>",
                unsafe_allow_html=True)
    ###########################################################
    def num_summary(dataframe, numerical_col, plot=False):
        quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1]
        print(dataframe[numerical_col].describe(quantiles).T)
        if plot:
            dataframe[numerical_col].hist()
            plt.xlabel(numerical_col)
            plt.title(numerical_col)
            plt.show(block=True)


    for col in num_cols:
        num_summary(df, col, plot=True)

    num_cols = [col for col in num_cols if 'Unnamed: 0' not in col]

    ################################################################################

    colors = sns.color_palette("pastel")
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(7, 5))
        risk_means = df.groupby("Risk").agg({col: "mean"})
        risk_plot = risk_means.plot(kind="bar", rot=0, ax=ax, color=colors)

        ax.tick_params(axis="both", labelsize=8)
        plt.title(col)
        plt.legend(title="Risk")
        st.pyplot(fig)


    # Kategorik Değişken Analizi
    st.markdown("<h3 style='text-align:left;'>- Kategorik Değişkenlere göre Hedef Değişken Analizi -</h3>",
                unsafe_allow_html=True)

    ######################################
    def cat_summary(dataframe, col_name, plot=False):
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        if plot:
            dataframe[col_name].hist()
            plt.xlabel(col_name)
            plt.title(col_name)
            plt.show(block=True)
        print('#################')

    for col in cat_cols:
        cat_summary(df, col, plot=True)
    #####################################################
    def plot_categorical_to_target(df, categorical_values, target):
        number_of_columns = 2
        number_of_rows = math.ceil(len(categorical_values) / 2)

        for index, column in enumerate(categorical_values, 1):
            fig, ax = plt.subplots(figsize=(12, 5 * number_of_rows))
            sns.countplot(x=column, data=df, hue=target, palette={"good": "g", "bad": "r"}, ax=ax)

            # Yüzde değerlerini ekle
            total = len(df)
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{height / total:.1%}',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom')

            ax.set_title(column)
            st.pyplot(fig)  # Grafiği Streamlit'te göster

    for col in cat_cols:
        plot_categorical_to_target(df, [col], "Risk")


    # Aykırı Değer

    #Age(-9.5, 82.5)
    #Credit amount(-6061.049999999998, 13188.549999999997)
    #Duration(-29.0, 75.0)

    st.markdown("<h3 style='text-align:left;'>- Aykırı Değerler-</h3>",
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns([5, 5, 5])
    col1.metric("Age:", -9.5, 82.5)
    col2.metric("Credit Amount:", -6061.049999999998, 13188.549999999997)
    col3.metric("Duration:", -29.0, 75.0)
    ################################################################################################
    def outlier_thresholds(dataframe, col_name):
        quartile1 = dataframe[col_name].quantile(0.15)
        quartile3 = dataframe[col_name].quantile(0.85)
        interquartile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquartile_range
        low_limit = quartile1 - 1.5 * interquartile_range
        return low_limit, up_limit

    for col in num_cols:
        print(col, outlier_thresholds(df, col))


    def check_outlier(dataframe, col_name):
        low_limit, up_limit = outlier_thresholds(dataframe, col_name)
        if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
            return True
        else:
            return False

    for col in num_cols:
        print(col, check_outlier(df, col), 0.05, 0.95)


    def grab_outliers(dataframe, col_name, index=False):
        low, up = outlier_thresholds(dataframe, col_name)
        if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
            print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
        else:
            print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])
        if index:
            outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
            return outlier_index
        for col in num_cols:
            print(col, grab_outliers(df, col))


    def remove_outlier(dataframe, col_name):
        low_limit, up_limit = outlier_thresholds(dataframe, col_name)
        df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
        return df_without_outliers

    for col in num_cols:
        print(col, remove_outlier(df, col))


    def replace_with_thresholds(dataframe, col_name):
        low_limit, up_limit = outlier_thresholds(dataframe, col_name)
        if low_limit > 0:
            dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
            dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
        else:
            dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit


    for col in num_cols:
        print(col, replace_with_thresholds(df, col))
    ################################################################################################
    # Age(-9.5, 82.5)
    # Credit amount(-6061.049999999998, 13188.549999999997)
    # Duration(-29.0, 75.0)

    st.markdown("<h3 style='text-align:left;'>- Outliers </h3>", unsafe_allow_html=True)
    color = "skyblue"
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.boxplot(x=df[col], ax=ax, color=color)
        ax.set_xlabel(None) # Grafiğin altında başlık olarak gözüken değişken ismini kaldırır
        ax.annotate(col, xy=(0.95, 0.95), xycoords='axes fraction',
                    fontsize=12, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5))

        st.pyplot(fig)

    st.markdown("<h3 style='text-align:left;'>- Describe </h3>", unsafe_allow_html=True)
    st.table(df.describe([0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))

    # Missing değerlerin doldurulması
    st.markdown("<h3 style='text-align:left;'>- Missing Values-</h3>",
                unsafe_allow_html=True)

    col1, col2 = st.columns([7.5, 7.5])
    col1.metric("Checking account:" ,394)
    col2.metric("Saving accounts:" ,183)

    def missing_values_table(dataframe, na_name=False):
        na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
        n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
        ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
        missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
        print(missing_df, end="\n")
        if na_name:
            return na_columns
        missing_values_table(df)

    df['Saving accounts'].fillna(df['Saving accounts'].mode()[0], inplace=True)
    df['Checking account'].fillna(df['Checking account'].mode()[0], inplace=True)

    # Korelasyon

    st.markdown("<h3 style='text-align:left;'>- Korelasyon Matrix</h3>", unsafe_allow_html=True)
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    corr = df.corr()
    #plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots(figsize=[18, 13])
    sns.heatmap(df.corr(), annot=True, ax=ax, fmt=".2f", cmap=cmap)

    plt.title("Korelasyon Isı Haritası")
    plt.show()
    st.pyplot(fig)


# TAHMİNLEME SAYFASI İÇERİĞİ

elif page == "Tahminleme":
    st.markdown("<h2 style='text-align:center;'> - TAHMİNLEME - </h2>", unsafe_allow_html=True)

    def grab_col_names(df, cat_th=10, car_th=20):
        cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
        num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]
        cat_but_car = [col for col in df.columns if
                       df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]
        num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
        num_cols = [col for col in num_cols if col not in cat_cols]

        print(f"Observations: {df.shape[0]}")
        print(f"Variables:{df.shape[1]}")
        print(f'cat_cols:{len(cat_cols)}')
        print(f'num_cols:{len(num_cols)}')
        print(f'cat_but_car:{len(cat_but_car)}')
        print(f'num_but_cat:{len(num_but_cat)}')
        return cat_cols, num_cols, cat_but_car
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    def num_summary(dataframe, numerical_col, plot=False):
        quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1]
        print(dataframe[numerical_col].describe(quantiles).T)
        if plot:
            dataframe[numerical_col].hist()
            plt.xlabel(numerical_col)
            plt.title(numerical_col)
            plt.show(block=True)

    for col in num_cols:
        num_summary(df, col, plot=True)

    num_cols = [col for col in num_cols if 'Unnamed: 0' not in col]
    cat_cols, num_cols, cat_but_car = grab_col_names(df)


    df = pd.read_csv("/Users/cansuuysal/Desktop/german_credit_data.csv", sep=';')
    df = df.drop(columns=["Unnamed: 0"])
    num_cols = [col for col in num_cols if 'Unnamed: 0' not in col]
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
    def label_encoder(dataframe, binary_col):
        labelencoder = LabelEncoder()
        dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
        return dataframe
    for col in binary_cols:
        df = label_encoder(df, col)

    def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
        dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
        return dataframe
    cat_cols = [col for col in cat_cols if col not in binary_cols and col not in "Risk"]
    df = one_hot_encoder(df, cat_cols, drop_first=True)
    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    st.markdown("<h3 style='text-align:left;'>Ön-İşlemeden Sonraki Genel Bilgiler</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([7.5, 7.5, 5])
    col1.metric("Verinin Şekli:", str(df.shape))
    col2.metric('Gözlem Sayısı:', df.shape[0])
    col3.metric("Değişken Sayısı:", df.shape[1])

    st.markdown("<h3 style='text-align:left;'>Değişken Bilgileri</h3>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([5, 5, 5, 5])
    col1.metric("Kategorik:", len(cat_cols))
    col2.metric('Numerik:', len(num_cols))
    col3.metric("Kategorik-Kardinal:", len(cat_but_car))

    ######CART######

    y = df["Risk"]
    X = df.drop(["Risk"], axis=1)
    cart_model = DecisionTreeClassifier(max_depth=10, random_state=1).fit(X, y)
    derinlik = cart_model.tree_.max_depth

    ###confusion matrix için y_pred
    y_pred = cart_model.predict(X)
    ###auc##
    y_prob = cart_model.predict_proba(X)[:, 1]
    # AUC
    roc_auc_score(y, y_prob)
    # confusion matrix
    print(classification_report(y, y_pred))

    st.markdown("<h3 style='text-align:left;'>Confusion Matrix</h3>", unsafe_allow_html=True)

    ##Holdout Yöntemi ile Başarı Değerlendirme
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

    cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)
    # train hatası
    y_pred = cart_model.predict(X_train)
    y_prob = cart_model.predict_proba(X_train)[:, 1]
    print(classification_report(y_train, y_pred))
    report = classification_report(y_train, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.table(report_df)
    roc_auc_score(y_train, y_prob)
    # test hatası
    y_pred = cart_model.predict(X_test)
    y_prob = cart_model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))
    roc_auc_score(y_test, y_prob)

    # CV ile Başarı Değerlendirme cross validation on katlı çapraz katlama
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_validate
    cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)
    cv_results = cross_validate(cart_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

    cv_results['test_accuracy'].mean()
    cv_results['test_f1'].mean()
    cv_results['test_roc_auc'].mean()


    def plot_importance(cart_model, X):
        feature_imp = pd.DataFrame({'Value': cart_model.feature_importances_, 'Feature': X.columns})
        plt.figure(figsize=(10, 15))
        sns.set(font_scale=1)
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                         ascending=False)[1:len(X)])
        plt.title('Feature Importance')
        plt.tight_layout()


    plot_importance(cart_model, X_test)
    st.pyplot(plt)

    ########################################################################################################
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    #########**********HOCANIN********#######
    def split_data(dataframe, test_size=0.20, random_state=45):
        y = dataframe["Risk"]
        X = dataframe.drop(["Risk"], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test


    def evaluate_model(X_train, y_train, X_test, y_test):
        cart_model = DecisionTreeClassifier().fit(X_train, y_train)
        y_pred = cart_model.predict(X_test)
        accuracy = round(accuracy_score(y_test, y_pred), 3)
        recall = round(recall_score(y_test, y_pred), 3)
        precision = round(precision_score(y_test, y_pred), 3)
        f1 = round(f1_score(y_test, y_pred), 3)
        auc = round(roc_auc_score(y_test, y_pred), 3)
        return cart_model, accuracy, recall, precision, f1, auc

        st.markdown("<h3 style='text-align:left;'>Model Bilgileri</h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([10, 5, 5])
        col1.metric("Kullanılan Model:", cart_model.__class__.__name__)
        col3.metric('Accuracy Skoru:', accuracy)

        col1, col2, col3, col4 = st.columns([5, 5, 5, 5])
        col1.metric("Recall Skoru:", recall)
        col2.metric("Precision Skoru:", precision)
        col3.metric('F1 Skoru:', f1)
        col4.metric("AUC Skoru:", auc)

        plot_importance(cart_model, X_test)
        st.pyplot(plt)

    st.markdown("<h2 style='text-align:left;'>Yeni Bir Kullanıcı İçin Tahmin Yapalım</h2>", unsafe_allow_html=True)

    # Inputları Oluşturma
    col1, col2, col3 = st.columns([5, 5, 5])
    sex_input = col1.selectbox("Sex", ["Female", "Male"])
    age_input = col2.number_input("Age", step=1)
    job_input= col3.selectbox("Job", ["unskilled-non resident", "unskilled-resident", "skilled", "highly skilled"])

    col1, col2, col3 = st.columns([5, 5, 5])
    housing_input = col1.selectbox("Housing", ["own", "rent", "free"])
    saving_input = col2.selectbox("Saving accounts", ["little", "moderate","quite rich","rich"])
    checking_input = col3.selectbox("Checking account",["little", "moderate", "rich"])

    col1, col2, col3 = st.columns([5, 5, 5])
    credit_amount_input = col1.number_input("Credit Amount", step=1)
    duration_input = col2.number_input("Duration", step=1)
    purpose_input = col3.selectbox("Purpose",
                                   ["car", "furniture/equipment", 'radio/TV', 'domestic appliances', 'repairs', 'education', 'business', 'vacation/others'])
    input_list = [sex_input, age_input, job_input, housing_input, saving_input, checking_input, credit_amount_input, duration_input, purpose_input]

    sample_df = pd.DataFrame()

    # Kategorik değişkenleri 1-0 formatına çevirme

    #Age
    sample_df.loc[0, "Age"] = age_input
    # sex
    if sex_input == "Female":
        sample_df.loc[0, "Sex"] = 0
    else:
        sample_df.loc[0, "Sex"] = 1
    #Credit Amount
    sample_df.loc[0, "Credit amount"] = credit_amount_input
    #duration
    sample_df.loc[0, "Duration"] = duration_input
    #  housing_input
    if housing_input == "rent":
        sample_df.loc[0, "Housing_own"] = 0
        sample_df.loc[0, "Housing_rent"] = 1
    elif housing_input == "own":
        sample_df.loc[0, "Housing_own"] = 1
        sample_df.loc[0, "Housing_rent"] = 0
    else:
        sample_df.loc[0, "Housing_own"] = 0
        sample_df.loc[0, "Housing_rent"] = 0
    #  housing_input
    if housing_input == "rent":
        sample_df.loc[0, "Housing_own"] = 0
        sample_df.loc[0, "Housing_rent"] = 1
    elif housing_input== "own":
        sample_df.loc[0, "Housing_own"] = 1
        sample_df.loc[0, "Housing_rent"] = 0
    else:
        sample_df.loc[0, "Housing_own"] = 0
        sample_df.loc[0, "Housing_rent"] = 0
    # saving_input
    if saving_input == "Moderate":
        sample_df.loc[0, "Saving accounts_moderate"] = 1
        sample_df.loc[0, "Saving accounts_quite rich"] = 0
        sample_df.loc[0, "Saving accounts_rich"] = 0
    if saving_input == "quite rich":
        sample_df.loc[0, "Saving accounts_moderate"] = 0
        sample_df.loc[0, "Saving accounts_quite rich"] = 1
        sample_df.loc[0, "Saving accounts_rich"] = 0
    if saving_input == "rich":
        sample_df.loc[0, "Saving accounts_moderate"] = 0
        sample_df.loc[0, "Saving accounts_quite rich"] = 0
        sample_df.loc[0, "Saving accounts_rich"] = 1
    else:
        sample_df.loc[0, "Saving accounts_moderate"] = 0
        sample_df.loc[0, "Saving accounts_quite rich"] = 0
        sample_df.loc[0, "Saving accounts_rich"] = 0
    # checking_input
    if checking_input == "Moderate":
        sample_df.loc[0, "Checking account_moderate"] = 1
        sample_df.loc[0, "Checking account_rich"] = 0
    if checking_input == "rich":
        sample_df.loc[0, "Checking account_moderate"] = 0
        sample_df.loc[0, "Checking account_rich"] = 1
    else:
        sample_df.loc[0, "Checking account_moderate"] = 0
        sample_df.loc[0, "Checking account_rich"] = 0
    # purpose
    if purpose_input == "car":
        sample_df.loc[0, "Purpose_car"] = 1
        sample_df.loc[0, "Purpose_domestic appliances"] = 0
        sample_df.loc[0, "Purpose_education"] = 0
        sample_df.loc[0, "Purpose_furniture/equipment"] = 0
        sample_df.loc[0, "Purpose_radio/TV"] = 0
        sample_df.loc[0, "Purpose_repairs"] = 0
        sample_df.loc[0, "Purpose_vacation/others"] = 0
    if purpose_input == "domestic appliances":
        sample_df.loc[0, "Purpose_car"] = 0
        sample_df.loc[0, "Purpose_domestic appliances"] = 1
        sample_df.loc[0, "Purpose_education"] = 0
        sample_df.loc[0, "Purpose_furniture/equipment"] = 0
        sample_df.loc[0, "Purpose_radio/TV"] = 0
        sample_df.loc[0, "Purpose_repairs"] = 0
        sample_df.loc[0, "Purpose_vacation/others"] = 0
    if purpose_input == "education":
        sample_df.loc[0, "Purpose_car"] = 0
        sample_df.loc[0, "Purpose_domestic appliances"] = 0
        sample_df.loc[0, "Purpose_education"] = 1
        sample_df.loc[0, "Purpose_furniture/equipment"] = 0
        sample_df.loc[0, "Purpose_radio/TV"] = 0
        sample_df.loc[0, "Purpose_repairs"] = 0
        sample_df.loc[0, "Purpose_vacation/others"] = 0
    if purpose_input == "furniture/equipment":
        sample_df.loc[0, "Purpose_car"] = 0
        sample_df.loc[0, "Purpose_domestic appliances"] = 0
        sample_df.loc[0, "Purpose_education"] = 0
        sample_df.loc[0, "Purpose_furniture/equipment"] = 1
        sample_df.loc[0, "Purpose_radio/TV"] = 0
        sample_df.loc[0, "Purpose_repairs"] = 0
        sample_df.loc[0, "Purpose_vacation/others"] = 0
    if purpose_input == "radio/TV":
        sample_df.loc[0, "Purpose_car"] = 0
        sample_df.loc[0, "Purpose_domestic appliances"] = 0
        sample_df.loc[0, "Purpose_education"] = 0
        sample_df.loc[0, "Purpose_furniture/equipment"] = 0
        sample_df.loc[0, "Purpose_radio/TV"] = 1
        sample_df.loc[0, "Purpose_repairs"] = 0
        sample_df.loc[0, "Purpose_vacation/others"] = 0
    if purpose_input == "repairs":
        sample_df.loc[0, "Purpose_car"] = 0
        sample_df.loc[0, "Purpose_domestic appliances"] = 0
        sample_df.loc[0, "Purpose_education"] = 0
        sample_df.loc[0, "Purpose_furniture/equipment"] = 0
        sample_df.loc[0, "Purpose_radio/TV"] = 0
        sample_df.loc[0, "Purpose_repairs"] = 1
        sample_df.loc[0, "Purpose_vacation/others"] = 0
    if purpose_input == "Purpose_vacation/others":
        sample_df.loc[0, "Purpose_car"] = 0
        sample_df.loc[0, "Purpose_domestic appliances"] = 0
        sample_df.loc[0, "Purpose_education"] = 0
        sample_df.loc[0, "Purpose_furniture/equipment"] = 0
        sample_df.loc[0, "Purpose_radio/TV"] = 0
        sample_df.loc[0, "Purpose_repairs"] = 0
        sample_df.loc[0, "Purpose_vacation/others"] = 1
    else:
        sample_df.loc[0, "Purpose_car"] = 0
        sample_df.loc[0, "Purpose_domestic appliances"] = 0
        sample_df.loc[0, "Purpose_education"] = 0
        sample_df.loc[0, "Purpose_furniture/equipment"] = 0
        sample_df.loc[0, "Purpose_radio/TV"] = 0
        sample_df.loc[0, "Purpose_repairs"] = 0
        sample_df.loc[0, "Purpose_vacation/others"] = 0
    # Job
    if job_input == "Unskilled-non resident":
        sample_df.loc[0, "Job_1"] = 1
        sample_df.loc[0, "Job_2"] = 0
        sample_df.loc[0, "Job_3"] = 0
    elif job_input == "Unskilled-resident":
        sample_df.loc[0, "Job_1"] = 0
        sample_df.loc[0, "Job_2"] = 1
        sample_df.loc[0, "Job_3"] = 0
    elif job_input == "Skilled":
        sample_df.loc[0, "Job_1"] = 0
        sample_df.loc[0, "Job_2"] = 0
        sample_df.loc[0, "Job_3"] = 1
    else:
        sample_df.loc[0, "Job_1"] = 0
        sample_df.loc[0, "Job_2"] = 0
        sample_df.loc[0, "Job_3"] = 0

    from sklearn.tree import DecisionTreeClassifier

    def forecast_sample(cart_model, sample):
        sample_pred = cart_model.predict(sample)
        return sample_pred

    Risk = forecast_sample(cart_model, sample_df)

    if Risk == 0:
        st.markdown("<h4 style='text-align:center;'>Bu müşterinin geri ödemede sorun yaşayacağı tahmin ediliyor</h4>", unsafe_allow_html=True)
    else:
        st.markdown("<h4 style='text-align:center;'>Bu müşterinin ödemelerini zamanında yapacağı tahmin ediliyor</h4>", unsafe_allow_html=True)
