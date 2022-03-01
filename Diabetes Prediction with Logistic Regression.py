import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

data = pd.read_csv("C:/Users/90537/Desktop/DSMLBC6/7.hafta/diabetes.csv")
df = data.copy()

def check_df(dataframe: object, head: object = 5) -> object:
    print("########################### Shape ###########################")
    print(dataframe.shape)

    print("########################### Types ###########################")
    print(dataframe.dtypes)

    print("########################### Head ###########################")
    print(dataframe.head(head))

    print("########################### Tail ###########################")
    print(dataframe.tail(head))

    print("########################### NA ###########################")
    print(dataframe.isnull().sum())

    print("########################### Quantiles ###########################")
    print(dataframe.quantile([0, 0.05,0.1, 0.25, 0.50,0.75, 0.90, 0.95, 0.99, 1]).T)

    check_df(df)

    df.groupby(['Outcome']).agg({"Age": ["mean", "median"],
                                 "Glucose": ["mean", "median"],
                                 "Pregnancies": ["mean", "median"],
                                 "BMI": ["mean", "median"],
                                 "SkinThickness": ["mean", "median"]})

    def grab_col_names(dataframe, cat_th=10, car_th=20):
        # cat_cols, cat_but_car
        cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
        num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                       dataframe[col].dtypes != "O"]
        cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                       dataframe[col].dtypes == "O"]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        # num_cols
        num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]

        print(f"Observations: {dataframe.shape[0]}")
        print(f"Variables: {dataframe.shape[1]}")
        print(f'cat_cols: {len(cat_cols)}')
        print(f'num_cols: {len(num_cols)}')
        print(f'cat_but_car: {len(cat_but_car)}')
        print(f'num_but_cat: {len(num_but_cat)}')

        return cat_cols, num_cols, cat_but_car

    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    def cat_summary(dataframe, col_name, plot=False):
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

        if plot:
            plt.style.use('seaborn-darkgrid')
            fig, ax = plt.subplots(1, 2)
            ax = np.reshape(ax, (1, 2))
            ax[0, 0] = sns.countplot(x=dataframe[col_name], color="green", ax=ax[0, 0])
            ax[0, 0].set_ylabel('Count')
            ax[0, 0].set_xticklabels(ax[0, 0].get_xticklabels(), rotation=-45)
            ax[0, 1] = plt.pie(dataframe[col_name].value_counts().values,
                               labels=dataframe[col_name].value_counts().keys(),
                               colors=sns.color_palette('bright'), shadow=True, autopct='%.0f%%')
            plt.title("Percent")
            fig.set_size_inches(10, 6)
            fig.suptitle('Analysis of Categorical Variables', fontsize=13)
            plt.show()

        for col in cat_cols:
            cat_summary(df, col, plot=True)

    def num_summary(dataframe, numerical_col):
        # setup the plot grid
        plt.style.use('seaborn-darkgrid')
        fig, ax = plt.subplots(1, 2)
        ax = np.reshape(ax, (1, 2))
        ax[0, 0] = sns.histplot(x=dataframe[numerical_col], color="green", bins=20, ax=ax[0, 0])
        ax[0, 0].set_ylabel('Count')
        ax[0, 0].set_title('Distribution')
        ax[0, 1] = sns.boxplot(y=dataframe[numerical_col], color="purple", ax=ax[0, 1])
        ax[0, 1].set_title('Quantiles')

        fig.set_size_inches(12, 8)
        fig.suptitle('Analysis of Numerical Variables', fontsize=13)
        plt.show()

    for col in df[num_cols]:
        num_summary(df, col)

    def correlated_map(dataframe, plot=False):
        corr = dataframe.corr()
        if plot:
            sns.set(rc={'figure.figsize': (10, 10)})
            sns.heatmap(corr, cmap="YlGnBu", annot=True, linewidths=.7)
            plt.xticks(rotation=60, size=10)
            plt.yticks(size=10)
            plt.title('Correlation Map', size=20)
            plt.show()

    correlated_map(df, plot=True)


    def missing_values_table(dataframe, na_name=False):
        na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
        n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
        ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
        missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
        print(missing_df, end="\n")
        if na_name:
            return na_columns

    missing_values_table(df, na_name=True)

    def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
        quartile1 = dataframe[col_name].quantile(q1)
        quartile3 = dataframe[col_name].quantile(q3)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        return low_limit, up_limit

    for col in num_cols:
        print(outlier_thresholds(df, col))

    def replace_with_thresholds(dataframe, variable):
        low_limit, up_limit = outlier_thresholds(dataframe, variable)
        dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

    for col in num_cols:
        replace_with_thresholds(df, col)

####    Feature Extraction

    bmi_labels = ["underweight", "normal_weight", "overweight", "obesity_class_1", "obesity_class_2", "obesity_class_3"]
    df['BMI_Cat'] = pd.cut(df['BMI'], [-1, 18.5, 25, 30, 35, 40, df['BMI'].max()],
                           labels=bmi_labels)
    cat_summary(df, "BMI_Cat", plot=True)

    df["New_Age_Cat"] = df["Age"].apply(
        lambda x: "youngmale" if x < 30 else ("maturemale" if 30 <= x <= 50 else "seniormale"))

    cat_summary(df, "New_Age_Cat", plot=True)

    df["Glucose_Cat"] = df["Glucose"].apply(lambda x: "Normal" if x < 140 else ("IGT" if 140 <= x <= 200 else "DM"))

    cat_summary(df, "Glucose_Cat", plot=True)

    bp_labels = ["optimal", "normal", "high_normal", "grade_1_hypertension", "grade_2_hypertension",
                 "grade_3_hypertension"]
    df['Blood_Pressure_Cat'] = pd.cut(df['BloodPressure'], [-1, 80, 85, 90, 100, 110, df['BloodPressure'].max()],
                                      labels=bp_labels)
    cat_summary(df, "Blood_Pressure_Cat", plot=True)

### Encoding ####
    def label_encoder(dataframe, binary_col):
        labelencoder = LabelEncoder()
        dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
        return dataframe

    binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]
    len(binary_cols)

    for col in binary_cols:
        label_encoder(df, col)

    df.head()

    def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
        dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
        return dataframe

    ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
    df = one_hot_encoder(df, ohe_cols, drop_first=True)

    df.head()

    cat_cols, num_cols, cat_but_car = grab_col_names(df)

#### Scaling #####

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    check_df(df)

#### Model####

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)

#### Model Validation ####

cv_results = cross_validate(log_model,
                            X, y,
                            cv=10,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

#### Predict####
y_pred = log_model.predict(X)

#### Model Evaluation ####
# Confusion Matrix
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))

#### ROC AUC####
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)


# Prediction for A New Observation

X.columns

random_user = X.sample(1, random_state=42)

log_model.predict(random_user)

