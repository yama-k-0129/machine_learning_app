import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler
import optuna
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier



# Title
st.title('機械学習アプリ')
st.write('streamlitで実装')

# サイドバーに表示
st.sidebar.markdown("### 機械学習に用いるcsvファイルを入力してください")
# ファイルアップロード
uploaded_files = st.sidebar.file_uploader("Choose a CSV file", accept_multiple_files=False)
# ファイルアップロード時実行
if uploaded_files:
    df = pd.read_csv(uploaded_files)
    df_columns = df.columns
    # データの表示
    st.markdown("### 入力データ")
    st.dataframe(df)
    # matplotlibで可視化。X軸Y軸選択
    st.markdown("### 可視化 単変量")
    # データフレームのカラムを選択
    x = st.selectbox('X軸', df_columns)
    y = st.selectbox('Y軸', df_columns)
    fig = plt.figure(figsize=(12, 8))
    plt.scatter(df[x], df[y])
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=18)
    execute_simplot = st.button("単変量プロット描画")
    if execute_simplot:
        st.pyplot(fig)

    st.markdown('### 可視化 ペアプロット')
    item = st.multiselect('可視化するカラム', df_columns)
    hue = st.selectbox("色の基準", df_columns)  # 散布図の色分け基準を一つ選択

    execute_pairplot = st.button("ペアプロット描画")  # 実行ボタン
    if execute_pairplot:
        df_sns = df[item]
        df_sns['hue'] = df[hue]

        fig = sns.pairplot(df_sns, hue='hue')
        st.pyplot(fig)  # 実行ボタンを押すとstreamlit上でseabornのペアプロットを表示

    st.markdown('### モデリング')
    ex = st.multiselect("説明変数を選択してください（複数選択可）", df_columns)
    ob = st.selectbox('目的変数を選択してください', df_columns)
    if st.checkbox('undersampling'):
        df_ex = df[ex]
        df_ob = df[ob]
        positive_count_train = df_ob.value_counts()[1]
        strategy = {0: positive_count_train * 9, 1: positive_count_train}
        rus = RandomUnderSampler(random_state=0, sampling_strategy=strategy)# type: ignore
        df_ex, df_ob = rus.fit_resample(df_ex, df_ob)# type: ignore
    else:
        df_ex = df[ex]
        df_ob = df[ob]

    ml_menu = st.sidebar.selectbox(
        '機械学習アルゴリズムを選択してください',
        ('ロジスティック回帰', '決定木', 'ニューラルネットワーク', 'ランダムフォレスト', 'LightGBM', 'XGBoost', "SVM", "k-NN","Naive Bayes","Ensemble Methods","Deep Learning Models")
    )
    X_train = df_ex
    y_train = df_ob
    
 

    # Objective function for Logistic Regression hyperparameter tuning
    def objective_lr(trial):
        # パラメータの候補を生成
        C = trial.suggest_loguniform("C", 1e-5, 1e2)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
        solver = "saga" if penalty == "elasticnet" else "liblinear"
        l1_ratio = trial.suggest_uniform("l1_ratio", 0, 1) if penalty == "elasticnet" else None

        # モデルを作成
        lr = LogisticRegression(C=C, penalty=penalty, l1_ratio=l1_ratio, solver=solver, max_iter=1000)

        # 交差検証で評価
        scores = cross_val_score(lr, X_train, y_train, cv=5, scoring="roc_auc")

        return np.mean(scores)

    # Objective function for Decision Tree hyperparameter tuning
    def objective_dt(trial):
        # パラメータの候補を生成
        max_depth = trial.suggest_int("max_depth", 1, 30)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
        max_features = trial.suggest_categorical("max_features", ["auto", "sqrt", "log2", None])
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

        # モデルを作成
        dt = DecisionTreeClassifier(
            max_depth=max_depth, 
            min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf, 
            max_features=max_features, 
            criterion=criterion
        )

        # 交差検証で評価
        scores = cross_val_score(dt, X_train, y_train, cv=5, scoring="roc_auc")

        return np.mean(scores)

    # Objective function for Neural Network hyperparameter tuning
    # ニューラルネットワークのモデルを定義
    def create_model(optimizer='adam', dropout_rate=0.1, init_mode='uniform', activation='relu'):
        model = Sequential()
        model.add(Dense(units=trial.suggest_int("units", 10, 50),
                        activation=activation,
                        kernel_initializer=init_mode,
                        input_shape=(X_train.shape[1],)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid', kernel_initializer=init_mode))
        model.compile(optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        return model

    # Optunaでのハイパーパラメータチューニングの目的関数
    def objective_nn(trial):
        # チューニング対象のパラメータ
        optimizer_options = ['RMSprop', 'Adam', 'SGD']
        dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
        init_mode = trial.suggest_categorical('init_mode', ['uniform', 'normal', 'zero'])
        activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])
        optimizer = trial.suggest_categorical("optimizer", optimizer_options)

        model = KerasClassifier(build_fn=create_model, 
                                epochs=50, 
                                batch_size=trial.suggest_categorical("batch_size", [10, 50, 100]),
                                verbose=0,
                                optimizer=optimizer,
                                dropout_rate=dropout_rate,
                                init_mode=init_mode,
                                activation=activation)
        score = cross_val_score(model, X_train, y_train, cv=3)
        return score.mean()
    # Objective function for Random Forest hyperparameter tuning


    def objective_rf(trial):
        # パラメータの候補を生成
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 10, 100)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 15)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        max_features = trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"])
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])

        # モデルを作成
        rf = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf, 
            max_features=max_features,
            bootstrap=bootstrap
        )

        # 交差検証で評価
        scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="roc_auc")

        return np.mean(scores)

    # Objective function for LightGBM hyperparameter tuning
    def objective_lgb(trial):
        # パラメータの候補を生成
        num_leaves = trial.suggest_int("num_leaves", 2, 256)
        min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 10, 100)
        max_depth = trial.suggest_int("max_depth", 5, 30)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-8, 1.0)
        n_estimators = trial.suggest_int("n_estimators", 10, 500)
        subsample = trial.suggest_uniform("subsample", 0.1, 1.0)
        colsample_bytree = trial.suggest_uniform("colsample_bytree", 0.1, 1.0)
        reg_lambda = trial.suggest_loguniform("reg_lambda", 1e-8, 10.0)
        reg_alpha = trial.suggest_loguniform("reg_alpha", 1e-8, 10.0)
        scale_pos_weight = trial.suggest_uniform("scale_pos_weight", 0.1, 1.0)

        # モデルを作成
        lgbm = lgb.LGBMClassifier(
            num_leaves=num_leaves, 
            min_data_in_leaf=min_data_in_leaf,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            scale_pos_weight=scale_pos_weight
        )

        # 交差検証で評価
        scores = cross_val_score(lgbm, X_train, y_train, cv=5, scoring="roc_auc")

        return np.mean(scores)

    def objective_xgb(trial):
        # パラメータの候補を生成
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-8, 1.0)
        max_depth = trial.suggest_int("max_depth", 1, 30)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
        gamma = trial.suggest_loguniform("gamma", 1e-8, 1.0)
        subsample = trial.suggest_uniform("subsample", 0.1, 1.0)
        colsample_bytree = trial.suggest_uniform("colsample_bytree", 0.1, 1.0)
        reg_lambda = trial.suggest_loguniform("reg_lambda", 1e-8, 10.0)
        reg_alpha = trial.suggest_loguniform("reg_alpha", 1e-8, 10.0)
        scale_pos_weight = trial.suggest_uniform("scale_pos_weight", 0.1, 1.0)

        # モデルを作成
        xgbm = xgb.XGBClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            gamma=gamma,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            scale_pos_weight=scale_pos_weight
        )

        # 交差検証で評価
        scores = cross_val_score(xgbm, X_train, y_train, cv=5, scoring="roc_auc")

        return np.mean(scores)

    def objective_svm(trial):
        C = trial.suggest_loguniform('C', 1e-5, 1e2)
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        if kernel == 'poly':
            degree = trial.suggest_int('degree', 1, 5)
        else:
            degree = 1
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        
        svm = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
        scores = cross_val_score(svm, X_train, y_train, cv=5, scoring="roc_auc")
        return np.mean(scores)

    def objective_knn(trial):
        n_neighbors = trial.suggest_int('n_neighbors', 1, 30)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])

        knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring="roc_auc")
        return np.mean(scores)

    def objective_nb(trial):
        var_smoothing = trial.suggest_loguniform('var_smoothing', 1e-10, 1e-2)

        nb = GaussianNB(var_smoothing=var_smoothing)
        scores = cross_val_score(nb, X_train, y_train, cv=5, scoring="roc_auc")
        return np.mean(scores)

    def objective_adaboost(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 1)

        adaboost = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        scores = cross_val_score(adaboost, X_train, y_train, cv=5, scoring="roc_auc")
        return np.mean(scores)

    def create_model(trial):
        n_layers = trial.suggest_int('n_layers', 1, 3)
        model = tf.keras.models.Sequential()
        for i in range(n_layers):
            n_units = trial.suggest_int('n_units_l{}'.format(i), 4, 128)
            model.add(tf.keras.layers.Dense(n_units, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=lr), metrics=['accuracy'])
        return model

    def objective_dl(trial):
        batch_size = trial.suggest_int('batch_size', 32, 256)
        epochs = trial.suggest_int('epochs', 10, 100)
        model = KerasClassifier(build_fn=create_model(trial), batch_size=batch_size, epochs=epochs, verbose=0)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
        return np.mean(scores)
    
    if ml_menu == 'ロジスティック回帰':
        objective = objective_lr
    elif ml_menu == '決定木':
        objective = objective_dt
    elif ml_menu == "ニューラルネットワーク":
        objective = objective_nn
    elif ml_menu == 'ランダムフォレスト':
        objective = objective_rf
    elif ml_menu == 'LightGBM':
        objective = objective_lgb
    elif ml_menu == 'XGBoost':
        objective = objective_xgb
    elif ml_menu == "SVM":
        objective = objective_svm
    elif ml_menu == "k-NN":
        objective = objective_knn
    elif ml_menu == "Naive Bayes":
        objective = objective_nb
    elif ml_menu == "Ensemble Methods":
        objective = objective_adaboost
    else:
        objective = objective_dl
    
    
    execute = st.sidebar.button("実行")

    if execute:    
        study = optuna.create_study(direction = 'maximize')
        progress_bar = st.progress(0)
        progress_text = st.empty()
        for i in range(100):
            study.optimize(objective, n_trials=1)
            progress = (i + 1) / 100
            progress_bar.progress(progress)
            progress_text.text(f"Progress: {int(progress * 100)}%")
        
        # 最適なパラメータを表示
        st.write(f"最も良いパラメーター: {study.best_trial.params}")
        st.write(f"最も良いパラメーター: {study.best_trial.value:.3f}")
       
    

