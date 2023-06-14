import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBClassifier
import numpy as np
import optuna

# 初期設定
lr = LogisticRegression()
dt = DecisionTreeClassifier()
mlp = MLPClassifier()
rf = RandomForestClassifier()
lgbm = lgb.LGBMClassifier()
xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

st.title('機械学習手法の比較')

st.sidebar.header('機械学習手法')

ml_menu = st.sidebar.selectbox(
    "機械学習手法を選択してください",
    ('すべて', 'ロジスティック回帰', '決定木', 'ニューラルネットワーク', 'ランダムフォレスト', 'LightGBM', 'XGBoost'))

all_models = st.sidebar.checkbox("すべての機械学習手法で実施する")

if all_models:
    best_only = st.sidebar.checkbox("最もAUCが高い手法の結果のみを出力する")

# データセットのアップロード
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロードしてください", type="csv")

# データセットの読み込み
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    # 説明変数と目的変数の選択
    ex = st.multiselect('説明変数を選択してください', df.columns)
    ob = st.selectbox('目的変数を選択してください', df.columns)

    # 実行ボタン
    execute = st.sidebar.button("実行")

    if execute:
        df_ex = df[ex]
        df_ob = df[ob]

        X_train, X_test, y_train, y_test = train_test_split(
            df_ex.values, df_ob.values, stratify=df_ob , test_size=0.3, random_state=0)
        sc = StandardScaler()
        sc.fit(X_train)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)

        models = {
            'ロジスティック回帰': lr,
            '決定木': dt,
            'ニューラルネットワーク': mlp,
            'ランダムフォレスト': rf,
            'LightGBM': lgbm,
            'XGBoost': xgb
    }

        def objective(trial):
            if ml_menu == 'ロジスティック回帰':
                C = trial.suggest_loguniform('C', 1e-10, 1e10)
                model = LogisticRegression(C=C)
            elif ml_menu == '決定木':
                max_depth = trial.suggest_int('max_depth', 1, 30)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
            elif ml_menu == 'ニューラルネットワーク':
                alpha = trial.suggest_loguniform('alpha', 1e-10, 1e0)
                hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50, 50), (100, 50)])
                model = MLPClassifier(alpha=alpha, hidden_layer_sizes=hidden_layer_sizes)
            elif ml_menu == 'ランダムフォレスト':
                n_estimators = trial.suggest_int('n_estimators', 10, 1000)
                max_depth = trial.suggest_int('max_depth', 1, 30)
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
            elif ml_menu == 'LightGBM':
                num_leaves = trial.suggest_int('num_leaves', 2, 100)
                max_depth = trial.suggest_int('max_depth', 1, 30)
                learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1)
                model = lgb.LGBMClassifier(num_leaves=num_leaves, max_depth=max_depth, learning_rate=learning_rate)
            else:
                max_depth = trial.suggest_int('max_depth', 1, 30)
                learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1)
                model = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, use_label_encoder=False, eval_metric='logloss')

            model.fit(X_train, y_train)
            Y_score = model.predict_proba(X_test)[:, 1]
            fpr_test, tpr_test, thresholds_test = roc_curve(y_true=y_test, y_score=Y_score)
            return 1.0 - auc(fpr_test, tpr_test)

        study = optuna.create_study()
        progress_bar = st.progress(0)
        for i in range(50):
            study.optimize(objective, n_trials=1)
            progress_bar.progress((i + 1) / 50)

        best_params = study.best_params
        best_model = models[ml_menu].set_params(**best_params)
        best_model.fit(X_train, y_train)

        # 以降のコードは変更なし


    if all_models:
        best_auc = 0
        best_name = None

        for name, model in models.items():
            Y_score = model.predict_proba(X_test)[:, 1]
            fpr_test, tpr_test, thresholds_test = roc_curve(y_true=y_test, y_score=Y_score)
            auc_score = auc(fpr_test, tpr_test)

            if not best_only:
                st.write(f"{name} のテストAUC: {auc_score:.3f}")

            if auc_score > best_auc:
                best_auc = auc_score
                best_model = model
                best_name = name

        st.write(f"最も良いモデル: {best_name} (AUC: {best_auc:.3f})")

    else:
        Y_score = best_model.predict_proba(X_test)[:, 1]
        fpr_test, tpr_test, thresholds_test = roc_curve(y_true=y_test, y_score=Y_score)
        auc_score = auc(fpr_test, tpr_test)
        st.write(f"{ml_menu} のテストAUC: {auc_score:.3f}")

    if best_name in ['ロジスティック回帰', 'ランダムフォレスト', 'LightGBM', 'XGBoost']:
        if best_name == 'ロジスティック回帰':
            feature_importances = np.abs(best_model.coef_[0])
        else:
            feature_importances = best_model.feature_importances_

        importance_df = pd.DataFrame({'feature': ex, 'importance': feature_importances})
        importance_df = importance_df.sort_values('importance', ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importance_df, ax=ax)
        plt.title('Feature Importance')
        st.pyplot(fig)

