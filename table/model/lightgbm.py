# 特徴
1.モデル訓練に掛かる時間が短い 
2.メモリ効率が高い
3.推測精度が高い 
  同じデータセットに対して他のブースティングの機械学習アルゴリズムと比較した場合、Leaf-Wiseのため推測精度が改善する傾向にあります。
  これはLeaf-Wiseの方がLevel-Wiseと比較して、より複雑な決定木となるため
4.過学習しやすい 
  Leaf-wiseは決定木が複雑になります。
  つまり、決定木の構造をハイパーパラメータで適切に調整しないと過学習となる可能性が高い
5.大規模なデータセットも訓練可能 
  XGBoostよりもLightGBMはモデル訓練時間が大幅に短い傾向
  またサポートベクターマシンなど一部の機械学習手法はその構造上、大規模データに適していない



# lightgbmによる多値分類
import lightgbm as lgb
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    # Iris データセットを読み込む
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # データセットを学習用とテスト用に分割する
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # テスト用のデータを評価用と検証用に分ける
    X_eval, X_valid, y_eval, y_valid = train_test_split(X_test, y_test, random_state=42)

    # データセットを生成する
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)

    # ハイパーパラメータ
    lgbm_params = {
        # 多値分類問題
        'objective': 'multiclass',
        # クラス数は 3
        'num_class': 3,
    }

    # 交差検証の結果を確認する
    lgb.cv(lgbm_params,
           lgb_train,
           num_boost_round=1000,
           early_stopping_rounds=10,
           nfold=5,
           shuffle=True,
           stratified=True,
           seed=42,
           callbacks=callbacks,
           )

    # 上記のパラメータでモデルを学習する
    model = lgb.train(lgbm_params, lgb_train,
                      # モデルの評価用データを渡す
                      valid_sets=lgb_eval,
                      # 最大で 1000 ラウンドまで学習する
                      num_boost_round=1000,
                      # 10 ラウンド経過しても性能が向上しないときは学習を打ち切る
                      early_stopping_rounds=10)

    # 学習したモデルでホールドアウト検証する
    y_pred_proba = model.predict(X_valid, num_iteration=model.best_iteration)
    # 返り値は確率になっているので最尤に寄せる
    y_pred = np.argmax(y_pred_proba, axis=1)

    # 精度 (Accuracy) を計算する
    accuracy = accuracy_score(y_valid, y_pred)
    print(accuracy)

    # 各特徴量がどれくらい予測に寄与したのか数値で確認
    # 特徴量の重要度をプロットする
    lgb.plot_importance(model, figsize=(12, 6))
    plt.show()

if __name__ == '__main__':
    main()


# kfold_lightgbm
def kfold_lightgbm_sklearn(data, categorical_feature = None):
    df = data[data['TARGET'].notnull()]
    test = data[data['TARGET'].isnull()]
    print("Train/valid shape: {}, test shape: {}".format(df.shape, test.shape))
    del_features = ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index', 'level_0']
    predictors = list(filter(lambda v: v not in del_features, df.columns))

    if not STRATIFIED_KFOLD:
        folds = KFold(n_splits= NUM_FOLDS, shuffle=True, random_state= RANDOM_SEED)
    else:
        folds = StratifiedKFold(n_splits= NUM_FOLDS, shuffle=True, random_state= RANDOM_SEED)

    # Hold oof predictions, test predictions, feature importance and training/valid auc
    oof_preds = np.zeros(df.shape[0])
    sub_preds = np.zeros(test.shape[0])
    importance_df = pd.DataFrame()
    eval_results = dict()

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df[predictors], df['TARGET'])):
        train_x, train_y = df[predictors].iloc[train_idx], df['TARGET'].iloc[train_idx]
        valid_x, valid_y = df[predictors].iloc[valid_idx], df['TARGET'].iloc[valid_idx]

        params = {'random_state': RANDOM_SEED, 'nthread': NUM_THREADS}
        clf = LGBMClassifier(**{**params, **LIGHTGBM_PARAMS})
        if not categorical_feature:
            clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                    eval_metric='auc', verbose=400, early_stopping_rounds= EARLY_STOPPING)
        else:
            clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                    eval_metric='auc', verbose=400, early_stopping_rounds=EARLY_STOPPING,
                    feature_name= list(df[predictors].columns), categorical_feature= categorical_feature)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test[predictors], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        # Feature importance by GAIN and SPLIT
        fold_importance = pd.DataFrame()
        fold_importance["feature"] = predictors
        fold_importance["gain"] = clf.booster_.feature_importance(importance_type='gain')
        fold_importance["split"] = clf.booster_.feature_importance(importance_type='split')
        importance_df = pd.concat([importance_df, fold_importance], axis=0)
        eval_results['train_{}'.format(n_fold+1)]  = clf.evals_result_['training']['auc']
        eval_results['valid_{}'.format(n_fold+1)] = clf.evals_result_['valid_1']['auc']

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(df['TARGET'], oof_preds))
    test['TARGET'] = sub_preds.copy()

    # Get the average feature importance between folds
    mean_importance = importance_df.groupby('feature').mean().reset_index()
    mean_importance.sort_values(by= 'gain', ascending=False, inplace=True)
    # Save feature importance, test predictions and oof predictions as csv
    if GENERATE_SUBMISSION_FILES:

        # Generate oof csv
        oof = pd.DataFrame()
        oof['SK_ID_CURR'] = df['SK_ID_CURR'].copy()
        df['PREDICTIONS'] = oof_preds.copy()
        df['TARGET'] = df['TARGET'].copy()
        df.to_csv('oof{}.csv'.format(SUBMISSION_SUFIX), index=False)
        # Save submission (test data) and feature importance
        test[['SK_ID_CURR', 'TARGET']].to_csv('submission{}.csv'.format(SUBMISSION_SUFIX), index=False)
        mean_importance.to_csv('feature_importance{}.csv'.format(SUBMISSION_SUFIX), index=False)
    return mean_importance