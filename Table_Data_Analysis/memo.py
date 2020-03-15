# 複数のカラムに一度で処理を行う、apply
# https://qiita.com/Hiroyuki1993/items/ab4ff4cfd378dca0f099
def some_function1(x):
    return pd.Series([x*2, x/2])

data[['new_column1', 'new_columns2']] = data['existing_column'].apply(some_function)