import pandas as pd
data = pd.read_csv("train.csv")

data.isnull()
data.isnull().sum()

df = df.fillna(0)

# null値のある行を全削除
df.dropna()

def missing_values_table(df):
  # Total missing values
  mis_val = df.isnull().sum()
  
  # Percentage of missing values
  mis_val_percent = 100 * df.isnull().sum() / len(df)
  
  # Make a table with the results
  mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
  
  # Rename the columns
  mis_val_table_ren_columns = mis_val_table.rename(
  columns = {0 : 'Missing Values', 1 : '% of Total Values'})
  
  # Sort the table by percentage of missing descending
  mis_val_table_ren_columns = mis_val_table_ren_columns[
      mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
  '% of Total Values', ascending=False).round(1)
  
  # Print some summary information
  print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
      "There are " + str(mis_val_table_ren_columns.shape[0]) +
        " columns that have missing values.")
  
  # Return the dataframe with missing information
  return mis_val_table_ren_columns

# 欠損している割合の低いものを表示
# underlineは制限〇〇％以下
def missing_few_values_table(df, underline):
  # 欠損全体
  mis_val = df.isnull().sum()
  
  # 欠損割合
  mis_val_percent = 100 * df.isnull().sum() / len(df)
  
  # テーブル作成
  mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
  
  # カラム名変更
  mis_val_table_ren_columns = mis_val_table.rename(
  columns = {0 : 'Missing Values', 1 : '% of Total Values'})
  
  # 欠損のみに限定（欠損数ゼロを排除）
  mis_val_table_ren_columns = mis_val_table_ren_columns[
      mis_val_table_ren_columns.iloc[:,1] != 0]
  
  # 下限underline割合以下に限定
  mis_few_val_table_ren_columns = mis_val_table_ren_columns[
      mis_val_table_ren_columns.iloc[:,1] <= underline].sort_values(
  '% of Total Values', ascending=False).round(1)
  
  # info表示
  print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
      "There are " + str(mis_val_table_ren_columns.shape[0]) +
        " columns that have missing values less than " + str(underline) + "%" )
  
  # テーブル表示
  return mis_few_val_table_ren_columns