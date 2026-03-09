"""
本地预计算脚本：运行 ML 流程，将结果保存到 data.json
在部署到 Render 之前本地执行一次即可。
用法: python precompute.py
"""
import sys, os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(BASE_DIR, "专业录取分数统计表.csv")
OUTPUT     = os.path.join(BASE_DIR, "data.json")

print("正在读取CSV...")
df_raw = pd.read_csv(CSV_PATH, quotechar='"', encoding='utf-8')
df = df_raw.copy()

expected_columns = ['ZYMC','SZD','SFSYL','SF985','SF211',
    'JHRS_3','JHRS_2','JHRS_1','SJRS_3','SJRS_2','SJRS_1',
    'XX_Med_3','XX_Med_2','XX_Med_1',
    'ZY_Max_3','ZY_Max_2','ZY_Max_1',
    'ZY_Min_3','ZY_Min_2','ZY_Min_1','CS','NX','YJ']
for col in expected_columns:
    if col not in df.columns: df[col] = np.nan

numeric_cols = [c for c in expected_columns if c not in ('ZYMC','SZD','SFSYL','SF985','SF211')]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df[numeric_cols] = SimpleImputer(strategy='median').fit_transform(df[numeric_cols])

# 特征工程
df['ZY_Min_avg']        = df[['ZY_Min_1','ZY_Min_2','ZY_Min_3']].mean(axis=1)
df['ZY_Min_trend']      = (df['ZY_Min_3']-df['ZY_Min_1'])/(df['ZY_Min_1']+1e-6)
df['ZY_Min_volatility'] = df[['ZY_Min_1','ZY_Min_2','ZY_Min_3']].std(axis=1)/(df['ZY_Min_avg']+1e-6)
df['JHRS_avg']          = df[['JHRS_1','JHRS_2','JHRS_3']].mean(axis=1)
df['JHRS_trend']        = (df['JHRS_3']-df['JHRS_1'])/(df['JHRS_1']+1e-6)
for i in (1,2,3):
    df[f'LR_{i}']          = df[f'SJRS_{i}']/(df[f'JHRS_{i}']+1e-6)
    df[f'Competition_{i}'] = df[f'XX_Med_{i}']/(df[f'ZY_Min_{i}']+1e-6)
df['LR_avg']          = df[['LR_1','LR_2','LR_3']].mean(axis=1)
df['Competition_avg'] = df[['Competition_1','Competition_2','Competition_3']].mean(axis=1)
for c in ('SFSYL','SF985','SF211'):
    df[c] = df[c].fillna(0).astype(int)

feature_cols = [
    'JHRS_1','JHRS_2','JHRS_3','SJRS_1','SJRS_2','SJRS_3',
    'XX_Med_1','XX_Med_2','XX_Med_3','ZY_Max_1','ZY_Max_2','ZY_Max_3',
    'ZY_Min_1','ZY_Min_2','ZY_Min_3','CS','NX','YJ',
    'ZY_Min_avg','ZY_Min_trend','ZY_Min_volatility',
    'JHRS_avg','JHRS_trend','LR_avg','Competition_avg',
    'SFSYL','SF985','SF211'
]

print("LassoCV 特征选择...")
lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso.fit(df[feature_cols], df['ZY_Min_3'])
important = [feature_cols[i] for i in range(len(feature_cols)) if lasso.coef_[i] != 0]
print(f"  选定特征: {important}")

print("XGBoost 训练...")
X_tr, X_te, y_tr, y_te = train_test_split(df[important], df['ZY_Min_3'], test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                     subsample=0.8, colsample_bytree=0.8, random_state=42)
model.fit(X_tr, y_tr)

print("生成预测排名...")
df['ZY_Min_4'] = model.predict(df[important])
df['ZY_Min_4'] = np.maximum(df['ZY_Min_4'], 0)
df['ZY_Min_4'] = (0.4*df['ZY_Min_3'] + 0.3*df['ZY_Min_2'] +
                  0.2*df['ZY_Min_1'] + 0.1*df['ZY_Min_4'])

records = []
for idx, row in df.iterrows():
    min_r = int(round(row.get('ZY_Min_4', 0)))
    max_r = int(round(row.get('ZY_Max_3', 0)))
    med_r = int(round(row.get('XX_Med_3', 0)))
    vals = sorted([min_r, max_r, med_r], reverse=True)
    records.append({
        "id":        int(idx),
        "region":    str(row['SZD']),
        "major":     str(row['ZYMC']),
        "min_rank":  vals[0],
        "max_rank":  vals[2],
        "median_rank": vals[1],
        "SFSYL":     int(row.get('SFSYL', 0)),
        "SF985":     int(row.get('SF985', 0)),
        "SF211":     int(row.get('SF211', 0)),
    })

with open(OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(records, f, ensure_ascii=False)

print(f"\n✅ 预计算完成！共 {len(records)} 条记录 → {OUTPUT}")
