"""
本地预计算脚本：运行 ML 流程，将结果保存到 data.json
在部署到 Render 之前本地执行一次即可。
用法: python precompute.py
"""
import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(BASE_DIR, "专业录取分数统计表.csv")
OUTPUT     = os.path.join(BASE_DIR, "data.json")

class RankForecastPipeline:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.le_zymc = LabelEncoder()
        self.le_szd = LabelEncoder()
        self.model_1 = self._new_model()
        self.model_2 = self._new_model()
        self.model_3 = self._new_model()
        self.model_4 = self._new_model()
        self.model_5 = self._new_model()

    @staticmethod
    def _new_model() -> XGBRegressor:
        return XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            reg_lambda=1.0,
            min_child_weight=1,
            n_jobs=-1,
        )

    @staticmethod
    def _check_columns(df: pd.DataFrame, required_cols):
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"缺少必要列: {missing}")

    def _prepare_data(self, csv_path: str):
        df_raw = pd.read_csv(csv_path)
        required_cols = [
            "ZYMC", "SZD", "SFSYL", "SF985", "SF211",
            "JHRS_3", "JHRS_2", "JHRS_1",
            "SJRS_3", "SJRS_2", "SJRS_1",
            "XX_Med_3", "XX_Med_2", "XX_Med_1",
            "ZY_Max_3", "ZY_Max_2", "ZY_Max_1",
            "ZY_Min_3", "ZY_Min_2", "ZY_Min_1",
            "CS", "NX", "YJ",
        ]
        self._check_columns(df_raw, required_cols)

        df = df_raw.copy()
        numeric_cols = [c for c in required_cols if c not in ["ZYMC", "SZD"]]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(axis=0).reset_index(drop=True)
        df["XXCC"] = df["SFSYL"] + df["SF985"] + df["SF211"]
        df["ZYMC"] = self.le_zymc.fit_transform(df["ZYMC"].astype(str))
        df["SZD"] = self.le_szd.fit_transform(df["SZD"].astype(str))
        return df, df_raw

    def run(self, csv_path: str):
        df, df_raw = self._prepare_data(csv_path)
        base_features = [
            "ZYMC", "SZD", "XXCC", "JHRS_3", "JHRS_2", "SJRS_3", "SJRS_2",
            "XX_Med_3", "XX_Med_2", "ZY_Max_3", "ZY_Max_2", "ZY_Min_3", "ZY_Min_2",
            "CS", "NX", "YJ",
        ]

        train_df, _ = train_test_split(df, train_size=0.7, random_state=self.seed, shuffle=True)
        train_df = train_df.copy()

        m1_features = base_features.copy()
        self.model_1.fit(train_df[m1_features], train_df["JHRS_1"])
        train_df["JHRS_1_hat"] = self.model_1.predict(train_df[m1_features])

        m2_features = [
            "ZYMC", "SZD", "XXCC", "JHRS_3", "JHRS_2", "JHRS_1_hat", "SJRS_3", "SJRS_2",
            "XX_Med_3", "XX_Med_2", "ZY_Max_3", "ZY_Max_2", "ZY_Min_3", "ZY_Min_2",
            "CS", "NX", "YJ",
        ]
        self.model_2.fit(train_df[m2_features], train_df["SJRS_1"])
        train_df["SJRS_1_hat"] = self.model_2.predict(train_df[m2_features])

        m345_features = [
            "ZYMC", "SZD", "XXCC", "JHRS_3", "JHRS_2", "JHRS_1_hat", "SJRS_3", "SJRS_2", "SJRS_1_hat",
            "XX_Med_3", "XX_Med_2", "ZY_Max_3", "ZY_Max_2", "ZY_Min_3", "ZY_Min_2",
            "CS", "NX", "YJ",
        ]
        self.model_3.fit(train_df[m345_features], train_df["XX_Med_1"])
        self.model_4.fit(train_df[m345_features], train_df["ZY_Max_1"])
        self.model_5.fit(train_df[m345_features], train_df["ZY_Min_1"])

        full_df = df.copy()
        self.model_1.fit(full_df[m1_features], full_df["JHRS_1"])
        full_df["JHRS_1_hat"] = self.model_1.predict(full_df[m1_features])
        self.model_2.fit(full_df[m2_features], full_df["SJRS_1"])
        full_df["SJRS_1_hat"] = self.model_2.predict(full_df[m2_features])
        self.model_3.fit(full_df[m345_features], full_df["XX_Med_1"])
        self.model_4.fit(full_df[m345_features], full_df["ZY_Max_1"])
        self.model_5.fit(full_df[m345_features], full_df["ZY_Min_1"])

        pred_df = full_df.copy()
        m1_pred_input = pd.DataFrame({
            "ZYMC": pred_df["ZYMC"], "SZD": pred_df["SZD"], "XXCC": pred_df["XXCC"],
            "JHRS_3": pred_df["JHRS_2"], "JHRS_2": pred_df["JHRS_1"],
            "SJRS_3": pred_df["SJRS_2"], "SJRS_2": pred_df["SJRS_1"],
            "XX_Med_3": pred_df["XX_Med_2"], "XX_Med_2": pred_df["XX_Med_1"],
            "ZY_Max_3": pred_df["ZY_Max_2"], "ZY_Max_2": pred_df["ZY_Max_1"],
            "ZY_Min_3": pred_df["ZY_Min_2"], "ZY_Min_2": pred_df["ZY_Min_1"],
            "CS": pred_df["CS"], "NX": pred_df["NX"], "YJ": pred_df["YJ"],
        })
        pred_df["JHRS_0"] = self.model_1.predict(m1_pred_input[m1_features])

        m2_pred_input = pd.DataFrame({
            "ZYMC": pred_df["ZYMC"], "SZD": pred_df["SZD"], "XXCC": pred_df["XXCC"],
            "JHRS_3": pred_df["JHRS_2"], "JHRS_2": pred_df["JHRS_1"], "JHRS_1_hat": pred_df["JHRS_0"],
            "SJRS_3": pred_df["SJRS_2"], "SJRS_2": pred_df["SJRS_1"],
            "XX_Med_3": pred_df["XX_Med_2"], "XX_Med_2": pred_df["XX_Med_1"],
            "ZY_Max_3": pred_df["ZY_Max_2"], "ZY_Max_2": pred_df["ZY_Max_1"],
            "ZY_Min_3": pred_df["ZY_Min_2"], "ZY_Min_2": pred_df["ZY_Min_1"],
            "CS": pred_df["CS"], "NX": pred_df["NX"], "YJ": pred_df["YJ"],
        })
        pred_df["SJRS_0"] = self.model_2.predict(m2_pred_input[m2_features])

        m345_pred_input = pd.DataFrame({
            "ZYMC": pred_df["ZYMC"], "SZD": pred_df["SZD"], "XXCC": pred_df["XXCC"],
            "JHRS_3": pred_df["JHRS_2"], "JHRS_2": pred_df["JHRS_1"], "JHRS_1_hat": pred_df["JHRS_0"],
            "SJRS_3": pred_df["SJRS_2"], "SJRS_2": pred_df["SJRS_1"], "SJRS_1_hat": pred_df["SJRS_0"],
            "XX_Med_3": pred_df["XX_Med_2"], "XX_Med_2": pred_df["XX_Med_1"],
            "ZY_Max_3": pred_df["ZY_Max_2"], "ZY_Max_2": pred_df["ZY_Max_1"],
            "ZY_Min_3": pred_df["ZY_Min_2"], "ZY_Min_2": pred_df["ZY_Min_1"],
            "CS": pred_df["CS"], "NX": pred_df["NX"], "YJ": pred_df["YJ"],
        })
        pred_df["XX_Med_0"] = self.model_3.predict(m345_pred_input[m345_features])
        pred_df["ZY_Max_0"] = self.model_4.predict(m345_pred_input[m345_features])
        pred_df["ZY_Min_0"] = self.model_5.predict(m345_pred_input[m345_features])

        out_df = df_raw.loc[pred_df.index].copy().reset_index(drop=True)
        out_df["JHRS_0"] = pred_df["JHRS_0"].values
        out_df["SJRS_0"] = pred_df["SJRS_0"].values
        out_df["XX_Med_0"] = pred_df["XX_Med_0"].values
        out_df["ZY_Max_0"] = pred_df["ZY_Max_0"].values
        out_df["ZY_Min_0"] = pred_df["ZY_Min_0"].values
        return out_df


print("运行级联XGBoost预计算...")
df = RankForecastPipeline(seed=42).run(CSV_PATH)

records = []
for idx, row in df.iterrows():
    min_r = int(round(row.get('ZY_Min_0', 0)))
    max_r = int(round(row.get('ZY_Max_0', 0)))
    med_r = int(round(row.get('XX_Med_0', 0)))
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
