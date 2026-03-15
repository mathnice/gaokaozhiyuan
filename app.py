"""
高考志愿模拟填报系统 - Web版后端
使用 Flask 提供 REST API，前端调用进行志愿查询与管理
"""
import sys
import os
import json
import csv
import io
import datetime
import threading
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = BASE_DIR
sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 机器学习流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ModelMetrics:
    name: str
    rmse: float
    mae: float
    r2: float


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
    def _check_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"缺少必要列: {missing}")

    def _prepare_data(self, csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

        if df.empty:
            raise ValueError("清洗后数据为空，请检查原始数据质量。")
        return df, df_raw

    def run(self, csv_path: str) -> pd.DataFrame:
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 冲/稳/保算法
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

UNREACHABLE_THRESHOLD = 1.0
BAO_PROB_THRESHOLD = 0.80
STEADY_PROB_THRESHOLD = 0.55
MIN_CHONG_DISPLAY_PROB = 0.20
SIGMA_MODE = "neutral"
SIGMA_MODE_FACTOR = {
    "conservative": 0.80,
    "neutral": 1.00,
    "aggressive": 1.20,
}


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def _estimate_distribution_params(major_data: Dict[str, Any]) -> Tuple[float, float]:
    low_rank = float(major_data.get('min_rank', 0) or 0)
    median_rank = float(major_data.get('median_rank', 0) or 0)
    high_rank = float(major_data.get('max_rank', 0) or 0)

    valid_ranks = [rank for rank in (low_rank, median_rank, high_rank) if rank > 0]
    if not valid_ranks:
        return 1.0, 1.0

    actual_best = min(valid_ranks)
    actual_worst = max(valid_ranks)
    mu = median_rank if median_rank > 0 else float(sum(valid_ranks) / len(valid_ranks))
    sample_sigma = float(np.std(valid_ranks, ddof=0)) if len(valid_ranks) >= 2 else 0.0
    sigma_from_range = (actual_worst - actual_best) / 3.0 if actual_worst > actual_best else 0.0
    sigma_floor = max(mu * 0.08, 50.0)
    sigma = max(sample_sigma, sigma_from_range, sigma_floor)
    sigma *= SIGMA_MODE_FACTOR.get(SIGMA_MODE, 1.0)
    return mu, sigma


def _calculate_admission_probability(user_rank: int, major_data: Dict[str, Any]) -> float:
    mu, sigma = _estimate_distribution_params(major_data)
    cdf_value = _normal_cdf(float(user_rank), mu, sigma)
    admit_prob = 1.0 - cdf_value
    return min(max(admit_prob, 0.0), 1.0)


def _tag_from_probability(prob: float) -> str:
    if prob >= BAO_PROB_THRESHOLD:
        return "保"
    if prob >= STEADY_PROB_THRESHOLD:
        return "稳"
    return "冲"


def _should_hide_result(tag: Optional[str], prob: float) -> bool:
    return tag == "冲" and prob < MIN_CHONG_DISPLAY_PROB


def is_unreachable(user_rank: Optional[int], major_data: Dict[str, Any]) -> bool:
    if user_rank is None or user_rank <= 0:
        return False
    min_rank = major_data.get('min_rank', 0)
    if min_rank <= 0:
        return False
    rank_diff = (user_rank - min_rank) / min_rank
    return rank_diff > UNREACHABLE_THRESHOLD


def should_recommend(user_rank: Optional[int], major_data: Dict[str, Any]) -> bool:
    if is_unreachable(user_rank, major_data):
        return False
    if user_rank is None or user_rank <= 0:
        return True
    prob = _calculate_admission_probability(user_rank, major_data)
    tag = _tag_from_probability(prob)
    return not _should_hide_result(tag, prob)

def calculate_tag_with_prob(user_rank, major_data):
    if user_rank is None or user_rank <= 0:
        return "稳", 50
    if major_data.get('min_rank', 0) <= 0:
        return "稳", 50
    if is_unreachable(user_rank, major_data):
        return None, None
    prob = _calculate_admission_probability(user_rank, major_data)
    tag = _tag_from_probability(prob)
    prob_percent = int(round(prob * 100))
    prob_percent = min(max(prob_percent, 1), 99)
    return tag, prob_percent

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 数据初始化（懒加载，线程安全）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALL_MAJORS: List[Dict] = []
_data_lock = threading.Lock()
_data_ready = False

def init_data():
    global _data_ready
    with _data_lock:
        if _data_ready:
            return
        # 优先读预计算 JSON（快速）
        json_path = os.path.join(BASE_DIR, "data.json")
        if os.path.exists(json_path):
            print("从 data.json 快速加载...")
            with open(json_path, encoding="utf-8") as f:
                ALL_MAJORS[:] = json.load(f)
        else:
            # 回退：运行完整 ML 流程
            print("data.json 不存在，启动 ML 流程...")
            csv_path = os.path.join(PARENT_DIR, "专业录取分数统计表.csv")
            df = RankForecastPipeline(seed=42).run(csv_path)
            for idx, row in df.iterrows():
                min_r = int(round(row.get('ZY_Min_0', 0)))
                max_r = int(round(row.get('ZY_Max_0', 0)))
                med_r = int(round(row.get('XX_Med_0', 0)))
                vals = sorted([min_r, max_r, med_r], reverse=True)
                ALL_MAJORS.append({
                    "id": int(idx), "region": str(row['SZD']), "major": str(row['ZYMC']),
                    "min_rank": vals[0], "max_rank": vals[2], "median_rank": vals[1],
                    "SFSYL": int(row.get('SFSYL',0)), "SF985": int(row.get('SF985',0)),
                    "SF211": int(row.get('SF211',0)),
                })
        _data_ready = True
        print(f"数据就绪，共 {len(ALL_MAJORS)} 条记录")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 用户系统
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ACCOUNTS = {
    ("guest","guest"): "guest",
    ("vip","vip"):     "vip",
    ("dev","dev"):     "dev",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REST API 路由
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/health")
def health():
    return jsonify({"ok": True, "records": len(ALL_MAJORS)})

@app.route("/api/login", methods=["POST"])
def login():
    body = request.json or {}
    username = body.get("username","").strip()
    password = body.get("password","").strip()
    user_type = ACCOUNTS.get((username, password))
    if user_type:
        return jsonify({"success": True, "username": username, "user_type": user_type})
    return jsonify({"success": False, "message": "用户名或密码错误"}), 401

@app.route("/api/regions", methods=["GET"])
def get_regions():
    init_data()
    return jsonify(sorted(set(m["region"] for m in ALL_MAJORS)))

@app.route("/api/majors_list", methods=["GET"])
def get_majors_list():
    init_data()
    return jsonify(sorted(set(m["major"] for m in ALL_MAJORS)))

@app.route("/api/search", methods=["POST"])
def search():
    init_data()
    body = request.json or {}
    user_rank = body.get("user_rank")
    if user_rank is not None:
        try: user_rank = int(user_rank)
        except: user_rank = None
    regions = body.get("regions", [])
    majors  = body.get("majors",  [])
    conditions    = body.get("conditions", [])
    sort_settings = body.get("sort", [])

    results = []
    for m in ALL_MAJORS:
        if regions and m["region"] not in regions: continue
        if majors  and m["major"]  not in majors:  continue
        if user_rank and not should_recommend(user_rank, m): continue
        if conditions and not _check_conditions(m, conditions, user_rank): continue
        item = dict(m)
        if user_rank:
            tag, prob = calculate_tag_with_prob(user_rank, m)
            item["tag"] = tag; item["prob"] = prob
        else:
            item["tag"] = "稳"; item["prob"] = None
        results.append(item)
    return jsonify({"total": len(results), "data": _apply_sort(results, sort_settings)})


def _check_conditions(item, conditions, user_rank=None):
    keep = None
    for cond in conditions:
        field = cond.get("field","");  op = cond.get("operator","=")
        val   = cond.get("value","").strip(); logic = cond.get("logic","AND")
        if not val: continue
        try:
            if   field=="最低排名": iv,tv = item.get("min_rank",0),    int(val)
            elif field=="中位数":   iv,tv = item.get("median_rank",0), int(val)
            elif field=="地区":     iv,tv = item.get("region",""),     val
            elif field=="专业":     iv,tv = item.get("major",""),      val
            elif field=="标签":
                iv = calculate_tag_with_prob(user_rank,item)[0] if user_rank else "稳"
                tv = val
            else: continue
            ops = {"=":lambda a,b:a==b,"≠":lambda a,b:a!=b,"<":lambda a,b:a<b,
                   ">":lambda a,b:a>b,"≤":lambda a,b:a<=b,"≥":lambda a,b:a>=b,
                   "包含":lambda a,b:str(b) in str(a),"不包含":lambda a,b:str(b) not in str(a)}
            r = ops.get(op, lambda a,b: True)(iv, tv)
        except: r = True
        if keep is None: keep = r
        elif logic=="AND": keep = keep and r
        else: keep = keep or r
    return bool(keep) if keep is not None else True


def _apply_sort(data, sort_settings):
    keys = {"min_rank":lambda x:x.get("min_rank",0), "max_rank":lambda x:x.get("max_rank",0),
            "median_rank":lambda x:x.get("median_rank",0),
            "region":lambda x:x.get("region",""), "major":lambda x:x.get("major",""),
            "prob":lambda x:x.get("prob", -1)}
    for s in reversed(sort_settings):
        f = s.get("field","min_rank")
        if f in keys: data.sort(key=keys[f], reverse=not s.get("ascending",True))
    return data


@app.route("/api/tag", methods=["POST"])
def get_tag():
    body = request.json or {}
    user_rank = body.get("user_rank"); major_data = body.get("major_data",{})
    if user_rank:
        tag, prob = calculate_tag_with_prob(int(user_rank), major_data)
    else: tag, prob = "稳", None
    return jsonify({"tag": tag, "prob": prob})


@app.route("/api/export", methods=["POST"])
def export_scheme():
    body = request.json or {}
    scheme_name = body.get("scheme_name","未命名方案")
    user_rank   = body.get("user_rank"); data_list = body.get("data",[])
    output = io.StringIO(); writer = csv.writer(output)
    writer.writerow(["方案名称", scheme_name])
    writer.writerow(["导出时间", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow(["我的排名", user_rank if user_rank else "未设置"]); writer.writerow([])
    writer.writerow(["ID","地区","专业","最低排名","最高排名","中位排名","双一流","985","211","冲稳保","概率"])
    for d in data_list:
        if user_rank: tag,prob = calculate_tag_with_prob(int(user_rank),d)
        else:         tag,prob = d.get("tag","稳"), d.get("prob","")
        writer.writerow([d["id"],d["region"],d["major"],d["min_rank"],d["max_rank"],d["median_rank"],
                         d.get("SFSYL",0),d.get("SF985",0),d.get("SF211",0),
                         tag, f"{prob}%" if prob else ""])
    return jsonify({"csv": output.getvalue()})


@app.route("/api/ai_analyze", methods=["POST"])
def ai_analyze():
    body = request.json or {}
    scheme_data = body.get("scheme_data",[]); api_key = body.get("api_key","sk-5f4ada1f0fe8430abed056d9622634be")
    if not scheme_data: return jsonify({"result":"方案为空，无法分析。"})
    try:
        import dashscope; dashscope.api_key = api_key
        tag_map = {"冲":"冲(高风险)","稳":"稳(适中)","保":"保(稳妥)"}
        lines = []
        for i,item in enumerate(scheme_data):
            lbl = tag_map.get(item.get("_tag") or item.get("tag",""),"未知")
            syls = [k for k in ("SF985","SF211","SFSYL") if item.get(k)]
            syl_str = f"[{','.join(syls)}]" if syls else ""
            lines.append(f"{i+1:2d}. 【{lbl}】 {item['region']} · {item['major']} {syl_str}\n"
                         f"     最低:{item['min_rank']}  中位:{item['median_rank']}  最高:{item['max_rank']}")
        prompt = f"""你是一位权威的高考志愿填报专家，请对以下志愿方案进行专业评估。

【志愿方案清单（共{len(scheme_data)}个）】
{chr(10).join(lines)}

请按照以下固定格式输出分析报告，每个章节必须包含，不得省略：

## 一、冲稳保结构评估
- 冲/稳/保各多少个，比例是否合理（建议4:4:2），说明原因

## 二、排名梯度分析
- 各志愿最低录取排名梯度是否合理，是否存在断层

## 三、专业方向评估
- 专业是否集中，跨度评价，就业前景简评

## 四、地域布局评估
- 城市/省份分布合理性评价

## 五、风险识别与改进建议
- ⚠️ 风险X：...（1~3条）
- ✅ 建议X：...（对应改进）

## 六、综合评分
综合评分：X/10 分 —— 理由：...

语言专业客观，针对具体内容，避免泛泛而谈。"""
        resp = dashscope.Generation.call(model="qwen-plus",
            messages=[{"role":"user","content":prompt}], temperature=0.7,top_p=0.9)
        try:    content = resp.output.choices[0]["message"]["content"]
        except: content = str(resp)
        return jsonify({"result": content})
    except ImportError:
        return jsonify({"result":"服务器未安装 dashscope，AI功能不可用。"})
    except Exception as e:
        return jsonify({"result": f"AI分析出错：{str(e)}"})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 启动
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)