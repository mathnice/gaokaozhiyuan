"""
高考志愿模拟填报系统 - Web版后端
使用 Flask 提供 REST API，前端调用进行志愿查询与管理
"""
import sys
import os
import json
import csv
import datetime
import random
from typing import List, Dict, Any, Optional, Tuple

# ── 目录配置 ──────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = BASE_DIR  # CSV 已移至同一目录
sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)  # 开发阶段允许跨域

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 机器学习 & 数据处理（复用原逻辑）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_and_preprocess_data():
    csv_path = os.path.join(PARENT_DIR, "专业录取分数统计表.csv")
    df_raw = pd.read_csv(csv_path, quotechar='"', encoding='utf-8')
    df = df_raw.copy()

    expected_columns = ['ZYMC', 'SZD', 'SFSYL', 'SF985', 'SF211',
                        'JHRS_3', 'JHRS_2', 'JHRS_1', 'SJRS_3', 'SJRS_2', 'SJRS_1',
                        'XX_Med_3', 'XX_Med_2', 'XX_Med_1',
                        'ZY_Max_3', 'ZY_Max_2', 'ZY_Max_1',
                        'ZY_Min_3', 'ZY_Min_2', 'ZY_Min_1', 'CS', 'NX', 'YJ']
    for col in expected_columns:
        if col not in df.columns:
            df[col] = np.nan
        if col not in df_raw.columns:
            df_raw[col] = np.nan

    numeric_cols = ['JHRS_3', 'JHRS_2', 'JHRS_1', 'SJRS_3', 'SJRS_2', 'SJRS_1',
                    'XX_Med_3', 'XX_Med_2', 'XX_Med_1', 'ZY_Max_3', 'ZY_Max_2', 'ZY_Max_1',
                    'ZY_Min_3', 'ZY_Min_2', 'ZY_Min_1', 'CS', 'NX', 'YJ']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df, df_raw


def feature_engineering(df):
    df['ZY_Min_avg'] = df[['ZY_Min_1', 'ZY_Min_2', 'ZY_Min_3']].mean(axis=1)
    df['ZY_Min_trend'] = (df['ZY_Min_3'] - df['ZY_Min_1']) / (df['ZY_Min_1'] + 1e-6)
    df['ZY_Min_volatility'] = df[['ZY_Min_1', 'ZY_Min_2', 'ZY_Min_3']].std(axis=1) / (df['ZY_Min_avg'] + 1e-6)
    df['JHRS_avg'] = df[['JHRS_1', 'JHRS_2', 'JHRS_3']].mean(axis=1)
    df['JHRS_trend'] = (df['JHRS_3'] - df['JHRS_1']) / (df['JHRS_1'] + 1e-6)
    df['LR_1'] = df['SJRS_1'] / (df['JHRS_1'] + 1e-6)
    df['LR_2'] = df['SJRS_2'] / (df['JHRS_2'] + 1e-6)
    df['LR_3'] = df['SJRS_3'] / (df['JHRS_3'] + 1e-6)
    df['LR_avg'] = df[['LR_1', 'LR_2', 'LR_3']].mean(axis=1)
    df['Competition_1'] = df['XX_Med_1'] / (df['ZY_Min_1'] + 1e-6)
    df['Competition_2'] = df['XX_Med_2'] / (df['ZY_Min_2'] + 1e-6)
    df['Competition_3'] = df['XX_Med_3'] / (df['ZY_Min_3'] + 1e-6)
    df['Competition_avg'] = df[['Competition_1', 'Competition_2', 'Competition_3']].mean(axis=1)
    df['SFSYL'] = df['SFSYL'].fillna(0).astype(int)
    df['SF985'] = df['SF985'].fillna(0).astype(int)
    df['SF211'] = df['SF211'].fillna(0).astype(int)
    return df


def lasso_feature_selection(df):
    feature_cols = [
        'JHRS_1', 'JHRS_2', 'JHRS_3', 'SJRS_1', 'SJRS_2', 'SJRS_3',
        'XX_Med_1', 'XX_Med_2', 'XX_Med_3', 'ZY_Max_1', 'ZY_Max_2', 'ZY_Max_3',
        'ZY_Min_1', 'ZY_Min_2', 'ZY_Min_3', 'CS', 'NX', 'YJ',
        'ZY_Min_avg', 'ZY_Min_trend', 'ZY_Min_volatility',
        'JHRS_avg', 'JHRS_trend', 'LR_avg', 'Competition_avg',
        'SFSYL', 'SF985', 'SF211'
    ]
    X = df[feature_cols]
    y = df['ZY_Min_3']
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X, y)
    important_features = [feature_cols[i] for i in range(len(feature_cols)) if lasso.coef_[i] != 0]
    return important_features


def xgboost_prediction(df, important_features):
    X = df[important_features]
    y = df['ZY_Min_3']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.fit(X_train, y_train)
    return model


def predict_next_year(df, model, important_features):
    X = df[important_features]
    df['ZY_Min_4'] = model.predict(X)
    df['ZY_Min_4'] = np.maximum(df['ZY_Min_4'], 0)
    df['ZY_Min_4'] = (0.4 * df['ZY_Min_3'] + 0.3 * df['ZY_Min_2'] +
                      0.2 * df['ZY_Min_1'] + 0.1 * df['ZY_Min_4'])
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 冲/稳/保标签算法
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BAO_THRESHOLD = -0.15
# 超过此位次差（用户排名 > 学校最低排名 * 2）视为完全不可达，不纳入推荐
UNREACHABLE_THRESHOLD = 1.0

def calculate_tag_with_prob(user_rank: Optional[int], major_data: Dict) -> Tuple[Optional[str], Optional[int]]:
    """返回 (tag, prob)，当差距过大视为不可达时返回 (None, None)"""
    if user_rank is None or user_rank <= 0:
        return "稳", 50
    min_rank = major_data.get('min_rank', 0)
    if min_rank <= 0:
        return "稳", 50
    rank_diff = (user_rank - min_rank) / min_rank
    # 排名差距超过100%（用户排名是学校线2倍以上）→ 此校不在可选范围内
    if rank_diff > UNREACHABLE_THRESHOLD:
        return None, None
    if rank_diff >= 0.30:
        return "冲", 20
    elif rank_diff >= 0.15:
        return "冲", 30
    elif rank_diff >= 0:
        return "冲", 40
    elif rank_diff >= -0.10:
        return "稳", 60
    elif rank_diff >= -0.15:
        return "稳", 75
    elif rank_diff >= -0.30:
        return "保", 85
    else:
        return "保", 95


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 初始化数据（应用启动时执行一次）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALL_MAJORS: List[Dict] = []

def init_data():
    global ALL_MAJORS
    print("正在加载并处理数据...")
    df, raw_df = load_and_preprocess_data()
    df = feature_engineering(df)
    important_features = lasso_feature_selection(df)
    model = xgboost_prediction(df, important_features)
    df = predict_next_year(df, model, important_features)

    ALL_MAJORS = []
    for idx, row in df.iterrows():
        min_r = int(round(row.get('ZY_Min_4', 0)))
        max_r = int(round(row.get('ZY_Max_3', 0)))
        med_r = int(round(row.get('XX_Med_3', 0)))
        vals = sorted([min_r, max_r, med_r], reverse=True)
        rec = {
            "id": int(idx),
            "region": str(row['SZD']),
            "major": str(row['ZYMC']),
            "min_rank": vals[0],
            "max_rank": vals[2],
            "median_rank": vals[1],
            "SFSYL": int(row.get('SFSYL', 0)),
            "SF985": int(row.get('SF985', 0)),
            "SF211": int(row.get('SF211', 0)),
        }
        ALL_MAJORS.append(rec)
    print(f"数据初始化完成，共 {len(ALL_MAJORS)} 条专业记录")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 用户系统（内存简单实现）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ACCOUNTS = {
    ("guest", "guest"): "guest",
    ("vip", "vip"): "vip",
    ("dev", "dev"): "dev",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REST API 路由
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/login", methods=["POST"])
def login():
    body = request.json or {}
    username = body.get("username", "").strip()
    password = body.get("password", "").strip()
    user_type = ACCOUNTS.get((username, password))
    if user_type:
        return jsonify({"success": True, "username": username, "user_type": user_type})
    return jsonify({"success": False, "message": "用户名或密码错误"}), 401


@app.route("/api/regions", methods=["GET"])
def get_regions():
    regions = sorted(set(m["region"] for m in ALL_MAJORS))
    return jsonify(regions)


@app.route("/api/majors_list", methods=["GET"])
def get_majors_list():
    majors = sorted(set(m["major"] for m in ALL_MAJORS))
    return jsonify(majors)


@app.route("/api/search", methods=["POST"])
def search():
    """
    请求体:
    {
        "user_rank": 5000,          // 可选
        "regions": ["北京","上海"],  // [] 表示全部
        "majors": ["计算机类"],      // [] 表示全部
        "conditions": [...],         // 高级筛选条件（可选）
        "sort": [{"field":"min_rank","ascending":true}]  // 可选
    }
    """
    body = request.json or {}
    user_rank = body.get("user_rank")
    if user_rank is not None:
        try:
            user_rank = int(user_rank)
        except:
            user_rank = None

    regions = body.get("regions", [])   # [] = 全部
    majors = body.get("majors", [])     # [] = 全部
    conditions = body.get("conditions", [])
    sort_settings = body.get("sort", [])

    results = []
    for m in ALL_MAJORS:
        if regions and m["region"] not in regions:
            continue
        if majors and m["major"] not in majors:
            continue
        # 高级筛选
        if conditions and not _check_conditions(m, conditions, user_rank):
            continue

        item = dict(m)
        if user_rank:
            tag, prob = calculate_tag_with_prob(user_rank, m)
            # 位次差距过大（完全不可达）→ 直接过滤，不纳入结果
            if tag is None:
                continue
            item["tag"] = tag
            item["prob"] = prob
        else:
            item["tag"] = "稳"
            item["prob"] = None
        results.append(item)

    # 排序
    results = _apply_sort(results, sort_settings)
    return jsonify({"total": len(results), "data": results})


def _check_conditions(item: Dict, conditions: List[Dict], user_rank=None) -> bool:
    keep = None
    for cond in conditions:
        field = cond.get("field", "")
        op = cond.get("operator", "=")
        val = cond.get("value", "").strip()
        logic = cond.get("logic", "AND")
        if not val:
            continue

        try:
            if field == "最低排名":
                iv = item.get("min_rank", 0)
                tv = int(val)
            elif field == "中位数":
                iv = item.get("median_rank", 0)
                tv = int(val)
            elif field == "地区":
                iv = item.get("region", "")
                tv = val
            elif field == "专业":
                iv = item.get("major", "")
                tv = val
            elif field == "标签":
                if user_rank:
                    iv = calculate_tag_with_prob(user_rank, item)[0]
                else:
                    iv = "稳"
                tv = val
            else:
                continue

            if op == "=":
                r = (iv == tv)
            elif op == "≠":
                r = (iv != tv)
            elif op == "<":
                r = (iv < tv)
            elif op == ">":
                r = (iv > tv)
            elif op == "≤":
                r = (iv <= tv)
            elif op == "≥":
                r = (iv >= tv)
            elif op == "包含":
                r = str(tv) in str(iv)
            elif op == "不包含":
                r = str(tv) not in str(iv)
            else:
                r = True
        except:
            r = True

        if keep is None:
            keep = r
        elif logic == "AND":
            keep = keep and r
        else:
            keep = keep or r

    return bool(keep) if keep is not None else True


def _apply_sort(data: List[Dict], sort_settings: List[Dict]) -> List[Dict]:
    for s in reversed(sort_settings):
        field = s.get("field", "min_rank")
        asc = s.get("ascending", True)
        if field == "min_rank":
            data.sort(key=lambda x: x.get("min_rank", 0), reverse=not asc)
        elif field == "max_rank":
            data.sort(key=lambda x: x.get("max_rank", 0), reverse=not asc)
        elif field == "median_rank":
            data.sort(key=lambda x: x.get("median_rank", 0), reverse=not asc)
        elif field == "region":
            data.sort(key=lambda x: x.get("region", ""), reverse=not asc)
        elif field == "major":
            data.sort(key=lambda x: x.get("major", ""), reverse=not asc)
    return data


@app.route("/api/tag", methods=["POST"])
def get_tag():
    """计算单条记录的冲/稳/保标签"""
    body = request.json or {}
    user_rank = body.get("user_rank")
    major_data = body.get("major_data", {})
    if user_rank:
        tag, prob = calculate_tag_with_prob(int(user_rank), major_data)
    else:
        tag, prob = "稳", None
    return jsonify({"tag": tag, "prob": prob})


@app.route("/api/export", methods=["POST"])
def export_scheme():
    """
    将方案数据转为 CSV 字符串返回
    请求体: { "scheme_name": "...", "user_rank": ..., "data": [...] }
    """
    body = request.json or {}
    scheme_name = body.get("scheme_name", "未命名方案")
    user_rank = body.get("user_rank")
    data_list = body.get("data", [])

    import io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["方案名称", scheme_name])
    writer.writerow(["导出时间", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow(["我的排名", user_rank if user_rank else "未设置"])
    writer.writerow([])
    writer.writerow(["ID", "地区", "专业", "最低排名", "最高排名", "中位数排名",
                     "双一流", "985", "211", "冲稳保标签", "录取概率"])
    for d in data_list:
        if user_rank:
            tag, prob = calculate_tag_with_prob(int(user_rank), d)
        else:
            tag = d.get("tag", "稳")
            prob = d.get("prob", "")
        writer.writerow([
            d["id"], d["region"], d["major"], d["min_rank"], d["max_rank"], d["median_rank"],
            d.get("SFSYL", 0), d.get("SF985", 0), d.get("SF211", 0),
            tag, f"{prob}%" if prob else ""
        ])
    return jsonify({"csv": output.getvalue()})


@app.route("/api/ai_analyze", methods=["POST"])
def ai_analyze():
    """
    调用通义千问分析志愿方案
    请求体: { "scheme_data": [...], "api_key": "sk-..." }
    """
    body = request.json or {}
    scheme_data = body.get("scheme_data", [])
    api_key = body.get("api_key", "sk-5f4ada1f0fe8430abed056d9622634be")

    if not scheme_data:
        return jsonify({"result": "方案为空，无法分析。"})

    try:
        import dashscope
        dashscope.api_key = api_key

        # 构建包含冲稳保标签的方案摘要
        tag_map = {"冲": "冲(高风险)", "稳": "稳(适中)", "保": "保(稳妥)"}
        scheme_lines = []
        for i, item in enumerate(scheme_data):
            tag_label = tag_map.get(item.get('_tag') or item.get('tag', ''), '未知')
            syl_tags = []
            if item.get('SF985'): syl_tags.append('985')
            if item.get('SF211'): syl_tags.append('211')
            if item.get('SFSYL'): syl_tags.append('双一流')
            syl_str = f"[{','.join(syl_tags)}]" if syl_tags else ""
            scheme_lines.append(
                f"{i+1:2d}. 【{tag_label}】 {item['region']} · {item['major']} {syl_str}\n"
                f"     最低排名:{item['min_rank']}  中位:{item['median_rank']}  最高:{item['max_rank']}"
            )
        scheme_text = "\n".join(scheme_lines)

        prompt = f"""你是一位权威的高考志愿填报专家，请对以下志愿方案进行专业评估。

【志愿方案清单（共{len(scheme_data)}个）】
{scheme_text}

请按照以下固定格式输出分析报告，每个章节必须包含，不得省略：

## 一、冲稳保结构评估
- 本方案中"冲"有多少个、"稳"有多少个、"保"有多少个
- 比例是否合理（建议冲:稳:保 ≈ 4:4:2），并说明原因
- 若结构不合理，指出具体问题

## 二、排名梯度分析
- 列出各志愿最低录取排名，判断相邻志愿间的梯度是否合理（过大或过小都是风险）
- 是否存在梯度断层（即某两个相邻志愿之间分差悬殊）

## 三、专业方向评估
- 所选专业是否属于同一方向，还是跨度过大
- 就业前景与专业匹配度的简要评价

## 四、地域布局评估
- 考生选择的城市/省份分布是否过于集中或过于分散
- 给出合理性判断

## 五、风险识别与改进建议
- 列出该方案存在的1~3个主要风险点（格式：⚠️ 风险X：...）
- 给出具体、可操作的改进建议（格式：✅ 建议X：...）

## 六、综合评分
- 给出该方案的综合评分（满分10分），并用一句话说明理由
- 格式：综合评分：X/10 分 —— 理由：...

请确保语言专业、客观，针对具体志愿内容给出实质性建议，避免泛泛而谈。"""

        response = dashscope.Generation.call(
            model='qwen-plus',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.7, top_p=0.9
        )
        try:
            content = response.output.choices[0]['message']['content']
        except Exception:
            content = str(response)
        return jsonify({"result": content})
    except ImportError:
        return jsonify({"result": "服务器未安装 dashscope，AI功能不可用。"})
    except Exception as e:
        return jsonify({"result": f"AI分析出错：{str(e)}"})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 启动
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 初始化数据（gunicorn 直接导入模块时也会执行）
init_data()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
