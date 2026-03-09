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
from typing import List, Dict, Optional, Tuple

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = BASE_DIR
sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 机器学习流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_and_preprocess_data():
    csv_path = os.path.join(PARENT_DIR, "专业录取分数统计表.csv")
    df_raw = pd.read_csv(csv_path, quotechar='"', encoding='utf-8')
    df = df_raw.copy()
    expected = ['ZYMC','SZD','SFSYL','SF985','SF211',
                'JHRS_3','JHRS_2','JHRS_1','SJRS_3','SJRS_2','SJRS_1',
                'XX_Med_3','XX_Med_2','XX_Med_1',
                'ZY_Max_3','ZY_Max_2','ZY_Max_1',
                'ZY_Min_3','ZY_Min_2','ZY_Min_1','CS','NX','YJ']
    for col in expected:
        if col not in df.columns:    df[col] = np.nan
        if col not in df_raw.columns: df_raw[col] = np.nan
    numeric = [c for c in expected if c not in ('ZYMC','SZD','SFSYL','SF985','SF211')]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[numeric] = SimpleImputer(strategy='median').fit_transform(df[numeric])
    return df, df_raw

def feature_engineering(df):
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
    return df

def lasso_feature_selection(df):
    feature_cols = [
        'JHRS_1','JHRS_2','JHRS_3','SJRS_1','SJRS_2','SJRS_3',
        'XX_Med_1','XX_Med_2','XX_Med_3','ZY_Max_1','ZY_Max_2','ZY_Max_3',
        'ZY_Min_1','ZY_Min_2','ZY_Min_3','CS','NX','YJ',
        'ZY_Min_avg','ZY_Min_trend','ZY_Min_volatility',
        'JHRS_avg','JHRS_trend','LR_avg','Competition_avg',
        'SFSYL','SF985','SF211'
    ]
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(df[feature_cols], df['ZY_Min_3'])
    return [feature_cols[i] for i in range(len(feature_cols)) if lasso.coef_[i] != 0]

def xgboost_prediction(df, feats):
    X_tr, X_te, y_tr, _ = train_test_split(df[feats], df['ZY_Min_3'], test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.fit(X_tr, y_tr)
    return model

def predict_next_year(df, model, feats):
    df['ZY_Min_4'] = model.predict(df[feats])
    df['ZY_Min_4'] = np.maximum(df['ZY_Min_4'], 0)
    df['ZY_Min_4'] = (0.4*df['ZY_Min_3'] + 0.3*df['ZY_Min_2'] +
                      0.2*df['ZY_Min_1'] + 0.1*df['ZY_Min_4'])
    return df

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 冲/稳/保算法
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

UNREACHABLE_THRESHOLD = 1.0

def calculate_tag_with_prob(user_rank, major_data):
    if user_rank is None or user_rank <= 0: return "稳", 50
    min_rank = major_data.get("min_rank", 0)
    if min_rank <= 0: return "稳", 50
    diff = (user_rank - min_rank) / min_rank
    if diff > UNREACHABLE_THRESHOLD: return None, None
    if diff >= 0.30: return "冲", 20
    if diff >= 0.15: return "冲", 30
    if diff >= 0:    return "冲", 40
    if diff >= -0.10: return "稳", 60
    if diff >= -0.15: return "稳", 75
    if diff >= -0.30: return "保", 85
    return "保", 95

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
            df, _ = load_and_preprocess_data()
            df = feature_engineering(df)
            feats = lasso_feature_selection(df)
            model = xgboost_prediction(df, feats)
            df = predict_next_year(df, model, feats)
            for idx, row in df.iterrows():
                min_r = int(round(row.get('ZY_Min_4', 0)))
                max_r = int(round(row.get('ZY_Max_3', 0)))
                med_r = int(round(row.get('XX_Med_3', 0)))
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
        if conditions and not _check_conditions(m, conditions, user_rank): continue
        item = dict(m)
        if user_rank:
            tag, prob = calculate_tag_with_prob(user_rank, m)
            if tag is None: continue
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
            "region":lambda x:x.get("region",""), "major":lambda x:x.get("major","")}
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