"""
高考志愿模拟填报系统 - Web版后端
使用 Flask 提供 REST API，前端调用进行志愿查询与管理
"""
import os
import json
import csv
import io
import datetime
import threading
from typing import List, Dict, Optional, Tuple

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 数据加载（读预计算的 data.json）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALL_MAJORS: List[Dict] = []
_data_lock = threading.Lock()
_data_ready = False

def init_data():
    global _data_ready
    with _data_lock:
        if _data_ready:
            return
        json_path = os.path.join(BASE_DIR, "data.json")
        print(f"正在加载数据: {json_path}")
        with open(json_path, encoding="utf-8") as f:
            ALL_MAJORS[:] = json.load(f)
        _data_ready = True
        print(f"数据加载完成，共 {len(ALL_MAJORS)} 条记录")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 冲/稳/保算法
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

UNREACHABLE_THRESHOLD = 1.0

def calculate_tag_with_prob(user_rank: Optional[int], major_data: Dict) -> Tuple[Optional[str], Optional[int]]:
    if user_rank is None or user_rank <= 0:
        return "稳", 50
    min_rank = major_data.get("min_rank", 0)
    if min_rank <= 0:
        return "稳", 50
    diff = (user_rank - min_rank) / min_rank
    if diff > UNREACHABLE_THRESHOLD:
        return None, None
    if diff >= 0.30: return "冲", 20
    if diff >= 0.15: return "冲", 30
    if diff >= 0:    return "冲", 40
    if diff >= -0.10: return "稳", 60
    if diff >= -0.15: return "稳", 75
    if diff >= -0.30: return "保", 85
    return "保", 95

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 用户系统
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ACCOUNTS = {
    ("guest", "guest"): "guest",
    ("vip",   "vip"):   "vip",
    ("dev",   "dev"):   "dev",
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
    username = body.get("username", "").strip()
    password = body.get("password", "").strip()
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
    majors  = body.get("majors", [])
    conditions   = body.get("conditions", [])
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

    results = _apply_sort(results, sort_settings)
    return jsonify({"total": len(results), "data": results})


def _check_conditions(item, conditions, user_rank=None):
    keep = None
    for cond in conditions:
        field = cond.get("field", "")
        op    = cond.get("operator", "=")
        val   = cond.get("value", "").strip()
        logic = cond.get("logic", "AND")
        if not val: continue
        try:
            if   field == "最低排名": iv, tv = item.get("min_rank", 0),    int(val)
            elif field == "中位数":   iv, tv = item.get("median_rank", 0), int(val)
            elif field == "地区":     iv, tv = item.get("region", ""),     val
            elif field == "专业":     iv, tv = item.get("major", ""),      val
            elif field == "标签":
                iv = calculate_tag_with_prob(user_rank, item)[0] if user_rank else "稳"
                tv = val
            else: continue
            if op == "=":  r = iv == tv
            elif op == "≠": r = iv != tv
            elif op == "<": r = iv < tv
            elif op == ">": r = iv > tv
            elif op == "≤": r = iv <= tv
            elif op == "≥": r = iv >= tv
            elif op == "包含":   r = str(tv) in str(iv)
            elif op == "不包含": r = str(tv) not in str(iv)
            else: r = True
        except: r = True
        if keep is None: keep = r
        elif logic == "AND": keep = keep and r
        else: keep = keep or r
    return bool(keep) if keep is not None else True


def _apply_sort(data, sort_settings):
    for s in reversed(sort_settings):
        field = s.get("field", "min_rank")
        asc   = s.get("ascending", True)
        key_map = {
            "min_rank":    lambda x: x.get("min_rank", 0),
            "max_rank":    lambda x: x.get("max_rank", 0),
            "median_rank": lambda x: x.get("median_rank", 0),
            "region":      lambda x: x.get("region", ""),
            "major":       lambda x: x.get("major", ""),
        }
        if field in key_map:
            data.sort(key=key_map[field], reverse=not asc)
    return data


@app.route("/api/tag", methods=["POST"])
def get_tag():
    body = request.json or {}
    user_rank  = body.get("user_rank")
    major_data = body.get("major_data", {})
    if user_rank:
        tag, prob = calculate_tag_with_prob(int(user_rank), major_data)
    else:
        tag, prob = "稳", None
    return jsonify({"tag": tag, "prob": prob})


@app.route("/api/export", methods=["POST"])
def export_scheme():
    body = request.json or {}
    scheme_name = body.get("scheme_name", "未命名方案")
    user_rank   = body.get("user_rank")
    data_list   = body.get("data", [])
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["方案名称", scheme_name])
    writer.writerow(["导出时间", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow(["我的排名", user_rank if user_rank else "未设置"])
    writer.writerow([])
    writer.writerow(["ID","地区","专业","最低排名","最高排名","中位数排名","双一流","985","211","冲稳保","概率"])
    for d in data_list:
        if user_rank:
            tag, prob = calculate_tag_with_prob(int(user_rank), d)
        else:
            tag = d.get("tag", "稳"); prob = d.get("prob", "")
        writer.writerow([d["id"], d["region"], d["major"],
                         d["min_rank"], d["max_rank"], d["median_rank"],
                         d.get("SFSYL",0), d.get("SF985",0), d.get("SF211",0),
                         tag, f"{prob}%" if prob else ""])
    return jsonify({"csv": output.getvalue()})


@app.route("/api/ai_analyze", methods=["POST"])
def ai_analyze():
    body = request.json or {}
    scheme_data = body.get("scheme_data", [])
    api_key     = body.get("api_key", "sk-5f4ada1f0fe8430abed056d9622634be")
    if not scheme_data:
        return jsonify({"result": "方案为空，无法分析。"})
    try:
        import dashscope
        dashscope.api_key = api_key
        tag_map = {"冲": "冲(高风险)", "稳": "稳(适中)", "保": "保(稳妥)"}
        lines = []
        for i, item in enumerate(scheme_data):
            lbl  = tag_map.get(item.get("_tag") or item.get("tag",""), "未知")
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
        resp = dashscope.Generation.call(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7, top_p=0.9
        )
        try:    content = resp.output.choices[0]["message"]["content"]
        except: content = str(resp)
        return jsonify({"result": content})
    except ImportError:
        return jsonify({"result": "服务器未安装 dashscope，AI功能不可用。"})
    except Exception as e:
        return jsonify({"result": f"AI分析出错：{str(e)}"})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 启动
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)