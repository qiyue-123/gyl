import os
import pandas as pd
from datetime import datetime, timedelta
from dateutil.parser import isoparse
from openai import OpenAI
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from external_features import fetch_holidays, fetch_weather_daily, fetch_epidemic, compute_event_multipliers


def compute_discount(row):
    rp = float(row["regular_price"]) if pd.notna(row["regular_price"]) else 0.0
    dp = float(row["discount_price"]) if pd.notna(row["discount_price"]) else rp
    if rp <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (dp / rp)))


def build_baseline(df: pd.DataFrame, sku_id: str):
    sku = df[df["sku_id"] == sku_id].copy()
    if sku.empty:
        return [], 0
    sku.sort_values("date", inplace=True)
    cycle_days = int(sku["cycle_days"].iloc[-1]) if pd.notna(sku["cycle_days"].iloc[-1]) else 30
    # 简单基线：近7天移动平均 + 周期天数的平均（两者加权）
    sales = sku["sales"].astype(float)
    ma7 = sales.tail(7).mean() if len(sales) >= 1 else 0.0
    # 周期平均（取最近一个周期窗口或全局平均）
    ma_cycle = sales.tail(cycle_days).mean() if len(sales) >= cycle_days else sales.mean()
    baseline_val = 0.6 * ma7 + 0.4 * ma_cycle
    # 生成最近30天的日期与销量视图
    recent = sku.tail(min(30, len(sku)))[["date", "sales", "regular_price", "discount_price", "coupon", "cycle_days"]]
    return recent.to_dict(orient="records"), baseline_val


def fit_time_series_baseline(sales_series: pd.Series, cycle_days: int):
    # 判断季节性长度是否足够
    try:
        if len(sales_series) >= max(20, cycle_days * 3):
            # SARIMA (简化参数，可改为自动调参)
            model = SARIMAX(sales_series, order=(1, 1, 1), seasonal_order=(1, 0, 1, max(7, cycle_days)))
            res = model.fit(disp=False)
            return res
        else:
            model = ARIMA(sales_series, order=(1, 1, 1))
            res = model.fit()
            return res
    except Exception:
        # 回退到ARIMA
        model = ARIMA(sales_series, order=(1, 1, 1))
        res = model.fit()
        return res


def train_xgb_with_promotions(df: pd.DataFrame, sku_id: str, country_code: str = "CN",
                              weather_hist: pd.DataFrame | None = None,
                              epidemic_hist: pd.DataFrame | None = None):
    sku = df[df["sku_id"] == sku_id].copy()
    sku.sort_values("date", inplace=True)
    if len(sku) < 30:
        return None, None, None
    # 特征工程（加入节假日/天气/疫情外部特征，占位缺省值0）
    sku["day_of_week"] = pd.to_datetime(sku["date"]).dt.dayofweek
    sku["month"] = pd.to_datetime(sku["date"]).dt.month
    years = pd.to_datetime(sku["date"]).dt.year.unique().tolist()
    holiday_set = set()
    for y in years:
        holiday_set |= fetch_holidays(country_code, int(y))
    sku["is_holiday"] = pd.to_datetime(sku["date"]).dt.date.astype(str).isin(holiday_set).astype(int)
    # 天气历史（可选CSV）：列 date,tmax,tmin,precip
    if weather_hist is not None and not weather_hist.empty and all(col in weather_hist.columns for col in ["date", "tmax", "tmin", "precip"]):
        wh = weather_hist.copy()
        wh["date"] = pd.to_datetime(wh["date"]).dt.date.astype(str)
        wh = wh.set_index("date")
        sku["precip"] = pd.to_datetime(sku["date"]).dt.date.astype(str).map(wh["precip"]).fillna(0.0)
        sku["tmax"] = pd.to_datetime(sku["date"]).dt.date.astype(str).map(wh["tmax"]).fillna(0.0)
        sku["tmin"] = pd.to_datetime(sku["date"]).dt.date.astype(str).map(wh["tmin"]).fillna(0.0)
    else:
        sku["precip"] = 0.0
        sku["tmax"] = 0.0
        sku["tmin"] = 0.0

    # 疫情历史（可选CSV）：列 date,new_cases
    if epidemic_hist is not None and not epidemic_hist.empty and all(col in epidemic_hist.columns for col in ["date", "new_cases"]):
        eh = epidemic_hist.copy()
        eh["date"] = pd.to_datetime(eh["date"]).dt.date.astype(str)
        eh = eh.set_index("date")
        sku["epi_cases"] = pd.to_datetime(sku["date"]).dt.date.astype(str).map(eh["new_cases"]).fillna(0.0)
    else:
        sku["epi_cases"] = 0.0

    X = sku[[
        "discount", "coupon", "cycle_days", "day_of_week", "month",
        "is_holiday", "precip", "tmax", "tmin", "epi_cases"
    ]]
    y = sku["sales"].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    importance = list(zip(
        ["discount", "coupon", "cycle_days", "day_of_week", "month", "is_holiday", "precip", "tmax", "tmin", "epi_cases"],
        model.feature_importances_.tolist()
    ))
    importance = sorted(importance, key=lambda x: x[1], reverse=True)
    return model, mae, importance


def get_llm_client():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", None)
    # Proxy support via environment variables (HTTP_PROXY/HTTPS_PROXY)
    # OpenAI SDK respects standard env proxies; ensure they are set if needed.
    if not api_key:
        raise RuntimeError("环境变量 OPENAI_API_KEY 未设置")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)

def llm_decision_chat(messages: list[dict], context: str, model: str = "gpt-4o-mini"):
    client = get_llm_client()
    sys_prompt = (
        "你是供应链/需求预测决策助手，帮助用户讨论：\n"
        "- 用哪些数据/外部特征（节假日、天气、疫情、促销、周期）。\n"
        "- 选择哪些预测方法（统计基线、XGBoost、融合、是否使用LLM解释）。\n"
        "- 如何设置服务水平/前置期/审查周期。\n"
        "- 给出清晰步骤：数据检查→特征→模型→情景→输出与风险提示。\n"
        "请输出简洁的可执行建议，可包含检查清单或下一步行动。"
    )
    msg = [{"role": "system", "content": sys_prompt + "\n上下文:\n" + context}] + messages
    resp = client.chat.completions.create(
        model=model,
        messages=msg,
        temperature=0.3,
    )
    return resp.choices[0].message.content

def llm_generate_flowchart(messages: list[dict], context: str, model: str = "gpt-4o-mini"):
    client = get_llm_client()
    sys_prompt = (
        "你是一名架构制图助手。请基于上下文与对话，总结当前预测决策流程，"
        "并返回一个 Graphviz DOT 流程图（只返回代码，无额外说明）。"
        "要求：\n"
        "- 使用 digraph G { rankdir=LR; node [shape=box, style=rounded]; ... }\n"
        "- 包含：数据获取、外部特征、特征工程、所选模型(统计/ML/融合)、库存策略、生成式AI解释/问答、可视化/导出。\n"
        "- 节点名称尽量短；连线体现当前选择的处理路径。\n"
    )
    msg = [{"role": "system", "content": sys_prompt + "\n上下文:\n" + context}] + messages
    resp = client.chat.completions.create(
        model=model,
        messages=msg,
        temperature=0.2,
    )
    return resp.choices[0].message.content


SYSTEM_PROMPT = (
    "你是医药电商需求预测与供应链决策助手。"
    "给出短期预测(7天)与解释，考虑：服用周期、价格折扣、优惠券、节假日/事件。"
    "在输出中包含：预测表(每日销量)、关键影响因素、补货建议(安全库存/保质期)。"
    "保持简洁，给出风险提示。"
    "如需查询外部特征(节假日/天气)，请用一行指令提出：\n"
    "TOOL: fetch_holidays country=CN year=2025\n"
    "或\nTOOL: fetch_weather lat=39.9 lon=116.4 start=2025-12-14 days=7\n"
    "我们将执行工具并把结果回传再继续。"
)


def llm_forecast_and_advise(df: pd.DataFrame, sku_id: str, scenario: dict,
                           weather_hist: pd.DataFrame | None = None,
                           epidemic_hist: pd.DataFrame | None = None):
    try:
        recent, baseline_val = build_baseline(df, sku_id)
        if len(recent) == 0:
            raise ValueError("指定SKU无历史数据")
        # 构造未来7天场景
        last_date = isoparse(recent[-1]["date"]) if isinstance(recent[-1]["date"], str) else recent[-1]["date"]
        future_days = []
        for i in range(1, 8):
            d = (last_date + timedelta(days=i)).date().isoformat()
            # 应用场景：覆盖折扣/优惠券
            discount = scenario.get("discount", 0.0)
            coupon = scenario.get("coupon", 0)
            future_days.append({"date": d, "discount": discount, "coupon": coupon})

        # 训练统计模型与XGBoost以提供更真实的参考
        sku_hist = df[df["sku_id"] == sku_id].copy()
        sku_hist.sort_values("date", inplace=True)
        cycle_days = int(sku_hist["cycle_days"].iloc[-1]) if pd.notna(sku_hist["cycle_days"].iloc[-1]) else 30
        ts_res = fit_time_series_baseline(sku_hist["sales"].astype(float), cycle_days)
        # 预测未来7天的常规销量（不含促销）
        ts_forecast = []
        try:
            ts_vals = ts_res.get_forecast(steps=7).predicted_mean
            ts_forecast = [float(x) for x in ts_vals]
        except Exception:
            ts_forecast = [baseline_val] * 7

        country_code = scenario.get("country_code", "CN")
        lat = float(scenario.get("lat", 39.9))
        lon = float(scenario.get("lon", 116.4))
        start_date = last_date.date().isoformat() if hasattr(last_date, "date") else str(last_date)
        try:
            holidays = fetch_holidays(country_code, int(start_date[:4]))
            weather = fetch_weather_daily(lat, lon, start_date, days=7)
            epidemic = fetch_epidemic(country_code, start_date, days=7)
        except Exception:
            holidays, weather, epidemic = set(), [], []

        xgb_model, xgb_mae, feat_importance = train_xgb_with_promotions(
            df, sku_id, country_code=country_code,
            weather_hist=weather_hist, epidemic_hist=epidemic_hist
        )
        promo_forecast = []
        if xgb_model is not None:
            # 基于情景生成X输入，加入外部特征
            dow_start = pd.to_datetime(recent[-1]["date"]).weekday() if isinstance(recent[-1]["date"], str) else pd.to_datetime(str(recent[-1]["date"]).split(' ')[0]).weekday()
            # 为未来7天准备外部特征映射
            holiday_set_future = holidays if isinstance(holidays, set) else set(holidays) if holidays else set()
            weather_map = {w["date"]: w for w in weather} if weather else {}
            epi_map = {e["date"]: e for e in epidemic} if epidemic else {}
            for i in range(7):
                future_date = (last_date + timedelta(days=i)).date().isoformat()
                month = (last_date + timedelta(days=i)).month
                is_holiday = 1 if future_date in holiday_set_future else 0
                w = weather_map.get(future_date, {})
                precip = w.get("precip", 0.0) or 0.0
                tmax = w.get("tmax", 0.0) or 0.0
                tmin = w.get("tmin", 0.0) or 0.0
                epi_cases = epi_map.get(future_date, {}).get("new_cases", 0.0) or 0.0
                X_row = np.array([[
                    scenario.get("discount", 0.0), scenario.get("coupon", 0), cycle_days,
                    (dow_start + i) % 7, month, is_holiday, precip, tmax, tmin, epi_cases
                ]])
                promo_forecast.append(float(xgb_model.predict(X_row)[0]))
        else:
            promo_forecast = ts_forecast
            feat_importance = None

        # 融合预测：简单加权（可改为学习融合）
        fused = [0.6 * ts + 0.4 * promo for ts, promo in zip(ts_forecast, promo_forecast)]

        # 外部事件特征：节假日/天气风险对方差进行微调
        try:
            event_mult = compute_event_multipliers(future_days, holidays, weather, epidemic)
            # 用事件乘子对不确定性放大（均值保持）
        except Exception:
            event_mult = {fd["date"]: 1.0 for fd in future_days}

        # 安全库存与服务水平：正态近似
        # z值近似（常用服务水平：0.95->1.645, 0.99->2.33, 0.997->3.0）
        sl = float(scenario.get("service_level", 0.95))
        if sl >= 0.997:
            z = 3.0
        elif sl >= 0.99:
            z = 2.33
        elif sl >= 0.95:
            z = 1.645
        else:
            z = 1.282
        lead_time = int(scenario.get("lead_time", 7))
        review_period = int(scenario.get("review_period", 0))  # 0 => 连续补货
        on_hand = int(scenario.get("on_hand", 0))
        # 用历史残差估计sigma
        try:
            resid = sku_hist["sales"].astype(float) - ts_res.fittedvalues
            sigma = float(np.std(resid.dropna()))
        except Exception:
            sigma = max(1.0, float(np.std(sku_hist["sales"].astype(float))))

        # 更严谨：考虑促销引入的波动增量，对需求方差进行放大；周期性审查下考虑 L+R
        # 估计日方差
        sigma_day = max(1e-6, sigma)
        # 促销波动系数（根据未来情景近似）
        promo_multiplier = 1.0 + 0.3 * float(scenario.get("discount", 0.0)) + 0.15 * int(scenario.get("coupon", 0))
        # 需求期（连续补货用 L；周期性审查用 L+R）
        D_period = lead_time + max(0, review_period)
        # 需求均值（使用融合预测均值近似）
        demand_mean_period = float(np.mean(fused[:min(len(fused), max(1, lead_time))])) * D_period
        # 方差聚合（独立同分布近似 + 促销放大系数）
        # 将事件乘子整合（取未来期平均乘子）
        avg_event_multiplier = float(np.mean(list(event_mult.values()))) if event_mult else 1.0
        sigma_period = sigma_day * np.sqrt(D_period) * promo_multiplier * avg_event_multiplier
        safety_stock = max(0, int(round(z * sigma_period)))
        reorder_point = int(round(demand_mean_period + safety_stock))
        # 建议订货量（周期性审查下按目标库存位计算）
        target_level = reorder_point
        order_qty = max(0, target_level - on_hand)

        # 组装上下文
        context = {
            "sku_id": sku_id,
            "baseline_recent": recent,
            "baseline_value": baseline_val,
            "future_days": future_days,
            "notes": scenario.get("notes", ""),
            "ts_forecast": ts_forecast,
            "xgb_mae": xgb_mae if xgb_model is not None else None,
            "fused_forecast": fused,
            "safety_stock": safety_stock,
            "reorder_point": reorder_point,
            "order_qty": order_qty,
            "service_level": sl,
            "lead_time": lead_time,
            "review_period": review_period,
            "on_hand": on_hand,
            "event_multiplier": avg_event_multiplier,
            "xgb_feature_importance": feat_importance,
        }
    except Exception as e:
        return f"预测管道失败: {e}", None

    user_prompt = (
        "请基于以下上下文进行7日销量预测，并给出解释与补货建议：\n"
        f"SKU: {sku_id}\n"
        f"近期基线值(移动平均融合): {baseline_val:.2f}\n"
        f"历史片段(近30天): {recent}\n"
        f"未来7天情景(折扣/优惠券): {future_days}\n"
        f"备注/事件: {scenario.get('notes', '')}\n"
        "请返回：\n- 每日预测销量(日期与数值，包含统计基线与融合值)\n- 关键影响因素(折扣力度、优惠券、周期性)\n- 补货建议(安全库存、安全服务水平、补货前置期与建议订货量，考虑保质期/冷链；若审查周期>0，按周期性审查策略说明)\n- 风险提示(促销峰值、断货风险)\n"
    )

    client = get_llm_client()
    # 使用最新兼容模型名称占位，实际以环境为准（在README中说明）
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt + "\n参考数据:" + str(context)},
        ],
        temperature=0.3,
    )
    content = resp.choices[0].message.content
    # 简易工具调用解析：检测以TOOL:开头的一行
    lines = (content or "").splitlines()
    tool_line = None
    for ln in lines:
        if ln.strip().startswith("TOOL:"):
            tool_line = ln.strip()
            break
    if tool_line:
        # 解析工具指令
        try:
            # 示例：TOOL: fetch_holidays country=CN year=2025
            parts = tool_line.replace("TOOL:", "").strip().split()
            tool_name = parts[0]
            args = {k: v for k, v in [p.split("=") for p in parts[1:] if "=" in p]}
            tool_result = {}
            if tool_name == "fetch_holidays":
                cc = args.get("country", "CN")
                year = int(args.get("year", datetime.now().year))
                tool_result = {"holidays": list(fetch_holidays(cc, year))}
            elif tool_name == "fetch_weather":
                lat = float(args.get("lat", 39.9))
                lon = float(args.get("lon", 116.4))
                start = args.get("start", datetime.now().date().isoformat())
                days = int(args.get("days", 7))
                tool_result = {"weather": fetch_weather_daily(lat, lon, start, days)}
            elif tool_name == "fetch_epidemic":
                cc = args.get("country", "CN")
                start = args.get("start", datetime.now().date().isoformat())
                days = int(args.get("days", 7))
                tool_result = {"epidemic": fetch_epidemic(cc, start, days)}
            # 二次调用，附加工具结果
            follow_up = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt + "\n参考数据:" + str(context)},
                    {"role": "assistant", "content": content},
                    {"role": "tool", "content": str(tool_result)},
                ],
                temperature=0.3,
                )
            return follow_up.choices[0].message.content, feat_importance
        except Exception:
            # 若工具调用失败，返回原回答
            return content, feat_importance
    # 未请求工具时，直接返回首轮回答
    return content, feat_importance


def generate_numeric_forecast(df: pd.DataFrame, sku_id: str, scenario: dict,
                              weather_hist: pd.DataFrame | None = None,
                              epidemic_hist: pd.DataFrame | None = None,
                              baseline_only: bool = False):
    """生成未来7日数值预测与库存策略，不调用LLM。
    baseline_only=True 时跳过促销ML与外部特征，只用统计基线。
    返回 (rows, metrics)。
    rows: [{date, ts_forecast, promo_forecast, fused_forecast}...]
    metrics: {safety_stock, reorder_point, order_qty, service_level, lead_time, review_period, on_hand}
    """
    sku_hist = df[df["sku_id"] == sku_id].copy()
    if sku_hist.empty:
        raise ValueError("指定SKU无历史数据")
    sku_hist.sort_values("date", inplace=True)

    cycle_days = int(sku_hist["cycle_days"].iloc[-1]) if pd.notna(sku_hist["cycle_days"].iloc[-1]) else 30
    ts_res = fit_time_series_baseline(sku_hist["sales"].astype(float), cycle_days)
    try:
        ts_vals = ts_res.get_forecast(steps=7).predicted_mean
        ts_forecast = [float(x) for x in ts_vals]
    except Exception:
        # 回退到简单均值
        ts_forecast = [float(sku_hist["sales"].astype(float).tail(7).mean())] * 7

    country_code = scenario.get("country_code", "CN")
    lat = float(scenario.get("lat", 39.9))
    lon = float(scenario.get("lon", 116.4))
    last_date = pd.to_datetime(sku_hist["date"].iloc[-1])
    start_date = last_date.date().isoformat()
    try:
        holidays = fetch_holidays(country_code, int(start_date[:4]))
        weather = fetch_weather_daily(lat, lon, start_date, days=7)
        epidemic = fetch_epidemic(country_code, start_date, days=7)
    except Exception:
        holidays, weather, epidemic = set(), [], []

    promo_forecast = []
    if baseline_only:
        promo_forecast = ts_forecast
    else:
        xgb_model, _, _ = train_xgb_with_promotions(
            df, sku_id, country_code=country_code,
            weather_hist=weather_hist, epidemic_hist=epidemic_hist
        )
        if xgb_model is not None:
            dow_start = last_date.weekday()
            holiday_set_future = holidays if isinstance(holidays, set) else set(holidays) if holidays else set()
            weather_map = {w["date"]: w for w in weather} if weather else {}
            epi_map = {e["date"]: e for e in epidemic} if epidemic else {}
            for i in range(7):
                future_date = (last_date + timedelta(days=i+1)).date().isoformat()
                month = (last_date + timedelta(days=i+1)).month
                is_holiday = 1 if future_date in holiday_set_future else 0
                w = weather_map.get(future_date, {})
                precip = w.get("precip", 0.0) or 0.0
                tmax = w.get("tmax", 0.0) or 0.0
                tmin = w.get("tmin", 0.0) or 0.0
                epi_cases = epi_map.get(future_date, {}).get("new_cases", 0.0) or 0.0
                X_row = np.array([[
                    scenario.get("discount", 0.0), scenario.get("coupon", 0), cycle_days,
                    (dow_start + i + 1) % 7, month, is_holiday, precip, tmax, tmin, epi_cases
                ]])
                promo_forecast.append(float(xgb_model.predict(X_row)[0]))
        else:
            promo_forecast = ts_forecast

    fused = [0.6 * ts + 0.4 * promo for ts, promo in zip(ts_forecast, promo_forecast)]

    # 安全库存与策略
    sl = float(scenario.get("service_level", 0.95))
    if sl >= 0.997:
        z = 3.0
    elif sl >= 0.99:
        z = 2.33
    elif sl >= 0.95:
        z = 1.645
    else:
        z = 1.282
    lead_time = int(scenario.get("lead_time", 7))
    review_period = int(scenario.get("review_period", 0))
    on_hand = int(scenario.get("on_hand", 0))
    try:
        resid = sku_hist["sales"].astype(float) - ts_res.fittedvalues
        sigma = float(np.std(resid.dropna()))
    except Exception:
        sigma = max(1.0, float(np.std(sku_hist["sales"].astype(float))))
    sigma_day = max(1e-6, sigma)
    promo_multiplier = 1.0 + 0.3 * float(scenario.get("discount", 0.0)) + 0.15 * int(scenario.get("coupon", 0))
    D_period = lead_time + max(0, review_period)
    demand_mean_period = float(np.mean(fused[:min(len(fused), max(1, lead_time))])) * D_period
    avg_event_multiplier = 1.0
    try:
        future_days = [{"date": (last_date + timedelta(days=i+1)).date().isoformat()} for i in range(7)]
        event_mult = compute_event_multipliers(future_days, holidays, weather, epidemic)
        avg_event_multiplier = float(np.mean(list(event_mult.values()))) if event_mult else 1.0
    except Exception:
        pass
    sigma_period = sigma_day * np.sqrt(D_period) * promo_multiplier * avg_event_multiplier
    safety_stock = max(0, int(round(z * sigma_period)))
    reorder_point = int(round(demand_mean_period + safety_stock))
    order_qty = max(0, reorder_point - on_hand)

    rows = []
    for i in range(7):
        d = (last_date + timedelta(days=i+1)).date().isoformat()
        rows.append({
            "date": d,
            "ts_forecast": float(ts_forecast[i]),
            "promo_forecast": float(promo_forecast[i]),
            "fused_forecast": float(fused[i]),
        })
    metrics = {
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "order_qty": order_qty,
        "service_level": sl,
        "lead_time": lead_time,
        "review_period": review_period,
        "on_hand": on_hand,
    }
    return rows, metrics


def load_sales_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 规范字段类型
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    df["coupon"] = df["coupon"].astype(int)
    df["cycle_days"] = df["cycle_days"].astype(int)
    # 添加折扣特征
    df["discount"] = df.apply(compute_discount, axis=1)
    return df
