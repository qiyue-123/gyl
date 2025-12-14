import os
import streamlit as st
import logging
import pandas as pd
from ai_backend import load_sales_csv, llm_forecast_and_advise, generate_numeric_forecast, llm_decision_chat, llm_generate_flowchart
from external_features import fetch_holidays, fetch_weather_daily


st.set_page_config(page_title="医药电商需求预测助手", layout="wide")
# 运行期日志到控制台
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("med-forecast")
st.title("医药电商需求预测助手（LLM 智能决策/问答）")

with st.sidebar:
    st.header("配置")
    api_key_ok = bool(os.getenv("OPENAI_API_KEY"))
    st.write("API Key 已设置" if api_key_ok else "请设置 OPENAI_API_KEY 环境变量")
    default_path = "data/sample_sales.csv"
    data_file = st.file_uploader("上传销售CSV（参见 README 数据格式）", type=["csv"])
    use_sample = st.checkbox("使用示例数据", value=True)

    st.subheader("外部事件特征输入")
    country_code = st.text_input("国家代码(节假日)", "CN")
    lat = st.number_input("纬度(天气)", value=39.9, help="例如北京约39.9")
    lon = st.number_input("经度(天气)", value=116.4, help="例如北京约116.4")

    st.subheader("可选：历史外部特征CSV")
    weather_file = st.file_uploader("历史天气CSV (date,tmax,tmin,precip)", type=["csv"], key="weather")
    epi_file = st.file_uploader("历史疫情CSV (date,new_cases)", type=["csv"], key="epi")

# 读取销售数据
try:
    if use_sample:
        df = load_sales_csv(default_path)
    else:
        if data_file is None:
            st.warning("未选择文件，已回退使用示例数据")
            df = load_sales_csv(default_path)
        else:
            df = load_sales_csv(data_file)
except Exception as e:
    msg = f"运行出错：{e}"
    st.error(msg)
    logger.exception(msg)
    st.stop()

# 读取可选外部历史特征
weather_hist = None
epidemic_hist = None
if 'weather_file' in locals() and weather_file is not None:
    try:
        weather_hist = pd.read_csv(weather_file)
    except Exception as e:
        st.warning(f"历史天气CSV读取失败: {e}")
if 'epi_file' in locals() and epi_file is not None:
    try:
        epidemic_hist = pd.read_csv(epi_file)
    except Exception as e:
        st.warning(f"历史疫情CSV读取失败: {e}")

st.subheader("数据预览")
st.dataframe(df.head(20))

# SKU 选择
sku_list = sorted(df["sku_id"].unique().tolist())
sku_id = st.selectbox("选择SKU", sku_list)

# 场景参数
col1, col2, col3 = st.columns(3)
with col1:
    discount = st.slider("未来7天折扣力度(0-1)", 0.0, 1.0, 0.1, 0.01)
with col2:
    coupon = st.selectbox("未来7天是否有优惠券", [0, 1], index=0)
with col3:
    notes = st.text_input("备注/事件（节假日/疫情/促销主题）", "---")

col4, col5, col6 = st.columns(3)
with col4:
    lead_time = st.number_input("补货前置期(天)", min_value=1, max_value=60, value=7)
with col5:
    service_level = st.slider("服务水平(满足率)", 0.90, 0.999, 0.95)
with col6:
    on_hand = st.number_input("当前库存(件)", min_value=0, value=200)

col7, _ , _ = st.columns(3)
with col7:
    review_period = st.number_input("补货审查周期(天)", min_value=0, max_value=60, value=7,
                                   help="周期性审查与下单；0表示连续补货策略(R,Q)")

scenario = {
    "discount": float(discount),
    "coupon": int(coupon),
    "notes": notes,
    "lead_time": int(lead_time),
    "service_level": float(service_level),
    "on_hand": int(on_hand),
    "review_period": int(review_period),
    "country_code": country_code,
    "lat": float(lat),
    "lon": float(lon),
}

run = st.button("生成预测与建议")

if run:
    try:
        # 获取未来7天外部事件特征（用于页面参考展示）
        sku_dates = pd.to_datetime(df[df["sku_id"] == sku_id]["date"]).dt.date
        last_date = max(sku_dates) if len(sku_dates) else pd.to_datetime("today").date()
        future_start = pd.to_datetime(last_date).date().isoformat()
        holidays = fetch_holidays(country_code, pd.to_datetime(future_start).year)
        weather = fetch_weather_daily(float(lat), float(lon), future_start, days=7)

        res = llm_forecast_and_advise(
            df, sku_id, scenario,
            weather_hist=weather_hist,
            epidemic_hist=epidemic_hist,
        )
        if res is None:
            st.error("预测与建议生成失败")
            st.stop()
        result, feat_importance = res

        st.subheader("预测与建议（生成式AI）")
        st.write(result)
        try:
            logger.info(f"Forecast generated for SKU={sku_id}")
            print("[Forecast Output]", result)
        except Exception:
            pass
        if isinstance(result, str) and ("失败" in result or "error" in result.lower()):
            st.error("预测生成遇到问题，已在控制台与页面显示详情。")
            logger.error(f"Forecast error detail: {result}")

        st.subheader("外部事件特征(参考)")
        st.write({"holidays_count": len(holidays), "weather_preview": weather[:3]})

        if feat_importance is not None:
            st.subheader("XGBoost 特征重要性")
            st.dataframe(pd.DataFrame(feat_importance, columns=["feature", "importance"]))
    except Exception as e:
        st.error(f"运行出错：{e}")

# 情景对比与导出
st.subheader("情景对比（数值预测，不调用LLM）")
with st.expander("配置两种情景并比较未来7日融合预测与订货建议"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**情景A**")
        a_discount = st.slider("A 折扣", 0.0, 1.0, float(discount), 0.01, key="a_discount")
        a_coupon = st.selectbox("A 优惠券", [0, 1], index=int(coupon), key="a_coupon")
    with c2:
        st.markdown("**情景B**")
        b_discount = st.slider("B 折扣", 0.0, 1.0, 0.2, 0.01, key="b_discount")
        b_coupon = st.selectbox("B 优惠券", [0, 1], index=0, key="b_coupon")

    compare = st.button("生成情景A/B对比")
    if compare:
        try:
            scenario_a = dict(scenario)
            scenario_a.update({"discount": float(a_discount), "coupon": int(a_coupon)})
            scenario_b = dict(scenario)
            scenario_b.update({"discount": float(b_discount), "coupon": int(b_coupon)})
            rows_a, metrics_a = generate_numeric_forecast(df, sku_id, scenario_a, weather_hist, epidemic_hist)
            rows_b, metrics_b = generate_numeric_forecast(df, sku_id, scenario_b, weather_hist, epidemic_hist)
            df_a = pd.DataFrame(rows_a)
            df_b = pd.DataFrame(rows_b)
            df_join = df_a.merge(df_b, on="date", suffixes=("_A", "_B"))
            st.dataframe(df_join)
            st.write({"metrics_A": metrics_a, "metrics_B": metrics_b})

            # 可视化曲线对比（融合预测）
            try:
                chart_df = df_join[["date", "fused_forecast_A", "fused_forecast_B"]]
                chart_df = chart_df.set_index("date")
                st.line_chart(chart_df)
            except Exception:
                pass

            # 导出CSV
            csv_bytes = df_join.to_csv(index=False).encode("utf-8")
            st.download_button("下载对比CSV", data=csv_bytes, file_name=f"compare_{sku_id}.csv", mime="text/csv")
        except Exception as e:
            st.error(f"情景对比出错：{e}")

st.subheader("情景网格与热力图（数值预测）")
with st.expander("批量评估折扣×优惠券组合，查看融合预测均值与订货建议差异"):
    grid_discounts = st.multiselect("选择折扣网格", options=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4], default=[0.0, 0.1, 0.2])
    grid_coupons = st.multiselect("选择优惠券网格", options=[0, 1], default=[0, 1])
    run_grid = st.button("生成网格热力图")
    if run_grid:
        try:
            results = []
            for d in grid_discounts:
                for c in grid_coupons:
                    sc = dict(scenario)
                    sc.update({"discount": float(d), "coupon": int(c)})
                    rows_g, metrics_g = generate_numeric_forecast(df, sku_id, sc, weather_hist, epidemic_hist)
                    fused_mean = float(pd.DataFrame(rows_g)["fused_forecast"].mean())
                    results.append({"discount": d, "coupon": c, "fused_mean": fused_mean, "order_qty": metrics_g["order_qty"]})
            grid_df = pd.DataFrame(results)
            st.dataframe(grid_df)
            if not grid_df.empty:
                pivot = grid_df.pivot(index="discount", columns="coupon", values="fused_mean")
                st.caption("融合预测均值热力图")
                st.dataframe(pivot.style.background_gradient(cmap="Blues"))
        except Exception as e:
            st.error(f"网格生成出错：{e}")

st.subheader("导出 LLM 预测结果为 Markdown")
st.caption("运行上方按钮后可在此导出最近一次预测文本")
if 'result' in locals() and isinstance(result, str):
    st.download_button("下载预测Markdown", data=result.encode("utf-8"), file_name=f"forecast_{sku_id}.md", mime="text/markdown")

st.subheader("一键导出报告 (Markdown)")
st.caption("包含数值预测、库存策略、情景对比与LLM输出摘要")
if 'result' in locals() and isinstance(result, str):
    try:
        # 准备最新数值预测
        rows_num, metrics_num = generate_numeric_forecast(df, sku_id, scenario, weather_hist, epidemic_hist)
        num_df = pd.DataFrame(rows_num)
        report_parts = []
        report_parts.append(f"# 预测与决策报告 - SKU {sku_id}\n")
        report_parts.append("## 场景参数\n" + pd.Series(scenario).to_string())
        report_parts.append("## 数值预测 (融合)\n" + num_df.to_markdown(index=False))
        report_parts.append("## 库存策略\n" + pd.Series(metrics_num).to_string())
        report_parts.append("## 生成式AI 输出\n" + result)
        md_report = "\n\n".join(report_parts)
        st.download_button("下载完整报告", data=md_report.encode("utf-8"), file_name=f"report_{sku_id}.md", mime="text/markdown")
    except Exception as e:
        st.warning(f"生成报告失败：{e}")

st.subheader("智能决策 / 问答（与LLM逐步协同）")
st.caption("与助手讨论：使用哪些数据、预测方法、影响因子，何时让 LLM 生成解释/建议，再逐步完成预测。")

# 构造上下文供对话参考
context_parts = []
context_parts.append(f"可用SKU: {len(sku_list)} 个，当前选择: {sku_id}")
context_parts.append(f"场景参数: {scenario}")
context_parts.append(f"数据列: {list(df.columns)}")
context_parts.append("可选方法: 统计基线(SARIMA/ARIMA), 促销ML(XGBoost), 融合, LLM解释/问答")
context_parts.append("外部特征: 节假日、天气、疫情；可上传历史或在线获取未来7天")
decision_context = "\n".join(context_parts)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "你好，我可以帮你讨论本次需求预测要用的数据、方法和外部特征。你想先确认哪些因子？"}
    ]

chat_container = st.container()
for m in st.session_state.chat_messages:
    with chat_container:
        st.markdown(f"**{'用户' if m['role']=='user' else '助手'}：** {m['content']}")

user_chat = st.text_input("输入与助手对话的问题或决策意图", key="chat_input")
col_chat1, col_chat2 = st.columns([1,1])
if "chat_busy" not in st.session_state:
    st.session_state.chat_busy = False
with col_chat1:
    send_chat = st.button("发送并协同决策", disabled=st.session_state.chat_busy)
with col_chat2:
    clear_chat = st.button("清空对话")

run_from_chat = st.button("根据对话执行预测")

if clear_chat:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "对话已重置。请告诉我本次要确认的数据和方法。"}
    ]

if send_chat and user_chat.strip():
    st.session_state.chat_busy = True
    st.session_state.chat_messages.append({"role": "user", "content": user_chat.strip()})
    try:
        assistant_reply = llm_decision_chat(st.session_state.chat_messages, decision_context)
    except Exception as e:
        assistant_reply = f"调用LLM失败：{e}"
    st.session_state.chat_messages.append({"role": "assistant", "content": assistant_reply})
    st.session_state.chat_busy = False
    st.rerun()

if run_from_chat:
    try:
        scenario_chat = dict(scenario)
        rows_chat, metrics_chat = generate_numeric_forecast(
            df, sku_id, scenario_chat, weather_hist, epidemic_hist, baseline_only=False
        )
        st.markdown("**对话触发的数值预测**")
        st.dataframe(pd.DataFrame(rows_chat))
        st.write(metrics_chat)
        # 将结果写回对话摘要
        summary = f"已执行融合预测，建议订货量 {metrics_chat.get('order_qty')}，安全库存 {metrics_chat.get('safety_stock')}。"
        st.session_state.chat_messages.append({"role": "assistant", "content": summary})
    except Exception as e:
        st.session_state.chat_messages.append({"role": "assistant", "content": f"执行预测失败：{e}"})
    st.rerun()

# 动态生成流程图（由LLM辅助）
st.subheader("根据对话动态生成流程图")
st.caption("由 LLM 结合上下文与对话生成 Graphviz 流程图，展示当前决策路径。")
gen_flow = st.button("生成流程图（LLM）")
if gen_flow:
    try:
        dot = llm_generate_flowchart(st.session_state.chat_messages, decision_context)
        st.graphviz_chart(dot)
    except Exception as e:
        st.warning(f"生成流程图失败：{e}")

# 预测全流程可视化（一步步对比一般预测与生成式AI）
st.subheader("预测流程可视化")
st.caption("对比统计基线、促销ML与融合预测的数值曲线；保留生成式AI的对话输出以体现解释与补位。")
flow_chart = r"""
digraph G {
    rankdir=LR;
    node [shape=box, style=rounded];
    sales [label="销售数据\nCSV/上传"];
    external [label="外部特征\n节假日/天气/疫情"];
    features [label="特征工程\n折扣/优惠券/周期/外部特征"];
    sarima [label="统计基线\nSARIMA/ARIMA"];
    xgb [label="促销ML\nXGBoost"];
    fuse [label="融合预测\n0.6*ts + 0.4*ml"];
    inv [label="库存策略\n安全库存/订货点/订货量"];
    llm [label="生成式AI\n解释/问答/风险提示"];
    viz [label="可视化与导出\n曲线/Markdown/CSV"];
    sales -> features;
    external -> features;
    features -> sarima;
    features -> xgb;
    sarima -> fuse;
    xgb -> fuse;
    fuse -> inv;
    fuse -> llm;
    inv -> llm;
    fuse -> viz;
    llm -> viz;
}
"""
st.graphviz_chart(flow_chart)
try:
    rows_num, metrics_num = generate_numeric_forecast(df, sku_id, scenario, weather_hist, epidemic_hist)
    num_df = pd.DataFrame(rows_num)
    st.markdown("**数值预测表**")
    st.dataframe(num_df)
    # 曲线：统计基线 vs 促销ML vs 融合
    plot_df = num_df.set_index("date")[["ts_forecast", "promo_forecast", "fused_forecast"]]
    st.line_chart(plot_df)
    st.markdown("**库存策略指标(数值)**")
    st.write(metrics_num)
    # 生成式AI对话输出回显（若已生成）
    if 'result' in locals() and isinstance(result, str):
        st.markdown("**生成式AI输出(回显)**")
        st.write(result)
except Exception as e:
    st.warning(f"流程可视化生成失败：{e}")
