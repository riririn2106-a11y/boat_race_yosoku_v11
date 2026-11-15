import streamlit as st
import pandas as pd
import itertools
import json
from PIL import Image
from google import genai
from google.genai import types
from pydantic import BaseModel

STADIUM_PROFILES = {
    "大村": {"in_base": 65.0, "local_weight": 1.0}, "徳山": {"in_base": 62.0, "local_weight": 1.0},
    "芦屋": {"in_base": 61.0, "local_weight": 1.0}, "下関": {"in_base": 60.0, "local_weight": 1.1},
    "若松": {"in_base": 58.0, "local_weight": 1.1}, "宮島": {"in_base": 57.0, "local_weight": 1.2},
    "丸亀": {"in_base": 56.0, "local_weight": 1.1}, "児島": {"in_base": 56.0, "local_weight": 1.1},
    "多摩川": {"in_base": 55.0, "local_weight": 1.0}, "尼崎": {"in_base": 55.0, "local_weight": 1.0},
    "津": {"in_base": 54.0, "local_weight": 1.1}, "住之江": {"in_base": 54.0, "local_weight": 1.1},
    "蒲郡": {"in_base": 53.0, "local_weight": 1.1}, "常滑": {"in_base": 53.0, "local_weight": 1.1},
    "三国": {"in_base": 53.0, "local_weight": 1.2}, "びわこ": {"in_base": 51.0, "local_weight": 1.5},
    "浜名湖": {"in_base": 51.0, "local_weight": 1.2}, "桐生": {"in_base": 50.0, "local_weight": 1.3},
    "鳴門": {"in_base": 48.0, "local_weight": 1.5}, "唐津": {"in_base": 51.0, "local_weight": 1.1},
    "福岡": {"in_base": 45.0, "local_weight": 2.5}, "平和島": {"in_base": 44.0, "local_weight": 2.0},
    "戸田": {"in_base": 43.0, "local_weight": 2.2}, "江戸川": {"in_base": 42.0, "local_weight": 3.0},
}

FRAME_BASE_PROB = {
    1: {"win_1st": 0.556, "win_2nd": 0.180, "win_3rd": 0.100},
    2: {"win_1st": 0.148, "win_2nd": 0.250, "win_3rd": 0.185},
    3: {"win_1st": 0.132, "win_2nd": 0.200, "win_3rd": 0.205},
    4: {"win_1st": 0.090, "win_2nd": 0.170, "win_3rd": 0.170},
    5: {"win_1st": 0.060, "win_2nd": 0.120, "win_3rd": 0.175},
    6: {"win_1st": 0.018, "win_2nd": 0.080, "win_3rd": 0.155},
}

DEVELOPMENT_BIAS = {
    1: {2: 1.2, 3: 1.1, 4: 0.9, 5: 0.8, 6: 0.7},
    2: {1: 1.3, 3: 1.0, 4: 1.0, 5: 0.8, 6: 0.7},
    3: {1: 1.1, 2: 1.1, 4: 1.1, 5: 0.9, 6: 0.7},
    4: {1: 0.7, 2: 0.6, 3: 0.7, 5: 1.4, 6: 1.3},
    5: {1: 0.8, 2: 0.7, 4: 1.1, 6: 1.2},
    6: {1: 0.9, 2: 0.8, 4: 1.0, 5: 1.1},
}

WEIGHTS = {
    "win_rate_national": 0.4835, "rate_national": -0.9146, "win_rate_local": 0.2205,
    "rate_local": 1.8718, "motor": 0.5000, "st_relative": 1.1513,
    "raw_win_rate_add": 4.3199,
    "course_1st_weight": 0.65, "course_2nd_weight": 0.50, "course_3rd_weight": 0.40
}
CLASS_RELIABILITY = {"A1": 1.15, "A2": 1.00, "B1": 0.85, "B2": 0.50}

class BaseBoatInfo(BaseModel):
    boat_no: int
    racer_name: str
    class_type: str
    win_rate_national: float
    rate_national: float
    win_rate_local: float
    rate_local: float
    motor_rate: float
    c1_rates: list[float]
    c2_rates: list[float]
    c3_rates: list[float]
    c4_rates: list[float]
    c5_rates: list[float]
    c6_rates: list[float]

class SingleRaceBase(BaseModel):
    race_no: int
    boats: list[BaseBoatInfo]

class AllRacesBaseData(BaseModel):
    stadium_name: str
    races: list[SingleRaceBase]

class LiveBoatInfo(BaseModel):
    boat_no: int
    exhibition_st: float
    tilt: float
    straight_time: float
    turn_time: float
    inspect_time: float
    target_course: int
    course_1st_rate: float
    course_2nd_rate: float
    course_3rd_rate: float
    makuri_rate: float
    sashi_rate: float
    is_flying: bool

class LiveRaceData(BaseModel):
    boats: list[LiveBoatInfo]

def calculate_deviation_diff(series, reverse=False):
    values = pd.Series(series, copy=False).astype(float)
    if values.empty: return pd.Series(dtype=float)
    mean = values.mean()
    std = values.std(ddof=0) + 1e-5
    factor = -1.0 if reverse else 1.0
    return (10 * (values - mean) / std * factor).clip(lower=-25.0, upper=25.0)

def predict_race_v9(race_data, stadium_name, profiles):
    df = race_data.copy()
    profile = profiles.get(stadium_name, {"in_base": 55.0, "local_weight": 1.0})
    
    in_scale = profile["in_base"] / 55.0
    df["base_1st"] = df["target_course"].map(lambda x: FRAME_BASE_PROB[x]["win_1st"])
    df["base_2nd"] = df["target_course"].map(lambda x: FRAME_BASE_PROB[x]["win_2nd"])
    df["base_3rd"] = df["target_course"].map(lambda x: FRAME_BASE_PROB[x]["win_3rd"])
    
    orig_1st = df.loc[df["target_course"] == 1, "base_1st"].values
    if len(orig_1st) > 0:
        new_1st = orig_1st * in_scale
        df.loc[df["target_course"] == 1, "base_1st"] = new_1st
        remain_weight = (1.0 - new_1st) / (1.0 - orig_1st) if orig_1st != 1.0 else 1.0
        df.loc[df["target_course"] != 1, "base_1st"] *= remain_weight

    df["diff_win_national"] = calculate_deviation_diff(df["win_rate_national"])
    df["diff_rate_national"] = calculate_deviation_diff(df["2rate_national"])
    w_national = WEIGHTS["win_rate_national"] * (0.5 if profile["local_weight"] >= 2.0 else 1.0)
    w_local = WEIGHTS["win_rate_local"] * profile["local_weight"]
    
    df["diff_win_local"] = calculate_deviation_diff(df["win_rate_local"])
    df["diff_rate_local"] = calculate_deviation_diff(df["2rate_local"])
    df["diff_motor"] = calculate_deviation_diff(df["motor_rate"])
    
    adjusted_st = df["exhibition_st"].copy().astype(float)
    adjusted_st[df["is_flying"] == True] += 0.03
    df["st_relative_base"] = (adjusted_st.mean() - adjusted_st) / 0.01
    
    min_straight = df["straight_time"].astype(float).min()
    min_turn = df["turn_time"].astype(float).min()
    min_inspect = df["inspect_time"].astype(float).min()
    
    df["gap_straight"] = df["straight_time"].astype(float) - min_straight
    df["gap_turn"] = df["turn_time"].astype(float) - min_turn
    df["gap_inspect"] = df["inspect_time"].astype(float) - min_inspect
    
    df["diff_straight"] = calculate_deviation_diff(df["straight_time"], reverse=True)
    df["diff_turn"] = calculate_deviation_diff(df["turn_time"], reverse=True)
    df["diff_inspect"] = calculate_deviation_diff(df["inspect_time"], reverse=True)
    
    df["spec_nobi"] = (df["diff_straight"] + df["diff_inspect"]) / 2.0
    df["spec_deashi"] = df["diff_turn"]
    
    df.loc[df["gap_straight"] == 0.0, "spec_nobi"] += 3.0
    df.loc[df["gap_turn"] == 0.0, "spec_deashi"] += 3.0
    
    df["ex_score"] = 0.0
    in_courses = (1, 2, 3)
    out_courses = (4, 5, 6)
    df.loc[df["target_course"].isin(in_courses), "ex_score"] = (df["spec_deashi"] * 1.4) + (df["spec_nobi"] * 1.1)
    df.loc[df["target_course"].isin(out_courses), "ex_score"] = (df["spec_nobi"] * 1.5) + (df["spec_deashi"] * 1.0)
    
    df["trust_factor"] = df["class_type"].map(CLASS_RELIABILITY).fillna(0.85)
    
    df["base_score"] = (
        df["diff_win_national"] * w_national + df["diff_rate_national"] * WEIGHTS["rate_national"]
        + df["diff_win_local"] * w_local + df["diff_rate_local"] * WEIGHTS["rate_local"]
        + df["diff_motor"] * WEIGHTS["motor"] + df["st_relative_base"] * WEIGHTS["st_relative"]
        + df["win_rate_national"] * WEIGHTS["raw_win_rate_add"] + df["ex_score"]
    )
    df["pure_diff_score"] = df["base_score"] * df["trust_factor"]
    df.loc[df["is_flying"] == True, "pure_diff_score"] -= 10.0
    
    f_1st = 1.0 + df["pure_diff_score"] * 0.015 + (df["course_1st_rate"] * 0.01 * WEIGHTS["course_1st_weight"])
    f_2nd = 1.0 + df["pure_diff_score"] * 0.015 + (df["course_2nd_rate"] * 0.01 * WEIGHTS["course_2nd_weight"])
    f_3rd = 1.0 + df["pure_diff_score"] * 0.015 + (df["course_3rd_rate"] * 0.01 * WEIGHTS["course_3rd_weight"])
    
    df["pred_1st_prob"] = (df["base_1st"] * f_1st).clip(lower=0)
    df["pred_1st_prob"] /= df["pred_1st_prob"].sum()
    df["pred_2nd_prob"] = (df["base_2nd"] * f_2nd).clip(lower=0, upper=1.0)
    df["pred_3rd_prob"] = (df["base_3rd"] * f_3rd).clip(lower=0, upper=1.0)
    return df

def generate_custom_bets_v9(res_df, conf_rank):
    prob_1st = res_df.set_index("boat_no")["pred_1st_prob"].to_dict()
    prob_2nd = res_df.set_index("boat_no")["pred_2nd_prob"].to_dict()
    prob_3rd = res_df.set_index("boat_no")["pred_3rd_prob"].to_dict()
    
    makuri_dict = res_df.set_index("boat_no")["makuri_rate"].to_dict()
    sashi_dict = res_df.set_index("boat_no")["sashi_rate"].to_dict()
    
    trifecta_list = []
    boats = (1, 2, 3, 4, 5, 6)
    for p1, p2, p3 in itertools.permutations(boats, 3):
        bias_map = DEVELOPMENT_BIAS.get(p1, {})
        rem_sum_2nd = sum(prob_2nd[b] * bias_map.get(b, 1.0) for b in boats if b != p1)
        p2_cond = (prob_2nd[p2] * bias_map.get(p2, 1.0)) / rem_sum_2nd if rem_sum_2nd > 0 else 0.2
        rem_sum_3rd = sum(prob_3rd[b] * bias_map.get(b, 1.0) for b in boats if b not in (p1, p2))
        p3_cond = (prob_3rd[p3] * bias_map.get(p3, 1.0)) / rem_sum_3rd if rem_sum_3rd > 0 else 0.2
        
        raw_prob = prob_1st[p1] * p2_cond * p3_cond
        trifecta_list.append({"買い目": f"{p1}-{p2}-{p3}", "prob": raw_prob})
        
    df_tri = pd.DataFrame(trifecta_list)
    df_tri["確率"] = df_tri["prob"] / df_tri["prob"].sum()
    
    def estimate_market_pop(comb):
        parts = comb.split("-")
        p1 = int(parts[0])
        p2 = int(parts[1])
        score = 1.5 if p1 == 1 else (0.8 if p1 in (2, 3, 4) else 0.4)
        if p2 in (5, 6):
            score *= 0.8
        return score

    df_tri["market_pop"] = df_tri["買い目"].apply(estimate_market_pop)
    df_tri["妙味度"] = df_tri["確率"] / df_tri["market_pop"]
    
    final_bets = []
    report_data = []
    
    if conf_rank in ("D", "E"):
        enemy_probs = {b: prob_1st[b] for b in boats if b != 1}
        sorted_enemies = sorted(enemy_probs.items(), key=lambda x: x[1], reverse=True)
        
        for k_boat, k_prob in sorted_enemies[:3]:
            m_rate = makuri_dict.get(k_boat, 0.0)
            s_rate = sashi_dict.get(k_boat, 0.0)
            
            if m_rate >= s_rate and k_boat > 2:
                t1 = k_boat + 1 if k_boat + 1 <= 6 else 5
                t2 = 5 if k_boat != 5 else 6
                tactics = "まくり（外枠・ダッシュ追従軸）"
            else:
                t1 = 1
                t2 = k_boat - 1 if k_boat - 1 >= 2 else 2
                tactics = "差し・まくり差し（イン残り・内寄り追従軸）"
                
            report_data.append({
                "killer_boat": k_boat,
                "killer_prob": k_prob,
                "tactics": tactics,
                "target1": t1,
                "target2": t2
            })
    else:
        head_count = 10 if conf_rank == "A" else 6
        final_bets = df_tri.sort_values(by="確率", ascending=False).head(head_count)["買い目"].tolist()
        
        if conf_rank in ("C+", "C", "C-"):
            df_ana_pool = df_tri[df_tri["確率"] >= 0.015].copy()
            if df_ana_pool.empty or len(df_ana_pool) < 6:
                df_ana_pool = df_tri.iloc[6:].copy()
            df_ana = df_ana_pool.sort_values(by="妙味度", ascending=False).reset_index(drop=True)
            report_data = df_ana.head(6).to_dict(orient="records")
        
    return final_bets, report_data

import time

def generate_with_retry(client, contents, schema, retries=3):
    models_to_try = ["gemini-3.8-flash", "gemini-3.8-flash-lite"]
    last_err = None
    for model_name in models_to_try:
        for attempt in range(retries):
            try:
                return client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.1,
                    ),
                )
            except Exception as e:
                last_err = e
                if "503" in str(e) or "429" in str(e) or "UNAVAILABLE" in str(e):
                    time.sleep(3)
                    continue
                raise e
    raise last_err

def parse_base_files(uploaded_files, api_key):
    client = genai.Client(api_key=api_key)
    prompt = """
    添付された競艇の出走表（PDFまたは画像）から、記載されている全レース（1R〜12Rなど）の基本データを抽出してください。

    【抽出ルール】
    - stadium_name: 競艇場名（「戸田」「大村」「住之江」など場名のみ）
    - races: 掲載されている各レースのリスト
      - race_no: レース番号（1〜12の整数）
      - boats: 1号艇〜6号艇のリスト
        - boat_no: 艇番（1〜6）
        - racer_name: 選手名
        - class_type: 級別（"A1", "A2", "B1", "B2"）
        - win_rate_national: 全国勝率（上段の勝率）
        - rate_national: 全国2連率（上段の2連率）
        - win_rate_local: 当地勝率（下段の勝率）
        - rate_local: 当地2連率（下段の2連率）
        - motor_rate: モーター2連率
        - c1_rates 〜 c6_rates: 全国コース別欄（1〜6コース）の成績から計算した [1着率(%), 2着率(%), 3着率(%)] の3要素配列。
          ※表に「上段：進入回数」「下段：1・2・3着回数」が書かれている場合は、(回数 ÷ 進入回数 × 100) で1着率・2着率・3着率を計算してください。進入回数が0または「-」の場合は [0.0, 0.0, 0.0] としてください。
    """
    contents = [prompt]
    for f in uploaded_files:
        file_bytes = f.getvalue()
        mime_type = f.type if f.type else "application/pdf"
        contents.append(types.Part.from_bytes(data=file_bytes, mime_type=mime_type))

    response = generate_with_retry(client, contents, AllRacesBaseData)
    return json.loads(response.text)

def parse_live_images(uploaded_images, api_key):
    client = genai.Client(api_key=api_key)
    prompt = """
    添付された直前情報・展示タイム・ボートレース日和などのスクリーンショットから、1号艇〜6号艇の直前データを抽出してください。
    記載がない項目は 0.0（target_courseは枠番と同じ1〜6）にしてください。

    【抽出ルール】
    - boat_no: 艇番（1〜6）
    - exhibition_st: 展示ST（例: .12 -> 0.12, F.01 -> 0.01 で is_flying=true）
    - tilt: チルト角度（例: -0.5, 0.0, 0.5）
    - straight_time: 直線タイム（記載がなければ展示タイムと同じ値）
    - turn_time: 回り足タイム（記載がなければ展示タイムと同じ値）
    - inspect_time: 展示タイム（1周タイムまたは展示タイム）
    - target_course: スタート展示の進入コース（1〜6）
    - course_1st_rate: 当該進入コースでの1着率（%）（画像に記載がある場合のみ抽出、なければ0.0）
    - course_2nd_rate: 当該進入コースでの2着率（%）（画像に記載がある場合のみ抽出、なければ0.0）
    - course_3rd_rate: 当該進入コースでの3着率（%）（画像に記載がある場合のみ抽出、なければ0.0）
    - makuri_rate: 当該コースでの1着時「まくり」率（%）
    - sashi_rate: 当該コースでの1着時「差し」＋「まくり差し」の合計率（%）
    - is_flying: 展示STでフライング（F）している場合は true、それ以外は false
    """
    images = [Image.open(f) for f in uploaded_images]
    contents = [prompt] + images
    response = generate_with_retry(client, contents, LiveRaceData)
    return json.loads(response.text)

st.set_page_config(page_title="競艇予想システム v9.0", page_icon="🚤", layout="centered")
st.title("🔮 競艇AI予想システム v9.0")
st.caption("【2段階入力版】前日出走表PDF一括読み込み ＋ 直前情報スクショ合体")

api_key = st.secrets.get("GEMINI_API_KEY", "")
if not api_key:
    api_key = st.text_input("Gemini APIキーを入力", type="password")

# ==========================================
# STEP 1: 前日出走表PDF（または画像）の一括読み込み
# ==========================================
with st.expander("📂 ステップ1：出走表PDF・基本データの事前読み込み（1日1回）", expanded=("base_races" not in st.session_state)):
    base_files = st.file_uploader(
        "公式出走表PDF（1R〜6R、7R〜12Rなど複数選択可）をアップロード",
        type=("pdf", "png", "jpg", "jpeg"),
        accept_multiple_files=True,
        key="base_uploader"
    )
    if st.button("📥 出走表から全レース基本データを抽出・保存", use_container_width=True):
        if not api_key:
            st.error("APIキーが設定されていません。")
        elif not base_files:
            st.warning("出走表PDFまたは画像を選択してください。")
        else:
            with st.spinner("出走表から全レースの選手・勝率・コース別入着率を解析中（約10〜15秒）..."):
                try:
                    parsed = parse_base_files(base_files, api_key)
                    st.session_state["base_stadium"] = parsed["stadium_name"]
                    race_dict = {}
                    for r in parsed["races"]:
                        race_dict[r["race_no"]] = r["boats"]
                    st.session_state["base_races"] = race_dict
                    st.success(f"✅ ボートレース【{parsed['stadium_name']}】の {len(race_dict)} レース分を保存しました！")
                except Exception as e:
                    st.error(f"出走表の解析エラー: {e}")

# ==========================================
# STEP 2: レース選択 ＋ 直前情報スクショの追加
# ==========================================
st.divider()
st.subheader("⚡ ステップ2：レース選択 ＆ 直前情報スクショ合体")

stadium_keys = list(STADIUM_PROFILES.keys())
saved_stadium = st.session_state.get("base_stadium", "戸田")
default_st_idx = stadium_keys.index(saved_stadium) if saved_stadium in stadium_keys else 0

col1, col2 = st.columns(2)
selected_stadium = col1.selectbox("競艇場", stadium_keys, index=default_st_idx)

available_races = sorted(list(st.session_state["base_races"].keys())) if "base_races" in st.session_state else list(range(1, 13))
selected_race_no = col2.selectbox("予想するレース番号", available_races, format_func=lambda x: f"{x} R")

if "base_races" in st.session_state and selected_race_no in st.session_state["base_races"]:
    st.caption(f"✅ {selected_stadium} {selected_race_no}R の前日出走表データセット済み")
else:
    st.warning("⚠️ このレースの出走表データが未登録です（ステップ1でPDFを読み込むか、直前画像に全情報を含めてください）")

live_files = st.file_uploader(
    "📸 直前情報・展示タイム・ボートレース日和のスクショを選択",
    type=("png", "jpg", "jpeg", "webp"),
    accept_multiple_files=True,
    key="live_uploader"
)

if st.button("🚀 直前情報を合体して予想実行", type="primary", use_container_width=True):
    if not api_key:
        st.error("APIキーが設定されていません。")
    elif not live_files:
        st.warning("直前情報のスクリーンショットを1枚以上選択してください。")
    else:
        with st.spinner("直前情報を解析し、出走表データと合体して計算中..."):
            try:
                live_parsed = parse_live_images(live_files, api_key)
                live_map = {b["boat_no"]: b for b in live_parsed["boats"]}
                
                base_boats = st.session_state.get("base_races", {}).get(selected_race_no, [])
                base_map = {b["boat_no"]: b for b in base_boats}
                
                df_records = []
                for b_no in range(1, 7):
                    base_b = base_map.get(b_no, {})
                    live_b = live_map.get(b_no, {})
                    
                    t_course = live_b.get("target_course", b_no)
                    if t_course not in (1, 2, 3, 4, 5, 6):
                        t_course = b_no
                        
                    # コース別1・2・3着率：直前画像にあれば優先、なければ前日PDFの進入コース別率を使用
                    c_rates = base_b.get(f"c{t_course}_rates", [0.0, 0.0, 0.0])
                    if len(c_rates) < 3:
                        c_rates = [0.0, 0.0, 0.0]
                        
                    c1_r = live_b.get("course_1st_rate", 0.0) or c_rates[0]
                    c2_r = live_b.get("course_2nd_rate", 0.0) or c_rates[1]
                    c3_r = live_b.get("course_3rd_rate", 0.0) or c_rates[2]
                    
                    df_records.append({
                        "boat_no": b_no,
                        "class_type": base_b.get("class_type", "B1"),
                        "win_rate_national": base_b.get("win_rate_national", 5.0),
                        "2rate_national": base_b.get("rate_national", 30.0),
                        "win_rate_local": base_b.get("win_rate_local", 5.0),
                        "2rate_local": base_b.get("rate_local", 30.0),
                        "motor_rate": base_b.get("motor_rate", 30.0),
                        "exhibition_st": live_b.get("exhibition_st", 0.15),
                        "tilt": live_b.get("tilt", 0.0),
                        "straight_time": live_b.get("straight_time", 6.70),
                        "turn_time": live_b.get("turn_time", 6.70),
                        "inspect_time": live_b.get("inspect_time", 6.70),
                        "target_course": t_course,
                        "course_1st_rate": c1_r,
                        "course_2nd_rate": c2_r,
                        "course_3rd_rate": c3_r,
                        "makuri_rate": live_b.get("makuri_rate", 0.0),
                        "sashi_rate": live_b.get("sashi_rate", 0.0),
                        "is_flying": live_b.get("is_flying", False)
                    })
                
                st.session_state["merged_df"] = pd.DataFrame(df_records)
                st.session_state["merged_stadium"] = selected_stadium
                st.session_state["merged_race_no"] = selected_race_no
                
            except Exception as e:
                st.error(f"解析・合体エラーが発生しました: {e}")

# ==========================================
# 結果表示＆手動微調整テーブル
# ==========================================
if "merged_df" in st.session_state:
    st.divider()
    st.subheader(f"📋 {st.session_state['merged_stadium']} {st.session_state['merged_race_no']}R 統合データ（タップで修正可）")
    edited_df = st.data_editor(st.session_state["merged_df"], use_container_width=True, hide_index=True)
    
    res = predict_race_v9(edited_df, st.session_state["merged_stadium"], STADIUM_PROFILES)
    confidence_score = res.set_index("boat_no")["pred_1st_prob"].get(1, 0.0)
    
    if confidence_score >= 0.80: conf_rank = "A"
    elif confidence_score >= 0.60: conf_rank = "B"
    elif confidence_score >= 0.50: conf_rank = "C+"
    elif confidence_score >= 0.40: conf_rank = "C"
    elif confidence_score >= 0.30: conf_rank = "C-"
    elif confidence_score >= 0.15: conf_rank = "D"
    else: conf_rank = "E"
        
    final_bets, report_data = generate_custom_bets_v9(res, conf_rank)
    
    st.divider()
    st.markdown(f"### 🏟️ {st.session_state['merged_stadium']} {st.session_state['merged_race_no']}R | ランク 【 {conf_rank} 】")
    st.metric(label="1号艇 1着予測確率", value=f"{confidence_score*100:.1f}%")
    
    if conf_rank in ("D", "E"):
        st.error("🔥 【警告】イン飛び波乱レース。決まり手特性から展開軸を自動構成します。")
        for idx, r_data in enumerate(report_data):
            with st.container(border=True):
                st.markdown(f"**🥇 1着予想 第{idx+1}候補：【 {r_data['killer_boat']} 号艇 】** (単勝確率: {r_data['killer_prob']*100:.1f}%)")
                st.write(f"⚔️ **想定展開**：{r_data['tactics']}")
                st.write(f"👉 **展開軸**：{r_data['target1']} 号艇 / {r_data['target2']} 号艇")
                st.code(
                    f"【 {r_data['killer_boat']} - {r_data['target1']} - 全 】 【 {r_data['killer_boat']} - {r_data['target2']} - 全 】 (各4点)\n"
                    f"【 {r_data['killer_boat']} - 全 - {r_data['target1']} 】 【 {r_data['killer_boat']} - 全 - {r_data['target2']} 】 (各3点)"
                )
    else:
        st.subheader("🟩 三連単ターゲット（通常表示）")
        if conf_rank in ("C+", "C", "C-"):
            st.caption(f"※1号艇が頭の可能性もありますが、相手が絞りづらい混戦（{conf_rank}）の番組構成です。")
        
        cols = st.columns(2)
        for idx, bet in enumerate(final_bets):
            cols[idx % 2].success(f"第{idx+1}点 : **【 {bet} 】**")
            
        if conf_rank in ("C+", "C", "C-") and isinstance(report_data, list):
            st.subheader("🏴‍☠️ 三連単：妙味・穴枠厳選6点")
            for idx, row in enumerate(report_data):
                st.warning(f"第{idx+1}点 : **【 {row['買い目']} 】** ({row['確率']*100:.2f}%) [妙味度:{row['妙味度']:.2f}]")
