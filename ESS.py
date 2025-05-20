import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import streamlit as st

st.set_page_config(layout="wide")
st.title("ESS 최적화 기반 ROI 분석 도구")

# === 설정 입력 ===
SoC_max = st.sidebar.slider("최대 SoC", 0.5, 1.0, 0.8)
SoC_min = st.sidebar.slider("최소 SoC", 0.0, 0.5, 0.2)
Einit_ratio = st.sidebar.slider("초기 SoC 비율", 0.0, 1.0, 0.3)
loadSelect = st.sidebar.selectbox("부하 선택 (0~3)", [0, 1, 2, 3])
loadBase = st.sidebar.number_input("기본 부하 [W]", value=350000.0)
panelArea = st.sidebar.number_input("패널 면적 [m^2]", value=2500)
panelEff = st.sidebar.number_input("패널 효율", value=0.3)
cloudy = st.sidebar.selectbox("날씨", ["맑음", "흐림"])
timeOptimize = st.sidebar.number_input("최적화 시간 간격 [min]", value=60)
battery_cost_per_kWh = st.sidebar.number_input("배터리 가격 ($/kWh)", value=400)
pcs_cost_per_kW = st.sidebar.number_input("PCS 가격 ($/kW)", value=300)

# === 파일 로드 ===
base_path = "https://raw.githubusercontent.com/annemayer30/UOO/main/"
load_df = pd.read_excel(base_path + "loadData.xlsx", header=None)
cost_df = pd.read_excel(base_path + "costData.xlsx", header=None)
clear_df = pd.read_excel(base_path + "clearDay.xlsx", header=None)
cloudy_df = pd.read_excel(base_path + "cloudyDay.xlsx", header=None)
time = pd.read_excel(base_path + "time.xlsx", header=None).values.flatten()

# === 시간 설정 ===
dt = timeOptimize * 60
stepAdjust = int(dt / (time[1] - time[0]))
N = len(time[2::stepAdjust])

# === 입력 데이터 처리 ===
Pload_raw = load_df.iloc[2::stepAdjust, loadSelect].values[:N] + loadBase
Cost = cost_df.iloc[2::stepAdjust, 0].values[:N]
Ppv_raw = clear_df if cloudy == "맑음" else cloudy_df
Ppv = panelArea * panelEff * Ppv_raw.iloc[2::stepAdjust, 0].values[:N]

# === 탐색 범위 ===
battery_range = np.arange(500, 3001, 500)
pcs_range = np.arange(100, 901, 200)

best_result = None
best_ROI = -np.inf

for batt_kWh in battery_range:
    for pcs_kW in pcs_range:
        battEnergy = batt_kWh * 3600 * 1e3
        Pmin = -pcs_kW * 1e3
        Pmax = pcs_kW * 1e3
        Emin = SoC_min * battEnergy
        Emax = SoC_max * battEnergy
        Einit = Einit_ratio * battEnergy

        c = np.concatenate([dt * Cost, np.zeros(N), np.zeros(N)])
        bounds = [(0, max(Pload_raw)*0.9)] * N + [(Pmin, Pmax)] * N + [(Emin, Emax)] * N

        # === 제약조건 ===
        A_eq = np.zeros((N + 1, 3 * N))
        b_eq = np.zeros(N + 1)
        A_eq[0, N] = dt
        A_eq[0, 2 * N] = 1
        b_eq[0] = Einit
        for i in range(1, N):
            A_eq[i, N + i] = dt
            A_eq[i, 2 * N + i] = 1
            A_eq[i, 2 * N + i - 1] = -1
        A_eq[N, 2 * N + N - 1] = 1
        b_eq[N] = Einit

        A_load = np.zeros((N, 3 * N))
        for i in range(N):
            A_load[i, i] = 1
            A_load[i, N + i] = 1
        b_load = Pload_raw - Ppv

        res = linprog(
            c=c,
            A_eq=np.vstack([A_eq, A_load]),
            b_eq=np.concatenate([b_eq, b_load]),
            bounds=bounds,
            method='highs'
        )

        if not res.success:
            continue

        x = res.x
        Pgrid = x[:N]
        Pbatt = x[N:2*N]
        Ebatt = x[2*N:]
        SoC = Ebatt / battEnergy * 100

        # ESS 유무 차이로만 절감 계산
        LoadCost = np.sum(((Pload_raw - Ppv) / 1e3) * Cost)
        GridCost = np.sum((Pgrid / 1e3) * Cost)
        true_saving = max(0, LoadCost - GridCost)

        battery_saving_energy = np.sum((Pbatt < 0) * (-Pbatt) * dt) / 3600 / 1e3
        battery_saving_cost = battery_saving_energy * np.mean(Cost)

        annual_saving = true_saving * 365
        saving_10yr = annual_saving * 10
        batt_cost = batt_kWh * battery_cost_per_kWh
        pcs_cost = pcs_kW * pcs_cost_per_kW
        total_cost = batt_cost + pcs_cost
        ROI = (saving_10yr - total_cost) / total_cost * 100
        payback = total_cost / true_saving if true_saving > 0 else np.inf

        if ROI > best_ROI:
            best_result = {
                "batt_kWh": batt_kWh,
                "pcs_kW": pcs_kW,
                "Pgrid": Pgrid,
                "Pbatt": Pbatt,
                "Ebatt": Ebatt,
                "SoC": SoC,
                "Cost": Cost,
                "Pload": Pload_raw,
                "Ppv": Ppv,
                "thour": np.arange(1, N+1) * dt / 3600,
                "summary": {
                    "배터리 절약량 (kWh)": battery_saving_energy,
                    "배터리 절약 금액 ($)": battery_saving_cost,
                    "배터리 투자비용 ($)": batt_cost,
                    "PCS 투자비용 ($)": pcs_cost,
                    "총 투자비용 ($)": total_cost,
                    "연간 절감액 ($)": annual_saving,
                    "10년 누적 절감액 ($)": saving_10yr,
                    "ROI (%)": ROI,
                    "손익분기점 (일)": payback
                }
            }
            best_ROI = ROI

# === 결과 출력 ===
if best_result:
    st.subheader("최적화 결과 요약")
    st.markdown(f"**배터리 용량**: {best_result['batt_kWh']} kWh")
    st.markdown(f"**PCS 용량**: {best_result['pcs_kW']} kW")
    st.dataframe(pd.DataFrame([best_result["summary"]]))

    # === 그래프 ===
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Battery + SoC
    axes[0].plot(best_result["thour"], best_result["Ebatt"] / 3.6e6, label="Battery Energy [kWh]", color='tab:blue')
    ax2 = axes[0].twinx()
    ax2.plot(best_result["thour"], best_result["SoC"], 'r-', label="SoC (%)")
    ax2.axhspan(SoC_min * 100, SoC_max * 100, color='gray', alpha=0.2)
    ax2.set_ylim(0, 100)
    axes[0].set_xlim([1, 24])
    axes[0].set_ylabel("Battery [kWh]")
    ax2.set_ylabel("SoC (%)")
    axes[0].grid(True)

    # Grid Price
    axes[1].plot(best_result["thour"], best_result["Cost"], linewidth=1.5)
    axes[1].set_xlim([1, 24])
    axes[1].set_ylabel("Grid Price [$/kWh]")
    axes[1].grid(True)

    # Power
    axes[2].plot(best_result["thour"], best_result["Pgrid"] / 1e3, label="Grid")
    axes[2].plot(best_result["thour"], best_result["Pload"] / 1e3, label="Load")
    axes[2].plot(best_result["thour"], best_result["Pbatt"] / 1e3, label="Battery")
    axes[2].plot(best_result["thour"], best_result["Ppv"] / 1e3, label="PV")
    axes[2].set_xlim([1, 24])
    axes[2].set_ylabel("Power [kW]")
    axes[2].legend()
    axes[2].grid(True)

    st.pyplot(fig)
else:
    st.error("최적화 실패 또는 유효한 해를 찾을 수 없습니다.")
