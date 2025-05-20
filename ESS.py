import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# === 설정 ===
SoC_max = 0.8
SoC_min = 0.2
Einit_ratio = 0.3
loadSelect = 2
loadBase = 350e3
panelArea = 2500
panelEff = 0.3
cloudy = 0
timeOptimize = 60
numDays = 1
battery_cost_per_kWh = 400
pcs_cost_per_kW = 300

# === 엑셀 경로에서 로드 ===
base_path = "E:/가천대학교/2025-1학기/최적화기반 에너지관리/"
load_df = pd.read_excel(base_path + "loadData.xlsx", header=None)
cost_df = pd.read_excel(base_path + "costData.xlsx", header=None)
clear_df = pd.read_excel(base_path + "clearDay.xlsx", header=None)
cloudy_df = pd.read_excel(base_path + "cloudyDay.xlsx", header=None)
time = pd.read_excel(base_path + "time.xlsx", header=None).values.flatten()

# === 시간 설정 ===
dt = timeOptimize * 60
stepAdjust = int(dt / (time[1] - time[0]))
N = len(time[2::stepAdjust])

# === 기본 데이터 ===
Pload_raw = np.tile(load_df.iloc[2::stepAdjust, loadSelect].values[:N], 1) + loadBase
Cost = np.tile(cost_df.iloc[2::stepAdjust, 0].values[:N], 1)
Ppv_raw = clear_df if cloudy == 0 else cloudy_df
Ppv = panelArea * panelEff * np.tile(Ppv_raw.iloc[2::stepAdjust, 0].values[:N], 1)

# === 최적화 범위 ===
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

        # === 제약 ===
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

        # === ROI 기준 변경 (ESS 유무 기준 차이만)
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
    print(f"최적 배터리 용량: {best_result['batt_kWh']} kWh")
    print(f"최적 PCS 용량: {best_result['pcs_kW']} kW")
    for k, v in best_result["summary"].items():
        print(f"{k}: {v:.2f}")

    # === 시각화 ===
    plt.figure(figsize=(10, 12))
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(best_result["thour"], best_result["Ebatt"] / 3.6e6, label="Battery Energy [kWh]", color='tab:blue', linewidth=1.5)
    ax1.set_xlim([1, 24])
    ax1.set_xlabel("Time [hrs]")
    ax1.set_ylabel("Battery Energy [kWh]", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(best_result["thour"], best_result["SoC"], 'r-', label="SoC (%)", linewidth=1.5)
    ax2.axhspan(SoC_min * 100, SoC_max * 100, color='gray', alpha=0.2)
    ax2.set_ylabel("SoC (%)", color='tab:red')
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax1.legend(loc='best')

    plt.subplot(3, 1, 2)
    plt.plot(best_result["thour"], best_result["Cost"], linewidth=1.5)
    plt.grid(True)
    plt.xlim([1, 24])
    plt.ylim([8, 26])
    plt.xlabel("Time [hrs]")
    plt.ylabel("Grid Price [$/kWh]")

    plt.subplot(3, 1, 3)
    plt.plot(best_result["thour"], best_result["Pgrid"] / 1e3, label="Grid", linewidth=1.5)
    plt.plot(best_result["thour"], best_result["Pload"] / 1e3, label="Load", linewidth=1.5)
    plt.plot(best_result["thour"], best_result["Pbatt"] / 1e3, label="Battery", linewidth=1.5)
    plt.plot(best_result["thour"], best_result["Ppv"] / 1e3, label="PV", linewidth=1.5)
    plt.grid(True)
    plt.xlim([1, 24])
    plt.xlabel("Time [hrs]")
    plt.ylabel("Power [kW]")
    plt.legend()

    plt.tight_layout()
    plt.show()
