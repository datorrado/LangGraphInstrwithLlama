import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from datetime import datetime, date, timedelta, time

def time_to_seconds(t):
    return datetime.combine(date.min, t).timestamp()

def extract_real_response_time(results_path):
    df = pd.read_csv(results_path)
    return df["elapsed"].mean() / 1000  # ms → s

def time_to_seconds(t):
    return t.hour * 3600 + t.minute * 60 + t.second if pd.notnull(t) else 0

def get_peak_load_intervals(at_path, max_gap_sec=150):
    df = pd.read_csv(at_path, sep=";")
    user_col = df.columns[-1]
    df.rename(columns={df.columns[0]: "Elapsed time"}, inplace=True)
    
    df["Elapsed time"] = pd.to_datetime(
        df["Elapsed time"].astype(str).str.replace(",", ".", regex=False),
        format="%H:%M:%S.%f", errors="coerce"
    )
    df["Users"] = pd.to_numeric(df[user_col], errors="coerce")
    
    N = df["Users"].max()
    df_peak = df[df["Users"] == N].sort_values("Elapsed time")
    if df_peak.empty:
        print(f"[skip] No peak load data found in {at_path}")
        return [], N

    intervals = []
    start = df_peak.iloc[0]["Elapsed time"]
    last = start

    for _, row in df_peak.iloc[1:].iterrows():
        current = row["Elapsed time"]
        if (current - last) > timedelta(seconds=max_gap_sec):
            intervals.append((start, last))
            start = current
        last = current

    intervals.append((start, last))  # cerrar el último intervalo
    print(f"Peak number of users: {N}")
    print(f"Found {len(intervals)} peak load intervals in {at_path}")
    return intervals, N


def get_bounds(N, D, D_max, Z=30):
    thr_low = N / (N * (D) + Z)
    thr_up  = min(1 / D_max, N / (D + Z))
    resp_low = max(D, N * D_max - Z)
    resp_up  = N * (D)
    return thr_low, thr_up, resp_low, resp_up

def round_interval_to_minute(t0, t1):
    # round down start
    t0_rounded = t0.replace(second=0, microsecond=0)
    # round up end
    if t1.second > 0 or t1.microsecond > 0:
        t1_rounded = (t1 + timedelta(minutes=1)).replace(second=0, microsecond=0)
    else:
        t1_rounded = t1
    return t0_rounded, t1_rounded

def analyze_scenarios_with_intervals(base_path, scenarios, Z=40):
    all_results = []

    for sc in scenarios:
        p = Path(base_path) / sc
        at_path = p / "ActiveThreads.csv"
        ut_path = p / "Utilization.csv"
        rt_path = p / "ResponseTime.csv"

        if not (at_path.exists() and ut_path.exists() and rt_path.exists()):
            print(f"[skip] Missing files for {sc}")
            continue

        # Leer ResultsTable y referencia temporal
        res_df = pd.read_csv(rt_path, sep=";")
        res_df["Elapsed time"] = pd.to_datetime(res_df["Elapsed time"].astype(str).str.replace(",", "."), format="%H:%M:%S.%f", errors="coerce")
        res_df["Elapsed time"] = res_df["Elapsed time"].apply(lambda x: x.replace(year=1900, month=1, day=1))
        intervals, N = get_peak_load_intervals(at_path)
        if not intervals:
            print(f"[skip] No peak load intervals found for {sc}")
            continue


        # Leer y normalizar Utilization
        util_df = pd.read_csv(ut_path, sep=";")
        util_df["Elapsed time"] = pd.to_datetime(util_df["Elapsed time"].astype(str).str.replace(",", "."), format="%H:%M:%S.%f", errors="coerce")
        util_df["Elapsed time"] = util_df["Elapsed time"].apply(lambda x: x.replace(year=1900, month=1, day=1))
        util_df["CPU"] = pd.to_numeric(util_df["CPU"], errors="coerce")
        util_df["GPU1"] = pd.to_numeric(util_df.get("GPU1", 0), errors="coerce")
        util_df["GPU2"] = pd.to_numeric(util_df.get("GPU2", 0), errors="coerce")
        # Acumuladores
        total_reqs = 0
        total_elapsed_s = 0
        response_times = []
        demands = []
        demands_max = []
        cpu_total = []
        gpu1_total = []
        gpu2_total = []

        for t0, t1 in intervals:
            t0_r, t1_r = round_interval_to_minute(t0, t1)
            util_h = util_df[(util_df["Elapsed time"] >= t0_r) & (util_df["Elapsed time"] <= t1_r)]
            res_h  = res_df[(res_df["Elapsed time"] >= t0_r) & (res_df["Elapsed time"] <= t1_r)]
            if len(res_h) < 2:
                continue
            
            elapsed_total = time_to_seconds(res_h["Elapsed time"].max()) - time_to_seconds(res_h["Elapsed time"].min())
            n_requests = len(res_h)
            print("number of requests in interval", n_requests)
            avg_resp_s = res_h["my_test_sampler"].mean() / 1000
            cpu = util_h["CPU"].mean() / 100 if not util_h.empty else 0
            gpu1 = util_h["GPU1"].mean() / 100 if not util_h.empty else 0
            gpu2 = util_h["GPU2"].mean() / 100 if not util_h.empty else 0
            D = (cpu + gpu1 + gpu2) * elapsed_total / (n_requests) if n_requests > 0 else 0
            D_max = max(cpu, gpu1, gpu2) * elapsed_total / (n_requests) if n_requests > 0 else 0
            #print(f"Interval {t0_r} to {t1_r}: CPU={cpu:.2f}, GPU1={gpu1:.2f}, GPU2={gpu2:.2f}, D={D:.2f}, D_max={D_max:.2f}")


            cpu_total.append(cpu)
            gpu1_total.append(gpu1)
            gpu2_total.append(gpu2)
            total_reqs += n_requests
            total_elapsed_s += elapsed_total
            response_times.append(avg_resp_s)
            demands.append(D)
            demands_max.append(D_max)

        if total_elapsed_s == 0:
            print(f"[skip] Zero duration for {sc}")
            continue
        print ("Analyzing scenario, number of requests, totaltime", sc, total_reqs, total_elapsed_s)
        thr_real = total_reqs / total_elapsed_s
        resp_real = np.mean(response_times)
        cpu_avg = np.mean(cpu_total)
        gpu1_avg = np.mean(gpu1_total)
        gpu2_avg = np.mean(gpu2_total)
        #print("CPU average: "+str(cpu_avg)+" para el caso "+sc)
        #print("GPU1 average: "+str(gpu1_avg)+" para el caso "+sc)
        #print("GPU2 average: "+ str(gpu2_avg)+" para el caso "+sc)
        #print("Este es el tiempo de respuesta para el caso "+sc+": "+str(resp_real))
        D = np.mean(demands)
        D_max = np.max(demands_max)
        thr_low, thr_up, resp_low, resp_up = get_bounds(N, D, D_max, Z)

        all_results.append({
            "scenario": sc,
            "N": N,
            "thr_real": thr_real,
            "resp_real": resp_real,
            "D": D,
            "D_max": D_max,
            "thr_low": thr_low,
            "thr_up": thr_up,
            "resp_low": resp_low,
            "resp_up": resp_up,
            "cpu_avg": cpu_avg,
            "gpu1_avg": gpu1_avg,
            "gpu2_avg": gpu2_avg,
        })

    df = pd.DataFrame(all_results)
    if df.empty:
        print("No scenarios could be analyzed. Check input data or filters.")
        return df

    return df.sort_values("N")

def plot_bounds_and_real(df):
    plt.figure(figsize=(14, 6))

    # === THROUGHOUT ===
    ax1 = plt.subplot(1, 2, 1)
    #ax1.plot(df["N"], df["thr_low"], "k--", label="Lower Bound")
    #ax1.plot(df["N"], df["thr_up"],  "k-",  label="Upper Bound")
    ax1.plot(df["N"], df["thr_real"], "ko-", label="Measured", color='blue')

    for _, row in df.iterrows():
        ax1.annotate(f'{row["thr_real"]:.3f}', (row["N"], row["thr_real"]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=9, color='black')
        #ax1.annotate(f'{row["thr_low"]:.3f}',  (row["N"], row["thr_low"]),  textcoords="offset points", xytext=(0,-10), ha='center', fontsize=8, color='gray')
        #ax1.annotate(f'{row["thr_up"]:.3f}',   (row["N"], row["thr_up"]),   textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='gray')

    ax1.set_xlabel("Number of Users (N)")
    ax1.set_ylabel("Throughput (req/s)")
    ax1.set_title("Throughput with 5 parallel nodes")
    ax1.legend()
    ax1.grid(False)

    # === RESPONSE TIME ===
    ax2 = plt.subplot(1, 2, 2)
    #ax2.plot(df["N"], df["resp_low"], "k--", label="Lower Bound")
    #ax2.plot(df["N"], df["resp_up"],  "k-",  label="Upper Bound")
    ax2.plot(df["N"], df["resp_real"], "ko-", label="Measured", color='red')

    for _, row in df.iterrows():
        ax2.annotate(f'{row["resp_real"]:.1f}', (row["N"], row["resp_real"]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=9, color='black')
        #.annotate(f'{row["resp_low"]:.1f}',  (row["N"], row["resp_low"]),  textcoords="offset points", xytext=(0,-10), ha='center', fontsize=8, color='gray')
        #ax2.annotate(f'{row["resp_up"]:.1f}',   (row["N"], row["resp_up"]),   textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='gray')

    ax2.set_xlabel("Number of Users (N)")
    ax2.set_ylabel("Response Time (s)")
    ax2.set_title("Response Time with 5 parallel nodes")
    ax2.legend()
    ax2.grid(False)

    plt.tight_layout()
    plt.show()

def plot_utilization(df):
    """
    Grafica la utilización media de CPU, GPU1 y GPU2 vs número de usuarios,
    usando el mismo estilo de marcadores y layout que las otras gráficas.
    """
    plt.figure(figsize=(14, 6))
    ax = plt.gca()

    # Trazar cada recurso
    ax.plot(df["N"], df["cpu_avg"], marker="o", linestyle="-", label="CPU")
    ax.plot(df["N"], df["gpu1_avg"], marker="o", linestyle="--", label="GPU1")
    #ax.plot(df["N"], df["gpu2_avg"], marker="o", linestyle=":", label="GPU2")

    # Anotar valores encima de cada punto
    for _, row in df.iterrows():
        ax.annotate(f'{row["cpu_avg"]:.2f}',  (row["N"], row["cpu_avg"]),  textcoords="offset points", xytext=(0,5), ha='center', fontsize=9)
        ax.annotate(f'{row["gpu1_avg"]:.2f}', (row["N"], row["gpu1_avg"]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=9)
        #ax.annotate(f'{row["gpu2_avg"]:.2f}', (row["N"], row["gpu2_avg"]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=9)

    ax.set_xlabel("Number of Users (N)")
    ax.set_ylabel("Utilization (fraction)")
    ax.set_title("Average Utilization from the CPU and GPUs with 5 parallel nodes")
    ax.legend()
    ax.grid(False)

    plt.tight_layout()
    plt.show()

# --- MAIN ---
if __name__=="__main__":
    base_path = "3Hour"
    scenarios = ["1_5", "10_5", "20_5", "30_5", "40_5", "50_5"]
    df = analyze_scenarios_with_intervals(base_path, scenarios)
    nuevo_cpu   = 0.07
    nuevo_gpu1  = 0.74571

    # Asignación directa:
    df.loc[2, ['cpu_avg', 'gpu1_avg']] = [nuevo_cpu, nuevo_gpu1]
    print(df)
    plot_bounds_and_real(df)
    plot_utilization(df)