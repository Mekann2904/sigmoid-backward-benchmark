import numpy as np
import time
import matplotlib.pyplot as plt

def main():
    # データサイズ: 桁をまたぐように設定
    data_sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000, 3_000_000, 5_000_000]
    
    naive_times = []
    opt_times = []

    print(f"{'Size (N)':>12} | {'Naive (sec)':>12} | {'Optimized (sec)':>15} | {'Speedup':>10}")
    print("-" * 65)

    for N in data_sizes:
        # --- データ準備 ---
        x = np.random.randn(N)
        dout = np.random.randn(N)
        y = 1 / (1 + np.exp(-x))

        loops = 10  # ループ回数（少し軽めに調整）
        
        # --- 1. Naive (expあり) ---
        start = time.time()
        for _ in range(loops):
            _ = dout * (y ** 2) * np.exp(-x)
        naive_times.append((time.time() - start) / loops)

        # --- 2. Optimized (yのみ) ---
        start = time.time()
        for _ in range(loops):
            _ = dout * y * (1.0 - y)
        opt_times.append((time.time() - start) / loops)

        # 進捗表示
        speedup = naive_times[-1] / opt_times[-1]
        print(f"{N:>12,} | {naive_times[-1]:.5f} | {opt_times[-1]:.5f} | {speedup:.1f}x")

    plot_results_log(data_sizes, naive_times, opt_times)


def plot_results_log(sizes, naive, opt):
    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # --- 左: 処理時間 ---
    ax[0].plot(sizes, naive, marker='o', label='Naive: exp(-x)', color='#FF5733', linestyle='--')
    ax[0].plot(sizes, opt, marker='o', label='Optimized: y(1-y)', color='#33C1FF', linewidth=3)
    
    # ★ここが変更点: 横軸と縦軸を対数スケールにする
    ax[0].set_xscale('log') 
    ax[0].set_yscale('log') # 時間も桁が違うので対数の方が見やすい場合があります（お好みで外してもOK）
    
    ax[0].set_title('Execution Time (Log Scale)')
    ax[0].set_xlabel('Data Size (N) - Log Scale')
    ax[0].set_ylabel('Time (seconds) - Log Scale')
    ax[0].legend()
    ax[0].grid(True, which="both", ls="-", alpha=0.5) # グリッドを細かく表示

    # --- 右: 倍率 ---
    ratios = [n / o for n, o in zip(naive, opt)]
    ax[1].plot(sizes, ratios, marker='s', color='#28A745', linewidth=2)
    
    # ★ここが変更点: 横軸を対数スケールにする
    ax[1].set_xscale('log')
    
    ax[1].set_title('Speedup Ratio (Naive / Optimized)')
    ax[1].set_xlabel('Data Size (N) - Log Scale')
    ax[1].set_ylabel('Speedup Factor (times)')
    ax[1].set_ylim(0, max(ratios) * 1.2)
    ax[1].grid(True, which="both", ls="-", alpha=0.5)

    # 平均倍率の表示
    avg_speedup = sum(ratios) / len(ratios)
    ax[1].text(sizes[0], avg_speedup + 0.5, f" Avg Speedup: ~{avg_speedup:.1f}x", 
               color='green', fontweight='bold', verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig('sigmoid_benchmark_log.png')
    print("\n[INFO] 'sigmoid_benchmark_log.png' に保存しました。")
    plt.show()

if __name__ == "__main__":
    main()
