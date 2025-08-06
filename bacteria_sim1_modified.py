# bacteria_sim1.py の先頭部分（必要なインポートなど）
import os
import pandas as pd
import random
import pickle
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUIレスのバックエンド
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path

# この .py ファイルのあるフォルダをベースに
script_dir = Path(__file__).resolve().parent

# 出力は script_dir/“histogram_data” にまとめる
output_dir = script_dir / "histogram_data"
output_dir.mkdir(parents=True, exist_ok=True)

# CSVデータの前処理など（これはシミュレーション全体で共通ならグローバルに置くか、必要なら各シミュレーション内で再実行）
folder_path = 'Data_Durvernoy/coli'
split_lengths = []
all_filtered_max_lengths = []

for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        try:
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
            cname_column = next((col for col in df.columns if 'cname' in col.lower()), None)
            length_column = next((col for col in df.columns if 'length' in col.lower()), None)
            if cname_column and length_column:
                df[cname_column] = df[cname_column].str.strip()
                df_filtered = df[df[cname_column] != '0']
                all_cnames = set(df_filtered[cname_column].unique())
                cnames_with_children = [cname for cname in all_cnames if any((cname + suffix) in all_cnames for suffix in ['T', 'H'])]
                df_filtered = df_filtered[df_filtered[cname_column].isin(cnames_with_children)]
                if not df_filtered.empty:
                    df_max_lengths = df_filtered.loc[df_filtered.groupby(cname_column)[length_column].idxmax()]
                    all_filtered_max_lengths.append(df_max_lengths)
                    split_lengths.extend(df_max_lengths[length_column].dropna().values)
        except Exception:
            continue

def get_random_split_length():
    if split_lengths:
        return random.choice(split_lengths)
    else:
        raise ValueError("split_lengthデータが存在しません。")

# 成長率リスト（各ケースを区別するためのリスト）
#growth_rate_list = [
#    0.0318923017950924,
#    0.02391296425097043,
#    0.03181780533101043,
#    0.03219089428717173,
#    0.028775610031598328,
#    0.022741131368990135,
#    0.032767027317560454,
#    0.025050534305706414,
#    0.02832049680944065,
#    0.025879112141591276
#]
growth_rate_list = [
    0.028775610031598328,
    0.028775610031598328,
    0.028775610031598328,
    0.028775610031598328,
    0.028775610031598328,
    0.028775610031598328,
    0.028775610031598328,
    0.028775610031598328,
    0.028775610031598328,
    0.028775610031598328
]

pole_kind = {}

def run_simulation(sim_index):
    global pole_kind
    pole_kind.clear()
    


    import time
    start_simulation = time.time()  # 各シミュレーションの開始時刻を記録
    script_dir = Path(__file__).resolve().parent

    import os
    import pandas as pd
    import random
    import pickle
    from datetime import datetime
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # GUIレスのバックエンド
    import matplotlib.pyplot as plt
  
    import matplotlib.patches as patches
    from collections import defaultdict
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    #np.random.seed(123)  # NumPy の乱数シードを設定
    #random.seed(123)     # Python の random モジュールのシードを設定
    np.random.seed(sim_index)  # NumPy の乱数シードを設定
    random.seed(sim_index)     # Python の random モジュールのシードを設定
    mean_interval=0.029
    std_interval=0.0029
    #g_r=max(np.random.normal(loc=mean_interval, scale=std_interval), 0.001)
    g_r = growth_rate_list[sim_index]
    print(g_r)
    k_s_values=[4000]#
    #k_s_values=[300,500,800,1000,1200,1500,1800,2000,3000,4000,5000,8000,10000,15000,20000,25000,30000]
    for k_s in k_s_values:
        # ディスクの物理特性
        R = 11  # 半径
        m = 1.0  # 質量

        # バネの特性
        #k_s =1500.0  # バネ定数
        l = R  # 自然長

        # 反発力の特性
        k_repulsion = 1.0  # 反発力の定数

        # 反発力の特性
        k_c =2000.0  # 反発力の定数

        # トーションスプリングの特性
        kt=1000.0
        kt_par = kt  # トーションスプリング定数 (parallel component)
        kt_bot = kt  # トーションスプリング定数 (bottom component)
        theta0 = np.pi  # 自然状態の角度（ラジアン）

        # グローバルな粒子IDカウンターの初期化
        global_particle_id_counter = 0

        # 初期の粒子にIDを割り当て
        initial_positions = [
            np.array([0, 0]),
            np.array([l, 0]),
            np.array([2 * l, 0]),
            np.array([3 * l, 0])
        ]
        initial_velocities = [np.array([0.0, 0.0]) for _ in initial_positions]
        initial_ids = []

        # 時間ステップとシミュレーション上限
        initial_dt = 0.01
        dt=initial_dt
        eps = 1e-4
        t_max = 6000
        max_chains = 128  # 計算を止める粒子鎖数の上限
        current_time = 0

        # 粒子追加のタイミング
        c1 = 0.01  # parameter for stability
        # 成長率のパラメータ

        #mean_interval = 0.029  # 成長率の平均
        #mean_interval =0.0553988

        #最小値: 0.00379238
        #中央値: 0.0283913
        #最大値: 0.0553988


        #std_interval = 0.0029   # 成長率の標準偏差
        #std_interval = 0.
        particle_add_interval_counter = 0  # 粒子追加間隔カウンター
        add_particle_interval_steps=1.0


        for _ in initial_positions:
            initial_ids.append(global_particle_id_counter)
            global_particle_id_counter += 1

        # 粒子生成の記録用リストを初期化
        particle_creation_history = []
        # 初期粒子の生成時に記録
        for new_id in initial_ids:
            particle_creation_history.append({
                'particle_id': new_id,
                'time_created': current_time
            })


        # 成長率を決定する関数
        def determine_growth_rate(distribution="normal", **params):
            if distribution == "normal":
                mean = params.get('mean', 1.0)
                std = params.get('std', 0.1)
                return np.random.normal(mean, std)
            # 他の分布も追加可能
            else:
                raise ValueError(f"Unknown distribution type: {distribution}")

        # 粒子追加用に成長率を初期化
        def initialize_growth_rate_for_chain(c1, distribution="normal", **params):
            growth_rate = determine_growth_rate(distribution, **params)/c1
            add_particle_interval_steps = 1.0 / growth_rate  # 成長率から間隔を計算
            return growth_rate, add_particle_interval_steps

        # 記録用の成長率
        growth_rates = []



        # 逆流防止のための小さい範囲
        overlap_threshold = 0.1  # 0.1以下の距離で速度を0にする


        # 粒子の位置をランダムに少しずらすための関数
        def random_shift(position, scale=0.1):
            return position + np.random.uniform(-scale, scale, size=position.shape)

        # バウンディングボックスの計算
        def get_bounding_box(chain):
            positions, _, _, _ = chain
            x_min = min(pos[0] for pos in positions) - R
            x_max = max(pos[0] for pos in positions) + R
            y_min = min(pos[1] for pos in positions) - R
            y_max =max(pos[1] for pos in positions) + R
            return x_min, x_max, y_min, y_max

        # バウンディングボックスの重なり判定
        def are_bounding_boxes_overlapping(box1, box2):
            x_min1, x_max1, y_min1, y_max1 = box1
            x_min2, x_max2, y_min2, y_max2 = box2
            if x_max1 < x_min2 or x_max2 < x_min1 or y_max1 < y_min2 or y_max2 < y_min1:
                return False
            return True

        # シンプルな反発力の計算
        """
        def simple_repulsion_force(Xj, Xl, k_c, R):
            r = np.linalg.norm(Xj - Xl) / (2 * R)
            if r < 1:
                f = -k_c / (2 * R ** 2) * (1 - 2 * R / np.linalg.norm(Xj - Xl)) * (Xj - Xl)
                return f
            else:
                return np.array([0.0, 0.0])
        """
        def phi(x):
            """
            距離比 x = |X_j^k - X_i^m|/(2R) に対する反発エネルギーの寄与を返す。
            x <= 1 のとき: (k_c / 2) * (x - 1)^2
            x > 1 のとき: 0
            """
            if x <= 1:
                return (k_c / 2.0) * (x - 1) ** 2
            else:
                return 0.0



        def simple_repulsion_force(Xj, Xl, k_c, R, epsilon=1e-8):
            distance_vector = Xj - Xl
            distance = np.linalg.norm(distance_vector)
            if distance < 2 * R:
                if distance < epsilon:
                    # 完全に重なっている場合、ランダムな方向に小さな力を加える
                    force_direction = np.random.uniform(-1, 1, size=2)
                    force_direction /= np.linalg.norm(force_direction) + epsilon
                else:
                    force_direction = distance_vector / distance
                overlap_factor = 1 - (2 * R) / distance
                f = -k_c / (2 * R ** 2) * overlap_factor * distance_vector
                return f
            else:
                return np.array([0.0, 0.0])


        # 粒子鎖同士の衝突をチェックし、反発力を適用
        def check_and_resolve_chain_collision(chain1, chain2, k_c, R):
            pos1, vel1, ids1, _ = chain1
            pos2, vel2, ids2, _ = chain2

            box1 = get_bounding_box(chain1)
            box2 = get_bounding_box(chain2)

            # Check if bounding boxes overlap
            if are_bounding_boxes_overlapping(box1, box2):
                for _ in range(20):  # 重なりが解消されるまで最大10回反発力を適用
                    overlap_exists = False
                    for i, pos1_i in enumerate(pos1):
                        for j, pos2_j in enumerate(pos2):
                            if np.linalg.norm(pos1_i - pos2_j) < 2 * R:
                                force = simple_repulsion_force(pos1_i, pos2_j, k_c, R)
                                if np.linalg.norm(force) > 0:
                                    overlap_exists = True
                                    vel1[i] += force * dt
                                    vel2[j] -= force * dt

                    if not overlap_exists:
                        break  # 重なりが解消されたらループを抜ける

                    # Update positions
                    for i in range(len(pos1)):
                        pos1[i] += vel1[i] * dt
                    for j in range(len(pos2)):
                        pos2[j] += vel2[j] * dt

        # 端の粒子同士の重なりを解消
        def resolve_end_particle_overlap(chain1, chain2, k_c, R):
            pos1, vel1, ids1, _ = chain1
            pos2, vel2, ids2, _ = chain2

            for _ in range(20):  # 最大10回反発力を適用
                overlap_exists = False
                for pos1_i, vel1_i in zip(pos1, vel1):
                    for pos2_j, vel2_j in zip(pos2, vel2):
                        if np.linalg.norm(pos1_i - pos2_j) < 2 * R:
                            force = simple_repulsion_force(pos1_i, pos2_j, k_c, R)
                            if np.linalg.norm(force) > 0:
                                overlap_exists = True
                                vel1_i += force * dt
                                vel2_j -= force * dt

                if not overlap_exists:
                    break  # 重なりが解消されたらループを抜ける

                # Update positions
                for i in range(len(pos1)):
                    pos1[i] += vel1[i] * dt
                for j in range(len(pos2)):
                    pos2[j] += vel2[j] * dt


        # トーションスプリング力の計算（parallel component）
        def torsion_spring_par(k, positions, kt_par, theta0, eps=1e-6):
            V = np.zeros(2)
            n = len(positions)

            def norm(vec):
                return np.linalg.norm(vec)

            # 必要な粒子数が揃っているか確認
            if n < 3:
                return V  # 粒子数が3未満の場合は計算しない

            if k == 0 and n >= 3:
                if k + 2 >= n:
                    return V  # インデックス範囲外を防ぐ
                Xj, Xj1, Xj2 = positions[k], positions[k + 1], positions[k + 2]
                # 以下、元の計算...
                V -= kt_par / (norm(Xj2 - Xj1) * norm(Xj - Xj1) + eps) * \
                     ((np.dot(Xj2 - Xj1, Xj - Xj1)) / (norm(Xj2 - Xj1) * norm(Xj - Xj1) + eps) - np.cos(theta0)) * \
                     ((Xj2 - Xj1) - np.dot(Xj2 - Xj1, Xj - Xj1) * (Xj - Xj1) / (norm(Xj - Xj1) ** 2 + eps))
            elif k == 1:
                if k - 1 < 0:
                    return V
                Xj_1, Xj = positions[k - 1], positions[k]
                if n >= 3 and k + 1 < n:
                    Xj1 = positions[k + 1]
                    V += kt_par / (norm(Xj1 - Xj) * norm(Xj_1 - Xj) + eps) * \
                         ((np.dot(Xj1 - Xj, Xj_1 - Xj)) / (norm(Xj1 - Xj) * norm(Xj_1 - Xj) + eps) - np.cos(theta0)) * \
                         ((Xj_1 - Xj) - np.dot(Xj1 - Xj, Xj_1 - Xj) * (Xj1 - Xj) / (norm(Xj1 - Xj) ** 2 + eps) + \
                          (Xj1 - Xj) - np.dot(Xj1 - Xj, Xj_1 - Xj) * (Xj_1 - Xj) / (norm(Xj_1 - Xj) ** 2 + eps))
                if n > 3 and k + 2 < n:
                    Xj2 = positions[k + 2]
                    V -= kt_par / (norm(Xj2 - Xj1) * norm(Xj - Xj1) + eps) * \
                         ((np.dot(Xj2 - Xj1, Xj - Xj1)) / (norm(Xj2 - Xj1) * norm(Xj - Xj1) + eps) - np.cos(theta0)) * \
                         ((Xj2 - Xj1) - np.dot(Xj2 - Xj1, Xj - Xj1) * (Xj - Xj1) / (norm(Xj - Xj1) ** 2 + eps))
            elif k == n - 1 and n >= 3:
                if k - 2 < 0:
                    return V
                Xj_2, Xj_1, Xj = positions[k - 2], positions[k - 1], positions[k]
                V -= kt_par / (norm(Xj - Xj_1) * norm(Xj_2 - Xj_1) + eps) * \
                     ((np.dot(Xj - Xj_1, Xj_2 - Xj_1)) / (norm(Xj - Xj_1) * norm(Xj_2 - Xj_1) + eps) - np.cos(theta0)) * \
                     ((Xj_2 - Xj_1) - np.dot(Xj - Xj_1, Xj_2 - Xj_1) * (Xj - Xj_1) / (norm(Xj - Xj_1) ** 2 + eps))
            elif k == n - 2:
                if k - 1 < 0:
                    return V
                Xj = positions[k]
                if n > 3 and k - 2 >= 0:
                    Xj_2, Xj_1 = positions[k - 2], positions[k - 1]
                    V -= kt_par / (norm(Xj - Xj_1) * norm(Xj_2 - Xj_1) + eps) * \
                         ((np.dot(Xj - Xj_1, Xj_2 - Xj_1)) / (norm(Xj - Xj_1) * norm(Xj_2 - Xj_1) + eps) - np.cos(theta0)) * \
                         ((Xj_2 - Xj_1) - np.dot(Xj - Xj_1, Xj_2 - Xj_1) * (Xj - Xj_1) / (norm(Xj - Xj_1) ** 2 + eps))
                if n >= 3 and k + 1 < n:
                    Xj1 = positions[k + 1]
                    Xj_1 = positions[k - 1]
                    V += kt_par / (norm(Xj1 - Xj) * norm(Xj_1 - Xj) + eps) * \
                         ((np.dot(Xj1 - Xj, Xj_1 - Xj)) / (norm(Xj1 - Xj) * norm(Xj_1 - Xj) + eps) - np.cos(theta0)) * \
                         ((Xj_1 - Xj) - np.dot(Xj1 - Xj, Xj_1 - Xj) * (Xj1 - Xj) / (norm(Xj1 - Xj) ** 2 + eps) + \
                          (Xj1 - Xj) - np.dot(Xj1 - Xj, Xj_1 - Xj) * (Xj_1 - Xj) / (norm(Xj_1 - Xj) ** 2 + eps))
            else:
                if k - 2 < 0 or k + 2 >= n:
                    return V
                Xj_2, Xj_1, Xj, Xj1, Xj2 = positions[k - 2], positions[k - 1], positions[k], positions[k + 1], positions[k + 2]
                V -= kt_par / (norm(Xj - Xj_1) * norm(Xj_2 - Xj_1) + eps) * \
                     ((np.dot(Xj - Xj_1, Xj_2 - Xj_1)) / (norm(Xj - Xj_1) * norm(Xj_2 - Xj_1) + eps) - np.cos(theta0)) * \
                     ((Xj_2 - Xj_1) - np.dot(Xj - Xj_1, Xj_2 - Xj_1) * (Xj - Xj_1) / (norm(Xj - Xj_1) ** 2 + eps))
                V += kt_par / (norm(Xj1 - Xj) * norm(Xj_1 - Xj) + eps) * \
                     ((np.dot(Xj1 - Xj, Xj_1 - Xj)) / (norm(Xj1 - Xj) * norm(Xj_1 - Xj) + eps) - np.cos(theta0)) * \
                     ((Xj_1 - Xj) - np.dot(Xj1 - Xj, Xj_1 - Xj) * (Xj1 - Xj) / (norm(Xj1 - Xj) ** 2 + eps) + \
                      (Xj1 - Xj) - np.dot(Xj1 - Xj, Xj_1 - Xj) * (Xj_1 - Xj) / (norm(Xj_1 - Xj) ** 2 + eps))
                V -= kt_par / (norm(Xj2 - Xj1) * norm(Xj - Xj1) + eps) * \
                     ((np.dot(Xj2 - Xj1, Xj - Xj1)) / (norm(Xj2 - Xj1) * norm(Xj - Xj1) + eps) - np.cos(theta0)) * \
                     ((Xj2 - Xj1) - np.dot(Xj2 - Xj1, Xj - Xj1) * (Xj - Xj1) / (norm(Xj - Xj1) ** 2 + eps))
            return V

        # トーションスプリング力の計算（bottom component）
        def torsion_spring_bot(k, positions, kt_bot, theta0, eps=1e-6):
            V = np.zeros(2)
            n = len(positions)

            def norm(vec):
                return np.linalg.norm(vec)

            # 必要な粒子数が揃っているか確認
            if n < 3:
                return V  # 粒子数が3未満の場合は計算しない

            if k == 0 and n >= 3:
                if k + 2 >= n:
                    return V
                Xj, Xj1, Xj2 = positions[k], positions[k + 1], positions[k + 2]
                # 以下、元の計算...
                V -= kt_bot / (norm(Xj2 - Xj1) * norm(Xj - Xj1) + eps) * \
                     ((np.cross(Xj2 - Xj1, Xj - Xj1)) / (norm(Xj2 - Xj1) * norm(Xj - Xj1) + eps) - np.sin(theta0)) * \
                     np.array([-(Xj2 - Xj1)[1] + ((Xj - Xj1)[0] / (norm(Xj - Xj1) ** 2 + eps)) * np.cross(Xj - Xj1, Xj2 - Xj1),
                               (Xj2 - Xj1)[0] + ((Xj - Xj1)[1] / (norm(Xj - Xj1) ** 2 + eps)) * np.cross(Xj - Xj1, Xj2 - Xj1)])
            elif k == 1:
                if k - 1 < 0:
                    return V
                Xj, Xj_1 = positions[k], positions[k - 1]
                if n >= 3 and k + 1 < n:
                    Xj1 = positions[k + 1]
                    V += kt_bot / (norm(Xj1 - Xj) * norm(Xj_1 - Xj) + eps) * \
                         ((np.cross(Xj1 - Xj, Xj_1 - Xj)) / (norm(Xj1 - Xj) * norm(Xj_1 - Xj) + eps) - np.sin(theta0)) * \
                         (np.array([(Xj_1 - Xj)[1] - (((Xj1 - Xj)[0]) / (norm(Xj1 - Xj) ** 2 + eps)) * np.cross(Xj1 - Xj, Xj_1 - Xj),
                                    -(Xj_1 - Xj)[0] - (((Xj1 - Xj)[1]) / (norm(Xj1 - Xj) ** 2 + eps)) * np.cross(Xj1 - Xj, Xj_1 - Xj)]) +
                          np.array([-(Xj1 - Xj)[1] + (((Xj_1 - Xj)[0]) / (norm(Xj_1 - Xj) ** 2 + eps)) * np.cross(Xj_1 - Xj, Xj1 - Xj),
                                    (Xj1 - Xj)[0] + (((Xj_1 - Xj)[1]) / (norm(Xj_1 - Xj) ** 2 + eps)) * np.cross(Xj_1 - Xj, Xj1 - Xj)]))
                if n > 3 and k + 2 < n:
                    Xj2 = positions[k + 2]
                    V -= kt_bot / (norm(Xj2 - Xj1) * norm(Xj - Xj1) + eps) * \
                         ((np.cross(Xj2 - Xj1, Xj - Xj1)) / (norm(Xj2 - Xj1) * norm(Xj - Xj1) + eps) - np.sin(theta0)) * \
                         np.array([-(Xj2 - Xj1)[1] + ((Xj - Xj1)[0] / (norm(Xj - Xj1) ** 2 + eps)) * np.cross(Xj - Xj1, Xj2 - Xj1),
                                   (Xj2 - Xj1)[0] + ((Xj - Xj1)[1] / (norm(Xj - Xj1) ** 2 + eps)) * np.cross(Xj - Xj1, Xj2 - Xj1)])
            elif k == n - 1 and n >= 3:
                if k - 2 < 0:
                    return V
                Xj_2, Xj_1, Xj = positions[k - 2], positions[k - 1], positions[k]
                V -= kt_bot / (norm(Xj - Xj_1) * norm(Xj_2 - Xj_1) + eps) * \
                     ((np.cross(Xj - Xj_1, Xj_2 - Xj_1)) / (norm(Xj - Xj_1) * norm(Xj_2 - Xj_1) + eps) - np.sin(theta0)) * \
                     np.array([(Xj_2 - Xj_1)[1] - ((Xj - Xj_1)[0]) / (norm(Xj - Xj_1) ** 2 + eps) * np.cross(Xj - Xj_1, Xj_2 - Xj_1),
                               -(Xj_2 - Xj_1)[0] - ((Xj - Xj_1)[1]) / (norm(Xj - Xj_1) ** 2 + eps) * np.cross(Xj - Xj_1, Xj_2 - Xj_1)])
            elif k == n - 2:
                if k - 1 < 0:
                    return V
                Xj = positions[k]
                if n > 3 and k - 2 >= 0:
                    Xj_2, Xj_1 = positions[k - 2], positions[k - 1]
                    V -= kt_bot / (norm(Xj - Xj_1) * norm(Xj_2 - Xj_1) + eps) * \
                         ((np.cross(Xj - Xj_1, Xj_2 - Xj_1)) / (norm(Xj - Xj_1) * norm(Xj_2 - Xj_1) + eps) - np.sin(theta0)) * \
                         np.array([(Xj_2 - Xj_1)[1] - ((Xj - Xj_1)[0]) / (norm(Xj - Xj_1) ** 2 + eps) * np.cross(Xj - Xj_1, Xj_2 - Xj_1),
                                   -(Xj_2 - Xj_1)[0] - ((Xj - Xj_1)[1]) / (norm(Xj - Xj_1) ** 2 + eps) * np.cross(Xj - Xj_1, Xj_2 - Xj_1)])
                if n >= 3 and k + 1 < n:
                    Xj1 = positions[k + 1]
                    Xj_1 = positions[k - 1]
                    V += kt_bot / (norm(Xj1 - Xj) * norm(Xj_1 - Xj) + eps) * \
                         ((np.cross(Xj1 - Xj, Xj_1 - Xj)) / (norm(Xj1 - Xj) * norm(Xj_1 - Xj) + eps) - np.sin(theta0)) * \
                         (np.array([(Xj_1 - Xj)[1] - (((Xj1 - Xj)[0]) / (norm(Xj1 - Xj) ** 2 + eps)) * np.cross(Xj1 - Xj, Xj_1 - Xj),
                                    -(Xj_1 - Xj)[0] - (((Xj1 - Xj)[1]) / (norm(Xj1 - Xj) ** 2 + eps)) * np.cross(Xj1 - Xj, Xj_1 - Xj)]) +
                          np.array([-(Xj1 - Xj)[1] + (((Xj_1 - Xj)[0]) / (norm(Xj_1 - Xj) ** 2 + eps)) * np.cross(Xj_1 - Xj, Xj1 - Xj),
                                    (Xj1 - Xj)[0] + (((Xj_1 - Xj)[1]) / (norm(Xj_1 - Xj) ** 2 + eps)) * np.cross(Xj_1 - Xj, Xj1 - Xj)]))
            else:
                if k - 2 < 0 or k + 2 >= n:
                    return V
                Xj_2, Xj_1, Xj, Xj1, Xj2 = positions[k - 2], positions[k - 1], positions[k], positions[k + 1], positions[k + 2]
                V -= kt_bot / (norm(Xj - Xj_1) * norm(Xj_2 - Xj_1) + eps) * \
                     ((np.cross(Xj - Xj_1, Xj_2 - Xj_1)) / (norm(Xj - Xj_1) * norm(Xj_2 - Xj_1) + eps) - np.sin(theta0)) * \
                     np.array([(Xj_2 - Xj_1)[1] - ((Xj - Xj_1)[0]) / (norm(Xj - Xj_1) ** 2 + eps) * np.cross(Xj - Xj_1, Xj_2 - Xj_1),
                               -(Xj_2 - Xj_1)[0] - ((Xj - Xj_1)[1]) / (norm(Xj - Xj_1) ** 2 + eps) * np.cross(Xj - Xj_1, Xj_2 - Xj_1)])
                V += kt_bot / (norm(Xj1 - Xj) * norm(Xj_1 - Xj) + eps) * \
                     ((np.cross(Xj1 - Xj, Xj_1 - Xj)) / (norm(Xj1 - Xj) * norm(Xj_1 - Xj) + eps) - np.sin(theta0)) * \
                     (np.array([(Xj_1 - Xj)[1] - (((Xj1 - Xj)[0]) / (norm(Xj1 - Xj) ** 2 + eps)) * np.cross(Xj1 - Xj, Xj_1 - Xj),
                                -(Xj_1 - Xj)[0] - (((Xj1 - Xj)[1]) / (norm(Xj1 - Xj) ** 2 + eps)) * np.cross(Xj1 - Xj, Xj_1 - Xj)]) +
                      np.array([-(Xj1 - Xj)[1] + (((Xj_1 - Xj)[0]) / (norm(Xj_1 - Xj) ** 2 + eps)) * np.cross(Xj_1 - Xj, Xj1 - Xj),
                                (Xj1 - Xj)[0] + (((Xj_1 - Xj)[1]) / (norm(Xj_1 - Xj) ** 2 + eps)) * np.cross(Xj_1 - Xj, Xj1 - Xj)]))
                V -= kt_bot / (norm(Xj2 - Xj1) * norm(Xj - Xj1) + eps) * \
                     ((np.cross(Xj2 - Xj1, Xj - Xj1)) / (norm(Xj2 - Xj1) * norm(Xj - Xj1) + eps) - np.sin(theta0)) * \
                     np.array([-(Xj2 - Xj1)[1] + ((Xj - Xj1)[0] / (norm(Xj - Xj1) ** 2 + eps)) * np.cross(Xj - Xj1, Xj2 - Xj1),
                               (Xj2 - Xj1)[0] + ((Xj - Xj1)[1] / (norm(Xj - Xj1) ** 2 + eps)) * np.cross(Xj - Xj1, Xj2 - Xj1)])
            return V


        def compute_normalized_torsion_energy(positions, kt_par, kt_bot, theta0, eps=1e-6):

            N = len(positions)
            if N < 3:
                return 0.0, 0  # バネが存在しない場合はエネルギー 0, バネ数 0

            energy_parallel_sum = 0.0
            energy_perp_sum = 0.0
            count = 0  # 有効バネの数

            for j in range(1, N - 1):
                vec1 = positions[j-1] - positions[j ]
                vec2 = positions[j + 1] - positions[j]
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)

                if norm1 < eps or norm2 < eps:
                    continue  # 0除算を避ける -> このバネはスキップ

                cos_angle = np.dot(vec2, vec1) / (norm1 * norm2)
                sin_angle = (vec1[0] * vec2[1] - vec1[1] * vec2[0]) / (norm1 * norm2)

                # 並列成分
                energy_parallel = 0.5 * kt_par * (cos_angle - np.cos(theta0))**2
                # 垂直成分
                energy_perp = 0.5 * kt_bot * (sin_angle - np.sin(theta0))**2

                energy_parallel_sum += energy_parallel
                energy_perp_sum     += energy_perp
                count += 1

            if count == 0:
                # バネが全部スキップされたら 0 と返す
                return 0.0, 0

            total_energy = energy_parallel_sum + energy_perp_sum
            #print(energy_parallel_sum , energy_perp_sum)
            normalized_energy = total_energy / count
            return normalized_energy, count


        # 接着バネの力を計算する関数
        def adhesion_force(position, k_adhesion, ground_position):
            distance_vector = ground_position - position
            force = k_adhesion * distance_vector
            return force, np.linalg.norm(force)

        def get_num_adhesive_springs(particle_id):
            # この関数は、粒子IDに対応する接着バネの数を返します
            # ここでは仮に、すべての粒子に1つの接着バネが付くとします
            return 1  # または他のロジックに従って数を返すように実装します


        # 接着バネの力を適用する関数
        adhesion_points = {}
        # 初期化時に adhesion_points をクリア
        adhesion_points.clear()


        max_adhesive_springs = 50

        # 接着バネの定数
        k_adhesion = 0.004 # 接着バネ定数
        adhesion_break_threshold = 0.24  # バネが壊れるしきい値
        lambda_poisson = 1.25

        # 接着バネの力を計算する関数
        def adhesion_force(position, k_adhesion, ground_position):
            distance_vector = ground_position - position
            force = k_adhesion * distance_vector
            return force, np.linalg.norm(force)




        # 接着バネの力を適用する関数
        def apply_adhesion_forces(chain, k_adhesion, adhesion_break_threshold, current_step, current_time):
            positions, velocities, ids, split_index, chain_id, parent_chain_id = chain
            forces = [np.zeros(2) for _ in positions]  # 各粒子の力を初期化

            for i in [0, len(positions) - 1]:  # 先頭と最後尾の粒子のみ対象
                if ids[i] not in adhesion_points:
                    continue

                remaining_springs = []
                for ground_position in adhesion_points[ids[i]]:
                    # 接着バネの力を計算
                    force, force_magnitude = adhesion_force(positions[i], k_adhesion, ground_position)

                    # 力の大きさがしきい値を超えている場合、バネが壊れる
                    if force_magnitude > adhesion_break_threshold:
                        print(f"Step {current_step}: Adhesion spring at particle {ids[i]} broke due to excessive force at time {current_time:.4f} (Force magnitude: {force_magnitude:.4f}, Threshold: {adhesion_break_threshold}).")

                        # 壊れたタイミングを履歴に追加
                        adhesion_history.append({
                            'time': current_time,
                            'particle_id': ids[i],
                            'position': positions[i].copy(),
                            'action': 'broken'
                        })
                        # 破壊されたバネは remaining_springs に追加しない
                    else:
                        # 力を適用
                        forces[i] += force
                        # バネはまだ有効なので remaining_springs に追加
                        remaining_springs.append(ground_position)

                # 残ったバネだけを保持
                if remaining_springs:
                    adhesion_points[ids[i]] = remaining_springs
                else:
                    del adhesion_points[ids[i]]
                    print(f"Step {current_step}: All adhesion springs removed for particle {ids[i]}.")

            return forces



        # 記録用
        all_positions = []
        all_times = []  # 各ステップの時間を記録
        add_particle_interval_steps = 1.0
        particle_addition_log = []
        adhesion_points_history = []

        # カラーマップの設定
        chain_colors = ['gray']
        next_color_index = len(chain_colors)  # 新しい色の割り当てに使用するインデックス

        # 粒子鎖の色を追跡するリスト
        particle_chain_colors = [chain_colors[0]]  # 最初のチェインは灰色
        all_chain_colors = []

        # チェインIDのカウンターを初期化
        chain_id_counter = 0  # チェインIDのカウンター

        # シミュレーション用のチェインを初期化
        particle_chains = [(initial_positions, initial_velocities, initial_ids, None, chain_id_counter, None)]
        chain_id_counter += 1

        # チェインごとのデータを保持する辞書を初期化
        chain_growth_rates = {}
        chain_L_theoretical = {}
        chain_L_actual = {}
        chain_split_thresholds = {}
        parent_chain_ids = {}

        
        
        # 初期状態の履歴を保存する辞書
        chain_lengths_theoretical = defaultdict(list)
        chain_lengths_actual = defaultdict(list)
        chain_times = defaultdict(list)
        chain_particle_counts = defaultdict(list)  # 必要なら初期状態も記録する




        # 位置の範囲を設定
        def calculate_bounds(all_positions):
            x_min = min(pos[0] for frame in all_positions for chain in frame for pos in chain[0])
            x_max = max(pos[0] for frame in all_positions for chain in frame for pos in chain[0])
            y_min = min(pos[1] for frame in all_positions for chain in frame for pos in chain[0])
            y_max = max(pos[1] for frame in all_positions for chain in frame for pos in chain[0])
            return x_min, x_max, y_min, y_max


        adhesion_history = []

        # 接着バネの発生時刻を記録する辞書を初期化（この行を追加）
        adhesion_creation_time = {}



        division_history = []
        chain_lengths_over_time = []
        time_steps = []

        #front_shift = -4.520499999999779#ks=300
        front_shift = 0.0
        

        dummy_L_data=[]
        repulsion_force_history = []  # 各タイムステップでの「反発力絶対値合計」を記録
        time_steps_repulsion   = []   # その時刻を記録 (repulsion_force_historyと同じ長さになる)
        number_of_chains_history = []
        repulsion_energy_time=[]
        torsion_energy_time=[]
        
        
        # --- 初期チェインに対してデータを設定 ---
        pole_kind[initial_ids[0]]  = 'new'
        pole_kind[initial_ids[-1]] = 'new'
        for idx, (positions, velocities, ids, _, chain_id, parent_chain_id) in enumerate(particle_chains):
            # 成長レートを割り当て
            growth_rate = g_r
            chain_growth_rates[chain_id] = growth_rate

            N = len(positions)
            L_actual = 2 * R + sum(np.linalg.norm(positions[i + 1] - positions[i]) for i in range(len(positions) - 1))
            print(L_actual,11111)
            chain_L_actual[chain_id] = L_actual
            chain_L_theoretical[chain_id] = L_actual

            split_threshold = 121.43950000000001
            chain_split_thresholds[chain_id] = split_threshold

            # 親チェインIDを設定
            parent_chain_ids[chain_id] = parent_chain_id

        L0=L_actual
        chain0_info = {}
        
        
        if 0 in chain_growth_rates:  # チェイン0が存在すれば
            chain0_info["growth_rate"] = chain_growth_rates[0]
            chain0_info["L_initial"]   = chain_L_theoretical[0]
            chain0_info["threshold"]   = chain_split_thresholds[0]


        current_positions = []
        # --- シミュレーション開始前に初期状態を記録する（t = 0） ---
        for idx, (positions, velocities, ids, split_index, chain_id, parent_chain_id) in enumerate(particle_chains):
            chain_lengths_theoretical[chain_id].append(chain_L_theoretical[chain_id])
            chain_lengths_actual[chain_id].append(chain_L_actual[chain_id])
            chain_times[chain_id].append(0)  # t = 0 の時刻を記録
            chain_particle_counts[chain_id].append(len(ids))
            current_positions.append((positions.copy(), ids.copy()))

        all_positions.append(current_positions)
        all_times.append(current_time)  # current_time は0のはず
        

        # 初期のチェインカラーも保存する
        all_chain_colors.append(particle_chain_colors.copy())
        adhesion_points_history.append(adhesion_points.copy())
        adhesion_time_series = []




        try:
            next_sample_time = 1.0
            while current_time < t_max and len(particle_chains) <= max_chains:
                #max_force = 0
                #### ①【分割処理の判定】（旧：位置更新後に行っていた部分を先頭へ移動）
                updated_chains = []
                updated_chain_colors = []
                updated_chain_growth_rates = {}
                updated_chain_L_theoretical = {}
                updated_chain_L_actual = {}
                updated_chain_split_thresholds = {}
                for chain_idx, (positions, velocities, ids, split_index, chain_id, parent_chain_id) in enumerate(particle_chains):
                    total_distance = 2 * R + sum(np.linalg.norm(positions[i + 1] - positions[i]) for i in range(len(positions) - 1))
                    split_threshold = 121.43950000000001
                    if total_distance > split_threshold:
                        # 分割する場合（元の分割ロジックそのまま）
                        half = len(positions) // 2
                        chain1_positions = positions[:half]
                        chain1_velocities = velocities[:half]
                        chain1_ids = ids[:half]
                        chain2_positions = positions[half:]
                        chain2_velocities = velocities[half:]
                        chain2_ids = ids[half:]
                        # ランダムシフト
                        chain1_positions = [random_shift(pos) for pos in chain1_positions]
                        chain2_positions = [random_shift(pos) for pos in chain2_positions]
                        
                        # ───────── 新旧ポールのラベル付け ─────────
                        #   chain1_ids[0]        : 母由来 old
                        #   chain1_ids[-1]       : セプタム new
                        #   chain2_ids[0]        : セプタム new
                        #   chain2_ids[-1]       : 母由来 old
                        pole_kind[chain1_ids[0]]  = 'old'
                        pole_kind[chain2_ids[-1]] = 'old'
                        pole_kind[chain1_ids[-1]] = 'new'
                        pole_kind[chain2_ids[0]]  = 'new'

                        # 端でなくなった粒子のラベルは削除（None 扱いにする）
                        for pid in (set(chain1_ids[1:-1]) | set(chain2_ids[1:-1])):
                            pole_kind.pop(pid, None)
                        # ────────────────────────────────────
                        
                        #new_growth_rate1 = max(np.random.normal(loc=mean_interval, scale=std_interval), 0.001)
                        #new_growth_rate2 = max(np.random.normal(loc=mean_interval, scale=std_interval), 0.001)
                        
                        new_growth_rate1 = g_r
                        new_growth_rate2 = g_r
                        print(f"Time {current_time:.4f}: Split occurred.")
                        print(f"Chain 1: {chain1_positions}")
                        print(f"Chain 2: {chain2_positions}")
                        print(f"growth_rate 1: {new_growth_rate1}")
                        print(f"growth_rate 2: {new_growth_rate2}")
                        split_particle_id = ids[half - 1]
                        chain_id1 = chain_id
                        chain_id2 = chain_id_counter
                        chain_id_counter += 1
                        division_history.append({
                            'time': current_time,
                            'chain_id': chain_id,
                            'split_particle_id': split_particle_id,
                            'particle_ids_before_split': ids.copy(),
                            'particle_ids_chain1': chain1_ids.copy(),
                            'particle_ids_chain2': chain2_ids.copy(),
                            'new_chain_ids': [chain_id1, chain_id2]
                        })
                        L_actual1 = 2 * R + sum(np.linalg.norm(chain1_positions[i + 1] - chain1_positions[i]) for i in range(len(chain1_positions) - 1))
                        L_actual2 = 2 * R + sum(np.linalg.norm(chain2_positions[i + 1] - chain2_positions[i]) for i in range(len(chain2_positions) - 1))
                        L_d = chain_L_theoretical[chain_id]
                        N_actual1 = len(chain1_positions)
                        N_actual2 = len(chain2_positions)
                        # 親チェインの理論長を各新チェインに粒子数の比率で分割
                        L_theoretical1 = L_d * (N_actual1) / (N_actual1 + N_actual2)
                        L_theoretical2 = L_d * (N_actual2) / (N_actual1 + N_actual2)
                        print(L_theoretical1,L_theoretical2 )

                        new_split_threshold1 = 121.43950000000001
                        new_split_threshold2 = 121.43950000000001
                        updated_chains.append((chain1_positions, chain1_velocities, chain1_ids, None, chain_id1, chain_id))
                        updated_chain_growth_rates[chain_id1] = new_growth_rate1
                        updated_chain_L_theoretical[chain_id1] = L_theoretical1
                        updated_chain_L_actual[chain_id1] = L_actual1
                        updated_chain_split_thresholds[chain_id1] = new_split_threshold1
                        parent_chain_ids[chain_id1] = chain_id
                        # 親チェインの履歴を引き継ぐ（存在すれば）
                        chain_lengths_theoretical[chain_id1] = chain_lengths_theoretical.get(chain_id, []).copy()
                        chain_lengths_actual[chain_id1] = chain_lengths_actual.get(chain_id, []).copy()
                        chain_times[chain_id1] = chain_times.get(chain_id, []).copy()
                        chain_particle_counts[chain_id1] = chain_particle_counts.get(chain_id, [])[:len(chain_times.get(chain_id, []))].copy()
                        #chain_lengths_theoretical[chain_id1].append(L_theoretical1)
                        #chain_lengths_actual[chain_id1].append(L_actual1)
                        #chain_times[chain_id1].append(current_time)
                        #chain_particle_counts[chain_id1].append(len(chain1_positions))
                        updated_chains.append((chain2_positions, chain2_velocities, chain2_ids, None, chain_id2, chain_id))
                        updated_chain_growth_rates[chain_id2] = new_growth_rate2
                        updated_chain_L_theoretical[chain_id2] = L_theoretical2
                        updated_chain_L_actual[chain_id2] = L_actual2
                        updated_chain_split_thresholds[chain_id2] = new_split_threshold2
                        parent_chain_ids[chain_id2] = chain_id
                        #chain_lengths_theoretical[chain_id2] = [L_theoretical2]
                        #chain_lengths_actual[chain_id2] = [L_actual2]
                        #chain_times[chain_id2] = [current_time]
                        #chain_particle_counts[chain_id2] = [len(chain2_positions)]
                        chain_lengths_theoretical[chain_id2] = []
                        chain_lengths_actual[chain_id2] = []
                        chain_times[chain_id2] = []
                        chain_particle_counts[chain_id2] = []

                        updated_chain_colors.append(chain_colors[next_color_index % len(chain_colors)])
                        next_color_index += 1
                        updated_chain_colors.append(chain_colors[next_color_index % len(chain_colors)])
                        next_color_index += 1
                    else:
                        updated_chains.append((positions, velocities, ids, split_index, chain_id, parent_chain_id))
                        updated_chain_growth_rates[chain_id] = chain_growth_rates[chain_id]
                        updated_chain_L_theoretical[chain_id] = chain_L_theoretical[chain_id]
                        updated_chain_L_actual[chain_id] = chain_L_actual[chain_id]
                        updated_chain_split_thresholds[chain_id] = chain_split_thresholds[chain_id]
                        updated_chain_colors.append(particle_chain_colors[chain_idx])
                # 更新後のチェイン情報で particle_chains を上書き
                particle_chains = updated_chains
                particle_chain_colors = updated_chain_colors
                chain_growth_rates = updated_chain_growth_rates
                chain_L_theoretical = updated_chain_L_theoretical
                chain_L_actual = updated_chain_L_actual
                chain_split_thresholds = updated_chain_split_thresholds

                #### ②【集団成長】（長さ更新＋新規粒子追加）
                new_chains = []
                new_chain_growth_rates = {}
                new_chain_L_theoretical = {}
                new_chain_L_actual = {}
                new_chain_split_thresholds = {}
                new_chain_colors = []
                current_positions = []
                for chain_idx, (positions, velocities, ids, split_index, chain_id, parent_chain_id) in enumerate(particle_chains):
                    # 理論長さの更新（成長）
                    growth_rate = chain_growth_rates[chain_id]
                    L_theoretical = chain_L_theoretical[chain_id]
                    
                    L_theoretical += growth_rate * L_theoretical * dt
                    #print(dt,L_theoretical,current_time)
                    chain_L_theoretical[chain_id] = L_theoretical
                    # 実際の長さの更新
                    L_actual = 2 * R + sum(np.linalg.norm(positions[i + 1] - positions[i]) for i in range(len(positions) - 1))
                    chain_L_actual[chain_id] = L_actual
                    dummy_time = current_time - front_shift
                    dummy_L = L0 * np.exp(growth_rate * dummy_time)
                    dummy_L_data.append(dummy_L)
                    # 粒子数の目標値計算
                    N_theoretical = int(np.ceil((L_theoretical - 2 * R) / l)) + 1
                    N_actual = len(positions)
                    num_new_particles = N_theoretical - N_actual
                    new_chains.append((positions, velocities, ids, split_index, chain_id, parent_chain_id))
                    new_chain_growth_rates[chain_id] = growth_rate
                    new_chain_L_theoretical[chain_id] = L_theoretical
                    new_chain_L_actual[chain_id] = chain_L_actual[chain_id]
                    new_chain_split_thresholds[chain_id] = chain_split_thresholds[chain_id]
                    new_chain_colors.append(particle_chain_colors[chain_idx])
                    if num_new_particles > 0:
                        for _ in range(num_new_particles):
                            if len(positions) == 1:
                                insert_position = 0
                                new_id = global_particle_id_counter
                                global_particle_id_counter += 1
                                new_position = positions[insert_position] - np.array([l, 0.0])
                                new_velocity = np.array([0.0, 0.0])
                                positions.insert(insert_position, new_position)
                                velocities.insert(insert_position, new_velocity)
                                ids.insert(insert_position, new_id)
                                particle_creation_history.append({'particle_id': new_id, 'time_created': current_time})
                                print(f"Time {current_time:.4f}: Particle added to chain {chain_idx} at the beginning at {new_position} with ID {new_id}.")
                                particle_addition_log.append({'chain_idx': chain_idx, 'particle_id': new_id, 'time_added': current_time})
                            else:
                                new_id = global_particle_id_counter
                                global_particle_id_counter += 1
                                insert_index = np.random.randint(0, len(positions) - 1)
                                new_position = (positions[insert_index] + positions[insert_index + 1]) / 2
                                new_velocity = np.array([0.0, 0.0])
                                positions.insert(insert_index + 1, new_position)
                                velocities.insert(insert_index + 1, new_velocity)
                                ids.insert(insert_index + 1, new_id)
                                particle_creation_history.append({'particle_id': new_id, 'time_created': current_time})
                                print(f"Time {current_time:.4f}: Particle added to chain {chain_idx} between particles {insert_index} and {insert_index + 1} at {new_position} with ID {new_id}.")
                                particle_addition_log.append({'chain_idx': chain_idx, 'particle_id': new_id, 'time_added': current_time})


                
                #### ③【機械的相互作用の計算と位置更新】
                total_forces = {}
                for chain_idx, (positions, velocities, ids, split_index, chain_id, parent_chain_id) in enumerate(particle_chains):
                    forces = [np.zeros(2) for _ in positions]
                    # Linear Spring の力
                    for i in range(len(positions) - 1):
                        distance = positions[i + 1] - positions[i]
                        distance_norm = np.linalg.norm(distance)
                        ep = 1e-8
                        if abs(distance_norm - l) < ep:
                            force_spring = np.zeros(2)
                        else:
                            force_spring = k_s * (distance_norm - l) / (l ** 2) * (distance / distance_norm)
                        forces[i] += force_spring
                        forces[i + 1] -= force_spring
                    # Torsion Spring の力
                    for i in range(len(forces)):
                        forces[i] += torsion_spring_par(i, positions, kt_par, theta0)
                        forces[i] += torsion_spring_bot(i, positions, kt_bot, theta0)
                    # 接着バネの追加処理（必要に応じて）
                    if current_time > 0:
                        for i in [0, len(positions) - 1]:
                            p_add_spring = 1 - np.exp(-lambda_poisson * dt)
                            rand_value = np.random.rand()
                            if rand_value < p_add_spring:
                                if ids[i] not in adhesion_points:
                                    adhesion_points[ids[i]] = []
                                if len(adhesion_points[ids[i]]) < max_adhesive_springs:
                                    ground_position = positions[i].copy()
                                    adhesion_points[ids[i]].append(ground_position)
                                    adhesion_history.append({'time': current_time, 'particle_id': ids[i],
                                                              'position': positions[i].copy(), 'action': 'added'})
                                    particle_position = 'start' if i == 0 else 'end'
                                    chain_length = len(positions)
                                    print(f"\033[32mAdhesion spring added to particle {i} (ID: {ids[i]}) in chain {chain_id} of length {chain_length}, which is at the {particle_position}, at position {ground_position} at time {current_time:.4f}.\033[0m")
                                else:
                                    particle_position = 'start' if i == 0 else 'end'
                                    print(f"\033[33mParticle {i} (ID: {ids[i]}), which is at the {particle_position} of chain {chain_id}, already has the maximum number of adhesive springs ({max_adhesive_springs}).\033[0m")
                    adhesion_forces = apply_adhesion_forces(
                        (positions, velocities, ids, split_index, chain_id, parent_chain_id),
                        k_adhesion, adhesion_break_threshold, int(current_time / dt), current_time)
                    forces = [f + af for f, af in zip(forces, adhesion_forces)]
                    total_forces[chain_id] = forces
                
                repulsion_force_step_sum = 0.0
                total_repulsion_energy = 0.0
                contact_count = 0
                for i in range(len(particle_chains)):
                    positions_i, velocities_i, ids_i, _, chain_id_i, _ = particle_chains[i]
                    forces_i = total_forces[chain_id_i]
                    for j in range(i + 1, len(particle_chains)):
                        positions_j, velocities_j, ids_j, _, chain_id_j, _ = particle_chains[j]
                        forces_j = total_forces[chain_id_j]
                        for idx_i, pos_i in enumerate(positions_i):
                            for idx_j, pos_j in enumerate(positions_j):
                                force = simple_repulsion_force(pos_i, pos_j, k_c, R)
                                force_norm = np.linalg.norm(force)
                                if force_norm > 0:
                                    forces_i[idx_i] += force
                                    forces_j[idx_j] -= force
                                distance = np.linalg.norm(pos_i - pos_j)
                                distance_ratio = distance / (2 * R)
                                energy_contribution = phi(distance_ratio)
                                total_repulsion_energy += energy_contribution
                                if distance <= 2 * R:
                                    contact_count += 1
                        total_forces[chain_id_i] = forces_i
                normalized_energy = total_repulsion_energy / contact_count if contact_count > 0 else 0.0
                repulsion_energy_time.append((current_time, normalized_energy, contact_count))
                
                # 位置の更新（オーバーダンピング仮定）
                for chain_idx, (positions, velocities, ids, split_index, chain_id, parent_chain_id) in enumerate(particle_chains):
                    forces = total_forces[chain_id]
                    # 位置更新
                    for i in range(len(positions)):
                        velocities[i] = forces[i]
                        positions[i] = positions[i] + velocities[i] * dt
                    # 最新の粒子配置に基づいて L_actual を再計算
                    L_actual = 2 * R + sum(np.linalg.norm(positions[i + 1] - positions[i]) for i in range(len(positions) - 1))
                    chain_L_actual[chain_id] = L_actual
                    new_chain_L_actual[chain_id] = L_actual  # ← この行を追加
                    chain_particle_counts[chain_id].append(len(positions))
                    current_positions.append((positions.copy(), ids.copy()))
                    
                if current_time > 0:
                    for idx, (positions, velocities, ids, split_index, chain_id, parent_chain_id) in enumerate(particle_chains):
                        if chain_id not in chain_lengths_theoretical:
                            if parent_chain_id is not None:
                                chain_lengths_theoretical[chain_id] = chain_lengths_theoretical[parent_chain_id].copy()
                                chain_lengths_actual[chain_id] = chain_lengths_actual[parent_chain_id].copy()
                            else:
                                chain_lengths_theoretical[chain_id] = []
                                chain_lengths_actual[chain_id] = []
                        if chain_id not in chain_times:
                            chain_times[chain_id] = []
                        chain_lengths_theoretical[chain_id].append(chain_L_theoretical[chain_id])
                        chain_lengths_actual[chain_id].append(chain_L_actual[chain_id])
                        chain_times[chain_id].append(current_time)
                
                
                
                if current_time + dt >= next_sample_time:
                    dt = next_sample_time - current_time

                # ここで、dt が決まった時点で current_time を更新
                current_time += dt
                # [Rounding Fix] 数値誤差の丸め（必要に応じて）
                if abs(current_time - round(current_time)) < 1e-8:
                    current_time = round(current_time)
                # 固定サンプリングの更新
                if abs(current_time - next_sample_time) < 1e-8:
                    next_sample_time += 1.0
                #### 各種記録・更新（サンプルデータ、時間更新などは元と同様）
                all_positions.append(current_positions)
                all_times.append(current_time)
                all_chain_colors.append(new_chain_colors.copy())
                particle_chains = new_chains
                particle_chain_colors = new_chain_colors
                chain_growth_rates = new_chain_growth_rates
                chain_L_theoretical = new_chain_L_theoretical
                chain_L_actual = new_chain_L_actual
                chain_split_thresholds = new_chain_split_thresholds
                repulsion_force_history.append(repulsion_force_step_sum)
                time_steps_repulsion.append(current_time)
                adhesion_points_history.append(adhesion_points.copy())
                current_chain_lengths = [chain_L_actual[chain[4]] for chain in particle_chains]
                chain_lengths_over_time.append(current_chain_lengths)
                time_steps.append(current_time)
                # ←ここで記録も行う
                for pid, springs in adhesion_points.items():
                    adhesion_time_series.append({
                        'time'     : current_time,
                        'particle' : pid,
                        'adhesion' : len(springs),
                        'pole_kind': pole_kind.get(pid)   # ← この時点のラベルを凍結
                    })

                


                                    # ====== ここで torsion_energy を計算して記録 =======
                total_energy_all = 0.0      # 全チェインの非正規化エネルギーの合計
                total_bond_count_all = 0    # 全チェインのバネ数の合計

                for chain in particle_chains:
                    positions, velocities, ids, split_index, chain_id, parent_chain_id = chain

                    # 各チェインごとに正規化エネルギーとバネ数を取得
                    normalized_energy, count_bonds = compute_normalized_torsion_energy(positions, kt_par, kt_bot, theta0)

                    # 非正規化のエネルギーは (normalized_energy * count_bonds) です
                    total_energy_all += normalized_energy * count_bonds
                    total_bond_count_all += count_bonds

                # 全体での正規化エネルギーを計算
                if total_bond_count_all > 0:
                    overall_normalized_torsion_energy = total_energy_all / total_bond_count_all
                else:
                    overall_normalized_torsion_energy = 0.0

                # 例として、各タイムステップの記録用リストに (time, overall_normalized_torsion_energy, total_bond_count_all) を追加
                torsion_energy_time.append((current_time, overall_normalized_torsion_energy, total_bond_count_all))
                
                # すべてのチェインのforcesを集めて最大力を計算
    
                
                all_forces = []
                for forces in total_forces.values():
                    all_forces.extend(forces)
                if all_forces:
                    forces_magnitudes = [np.linalg.norm(f) for f in all_forces]
                    max_force = max(forces_magnitudes)
                else:
                    max_force = 0

                # 時間ステップを計算
                if max_force > 0:
                    dt = min(initial_dt, c1*R / (max_force), 1/(lambda_poisson+eps))
                else:
                    dt = initial_dt
                    

                    
               



        except Exception as e:
            print(f"An error occurred: {e}")
            
            
        finally:
        
            print("[DEBUG-save] first 10 entries of adhesion_time_series:")
            for rec in adhesion_time_series[:10]:
                print(rec)

            if chain0_info:
                growth_rate_0   = chain0_info["growth_rate"]
                L_initial_0     = chain0_info["L_initial"]
                split_threshold = chain0_info["threshold"]

                # --- 純粋な理論的分割時間 T_theory を計算 ---
                #     L(t) = L0 * exp(growth_rate * t) = split_threshold
                #     => t = (1 / growth_rate) * ln(split_threshold / L0)
                T_theoretical_0 = (1.0 / growth_rate_0) * np.log(split_threshold / L_initial_0)

                # --- division_history から "chain_id=0" の最初の分割時間を探す ---
                T_actual_0 = None
                for event in division_history:
                    if event.get("chain_id") == 0:
                        T_actual_0 = event["time"]
                        break

                # --- 結果を表示 ---
                print("\n--- Chain 0 Division Info ---")
                print(f"Growth rate      : {growth_rate_0}")
                print(f"Initial length   : {L_initial_0}")
                print(f"Split threshold  : {split_threshold}")

                print(f"Theoretical time : {T_theoretical_0:.3f}")

                if T_actual_0 is not None:
                    print(f"Actual time      : {T_actual_0:.3f}")
                    dt = T_actual_0 - T_theoretical_0
                    print(f"Difference (actual - theoretical): {dt:.3f}")
                else:
                    print("No division event found for chain 0.\n")
            else:
                print("No chain0_info found. Possibly chain 0 was never created.\n")

            import matplotlib.colors as mcolors
            import matplotlib.pyplot as plt
            from matplotlib import cm

            # シミュレーション終了後に粒子追加ログをデータフレームとして表示
            if particle_addition_log:
                df_addition = pd.DataFrame(particle_addition_log)
                print("\n--- Particle Addition Log ---")
                print(df_addition)
            else:
                print("No particles were added during the simulation.")

            max_adhesive_springs = 30
            tolerance_factor = 0.005
            desired_frame_count = 500

            # エラーが発生しても、正常に終了してもアニメーションを保存
            print("Saving animation...")
            frame_indices = np.linspace(0, len(all_positions) - 1, num=desired_frame_count).astype(int)
            all_positions_sampled = [all_positions[i] for i in frame_indices]
            all_chain_colors_sampled = [all_chain_colors[i] for i in frame_indices]
            all_times_sampled = [all_times[i] for i in frame_indices]

            # 位置の範囲を計算
            x_min, x_max, y_min, y_max = calculate_bounds(all_positions_sampled)

            # アニメーションの設定と保存
            fig, ax = plt.subplots()
            ax.set_xlim(x_min - 2 * R, x_max + 2 * R)
            ax.set_ylim(y_min - 2 * R, y_max + 2 * R)

            # カラーマップから色を取得する関数
            def get_color_from_cmap(value, min_value, max_value, cmap):
                normalized_value = (value - min_value) / (max_value - min_value)  # 値を0から1の範囲に正規化
                color = cmap(normalized_value)
                return color

            # 白から赤のカスタムカラーマップを定義
            cdict = {
                'red':   ((0.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
                'green': ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
                'blue':  ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0))
            }

            custom_cmap = mcolors.LinearSegmentedColormap('WhiteRed', cdict)

            # カラーバーの範囲を設定（最小値は1、最大値は max_adhesive_springs）
            norm = mcolors.Normalize(vmin=1, vmax=max_adhesive_springs)

            # ScalarMappable オブジェクトの作成（カラーバー用）
            sm = cm.ScalarMappable(cmap=custom_cmap, norm=norm)
            sm.set_array([])  # 必須ではないが、カラーバーの作成に必要

            # カラーバーの追加（これを一度だけ行う）
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
            cbar.set_label('Number of Adhesion Springs')

            def sample_adhesion_points_history(adhesion_points_history, total_frames):
                return [adhesion_points_history[int(i)] for i in np.linspace(0, len(adhesion_points_history) - 1, total_frames).astype(int)]

            # サンプリングされたデータを使用
            sampled_adhesion_points_history = sample_adhesion_points_history(adhesion_points_history, desired_frame_count)

            # ======================= ここから追加・変更部分 =======================
            def get_spring_color(distance, equilibrium_length, tolerance):
                """
                4つの区間で色分け:
                  1) distance < (l - 1.25*t) → 白 (white)
                  2) (l - 1.25*t) ≤ distance < (l - t) → 白→暗緑 (white→dark green) のグラデーション
                  3) (l - t) ≤ distance ≤ (l + t) → 黒 (black)
                  4) distance > (l + t) → 赤 (red)
                """
                # グラデーションの開始点を狭める
                lower_grad_start = equilibrium_length*0.7  # 白→暗緑開始点
                lower_black_zone = equilibrium_length - tolerance
                upper_black_zone = equilibrium_length + tolerance

                if distance > upper_black_zone:
                    # 4) [l + t, ∞) → 赤
                    return 'red'
                elif distance >= lower_black_zone:
                    # 3) [l - t, l + t] → 黒
                    return 'black'
                elif distance < lower_grad_start:
                    # 1) (-∞, l - 1.25t) → 白
                    return 'white'
                else:
                    # 2) [l - 1.25t, l - t) → 白→暗緑 の線形補間
                    alpha = (distance - lower_grad_start) / (lower_black_zone - lower_grad_start)
                    alpha = max(0.0, min(1.0, alpha))  # 0～1にクリップ

                    # 白 (1,1,1) → 暗緑 (0,0.5,0) への補間
                    r = 1.0 - alpha * 1.0  # 1→0
                    g = 1.0 - alpha * 0.5  # 1→0.5
                    b = 1.0 - alpha * 1.0  # 1→0
                    return (r, g, b)  # (R, G, B)
            # ======================= ここまで追加・変更部分 =======================

            # 初期化関数
            def init():
                return []

            # フレーム更新関数
            def update(frame_idx):
                frame = all_positions_sampled[frame_idx]
                colors = all_chain_colors_sampled[frame_idx]
                sim_time = all_times_sampled[frame_idx]
                ax.clear()
                ax.set_xlim(x_min - 2 * R, x_max + 2 * R)
                ax.set_ylim(y_min - 2 * R, y_max + 2 * R)
                elements = []

                # sim_time に最も近い adhesion_points_history のエントリを検索
                closest_time_idx = min(range(len(all_times)), key=lambda i: abs(all_times[i] - sim_time))
                current_adhesion_points = adhesion_points_history[closest_time_idx]

                # Adhesion springs for each particle
                print(f"Frame {frame_idx}, Time {sim_time:.2f}: Adhesion Springs Information")
                for particle_id, springs in current_adhesion_points.items():
                    cnt = len(springs)
                    print(f"Particle ID {particle_id}: Number of Adhesion Springs: {len(springs)}")
                    

                for chain_idx, (positions, ids) in enumerate(frame):
                    for i, pos in enumerate(positions):
                        # 元の粒子を描画
                        disk = patches.Circle(pos, R, color=colors[chain_idx])
                        ax.add_patch(disk)
                        elements.append(disk)

                        # 接着バネを持つ粒子に色を適用
                        num_adhesive_springs = len(current_adhesion_points.get(ids[i], []))
                        if num_adhesive_springs > 0 and i in [0, len(positions) - 1]:  # 先頭と最後尾のみ
                            adhesive_color = get_color_from_cmap(num_adhesive_springs, 1, max_adhesive_springs, custom_cmap)
                            adhesive_circle = patches.Circle(pos, R * 0.5, color=adhesive_color, alpha=0.8)
                            ax.add_patch(adhesive_circle)
                            elements.append(adhesive_circle)

                    # バネの描画
                    for i in range(len(positions) - 1):
                        distance = np.linalg.norm(positions[i + 1] - positions[i])
                        tolerance = tolerance_factor * l

                        # ======================= ここを変更 =======================
                        spring_color = get_spring_color(distance, l, tolerance)
                        # ======================= ここまで変更 =====================

                        spring, = ax.plot([positions[i][0], positions[i + 1][0]],
                                          [positions[i][1], positions[i + 1][1]],
                                          color=spring_color)
                        elements.append(spring)

                time_text = ax.text(0.05, 0.95, f'Time: {sim_time:.2f} s', transform=ax.transAxes, fontsize=12, verticalalignment='top')
                elements.append(time_text)

                return elements

            ani = FuncAnimation(fig, update, frames=desired_frame_count, init_func=init, blit=True, interval=20)
            plt.xlabel('x position')
            plt.ylabel('y position')
            plt.title('Disks Connected by Springs (with Repulsion) and Adding New Particles')
            plt.gca().set_aspect('equal', adjustable='box')

            # アニメーションを保存
            writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
            ani.save(f'particle_simulation_by_chain_ks_{k_s}_kt_{kt}_growth_rate_{g_r}_maxchain_{max_chains}_division_length_{split_threshold}_{sim_index}.mp4', writer=writer)

            # アニメーション表示
            plt.show()

            def save_chains_to_directory(chain_lengths_theoretical, chain_lengths_actual, time_steps, k_s, base_dir="chain_data"):
                """
                各チェインのデータを個別のCSVファイルとして保存し、
                k_sの値に基づいた新しいディレクトリにまとめます。

                Parameters
                ----------
                chain_lengths_theoretical : dict
                    {chain_id: [L_th_step0, L_th_step1, ...], ...}
                chain_lengths_actual : dict
                    {chain_id: [L_ac_step0, L_ac_step1, ...], ...}
                time_steps : list or array
                    [t0, t1, t2, ...] (各ステップ時刻)
                k_s : float or int
                    シミュレーションパラメータk_sの値
                base_dir : str, optional
                    基本となる出力ディレクトリ名（デフォルトは"chain_data"）
                """

                
                try:
                    # 新しいディレクトリ名を生成（例: chain_data_ks_1000_loop_0_growth_rate_0.029）
                    dir_name = f"{base_dir}_ks_{k_s}_kt_{kt}_loop_{sim_index}_growth_rate_{g_r}"
                    os.makedirs(dir_name, exist_ok=True)
                    print(f"Created directory: {dir_name}")

                    # 各チェインごとにCSVファイルを作成
                    for chain_id in chain_times.keys():
                        time_list = chain_times.get(chain_id, [])
                        L_th_list = chain_lengths_theoretical.get(chain_id, [])
                        L_ac_list = chain_lengths_actual.get(chain_id, [])
                        np_list = chain_particle_counts.get(chain_id, [])

                        # 各リストの長さを確認
                        if not (len(time_list) == len(L_th_list) == len(L_ac_list) == len(np_list)):
                            print(f"[ERROR] Chain {chain_id} のデータ長が一致していません。"
                                  f" time: {len(time_list)}, L_theoretical: {len(L_th_list)}, L_actual: {len(L_ac_list)}, N_particles: {len(np_list)}")
                            # 必要に応じて、足りないリストを埋める処理を追加するか、単にスキップする
                            continue

                        # そのまま保存（または desired_length を使って全リストを切り詰める）
                        df = pd.DataFrame({
                            'time': time_list,
                            'L_theoretical': L_th_list,
                            'L_actual': L_ac_list,
                            'N_particles': np_list,
                        })
                        file_name = f"chain_{chain_id}.csv"
                        file_path = os.path.join(dir_name, file_name)
                        df.to_csv(file_path, index=False, encoding='utf-8')
                        print(f"Saved chain {chain_id} to {file_path}")


                except Exception as e:
                    print(f"An error occurred: {e}")



            # 関数の呼び出し
            save_chains_to_directory(
                chain_lengths_theoretical=chain_lengths_theoretical,
                chain_lengths_actual=chain_lengths_actual,
                time_steps=time_steps,
                k_s=k_s,
                base_dir="chain_data"
            )
            df_repulsion = pd.DataFrame(repulsion_energy_time, columns=["time", "repulsion_energy", "contact_count"])

            # 変数 k_s を使ってファイル名に埋め込む (例)
            end_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"repulsion_energy_ks_{k_s}_kt_{kt}_kc_{k_c}_growth_rate_{g_r}_endtime_{end_time_str}_loop_{sim_index}.csv"

            df_repulsion.to_csv(output_filename, index=False, encoding="utf-8")
            print(f"Saved repulsion energy history to {output_filename}")

            #end_time_str = f"{current_time:.2f}"

            # または、実際の保存時刻を使う場合はこちら
            #end_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # ファイル名に k_s, max_chains, 終了時刻を含める
            filename = f"all_positions_ks_{k_s}_kt_{kt}_maxchain_{max_chains}_growth_rate_{g_r}_endtime_{end_time_str}_loop_{sim_index}.pkl"

            with open(filename, 'wb') as f:
                pickle.dump(all_positions, f)

            print(f"Data saved to {filename}")

            times, energies, bond_counts = zip(*torsion_energy_time)
            df_torsion = pd.DataFrame({
                "time": times,
                "torsion_energy_normalized": energies,
                "total_bond_count": bond_counts
            })

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"torsion_energy_ks_{k_s}_kt_{kt}_kc_{k_c}_maxchain_{max_chains}_growth_rate_{g_r}_endtime_{timestamp}_loop_{sim_index}.csv"
            df_torsion.to_csv(filename, index=False)
            print(f"Saved torsion energy history to {filename}")
            
            
            # DataFrame にまとめる
            df = pd.DataFrame(adhesion_time_series)

            # ファイル名に max_adhesive_springs, lambda_poisson, sim_index を自動で含める
            filename = (
                f"adhesion_per_particle"
                f"_max{max_adhesive_springs}"
                f"_poisson{lambda_poisson:.6f}"
                f"_sim{sim_index}.pkl"
            )

            # output_dir を使ってパスを組み立て
            pkl_path = script_dir / filename

            # pickle で保存
            df.to_pickle(pkl_path)
            print(f"Saved data to {pkl_path}")






            import numpy as np
            import pickle
            import os
            from datetime import datetime

            # リアルタイムのタイムスタンプを取得
            current_real_time = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 保存ディレクトリ
            script_dir  = Path(__file__).resolve().parent
            output_dir  = script_dir / "histogram_data"
            output_dir.mkdir(parents=True, exist_ok=True)

            # データを保存するリスト
            histogram_data = []

            # 保存したい時間間隔
            target_times = np.arange(3, max(time_steps) + 1, 3)  # 最大の時間まで3刻み

            for target_time in target_times:
                # ターゲットの時間に最も近いタイムステップを見つける
                closest_idx = min(range(len(time_steps)), key=lambda i: abs(time_steps[i] - target_time))
                closest_time = time_steps[closest_idx]

                # closest_time がすでにシミュレーション終了点を越えている場合、停止
                if closest_time > max(time_steps):
                    break

                chain_lengths_at_time = chain_lengths_over_time[closest_idx]

                # ヒストグラムデータを計算
                counts, bin_edges = np.histogram(chain_lengths_at_time, bins=20)

                # データを保存用リストに追加
                histogram_data.append({
                    'time': closest_time,
                    'counts': counts.tolist(),
                    'bin_edges': bin_edges.tolist()
                })

            # 保存ファイル名にパラメータとリアルタイムを含める
            filename = f"histogram_ks_{k_s}_ktpar_{kt_par}_realtime_{current_real_time}.pkl"
            file_path = os.path.join(output_dir, filename)

            # データをファイルに保存
            #with open(file_path, "wb") as f:
                #pickle.dump(histogram_data, f)

            #print(f"データを {file_path} に保存しました。")

        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd

        ###############################################################################
        # 0) L2ノルムの計算結果を保存するリスト (グローバル変数)
        ###############################################################################
        # グローバル変数が既に存在しない場合のみ初期化
        if 'results_original' not in globals():
            results_original = []  # [(k_s, l2_norm_original), ...]

        ###############################################################################
        # 1) 「元のコード」を実行する関数
        ###############################################################################
        def calculate_l2_norm():
            """
            変更したくない元の計算コードをここに貼り付ける。
            chain_lengths_theoretical, chain_lengths_actual, time_steps, k_s など
            グローバル変数を使って計算する想定。
            """
            chain_id = 0

            # データ取得
            y_theory = chain_lengths_theoretical[chain_id]  # 理論長さ
            y_actual = chain_lengths_actual[chain_id]       # 実際の長さ
            x = time_steps                                  # 時刻

            # 配列長を短い方に合わせる
            min_len = min(len(x), len(y_theory), len(y_actual))
            x = x[:min_len]
            y_theory = y_theory[:min_len]
            y_actual = y_actual[:min_len]

            # 1) Pythonリスト -> NumPy 配列
            x_arr = np.array(x)
            y_theory_arr = np.array(y_theory)
            y_actual_arr = np.array(y_actual)

            # 2) 差の二乗を計算
            diff_sq = (y_theory_arr - y_actual_arr) ** 2

            # 3) 数値積分 (台形則)
            l2_norm_sq = np.trapz(diff_sq, x_arr)
            l2_norm = np.sqrt(l2_norm_sq)

            print(f"L2-norm squared of difference: {l2_norm_sq:.4f}")
            print(f"L2-norm of difference:         {l2_norm:.4f}")

            return l2_norm

        ###############################################################################
        # 2) シミュレーション結果を処理してリストに追加する関数
        ###############################################################################
        def process_simulation_results():
            """
            シミュレーション終了後に、この関数を呼び出す。
              1) 元の actual で L2ノルム計算 → results_original に (k_s, l2_norm_original)
            """
            print("\n--- (A) Original actual ---")
            l2_norm_original = calculate_l2_norm()
            results_original.append((k_s, l2_norm_original))
            print(f"Appended to results_original: (k_s={k_s}, L2_norm_original={l2_norm_original:.4f})")

        ###############################################################################
        # 3) プロットおよびデータ保存する関数
        ###############################################################################
        def plot_and_save_results_sorted(save_plot=False, plot_filename='k_s_vs_L2norm.pdf', csv_filename='L2norm_results.csv'):
            """
            これまでに貯めた results_original を
            k_s の昇順にソートしてからプロットし、必要に応じて保存する。
            また、データもCSVファイルとして保存する。
            """
            # Pandas DataFrame に変換
            df_original = pd.DataFrame(results_original, columns=['k_s', 'L2_norm_original'])

            # 重複するk_sを削除 (最初の出現のみ保持)
            df_original = df_original.drop_duplicates(subset='k_s', keep='first')

            # k_sでソート
            df_combined = df_original.sort_values(by='k_s')

            # 重複チェックのために表示（オプション）
            print("\n=== Combined DataFrame ===")
            print(df_combined)

            if df_combined.empty:
                print("Warning: Combined DataFrame is empty. No data to plot.")
            else:
                # プロット
                plt.figure(figsize=(8, 5))
                plt.plot(df_combined['k_s'], df_combined['L2_norm_original'], 'o-', label="Original actual")

                plt.xlabel("k_s")
                plt.ylabel("L2-norm of difference")
                plt.title("Comparison: original, sorted by k_s")
                plt.legend()
                plt.grid(True)

                if save_plot:
                    plt.savefig(plot_filename, format='pdf')
                    print(f"Plot saved as {plot_filename}")

                plt.show()

                # データの保存: CSVに保存
                df_combined.to_csv(csv_filename, index=False)
                print(f"Results saved to {csv_filename}")

        ###############################################################################
        # 4) シミュレーション実行のサンプル (ダミー)
        ###############################################################################
        if __name__ == "__main__":




            process_simulation_results()

            # 全シミュレーションが終わったら、プロットおよびデータ保存
            plot_and_save_results_sorted(save_plot=True, plot_filename='k_s_vs_L2norm.pdf', csv_filename='L2norm_results.csv')

        #%matplotlib inline
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd  # 追加: pandasをインポート

        # 可視化したいチェイン ID
        chain_id = 0

        # データ取得
        y_theory = chain_lengths_theoretical[chain_id]  # 理論長さ
        y_actual = chain_lengths_actual[chain_id]       # 実際の長さ
        x = time_steps                                  # 時刻

        # 配列長を短い方に合わせる
        min_len = min(len(x), len(y_theory), len(y_actual))
        x = x[:min_len]
        y_theory = y_theory[:min_len]
        y_actual = y_actual[:min_len]

        # 可視化
        plt.figure(figsize=(10, 6))
        plt.plot(x, y_theory, label="Theoretical length", color="blue")
        plt.plot(x, y_actual, label="Actual length", color="red", linestyle="--")

        plt.xlabel("Time")
        plt.ylabel("Length")
        plt.title(f"Chain ID = {chain_id}: Theoretical vs Actual")
        plt.legend()
        plt.grid()
        plt.show()

        # データの保存
        # データフレームを作成
        df = pd.DataFrame({
            'x': x,
            'y_theory': y_theory,
            'y_actual': y_actual
        })

        # ファイル名にchain_idを含める（必要に応じてk_sも含める）
        # 例: chain_0_data.csv
        csv_filename = f'chain_{chain_id}_k_s_{k_s}_kt_{kt}_data_below.csv'

        # CSVに保存
        df.to_csv(csv_filename, index=False)
        print(f"Data saved to {csv_filename}")
        end_simulation = time.time()  # 終了時刻を記録
        simulation_time = end_simulation - start_simulation  # 経過時間を計算

    return f"Simulation {sim_index} completed with growth_rate {g_r} in {simulation_time:.2f} seconds"




import time
from pathos.multiprocessing import ProcessingPool as Pool

if __name__ == '__main__':
    overall_start_time = time.time()  # 全体の開始時刻を記録
    pool = Pool()  # 利用可能な全コアを使用
    simulation_indices = list(range(5))
    
    # 各シミュレーションは個別の実行時間を含む結果を返す想定
    results = pool.map(run_simulation, simulation_indices)
    
    overall_end_time = time.time()  # 全体の終了時刻を記録
    overall_elapsed_time = overall_end_time - overall_start_time
    
    print("All simulations completed.")
    print(f"Total elapsed time: {overall_elapsed_time:.2f} seconds")
    
    # 各シミュレーションの実行時間などの情報を表示
    for res in results:
        print(res)
