# import python 各種套件
# 這裡的套件是用來處理數據、隨機數、畫圖等功能
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import copy
# 讀取 Excel 檔案
df = pd.read_excel("data.xlsx")
plt.ion()
# 提取數據
coords = list(zip(df["X"], df["Y"]))  # 讀取所有座標
demands = list(df["Demand"])  # 讀取所有需求量
# 從 Excel 取得目的地座標名稱
dot = list(df["Name"])[1:]  # 讀取所有目的地名稱
# 從 Excel 取得配送中心的座標
depot = coords[0]  # 第一列為配送中心
destinations = coords[1:]  # 目的地
demands = demands[1:]  # 需求量（不包含配送中心）
# 車輛最大容量
vehicle_capacity = 10 
# 迭代次數 
num_generations = 100
# 初始種群大小
population_size = 30
# 突變率
mutation_rate = 0.1

# 計算歐基里得距離
def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# 計算總距離
def total_distance(route):
    # 紀錄總距離的變數
    dist = 0
    # 當前載貨量
    current_load = vehicle_capacity
    # 一開始車輛在配送中心
    current_position = depot
    # route : 一開始隨機生成的路徑
    for stop in route:
        # 取得對應目的地的需求量
        demand = demands[stop]
        # 如果當前負載小於需求量，則需要回到配送中心
        if current_load < demand:
            # 總距離加上上一點回到配送中心的距離
            dist += distance(current_position, depot)
            # 重設當前負載為車輛容量
            current_load = vehicle_capacity
            # 回到配送中心 座標變成 depot (配送中心)
            current_position = depot
        # 總距離加上當前點到目的地的距離
        dist += distance(current_position, destinations[stop])
        # 更新當前負載，減去需求量
        current_load -= demand
        # 更新當前位置為目的地
        current_position = destinations[stop]
    # 總距離加上當前點到目的地的距離    
    dist += distance(current_position, depot)
    return dist

# 生成初始種群
def generate_initial_population():
    #population是一個裝各種路徑list的列表
    population = [random.sample(range(len(destinations)), len(destinations)) for _ in range(population_size)]
    return population

#定義適應度函數
#使用倒數是為了將 "較短距離" 轉換為 "較高適應度"。
def fitness(route):
    return 1 / total_distance(route)

# 選擇
# 用來從目前的種群中選擇出兩個解，這兩個個體會成為父母，進行交叉操作生成下一代。
def selection(population):
    # 適應度高的個體會在選擇中有更大的機會被選中
    weights = [fitness(ind) for ind in population]
    #這裡的 k=2 表示選擇兩個個體作為父母
    #這樣的選擇方式可以確保適應度較高的個體有更大的機會被選中，從而在下一代中保留優秀的基因
    return random.choices(population, weights=weights, k=2)

# 交叉 
# 目的是從兩個「父代」生成一個新的「子代」
def crossover(parent1, parent2):
    size = len(parent1)
    #使用 random.sample() 函數隨機選擇兩個索引，並使用 sorted() 函數確保 start 小於 end
    start, end = sorted(random.sample(range(size), 2))
    #創建一個空的子代列表 child，初始化為 -1，表示還未填充。
    child = [-1] * size
    #將父代 parent1 的部分基因（即從 start 到 end 的部分）複製到子代 child 中對應的位置。
    #start 和 end 是隨機選擇的索引
    child[start:end] = parent1[start:end]
    #用另一個父代填充子代的剩餘部分
    fill_pos = [i for i in parent2 if i not in child]
    index = 0
    for i in range(size):
        #如果 child[i] 是 -1，則表示這個位置還沒有被填充
        if child[i] == -1:
            # 逐一填充剩餘的部分
            child[i] = fill_pos[index]
            index += 1
    return child

# 變異
# 通過隨機交換路徑中的兩個城市
def mutate(individual):
    #random.random() 生成一個 0 到 1 之間的隨機數，這裡用來決定是否進行變異
    #如果這個隨機數小於上面所設的變異率，則進行變異
    if random.random() < mutation_rate:
        #隨機選擇兩個索引 i 和 j，並交換這兩個位置的基因（城市）
        i, j = random.sample(range(len(individual)), 2)
        #交換個體中兩個基因的位置
        individual[i], individual[j] = individual[j], individual[i]

# 遺傳算法找出最短的路徑
def genetic_algorithm():
    # 初始化種群
    population = generate_initial_population()
    # 記錄最佳距離和路徑
    # 初始化最短距離變數
    best_distance = float('inf')
    # best_route：記錄目前找到的最佳路徑的list
    best_route = None
    # 用來記錄每一代的最佳距離
    history = []
    # 用來記錄目前的世代數
    stagnant_generations = 0
    # 用來記錄沒有改進的世代數
    max_stagnant_generations = 50
    # 演算法會執行 num_generations 代，每一代都會產生一批新的解（也就是新的路線）。
    for generation in range(num_generations):
        # 每一代都會產生新的種群，這裡的 population_size 是指每一代的個體數量
        new_population = []
        # 重複這段交配 + 突變的邏輯，直到我生出跟原本一樣多的新族群
        for _ in range(population_size // 2):
            # 選擇父母
            parent1, parent2 = selection(population)
            # 交配 ➜ 產生兩個小孩
            # 這裡的 crossover 是用來生成新的個體（子代）
            child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)
            # 每個孩子都會經過 mutate 函數處理一次。
            # 會根據 mutation_rate決定要不要真的突變。
            mutate(child1)
            # 這裡的 mutate 是用來增加解的多樣性，防止算法陷入局部最優解
            mutate(child2)
            # 將新產生的兩個小孩加入到新的種群中
            new_population.extend([child1, child2])   
        # 菁英保留
        if best_route is not None:
            new_population[-1] = best_route
        # 更新整個種群    
        population = new_population
        # 計算當前世代的最佳路徑    
        current_best = min(population, key=total_distance)
        # 把上面這個current_best 路徑列表丟進去，計算出這個路徑的距離
        # 把數字距離存在 current_best_distance 裡面
        current_best_distance = total_distance(current_best)
        # best_distance 是一開始設的正無窮大，這裡會跟 current_best_distance 比較
        # 如果 current_best_distance 比 best_distance 還小，表示這個路徑更短
        if current_best_distance < best_distance:
            # 存目前最佳路徑的距離
            best_distance = current_best_distance
            # 存目前最佳路徑的路徑列表
            best_route = current_best
            # 是用來記錄 連續幾代演化都沒有找到更好的解（沒有進步）。
            # 如果有進步，就把 stagnant_generations 重設為 0
            # 這樣就可以知道目前的演化進度
            stagnant_generations = 0
        else:
            # 沒進步，加 1
            stagnant_generations += 1
        # list history 是用來記錄每一代的最佳距離，方便後續畫圖分析演化過程
        history.append(best_distance)
        # 這裡的 max_stagnant_generations 是一個參數，用來控制演化的停止條件
        if stagnant_generations >= max_stagnant_generations:
            print("Early stopping due to no improvement.")
            # 可以提前停止，避免浪費更多的時間
            break
    # 畫收斂圖
    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Best Distance")
    plt.title("GA Convergence")
    plt.show()
    return best_route, best_distance
# 計算最佳解經過的所有位置
def plot_route(best_route, depot, destinations, demands, vehicle_capacity, dot):
    end_route = ['Depot']
    current_position = depot
    current_load = vehicle_capacity
    local_demands = copy.deepcopy(demands)
    i = 0
    while i < len(best_route):
        stop = best_route[i]
        stop_name = dot[stop]
        # 該站已送完 → 下一站
        if local_demands[stop] == 0:
            i += 1
            continue
        # 沒貨了 → 回倉補貨
        if current_load == 0:
            end_route.append("Depot")
            current_load = vehicle_capacity
            current_position = depot
            continue
        # 實際配送量
        deliver = min(current_load, local_demands[stop])
        local_demands[stop] -= deliver
        current_load -= deliver
        # 紀錄這次拜訪該站
        end_route.append(stop_name)
        # 如果這次還沒送完，下一輪會繼續處理這個 stop
        # 如果送完了，就讓它進下一站（在上方 if 判斷）
        # 如果還有需求且車已空，下一輪會自動回補再來
    # 結尾補倉
    if end_route[-1] != "Depot":
        end_route.append("Depot")
    return end_route

def main():
    print("waiting...")
    # 呼叫基因演算法
    best_route, best_distance = genetic_algorithm()
    print("Best Distance:", best_distance)
    end_route=plot_route(best_route, depot, destinations, demands, vehicle_capacity,dot)
    for i in range(len(end_route)):
        print(end_route[i], end=" ➜ ")
    print("end")
    print()
    # 畫出最佳路徑
    # 程式等待直到你按下 Enter 鍵
    #才不會產生圖後就被關掉了
    input("Press Enter to exit...")
main()
