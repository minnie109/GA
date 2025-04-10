import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import copy

st.set_page_config(page_title="🚚 配送路徑最佳化 GA", layout="centered")
st.title("🚚 配送路徑最佳化系統")

uploaded_file = st.file_uploader("📁 上傳包含 X, Y, Demand, Name 欄位的 Excel 檔案", type=["xlsx"])

vehicle_capacity = st.number_input("🚗 車輛容量", min_value=1, value=10)
num_generations = st.number_input("🔁 演化代數", min_value=1, value=100)
population_size = st.number_input("👨‍👩‍👧‍👦 種群大小", min_value=2, value=30)
mutation_rate = 0.1

if uploaded_file and st.button("開始運算 🚀"):

    df = pd.read_excel(uploaded_file)
    coords = list(zip(df["X"], df["Y"]))
    demands_raw = list(df["Demand"])
    dot_raw = list(df["Name"])

    depot = coords[0]
    destinations = coords[1:]
    demands = demands_raw[1:]
    dot = dot_raw[1:]

    def distance(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def total_distance(route):
        dist = 0
        current_load = vehicle_capacity
        current_position = depot
        for stop in route:
            demand = demands[stop]
            if current_load < demand:
                dist += distance(current_position, depot)
                current_load = vehicle_capacity
                current_position = depot
            dist += distance(current_position, destinations[stop])
            current_load -= demand
            current_position = destinations[stop]
        dist += distance(current_position, depot)
        return dist

    def generate_initial_population():
        return [random.sample(range(len(destinations)), len(destinations)) for _ in range(population_size)]

    def fitness(route):
        return 1 / total_distance(route)

    def selection(population):
        weights = [fitness(ind) for ind in population]
        return random.choices(population, weights=weights, k=2)

    def crossover(parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [-1] * size
        child[start:end] = parent1[start:end]
        fill_pos = [i for i in parent2 if i not in child]
        index = 0
        for i in range(size):
            if child[i] == -1:
                child[i] = fill_pos[index]
                index += 1
        return child

    def mutate(individual):
        if random.random() < mutation_rate:
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]

    def genetic_algorithm():
        population = generate_initial_population()
        best_distance = float('inf')
        best_route = None
        history = []
        stagnant_generations = 0
        max_stagnant_generations = 20

        for generation in range(num_generations):
            new_population = []
            for _ in range(population_size // 2):
                parent1, parent2 = selection(population)
                child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)
                mutate(child1)
                mutate(child2)
                new_population.extend([child1, child2])
            population = new_population
            current_best = min(population, key=total_distance)
            current_best_distance = total_distance(current_best)
            if current_best_distance < best_distance:
                best_distance = current_best_distance
                best_route = current_best
                stagnant_generations = 0
            else:
                stagnant_generations += 1
            history.append(best_distance)
            if stagnant_generations >= max_stagnant_generations:
                break
        return best_route, best_distance, history

    def plot_route(best_route, draw=False):
        end_route = ['Depot']
        current_position = depot
        current_load = vehicle_capacity
        local_demands = copy.deepcopy(demands)
        path_coords = [depot]
        colors = []
        i = 0
        while i < len(best_route):
            stop = best_route[i]
            stop_name = dot[stop]
            if local_demands[stop] == 0:
                i += 1
                continue
            if current_load == 0:
                path_coords.append(depot)
                end_route.append("Depot")
                colors.append("red")
                current_load = vehicle_capacity
                current_position = depot
                continue
            deliver = min(current_load, local_demands[stop])
            local_demands[stop] -= deliver
            current_load -= deliver
            path_coords.append(destinations[stop])
            end_route.append(f"{stop_name} (送{deliver})")
            colors.append("blue")

        if end_route[-1] != "Depot":
            path_coords.append(depot)
            end_route.append("Depot")
            colors.append("red")

        return end_route

    with st.spinner("🧠 運算中，請等待..."):
        best_route, best_distance, history = genetic_algorithm()

    st.success(f"✅ 最佳距離：{best_distance:.2f}")

    st.subheader("📈 收斂圖")
    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Distance")
    ax.set_title("GA Convergence")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("📋 拜訪紀錄分析")
    end_route = plot_route(best_route)
    pure_route = [stop.split(" (送")[0] if " (送" in stop else stop for stop in end_route]

    st.markdown("📋 **完整拜訪順序**：")
    st.write(" ➜ ".join(pure_route) + " ➜ End")

    st.markdown("📋 **完整配送紀錄（每趟配送）**：")
    trip = []
    trip_counter = 1

    for stop in end_route:
        if stop == "Depot":
            if trip:  # 如果不是第一次
                st.markdown(f"**🚚 配送趟次 {trip_counter}**")
                st.code(" ➔ ".join(trip))
                trip_counter += 1
            trip = ["Depot"]
        else:
            trip.append(stop)

    if trip and trip != ["Depot"]:
        st.markdown(f"**🚚 配送趟次 {trip_counter}**")
        st.code(" ➔ ".join(trip))
