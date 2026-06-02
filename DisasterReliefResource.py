from collections import deque
import heapq

# =====================================
# CO1 : Knowledge Representation
# =====================================

graph = {
    "Warehouse": [("CenterA", 4), ("CenterB", 6)],
    "CenterA": [("CenterC", 5), ("CenterD", 3)],
    "CenterB": [("CenterD", 2)],
    "CenterC": [("ReliefZone", 4)],
    "CenterD": [("ReliefZone", 2)],
    "ReliefZone": []
}

heuristic = {
    "Warehouse": 8,
    "CenterA": 5,
    "CenterB": 4,
    "CenterC": 2,
    "CenterD": 1,
    "ReliefZone": 0
}

# =====================================
# CO2 : BFS
# =====================================

def bfs(start):

    visited = set()
    queue = deque([start])

    print("\nBFS Traversal:")

    while queue:

        node = queue.popleft()

        if node not in visited:

            print(node, end=" -> ")
            visited.add(node)

            for neighbor, _ in graph[node]:
                queue.append(neighbor)

    print("END")


# =====================================
# CO2 : DFS
# =====================================

def dfs(node, visited=None):

    if visited is None:
        visited = set()

    visited.add(node)

    print(node, end=" -> ")

    for neighbor, _ in graph[node]:

        if neighbor not in visited:
            dfs(neighbor, visited)


# =====================================
# CO2 : UCS
# =====================================

def ucs(start, goal):

    pq = [(0, start)]
    visited = set()

    while pq:

        cost, node = heapq.heappop(pq)

        if node == goal:
            return cost

        if node in visited:
            continue

        visited.add(node)

        for neighbor, distance in graph[node]:
            heapq.heappush(
                pq,
                (cost + distance, neighbor)
            )

    return -1


# =====================================
# CO2 : A*
# =====================================

def astar(start, goal):

    pq = [(0, start)]
    g_cost = {start: 0}

    while pq:

        _, node = heapq.heappop(pq)

        if node == goal:
            return g_cost[node]

        for neighbor, distance in graph[node]:

            new_cost = g_cost[node] + distance

            if neighbor not in g_cost or new_cost < g_cost[neighbor]:

                g_cost[neighbor] = new_cost

                f_cost = new_cost + heuristic[neighbor]

                heapq.heappush(
                    pq,
                    (f_cost, neighbor)
                )

    return -1


# =====================================
# CO3 : CSP Allocation
# =====================================

class Truck:

    def __init__(self, name, capacity):
        self.name = name
        self.capacity = capacity


class ReliefCenter:

    def __init__(self, name, demand):
        self.name = name
        self.demand = demand


centers = [
    ReliefCenter("CenterA", 30),
    ReliefCenter("CenterB", 20),
    ReliefCenter("CenterC", 15)
]

trucks = [
    Truck("Truck1", 50),
    Truck("Truck2", 40)
]


def backtracking(index):

    if index == len(centers):
        return True

    center = centers[index]

    for truck in trucks:

        if truck.capacity >= center.demand:

            truck.capacity -= center.demand

            if backtracking(index + 1):
                return True

            truck.capacity += center.demand

    return False


# =====================================
# CO4 : Utility Function
# =====================================

def utility(delivered, time_taken):

    return delivered - (0.5 * time_taken)


# =====================================
# CO4 : Minimax
# =====================================

def minimax(depth, maximizing):

    if depth == 0:
        return 0

    if maximizing:

        return max(
            5 + minimax(depth - 1, False),
            2 + minimax(depth - 1, False)
        )

    else:

        return min(
            -3 + minimax(depth - 1, True),
            -1 + minimax(depth - 1, True)
        )


# =====================================
# CO5 : Bayes Theorem
# =====================================

def bayes(prior, likelihood, evidence):

    return (prior * likelihood) / evidence


# =====================================
# CO5 : Expected Utility
# =====================================

def expected_utility(probability, utility_value):

    return probability * utility_value


# =====================================
# CO6 : Hybrid AI Decision System
# =====================================

def hybrid_decision():

    route_cost = ucs(
        "Warehouse",
        "ReliefZone"
    )

    blockage_probability = bayes(
        0.4,
        0.8,
        0.6
    )

    print("\n----- AI DECISION REPORT -----")

    print(
        f"Shortest Route Cost : {route_cost}"
    )

    print(
        f"Road Block Probability : {round(blockage_probability,2)}"
    )

    if blockage_probability > 0.5:

        print(
            "Decision : Alternate Route Recommended"
        )

    else:

        print(
            "Decision : Route Safe"
        )


# =====================================
# Display Map
# =====================================

def display_graph():

    print("\nRoad Network")

    for node in graph:

        print(node, "->", graph[node])


# =====================================
# Main Menu
# =====================================

def main():

    while True:

        print("\n")
        print("=" * 50)
        print("RESOURCE PLANNER RELIEF SYSTEM")
        print("=" * 50)

        print("1. Display Map")
        print("2. BFS Search")
        print("3. DFS Search")
        print("4. Uniform Cost Search")
        print("5. A* Search")
        print("6. CSP Resource Allocation")
        print("7. Utility Evaluation")
        print("8. Minimax Decision")
        print("9. Bayesian Analysis")
        print("10. Hybrid AI Decision")
        print("11. Exit")

        choice = int(input("\nEnter Choice: "))

        if choice == 1:

            display_graph()

        elif choice == 2:

            bfs("Warehouse")

        elif choice == 3:

            print("\nDFS Traversal:")
            dfs("Warehouse")
            print("END")

        elif choice == 4:

            cost = ucs(
                "Warehouse",
                "ReliefZone"
            )

            print(
                "\nUCS Shortest Cost =",
                cost
            )

        elif choice == 5:

            cost = astar(
                "Warehouse",
                "ReliefZone"
            )

            print(
                "\nA* Shortest Cost =",
                cost
            )

        elif choice == 6:

            if backtracking(0):

                print(
                    "\nResources Successfully Allocated"
                )

            else:

                print(
                    "\nAllocation Failed"
                )

        elif choice == 7:

            value = utility(
                500,
                20
            )

            print(
                "\nUtility Value =",
                value
            )

        elif choice == 8:

            value = minimax(
                3,
                True
            )

            print(
                "\nMinimax Value =",
                value
            )

        elif choice == 9:

            result = bayes(
                0.4,
                0.8,
                0.6
            )

            print(
                "\nBayesian Probability =",
                round(result, 2)
            )

        elif choice == 10:

            hybrid_decision()

        elif choice == 11:

            print(
                "\nThank You!"
            )
            break

        else:

            print(
                "\nInvalid Choice"
            )


if __name__ == "__main__":
    main()
