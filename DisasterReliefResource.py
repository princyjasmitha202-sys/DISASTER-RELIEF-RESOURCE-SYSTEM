import heapq
import math
import random
import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict, deque

# ─────────────────────────────────────────────
# CO1: Logging & Step-by-Step Trace
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ResourcePlanner")

def trace(step: str, detail: str = ""):
    logger.info(f"TRACE | {step}" + (f" | {detail}" if detail else ""))

# ─────────────────────────────────────────────
# CO1: PEAS Definition
# ─────────────────────────────────────────────
PEAS = {
    "Performance": "Minimize total delivery cost + unmet demand; maximize coverage within time window",
    "Environment": "Road network graph with disaster zones, depots, and relief camps",
    "Actuators": "Route assignment, supply truck dispatch, priority override",
    "Sensors": "Road status, camp demand levels, weather severity, available stock",
}

# ─────────────────────────────────────────────
# CO1: State, Action, Transition Dataclasses
# ─────────────────────────────────────────────
@dataclass
class Location:
    id: str
    name: str
    lat: float
    lon: float
    demand: dict = field(default_factory=dict)   # {"food": 10, "medicine": 5}
    supply: dict = field(default_factory=dict)   # depot stock
    is_depot: bool = False
    is_camp: bool = False
    severity: int = 0  # 0-5 disaster severity

@dataclass
class State:
    current_node: str
    supplies_carried: dict
    visited_camps: frozenset
    total_cost: float
    path: list

    def __lt__(self, other):
        return self.total_cost < other.total_cost

    def __hash__(self):
        return hash((self.current_node, frozenset(self.supplies_carried.items()), self.visited_camps))

@dataclass
class Action:
    type: str          # "move", "deliver", "load"
    target: str
    payload: dict = field(default_factory=dict)
    cost: float = 0.0

# ─────────────────────────────────────────────
# CO1: Road Network Graph
# ─────────────────────────────────────────────
class RoadNetwork:
    def __init__(self):
        self.locations: dict[str, Location] = {}
        self.edges: dict[str, list] = defaultdict(list)  # node -> [(neighbor, weight, road_status)]

    def add_location(self, loc: Location):
        self.locations[loc.id] = loc
        trace("CO1:Graph", f"Added node {loc.id} ({loc.name})")

    def add_road(self, a: str, b: str, distance: float, passable: bool = True):
        status = "open" if passable else "blocked"
        self.edges[a].append((b, distance, status))
        self.edges[b].append((a, distance, status))
        trace("CO1:Graph", f"Road {a}<->{b}, dist={distance}, {status}")

    def neighbors(self, node: str, include_blocked: bool = False):
        return [
            (nb, w) for nb, w, st in self.edges[node]
            if include_blocked or st == "open"
        ]

    def heuristic(self, a: str, b: str) -> float:
        """Haversine heuristic for A*"""
        la, lb = self.locations[a], self.locations[b]
        R = 6371
        dlat = math.radians(lb.lat - la.lat)
        dlon = math.radians(lb.lon - la.lon)
        h = math.sin(dlat/2)**2 + math.cos(math.radians(la.lat)) * math.cos(math.radians(lb.lat)) * math.sin(dlon/2)**2
        return 2 * R * math.asin(math.sqrt(h))


# ─────────────────────────────────────────────
# CO2: Search Algorithms
# ─────────────────────────────────────────────
class SearchEngine:
    def __init__(self, graph: RoadNetwork):
        self.graph = graph
        self.stats = {"expansions": 0, "peak_memory": 0, "runtime": 0}

    def _reset_stats(self):
        self.stats = {"expansions": 0, "peak_memory": 0, "runtime": 0}

    def bfs(self, start: str, goal: str) -> tuple[list, float]:
        """CO2: BFS - unweighted shortest path"""
        self._reset_stats()
        t0 = time.time()
        queue = deque([(start, [start], 0)])
        visited = {start}
        while queue:
            self.stats["peak_memory"] = max(self.stats["peak_memory"], len(queue))
            node, path, cost = queue.popleft()
            self.stats["expansions"] += 1
            if node == goal:
                self.stats["runtime"] = time.time() - t0
                trace("CO2:BFS", f"Found path {path}, cost={cost:.2f}, expansions={self.stats['expansions']}")
                return path, cost
            for nb, w in self.graph.neighbors(node):
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb], cost + w))
        return [], float('inf')

    def dfs(self, start: str, goal: str, depth_limit: int = 20) -> tuple[list, float]:
        """CO2: DFS with depth limit"""
        self._reset_stats()
        t0 = time.time()
        stack = [(start, [start], 0, 0)]
        visited = set()
        while stack:
            self.stats["peak_memory"] = max(self.stats["peak_memory"], len(stack))
            node, path, cost, depth = stack.pop()
            if node in visited or depth > depth_limit:
                continue
            visited.add(node)
            self.stats["expansions"] += 1
            if node == goal:
                self.stats["runtime"] = time.time() - t0
                trace("CO2:DFS", f"Found path {path}, cost={cost:.2f}")
                return path, cost
            for nb, w in self.graph.neighbors(node):
                if nb not in visited:
                    stack.append((nb, path + [nb], cost + w, depth + 1))
        return [], float('inf')

    def ucs(self, start: str, goal: str) -> tuple[list, float]:
        """CO2: UCS - optimal cost"""
        self._reset_stats()
        t0 = time.time()
        pq = [(0, start, [start])]
        visited = {}
        while pq:
            self.stats["peak_memory"] = max(self.stats["peak_memory"], len(pq))
            cost, node, path = heapq.heappop(pq)
            if node in visited:
                continue
            visited[node] = cost
            self.stats["expansions"] += 1
            if node == goal:
                self.stats["runtime"] = time.time() - t0
                trace("CO2:UCS", f"Optimal path {path}, cost={cost:.2f}")
                return path, cost
            for nb, w in self.graph.neighbors(node):
                if nb not in visited:
                    heapq.heappush(pq, (cost + w, nb, path + [nb]))
        return [], float('inf')

    def astar(self, start: str, goal: str) -> tuple[list, float]:
        """CO2: A* with admissible haversine heuristic"""
        self._reset_stats()
        t0 = time.time()
        pq = [(self.graph.heuristic(start, goal), 0, start, [start])]
        g_cost = {start: 0}
        while pq:
            self.stats["peak_memory"] = max(self.stats["peak_memory"], len(pq))
            f, g, node, path = heapq.heappop(pq)
            if node == goal:
                self.stats["runtime"] = time.time() - t0
                trace("CO2:Astar", f"Path {path}, g={g:.2f}, expansions={self.stats['expansions']}")
                return path, g
            self.stats["expansions"] += 1
            for nb, w in self.graph.neighbors(node):
                ng = g + w
                if nb not in g_cost or ng < g_cost[nb]:
                    g_cost[nb] = ng
                    h = self.graph.heuristic(nb, goal)
                    heapq.heappush(pq, (ng + h, ng, nb, path + [nb]))
        return [], float('inf')

    def greedy(self, start: str, goal: str) -> tuple[list, float]:
        """CO2: Greedy Best-First"""
        self._reset_stats()
        t0 = time.time()
        pq = [(self.graph.heuristic(start, goal), start, [start], 0)]
        visited = set()
        while pq:
            h, node, path, cost = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            self.stats["expansions"] += 1
            if node == goal:
                self.stats["runtime"] = time.time() - t0
                return path, cost
            for nb, w in self.graph.neighbors(node):
                if nb not in visited:
                    heapq.heappush(pq, (self.graph.heuristic(nb, goal), nb, path + [nb], cost + w))
        return [], float('inf')

    def compare_algorithms(self, start: str, goal: str) -> dict:
        """CO2: Empirical profiling of all algorithms"""
        results = {}
        for name, fn in [("BFS", self.bfs), ("DFS", self.dfs), ("UCS", self.ucs),
                          ("A*", self.astar), ("Greedy", self.greedy)]:
            path, cost = fn(start, goal)
            results[name] = {
                "path": path, "cost": round(cost, 2),
                "expansions": self.stats["expansions"],
                "peak_memory": self.stats["peak_memory"],
                "runtime_ms": round(self.stats["runtime"] * 1000, 3)
            }
        return results


# ─────────────────────────────────────────────
# CO3: CSP - Supply Allocation
# ─────────────────────────────────────────────
class SupplyAllocationCSP:
    """
    Variables: allocation[camp][resource] = amount
    Domains: 0 .. available_stock[resource]
    Constraints:
      - Total allocated <= stock (capacity constraint)
      - Each camp gets >= min_demand (demand constraint)
      - Priority camps get >= priority_min (priority constraint)
      - No allocation to unreachable camps (connectivity constraint)
    """

    def __init__(self, camps: list, resources: list, stock: dict,
                 demand: dict, reachable: set, priority_camps: set):
        self.camps = camps
        self.resources = resources
        self.stock = stock       # {resource: total_available}
        self.demand = demand     # {camp: {resource: min_demand}}
        self.reachable = reachable
        self.priority_camps = priority_camps
        self.assignment = {}     # {(camp, resource): amount}
        self.failure_log = []

    def _remaining_stock(self, resource: str) -> float:
        used = sum(self.assignment.get((c, resource), 0) for c in self.camps)
        return self.stock.get(resource, 0) - used

    def _is_consistent(self, camp: str, resource: str, value: float) -> tuple[bool, str]:
        if camp not in self.reachable:
            return False, f"Camp {camp} is unreachable"
        if value < self.demand.get(camp, {}).get(resource, 0):
            return False, f"Value {value} < min demand {self.demand[camp][resource]} for {camp}/{resource}"
        if value > self._remaining_stock(resource) + self.assignment.get((camp, resource), 0):
            return False, f"Exceeds stock for {resource}"
        return True, "OK"

    def _select_unassigned(self) -> Optional[tuple]:
        """CO3: MRV heuristic - pick variable with fewest valid values"""
        unassigned = [
            (c, r) for c in self.camps for r in self.resources
            if (c, r) not in self.assignment
        ]
        if not unassigned:
            return None
        # MRV: minimize remaining domain size
        def domain_size(var):
            c, r = var
            stock = self._remaining_stock(r)
            min_d = self.demand.get(c, {}).get(r, 0)
            return int(stock - min_d)
        return min(unassigned, key=domain_size)

    def _order_values(self, camp: str, resource: str) -> list:
        """CO3: LCV heuristic - least constraining value first (give more first to priority)"""
        min_d = self.demand.get(camp, {}).get(resource, 0)
        stock = self._remaining_stock(resource)
        n_other = len([c for c in self.camps if c != camp])
        # LCV: leave max for others
        max_val = stock - (min_d * n_other) if n_other > 0 else stock
        max_val = max(min_d, min(max_val, stock))
        step = max(1, int((max_val - min_d) / 3)) if max_val > min_d else 1
        vals = list(range(int(min_d), int(max_val) + 1, step))
        if camp in self.priority_camps:
            vals = list(reversed(vals))  # give more to priority camps first
        return vals or [int(min_d)]

    def _forward_check(self, camp: str, resource: str, value: float) -> bool:
        """CO3: Forward checking - ensure remaining camps can still be satisfied"""
        remaining_stock = self._remaining_stock(resource)
        unassigned_camps = [c for c in self.camps if (c, resource) not in self.assignment and c != camp]
        min_needed = sum(self.demand.get(c, {}).get(resource, 0) for c in unassigned_camps)
        return remaining_stock >= min_needed

    def backtrack(self) -> Optional[dict]:
        """CO3: Backtracking with MRV + LCV + Forward Checking"""
        var = self._select_unassigned()
        if var is None:
            trace("CO3:CSP", "Solution found!")
            return dict(self.assignment)

        camp, resource = var
        for value in self._order_values(camp, resource):
            ok, reason = self._is_consistent(camp, resource, value)
            if ok and self._forward_check(camp, resource, value):
                self.assignment[(camp, resource)] = value
                result = self.backtrack()
                if result is not None:
                    return result
                del self.assignment[(camp, resource)]
                self.failure_log.append(f"Backtracked {camp}/{resource}={value}: {reason}")
            else:
                self.failure_log.append(f"Pruned {camp}/{resource}={value}: {reason}")

        return None

    def min_conflicts_local(self, max_iter: int = 1000) -> dict:
        """CO3: Local search with min-conflicts for CSP"""
        # Random init
        assignment = {}
        for c in self.camps:
            for r in self.resources:
                assignment[(c, r)] = self.demand.get(c, {}).get(r, 0)

        for _ in range(max_iter):
            # Find conflicting variables
            conflicts = []
            for r in self.resources:
                used = sum(assignment.get((c, r), 0) for c in self.camps)
                if used > self.stock.get(r, 0):
                    conflicts.append(r)
            if not conflicts:
                trace("CO3:MinConflicts", "Local search found valid allocation")
                return assignment
            # Fix worst conflict
            r = conflicts[0]
            over = sum(assignment.get((c, r), 0) for c in self.camps) - self.stock.get(r, 0)
            # Reduce lowest-priority camp
            non_priority = [c for c in self.camps if c not in self.priority_camps]
            if non_priority:
                c = random.choice(non_priority)
                assignment[(c, r)] = max(0, assignment[(c, r)] - over)

        return assignment


# ─────────────────────────────────────────────
# CO4: Game Theory / Minimax for Multi-Agent
# ─────────────────────────────────────────────
class MultiAgentPlanner:
    """
    Two agents: Relief Coordinator (MAX) vs Adversarial Nature/Blockage (MIN)
    - MAX tries to maximize supplies delivered
    - MIN tries to block roads / reduce supply
    CO4: Minimax with Alpha-Beta Pruning
    """

    def __init__(self, graph: RoadNetwork):
        self.graph = graph
        self.nodes_evaluated = 0

    def utility(self, delivered: float, total_demand: float, cost: float) -> float:
        """CO4: Evaluation function"""
        coverage = delivered / max(total_demand, 1)
        efficiency = 1.0 / max(cost, 1)
        return round(100 * coverage + 10 * efficiency, 2)

    def minimax(self, node_id: str, depth: int, is_max: bool,
                alpha: float, beta: float,
                current_delivery: float, total_demand: float,
                current_cost: float) -> tuple[float, str]:
        """CO4: Minimax with alpha-beta pruning"""
        self.nodes_evaluated += 1

        if depth == 0:
            return self.utility(current_delivery, total_demand, current_cost), node_id

        neighbors = self.graph.neighbors(node_id)
        if not neighbors:
            return self.utility(current_delivery, total_demand, current_cost), node_id

        best_action = neighbors[0][0]

        if is_max:
            best_val = float('-inf')
            for nb, w in neighbors:
                loc = self.graph.locations.get(nb)
                delivery_gain = sum(loc.demand.values()) * 0.8 if loc and loc.is_camp else 0
                val, _ = self.minimax(nb, depth-1, False, alpha, beta,
                                      current_delivery + delivery_gain,
                                      total_demand, current_cost + w)
                if val > best_val:
                    best_val = val
                    best_action = nb
                alpha = max(alpha, val)
                if beta <= alpha:
                    trace("CO4:Minimax", f"Alpha-Beta prune at {nb}, alpha={alpha:.2f}, beta={beta:.2f}")
                    break
            return best_val, best_action
        else:
            best_val = float('inf')
            for nb, w in neighbors:
                # Nature tries to add cost (simulate blockage)
                val, _ = self.minimax(nb, depth-1, True, alpha, beta,
                                      current_delivery, total_demand, current_cost + w * 1.5)
                if val < best_val:
                    best_val = val
                    best_action = nb
                beta = min(beta, val)
                if beta <= alpha:
                    trace("CO4:Minimax", f"Beta prune at {nb}")
                    break
            return best_val, best_action

    def expectimax(self, node_id: str, depth: int, is_max: bool,
                   delivery: float, demand: float, cost: float) -> float:
        """CO4: Expectimax for stochastic nature"""
        self.nodes_evaluated += 1
        if depth == 0:
            return self.utility(delivery, demand, cost)
        neighbors = self.graph.neighbors(node_id)
        if not neighbors:
            return self.utility(delivery, demand, cost)

        if is_max:
            return max(
                self.expectimax(nb, depth-1, False, delivery, demand, cost + w)
                for nb, w in neighbors
            )
        else:
            # Stochastic: average over nature's moves
            probs = [1/len(neighbors)] * len(neighbors)
            return sum(
                p * self.expectimax(nb, depth-1, True, delivery, demand, cost + w)
                for (nb, w), p in zip(neighbors, probs)
            )

    def policy_selection(self, start: str, camps: list[str], total_demand: float) -> dict:
        """CO4: Policy selection using bounded rationality"""
        policies = {}
        for camp in camps:
            self.nodes_evaluated = 0
            val, action = self.minimax(start, depth=3, is_max=True,
                                       alpha=float('-inf'), beta=float('inf'),
                                       current_delivery=0, total_demand=total_demand,
                                       current_cost=0)
            policies[camp] = {
                "recommended_next": action,
                "utility": val,
                "nodes_evaluated": self.nodes_evaluated
            }
        return policies


# ─────────────────────────────────────────────
# CO5: Bayesian Network for Demand Estimation
# ─────────────────────────────────────────────
class BayesianDemandEstimator:
    """
    CO5: Bayesian Network
    Nodes: Weather, Severity, Population -> Demand (food, medicine, water)
    Uses CPTs (Conditional Probability Tables) for inference.
    """

    def __init__(self):
        # P(Weather = Severe)
        self.p_weather = {"severe": 0.4, "mild": 0.6}

        # P(Severity | Weather)
        self.p_severity_given_weather = {
            "severe": {"high": 0.7, "low": 0.3},
            "mild":   {"high": 0.2, "low": 0.8},
        }

        # P(Demand_High | Severity, Population)
        self.p_demand_given_sv_pop = {
            ("high", "large"):  {"food": 0.9, "medicine": 0.85, "water": 0.95},
            ("high", "small"):  {"food": 0.7, "medicine": 0.6,  "water": 0.75},
            ("low",  "large"):  {"food": 0.4, "medicine": 0.3,  "water": 0.45},
            ("low",  "small"):  {"food": 0.2, "medicine": 0.15, "water": 0.25},
        }

    def prior_demand(self, population: str) -> dict:
        """CO5: Prior probability of high demand by variable elimination"""
        result = {}
        for resource in ["food", "medicine", "water"]:
            p = 0.0
            for weather, pw in self.p_weather.items():
                for severity, ps in self.p_severity_given_weather[weather].items():
                    p += pw * ps * self.p_demand_given_sv_pop[(severity, population)][resource]
            result[resource] = round(p, 3)
        trace("CO5:Bayes", f"Prior demand for {population}: {result}")
        return result

    def posterior_demand(self, observed_weather: str, population: str) -> dict:
        """CO5: Posterior P(Demand | Weather=observed) via Bayes Rule"""
        result = {}
        for resource in ["food", "medicine", "water"]:
            p = 0.0
            for severity, ps in self.p_severity_given_weather[observed_weather].items():
                p += ps * self.p_demand_given_sv_pop[(severity, population)][resource]
            result[resource] = round(p, 3)
        trace("CO5:Bayes", f"Posterior demand given {observed_weather} weather: {result}")
        return result

    def likelihood_weighting_sample(self, n_samples: int, weather: str, population: str) -> dict:
        """CO5: Approximate inference via likelihood weighting (sampling)"""
        totals = defaultdict(float)
        total_weight = 0.0
        for _ in range(n_samples):
            # Sample severity given evidence
            sev_probs = self.p_severity_given_weather[weather]
            severity = "high" if random.random() < sev_probs["high"] else "low"
            weight = sev_probs[severity]  # likelihood weight
            for resource in ["food", "medicine", "water"]:
                p = self.p_demand_given_sv_pop[(severity, population)][resource]
                totals[resource] += weight * p
            total_weight += weight
        return {r: round(totals[r] / total_weight, 3) for r in ["food", "medicine", "water"]}

    def markov_chain_severity(self, steps: int, start: str = "low") -> list:
        """CO5: Markov Chain for tracking evolving severity"""
        transition = {
            "low":  {"low": 0.7, "high": 0.3},
            "high": {"low": 0.4, "high": 0.6},
        }
        state = start
        sequence = [state]
        for _ in range(steps):
            probs = transition[state]
            state = "high" if random.random() < probs["high"] else "low"
            sequence.append(state)
        return sequence

    def estimate_demand_units(self, camp: Location, weather: str) -> dict:
        """CO5: Expected demand in units = P(high) * max_demand + P(low) * min_demand"""
        pop = "large" if camp.severity >= 3 else "small"
        probs = self.posterior_demand(weather, pop)
        max_d = {"food": 50, "medicine": 30, "water": 100}
        min_d = {"food": 10, "medicine": 5, "water": 20}
        return {
            r: round(probs[r] * max_d[r] + (1 - probs[r]) * min_d[r])
            for r in ["food", "medicine", "water"]
        }

    def expected_utility(self, allocation: dict, demand_probs: dict) -> float:
        """CO5: Expected utility = sum over outcomes of P(outcome) * utility"""
        eu = 0.0
        for resource, amount in allocation.items():
            p_high = demand_probs.get(resource, 0.5)
            # Utility: surplus penalized, shortfall penalized more
            expected_need = p_high * 50 + (1 - p_high) * 10
            if amount >= expected_need:
                eu += 10 - 0.1 * (amount - expected_need)  # slight waste penalty
            else:
                eu += 10 * (amount / expected_need) - 5     # shortage penalty
        return round(eu, 2)


# ─────────────────────────────────────────────
# CO6: Hybrid Resource Planner Agent
# ─────────────────────────────────────────────
class ResourcePlannerAgent:
    """
    CO6: Hybrid architecture:
    1. Bayesian demand estimation (CO5)
    2. CSP allocation (CO3)
    3. A* routing (CO2)
    4. Minimax policy selection (CO4)
    5. Full explainability trace (CO1, CO6)
    """

    def __init__(self):
        self.graph = RoadNetwork()
        self.search = SearchEngine(self.graph)
        self.bayes = BayesianDemandEstimator()
        self.multi_agent = MultiAgentPlanner(self.graph)
        self._build_network()

    def _build_network(self):
        """Build sample disaster relief network"""
        locations = [
            Location("D1", "Main Depot",        17.38, 78.47, supply={"food":500,"medicine":200,"water":1000}, is_depot=True),
            Location("D2", "Secondary Depot",   17.45, 78.50, supply={"food":200,"medicine":100,"water":400},  is_depot=True),
            Location("C1", "Camp Alpha",         17.30, 78.40, severity=4, is_camp=True),
            Location("C2", "Camp Beta",          17.35, 78.55, severity=3, is_camp=True),
            Location("C3", "Camp Gamma",         17.25, 78.45, severity=5, is_camp=True),
            Location("C4", "Camp Delta",         17.42, 78.35, severity=2, is_camp=True),
            Location("H1", "Hub Junction",       17.38, 78.42),
            Location("H2", "Highway Node",       17.40, 78.50),
            Location("B1", "Blocked Bridge",     17.32, 78.48),
        ]
        for loc in locations:
            self.graph.add_location(loc)

        roads = [
            ("D1","H1",5.0,True), ("D1","H2",8.0,True), ("D2","H2",4.0,True),
            ("H1","C1",6.0,True), ("H1","C4",7.0,True), ("H2","C2",5.0,True),
            ("H2","H1",3.0,True), ("C1","B1",4.0,False), ("B1","C3",3.0,False),
            ("H1","C3",12.0,True), ("C2","C3",8.0,True), ("D1","C4",10.0,True),
        ]
        for a, b, d, p in roads:
            self.graph.add_road(a, b, d, p)

    def run(self, weather: str = "severe") -> dict:
        trace("CO6:Hybrid", "=== Starting Hybrid Resource Planner ===")
        results = {}

        # Step 1: CO5 - Bayesian demand estimation
        trace("CO6:Step1", "Bayesian demand estimation")
        camp_demands = {}
        camp_eu = {}
        for cid, loc in self.graph.locations.items():
            if loc.is_camp:
                demand = self.bayes.estimate_demand_units(loc, weather)
                loc.demand = demand
                camp_demands[cid] = demand
                probs = self.bayes.posterior_demand(weather, "large" if loc.severity >= 3 else "small")
                camp_eu[cid] = self.bayes.expected_utility(demand, probs)
        results["bayesian_demands"] = camp_demands
        results["expected_utilities"] = camp_eu

        # Markov chain severity tracking
        results["severity_forecast"] = self.bayes.markov_chain_severity(5)

        # Step 2: CO3 - CSP supply allocation
        trace("CO6:Step2", "CSP supply allocation")
        camps = [c for c in self.graph.locations if self.graph.locations[c].is_camp]
        reachable = set()
        for camp in camps:
            _, cost = self.search.astar("D1", camp)
            if cost < float('inf'):
                reachable.add(camp)
        trace("CO3:CSP", f"Reachable camps: {reachable}")

        total_stock = {"food": 700, "medicine": 300, "water": 1400}
        priority_camps = {c for c in camps if self.graph.locations[c].severity >= 4}

        csp = SupplyAllocationCSP(
            camps=camps, resources=["food","medicine","water"],
            stock=total_stock, demand=camp_demands,
            reachable=reachable, priority_camps=priority_camps
        )
        allocation = csp.backtrack()
        if allocation is None:
            trace("CO3:CSP", "Backtracking failed, using min-conflicts local search")
            allocation = csp.min_conflicts_local()
        results["allocation"] = {f"{c}/{r}": v for (c,r), v in allocation.items()}
        results["csp_failure_log"] = csp.failure_log[:5]

        # Step 3: CO2 - Optimal routing with algorithm comparison
        trace("CO6:Step3", "Route planning with search algorithms")
        routes = {}
        algo_comparison = {}
        for camp in reachable:
            comparison = self.search.compare_algorithms("D1", camp)
            algo_comparison[camp] = comparison
            best_path, best_cost = self.search.astar("D1", camp)
            routes[camp] = {"path": best_path, "cost": round(best_cost, 2)}
        results["routes"] = routes
        results["algorithm_comparison"] = algo_comparison

        # Step 4: CO4 - Multi-agent policy with minimax
        trace("CO6:Step4", "Multi-agent policy selection")
        total_demand = sum(sum(d.values()) for d in camp_demands.values())
        policies = self.multi_agent.policy_selection("D1", list(reachable), total_demand)
        results["policies"] = policies

        # Step 5: CO6 - Explainability
        trace("CO6:Step5", "Generating explainability report")
        explain = []
        for camp in reachable:
            loc = self.graph.locations[camp]
            alloc_food = allocation.get((camp, "food"), 0)
            path, cost = routes[camp]["path"], routes[camp]["cost"]
            policy = policies.get(camp, {})
            eu = camp_eu.get(camp, 0)
            explain.append({
                "camp": loc.name,
                "severity": loc.severity,
                "estimated_demand": camp_demands.get(camp, {}),
                "allocated": {r: allocation.get((camp,r),0) for r in ["food","medicine","water"]},
                "route": " → ".join(self.graph.locations[n].name for n in path),
                "route_cost_km": cost,
                "expected_utility": eu,
                "minimax_utility": policy.get("utility", 0),
                "recommended_next_stop": self.graph.locations.get(policy.get("recommended_next",""), Location("?","?",0,0)).name,
                "why_priority": f"Severity {loc.severity}/5 → {'HIGH PRIORITY' if camp in priority_camps else 'standard'}",
            })
        results["explainability"] = explain
        results["weather"] = weather
        results["total_demand"] = total_demand
        results["reachable_camps"] = list(reachable)

        trace("CO6:Hybrid", "=== Planning Complete ===")
        return results


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    agent = ResourcePlannerAgent()

    print("\n" + "="*60)
    print("   RESOURCE PLANNER RELIEF SYSTEM")
    print("   Disaster Response AI Agent")
    print("="*60)

    for weather in ["severe", "mild"]:
        print(f"\n{'─'*50}")
        print(f"  Scenario: {weather.upper()} weather conditions")
        print(f"{'─'*50}")
        results = agent.run(weather)

        print("\n[CO5] Bayesian Demand Estimates:")
        for camp, demand in results["bayesian_demands"].items():
            name = agent.graph.locations[camp].name
            print(f"  {name}: {demand}")

        print("\n[CO5] Severity Forecast (Markov Chain):", results["severity_forecast"])

        print("\n[CO3] CSP Allocation (backtracking + MRV/LCV/FC):")
        for key, val in sorted(results["allocation"].items()):
            print(f"  {key}: {val}")

        print("\n[CO2] Optimal Routes (A*):")
        for camp, r in results["routes"].items():
            name = agent.graph.locations[camp].name
            print(f"  {name}: {r['path']} | cost={r['cost']} km")

        print("\n[CO2] Algorithm Comparison (D1 → C3):")
        if "C3" in results["algorithm_comparison"]:
            for algo, stats in results["algorithm_comparison"]["C3"].items():
                print(f"  {algo:8s} | cost={stats['cost']:6.1f} | expansions={stats['expansions']:3d} | mem={stats['peak_memory']:3d} | {stats['runtime_ms']:.2f}ms")

        print("\n[CO4] Minimax Policy Decisions:")
        for camp, pol in results["policies"].items():
            name = agent.graph.locations[camp].name
            print(f"  {name}: utility={pol['utility']:.1f}, nodes_evaluated={pol['nodes_evaluated']}")

        print("\n[CO6] Explainability Report:")
        for e in results["explainability"]:
            print(f"\n  Camp: {e['camp']} | {e['why_priority']}")
            print(f"    Estimated Demand: {e['estimated_demand']}")
            print(f"    Allocated:        {e['allocated']}")
            print(f"    Route:            {e['route']} ({e['route_cost_km']} km)")
            print(f"    Expected Utility: {e['expected_utility']} | Minimax: {e['minimax_utility']}")
