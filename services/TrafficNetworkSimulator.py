import json
import os
import math
import copy
import random
import time as _time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle


@dataclass
class TrafficLightPhase:
    green_incoming_roads: List[int]
    duration_seconds: float = 30.0


@dataclass
class TrafficLight:
    phases: List[TrafficLightPhase] = field(default_factory=list)
    offset_seconds: float = 0.0

    def cycle_length(self) -> float:
        return sum(p.duration_seconds for p in self.phases) or 1.0

    def active_phase(self, t: float) -> int:
        if not self.phases:
            return -1
        cycle = self.cycle_length()
        phase_t = (t + self.offset_seconds) % cycle
        acc = 0.0
        for i, p in enumerate(self.phases):
            acc += p.duration_seconds
            if phase_t < acc:
                return i
        return len(self.phases) - 1

    def is_green_for(self, road_id: int, t: float) -> bool:
        idx = self.active_phase(t)
        if idx < 0:
            return True
        return road_id in self.phases[idx].green_incoming_roads


@dataclass
class Road:
    road_id: int
    src: Optional[int]
    dst: Optional[int]
    base_travel_time: float
    capacity: int
    inflow_rate: float = 0.0
    cars_on_road: List["Car"] = field(default_factory=list)

    def current_travel_time(self) -> float:
        if self.capacity <= 0:
            return self.base_travel_time
        load = len(self.cars_on_road) / self.capacity
        return self.base_travel_time * (1.0 + 0.6 * (load ** 4))

    def is_full(self) -> bool:
        return len(self.cars_on_road) >= self.capacity

    def free_slots(self) -> int:
        return max(0, self.capacity - len(self.cars_on_road))


@dataclass
class Node:
    node_id: int
    position: Tuple[float, float]
    incoming_roads: List[int] = field(default_factory=list)
    outgoing_roads: List[int] = field(default_factory=list)
    traffic_light: Optional[TrafficLight] = None
    saturation_flow: float = 0.5


@dataclass
class Car:
    car_id: int
    path: List[int]
    path_pos: int = 0
    progress: float = 0.0
    spawn_time: float = 0.0
    finished: bool = False
    waiting: bool = False
    wait_time: float = 0.0


@dataclass
class GAConfig:
    population_size: int = 20
    num_generations: int = 25
    elite_fraction: float = 0.2
    crossover_rate: float = 0.7
    mutation_rate: float = 0.3
    mutation_sigma: float = 5.0
    tournament_size: int = 3
    min_phase_duration: float = 5.0
    max_phase_duration: float = 60.0
    min_phases: int = 2
    max_phases: int = 4
    eval_ticks: int = 600
    eval_seed: int = 42
    fitness: str = 'delay'


@dataclass
class Individual:
    phases: List[Tuple[Tuple[str, ...], float]]
    fitness: Optional[float] = None
    metrics: dict = field(default_factory=dict)

    def cycle_length(self) -> float:
        return sum(d for _, d in self.phases)

    def to_phase_list(self) -> List[TrafficLightPhase]:
        return [TrafficLightPhase(list(group), float(dur))
                for group, dur in self.phases]


class BaseTrafficSimulator:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.nodes: Dict[int, Node] = {}
        self.roads: Dict[int, Road] = {}
        self._next_road_id = 0
        self._next_car_id = 0
        self.source_road_ids: List[int] = []
        self.sink_road_ids: List[int] = []

        self.t = 0.0
        self.cars: Dict[int, Car] = {}
        self.finished_cars: List[Car] = []
        self.history: List[dict] = []
        self._spawn_accumulator: Dict[int, float] = defaultdict(float)

        self._intersection_budget: Dict[int, float] = defaultdict(float)

        self.cumulative_delay = 0.0
        self.total_wait_time = 0.0

    def _add_road(self, src, dst, base_travel_time, capacity=10, inflow_rate=0.0):
        rid = self._next_road_id
        self._next_road_id += 1
        road = Road(road_id=rid, src=src, dst=dst,
                    base_travel_time=float(base_travel_time),
                    capacity=int(capacity), inflow_rate=float(inflow_rate))
        self.roads[rid] = road
        if src is not None:
            self.nodes[src].outgoing_roads.append(rid)
        if dst is not None:
            self.nodes[dst].incoming_roads.append(rid)
        return rid

    def _bfs_reachable_nodes(self, start_node):
        seen = {start_node}
        q = deque([start_node])
        while q:
            u = q.popleft()
            for rid in self.nodes[u].outgoing_roads:
                v = self.roads[rid].dst
                if v is not None and v not in seen:
                    seen.add(v)
                    q.append(v)
        return seen

    def _shortest_path_roads(self, start_node, goal_sink_road_id):
        sink_road = self.roads[goal_sink_road_id]
        goal_node = sink_road.src
        if start_node == goal_node:
            return [goal_sink_road_id]
        dist = {start_node: 0.0}
        prev_road = {}
        visited = set()
        frontier = [(0.0, start_node)]
        while frontier:
            frontier.sort()
            d, u = frontier.pop(0)
            if u in visited:
                continue
            visited.add(u)
            if u == goal_node:
                break
            for rid in self.nodes[u].outgoing_roads:
                r = self.roads[rid]
                if r.dst is None:
                    continue
                nd = d + r.current_travel_time()
                if nd < dist.get(r.dst, float('inf')):
                    dist[r.dst] = nd
                    prev_road[r.dst] = rid
                    frontier.append((nd, r.dst))
        if goal_node not in prev_road and start_node != goal_node:
            return None
        path = []
        cur = goal_node
        while cur != start_node:
            rid = prev_road[cur]
            path.append(rid)
            cur = self.roads[rid].src
        path.reverse()
        path.append(goal_sink_road_id)
        return path

    def _pick_goal_sink(self, entry_node):
        reachable = self._bfs_reachable_nodes(entry_node)
        candidates = [rid for rid in self.sink_road_ids
                      if self.roads[rid].src in reachable
                      or self.roads[rid].src == entry_node]
        if not candidates:
            return None
        return self.rng.choice(candidates)

    def _spawn_cars(self, dt):
        for src_rid in self.source_road_ids:
            road = self.roads[src_rid]
            self._spawn_accumulator[src_rid] += road.inflow_rate * dt
            while self._spawn_accumulator[src_rid] >= 1.0:
                if road.is_full():
                    break
                self._spawn_accumulator[src_rid] -= 1.0
                entry_node = road.dst
                goal = self._pick_goal_sink(entry_node)
                if goal is None:
                    continue
                rest = self._shortest_path_roads(entry_node, goal)
                if rest is None:
                    continue
                full_path = [src_rid] + rest
                car = Car(car_id=self._next_car_id, path=full_path, spawn_time=self.t)
                self._next_car_id += 1
                road.cars_on_road.append(car)
                self.cars[car.car_id] = car

    def step(self, dt=1.0):
        self._spawn_cars(dt)

        for c in self.cars.values():
            c.waiting = False

        ready = []
        for road in self.roads.values():
            tt = road.current_travel_time()
            for car in road.cars_on_road:
                car.progress += dt
                if car.progress > tt:
                    car.progress = tt
                if car.progress >= tt:
                    ready.append((car, road))

        finishing = []
        by_node = defaultdict(list)
        for car, road in ready:
            next_idx = car.path_pos + 1
            if next_idx >= len(car.path):
                finishing.append((car, road))
                continue
            if road.dst is None:
                finishing.append((car, road))
                continue
            by_node[road.dst].append((car, road))

        for car, road in finishing:
            self._finish_car(car, road)
            
        for node_id, node in self.nodes.items():
            if node.traffic_light is not None:
                phase_idx = node.traffic_light.active_phase(self.t)
                green_roads = (set(node.traffic_light.phases[phase_idx].green_incoming_roads)
                               if phase_idx >= 0 else set(node.incoming_roads))
            else:
                green_roads = set(node.incoming_roads)

            for rid in node.incoming_roads:
                if rid in green_roads:
                    self._intersection_budget[rid] = min(
                        node.saturation_flow * 2.0,
                        self._intersection_budget[rid] + node.saturation_flow * dt
                    )
                else:
                    self._intersection_budget[rid] = 0.0

        for node_id, queue in by_node.items():
            queue.sort(key=lambda cr: (-cr[0].wait_time, -cr[0].progress))
            node = self.nodes[node_id]

            if node.traffic_light is not None:
                phase_idx = node.traffic_light.active_phase(self.t)
                green_roads = (set(node.traffic_light.phases[phase_idx].green_incoming_roads)
                               if phase_idx >= 0 else set(node.incoming_roads))
            else:
                green_roads = set(node.incoming_roads)

            for car, road in queue:
                next_idx = car.path_pos + 1
                next_road = self.roads[car.path[next_idx]]

                if road.road_id not in green_roads:
                    car.waiting = True
                    continue
                if self._intersection_budget[road.road_id] < 1.0:
                    car.waiting = True
                    continue
                if next_road.is_full():
                    car.waiting = True
                    continue

                self._intersection_budget[road.road_id] -= 1.0
                road.cars_on_road.remove(car)
                car.path_pos = next_idx
                car.progress = 0.0
                next_road.cars_on_road.append(car)

        for road in self.roads.values():
            if not road.cars_on_road:
                continue
            tt = road.current_travel_time()
            sorted_cars = sorted(road.cars_on_road, key=lambda c: -c.progress)
            if not sorted_cars[0].waiting:
                continue
            slot = tt / max(1, road.capacity)
            queue_back = tt
            for c in sorted_cars:
                if c.progress >= queue_back - 0.5 * slot:
                    c.waiting = True
                    queue_back -= slot
                else:
                    break

        waiting_now = 0
        for c in self.cars.values():
            if c.waiting:
                c.wait_time += dt
                waiting_now += 1
        self.cumulative_delay += waiting_now * dt

        self.t += dt
        self._snapshot(waiting_now)

    def _finish_car(self, car, road):
        if car in road.cars_on_road:
            road.cars_on_road.remove(car)
        car.finished = True
        self.total_wait_time += car.wait_time
        self.finished_cars.append(car)
        self.cars.pop(car.car_id, None)

    def run(self, num_ticks, dt=1.0):
        for _ in range(num_ticks):
            self.step(dt)

    def _snapshot(self, waiting_now):
        self.history.append({
            't': self.t,
            'road_loads': {rid: len(r.cars_on_road) for rid, r in self.roads.items()},
            'phases': {nid: (n.traffic_light.active_phase(self.t)
                              if n.traffic_light else -1)
                       for nid, n in self.nodes.items()},
            'active_cars': len(self.cars),
            'finished_cars': len(self.finished_cars),
            'waiting_cars': waiting_now,
            'cumulative_delay': self.cumulative_delay,
        })

    def metrics(self) -> dict:
        finished = self.finished_cars
        avg_wait = float(np.mean([c.wait_time for c in finished])) if finished else 0.0
        return {
            't_total': self.t,
            'cars_finished': len(finished),
            'cars_active': len(self.cars),
            'cumulative_delay_veh_seconds': self.cumulative_delay,
            'avg_wait_per_finished_car': avg_wait,
            'total_wait_time_finished': float(self.total_wait_time),
            'throughput_per_min': len(finished) / (self.t / 60.0) if self.t > 0 else 0.0,
        }

    def print_metrics(self, label=""):
        m = self.metrics()
        head = f"=== Metrics {label} ===" if label else "=== Metrics ==="
        print(head)
        print(f"  Sim time:              {m['t_total']:.0f}s")
        print(f"  Cars finished:         {m['cars_finished']}")
        print(f"  Cars still active:     {m['cars_active']}")
        print(f"  Cumulative delay:      {m['cumulative_delay_veh_seconds']:.0f} veh·s")
        print(f"  Avg wait/finished car: {m['avg_wait_per_finished_car']:.2f}s")
        print(f"  Throughput:            {m['throughput_per_min']:.2f} cars/min")

    def diagnose_road_loads(self, top_n=10):
        loads = sorted(self.roads.values(),
                       key=lambda r: -len(r.cars_on_road))[:top_n]
        print("\n--- Most congested roads ---")
        for r in loads:
            n = len(r.cars_on_road)
            n_wait = sum(1 for c in r.cars_on_road if c.waiting)
            tag = ""
            if r.src is None: tag = " [SOURCE]"
            elif r.dst is None: tag = " [SINK]"
            print(f"  R{r.road_id:>3}: {n:>3}/{r.capacity:<3} cars  "
                  f"({n_wait} waiting){tag}  src={r.src} dst={r.dst}  "
                  f"budget={self._intersection_budget.get(r.road_id, 0.0):.2f}")

    def set_traffic_light_phases(self, node_id, phases, offset=0.0):
        if node_id in self.nodes:
            self.nodes[node_id].traffic_light = TrafficLight(phases=phases,
                                                              offset_seconds=offset)

    def get_state_for_gnn(self) -> dict:
        node_features = {}
        for nid, node in self.nodes.items():
            inc_load = sum(len(self.roads[r].cars_on_road) for r in node.incoming_roads)
            inc_wait = sum(sum(1 for c in self.roads[r].cars_on_road if c.waiting)
                           for r in node.incoming_roads)
            node_features[nid] = {
                'incoming_load': inc_load,
                'incoming_waiting': inc_wait,
                'outgoing_load': sum(len(self.roads[r].cars_on_road)
                                     for r in node.outgoing_roads),
                'num_incoming': len(node.incoming_roads),
                'num_outgoing': len(node.outgoing_roads),
                'phase_idx': node.traffic_light.active_phase(self.t)
                              if node.traffic_light else -1,
            }
        edges = [{
            'road_id': rid, 'src': r.src, 'dst': r.dst,
            'load': len(r.cars_on_road), 'capacity': r.capacity,
            'travel_time': r.current_travel_time(),
            'base_travel_time': r.base_travel_time,
        } for rid, r in self.roads.items()]
        return {'t': self.t, 'nodes': node_features, 'edges': edges}

    def export_time_series(self, sequence_key='synthetic_sim',
                           bin_seconds=1.0, output_path=None):
        if not self.history:
            raise RuntimeError("No simulation history. Call run() first.")
        num_bins = len(self.history)
        roads_out = {}
        prev_loads = {rid: 0 for rid in self.roads}
        for rid, road in self.roads.items():
            entries, exits = [], []
            for snap in self.history:
                load = snap['road_loads'][rid]
                delta = load - prev_loads[rid]
                entries.append(max(0, delta))
                exits.append(max(0, -delta))
                prev_loads[rid] = load
            line = self._road_line_for_export(road)
            roads_out[str(rid)] = {
                'road_index': rid, 'fps': 1.0, 'bin_size_frames': 1,
                'bin_seconds': bin_seconds, 'num_bins': num_bins,
                'entry_direction': 'from_left', 'exit_direction': 'from_right',
                'time_series_entries': entries, 'time_series_exits': exits,
                'total_entries': int(sum(entries)), 'total_exits': int(sum(exits)),
                'unique_entry_tracks': int(sum(entries)),
                'unique_exit_tracks': int(sum(exits)),
                'entries_by_class': {'car': int(sum(entries))},
                'exits_by_class': {'car': int(sum(exits))},
                'line': line,
                'entry_events': [], 'exit_events': [], 'crossings': [],
            }
        record = {
            'video': '', 'sequence_key': sequence_key, 'fps': 1.0,
            'bin_size_frames': 1, 'bin_seconds': bin_seconds,
            'num_bins': num_bins, 'num_roads': len(roads_out), 'roads': roads_out,
        }
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(record, f, indent=2)
        return record

    def _road_line_for_export(self, road):
        a = list(self.nodes[road.src].position) if road.src is not None else [0, 0]
        b = list(self.nodes[road.dst].position) if road.dst is not None else [0, 0]
        return [a, b]

    @staticmethod
    def _draw_road_with_cars(ax, x1, y1, x2, y2, road, *,
                              show_label=True, label_extra="",
                              car_size=5, lateral_offset=6,
                              shrinkA=8, shrinkB=22):
        load = len(road.cars_on_road) / max(1, road.capacity)
        color = plt.cm.RdYlGn_r(min(1.0, load))
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle='-|>', mutation_scale=18,
                                     color=color, linewidth=2 + 3 * load,
                                     alpha=0.7, shrinkA=shrinkA, shrinkB=shrinkB))
        tt = road.current_travel_time()
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        nx_, ny_ = -uy * lateral_offset, ux * lateral_offset
        slot_px = max(8.0, min(20.0, (L - 30) / max(road.capacity, 1)))
        sorted_cars = sorted(road.cars_on_road, key=lambda c: -c.progress)
        for i, car in enumerate(sorted_cars):
            if car.waiting:
                d_from_end = 18 + i * slot_px
                px = x2 - ux * d_from_end + nx_
                py = y2 - uy * d_from_end + ny_
                cc = '#c0392b'
            else:
                frac = min(1.0, car.progress / max(0.001, tt))
                px = x1 + dx * frac + nx_
                py = y1 + dy * frac + ny_
                cc = '#2c3e50'
            ax.plot(px, py, 'o', color=cc, markersize=car_size,
                    markeredgecolor='white', markeredgewidth=0.5, zorder=4)
        if show_label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            n_cars = len(road.cars_on_road)
            n_wait = sum(1 for c in road.cars_on_road if c.waiting)
            label = f"R{road.road_id}\n{n_cars}/{road.capacity}"
            if n_wait:
                label += f"\nwait:{n_wait}"
            if label_extra:
                label += f"\n{label_extra}"
            ax.text(mx, my, label, fontsize=7, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.85, pad=2,
                              edgecolor='gray'),
                    zorder=5)


class SingleIntersectionSimulator(BaseTrafficSimulator):
    DIRECTIONS_4 = ['N', 'E', 'S', 'W']
    DIRECTIONS_3 = ['N', 'E', 'S']

    def __init__(self, num_approaches=4, approach_config=None,
                 traffic_light_phases=None, road_length=200.0,
                 base_travel_time=12.0, capacity=15,
                 saturation_flow=0.5, seed=None):
        super().__init__(seed=seed)
        if num_approaches not in (3, 4):
            raise ValueError("num_approaches must be 3 or 4")
        self.num_approaches = num_approaches
        self.directions = (self.DIRECTIONS_4 if num_approaches == 4
                           else self.DIRECTIONS_3)
        self.canvas_size = (600.0, 600.0)
        self.saturation_flow = saturation_flow
        self._build(approach_config or {}, road_length, base_travel_time, capacity)
        self._set_lights(traffic_light_phases)

    def _build(self, approach_config, road_length, base_travel_time, capacity):
        cx, cy = self.canvas_size[0] / 2, self.canvas_size[1] / 2
        self.nodes[0] = Node(node_id=0, position=(cx, cy),
                              saturation_flow=self.saturation_flow)

        offsets = {'N': (0, -road_length), 'S': (0, road_length),
                   'E': (road_length, 0), 'W': (-road_length, 0)}
        self.approach_in = {}
        self.approach_out = {}
        self._stub_points = {}

        for d in self.directions:
            cfg = approach_config.get(d, {})
            inflow = cfg.get('inflow_rate', 0.25)
            cap = cfg.get('capacity', capacity)
            btt = cfg.get('base_travel_time', base_travel_time)
            ox, oy = offsets[d]
            stub_pt = (cx + ox, cy + oy)

            in_rid = self._add_road(src=None, dst=0, base_travel_time=btt,
                                     capacity=cap, inflow_rate=inflow)
            self._stub_points[in_rid] = stub_pt
            self.source_road_ids.append(in_rid)
            self.approach_in[d] = in_rid

            out_rid = self._add_road(src=0, dst=None, base_travel_time=btt,
                                     capacity=999)
            self._stub_points[out_rid] = stub_pt
            self.sink_road_ids.append(out_rid)
            self.approach_out[d] = out_rid

    def _set_lights(self, phases):
        if phases is None:
            if self.num_approaches == 4:
                phases = [
                    TrafficLightPhase([self.approach_in['N'], self.approach_in['S']], 25.0),
                    TrafficLightPhase([self.approach_in['E'], self.approach_in['W']], 25.0),
                ]
            else:
                phases = [
                    TrafficLightPhase([self.approach_in['N']], 20.0),
                    TrafficLightPhase([self.approach_in['E']], 20.0),
                    TrafficLightPhase([self.approach_in['S']], 20.0),
                ]
        else:
            resolved = []
            for p in phases:
                green = []
                for entry in p.green_incoming_roads:
                    if isinstance(entry, str):
                        green.append(self.approach_in[entry])
                    else:
                        green.append(entry)
                resolved.append(TrafficLightPhase(green, p.duration_seconds))
            phases = resolved
        self.nodes[0].traffic_light = TrafficLight(phases=phases)

    def _pick_goal_sink(self, entry_node):
        return self.rng.choice(self.sink_road_ids)

    def _road_endpoints(self, road):
        cx, cy = self.nodes[0].position
        stub = self._stub_points.get(road.road_id)
        if road.src is None:
            return stub[0], stub[1], cx, cy
        return cx, cy, stub[0], stub[1]

    def draw(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 9))
        cx, cy = self.nodes[0].position
        for rid, road in self.roads.items():
            x1, y1, x2, y2 = self._road_endpoints(road)
            extra = f"λ={road.inflow_rate:.2f}/s" if road.src is None else ""
            self._draw_road_with_cars(ax, x1, y1, x2, y2, road, label_extra=extra)

        node = self.nodes[0]
        phase_idx = node.traffic_light.active_phase(self.t) if node.traffic_light else -1
        phase_color = (['#2ecc71', '#e74c3c', '#3498db', '#f39c12'][phase_idx % 4]
                       if phase_idx >= 0 else 'lightgray')
        ax.add_patch(Rectangle((cx - 25, cy - 25), 50, 50,
                                facecolor=phase_color, edgecolor='black', zorder=5))
        ax.text(cx, cy, f"P{phase_idx}" if phase_idx >= 0 else "—",
                fontsize=11, fontweight='bold', ha='center', va='center', zorder=6)

        offsets = {'N': (0, -1), 'S': (0, 1), 'E': (1, 0), 'W': (-1, 0)}
        for d in self.directions:
            ox, oy = offsets[d]
            ax.text(cx + ox * 250, cy + oy * 250, d,
                    fontsize=14, fontweight='bold', ha='center', va='center',
                    bbox=dict(facecolor='yellow', alpha=0.8, pad=3))

        ax.set_xlim(cx - 300, cx + 300)
        ax.set_ylim(cy - 300, cy + 300)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        waiting_now = sum(1 for c in self.cars.values() if c.waiting)
        ax.set_title(
            f"Intersection (t={self.t:.0f}s | active={len(self.cars)} | "
            f"finished={len(self.finished_cars)} | waiting={waiting_now} | "
            f"delay={self.cumulative_delay:.0f} veh·s | sat={self.saturation_flow:.2f}/s)",
            fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)
        return ax

    def animate(self, num_ticks=180, dt=1.0, interval_ms=120, save_path=None, dpi=100):
        fig, ax = plt.subplots(figsize=(9, 9))

        def update(frame):
            ax.clear()
            self.step(dt)
            self.draw(ax=ax)
            return []

        anim = animation.FuncAnimation(fig, update, frames=num_ticks,
                                       interval=interval_ms, blit=False, repeat=False)
        if save_path:
            ext = os.path.splitext(save_path)[1].lower()
            fps = max(1, int(1000 / interval_ms))
            try:
                writer = 'ffmpeg' if ext in ('.mp4', '.mov') else 'pillow'
                anim.save(save_path, writer=writer, fps=fps, dpi=dpi)
                print(f"  Saved animation to {save_path}")
            except Exception as e:
                print(f"  Could not save animation: {e}")
        plt.close(fig)
        return anim

    def generate_synthetic_dataset(self, dataset_rules, output_dir, seed=None):
        files = []
        for rule in dataset_rules:
            sequence_key = rule['sequence_key']
            num_bins = rule['num_bins']
            fps = rule['fps']
            bin_seconds = rule['bin_seconds']
            roads_spec = rule['roads']
            num_roads = len(roads_spec)

            if num_roads not in (3, 4):
                raise ValueError("num_roads must be 3 or 4")

            traffic_light = roads_spec[0].get('traffic_light', {})
            cycle_length = traffic_light.get('cycle_length_seconds', 60.0)
            green_time = traffic_light.get('green_time_seconds', 30.0)
            offset = traffic_light.get('offset_seconds', 0.0)
            green_flow = traffic_light.get('green_flow_mean', 5.0)

            approach_config = {}
            directions = self.DIRECTIONS_4 if num_roads == 4 else self.DIRECTIONS_3
            for d in directions:
                approach_config[d] = {'inflow_rate': green_flow}

            sim = SingleIntersectionSimulator(num_approaches=num_roads, approach_config=approach_config, seed=seed)

            # Set traffic light phases
            if num_roads == 4:
                phases = [
                    TrafficLightPhase([sim.approach_in['N'], sim.approach_in['S']], green_time),
                    TrafficLightPhase([sim.approach_in['E'], sim.approach_in['W']], cycle_length - green_time),
                ]
            else:
                phase_duration = green_time / 3
                red_duration = (cycle_length - green_time) / 2
                phases = [
                    TrafficLightPhase([sim.approach_in['N']], phase_duration),
                    TrafficLightPhase([sim.approach_in['E']], phase_duration),
                    TrafficLightPhase([sim.approach_in['S']], red_duration),
                ]
            sim.set_traffic_light_phases(0, phases, offset)

            # Run simulation
            duration_seconds = num_bins * bin_seconds
            num_ticks = int(duration_seconds / bin_seconds)
            sim.run(num_ticks, dt=bin_seconds)

            # Export
            output_path = os.path.join(output_dir, f'{sequence_key}.json')
            record = sim.export_time_series(sequence_key=sequence_key, bin_seconds=bin_seconds, output_path=output_path)
            files.append(output_path)

        return files


class TrafficNetworkSimulator(BaseTrafficSimulator):

    DEFAULT_CONFIG = {
        'num_nodes': 8,
        'canvas_size': (1000.0, 800.0),
        'travel_time_range': (5.0, 12.0),
        'capacity_range': (12, 25),
        'source_inflow_range': (0.05, 0.15),
        'num_sources': 3,
        'num_sinks': 3,
        'phase_duration_range': (15.0, 30.0),
        'saturation_flow_range': (0.5, 0.8),
        'sink_capacity': 999,
        'source_capacity': 999,
    }

    def __init__(self, config=None, seed=None):
        super().__init__(seed=seed)
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        self.cfg = cfg
        self.num_nodes = cfg['num_nodes']
        self.canvas_size = cfg['canvas_size']
        self.travel_time_range = cfg['travel_time_range']
        self.capacity_range = cfg['capacity_range']
        self.source_inflow_range = cfg['source_inflow_range']
        self.num_sources = cfg['num_sources']
        self.num_sinks = cfg['num_sinks']
        self.phase_duration_range = cfg['phase_duration_range']
        self.saturation_flow_range = cfg['saturation_flow_range']
        self.sink_capacity = cfg['sink_capacity']
        self.source_capacity = cfg['source_capacity']

        if cfg.get('auto_build', True):
            self.build_random_network()

    @staticmethod
    def _segments_intersect(p1, p2, p3, p4, eps=1e-6):
        def ccw(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        shared = lambda a, b: abs(a[0] - b[0]) < eps and abs(a[1] - b[1]) < eps
        if shared(p1, p3) or shared(p1, p4) or shared(p2, p3) or shared(p2, p4):
            return False
        d1, d2 = ccw(p3, p4, p1), ccw(p3, p4, p2)
        d3, d4 = ccw(p1, p2, p3), ccw(p1, p2, p4)
        return (((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps))
                and ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)))

    def _edge_would_cross(self, u, v, existing_pairs):
        p1, p2 = self.nodes[u].position, self.nodes[v].position
        for (a, b) in existing_pairs:
            if {a, b} & {u, v}:
                continue
            if self._segments_intersect(p1, p2, self.nodes[a].position,
                                        self.nodes[b].position):
                return True
        return False

    def _stub_endpoint(self, node_id, used_angles, stub_len=80):
        nx, ny = self.nodes[node_id].position
        cx, cy = self.canvas_size[0] / 2, self.canvas_size[1] / 2
        base_ang = math.atan2(ny - cy, nx - cx)
        best_ang, best_clear = base_ang, -1.0
        for delta in np.linspace(-math.pi / 2, math.pi / 2, 9):
            ang = base_ang + delta
            clear = min((abs((ang - a + math.pi) % (2 * math.pi) - math.pi)
                         for a in used_angles), default=math.pi)
            if clear > best_clear:
                best_clear, best_ang = clear, ang
        used_angles.append(best_ang)
        return (nx + math.cos(best_ang) * stub_len,
                ny + math.sin(best_ang) * stub_len), best_ang

    def _dist(self, u, v):
        x1, y1 = self.nodes[u].position
        x2, y2 = self.nodes[v].position
        return math.hypot(x1 - x2, y1 - y2)

    def _dist_to_time(self, d):
        lo, hi = self.travel_time_range
        max_d = math.hypot(*self.canvas_size)
        return lo + (d / max_d) * (hi - lo)

    def build_random_network(self):
        W, H = self.canvas_size
        cols = max(2, int(math.ceil(math.sqrt(self.num_nodes))))
        rows = int(math.ceil(self.num_nodes / cols))
        cw, ch = (0.8 * W) / cols, (0.8 * H) / rows
        positions = []
        for r in range(rows):
            for c in range(cols):
                if len(positions) >= self.num_nodes:
                    break
                x = 0.1 * W + (c + 0.5) * cw + self.rng.uniform(-cw * 0.2, cw * 0.2)
                y = 0.1 * H + (r + 0.5) * ch + self.rng.uniform(-ch * 0.2, ch * 0.2)
                positions.append((x, y))
        self.rng.shuffle(positions)
        for nid in range(self.num_nodes):
            sat = self.rng.uniform(*self.saturation_flow_range)
            self.nodes[nid] = Node(node_id=nid, position=positions[nid],
                                   saturation_flow=sat)

        existing_pairs = set()

        def has_directed(u, v):
            return any(self.roads[r].dst == v for r in self.nodes[u].outgoing_roads)

        connected = {0}
        unconnected = set(range(1, self.num_nodes))
        while unconnected:
            best = None
            for u in connected:
                for v in unconnected:
                    if self._edge_would_cross(u, v, [tuple(p) for p in existing_pairs]):
                        continue
                    d = self._dist(u, v)
                    if best is None or d < best[2]:
                        best = (u, v, d)
            if best is None:
                u, v, d = min(((u, v, self._dist(u, v))
                               for u in connected for v in unconnected),
                              key=lambda x: x[2])
            else:
                u, v, d = best
            cap = self.rng.randint(*self.capacity_range)
            self._add_road(u, v, base_travel_time=self._dist_to_time(d), capacity=cap)
            self._add_road(v, u, base_travel_time=self._dist_to_time(d), capacity=cap)
            existing_pairs.add(frozenset({u, v}))
            connected.add(v)
            unconnected.remove(v)

        for nid, node in self.nodes.items():
            target = self.rng.choice([3, 4])
            attempts = 0
            while (len(node.incoming_roads) + len(node.outgoing_roads)) < target and attempts < 50:
                attempts += 1
                others = sorted((n for n in self.nodes if n != nid),
                                key=lambda n: self._dist(nid, n)
                                )[:max(3, self.num_nodes // 2)]
                other = self.rng.choice(others)
                pair = frozenset({nid, other})
                if pair in existing_pairs:
                    if not has_directed(nid, other) and self.rng.random() < 0.5:
                        self._add_road(nid, other,
                                       base_travel_time=self._dist_to_time(self._dist(nid, other)),
                                       capacity=self.rng.randint(*self.capacity_range))
                    continue
                if self._edge_would_cross(nid, other, [tuple(p) for p in existing_pairs]):
                    continue
                cap = self.rng.randint(*self.capacity_range)
                btt = self._dist_to_time(self._dist(nid, other))
                roll = self.rng.random()
                if roll < 0.4:
                    self._add_road(nid, other, base_travel_time=btt, capacity=cap)
                elif roll < 0.8:
                    self._add_road(other, nid, base_travel_time=btt, capacity=cap)
                else:
                    self._add_road(nid, other, base_travel_time=btt, capacity=cap)
                    self._add_road(other, nid, base_travel_time=btt, capacity=cap)
                existing_pairs.add(pair)

        self._stub_angles_used = defaultdict(list)
        self._stub_points = {}
        for nid, node in self.nodes.items():
            nx, ny = node.position
            for rid in node.incoming_roads + node.outgoing_roads:
                r = self.roads[rid]
                other = r.src if r.dst == nid else r.dst
                if other is None:
                    continue
                ox, oy = self.nodes[other].position
                self._stub_angles_used[nid].append(math.atan2(oy - ny, ox - nx))

        for nid in self.rng.sample(range(self.num_nodes),
                                    k=min(self.num_sources, self.num_nodes)):
            inflow = self.rng.uniform(*self.source_inflow_range)
            stub_pt, _ = self._stub_endpoint(nid, self._stub_angles_used[nid])
            rid = self._add_road(src=None, dst=nid,
                                 base_travel_time=self.rng.uniform(*self.travel_time_range),
                                 capacity=self.source_capacity,
                                 inflow_rate=inflow)
            self._stub_points[rid] = stub_pt
            self.source_road_ids.append(rid)

        for nid in self.rng.sample(range(self.num_nodes),
                                    k=min(self.num_sinks, self.num_nodes)):
            stub_pt, _ = self._stub_endpoint(nid, self._stub_angles_used[nid])
            rid = self._add_road(src=nid, dst=None,
                                 base_travel_time=self.rng.uniform(*self.travel_time_range),
                                 capacity=self.sink_capacity)
            self._stub_points[rid] = stub_pt
            self.sink_road_ids.append(rid)

        self._build_traffic_lights()
        self._ensure_reachable_sinks()

    def _build_traffic_lights(self):
        for node in self.nodes.values():
            inc = list(node.incoming_roads)
            if len(inc) <= 1:
                node.traffic_light = None
                continue
            bearings = []
            for rid in inc:
                src = self.roads[rid].src
                bx, by = (self.nodes[src].position if src is not None
                           else (node.position[0] - 50, node.position[1] - 50))
                ang = math.atan2(by - node.position[1], bx - node.position[0])
                bearings.append((rid, ang))
            bearings.sort(key=lambda x: x[1])
            ga = [rid for i, (rid, _) in enumerate(bearings) if i % 2 == 0]
            gb = [rid for i, (rid, _) in enumerate(bearings) if i % 2 == 1]
            node.traffic_light = TrafficLight(phases=[
                TrafficLightPhase(ga, self.rng.uniform(*self.phase_duration_range)),
                TrafficLightPhase(gb, self.rng.uniform(*self.phase_duration_range)),
            ], offset_seconds=self.rng.uniform(0, 30))

    def _ensure_reachable_sinks(self):
        sink_nodes = {self.roads[rid].src for rid in self.sink_road_ids}
        for src_rid in self.source_road_ids:
            entry_node = self.roads[src_rid].dst
            reachable = self._bfs_reachable_nodes(entry_node)
            if not (reachable & sink_nodes):
                target = self.rng.choice(list(reachable))
                used = self._stub_angles_used[target]
                stub_pt, _ = self._stub_endpoint(target, used)
                rid = self._add_road(src=target, dst=None,
                                     base_travel_time=self.rng.uniform(*self.travel_time_range),
                                     capacity=self.sink_capacity)
                self._stub_points[rid] = stub_pt
                self.sink_road_ids.append(rid)

    def _road_endpoints(self, road):
        if road.src is not None and road.dst is not None:
            x1, y1 = self.nodes[road.src].position
            x2, y2 = self.nodes[road.dst].position
            return x1, y1, x2, y2
        stub = self._stub_points.get(road.road_id)
        if road.src is None:
            x2, y2 = self.nodes[road.dst].position
            x1, y1 = stub if stub else (x2 - 80, y2 - 80)
        else:
            x1, y1 = self.nodes[road.src].position
            x2, y2 = stub if stub else (x1 + 80, y1 + 80)
        return x1, y1, x2, y2

    def draw_network(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(13, 9))
        for rid, road in self.roads.items():
            x1, y1, x2, y2 = self._road_endpoints(road)
            extra = ""
            if road.src is None:
                extra = f"λ={road.inflow_rate:.2f}/s"
            self._draw_road_with_cars(ax, x1, y1, x2, y2, road,
                                       car_size=3, lateral_offset=4,
                                       label_extra=extra,
                                       shrinkA=12, shrinkB=12)

        for nid, node in self.nodes.items():
            x, y = node.position
            phase = node.traffic_light.active_phase(self.t) if node.traffic_light else -1
            face = 'lightgray' if phase < 0 else (['#2ecc71', '#e74c3c'][phase % 2])
            ax.add_patch(Circle((x, y), 18, facecolor=face, edgecolor='black', zorder=5))
            ax.text(x, y, str(nid), fontsize=11, fontweight='bold',
                    ha='center', va='center', zorder=6)

        for rid in self.source_road_ids:
            x1, y1, _, _ = self._road_endpoints(self.roads[rid])
            ax.plot(x1, y1, marker='s', color='blue', markersize=10)
        for rid in self.sink_road_ids:
            _, _, x2, y2 = self._road_endpoints(self.roads[rid])
            ax.plot(x2, y2, marker='X', color='purple', markersize=11)

        ax.set_xlim(-100, self.canvas_size[0] + 100)
        ax.set_ylim(-100, self.canvas_size[1] + 100)
        ax.set_aspect('equal')
        waiting_now = sum(1 for c in self.cars.values() if c.waiting)
        ax.set_title(f"Network (t={self.t:.0f}s | active={len(self.cars)} | "
                     f"finished={len(self.finished_cars)} | waiting={waiting_now} | "
                     f"delay={self.cumulative_delay:.0f} veh·s)",
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2)
        return ax

    def animate(self, num_ticks=120, dt=1.0, interval_ms=120, save_path=None, dpi=100):
        fig, ax = plt.subplots(figsize=(13, 9))

        def update(frame):
            ax.clear()
            self.step(dt)
            self.draw_network(ax=ax)
            return []

        anim = animation.FuncAnimation(fig, update, frames=num_ticks,
                                       interval=interval_ms, blit=False, repeat=False)
        if save_path:
            ext = os.path.splitext(save_path)[1].lower()
            fps = max(1, int(1000 / interval_ms))
            try:
                writer = 'ffmpeg' if ext in ('.mp4', '.mov') else 'pillow'
                anim.save(save_path, writer=writer, fps=fps, dpi=dpi)
                print(f"  Saved animation to {save_path}")
            except Exception as e:
                print(f"  Could not save animation: {e}")
        plt.close(fig)
        return anim


class TrafficLightGA:
    def __init__(self, sim_factory: Callable[[List[TrafficLightPhase]], "BaseTrafficSimulator"],
                 directions: List[str], ga_config: Optional[GAConfig] = None,
                 valid_groups: Optional[List[Tuple[str, ...]]] = None,
                 seed: int = 0):
        self.sim_factory = sim_factory
        self.directions = directions
        self.cfg = ga_config or GAConfig()
        self.rng = random.Random(seed)
        self.history = {
            'best_fitness': [],
            'mean_fitness': [],
            'worst_fitness': [],
            'best_individual': [],
            'best_metrics': [],
        }

        if valid_groups is None:
            opp = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
            groups = set()
            for d in directions:
                groups.add((d,))
                if d in opp and opp[d] in directions:
                    groups.add(tuple(sorted((d, opp[d]))))
            valid_groups = sorted(groups)
        self.valid_groups = valid_groups

    def _random_individual(self) -> Individual:
        n = self.rng.randint(self.cfg.min_phases, self.cfg.max_phases)
        phases = []
        covered = set()
        for _ in range(n):
            grp = self.rng.choice(self.valid_groups)
            dur = self.rng.uniform(self.cfg.min_phase_duration, self.cfg.max_phase_duration)
            phases.append((grp, dur))
            covered.update(grp)
        for d in self.directions:
            if d not in covered:
                idx = self.rng.randrange(len(phases))
                phases[idx] = ((d,), self.rng.uniform(self.cfg.min_phase_duration,
                                                       self.cfg.max_phase_duration))
                covered.add(d)
        return Individual(phases=phases)

    def _repair(self, ind: Individual) -> Individual:
        covered = set()
        new_phases = []
        for grp, dur in ind.phases:
            dur = max(self.cfg.min_phase_duration,
                      min(self.cfg.max_phase_duration, dur))
            new_phases.append((grp, dur))
            covered.update(grp)
        for d in self.directions:
            if d not in covered:
                new_phases.append(((d,),
                                   self.rng.uniform(self.cfg.min_phase_duration,
                                                     self.cfg.max_phase_duration)))
        if len(new_phases) > self.cfg.max_phases:
            new_phases = new_phases[:self.cfg.max_phases]
        ind.phases = new_phases
        return ind

    def _mutate(self, ind: Individual) -> Individual:
        new_phases = list(ind.phases)
        for i, (grp, dur) in enumerate(new_phases):
            if self.rng.random() < self.cfg.mutation_rate:
                dur = dur + self.rng.gauss(0, self.cfg.mutation_sigma)
            if self.rng.random() < self.cfg.mutation_rate * 0.5:
                grp = self.rng.choice(self.valid_groups)
            new_phases[i] = (grp, dur)
        roll = self.rng.random()
        if roll < 0.1 and len(new_phases) < self.cfg.max_phases:
            new_phases.append((self.rng.choice(self.valid_groups),
                               self.rng.uniform(self.cfg.min_phase_duration,
                                                 self.cfg.max_phase_duration)))
        elif roll < 0.2 and len(new_phases) > self.cfg.min_phases:
            new_phases.pop(self.rng.randrange(len(new_phases)))
        elif roll < 0.3 and len(new_phases) >= 2:
            i, j = self.rng.sample(range(len(new_phases)), 2)
            new_phases[i], new_phases[j] = new_phases[j], new_phases[i]
        return self._repair(Individual(phases=new_phases))

    def _crossover(self, p1: Individual, p2: Individual) -> Individual:
        if self.rng.random() > self.cfg.crossover_rate:
            return self._repair(Individual(phases=list(p1.phases)))
        cut1 = self.rng.randint(1, max(1, len(p1.phases) - 1))
        cut2 = self.rng.randint(1, max(1, len(p2.phases) - 1))
        child = list(p1.phases[:cut1]) + list(p2.phases[cut2:])
        if not child:
            child = list(p1.phases)
        return self._repair(Individual(phases=child))

    def _tournament(self, pop: List[Individual]) -> Individual:
        contenders = self.rng.sample(pop, k=min(self.cfg.tournament_size, len(pop)))
        return min(contenders, key=lambda ind: ind.fitness)

    def _evaluate(self, ind: Individual) -> Individual:
        sim = self.sim_factory(ind.to_phase_list())
        sim.run(num_ticks=self.cfg.eval_ticks, dt=1.0)
        m = sim.metrics()
        ind.metrics = m

        if self.cfg.fitness == 'delay':
            ind.fitness = m['cumulative_delay_veh_seconds']
        elif self.cfg.fitness == 'avg_wait':
            ind.fitness = m['avg_wait_per_finished_car'] or 1e9
        elif self.cfg.fitness == 'wait_plus_throughput':
            tp = m['throughput_per_min']
            ind.fitness = m['cumulative_delay_veh_seconds'] - 50.0 * tp
        else:
            raise ValueError(f"Unknown fitness: {self.cfg.fitness}")
        return ind

    def run(self, verbose: bool = True) -> Individual:
        population = [self._random_individual() for _ in range(self.cfg.population_size)]
        for ind in population:
            self._evaluate(ind)

        n_elite = max(1, int(self.cfg.elite_fraction * self.cfg.population_size))
        t0 = _time.time()

        for gen in range(self.cfg.num_generations):
            population.sort(key=lambda ind: ind.fitness)
            best = population[0]
            mean_fit = float(np.mean([ind.fitness for ind in population]))
            worst_fit = population[-1].fitness

            self.history['best_fitness'].append(best.fitness)
            self.history['mean_fitness'].append(mean_fit)
            self.history['worst_fitness'].append(worst_fit)
            self.history['best_individual'].append(copy.deepcopy(best))
            self.history['best_metrics'].append(copy.deepcopy(best.metrics))

            if verbose:
                elapsed = _time.time() - t0
                print(f"Gen {gen+1:02d}/{self.cfg.num_generations} "
                      f"| best={best.fitness:.0f} | mean={mean_fit:.0f} "
                      f"| worst={worst_fit:.0f} | "
                      f"throughput={best.metrics['throughput_per_min']:.2f} cars/min "
                      f"| t={elapsed:.1f}s")

            next_pop = [copy.deepcopy(ind) for ind in population[:n_elite]]
            while len(next_pop) < self.cfg.population_size:
                p1 = self._tournament(population)
                p2 = self._tournament(population)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                self._evaluate(child)
                next_pop.append(child)
            population = next_pop

        population.sort(key=lambda ind: ind.fitness)
        return population[0], 

    def plot_evolution(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        gens = np.arange(1, len(self.history['best_fitness']) + 1)
        ax.plot(gens, self.history['best_fitness'], 'g-o',
                label='Best', linewidth=2, markersize=5)
        ax.plot(gens, self.history['mean_fitness'], 'b-',
                label='Population mean', alpha=0.7)
        ax.fill_between(gens, self.history['best_fitness'],
                        self.history['worst_fitness'], alpha=0.15, color='gray',
                        label='Best→worst spread')
        ax.set_xlabel('Generation')
        ax.set_ylabel(f'Fitness ({self.cfg.fitness})')
        ax.set_title('GA Evolution — Traffic Light Schedule', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        return ax

if __name__ == "__main__":
    print("Running single-intersection sanity check...")
    sim = SingleIntersectionSimulator(
        num_approaches=4,
        approach_config={
            'N': {'inflow_rate': 0.30},
            'S': {'inflow_rate': 0.30},
            'E': {'inflow_rate': 0.20},
            'W': {'inflow_rate': 0.20},
        },
        saturation_flow=0.5,
        seed=42,
    )
    sim.run(num_ticks=600, dt=1.0)
    sim.print_metrics(label="single intersection")
    sim.diagnose_road_loads(top_n=5)

    print("\nRunning network sanity check...")
    net = TrafficNetworkSimulator(seed=7)
    net.run(num_ticks=600, dt=1.0)
    net.print_metrics(label="network")
    net.diagnose_road_loads(top_n=5)