"""Interactive particle-based evolution simulation.

Run with `python evolution_sim.py`. The simulation uses pygame for
real-time rendering and numpy for numerically efficient updates. Particles
represent elemental types with distinct bonding affinities and energy needs.
"""
from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import pygame

# Window configuration
WIDTH, HEIGHT = 1280, 720
BACKGROUND = (8, 10, 24)
FPS = 60

# Simulation parameters
MAX_PARTICLES = 2000
INITIAL_PARTICLES = 360
ENERGY_SITE_COUNT = 26
ENERGY_RESPAWN_TIME = 5.2
ENERGY_RADIUS = 12
ENERGY_PER_SITE = 320.0
ENERGY_DECAY = 2.0
MUTATION_RATE = 0.1
REPLICATION_THRESHOLD = 140.0
BOND_DISTANCE = 28.0
BOND_STRENGTH = 28.0
COLLISION_DISTANCE = 8.0
WORLD_FRICTION = 0.93
THERMAL_NOISE = 18.0
MAX_SPEED = 120.0
GRID_SIZE = 48

# Cluster dynamics
CLUSTER_SIZE_BONUS = 5
CLUSTER_METABOLISM_MULT = 0.65
ENERGY_SHARE_RATE = 0.35
CLUSTER_BUD_THRESHOLD = 220.0
CLUSTER_BUD_COST = 120.0
CLUSTER_MAX_SIZE = 24
WANDER_SPEED = 18.0


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class Element:
    name: str
    color: Tuple[int, int, int]
    base_energy: float
    metabolism: float
    size: float
    affinities: Dict[int, float]


# Define a small palette of distinct elements.
ELEMENTS: List[Element] = [
    Element(
        name="Spark",
        color=(255, 149, 128),
        base_energy=85.0,
        metabolism=0.9,
        size=6,
        affinities={0: 0.5, 1: 1.0, 2: 0.35},
    ),
    Element(
        name="Grove",
        color=(120, 255, 171),
        base_energy=95.0,
        metabolism=1.1,
        size=7,
        affinities={0: 0.65, 1: 0.4, 2: 1.1},
    ),
    Element(
        name="Stone",
        color=(140, 177, 255),
        base_energy=110.0,
        metabolism=1.3,
        size=8,
        affinities={0: 0.2, 1: 0.4, 2: 0.5},
    ),
]


class EnergySite:
    def __init__(self, x: float, y: float) -> None:
        self.pos = np.array([x, y], dtype=np.float32)
        self.charge = ENERGY_PER_SITE
        self.cooldown = 0.0
        angle = random.random() * math.tau
        self.velocity = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32) * WANDER_SPEED

    def available(self) -> bool:
        return self.charge > 0

    def update(self, dt: float) -> None:
        if self.charge <= 0:
            self.cooldown -= dt
            if self.cooldown <= 0:
                self.charge = ENERGY_PER_SITE
                self.cooldown = ENERGY_RESPAWN_TIME
        # Gentle wandering keeps the landscape dynamic.
        jitter = (np.random.rand(2).astype(np.float32) - 0.5) * 2.0
        self.velocity += jitter
        speed = float(np.linalg.norm(self.velocity))
        if speed > WANDER_SPEED:
            self.velocity *= WANDER_SPEED / speed
        self.pos += self.velocity * dt
        self.pos[0] = clamp(self.pos[0], ENERGY_RADIUS, WIDTH - ENERGY_RADIUS)
        self.pos[1] = clamp(self.pos[1], ENERGY_RADIUS, HEIGHT - ENERGY_RADIUS)

    def draw(self, surface: pygame.Surface) -> None:
        ratio = clamp(self.charge / ENERGY_PER_SITE, 0.0, 1.0)
        inner = int(ENERGY_RADIUS * ratio)
        pygame.draw.circle(surface, (255, 234, 153), self.pos.astype(int), ENERGY_RADIUS)
        pygame.draw.circle(surface, (255, 197, 92), self.pos.astype(int), inner)


class ParticleWorld:
    def __init__(self) -> None:
        self.positions = np.zeros((0, 2), dtype=np.float32)
        self.velocities = np.zeros((0, 2), dtype=np.float32)
        self.energy = np.zeros(0, dtype=np.float32)
        self.types: List[int] = []
        self.bonds: List[Set[int]] = []
        self.energy_sites: List[EnergySite] = []
        self.paused = False

    def seed(self) -> None:
        self.positions = np.random.rand(INITIAL_PARTICLES, 2).astype(np.float32)
        self.positions[:, 0] *= WIDTH
        self.positions[:, 1] *= HEIGHT
        self.velocities = (np.random.rand(INITIAL_PARTICLES, 2).astype(np.float32) - 0.5) * 32
        self.types = [random.randrange(len(ELEMENTS)) for _ in range(INITIAL_PARTICLES)]
        self.energy = np.array([ELEMENTS[t].base_energy for t in self.types], dtype=np.float32)
        self.bonds = [set() for _ in range(INITIAL_PARTICLES)]

        self.energy_sites = [
            EnergySite(
                random.uniform(ENERGY_RADIUS, WIDTH - ENERGY_RADIUS),
                random.uniform(ENERGY_RADIUS, HEIGHT - ENERGY_RADIUS),
            )
            for _ in range(ENERGY_SITE_COUNT)
        ]
        for site in self.energy_sites:
            site.cooldown = random.uniform(0, ENERGY_RESPAWN_TIME)

    def world_grid(self) -> Dict[Tuple[int, int], List[int]]:
        grid: Dict[Tuple[int, int], List[int]] = {}
        coords = (self.positions / GRID_SIZE).astype(int)
        for idx, (gx, gy) in enumerate(coords):
            grid.setdefault((gx, gy), []).append(idx)
        return grid

    def neighbors(self, index: int, grid: Dict[Tuple[int, int], List[int]]) -> List[int]:
        x, y = self.positions[index]
        gx, gy = int(x / GRID_SIZE), int(y / GRID_SIZE)
        nearby: List[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell = (gx + dx, gy + dy)
                if cell in grid:
                    nearby.extend(grid[cell])
        return nearby

    def clusters(self) -> List[List[int]]:
        """Return connected components of the bond graph."""
        seen: Set[int] = set()
        comps: List[List[int]] = []
        for idx in range(len(self.types)):
            if idx in seen:
                continue
            stack = [idx]
            comp: List[int] = []
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                comp.append(cur)
                stack.extend(self.bonds[cur])
            comps.append(comp)
        return comps

    def share_energy(self, clusters: List[List[int]], dt: float) -> None:
        for comp in clusters:
            if len(comp) < 2:
                continue
            avg = float(np.mean(self.energy[comp]))
            delta = (avg - self.energy[comp]) * (ENERGY_SHARE_RATE * dt)
            self.energy[comp] += delta

    def cluster_bud(self, comp: List[int]) -> None:
        if len(self.types) >= MAX_PARTICLES or len(comp) >= CLUSTER_MAX_SIZE:
            return
        centroid = np.mean(self.positions[comp], axis=0)
        parent = random.choice(comp)
        child_type = self.types[parent]
        if random.random() < MUTATION_RATE:
            child_type = random.randrange(len(ELEMENTS))
        angle = random.random() * math.tau
        offset = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32) * (BOND_DISTANCE * 0.8)
        child_pos = centroid + offset
        child_vel = np.random.randn(2).astype(np.float32) * 4
        self.positions = np.vstack([self.positions, child_pos])
        self.velocities = np.vstack([self.velocities, child_vel])
        self.energy = np.append(self.energy, CLUSTER_BUD_COST * 0.4)
        self.types.append(child_type)
        self.bonds.append(set())
        new_idx = len(self.types) - 1
        self.bonds[new_idx].update({parent})
        self.bonds[parent].add(new_idx)

    def try_bond(self, a: int, b: int) -> None:
        if a == b or b in self.bonds[a] or len(self.bonds[a]) > 6:
            return
        da = self.positions[b] - self.positions[a]
        dist = float(np.linalg.norm(da))
        if dist > BOND_DISTANCE or dist <= 1e-3:
            return
        ta, tb = self.types[a], self.types[b]
        affinity = ELEMENTS[ta].affinities.get(tb, 0.0)
        if random.random() < affinity:
            self.bonds[a].add(b)
            self.bonds[b].add(a)

    def replicate(self, index: int) -> None:
        if len(self.types) >= MAX_PARTICLES:
            return
        child_pos = self.positions[index] + np.random.randn(2).astype(np.float32) * 6
        child_vel = self.velocities[index] * -0.4
        child_type = self.types[index]
        if random.random() < MUTATION_RATE:
            child_type = random.randrange(len(ELEMENTS))
        self.positions = np.vstack([self.positions, child_pos])
        self.velocities = np.vstack([self.velocities, child_vel])
        self.energy = np.append(self.energy, ELEMENTS[child_type].base_energy * 0.6)
        self.types.append(child_type)
        self.bonds.append({index})
        self.bonds[index].add(len(self.types) - 1)

    def update(self, dt: float) -> None:
        if self.paused:
            return
        if len(self.types) == 0:
            return

        clusters = self.clusters()
        # Energy drain from metabolism and collisions. Large bonded structures get a metabolism discount.
        metabolism = np.array([ELEMENTS[t].metabolism for t in self.types], dtype=np.float32)
        for comp in clusters:
            if len(comp) >= CLUSTER_SIZE_BONUS:
                metabolism[comp] *= CLUSTER_METABOLISM_MULT
        self.energy -= metabolism * dt * 3.6
        self.energy -= WORLD_FRICTION * dt

        # Spatial hashing for local interactions.
        grid = self.world_grid()
        for i in range(len(self.types)):
            if self.energy[i] <= 0:
                continue
            neigh = self.neighbors(i, grid)
            for j in neigh:
                if i >= j:
                    continue
                delta = self.positions[j] - self.positions[i]
                dist = float(np.linalg.norm(delta))
                if dist < 1e-4:
                    continue
                if dist < COLLISION_DISTANCE:
                    push = (delta / dist) * (COLLISION_DISTANCE - dist)
                    self.velocities[i] -= push * 0.5
                    self.velocities[j] += push * 0.5
                if dist < BOND_DISTANCE:
                    self.try_bond(i, j)

        # Bond springs pull particles together.
        for i, partners in enumerate(self.bonds):
            if not partners:
                continue
            for j in partners:
                delta = self.positions[j] - self.positions[i]
                dist = float(np.linalg.norm(delta))
                if dist <= 1e-4:
                    continue
                force = (delta / dist) * (dist - BOND_DISTANCE) * 0.38
                self.velocities[i] += force

        # Energy sites consumption and wandering for dynamic niches.
        for site in self.energy_sites:
            site.update(dt)
        for i in range(len(self.types)):
            if self.energy[i] <= 0:
                continue
            for site in self.energy_sites:
                if not site.available():
                    continue
                offset = self.positions[i] - site.pos
                dist_sq = float(np.dot(offset, offset))
                if dist_sq < (ENERGY_RADIUS + 6) ** 2:
                    gained = min(32.0 * dt, site.charge)
                    site.charge -= gained
                    self.energy[i] += gained

        # Share energy across bonded clusters to keep macro-structures alive.
        self.share_energy(clusters, dt)

        # Budding reproduction favors larger assemblies that hoard energy.
        for comp in clusters:
            if len(comp) < CLUSTER_SIZE_BONUS:
                continue
            avg_energy = float(np.mean(self.energy[comp]))
            if avg_energy > CLUSTER_BUD_THRESHOLD:
                cost = CLUSTER_BUD_COST / len(comp)
                self.energy[comp] -= cost
                self.cluster_bud(comp)

        # Integrate positions.
        self.velocities += np.random.randn(len(self.types), 2).astype(np.float32) * (THERMAL_NOISE * dt)
        speed = np.linalg.norm(self.velocities, axis=1)
        mask = speed > MAX_SPEED
        if np.any(mask):
            self.velocities[mask] *= (MAX_SPEED / speed[mask])[:, None]
        self.positions += self.velocities * dt
        self.velocities *= WORLD_FRICTION

        # Wrap around the world.
        self.positions[:, 0] = np.mod(self.positions[:, 0], WIDTH)
        self.positions[:, 1] = np.mod(self.positions[:, 1], HEIGHT)

        # Replication and culling.
        for i in range(len(self.types)):
            if self.energy[i] > REPLICATION_THRESHOLD and len(self.bonds[i]) >= 1:
                self.energy[i] *= 0.55
                self.replicate(i)
        self.energy = np.maximum(self.energy, 0.0)
        alive_mask = self.energy > 0
        if not np.all(alive_mask):
            self._prune(~alive_mask)

    def _prune(self, dead_mask: np.ndarray) -> None:
        keep_indices = np.where(~dead_mask)[0]
        self.positions = self.positions[keep_indices]
        self.velocities = self.velocities[keep_indices]
        self.energy = self.energy[keep_indices]
        self.types = [self.types[i] for i in keep_indices]
        mapping = {old: new for new, old in enumerate(keep_indices)}
        new_bonds: List[Set[int]] = []
        for old_idx in keep_indices:
            remapped = {mapping[p] for p in self.bonds[old_idx] if p in mapping}
            new_bonds.append(remapped)
        self.bonds = new_bonds

    def draw(self, surface: pygame.Surface) -> None:
        for site in self.energy_sites:
            site.draw(surface)
        for pos, t in zip(self.positions, self.types):
            element = ELEMENTS[t]
            pygame.draw.circle(surface, element.color, pos.astype(int), element.size)


class Overlay:
    def __init__(self, font: pygame.font.Font) -> None:
        self.font = font

    def draw(self, surface: pygame.Surface, world: ParticleWorld) -> None:
        lines = [
            "Evolution Playground",
            f"Particles: {len(world.types)}/{MAX_PARTICLES}",
            "Press SPACE to pause/resume, R to reseed, Q to quit",
            "Elements: "
            + ", ".join(f"{idx}:{elm.name}" for idx, elm in enumerate(ELEMENTS)),
        ]
        y = 10
        for text in lines:
            surf = self.font.render(text, True, (230, 230, 230))
            surface.blit(surf, (10, y))
            y += 22


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Evolution Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Menlo", 18)

    world = ParticleWorld()
    world.seed()
    overlay = Overlay(font)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    world.paused = not world.paused
                elif event.key == pygame.K_r:
                    world.seed()
                elif event.key == pygame.K_q:
                    running = False

        world.update(dt)

        screen.fill(BACKGROUND)
        world.draw(screen)
        overlay.draw(screen, world)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit(0)
