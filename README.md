# Emergence

A lightweight evolution sandbox built with Python, pygame, and numpy. Particles
represent different elemental types with unique bonding affinities and energy
needs. They compete for sparse energy sites, replicate when sufficiently
powered, and occasionally mutate to explore the rulespace.

## Features
- **Elemental diversity:** Three element archetypes (Spark, Grove, Stone) with
  unique colors, metabolism, and bonding preferences.
- **Energy scarcity:** Energy sites recharge slowly, forcing competition and
  cooperation via attachments.
- **Mutation and replication:** Particles divide when energized and may mutate
  their element type, altering future bonding behavior.
- **Spatial hashing:** Grid-based neighbor lookup keeps interactions smooth on
  modern Macs with hundreds of particles.

## Running the simulation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the interactive window:
   ```bash
   python evolution_sim.py
   ```

Controls:
- `SPACE`: pause/resume simulation
- `R`: reseed with a fresh population and energy landscape
- `Q`: quit

Tune the top-level constants in `evolution_sim.py` to experiment with the
energy abundance, mutation rate, and population caps.
