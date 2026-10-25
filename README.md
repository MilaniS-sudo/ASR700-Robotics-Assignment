# Component B — Multi-Agent Robotic Swarm with Learning-Based Decision-Making

Four e-puck robots cooperatively search a 2 m × 2 m Webots arena for a hidden
signal source (a stand-in for a gas leak).

| Notebook | Question | Contents |
|---|---|---|
| `question1/swarm_setup.ipynb` | B1 (10) | Task, roles, ring and star topologies, world and controller design, communication load |
| `question2/consensus_pso.ipynb` | B2 (10) | Average consensus with theoretical convergence rate, formation, PSO search, emergent behaviour, convergence evaluation |
| `question3/learning_and_decisions.ipynb` | B3 (10) | MLP trained by imitation of the navigation rule, leave-one-seed-out validation, confusion matrix, in-simulation test, centralised vs distributed comparison with fault injection |
| `question4/ethics_risk.ipynb` | B4 (20) | Bias and reliability, explainability (permutation importance), safety risks, fault tolerance, South African context, references |

`swarm_data.py` holds the shared loading and metric helpers.

## Running order
1. Open `webots/worlds/component_b_swarm.wbt` and press Play. This runs 12 experiments (4 types × 3 seeds), about 21 min of simulated time.
2. Run `question3/learning_and_decisions.ipynb`. It writes `webots/data/swarm/nn_policy.json`.
3. Press Play again in Webots. It now runs the 3 neural-network runs.
4. Run all four notebooks from their own folders.

Dependencies: numpy, pandas, matplotlib, scikit-learn.
