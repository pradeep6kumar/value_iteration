# Value Iteration for 4x4 GridWorld

This repository implements the Value Iteration algorithm for a 4x4 GridWorld problem where an agent starts at the top-left corner (state 0) and tries to reach the bottom-right corner (state 15). The agent can move up, down, left, or right with equal probability. The rewards are -1 for each move, and the terminal state (bottom-right) has a reward of 0.

## Problem Description

- 4x4 GridWorld environment
- Agent starts at the top-left corner (state 0)
- Goal is to reach the bottom-right corner (state 15)
- Actions: up, down, left, right (equal probability)
- Rewards: -1 for each move, 0 for reaching the terminal state
- No obstacles
- Discount factor (gamma) = 1.0

## Implementation

The repository contains three main files:
1. `value_iteration.py` - NumPy implementation of the Value Iteration algorithm
2. `value_iteration_pytorch.py` - PyTorch implementation of the Value Iteration algorithm
3. `value_iteration_gridworld.ipynb` - Jupyter notebook demonstrating both implementations

## Expected Output

After running the Value Iteration algorithm, the final value function should look like this:

```
[[-59.42367735 -57.42387125 -54.2813141  -51.71012579]
 [-57.42387125 -54.56699476 -49.71029394 -45.13926711]
 [-54.2813141  -49.71029394 -40.85391609 -29.99766609]
 [-51.71012579 -45.13926711 -29.99766609   0.        ]]
```

This value function shows that states closer to the goal have higher values (less negative), while states farther from the goal have lower values (more negative). The pattern reflects the structure of the problem: the values decrease as we move away from the goal state, with the lowest value at the top-left corner (the starting state).

## How to Run

1. To run the NumPy implementation:
   ```
   python value_iteration.py
   ```

2. To run the PyTorch implementation:
   ```
   python value_iteration_pytorch.py
   ```

3. To run the Jupyter notebook:
   ```
   jupyter notebook value_iteration_gridworld.ipynb
   ```

## Requirements

- NumPy
- PyTorch
- Matplotlib
- Jupyter (for running the notebook)

You can install the required packages using:
```
pip install numpy torch matplotlib jupyter
``` 