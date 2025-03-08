import numpy as np
import matplotlib.pyplot as plt

def value_iteration_gridworld():
    # Initialize parameters
    grid_size = 4  # 4x4 grid
    num_states = grid_size * grid_size  # 16 states
    num_actions = 4  # up, down, left, right
    gamma = 1.0  # discount factor
    theta = 1e-4  # convergence threshold
    
    # Initialize value function
    V = np.zeros((grid_size, grid_size))
    
    # Define rewards: -1 for all states except terminal state (bottom-right)
    rewards = np.full((grid_size, grid_size), -1)
    rewards[grid_size-1, grid_size-1] = 0  # terminal state has 0 reward
    
    # Define actions: up, down, left, right
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Value iteration
    iteration = 0
    while True:
        iteration += 1
        delta = 0  # track maximum change
        V_new = V.copy()
        
        # Update each state
        for i in range(grid_size):
            for j in range(grid_size):
                # Skip terminal state
                if i == grid_size-1 and j == grid_size-1:
                    continue
                
                # Calculate value for each action
                action_values = []
                for action in actions:
                    # Calculate next state
                    next_i = max(0, min(grid_size-1, i + action[0]))
                    next_j = max(0, min(grid_size-1, j + action[1]))
                    
                    # Calculate value
                    action_value = rewards[i, j] + gamma * V[next_i, next_j]
                    action_values.append(action_value)
                
                # Update value function with average (equal probability for all actions)
                V_new[i, j] = sum(action_values) / num_actions
                
                # Track maximum change
                delta = max(delta, abs(V_new[i, j] - V[i, j]))
        
        # Update value function
        V = V_new
        
        # Check for convergence
        if delta < theta:
            break
    
    print(f"Value iteration converged after {iteration} iterations")
    print("Final value function:")
    print(V)
    
    return V, iteration

def plot_value_function(V):
    """Plot the value function as a heatmap"""
    plt.figure(figsize=(8, 6))
    plt.imshow(V, cmap='viridis')
    plt.colorbar(label='Value')
    plt.title('Value Function for 4x4 GridWorld')
    
    # Add text annotations
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            plt.text(j, i, f'{V[i, j]:.2f}', 
                     ha='center', va='center', 
                     color='white' if V[i, j] < -30 else 'black')
    
    plt.tight_layout()
    plt.savefig('value_function.png')
    plt.show()

if __name__ == "__main__":
    V, iterations = value_iteration_gridworld()
    plot_value_function(V) 