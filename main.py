import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class Trajectory:
    def __init__(self, states, actions, rewards):
        self.T = len(rewards) + 1

        assert len(actions) == len(rewards), "Must have as many rewards as actions"
        assert len(states) == len(rewards) + 1, "Must be one more state than action"
        self.states = np.array(states, dtype=np.int64)
        self.actions = np.array(actions, dtype=np.int64)
        self.rewards = np.array(rewards)

class Problem:
    def __init__(self, seed=42, W=5, min_reward=-5, max_reward=-1):
        #### TODO: Seed numpy's random number generator here using the seed provided
        # ?
        #### END

        # W is the width of the grid world problem
        self.W = W
        # Starting state is the middle tile (// performs a floor integer divide)
        self.starting_state = int(W // 2)

        #### TODO: Change the goal state to be randomly either 0 or (W-1)
        self.goal_state = 0
        #### END

        # Randomly initialise some weights
        self.tile_rewards = np.random.rand(W) * (max_reward - min_reward) + min_reward
        # Init an array to store the "theta" values
        self.policy_parameters = np.zeros(W)

    def run(self):
        x = self.starting_state
        states = []
        actions = []
        rewards = []
        while x != self.goal_state:
            states.append(x)
            #### TODO: Calculate probability of going right (a=+1), and replace below
            # p_right = ?
            #### END

            # Select a=+1 with probability 'p_right' and a=-1 with probability '1-p_right'
            a = int((np.random.rand() < 0.5) * 2 - 1)
            ####

            #### TODO: Remove the following code block once probabilities are properly calculated
            if (x==0 and a == -1):
                a = 1
            elif (x==self.W - 1 and a== +1):
                a = -1
            #### END

            # Step to the next state by applying environment function
            x = x + a

            # Calculate rewards
            r = self.tile_rewards[x]

            # Update the actions and rewards
            actions.append(a)
            rewards.append(r)
        
        # Append the terminal state to the list
        states.append(x)
        return Trajectory(states, actions, rewards)

    def reinforce_gradient_estimate(self, N):
        grads = np.zeros_like(self.policy_parameters)
        avg_return = 0
        
        for _ in range(N):
            trajectory = self.run()
            avg_return += np.sum(trajectory.rewards)
            #### TODO: Add in the gradient contribution here for the current trajectory
            # grads += ?
            #### END
        
        # Divide by N to find means
        grads = grads / N
        avg_return = avg_return / N
        
        return grads, avg_return

    def train(self, epochs: int, N: int, learning_rate: float, use_progress: bool = True):
        avg_returns = np.zeros(epochs)
        it = range(epochs) if not use_progress else tqdm(range(epochs))
        for e in it:
            grads, avg_return = self.reinforce_gradient_estimate(N)
            #### TODO: Update the parameters with the gradient descent rule with learning rate alpha
            # self.policy_parameters = ? 
            #### END

            avg_returns[e] = avg_return
        return avg_returns
    
    def calculate_max_return(self):
        if self.goal_state == 0:
            return np.sum(self.tile_rewards[:self.starting_state])
        else:
            return np.sum(self.tile_rewards[self.starting_state+1:])


if __name__ == "__main__":
    problem = Problem()

    epochs = 5000
    N = 32
    alpha = 5e-4
    returns = problem.train(epochs, N, alpha)
    max_return = problem.calculate_max_return()
    
    filter_width = 5
    smoothed_returns = np.convolve(returns, np.ones(filter_width)/filter_width, mode='same')

    plt.figure()
    plt.plot(range(1+filter_width,len(returns)+1-filter_width), smoothed_returns[filter_width:-filter_width], "-", label="Sampled Mean Return", alpha=0.7)
    plt.plot([1, len(returns)+1], [max_return, max_return], '-.', label="Max Return")
    plt.legend(loc="lower right")
    plt.xlabel("Epochs")
    plt.ylabel("Avg Return")
    plt.show()