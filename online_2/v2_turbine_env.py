import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
from simulator_modules.Operators import HighVariability, LowVariability, NORMAL
from simulator_modules.Relations import LinkH2Metano, LinkMaxEnergyFuel, LinkGeneratedEnergy
from simulator_modules.TurbineSimulator import Graph, plot_nodes_history, plot_sum_history
import numpy as np
import argparse

register(
    id='turbine-env-v2',
    entry_point='v2_turbine_env:TurbineEnv', # module_name:class_name
)

def create_nodes_and_relations(prob=0.0001):
    # Create nodes
    H2 = HighVariability(lower_bound=30, upper_bound=70, typical_value=57.5, name="H2",
                         typical_bias=0.1, typical_bias_prob=0.1, theta=1.5, theta_prob=0.1,
                         exceed_prob=prob, exceed_duration_range=(180, 300),
                         return_duration_range=(180, 300), exceed_bias_range=(-5, 5),
                         exceed_bias_prob=0.05, exceed_peak_value_range=(0, 5),
                         hold_duration_range=(1500, 2000), hold_prob_vary=0.05, hold_variation=1)
    Metano = HighVariability(lower_bound=30, upper_bound=70, typical_value=40, name="Metano",
                             typical_bias=0.1, typical_bias_prob=0.1, theta=1.5, theta_prob=0.1)
    MaxEnergy = LowVariability(lower_bound=0, upper_bound=100, typical_value=20, name="MaxEnergy",
                           theta=0.7, steps_range=(100, 200))
    GeneratedEnergy = LowVariability(lower_bound=0, upper_bound=100, typical_value=40, name="GeneratedEnergy",
                                     theta=0.7, steps_range=(100,200))
    # Create relations
    relations = {
        "relation1": LinkH2Metano(limit_lower_bound=75, limit_upper_bound=100,
                                  typical_lower_bound=93, typical_upper_bound=98),
        "relation2": LinkMaxEnergyFuel(start_point=0, typical_bias_prob=0.01, typical_bias=1,
                                       theta_prob=0.01),
        "relation3": LinkGeneratedEnergy(typical_bias_prob=0, typical_bias=0.2,
                                         theta_prob=0.7, theta_bias=0.8),
    }

    return H2, Metano, MaxEnergy, GeneratedEnergy, relations

def create_graph(seed=None):
    H2, Metano, MaxEnergy, GeneratedEnergy, relations = create_nodes_and_relations()

    graph = Graph(random_seed=seed, debug=False, n_unstable_steps=480)
    node1 = graph.add_node(H2)
    node2 = graph.add_node(Metano)
    node3 = graph.add_node(MaxEnergy)
    node4 = graph.add_node(GeneratedEnergy)
    graph.add_edge(root=node1, child=node2, strategy=relations["relation1"])
    graph.add_edge(root=node1, child=node3, strategy=relations["relation2"], other_influences=[node2])
    graph.add_edge(root=node3, child=node4, strategy=relations["relation3"])

    return graph, [node1, node2, node3, node4]

class TurbineEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, seed=None, render_mode=None, history_length=20,
                 reward_1=-20, reward_2=-3,
                 reward_3=0.001, reward_4=-0.2,
                 lower_threshold=50, upper_threshold=70, steps=50000):
        self.render_mode = render_mode
        self.history_length = history_length
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.seed = seed

        # Reward parameters
        self.reward_1 = reward_1
        self.reward_2 = reward_2
        self.reward_3 = reward_3
        self.reward_4 = reward_4

        # Setup the graph simulator problem
        graph, nodes = create_graph(self.seed)
        self.graph = graph
        self.nodes = nodes
        self.n_steps = steps # Number of steps to simulate

        # Initialize history buffer for each node + thresholds
        self.history = np.zeros((len(self.nodes) + 2, self.history_length), dtype=np.float32)

        # Gym requires defining the action space. The action space is 0 or 1.
        # 0: do nothing, 1: perform action.
        # Training code can call action_space.sample() to randomly select an action.
        self.action_space = spaces.Discrete(2)
        self.last_action = None

        # Gym requires defining the observation space. The observation space consists of the current operators' values + thresholds.
        # Use a 1D vector: [node1.last_value, node2.last_value, ..., nodeN.last_value, lower_threshold, upper_threshold]
        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=(len(self.nodes) + 2, self.history_length),
            dtype=np.float32
        )

    # Gym required function (and parameters) to reset the environment
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Ensure deterministic randomization if seed is provided
        if seed is not None:
            self.seed = seed
            np.random.seed(self.seed)
        # Randomize thresholds at each reset
        self.lower_threshold = np.random.uniform(36, 80)
        self.upper_threshold = np.random.uniform(self.lower_threshold + 1, 99)

        # Reset the simulator. Optionally, pass in seed control randomness and reproduce scenarios.
        graph, nodes = create_graph(seed=self.seed)
        self.graph = graph
        self.nodes = nodes

        # Reset history buffer
        self.history = np.zeros((len(self.nodes) + 2, self.history_length), dtype=np.float32)

        # Simulate initial steps to populate history
        for _ in range(self.history_length):
            self.graph.simulate(steps=1)
            current_values = np.array([node.last_value for node in self.nodes], dtype=np.float32)
            self.history = np.roll(self.history, shift=-1, axis=1)
            self.history[:len(self.nodes), -1] = current_values
            self.history[len(self.nodes), -1] = self.lower_threshold
            self.history[len(self.nodes)+1, -1] = self.upper_threshold

        # Additional info to return. For debugging or whatever.
        info = {}

        # Render environment
        if(self.render_mode=='human'):
            self.render()

        # Return observation and info
        return self.history, info

    # Gym required function (and parameters) to perform an action
    def step(self, action):
        self.last_action = action
        other_information = {'alert': action == 1}

        # Simulate one step and update history
        self.graph.simulate(steps=1, other_informations=other_information)
        current_values = np.array([node.last_value for node in self.nodes], dtype=np.float32)
        self.history = np.roll(self.history, shift=-1, axis=1)
        self.history[:len(self.nodes), -1] = current_values
        self.history[len(self.nodes), -1] = self.lower_threshold
        self.history[len(self.nodes)+1, -1] = self.upper_threshold

        # Determine reward and termination
        generated_energy = self.history[3, -1]  # Last value of the GeneratedEnergy node
        state = self.nodes[0].op.state.get_type()
        # max_energy = self.history[2, -1]  # Last value of the MaxEnergy node

        if generated_energy > self.upper_threshold and self.last_action == 0:
            reward = self.reward_1
        elif generated_energy < self.lower_threshold and self.last_action == 1:
            reward = self.reward_3
        elif generated_energy < self.upper_threshold and self.last_action == 1:
            reward = self.reward_2
        else:
            reward = self.reward_4

        # Determine if the episode is terminated or truncated
        terminated = False
        truncated = self.n_steps <= self.graph.current_step

        # Additional info to return. For debugging or whatever.
        info = {
            # "last_alert": self.last_action,
            "current_step": self.graph.current_step,
            "last_action": action,
            "reward": reward
        }

        # Render environment
        if(self.render_mode == 'human'):
            self.render()

        # Return observation, reward, terminated, truncated (not used), info
        return self.history, reward, terminated, truncated, info

    # Gym required function to render environment
    def render(self):
        if(self.render_mode == 'human'):
            print("Step: ", self.graph.current_step)
            # print("Alert: ", self.graph.last_alert)
            print("Action: ", self.last_action)
            print("Values: ", {node.name: node.last_value for node in self.nodes})
            print("")
        else:
            raise NotImplementedError("Render mode not implemented.")

# For unit testing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turbine Environment Simulation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the environment")
    parser.add_argument("--simulate", action="store_true", help="Run the environment simulation")
    parser.add_argument("--render_mode", type=str, default="human", help="Render mode for the environment")
    parser.add_argument("--history_length", type=int, default=20, help="Length of the history buffer")
    parser.add_argument("--reward_1", type=float, default=-20, help="Reward for alert and no action")
    parser.add_argument("--reward_2", type=float, default=-3, help="Reward for alert and action")
    parser.add_argument("--reward_3", type=float, default=0.001, help="Reward for no alert and no action")
    parser.add_argument("--reward_4", type=float, default=-0.2, help="Reward for no alert and action")
    parser.add_argument("--lower_threshold", type=float, default=50, help="Lower threshold for the turbine's H2 level.")
    parser.add_argument("--upper_threshold", type=float, default=70, help="Upper threshold for the turbine's H2 level.")
    args = parser.parse_args()

    env = gym.make(
        'turbine-env-v2',
        seed=args.seed,
        render_mode=args.render_mode,
        history_length=args.history_length,
        reward_1=args.reward_1,
        reward_2=args.reward_2,
        reward_3=args.reward_3,
        reward_4=args.reward_4,
        lower_threshold=args.lower_threshold,
        upper_threshold=args.upper_threshold
    )
    print(env.observation_space)
    print(env.action_space)

    # Use this to check our custom environment
    print("Check environment begin")
    check_env(env.unwrapped)
    print("Check environment end")

    if args.simulate:
        print("\nSTARTING RANDOM RUN\n")
        # Reset environment
        env.reset()

        # Take some random actions
        truncated = False
        while not truncated:
            rand_action = env.action_space.sample()
            obs, _, _, truncated, _ = env.step(rand_action)