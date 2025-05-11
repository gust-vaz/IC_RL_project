import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
from simulator_modules.Operators import HighVariability, LowVariability
from simulator_modules.Relations import LinkH2Metano, LinkSimilarBehavior, LinkLongReturn
from simulator_modules.TurbineSimulator import Graph, plot_nodes_history, plot_sum_history
import numpy as np
import argparse

register(
    id='turbine-env-v0',
    entry_point='v0_turbine_env:TurbineEnv', # module_name:class_name
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
    Enxofre = LowVariability(lower_bound=0, upper_bound=8, typical_value=0.002, name="Enxofre",
                             theta=0.002, steps_range=(480, 700))
    Etileno = LowVariability(lower_bound=0.05, upper_bound=0.55, typical_value=0.2, name="Etileno",
                             theta=0.05, steps_range=(480, 700))
    NButano = LowVariability(lower_bound=0.01, upper_bound=0.45, typical_value=0.15, name="N-Butano",
                             theta=0.03, steps_range=(480, 700))
    Etano = LowVariability(lower_bound=0.06, upper_bound=1.8, typical_value=0.43, name="Etano",
                           theta=0.7, steps_range=(240, 480))
    Propano = LowVariability(lower_bound=0, upper_bound=0.69, typical_value=0.2, name="Propano",
                             theta=0.1, steps_range=(240, 480))
    C5 = LowVariability(lower_bound=0, upper_bound=0.3, typical_value=0.021, name="C5+",
                        theta=0.01, steps_range=(240, 480))
    CO2 = LowVariability(lower_bound=0, upper_bound=0.3, typical_value=0.2, name="CO2",
                         theta=0.1, steps_range=(240, 480))
    Propileno = LowVariability(lower_bound=0.06, upper_bound=0.3, typical_value=0.21, name="Propileno",
                               theta=0.02, steps_range=(480, 700))
    CO = LowVariability(lower_bound=0, upper_bound=0.8, typical_value=0.6, name="CO",
                        theta=0.1, steps_range=(480, 700))
    vazao = LowVariability(lower_bound=3, upper_bound=10, typical_value=7, name="Vazão",
                           theta=1, steps_range=(5, 10))

    # Create relations
    relations = {
        "relation1": LinkH2Metano(limit_lower_bound=75, limit_upper_bound=100,
                                  typical_lower_bound=93, typical_upper_bound=98),
        "relation2": LinkSimilarBehavior(correlation=0.3, typical_bias_prob=0.1,
                                         typical_bias=0.6, theta_prob=0.5),
        "relation3": LinkSimilarBehavior(correlation=0.6, typical_bias_prob=0.1,
                                         typical_bias=0.01, theta_prob=0.5, amplifier=1.8),
        "relation4": LinkSimilarBehavior(correlation=0.7, typical_bias_prob=0.4,
                                         typical_bias=0.4, theta_prob=0.5),
        "relation5": LinkSimilarBehavior(correlation=0.85, typical_bias_prob=0.1,
                                         typical_bias=0.4, theta_prob=0.7, amplifier=14,
                                         holding_range=(50, 60)),
        "relation6": LinkSimilarBehavior(correlation=0.94, typical_bias_prob=0.01,
                                         typical_bias=0.8, theta_prob=0.7, amplifier=14,
                                         holding_range=(50, 60)),
        "relation7": LinkSimilarBehavior(correlation=0.89, typical_bias=0.01,
                                         typical_bias_prob=0.1, theta_prob=0.7, amplifier=10,
                                         holding_range=(180, 300)),
        "relation8": LinkSimilarBehavior(correlation=0.92, typical_bias=0.1,
                                         typical_bias_prob=0.1, theta_prob=0.7, amplifier=10,
                                         holding_range=(180, 230)),
        "relation9": LinkLongReturn(correlation=0.5, typical_bias_prob=0.1,
                                    typical_bias=0.1, theta_prob=0.5, amplifier=1.8,
                                    holding_range=(480, 700), back_range=(6000, 7200),
                                    back_typical_prob=0.7, back_typical_range=(-0.02, 0.02)),
        "relation10": LinkLongReturn(correlation=0.5, typical_bias_prob=0.1,
                                     typical_bias=0.1, theta_prob=0.5, amplifier=1.8,
                                     holding_range=(480, 700), back_range=(6000, 7200),
                                     back_typical_prob=0.7, back_typical_range=(-0.02, 0.02)),
        "relation11": LinkSimilarBehavior(correlation=0.92, typical_bias=0.1,
                                          typical_bias_prob=0.5, theta_prob=1, amplifier=1.4,
                                          holding_range=(20, 30))
    }

    return H2, Metano, Enxofre, Etileno, NButano, Etano, Propano, C5, CO2, Propileno, CO, vazao, relations

def create_graph(seed=None):
    H2, Metano, Enxofre, Etileno, NButano, Etano, Propano, C5, CO2, Propileno, CO, vazao, relations = create_nodes_and_relations()

    graph = Graph(random_seed=seed, debug=False, n_unstable_steps=480)
    node1 = graph.add_node(H2)
    node2 = graph.add_node(Metano)
    graph.add_edge(root=node1, child=node2, strategy=relations["relation1"])
    node3 = graph.add_node(Enxofre)
    graph.add_edge(root=node1, child=node3, strategy=relations["relation2"])
    node4 = graph.add_node(Etileno)
    graph.add_edge(root=node1, child=node4, strategy=relations["relation3"])
    node5 = graph.add_node(NButano)
    graph.add_edge(root=node1, child=node5, strategy=relations["relation4"])
    node6 = graph.add_node(Etano)
    graph.add_edge(root=node2, child=node6, strategy=relations["relation5"])
    node7 = graph.add_node(Propano)
    graph.add_edge(root=node2, child=node7, strategy=relations["relation6"])
    node8 = graph.add_node(C5)
    graph.add_edge(root=node2, child=node8, strategy=relations["relation7"])
    node9 = graph.add_node(CO2)
    graph.add_edge(root=node2, child=node9, strategy=relations["relation8"])
    node10 = graph.add_node(Propileno)
    graph.add_edge(root=node1, child=node10, strategy=relations["relation9"])
    node11 = graph.add_node(CO)
    graph.add_edge(root=node1, child=node11, strategy=relations["relation10"])
    node12 = graph.add_node(vazao)
    graph.add_edge(root=node1, child=node12, strategy=relations["relation11"])

    return graph, [node1, node2, node3, node4, node5, node6, node7, node8, node9, node10, node11, node12]


class TurbineEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, seed=None, render_mode=None, history_length=20,
                 reward_alert_no_action=-2, reward_alert_action=1, 
                 reward_no_alert_no_action=0, reward_no_alert_action=-0.2):
        self.render_mode = render_mode
        self.history_length = history_length
        self.seed = seed

        # Reward parameters
        self.reward_alert_no_action = reward_alert_no_action
        self.reward_alert_action = reward_alert_action
        self.reward_no_alert_no_action = reward_no_alert_no_action
        self.reward_no_alert_action = reward_no_alert_action

        # Setup the graph simulator problem
        graph, nodes = create_graph(self.seed)
        self.graph = graph
        self.nodes = nodes
        self.n_steps = 100000 # Number of steps to simulate

        # Initialize history buffer for each node
        self.history = np.zeros((len(self.nodes), self.history_length), dtype=np.float32)

        # Gym requires defining the action space. The action space is 0 or 1.
        # 0: do nothing, 1: perform action.
        # Training code can call action_space.sample() to randomly select an action. 
        self.action_space = spaces.Discrete(2)
        self.last_action = None

        # Gym requires defining the observation space. The observation space consists of the current operators' values.
        # The observation space is used to validate the observation returned by reset() and step().
        # Use a 1D vector: [node1.last_value, node2.last_value, ..., node12.last_value]
        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=(len(self.nodes), self.history_length),
            dtype=np.float32
        )

    # Gym required function (and parameters) to reset the environment
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the simulator. Optionally, pass in seed control randomness and reproduce scenarios.
        self.seed += 1
        graph, nodes = create_graph(seed=self.seed)
        self.graph = graph
        self.nodes = nodes

        # Reset history buffer
        self.history = np.zeros((len(self.nodes), self.history_length), dtype=np.float32)

        # Simulate initial steps to populate history
        for _ in range(self.history_length):
            self.graph.simulate(steps=1)
            current_values = np.array([node.last_value for node in self.nodes], dtype=np.float32)
            self.history = np.roll(self.history, shift=-1, axis=1)
            self.history[:, -1] = current_values

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

        # Determine reward and termination
        last_alert = self.graph.last_alert
        if last_alert and action == 0:
            reward = self.reward_alert_no_action
        elif last_alert and action == 1:
            reward = self.reward_alert_action
        elif not last_alert and action == 0:
            reward = self.reward_no_alert_no_action
        elif not last_alert and action == 1:
            reward = self.reward_no_alert_action

        # Simulate one step and update history
        self.graph.simulate(steps=1)
        current_values = np.array([node.last_value for node in self.nodes], dtype=np.float32)
        self.history = np.roll(self.history, shift=-1, axis=1)
        self.history[:, -1] = current_values

        # Determine if the episode is terminated or truncated
        terminated = False
        truncated = self.n_steps <= self.graph.current_step

        # Additional info to return. For debugging or whatever.
        info = {
            "last_alert": last_alert,
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
            print("Alert: ", self.graph.last_alert)
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
    parser.add_argument("--reward_alert_no_action", type=float, default=-2, help="Reward for alert and no action")
    parser.add_argument("--reward_alert_action", type=float, default=1, help="Reward for alert and action")
    parser.add_argument("--reward_no_alert_no_action", type=float, default=0, help="Reward for no alert and no action")
    parser.add_argument("--reward_no_alert_action", type=float, default=-0.2, help="Reward for no alert and action")
    args = parser.parse_args()

    env = gym.make(
        'turbine-env-v0',
        seed=args.seed,
        render_mode=args.render_mode,
        history_length=args.history_length,
        reward_alert_no_action=args.reward_alert_no_action,
        reward_alert_action=args.reward_alert_action,
        reward_no_alert_no_action=args.reward_no_alert_no_action,
        reward_no_alert_action=args.reward_no_alert_action
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