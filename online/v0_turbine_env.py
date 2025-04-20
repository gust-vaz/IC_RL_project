import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
from simulator_modules.Operators import HighVariability, LowVariability
from simulator_modules.Relations import LinkH2Metano, LinkSimilarBehavior, LinkLongReturn
from simulator_modules.TurbineSimulator import Graph, plot_nodes_history, plot_sum_history
import numpy as np

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

    def __init__(self, seed=None, render_mode=None):
        self.render_mode = render_mode

        # Setup the graph simulator problem
        graph, nodes = create_graph()
        self.graph = graph
        self.nodes = nodes
        self.n_steps = 100000 # Number of steps to simulate

        # Gym requires defining the action space. The action space is 0 or 1.
        # 0: do nothing, 1: perform action.
        # Training code can call action_space.sample() to randomly select an action. 
        self.action_space = spaces.Discrete(2)

        # TODO make it works with a history of n steps
        # Gym requires defining the observation space. The observation space consists of the current operators' values.
        # The observation space is used to validate the observation returned by reset() and step().
        # Use a 1D vector: [node1.last_value, node2.last_value, ..., node12.last_value]
        self.observation_space = spaces.Box(
            low=0,
            high=np.array([100,100,100,100,100,100,100,100,100,100,100,100]),
            shape=(12,),
            dtype=np.float32
        )

    # Gym required function (and parameters) to reset the environment
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # gym requires this call to control randomness and reproduce scenarios.

        # Reset the simulator. Optionally, pass in seed control randomness and reproduce scenarios.
        graph, nodes = create_graph(seed=seed)
        self.graph = graph
        self.nodes = nodes

        # Construct the observation state:
        # [node1.last_value, node2.last_value, ..., node12.last_value]
        self.graph.simulate(steps=1) # Simulate one step to get the initial values of the nodes.
        obs = np.array([node.last_value for node in self.nodes], dtype=np.float32)
        
        # Additional info to return. For debugging or whatever.
        info = {}

        # Render environment
        if(self.render_mode=='human'):
            self.render()

        # Return observation and info
        return obs, info

    # Gym required function (and parameters) to perform an action
    def step(self, action):
        # Determine reward and termination
        last_alert = self.graph.last_alert
        if last_alert == True and action == 0:
            reward = -1
        elif last_alert == True and action == 1:
            reward = 1
        elif last_alert == False and action == 0:
            reward = 0
        elif last_alert == False and action == 1:
            reward = -0.2

        # Construct the observation state: 
        # [node1.last_value, node2.last_value, ..., node12.last_value]
        self.graph.simulate(steps=1) # Simulate one step to get the new values of the nodes.
        obs = np.array([node.last_value for node in self.nodes], dtype=np.float32)

        # Determine if the episode is terminated or truncated
        terminated = False
        truncated = self.n_steps <= self.graph.current_step

        # Additional info to return. For debugging or whatever.
        info = {}

        # Render environment
        if(self.render_mode == 'human'):
            self.render()

        # Return observation, reward, terminated, truncated (not used), info
        return obs, reward, terminated, truncated, info

    # Gym required function to render environment
    def render(self):
        if(self.render_mode == 'human'):
            print("Step: ", self.graph.current_step)
            print("Alert: ", self.graph.last_alert)
            print("Values: ", {node.name: node.last_value for node in self.nodes})
            print("")
        else:
            raise NotImplementedError("Render mode not implemented.")

# For unit testing
if __name__=="__main__":
    env = gym.make('turbine-env-v0', render_mode='human')
    print(env.observation_space)
    print(env.action_space)
    print(env.reward_space)

    # Use this to check our custom environment
    print("Check environment begin")
    check_env(env.unwrapped)
    print("Check environment end")

    print("\STARTING RANDOM RUN\n")
    # Reset environment
    env.reset()

    # Take some random actions
    truncated = False
    while(not truncated):
        rand_action = env.action_space.sample()
        obs, _, _, truncated, _ = env.step(rand_action)