from abc import ABC, abstractmethod
import random
from math import cos, sin
import matplotlib.pyplot as plt
import numpy as np

# Constants for state machine
BASELINE  = 0   # Normal
DEVIATION = 1   # Exceeding
PLATEAU   = 2   # Holding
RECOVERY  = 3   # Returning

class State(ABC):
	"""
	Abstract base class for defining states in the operator's state machine.
	"""
     
	@abstractmethod
	def next_step(self, context: 'HighVariability') -> float:
		"""
        Defines the behavior for the next step in the current state.

        Parameters:
            context (HighVariability): The operator context.

        Returns:
            float: The calculated value for the next step.
        """
		pass

	@abstractmethod
	def get_type(self) -> int:
		"""
        Returns the state type.
        
        Returns:
            int: The state type.
        """
		pass

class BaselineState(State):
	"""
	State representing normal behavior of the operator. The normal behavior means that
	the operator is not deviating its bounds and is generating values around the typical value.
	The operator can also deviate its bounds with a certain probability.
	"""
	def next_step(self, context: 'HighVariability') -> float:
		if context.current_value is None:
			# Generate a new value around the typical value
			if random.random() < context.theta_prob:
				new_value = random.uniform(
					max(context.lower_bound, context.typical_value - context.theta),
					min(context.upper_bound, context.typical_value + context.theta)
				)
			else:
				new_value = context.typical_value
		else:
			# Generate a new value based on the previous value
			new_value = context.current_value
			if random.random() < context.theta_prob:
				new_value += random.uniform(-1, 1) * context.theta
			if random.random() < context.typical_bias_prob:
				new_value += (context.typical_value - new_value) * context.typical_bias

			if random.random() < context.deviate_prob:
				# Move to a new state
				context.set_state(DeviationState())
				context.initialize_state_variables()
			else:
				new_value = max(context.lower_bound, min(new_value, context.upper_bound))
		return new_value

	def get_type(self) -> int:
		return BASELINE

class DeviationState(State):
	"""
    State representing the deviation behavior of the operator. The deviation behavior means
    that the operator is increasing/decreasing the values to a peak/valley.
    It stops when it reaches the peak/valley range.
    """
	def next_step(self, context: 'HighVariability') -> float:
		if context.deviate_steps_total > context.deviate_steps_counter:
            # Generate a new value based on a deterministic trend with noise
			new_value = context.ascending_descending(
				x0=0,
				y0=context.start_value,
				x1=context.deviate_steps_total,
				y1=context.deviate_peak,
				step=context.deviate_steps_counter
			)
			context.deviate_steps_counter += 1
		else:
    		# Move to a new state
			context.set_state(PlateauState())
			return context.state.next_step(context)
		return new_value

	def get_type(self) -> int:
		return DEVIATION

class PlateauState(State):
    """
    State representing the plateau behavior of the operator. The plateau behavior means
    that the operator is holding the value at the peak/valley for a certain duration.
    It stops when it finishes the plateau duration.
    """
    def next_step(self, context: 'HighVariability') -> float:
        if context.plateau_steps_total > context.plateau_steps_counter:
            # Generate a new value based on the plateau previous value and noise
            new_value = context.holding()
            context.plateau_steps_counter += 1
        else:
            # Move to a new state
            context.set_state(RecoveryState())
            return context.state.next_step(context)
        return new_value
    
    def get_type(self) -> int:
        return PLATEAU

class RecoveryState(State):
    """
    State representing the recovery behavior of the operator. The recovery behavior means
    that the operator is returning to the typical value range after deviation and plateau.
    It stops when it reaches the typical value range.
    """
    def next_step(self, context: 'HighVariability') -> float:
        if context.recover_steps_total > context.recover_steps_counter:
            # Generate a new value based on a deterministic trend with noise
            if context.recover_steps_counter == 0:
                context.start_value = context.current_value
            new_value = context.ascending_descending(
                x0=0,
                y0=context.start_value,
                x1=context.recover_steps_total,
                y1=context.typical_value,
                step=context.recover_steps_counter
            )
            context.recover_steps_counter += 1
        else:
            # Move to a new state
            context.set_state(BaselineState())
            new_value = context.current_value
        return new_value
  
    def get_type(self) -> int:
        return RECOVERY


class StandardOperator(ABC):
    """
    Abstract base class for operators in the simulation.

    This class defines the interface and common functionality for all operators.
    """
    @abstractmethod
    def next_step(self, value: float = None) -> float:
        """
        Processes the next step for the operator.

        Parameters:
            value (float, optional): A predefined value for the next step.

        Returns:
            float: The calculated value for the next step.
        """
        pass

    def simulate(self, steps: int) -> list:
        """
        Simulates the operator for a given number of steps.

        Parameters:
            steps (int): The number of steps to simulate.

        Returns:
            list: A list of values generated during the simulation.
        """
        results = []
        for _ in range(steps):
            results.append(self.next_step())
        return results

    def plot_history(self, size: tuple = (10, 5)) -> None:
        """
        Plots the history of the operator's values.

        Parameters:
            size (tuple, optional): The size of the plot. Defaults to (10, 5).
        """
        plt.figure(figsize=size)
        plt.plot(self.stack, label="Values")
        plt.axhline(self.typical_value, color="r", linestyle="--", label="Typical Value")
        plt.title(f"Value Stack History: {self.name}")
        plt.xlabel("Steps")
        plt.ylabel("Values")
        plt.legend()
        plt.show()

class HighVariability(StandardOperator): 
    """
    Operator with large variations and deviation behavior.

    This operator generates values around a typical value with occasional
    deviation behavior, plateau behavior at peaks/valleys, and then recovers to normal (baseline).

    Attributes:
        lower_bound (float): Minimum value the operator can take in normal operation.
        upper_bound (float): Maximum value the operator can take in normal operation.
        typical_value (float): The value the operator should hover around.
        name (str): The name of the operator.
        typical_bias (float): Bias towards the typical value.
        typical_bias_prob (float): Probability of applying the bias.
        theta (float): Maximum variation from the previous value.
        theta_prob (float): Probability of applying the variation.
        deviate_prob (float): Probability of deviation behavior.
        deviate_duration_range (tuple): Range of steps for deviation behavior.
        recover_duration_range (tuple): Range of steps for recovery to normal.
        deviate_bias_range (tuple): Range of bias for deviation behavior.
        deviate_bias_prob (float): Probability of applying the bias during deviation.
        deviate_cos_prob (float): Probability of applying a cosine variation.
        deviate_sin_prob (float): Probability of applying a sine variation.
        deviate_peak_value_range (tuple): Range of values for the peak of deviation behavior.
        plat_duration_range (tuple): Range of steps for plateau at the peak.
        plat_prob_vary (float): Probability of varying at the plateau behavior.
        plat_variation (float): Variation range for plateau behavior.
        stack (list[float]): History of generated values.
        state (State): Current state of the operator.
    """
    def __init__(self, lower_bound: float, 
               upper_bound: float, 
               typical_value: float, 
               name: str, 
               typical_bias: float = 0.1, 
               typical_bias_prob: float = 0.1, 
               theta: float = 1.5, 
               theta_prob: float = 0.1, 
               deviate_prob: float = 0.0, 
               deviate_duration_range: tuple = (10, 20), 
               recover_duration_range: tuple = (10, 20), 
               deviate_bias_range: tuple = (-15, 5), 
               deviate_bias_prob: float = 0.1, 
               deviate_cos_prob: float = 0.5, 
               deviate_sin_prob: float = 0.5, 
               deviate_peak_value_range: tuple = (0, 5), 
               plat_duration_range: tuple = (1000, 2000), 
               plat_prob_vary: float = 0.05, 
               plat_variation: float = 0.02):
        """
        Initializes the HighVariability operator.

        Parameters:
            lower_bound (float): Minimum value the operator can take in normal operation.
            upper_bound (float): Maximum value the operator can take in normal operation.
            typical_value (float): The value the operator should hover around.
            name (str): The name of the operator.
            typical_bias (float): Bias towards the typical value.
            typical_bias_prob (float): Probability of applying the bias.
            theta (float): Maximum variation from the previous value.
            theta_prob (float): Probability of applying the variation.
            deviate_prob (float): Probability of deviation behavior.
            deviate_duration_range (tuple): Range of steps for deviation behavior.
            recover_duration_range (tuple): Range of steps for recovery to normal.
            deviate_bias_range (tuple): Range of bias for deviation behavior.
            deviate_bias_prob (float): Probability of applying the bias during deviation.
            deviate_cos_prob (float): Probability of applying a cosine variation.
            deviate_sin_prob (float): Probability of applying a sine variation.
            deviate_peak_value_range (tuple): Range of values for the peak of deviation behavior.
            plat_duration_range (tuple): Range of steps for plateau at the peak.
            plat_prob_vary (float): Probability of varying at the plateau behavior.
            plat_variation (float): Variation range for plateau behavior.
        """
    
        # Operator parameters about the baseline behavior
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.typical_value = typical_value
        self.typical_bias = typical_bias
        self.typical_bias_prob = typical_bias_prob
        self.theta = theta
        self.theta_prob = theta_prob

        self.name = name
        self.stack = []
        self.current_value = None
        
        # Operator parameters about the deviation and recovery behavior
        self.deviate_prob = deviate_prob
        self.deviate_duration_range = deviate_duration_range
        self.recover_duration_range = recover_duration_range
        self.deviate_bias_range = deviate_bias_range
        self.deviate_bias_prob = deviate_bias_prob
        self.deviate_cos_prob = deviate_cos_prob
        self.deviate_sin_prob = deviate_sin_prob
        self.deviate_peak_value_range = deviate_peak_value_range

        # Operator parameters about the plateau behavior
        self.plat_duration_range = plat_duration_range
        self.plat_prob_vary = plat_prob_vary
        self.plat_variation = plat_variation

        # Operator state variables
        self.deviate_peak = None
        self.plateau_steps_total = 0
        self.plateau_steps_counter = 0
        self.deviate_steps_total = 0
        self.deviate_steps_counter = 0
        self.recover_steps_total = 0
        self.recover_steps_counter = 0
        self.start_value = typical_value

        # Operator state
        self.state = BaselineState()

    def initialize_state_variables(self) -> None:
        """
        Initializes the state variables for the operator.
        """
        self.deviate_peak = random.uniform(
            self.deviate_peak_value_range[0],
            self.deviate_peak_value_range[1]
        )
        # Randomize the duration of each state
        self.deviate_steps_total = random.randint(*self.deviate_duration_range)
        self.deviate_steps_counter = 0
        self.plateau_steps_total = random.randint(*self.plat_duration_range)
        self.plateau_steps_counter = 0
        self.recover_steps_total = random.randint(*self.recover_duration_range)
        self.recover_steps_counter = 0
        self.start_value = self.current_value

    def set_state(self, state: State) -> None:
        """
        Sets the current state of the operator.

        Parameters:
            state (State): The new state to transition to.
        """
        self.state = state

    def next_step(self, value: float = None) -> float:
        """
        Processes the next step based on the current state or the value provided.

        Parameters:
            value (float, optional): A predefined value to use for the next step.
                                     If None, the value is calculated based on the current state.

        Returns:
            float: The next value in the sequence.
        """
        if value is not None:
            new_value = value
        else:
            new_value = self.state.next_step(self)
        self.stack.append(new_value)
        self.current_value = new_value
        return new_value

    def ascending_descending(self, x0: int, y0: float, x1: int, y1: float, step: int) -> float:
        """
        Calculates a value that ascends or descends linearly applying a bias and noise.

        Parameters:
            x0 (int): Starting step.
            y0 (float): Starting value.
            x1 (int): Ending step.
            y1 (float): Ending value.
            step (int): Current step.

        Returns:
            float: Interpolated value at the given step.
        """
        dx = x1 - x0
        dy = y1 - y0
        y = y0 + dy * (step - x0) / dx

        if step < x1 - dx * 0.05 and step > x0 + dx * 0.05 and random.random() < self.deviate_bias_prob:
            y += random.uniform(*self.deviate_bias_range)
        if random.random() < self.deviate_cos_prob:
            y += cos(step / 5)
        if random.random() < self.deviate_sin_prob:
            y += sin(step / 10)

        y = max(0, min(y, 100))
        return y

    def holding(self):
        """
        Calculates the value during the plateau state with a variation noise.

        Returns:
            float: The value during the plateau state.
        """
        y = self.current_value
        if random.random() < self.plat_prob_vary:
            y += random.uniform(-self.plat_variation, self.plat_variation)
        y = max(self.deviate_peak_value_range[0], min(y, self.deviate_peak_value_range[1]))
        return y

class LowVariability(StandardOperator):
    """
    Operator with small variations and linear trends.

    This operator generates values around a typical value with linear trends
    for each step. The trends can be set to a specific range of steps.

    Attributes:
        lower_bound (float): Minimum value the operator can take in normal operation.
        upper_bound (float): Maximum value the operator can take in normal operation.
        typical_value (float): The value the operator should hover around.
        name (str): The name of the operator.
        theta (float): Maximum variation from the previous value.
        steps_range (tuple): Range of steps for the linear trend.
        stack (list[float]): History of generated values.
        total_steps (int): Total number of steps for the current trend.
        steps_counter (int): Counter for the current trend steps.
        start_value (float): Starting value of the current trend.
        end_value (float): Ending value of the current trend.
    """
    def __init__(self, lower_bound: float,
                    upper_bound: float,
                    typical_value: float,
                    name: str,
                    theta: float,
                    steps_range: tuple,
                    seed: int = random.randint(0, 10000)):
        """
        Initializes the LowVariability operator.

        Parameters:
            lower_bound (float): Minimum value the operator can take in normal operation.
            upper_bound (float): Maximum value the operator can take in normal operation.
            typical_value (float): The value the operator should hover around.
            name (str): The name of the operator.
            theta (float): Maximum variation from the previous value.
            steps_range (tuple): Range of steps for the linear trend.
            seed (int, optional): Seed for random number generation. Defaults to a random integer.
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.typical_value = typical_value
        self.theta = theta
        self.steps_range = steps_range
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.name = name
        self.stack = []
        self.current_value = None
        self.total_steps = 0
        self.steps_counter = 0
        self.start_value = None
        self.end_value = None

        self.state = BaselineState()

    def linear_trend(self, x0: int, y0: float, x1: int, y1: float, step: int) -> float:
        """
        Calculates a linear trend between two points.

        Parameters:
            x0 (int): Starting step.
            y0 (float): Starting value.
            x1 (int): Ending step.
            y1 (float): Ending value.
            step (int): Current step.

        Returns:
            float: Interpolated value at the given step.
        """
        dx = x1 - x0
        dy = y1 - y0
        return y0 + dy * (step - x0) / dx
  
    def set_new_trend(self, range: tuple = None) -> None:
        """
        Sets the number of steps and initializes a new trend.

        Parameters:
            range (tuple, optional): Custom range for the number of steps.
                                     Defaults to self.steps_range.
        """
        if range:
            self.total_steps = self.rng.integers(*range)
        else:
            self.total_steps = self.rng.integers(*self.steps_range)
        self.steps_counter = 0

    def next_step(self, value: float = None) -> float:
        """
        Processes the next step based on the current trend or the value provided.

        Parameters:
            value (float, optional): A predefined value to use for the next step.
                                     If None, the value is calculated based on the trend.

        Returns:
            float: The next value in the sequence.
        """
        # Use the provided value directly
        if value is not None:
            new_value = value

        # Initialize the stack with a value near the typical value
        elif self.current_value is None:
            new_value = self.rng.uniform(
                max(self.lower_bound, self.typical_value - self.theta),
                min(self.upper_bound, self.typical_value + self.theta)
            )
        
        # Process the next step in the trend
        else:
            if self.total_steps == self.steps_counter:
                self.set_new_trend()
                self.start_value = self.current_value
                self.end_value = self.rng.uniform(
                    max(self.lower_bound, self.typical_value - self.theta),
                    min(self.upper_bound, self.typical_value + self.theta)
                )

            self.steps_counter += 1
            new_value = self.linear_trend(
                x0=0,
                y0=self.start_value,
                x1=self.total_steps,
                y1=self.end_value,
                step=self.steps_counter
            )

        self.stack.append(new_value)
        self.current_value = new_value
        return new_value
