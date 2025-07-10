from abc import ABC, abstractmethod
import random
from math import cos, sin
import matplotlib.pyplot as plt

# TODO: class MaxEnergyPossible based on H2 and Metan
# TODO: class GeneratedEnergy   based on MaxEnergyPossible
# TODO: constants to define MaxAlertThreshold and MinAlertThreshold

# BASELINE, DEVIATION, PLATEAU, RECOVERY

# Constants for state machine
NORMAL    = 0
EXCEEDING = 1
HOLDING   = 2
RETURNING = 3

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

class NormalState(State):
	"""
	State representing normal behavior of the operator. The normal behavior means that
	the operator is not exceeding its bounds and is generating values around the typical value.
	The operator can also exceed its bounds with a certain probability.
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

			if random.random() < context.exceed_prob:
				# Move to a new state
				context.set_state(ExceedingState())
				context.initialize_state_variables()
			else:
				new_value = max(context.lower_bound, min(new_value, context.upper_bound))
		return new_value

	def get_type(self) -> int:
		return NORMAL

class ExceedingState(State):
	"""
    State representing the exceeding behavior of the operator. The exceeding behavior means
    that the operator is increasing/decreasing the values to the a peak/valley.
    It stops when it reaches the peak/valley range.
    """
	def next_step(self, context: 'HighVariability') -> float:
		if context.exceed_steps_total > context.exceed_steps_counter:
            # Generate a new value based on a deterministic trend with noise
			new_value = context.ascending_descending(
				x0=0,
				y0=context.start_value,
				x1=context.exceed_steps_total,
				y1=context.exceeding_peak,
				step=context.exceed_steps_counter
			)
			context.exceed_steps_counter += 1
		else:
    		# Move to a new state
			context.set_state(HoldingState())
			return context.state.next_step(context)
		return new_value

	def get_type(self) -> int:
		return EXCEEDING

class HoldingState(State):
    """
    State representing the holding behavior of the operator. The holding behavior means
    that the operator is holding the value at the peak/valley for a certain duration.
    It stops when it finishes the hold duration.
    """
    def next_step(self, context: 'HighVariability') -> float:
        if context.hold_steps_total > context.hold_steps_counter:
            # Generate a new value based on the holding previous value and noise
            new_value = context.holding()
            context.hold_steps_counter += 1
        else:
            # Move to a new state
            context.set_state(ReturningState())
            return context.state.next_step(context)
        return new_value
    
    def get_type(self) -> int:
        return HOLDING

class ReturningState(State):
    """
    State representing the returning behavior of the operator. The returning behavior means
    that the operator is returning to the typical value range after exceeding and holding.
    It stops when it reaches the typical value range.
    """
    def next_step(self, context: 'HighVariability') -> float:
        if context.return_steps_total > context.return_steps_counter:
            # Generate a new value based on a deterministic trend with noise
            if context.return_steps_counter == 0:
                context.start_value = context.current_value
            new_value = context.ascending_descending(
                x0=0,
                y0=context.start_value,
                x1=context.return_steps_total,
                y1=context.typical_value,
                step=context.return_steps_counter
            )
            context.return_steps_counter += 1
        else:
            # Move to a new state
            context.set_state(NormalState())
            new_value = context.current_value
        return new_value
  
    def get_type(self) -> int:
        return RETURNING


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
    Operator with large variations and exceeding behavior.

    This operator generates values around a typical value with occasional
    exceeding behavior, holding at peaks, and returning to normal.

    Attributes:
        lower_bound (float): Minimum value the operator can take in normal operation.
        upper_bound (float): Maximum value the operator can take in normal operation.
        typical_value (float): The value the operator should hover around.
        name (str): The name of the operator.
        typical_bias (float): Bias towards the typical value.
        typical_bias_prob (float): Probability of applying the bias.
        theta (float): Maximum variation from the previous value.
        theta_prob (float): Probability of applying the variation.
        exceed_prob (float): Probability of exceeding behavior.
        exceed_duration_range (tuple): Range of steps for exceeding behavior.
        return_duration_range (tuple): Range of steps for returning to normal.
        exceed_bias_range (tuple): Range of bias for exceeding behavior.
        exceed_bias_prob (float): Probability of applying the bias during exceeding.
        exceed_cos_prob (float): Probability of applying a cosine variation.
        exceed_sin_prob (float): Probability of applying a sine variation.
        exceed_peak_value_range (tuple): Range of values for the peak of exceeding behavior.
        hold_duration_range (tuple): Range of steps for holding at the peak.
        hold_prob_vary (float): Probability of varying the holding behavior.
        hold_variation (float): Variation range for holding behavior.
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
               exceed_prob: float = 0.0, 
               exceed_duration_range: tuple = (10, 20), 
               return_duration_range: tuple = (10, 20), 
               exceed_bias_range: tuple = (-15, 5), 
               exceed_bias_prob: float = 0.1, 
               exceed_cos_prob: float = 0.5, 
               exceed_sin_prob: float = 0.5, 
               exceed_peak_value_range: tuple = (0, 5), 
               hold_duration_range: tuple = (1000, 2000), 
               hold_prob_vary: float = 0.05, 
               hold_variation: float = 0.02):
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
            exceed_prob (float): Probability of exceeding behavior.
            exceed_duration_range (tuple): Range of steps for exceeding behavior.
            return_duration_range (tuple): Range of steps for returning to normal.
            exceed_bias_range (tuple): Range of bias for exceeding behavior.
            exceed_bias_prob (float): Probability of applying the bias during exceeding.
            exceed_cos_prob (float): Probability of applying a cosine variation.
            exceed_sin_prob (float): Probability of applying a sine variation.
            exceed_peak_value_range (tuple): Range of values for the peak of exceeding behavior.
            hold_duration_range (tuple): Range of steps for holding at the peak.
            hold_prob_vary (float): Probability of varying the holding behavior.
            hold_variation (float): Variation range for holding behavior.
        """
    
        # Operator parameters about the normal behavior
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
        
        # Operator parameters about the exceeding and returning behavior
        self.exceed_prob = exceed_prob
        self.exceed_duration_range = exceed_duration_range
        self.return_duration_range = return_duration_range
        self.exceed_bias_range = exceed_bias_range
        self.exceed_bias_prob = exceed_bias_prob
        self.exceed_cos_prob = exceed_cos_prob
        self.exceed_sin_prob = exceed_sin_prob
        self.exceed_peak_value_range = exceed_peak_value_range

        # Operator parameters about the holding behavior
        self.hold_duration_range = hold_duration_range
        self.hold_prob_vary = hold_prob_vary
        self.hold_variation = hold_variation

        # Operator state variables
        self.exceeding_peak = None
        self.hold_steps_total = 0
        self.hold_steps_counter = 0
        self.exceed_steps_total = 0
        self.exceed_steps_counter = 0
        self.return_steps_total = 0
        self.return_steps_counter = 0
        self.start_value = typical_value

        # Operator state
        self.state = NormalState()

    def initialize_state_variables(self) -> None:
        """
        Initializes the state variables for the operator.
        """
        self.exceeding_peak = random.uniform(
            self.exceed_peak_value_range[0],
            self.exceed_peak_value_range[1]
        )
        # Randomize the duration of each state
        self.exceed_steps_total = random.randint(*self.exceed_duration_range)
        self.exceed_steps_counter = 0
        self.hold_steps_total = random.randint(*self.hold_duration_range)
        self.hold_steps_counter = 0
        self.return_steps_total = random.randint(*self.return_duration_range)
        self.return_steps_counter = 0
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

        if step < x1 - dx * 0.05 and step > x0 + dx * 0.05 and random.random() < self.exceed_bias_prob:
            y += random.uniform(*self.exceed_bias_range)
        if random.random() < self.exceed_cos_prob:
            y += cos(step / 5)
        if random.random() < self.exceed_sin_prob:
            y += sin(step / 10)

        y = max(0, min(y, 100))
        return y

    def holding(self):
        """
        Calculates the value during the holding state with a variation noise.

        Returns:
            float: The value during the holding state.
        """
        y = self.current_value
        if random.random() < self.hold_prob_vary:
            y += random.uniform(-self.hold_variation, self.hold_variation)
        y = max(self.exceed_peak_value_range[0], min(y, self.exceed_peak_value_range[1]))
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
                    steps_range: tuple):
        """
        Initializes the LowVariability operator.

        Parameters:
            lower_bound (float): Minimum value the operator can take in normal operation.
            upper_bound (float): Maximum value the operator can take in normal operation.
            typical_value (float): The value the operator should hover around.
            name (str): The name of the operator.
            theta (float): Maximum variation from the previous value.
            steps_range (tuple): Range of steps for the linear trend.
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.typical_value = typical_value
        self.theta = theta
        self.steps_range = steps_range

        self.name = name
        self.stack = []
        self.current_value = None
        self.total_steps = 0
        self.steps_counter = 0
        self.start_value = None
        self.end_value = None

        self.state = NormalState()

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
            self.total_steps = random.randint(*range)
        else:
            self.total_steps = random.randint(*self.steps_range)
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
            new_value = random.uniform(
                max(self.lower_bound, self.typical_value - self.theta),
                min(self.upper_bound, self.typical_value + self.theta)
            )
        
        # Process the next step in the trend
        else:
            if self.total_steps == self.steps_counter:
                self.set_new_trend()
                self.start_value = self.current_value
                self.end_value = random.uniform(
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
