from abc import ABC, abstractmethod
from math import sin, cos, log
import random

# Constants for state machine
BASELINE  = 0
DEVIATION = 1
PLATEAU   = 2
RECOVERY  = 3

class Link(ABC):
    """
    Abstract base class for defining relationships between nodes in the simulation.
    """

    @abstractmethod
    def calculate(self, parent, child) -> float:
        """
        Calculates the relationship between the parent and child nodes.
        The parent node dictates the behavior of the child node.

        Parameters:
            parent (Node): The parent node.
            child (Node): The child node.
        
        Returns:
            float: The calculated value for the child node.
        """
        pass

class LinkH2Metano(Link):
    """
    Link class for modeling the relationship between hydrogen and methane nodes.

    Attributes:
        limit_lower_bound (float): The minimum value the sum of the operators can take.
        limit_upper_bound (float): The maximum value the sum of the operators can take.
        typical_lower_bound (float): The lower bound of the typical range of the sum.
        typical_upper_bound (float): The upper bound of the typical range of the sum.
    """
    def __init__(self, limit_lower_bound: float, 
                limit_upper_bound: float,
                typical_lower_bound: float,
                typical_upper_bound: float):
        """
        Initializes the LinkH2Metano with constraints for the sum of hydrogen and methane.

        Parameters:
            limit_lower_bound (float): The minimum value the sum of the operators can take.
            limit_upper_bound (float): The maximum value the sum of the operators can take.
            typical_lower_bound (float): The lower bound of the typical range of the sum.
            typical_upper_bound (float): The upper bound of the typical range of the sum.
        """
        self.limit_lower_bound = limit_lower_bound
        self.limit_upper_bound = limit_upper_bound
        self.typical_lower_bound = typical_lower_bound
        self.typical_upper_bound = typical_upper_bound

    def calculate(self, parent, child) -> float:
        # Get the base value from the parent node's stack
        base_value = parent.op.current_value
        # Synchronize the child's state with the parent's state
        child.op.state = parent.op.state
        
        if parent.op.state.get_type() == BASELINE:
            # Calculate the range for child based on the constraints
            min_child = max(self.limit_lower_bound - base_value, child.op.lower_bound)
            max_child = min(self.limit_upper_bound - base_value, child.op.upper_bound)

            # Bias towards the typical range
            typical_sum = (self.typical_lower_bound + self.typical_upper_bound) / 2
            typical_child = max(min_child, min(typical_sum - base_value, max_child))

            # Generate the next value for child
            new_value = self._generate_next_value(child, min_child, max_child, typical_child)

        else:
            # Handle non-baseline states
            min_child = max(self.limit_lower_bound - base_value, 0)
            max_child = min(self.limit_upper_bound - base_value, 100)

            # Bias towards the typical range
            typical_sum = (self.typical_lower_bound + self.typical_upper_bound) / 2
            typical_child = max(min_child, min(typical_sum - base_value, max_child))

            # Generate the next value for child
            new_value = self._generate_next_value(child, min_child, max_child, typical_child)
       
        # Ensure bounds and set the next step
        new_value = max(min_child, min(new_value, max_child))   
        return child.op.next_step(value=new_value)
    
    def _generate_next_value(self, child, min_child: float, max_child: float, typical_child: float) -> float:
        """
        Generates the next value for the child node based on constraints and biases.

        Parameters:
            child (Node): The child node.
            min_child (float): The minimum value the child can take.
            max_child (float): The maximum value the child can take.
            typical_child (float): The typical value the child should hover around.

        Returns:
            float: The next value for the child node.
        """
        if child.op.current_value is None:
            # Initialize the stack with a value near the typical value
            if random.random() < child.op.theta_prob:
                return random.uniform(
                    max(min_child, typical_child - child.op.theta),
                    min(max_child, typical_child + child.op.theta)
                )
            return typical_child

        new_value = child.op.current_value

        # Apply random variation
        if random.random() < child.op.theta_prob:
            new_value += random.uniform(-1, 1) * child.op.theta

        # Bias towards the typical value
        if random.random() < child.op.typical_bias_prob:
            new_value += (typical_child - new_value) * child.op.typical_bias

        return new_value

class LinkSimilarBehavior(Link):
    """
    Link class for modeling similar behavior between parent and child nodes.

    Attributes:
        correlation (float): Correlation factor between the parent and child nodes.
        typical_bias_prob (float): Probability of applying a bias towards the typical value.
        typical_bias (float): Bias factor towards the typical value.
        theta_prob (float): Probability of applying a random variation.
        amplifier (float): Amplifies the correlation effect.
        plateau_range (tuple, optional): Range of steps for plateau behavior.
    """
    def __init__(self, correlation: float,
                typical_bias_prob: float, 
                typical_bias: float,
                theta_prob: float,
                amplifier: float = 1.0,
                plateau_range: tuple = None):
        """
        Initializes the LinkSimilarBehavior with correlation and bias parameters.

        Parameters:
            correlation (float): Correlation factor between the parent and child nodes.
            typical_bias_prob (float): Probability of applying a bias towards the typical value.
            typical_bias (float): Bias factor towards the typical value.
            theta_prob (float): Probability of applying a random variation.
            amplifier (float): Amplifies the correlation effect.
            plateau_range (tuple, optional): Range of steps for plateau behavior.
        """
        self.correlation = correlation
        self.typical_bias_prob = typical_bias_prob
        self.typical_bias = typical_bias
        self.theta_prob = theta_prob
        self.amplifier = amplifier
        self.plateau_range = plateau_range

    def calculate(self, parent, child) -> float:
        if child.op.current_value is not None and child.op.total_steps == child.op.steps_counter:
            # Synchronize the child's state with the parent's state
            child.op.state = parent.op.state

            # Set a new trend for the child based on the parent's state
            if parent.op.state.get_type() != BASELINE:
                child.op.set_new_trend(range=self.plateau_range)
            else:
                child.op.set_new_trend()

            # Initialize the child's trend values
            child.op.start_value = child.op.current_value
            variation = (parent.op.current_value - parent.op.typical_value) 
            percentage_variation = variation / parent.op.typical_value

            # Calculate the child's end value based on the parent's variation
            if parent.op.state.get_type() != BASELINE:
                child.op.end_value = child.op.typical_value * (1 + percentage_variation * self.correlation * self.amplifier)
            else:
                child.op.end_value = child.op.typical_value * (1 + percentage_variation * self.correlation)

            # Bias of the operator itself weighted by correlation
            if random.random() < self.typical_bias_prob and not parent.op.state.get_type() == BASELINE:
                child.op.end_value += (child.op.typical_value - child.op.end_value) * self.typical_bias

            # Apply random variation
            if random.random() < self.theta_prob:
                child.op.end_value += random.uniform(-1, 1) * child.op.theta
            
            # Ensure the child's end value is within bounds
            child.op.end_value = max(0, min(child.op.end_value, 100))

        # Advance the child's operator to the next step
        return child.op.next_step()

class LinkLongRecovery(Link):
    """
    Link class for modeling long recovery behavior between parent and child nodes.

    Attributes:
        correlation (float): Correlation factor between the parent and child nodes.
        typical_bias_prob (float): Probability of applying a bias towards the typical value.
        typical_bias (float): Bias factor towards the typical value.
        theta_prob (float): Probability of applying a random variation.
        amplifier (float): Amplifies the correlation effect.
        plateau_range (tuple): Range of steps for plateau behavior.
        back_range (tuple): Range of steps for recovery behavior.
        back_typical_prob (float): Probability of returning to the typical value.
        back_typical_range (tuple): Range of typical values for recovery behavior.
        back_cos_prob (float): Probability of applying a cosine variation.
        back_sin_prob (float): Probability of applying a sine variation.
        trigonometric_function_coef (float): Coefficient for trigonometric variations.
    """
    def __init__(self, correlation: float,
                typical_bias_prob: float,
                typical_bias: float,
                theta_prob: float,
                amplifier: float,
                plateau_range: tuple,
                back_range: tuple = (0, 30),
                back_typical_prob: float = 0.7,
                back_typical_range: tuple = (-10, 10),
                back_cos_prob: float = 0.1,
                back_sin_prob: float = 0.1,
                trigonometric_function_coef: float = 0.01):
        """
        Initializes the LinkLongRecovery with correlation and recovery behavior parameters.

        Parameters:
            correlation (float): Correlation factor between the parent and child nodes.
            typical_bias_prob (float): Probability of applying a bias towards the typical value.
            typical_bias (float): Bias factor towards the typical value.
            theta_prob (float): Probability of applying a random variation.
            amplifier (float): Amplifies the correlation effect.
            plateau_range (tuple): Range of steps for plateau behavior.
            back_range (tuple): Range of steps for recovery behavior.
            back_typical_prob (float): Probability of returning to the typical value.
            back_typical_range (tuple): Range of typical values for recovery behavior.
            back_cos_prob (float): Probability of applying a cosine variation.
            back_sin_prob (float): Probability of applying a sine variation.
            trigonometric_function_coef (float): Coefficient for trigonometric variations.
        """
        self.correlation = correlation
        self.typical_bias_prob = typical_bias_prob
        self.typical_bias = typical_bias
        self.theta_prob = theta_prob
        self.amplifier = amplifier
        self.plateau_range = plateau_range

        # Back parameters
        self.back_range = back_range
        self.back_typical_prob = back_typical_prob
        self.back_typical_range = back_typical_range
        self.back_cos_prob = back_cos_prob
        self.back_sin_prob = back_sin_prob
        self.trigonometric_function_coef = trigonometric_function_coef

        # State variables for recovery behavior
        self.to_back = False
        self.back_end = None
        self.current_back = 0

    def calculate(self, parent, child) -> float:
        self.current_back += 1
        if parent.op.state.get_type() == RECOVERY:
            self.to_back = True

        if child.op.current_value is not None and child.op.total_steps == child.op.steps_counter:
            if self.to_back:
                self._handle_recovering_behavior(parent, child)
            else:
                self._handle_baseline_behavior(parent, child)
            
            # Ensure the child's end value is within bounds
            child.op.end_value = max(0, min(child.op.end_value, 100))

        return child.op.next_step()
    
    def _handle_recovering_behavior(self, parent, child) -> None:
        """
        Handles the behavior when the child is recovering to its typical value.

        Parameters:
            parent (Node): The parent node.
            child (Node): The child node.
        """
        # Setup the child operator to go back to the typical value
        if self.back_end is None:
            self.back_end = random.randint(*self.back_range)
            self.current_back = 0

        # Back until around the typical value
        if parent.op.state.get_type() == BASELINE or parent.op.state.get_type() == RECOVERY:
            child.op.state = parent.op.state
            child.op.set_new_trend()

            child.op.start_value = child.op.current_value
            child.op.end_value = self._calculate_recover_value(
                    x0=0,
                    y0=0,
                    x1=self.back_end,
                    y1=child.op.typical_value,
                    step=self.current_back
                )
            
            if self.current_back >= self.back_end:
                self.to_back = False
                self.back_end = None

    def _handle_baseline_behavior(self, parent, child) -> None:
        """
        Handles the baseline behavior when the child is not recovering.

        Parameters:
            parent (Node): The parent node.
            child (Node): The child node.
        """
        child.op.state = parent.op.state

        # Set a new trend for the child based on the parent's state
        if parent.op.state.get_type() != BASELINE:
            child.op.set_new_trend(range=self.plateau_range)
        else:
            child.op.set_new_trend()

        # Initialize the child's trend values
        child.op.start_value = child.op.current_value
        variation = parent.op.current_value - parent.op.typical_value
        percentage_variation = variation / parent.op.typical_value
        
        # Calculate the child's end value based on the parent's variation
        if parent.op.state.get_type() != BASELINE:
            child.op.end_value = child.op.typical_value * (1 + percentage_variation * self.correlation * self.amplifier)
        else:
            child.op.end_value = child.op.typical_value * (1 + percentage_variation * self.correlation)

        # Bias of the operator itself weighted by correlation
        if random.random() < self.typical_bias_prob:
            child.op.end_value += (child.op.typical_value - child.op.end_value) * self.typical_bias
        
        # Apply random variation
        if random.random() < self.theta_prob:
            child.op.end_value += random.uniform(-1, 1) * child.op.theta

    def _calculate_recover_value(self, x0: int, y0: float, x1: int, y1: float, step: int) -> float:
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

        if random.random() < self.back_typical_prob:
            y += random.uniform(*self.back_typical_range)
        if random.random() < self.back_cos_prob:
            y += cos(step / 5) * self.trigonometric_function_coef
        if random.random() < self.back_sin_prob:
            y += sin(step / 10) * self.trigonometric_function_coef

        y = max(0, min(y, 100))
        return y

class LinkMaxEnergyFuel(Link):
    """
    Link class for modeling maximum energy fuel behavior between parent and child nodes.
    Attributes:
        time (int): Internal time counter for tracking steps.
    """

    def __init__(self):
        """
        Initializes the LinkMaxEnergyFuel with parameters for energy fuel calculation.
        """

        self.time = -1

    def calculate(self, parents, child):
        self.time += 1
        if child.op.current_value is not None and child.op.total_steps == child.op.steps_counter:
            if parents[1].op.state.get_type() == RECOVERY:
                self.time = min(1000, self.time)
            # Set a new trend for the child
            child.op.set_new_trend()
            
            # Initialize the child's trend values
            child.op.start_value = child.op.current_value

            # Define end value
            end_time = self.time + child.op.total_steps
            child.op.end_value = 0.9 * (0.5 * parents[0].op.current_value + 0.3 * parents[1].op.current_value) * log((end_time + 20) / 15, 20)
            child.op.end_value = max(0, min(child.op.end_value, 100))
        
        elif child.op.current_value is None:
            return child.op.next_step(value=80*log((20) / 15, 20))
        
        return child.op.next_step()

class LinkGeneratedEnergy(Link):
    """
    Link class for modeling generated energy behavior between parent and child nodes.

    Attributes:
        typical_bias_prob (float): Probability of applying a bias towards the typical value.
        typical_bias (float): Bias factor towards the typical value.
        theta_prob (float): Probability of applying a random variation.
        theta_bias (float): Bias factor for the random variation.
        alert_mode (bool): Flag indicating if the system is in alert mode.
    """

    def __init__(self, typical_bias_prob: float,
                 typical_bias: float,
                 theta_prob: float,
                 theta_bias: float):
        """
        Initializes the LinkGeneratedEnergy with bias and variation parameters.

        Parameters:
            typical_bias_prob (float): Probability of applying a bias towards the typical value.
            typical_bias (float): Bias factor towards the typical value.
            theta_prob (float): Probability of applying a random variation.
            theta_bias (float): Bias factor for the random variation.
        """
        self.typical_bias_prob = typical_bias_prob
        self.typical_bias = typical_bias
        self.theta_prob = theta_prob
        self.theta_bias = theta_bias
        self.smooth_factor = 1
        self.alert_mode = False
    
    def calculate(self, parent, child, other_informations):
        if not self.alert_mode and other_informations is not None and 'alert' in other_informations and other_informations['alert'] == True:
            self.alert_mode = True
            child.op.set_new_trend()
            child.op.start_value = child.op.current_value
            child.op.end_value = parent.op.current_value * random.uniform(0.7, 0.8)
            if random.random() < self.theta_prob:
                child.op.end_value -= (parent.op.current_value - child.op.end_value) * self.theta_bias
            if random.random() < self.typical_bias_prob:
                child.op.end_value += (parent.op.current_value - child.op.end_value) * self.typical_bias
            self.smooth_factor = 0.5

        elif child.op.current_value is not None and child.op.total_steps == child.op.steps_counter:
            self.alert_mode = False

            # Set a new trend for the child
            child.op.set_new_trend()
            
            # Initialize the child's trend values
            child.op.start_value = child.op.current_value

            self.smooth_factor = min(1, self.smooth_factor + 0.05)

            # Gradually move child.op.end_value towards parent's end_value after alert
            target_value = parent.op.end_value * random.uniform(0.8, 1.0)
            child.op.end_value = (1 - self.smooth_factor) * child.op.current_value + self.smooth_factor * target_value

            if random.random() < self.theta_prob:
                child.op.end_value -= (parent.op.end_value - child.op.end_value) * self.theta_bias
            if random.random() < self.typical_bias_prob:
                child.op.end_value += (parent.op.end_value - child.op.end_value) * self.typical_bias

        elif child.op.current_value is None:
            return child.op.next_step(parent.op.current_value)
        
        return child.op.next_step()