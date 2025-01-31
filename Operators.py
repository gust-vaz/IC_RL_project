from abc import ABC, abstractmethod
import random
from math import cos, sin
import matplotlib.pyplot as plt


class State(ABC):
  @abstractmethod
  def next_step(self, context, value=None):
    pass

  @abstractmethod
  def name(self):
    pass

class NormalState(State):
  def next_step(self, context, value=None):
    # If the stack is empty, generate a new value around the typical value
    if not context.stack:
      if random.random() < context.theta_prob:
        new_value = random.uniform(
          max(context.lower_bound, context.typical_value - context.theta),
          min(context.upper_bound, context.typical_value + context.theta)
        )
      else:
        new_value = context.typical_value
    # Otherwise, generate a new value around the previous value
    else:
      new_value = context.stack[-1]
      
      # Generate a new value around the previous value
      if random.random() < context.theta_prob:
        new_value += random.uniform(-1, 1) * context.theta
      
      # Apply a bias towards the typical value
      if random.random() < context.typical_bias_prob:
        new_value += (context.typical_value - new_value) * context.typical_bias
      
      # Apply the bounds unless exceeding behavior is triggered
      if random.random() < context.exceed_prob:
        context.set_state(ExceedingState())
        context.exceeding_peak = random.uniform(
          context.exceed_peak_value_range[0],
          context.exceed_peak_value_range[1]
        )
        # Randomize the duration of the each state
        context.exceed_steps_total = random.randint(*context.exceed_duration_range)
        context.exceed_steps_counter = 0
        context.hold_steps_total = random.randint(*context.hold_duration_range)
        context.hold_steps_counter = 0
        context.return_steps_total = random.randint(*context.return_duration_range)
        context.return_steps_counter = 0
        context.start = len(context.stack) - 1
      else:
        new_value = max(context.lower_bound, min(new_value, context.upper_bound))
    
    return new_value

  def name(self):
    return "Normal"

class ExceedingState(State):
  def next_step(self, context, value=None):
    # Gradually increase the value to the peak
    if context.exceed_steps_total > context.exceed_steps_counter:
      new_value = context.ascending_descending(
        x0=0,
        y0=context.stack[context.start],
        x1=context.exceed_steps_total,
        y1=context.exceeding_peak,
        step=context.exceed_steps_counter
      )
      context.exceed_steps_counter += 1
    # Hold the value at the peak
    else:
      context.set_state(HoldingState())
      return context.state.next_step(context)

    return new_value

  def name(self):
    return "Exceeding"

class HoldingState(State):
  def next_step(self, context, value=None):
    # Maintain the value at the peak
    if context.hold_steps_total > context.hold_steps_counter:
      new_value = context.holding()
      context.hold_steps_counter += 1
    # Gradually decrease the value to the typical value
    else:
      context.set_state(ReturningState())
      return context.state.next_step(context)
    
    return new_value
  
  def name(self):
    return "Holding"

class ReturningState(State):
  def next_step(self, context, value=None):
    # Gradually decrease the value to the typical value
    if context.return_steps_total > context.return_steps_counter:
      if context.return_steps_counter == 0:
        context.start = len(context.stack) - 1
      new_value = context.ascending_descending(
        x0=0,
        y0=context.stack[context.start],
        x1=context.return_steps_total,
        y1=context.typical_value,
        step=context.return_steps_counter
      )
      context.return_steps_counter += 1
    # Return to the normal state
    else:
      context.set_state(NormalState())
      new_value = context.stack[-1]
    
    return new_value
  
  def name(self):
    return "Returning"


class StandardOperator(ABC):
  @abstractmethod
  def next_step(self, value=None):
    pass

  def simulate(self, steps) -> None:
    for _ in range(steps):
      self.next_step()

  def show_history(self, size=(10, 5)) -> None:
    plt.figure(figsize=size)
    plt.plot(self.stack, label="Values")
    plt.axhline(self.typical_value, color="r", linestyle="--", label="Typical Value")
    plt.title(f"Value Stack History: {self.name}")
    plt.xlabel("Steps")
    plt.ylabel("Values")
    plt.legend()
    plt.show()

class GreatVariation(StandardOperator): 
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
    '''
      Default operator class for the simulator with additional features for exceeding behavior.

        - lower_bound: The minimum value the operator can take
        - upper_bound: The maximum value the operator can take
        - typical_value: The value the operator should be around
        - name: The name of the operator
        - typical_bias: The bias towards the typical value
        - typical_bias_prob: The probability of applying the bias
        - theta: The maximum variation from the previous value
        - theta_prob: The probability of applying the variation
        - exceed_prob: The probability of exceeding behavior
        - exceed_duration_range: The range of steps for exceeding behavior
        - return_duration_range: The range of steps for returning to typical value
        - exceed_bias_range: The range of bias for exceeding behavior
        - exceed_bias_prob: The probability of applying the bias
        - exceed_cos_prob: The probability of applying a cosine variation
        - exceed_sin_prob: The probability of applying a sine variation
        - exceed_peak_value_range: The range of values for the peak of exceeding behavior
        - hold_duration_range: The range of steps for holding at the peak
        - hold_prob_vary: The probability of varying the holding behavior
        - hold_variation: The variation range for holding behavior
      '''
    
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
    self.start = 0

    # Operator state
    self.state = NormalState()

    if not (lower_bound <= typical_value <= upper_bound):
      raise ValueError("Typical value must be within the bounds")

  def set_state(self, state: State):
    self.state = state

  def next_step(self, value=None):
    # If a value is provided, use it
    if value is not None:
      new_value = value
    else:
      new_value = self.state.next_step(self, value)
    
    self.stack.append(new_value)
    return new_value

  def ascending_descending(self, x0, y0, x1, y1, step):
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
    y = self.stack[-1]

    if random.random() < self.hold_prob_vary:
      y += random.uniform(-self.hold_variation, self.hold_variation)
    y = max(self.exceed_peak_value_range[0], min(y, self.exceed_peak_value_range[1]))

    return y

class LittleVariation(StandardOperator):
  def __init__(self, lower_bound: float,
                 upper_bound: float,
                 typical_value: float,
                 name: str,
                 theta: float =1.5,
                 steps_range: tuple =(10, 20)):
    '''
    Default operator class for the simulator with additional features for big linear step behavior.

    - lower_bound: The minimum value the operator can take
    - upper_bound: The maximum value the operator can take
    - typical_value: The value the operator should be around
    - name: The name of the operator
    - theta: The maximum variation from the previous value
    - steps_range: The range of steps for the big linear step behavior
    '''
    self.lower_bound = lower_bound
    self.upper_bound = upper_bound
    self.typical_value = typical_value
    self.theta = theta
    self.steps_range = steps_range

    self.name = name
    self.stack = []
    self.total_steps = 0
    self.steps_counter = 0
    self.start_value = None
    self.end_value = None

    self.state = NormalState()

  def linear_trend(self, x0, y0, x1, y1, step):
    dx = x1 - x0
    dy = y1 - y0
    y = y0 + dy * (step - x0) / dx
    return y
  
  def set_steps(self, range=None):
    if range:
      self.total_steps = random.randint(*range)
    else:
      self.total_steps = random.randint(*self.steps_range)
    self.steps_counter = 0

  def next_step(self, value=None):
    # Defined value
    if value is not None:
      new_value = value
    
    # Start near the typical value
    elif not self.stack:
      new_value = random.uniform(
        max(self.lower_bound, self.typical_value - self.theta),
        min(self.upper_bound, self.typical_value + self.theta)
      )
    
    # Processing the next step
    else:
      if self.total_steps == self.steps_counter:
        self.set_steps()
        self.start_value = self.stack[-1]
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
    return new_value
