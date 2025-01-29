from abc import ABC, abstractmethod
import random
from math import cos, sin
import matplotlib.pyplot as plt

class StandardOperator(ABC):
  @abstractmethod
  def next_step(self, value=None):
    pass

  def simulate(self, steps) -> None:
    for _ in range(steps):
      self.next_step()

  def show_history(self,size=(10,5)) -> None:
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
                 typical_bias: float =0.1, 
                 typical_bias_prob: float =0.1,
                 theta: float =1.5, 
                 theta_prob: float =0.1,
                 exceed_prob: float =0.0, 
                 exceed_duration_range: tuple =(10, 20),
                 return_duration_range: tuple =(10, 20),
                 exceed_bias_range: tuple =(-15, 5), 
                 exceed_bias_prob: float =0.1,
                 exceed_cos_prob: float =0.5,
                 exceed_sin_prob: float =0.5,
                 exceed_peak_value_range: tuple =(0,5), 
                 hold_duration_range: tuple =(1000,2000), 
                 hold_prob_vary: float =0.05,
                 hold_variation: float =0.02):
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

      # Operator parameters about the holding exceeding behavior
      self.hold_duration_range = hold_duration_range
      self.hold_prob_vary = hold_prob_vary
      self.hold_variation = hold_variation

      # State variables for exceeding behavior
      self.exceeding = False
      self.exceeding_peak = None
      self.hold_steps_total = 0
      self.hold_steps_counter = 0
      self.exceed_steps_total = 0
      self.exceed_steps_counter = 0
      self.return_steps_total = 0
      self.return_steps_counter = 0
      self.start = 0

      if not (lower_bound <= typical_value <= upper_bound):
        raise ValueError("Typical value must be within the bounds")

  def ascending_descending(self, x0, y0, x1, y1, step):
    dx = x1 - x0
    dy = y1 - y0
    y = y0 + dy * (step - x0) / dx

    if step < x1 - dx*0.05 and step > x0 + dx*0.05 and random.random() < self.exceed_bias_prob:
      y += random.randint(self.exceed_bias_range[0], self.exceed_bias_range[1])
    
    if random.random() < self.exceed_cos_prob:
      y += cos(step/5)
    
    if random.random() < self.exceed_sin_prob:
      y += sin(step/10)
    
    y = max(0, min(y, 100))
    return y
    
  def holding(self):
      y = self.stack[-1]
      
      if random.random() < self.hold_prob_vary:
        y += random.uniform(-self.hold_variation, self.hold_variation)
      y = max(self.exceed_peak_value_range[0], min(y, self.exceed_peak_value_range[1]))
      
      return y

  def next_step(self, value=None):
    # Defined value
    if value is not None:
      new_value = value

    # Start near the typical value
    elif not self.stack:
      if random.random() < self.theta_prob:
        new_value = random.uniform(
          max(self.lower_bound, self.typical_value - self.theta),
          min(self.upper_bound, self.typical_value + self.theta)
        )
      else:
        new_value = self.typical_value

    # Processing the next step
    else:
      # Handle exceeding behavior
      if self.exceeding:
        if self.exceed_steps_total > self.exceed_steps_counter:
          # Gradually move to the peak with small random variations
          new_value = self.ascending_descending( 
            x0=0, 
            y0=self.stack[self.start], 
            x1=self.exceed_steps_total,
            y1=self.exceeding_peak, 
            step=self.exceed_steps_counter
          )
          self.exceed_steps_counter += 1
        elif self.hold_steps_total > self.hold_steps_counter:
          # Holding at the peak with small variations
          new_value = self.holding()
          self.hold_steps_counter += 1
        elif self.return_steps_total > self.return_steps_counter:
          if self.return_steps_counter == 0:
            self.start = len(self.stack) - 1
          # Gradually return to the typical value with small random variations
          new_value = self.ascending_descending(
            x0=0, 
            y0=self.stack[self.start], 
            x1=self.return_steps_total,
            y1=self.typical_value, 
            step=self.return_steps_counter
          )
          self.return_steps_counter += 1
        else:
          # Reset to normal behavior
          self.exceeding = False
          new_value = self.stack[-1]
      # Normal operation
      else:
        new_value = self.stack[-1]
        
        # Random walk with a bias towards the typical value
        if random.random() < self.theta_prob:
          variation = random.uniform(-1, 1) * self.theta
          new_value += variation

        # Bias towards the typical value
        if random.random() < self.typical_bias_prob:
          bias = (self.typical_value - new_value) * self.typical_bias
          new_value += bias

        # Ensure bounds unless exceeding probability triggers
        if random.random() < self.exceed_prob:
          self.exceeding = True
          self.exceeding_peak = random.uniform(
            self.exceed_peak_value_range[0],
            self.exceed_peak_value_range[1]
          )
          # Randomize durations
          self.exceed_steps_total = random.randint(*self.exceed_duration_range)
          self.exceed_steps_counter = 0
          self.hold_steps_total = random.randint(*self.hold_duration_range)
          self.hold_steps_counter = 0
          self.return_steps_total = random.randint(*self.return_duration_range)
          self.return_steps_counter = 0
          self.start = len(self.stack) - 1
        else:
          new_value = max(self.lower_bound, min(new_value, self.upper_bound))

    self.stack.append(new_value)
    return new_value


class LittleVariation(StandardOperator):
  def __init__(self, lower_bound: float,
                 upper_bound: float,
                 typical_value: float,
                 name: str,
                 theta: float =1.5,
                 steps_range: tuple =(10, 20)):
    
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
