from abc import ABC, abstractmethod
from math import sin, cos
import random


class Correlacao(ABC):
  @abstractmethod
  def calculate(self, root, to_node):
    pass


class CorrelacaoH2Metano(Correlacao):
  def __init__(self, limit_lower_bound, 
               limit_upper_bound,
               typical_lower_bound, 
               typical_upper_bound):
    '''
    Correlation strategy for the H2 and Metano operators.

      - limit_lower_bound: The minimum value the sum of the operators can take
      - limit_upper_bound: The maximum value the sum of the operators can take
      - typical_lower_bound: The lower bound of the typical range of the sum
      - typical_upper_bound: The upper bound of the typical range of the sum
    '''
    self.limit_lower_bound = limit_lower_bound
    self.limit_upper_bound = limit_upper_bound
    self.typical_lower_bound = typical_lower_bound
    self.typical_upper_bound = typical_upper_bound

  def calculate(self, root, child):
    base_value = root.op.stack[-1]
    child.op.state = root.op.state
    
    if root.op.state.name() == "Normal":
      # Calculate the range for child based on the constraints
      min_child = max(self.limit_lower_bound - base_value, child.op.lower_bound)
      max_child = min(self.limit_upper_bound - base_value, child.op.upper_bound)

      # Bias towards the typical range
      typical_sum = (self.typical_lower_bound + self.typical_upper_bound) / 2
      typical_child = max(min_child, min(typical_sum - base_value, max_child))

      # Generate the next value for child
      if not child.op.stack:
        new_value = random.uniform(
          max(min_child, typical_child - child.op.theta),
          min(max_child, typical_child + child.op.theta)
        ) if random.random() < child.op.theta_prob else typical_child
        child.op.next_step(value=new_value)
        return
      new_value = child.op.stack[-1]

    else:
      # Calculate the range for child based on the constraints
      min_child = max(self.limit_lower_bound - base_value, 0)
      max_child = min(self.limit_upper_bound - base_value, 100)

      # Bias towards the typical range
      typical_sum = (self.typical_lower_bound + self.typical_upper_bound) / 2
      typical_child = max(min_child, min(typical_sum - base_value, max_child))

      # Generate the next value for child
      new_value = child.op.stack[-1]

    # Random walk with a bias towards the typical value
    if random.random() < child.op.theta_prob:
      new_value += random.uniform(-1, 1) * child.op.theta
    
    # Bias towards the typical value
    if random.random() < child.op.typical_bias_prob:
      new_value += (typical_child - new_value) * child.op.typical_bias
    
    # Ensure bounds
    new_value = max(min_child, min(new_value, max_child))   
    child.op.next_step(value=new_value)


class CorrelacaoGreatLittle(Correlacao):
  def __init__(self, correlation,
               typical_bias_prob=0.1, 
               typical_bias=0.5,
               theta_prob=0.1):
    self.correlation = correlation
    self.typical_bias_prob = typical_bias_prob
    self.typical_bias = typical_bias
    self.theta_prob = theta_prob

  def calculate(self, root, child):
    if child.op.stack and child.op.total_steps == child.op.steps_counter:
      child.op.state = root.op.state
      child.op.set_steps()
      child.op.start_value = child.op.stack[-1]

      # Provide the root variation to the child 
      variation = (root.op.stack[-1] - root.op.typical_value) 
      percentage_variation = variation / root.op.typical_value
      child.op.end_value = child.op.typical_value * (1 + percentage_variation * self.correlation)
      
      # Bias of the operator itself weighted by correlation
      if random.random() < self.typical_bias_prob:
        child.op.end_value += (child.op.typical_value - child.op.end_value) * self.typical_bias
      if random.random() < self.theta_prob:
        child.op.end_value += random.uniform(-1, 1) * child.op.theta
      
      # Ensure bounds
      child.op.end_value = max(0, min(child.op.end_value, 100))

    child.op.next_step()


class CorrelacaoGreatLittle2(Correlacao):
  def __init__(self, correlation,
               typical_bias_prob=0.1, 
               typical_bias=0.5,
               theta_prob=0.1,
               amplifier=1,
               holding_range=(70, 100)):
    self.correlation = correlation
    self.typical_bias_prob = typical_bias_prob
    self.typical_bias = typical_bias
    self.theta_prob = theta_prob
    self.amplifier = amplifier
    self.holding_range = holding_range

  def calculate(self, root, child):
    if child.op.stack and child.op.total_steps == child.op.steps_counter:
      child.op.state = root.op.state

      if root.op.state.name() != "Normal":
        child.op.set_steps(range=self.holding_range)
      else:
        child.op.set_steps()
      child.op.start_value = child.op.stack[-1]

      # Provide the root variation to the child 
      variation = (root.op.stack[-1] - root.op.typical_value) 
      percentage_variation = variation / root.op.typical_value
      if root.op.state.name() != "Normal":
        child.op.end_value = child.op.typical_value * (1 + percentage_variation * self.correlation * self.amplifier)
      else:
        child.op.end_value = child.op.typical_value * (1 + percentage_variation * self.correlation)

      # Bias of the operator itself weighted by correlation
      if random.random() < self.typical_bias_prob and not root.op.state.name() == "Normal":
        child.op.end_value += (child.op.typical_value - child.op.end_value) * self.typical_bias
      if random.random() < self.theta_prob:
        child.op.end_value += random.uniform(-1, 1) * child.op.theta
      
      # Ensure bounds
      child.op.end_value = max(0, min(child.op.end_value, 100))

    child.op.next_step()


class CorrelacaoDownGrowing(Correlacao):
  def __init__(self, correlation,
               typical_bias_prob=0.1,
               typical_bias=0.5,
               theta_prob=0.1,
               amplifier=1,
               holding_range=(70, 100),
               back_range=(0, 30),
               back_typical_prob=0.7,
               back_typical_range=(-10, 10),
               back_cos_prob=0.1,
               back_sin_prob=0.1,
               trigonometric_function_coef=0.01):
    self.correlation = correlation
    self.typical_bias_prob = typical_bias_prob
    self.typical_bias = typical_bias
    self.theta_prob = theta_prob
    self.amplifier = amplifier
    self.holding_range = holding_range

    # Back parameters
    self.back_range = back_range
    self.back_typical_prob = back_typical_prob
    self.back_typical_range = back_typical_range
    self.back_cos_prob = back_cos_prob
    self.back_sin_prob = back_sin_prob
    self.trigonometric_function_coef = trigonometric_function_coef

    self.to_back = False
    self.back_end = None
    self.current_back = 0

  def calculate(self, root, child):
    self.current_back += 1
    if root.op.state.name() == "Returning":
      self.to_back = True

    if child.op.stack and child.op.total_steps == child.op.steps_counter:
      # Setup the child operator to go back to the typical value
      if self.to_back and self.back_end == None:
        self.to_back = True
        self.back_end = random.randint(*self.back_range)
        self.current_back = 0
      
      # Back until around the typical value
      if self.to_back and (root.op.state.name() == "Normal" or root.op.state.name() == "Returning"):
        child.op.state = root.op.state
        child.op.set_steps()

        child.op.start_value = child.op.stack[-1]
        child.op.end_value = self.decide_end(
          x0=0,
          y0=0,
          x1=self.back_end,
          y1=child.op.typical_value,
          step=self.current_back
        )

        if self.current_back >= self.back_end:
          self.to_back = False
          self.back_end = None
      # Normal operation
      else:
        self.to_back = False
        child.op.state = root.op.state

        if root.op.state.name() != "Normal":
          child.op.set_steps(range=self.holding_range)
        else:
          child.op.set_steps()
        child.op.start_value = child.op.stack[-1]

        # Provide the root variation to the child
        variation = (root.op.stack[-1] - root.op.typical_value)
        percentage_variation = variation / root.op.typical_value
        if root.op.state.name() != "Normal":
          child.op.end_value = child.op.typical_value * (1 + percentage_variation * self.correlation * self.amplifier)
        else:
          child.op.end_value = child.op.typical_value * (1 + percentage_variation * self.correlation)

        # Bias of the operator itself weighted by correlation
        if random.random() < self.typical_bias_prob:
          child.op.end_value += (child.op.typical_value - child.op.end_value) * self.typical_bias
        if random.random() < self.theta_prob:
          child.op.end_value += random.uniform(-1, 1) * child.op.theta

      # Ensure bounds
      child.op.end_value = max(0, min(child.op.end_value, 100))
    k = child.op.next_step()
  
  def decide_end(self, x0, y0, x1, y1, step):
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