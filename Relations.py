from abc import ABC, abstractmethod
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
    
    if not root.op.exceeding:
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
      else:
        child.op.exceeding = False
        new_value = child.op.stack[-1]

    else:
      child.op.exceeding = True
      child.op.hold_steps_total = root.op.hold_steps_total
      child.op.hold_steps_counter = root.op.hold_steps_counter
      new_value = child.op.stack[-1]

      # Calculate the range for child based on the constraints
      min_child = max(self.limit_lower_bound - base_value, 0)
      max_child = min(self.limit_upper_bound - base_value, 100)

      # Bias towards the typical range
      typical_sum = (self.typical_lower_bound + self.typical_upper_bound) / 2
      typical_child = max(min_child, min(typical_sum - base_value, max_child))

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
      if root.op.exceeding:
        child.op.set_steps(range=self.holding_range)
      else:
        child.op.set_steps()
      child.op.start_value = child.op.stack[-1]

      # Provide the root variation to the child 
      variation = (root.op.stack[-1] - root.op.typical_value) 
      percentage_variation = variation / root.op.typical_value
      if root.op.exceeding:
        child.op.end_value = child.op.typical_value * (1 + percentage_variation * self.correlation * self.amplifier)
      else:
        child.op.end_value = child.op.typical_value * (1 + percentage_variation * self.correlation)

      # Bias of the operator itself weighted by correlation
      if random.random() < self.typical_bias_prob and not root.op.exceeding:
        child.op.end_value += (child.op.typical_value - child.op.end_value) * self.typical_bias
      if random.random() < self.theta_prob:
        child.op.end_value += random.uniform(-1, 1) * child.op.theta
      
      # Ensure bounds
      child.op.end_value = max(0, min(child.op.end_value, 100))

    child.op.next_step()


# class CorrelacaoDownGrowing(Correlacao):
#   def __init__(self, correlation,
#                back_range=(0, 30),
#                typical_bias_prob=0.1,
#                typical_bias=0.5,
#                theta_prob=0.1):
#     self.correlation = correlation
#     self.back_range = back_range
#     self.typical_bias_prob = typical_bias_prob
#     self.typical_bias = typical_bias
#     self.theta_prob = theta_prob

#     self.to_back = False

#   def calculate(self, root, child):
#     if child.op.stack and child.op.total_steps == child.op.steps_counter:
#       # if root.op.exceeding and 
#       child.op.set_steps()
#       child.op.start_value = child.op.stack[-1]

#       # Provide the root variation to the child
#       variation = (root.op.stack[-1] - root.op.typical_value)
#       percentage_variation = variation / root.op.typical_value
#       child.op.end_value = child.op.typical_value * (1 + percentage_variation * self.correlation)

#       # Bias of the operator itself weighted by correlation
#       if random.random() < self.typical_bias_prob:
#         child.op.end_value += (child.op.typical_value - child.op.end_value) * self.typical_bias
#       if random.random() < self.theta_prob:
#         child.op.end_value += random.uniform(-1, 1) * child.op.theta

#       # Ensure bounds
#       child.op.end_value = max(0, min(child.op.end_value, 100))

#     child.op.next_step()
