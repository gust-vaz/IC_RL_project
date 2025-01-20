
import random
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from math import sin, cos

class Standard:
  def __init__(self, lower_bound: float, 
                 upper_bound: float, 
                 typical_value: float, 
                 name: str, 
                 typical_bias: float =0.1, 
                 typical_bias_prob: float =0.1,
                 theta: float =1.5, 
                 theta_prop: float =0.1,
                 exceed_prob: float =0.0, 
                 exceed_duration_range: tuple =(10, 20),
                 return_duration_range: tuple =(10, 20),
                 exceed_bias_range: tuple =(-15, 5), 
                 exceed_bias_prob: float =0.1,
                 exceed_cos_prob: float =0.5,
                 exceed_sin_prob: float =0.5,
                 exceed_peak_value_range: tuple =(-5, 5), 
                 hold_duration_range: tuple =(5, 15), 
                 hold_prob_vary: float =0.05,
                 hold_variation: float =0.02):
      '''
      Default operator class for the simulator with additional features for exceeding behavior.

        - lower_bound: The minimum value the operator can take
        - upper_bound: The maximum value the operator can take
        - typical_value: The value the operator should be around
        - name: The name of the operator
        - typical_bias: The bias towards the typical value
        - theta: The maximum variation from the previous value
        - exceed_prob: The probability of exceeding behavior
        - exceed_duration_range: The range of steps for exceeding behavior
        - return_duration_range: The range of steps for returning to typical value
        - exceed_peak_value_range: The range of values for the peak of exceeding behavior
        - hold_duration_range: The range of steps for holding at the peak
        - hold_variation: The variation range for holding behavior

      '''

      # Operator parameters about the normal behavior
      self.lower_bound = lower_bound
      self.upper_bound = upper_bound
      self.typical_value = typical_value
      self.typical_bias = typical_bias
      self.typical_bias_prob = typical_bias_prob
      self.theta = theta
      self.theta_prop = theta_prop

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
    
    return y
    
  def holding(self):
      y = self.stack[-1]
      
      if random.random() < self.hold_prob_vary:
        y += random.uniform(-self.hold_variation, self.hold_variation)
      y = max(self.exceed_peak_value_range[0], min(y, self.exceed_peak_value_range[1]))
      
      return y

  def next_step(self, value=None):
    if value is not None:
      new_value = value
    elif not self.stack:
      # Start near the typical value
      new_value = random.uniform(
        max(self.lower_bound, self.typical_value - self.theta),
        min(self.upper_bound, self.typical_value + self.theta)
      )
    else:
      if self.exceeding:
        # Handle exceeding behavior
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
        # Handle exceeding behavior
        elif self.hold_steps_total > self.hold_steps_counter:
          # Holding at the peak with small variations
          # new_value = self.exceeding_peak + random.uniform(-self.hold_variation, self.hold_variation)
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
      else:
        # Normal operation
        new_value = self.stack[-1]
        
        # Random walk with a bias towards the typical value
        if random.random() < self.typical_bias_prob:
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

  def simulate(self, steps):
    for _ in range(steps):
      self.next_step()

  def show_history(self):
    plt.figure(figsize=(10, 5))
    plt.plot(self.stack, label="Values")
    plt.axhline(self.typical_value, color="r", linestyle="--", label="Typical Value")
    plt.title(f"Value Stack History: {self.name}")
    plt.xlabel("Steps")
    plt.ylabel("Values")
    plt.legend()
    plt.show()


def show_history(nodes):
  plt.figure(figsize=(10, 5))
  title = ""
  for node in nodes:
    plt.plot(node.op.stack, label=node.name)
    title = title + node.name + ", "
  plt.title(f"Value Stack History: {title}")
  plt.xlabel("Steps")
  plt.ylabel("Values")
  plt.legend()
  plt.show()


def show_sum_history(nodeA, nodeB):
  sum_stack = [ope_1 + ope_2 for ope_1, ope_2 in zip(nodeA.op.stack, nodeB.op.stack)]
  plt.figure(figsize=(10, 5))
  plt.plot(sum_stack, label="Sum of Values", color="green")
  plt.title("Sum of Values Stack History")
  plt.xlabel("Steps")
  plt.ylabel("Values")
  plt.legend()
  plt.show()


class Correlacao(ABC):
  @abstractmethod
  def calculate(self, root, to_node):
    pass


class CorrelacaoH2Metano(Correlacao):
  def __init__(self, limit_lower_bound, limit_upper_bound,
               typical_lower_bound, typical_upper_bound):
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
    
    # Calculate the range for child based on the constraints
    min_child = max(self.limit_lower_bound - base_value, child.op.lower_bound)
    max_child = min(self.limit_upper_bound - base_value, child.op.upper_bound)
    
    # Bias towards the typical range
    typical_sum = (self.typical_lower_bound + self.typical_upper_bound) / 2
    typical_child = typical_sum - base_value
    typical_child = max(min_child, min(typical_child, max_child))
    
    # Generate the next value for child
    if not child.op.stack:
      new_value = random.uniform(
        max(min_child, typical_child - child.op.theta),
        min(max_child, typical_child + child.op.theta)
      )
    else:
      variation = random.uniform(-1, 1) * child.op.theta
      new_value = child.op.stack[-1] + variation
      
      # Bias towards the typical value
      bias = (typical_child - new_value) * child.op.typical_bias
      new_value += bias

      # Ensure bounds
      new_value = max(min_child, min(new_value, max_child))
    
    child.op.next_step(value=new_value)


class CorrelacaoUsual(Correlacao):
  def __init__(self, correlation):
    self.correlation = correlation

  def calculate(self, root, child):
    root_last_value = root.op.stack[-1]
    root_prev_value = root.op.stack[-2] if len(root.op.stack) > 1 else root_last_value
    root_range = root.op.upper_bound - root.op.lower_bound

    # Calculate the normalized trend of the root
    root_trend = (root_last_value - root_prev_value) / root_range

    # Scale the trend to the child's range and apply correlation
    child_range = child.op.upper_bound - child.op.lower_bound
    trend_bias = self.correlation * root_trend * child_range

    # Generate the next value for the child
    if not child.op.stack:
      new_value = random.uniform(
        max(child.op.lower_bound, child.op.typical_value - child.op.theta),
        min(child.op.upper_bound, child.op.typical_value + child.op.theta)
      )
    else:
      variation = random.uniform(-1, 1) * child.op.theta
      new_value = child.op.stack[-1] + variation + trend_bias

      # Bias towards the child's typical value
      bias = (child.op.typical_value - new_value) * child.op.typical_bias
      new_value += bias

      # Ensure bounds
      new_value = max(child.op.lower_bound, min(new_value, child.op.upper_bound))

    child.op.next_step(value=new_value)


class Node:
  def __init__(self, operator):
    self.op = operator
    self.name = operator.name
    self.root = None
    self.edges = []

  def add_edge(self, child, strategy: Correlacao):
    edge = Edge(self, child, strategy=strategy)
    self.edges.append(edge)
    child.root = self
  
  def simulate_component(self):
    if not self.root:
      self.op.next_step()
    for edge in self.edges:
      edge.next_step()
      edge.child.simulate_component()


class Edge:
  def __init__(self, root: Node, child: Node, strategy: Correlacao):
    self.root = root
    self.child = child
    self._strategy = strategy

  @property
  def strategy(self) -> Correlacao:
    """ 
    Returns the strategy itself 
    """
    return self._strategy

  @strategy.setter
  def strategy(self, strategy: Correlacao) -> None:
    """ 
    Defines a new strategy to the class 
    """
    self._strategy = strategy
  
  def next_step(self):
    self._strategy.calculate(self.root, self.child)


class Graph:
  def __init__(self):
    self.nodes = []

  def add_node(self, operator):
    node = Node(operator)
    self.nodes.append(node)
    return node

  def add_edge(self, root: Node, child: Node, strategy: Correlacao):
    root.add_edge(child, strategy)
  
  def simulate(self, steps):
    for i in range(steps):
      for node in self.nodes:
        if not node.root:
          node.simulate_component()

  def display(self):
    for node in self.nodes:
      print(f'Node {node.name}:', end=' ')
      for edge in node.edges:
        print(f'{edge.child.name}', end=' ')
      print()