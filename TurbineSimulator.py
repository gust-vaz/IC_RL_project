
import random
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class Standard:
  def __init__(self, lower_bound, upper_bound, typical_value, 
               name, typical_bias=0.1, theta=1):
    '''
    Default operator class for the simulator.

      - lower_bound: The minimum value the operator can take
      - upper_bound: The maximum value the operator can take
      - typical_value: The value the operator should be around
      - name: The name of the operator
      - typical_bias: The bias towards the typical value
      - theta: The maximum variation from the previous value
    '''
    self.lower_bound = lower_bound
    self.upper_bound = upper_bound
    self.typical_value = typical_value
    self.theta = theta
    self.typical_bias = typical_bias
    self.name = name
    self.stack = []
    if not (lower_bound <= typical_value <= upper_bound):
      raise ValueError("Typical value must be within the bounds")

  def next_step(self, value=None):
    if value:
      new_value = value
    elif not self.stack:
      # Start near the typical value
      new_value = random.uniform(
        max(self.lower_bound, self.typical_value - self.theta),
        min(self.upper_bound, self.typical_value + self.theta)
      )
    else:
      variation = random.uniform(-1, 1) * self.theta
      new_value = self.stack[-1] + variation
      
      # Bias towards the typical value
      bias = (self.typical_value - new_value) * self.typical_bias
      new_value += bias

      # Ensure bounds
      new_value = max(self.lower_bound, min(new_value, self.upper_bound))
        
    self.stack.append(new_value)

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