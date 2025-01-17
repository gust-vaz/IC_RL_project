
import random
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class Standard:
  def __init__(self, lower_bound, upper_bound, typical_value, 
               name, typical_bias=0.1, theta=1.5):
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


class BaseOperators:
  def __init__(self, ope1: Standard, ope2: Standard, lower_extra_range, upper_extra_range,
               lower_typical_range, upper_typical_range):
    self.ope1 = ope1
    self.ope2 = ope2
    self.lower_extra_range = lower_extra_range
    self.upper_extra_range = upper_extra_range
    self.lower_typical_range = lower_typical_range
    self.upper_typical_range = upper_typical_range
  

  def next_step(self):
    base_value = self.ope1.next_step()
    
    # Calculate the range for ope2 based on the constraints
    min_ope2 = max(self.lower_extra_range - base_value, self.ope2.lower_bound)
    max_ope2 = min(self.upper_extra_range - base_value, self.ope2.upper_bound)
    
    # Bias towards the typical range
    typical_sum = (self.lower_typical_range + self.upper_typical_range) / 2
    typical_ope2 = typical_sum - base_value
    typical_ope2 = max(min_ope2, min(typical_ope2, max_ope2))
    
    # Generate the next value for ope2
    if not self.ope2.stack:
      new_value = random.uniform(
        max(min_ope2, typical_ope2 - self.ope2.theta),
        min(max_ope2, typical_ope2 + self.ope2.theta)
      )
    else:
      variation = random.uniform(-1, 1) * self.ope2.theta
      new_value = self.ope2.stack[-1] + variation
      
      # Bias towards the typical value
      bias = (typical_ope2 - new_value) * self.ope2.typical_bias
      new_value += bias

      # Ensure bounds
      new_value = max(min_ope2, min(new_value, max_ope2))
    
    self.ope2.next_step(value=new_value)


  def simulate(self, steps):
    for _ in range(steps):
      self.next_step()


  def show_history(self):
    plt.figure(figsize=(10, 5))
    plt.plot(self.ope1.stack, label=self.ope1.name)
    plt.plot(self.ope2.stack, label=self.ope2.name)
    plt.title(f"Value Stack History: {self.ope1.name} e {self.ope2.name}")
    plt.xlabel("Steps")
    plt.ylabel("Values")
    plt.legend()
    plt.show()
  

  def show_sum_history(self):
    sum_stack = [ope_1 + ope_2 for ope_1, ope_2 in zip(self.ope1.stack, self.ope2.stack)]
    plt.figure(figsize=(10, 5))
    plt.plot(sum_stack, label="Sum of Values", color="green")
    plt.title("Sum of Values Stack History")
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


class CorrUsual(Correlacao):
  def __init__(self, veja):
    self.veja = veja

  def calculate(self, from_node, to_node):
    print(f'Strategy B: from {from_node.name} to {to_node.name} and {self.veja}')


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