
import random
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from math import sin, cos
from Operators import StandardOperator, GreatVariation, LittleVariation
from Relations import Correlacao, CorrelacaoH2Metano


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
  def __init__(self, random_seed=None):
    if random_seed:
      random.seed(random_seed)
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
        print(f'{edge.child.name}', end=', ')
      print()


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
