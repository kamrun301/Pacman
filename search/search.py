# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import*

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


# Complete the BFS, DFS, UCS, and A* search algorithms in the provided structure.

from queue import PriorityQueue

# Depth-First Search (DFS) implementation
def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    """Search the deepest nodes in the search tree first."""
    stack = [(problem.getStartState(), [], set())]  # Stack holds (state, path, visited)
    while stack:
        state, path, visited = stack.pop()
        if problem.isGoalState(state):
            return path
        if state not in visited:
            visited.add(state)
            for successor, action, _ in problem.getSuccessors(state):
                stack.append((successor, path + [action], visited))
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    return []  # Return an empty list if no solution is found


# Breadth-First Search (BFS) implementation
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    """Search the shallowest nodes in the search tree first."""
    """Search the shallowest nodes in the search tree first."""
    from collections import deque
    queue = deque([(problem.getStartState(), [])])  # Queue holds (state, path)
    visited = set()
    while queue:
        state, path = queue.popleft()
        if problem.isGoalState(state):
            return path
        if state not in visited:
            visited.add(state)
            for successor, action, _ in problem.getSuccessors(state):
                queue.append((successor, path + [action]))
    return []  # Return an empty list if no solution is found


# Uniform-Cost Search (UCS) implementation
def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    pq = PriorityQueue()  # Priority queue for cost tracking
    pq.put((0, problem.getStartState(), []))  # (cost, state, path)
    visited = {}
    while not pq.empty():
        cost, state, path = pq.get()
        if state in visited and visited[state] <= cost:
            continue
        visited[state] = cost
        if problem.isGoalState(state):
            return path
        for successor, action, step_cost in problem.getSuccessors(state):
            pq.put((cost + step_cost, successor, path + [action]))
    return []  # Return an empty list if no solution is found



def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    """Search the node that has the lowest combined cost and heuristic first."""
    pq = PriorityQueue()  # Priority queue for f(n) = g(n) + h(n)
    pq.put((0 + heuristic(problem.getStartState(), problem), 0, problem.getStartState(), []))
    visited = {}
    while not pq.empty():
        f, g, state, path = pq.get()
        if state in visited and visited[state] <= g:
            continue
        visited[state] = g
        if problem.isGoalState(state):
            return path
        for successor, action, step_cost in problem.getSuccessors(state):
            new_g = g + step_cost
            new_f = new_g + heuristic(successor, problem)
            pq.put((new_f, new_g, successor, path + [action]))
    return []  # Return an empty list if no solution is found
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
ucs = uniformCostSearch
astar = aStarSearch


#Time & Path for Tiny maze search 
import time

# Example Maze (0 = free, 1 = wall)
tiny_maze = [
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
]

start = (0, 0)  # Starting position
goal = (2, 2)  # Goal position


class MazeProblem:
    def __init__(self, maze, start, goal):
        self.maze = maze
        self.start = start
        self.goal = goal

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state == self.goal

    def getSuccessors(self, state):
        successors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Possible movements: N, E, S, W
        for dx, dy in directions:
            x, y = state[0] + dx, state[1] + dy
            if 0 <= x < len(self.maze) and 0 <= y < len(self.maze[0]) and self.maze[x][y] == 0:
                successors.append(((x, y), (dx, dy), 1))  # Each step has a cost of 1
        return successors

    def getCostOfActions(self, actions):
        return len(actions)  # Assuming each action has a unit cost

# Define the problem
problem = MazeProblem(tiny_maze, start, goal)

# Test each algorithm and measure time + path length
results = {}
algorithms = {"DFS": dfs, "BFS": bfs, "UCS": ucs}

for name, algorithm in algorithms.items():
    start_time = time.time()
    path = algorithm(problem)
    end_time = time.time()
    results[name] = {
        "Path Length": len(path),
        "Time Taken (s)": f"{(end_time - start_time):.6f}"
        
    }

results


#Time & Path for medium maze search 

import time

# Example Maze (0 = free, 1 = wall)
medium_maze = [
    [0, 0, 0, 0, 1],
    [0, 1, 1, 0, 1],
    [0, 0, 0, 0, 0],
    [1, 1, 0, 1, 1],
    [0, 0, 0, 0, 0],
]


start = (0, 0)  # Starting position
goal = (4, 4)  # Goal position


class MazeProblem:
    def __init__(self, maze, start, goal):
        self.maze = maze
        self.start = start
        self.goal = goal

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state == self.goal

    def getSuccessors(self, state):
        successors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Possible movements: N, E, S, W
        for dx, dy in directions:
            x, y = state[0] + dx, state[1] + dy
            if 0 <= x < len(self.maze) and 0 <= y < len(self.maze[0]) and self.maze[x][y] == 0:
                successors.append(((x, y), (dx, dy), 1))  # Each step has a cost of 1
        return successors

    def getCostOfActions(self, actions):
        return len(actions)  # Assuming each action has a unit cost

# Define the problem
problem = MazeProblem(medium_maze, start, goal)

# Test each algorithm and measure time + path length
results = {}
algorithms = {"DFS": dfs, "BFS": bfs, "UCS": ucs}


for name, algorithm in algorithms.items():
    start_time = time.time()
    path = algorithm(problem)
    end_time = time.time()
    results[name] = {
        "Path Length": len(path),
        "Time Taken (s)": f"{(end_time - start_time):.6f}"
    }

results


#Time & Path big maze search 

import time

# Example Maze (0 = free, 1 = wall)
big_maze = [
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
]

start = (0, 0)  # Starting position
goal = (9, 9)  # Goal position


class MazeProblem:
    def __init__(self, maze, start, goal):
        self.maze = maze
        self.start = start
        self.goal = goal

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state == self.goal

    def getSuccessors(self, state):
        successors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Possible movements: N, E, S, W
        for dx, dy in directions:
            x, y = state[0] + dx, state[1] + dy
            if 0 <= x < len(self.maze) and 0 <= y < len(self.maze[0]) and self.maze[x][y] == 0:
                successors.append(((x, y), (dx, dy), 1))  # Each step has a cost of 1
        return successors

    def getCostOfActions(self, actions):
        return len(actions)  # Assuming each action has a unit cost

# Define the problem
problem = MazeProblem(big_maze, start, goal)

# Test each algorithm and measure time + path length
results = {}
algorithms = {"DFS": dfs, "BFS": bfs, "UCS": ucs}

for name, algorithm in algorithms.items():
    start_time = time.time()
    path = algorithm(problem)
    end_time = time.time()
    results[name] = {
        "Path Length": len(path),
        "Time Taken (s)": f"{(end_time - start_time):.6f}"
    }

results


