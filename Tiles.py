from collections import deque
import heapq
import time
#TODO :: add dfs search algorthm to the mix
#TODO :: add manhaten distance to the heuristic search algorithm

# from symbol import return_stmt

def is_solvable(state):
    """Check if the 8-puzzle is solvable based on the number of inversions."""
    inversion_list = [tile for tile in state if tile != 0]  # Exclude 0
    inversions = 0
    for i in range(len(inversion_list)):
        for j in range(i + 1, len(inversion_list)):
            if inversion_list[i] > inversion_list[j]:
                inversions += 1

    # Return True if inversions are even
    return inversions % 2 == 0

#TODO :: create a class node that saves the 8 puzzle states
class Node:
    def __init__ (self,state ,parent=None,action=None ,path_cost=0,depth=0):
        self.state=state
        self.parent=parent
        self.action=action
        self.path_cost=path_cost
        self.depth=depth

    #return true if the node is the goal state
    def __eq__ (self ,other):
        return self.state == other.state

    #checks if the path cost of the node is less than the path of the other node
    def __lt__(self, other):
        return self.path_cost < other.path_cost

    # return the hash of the state
    def __hash__(self):
        return hash(tuple(self.state))

    def __str__(self):
        return f"Node(state={self.state}, action={self.action}, cost={self.path_cost})"

#TODO :: create a class problem that saves the initial state of the 8 puzzle
class Problem:
    def __init__(self,initial,goal):
        self.initial=initial
        self.goal=goal

    def actions(self, state):
        # Find the index of the empty space (0)
        empty_index = state.index(0)
        possible_actions = []

        # Define the possible moves based on the position of the empty space
        if empty_index % 3 > 0:  #can move left accepts positions 1, 2, 4, 5, 7, 8
            possible_actions.append('left')
        if empty_index % 3 < 2:  #can move right accepts positions 0, 1, 3, 4, 6, 7
            possible_actions.append('right')
        if empty_index // 3 > 0:  #can move up accepts positions 3, 4, 5, 6, 7, 8
            possible_actions.append('up')
        if empty_index // 3 < 2:  #can move down accepts positions 0, 1, 2, 3, 4, 5
            possible_actions.append('down')

        return possible_actions

    def result(self,state,action):
    # Define the result of the action taken
        new_state = state[:] # Copy the current state
        empty_index = new_state.index(0) # Find the index of the empty space (0)
        if action == 'left':
            new_state[empty_index], new_state[empty_index - 1] = new_state[empty_index - 1], new_state[empty_index]#creates new tupple with the new state
        elif action == 'right':
            new_state[empty_index], new_state[empty_index + 1] = new_state[empty_index + 1], new_state[empty_index]
        elif action == 'up':
            new_state[empty_index], new_state[empty_index - 3] = new_state[empty_index - 3], new_state[empty_index]
        elif action == 'down':
            new_state[empty_index], new_state[empty_index + 3] = new_state[empty_index + 3], new_state[empty_index]
        return new_state

    def is_goal(self, state):
        return state == self.goal

    def cost(self,state,action):
    # Define the cost of the action taken
        return 1
    def heuristic(self,state):
        pass

    def value(self,state):
        pass

#TODO :: create expand function that returns the children of the node
def expand(problem, node):
    for action in problem.actions(node.state):
        result = problem.result(node.state, action)
        cost = node.path_cost + problem.cost(node.state, action)
        yield Node(state=result, parent=node, action=action, path_cost=cost,depth=node.depth+1)




#print all the nodes of the path
def print_path(node):
    path = []
    while node:
        path.append(node)
        node = node.parent
    path.reverse()  # Reverse the list to get the correct order
    for t in path:
        print(t)

def get_solution_path(node):
    path = []
    while node.parent:
        empty_index = node.parent.state.index(0)
        moved_tile = node.state[empty_index]
        path.append(moved_tile)
        node = node.parent
    path.reverse()  # Reverse the list to get the correct order
    return ' -> '.join(map(str, path))

    # TODO :: get user input for the 9 nodes of the 8 puzzle 0 is the empty space and implament it as initial state
def get_user_input():
    while True:
        try:
            initial = list(map(int, input("Enter the numbers in the puzzle (0-8) separated by spaces: ").split()))
            if (len(initial) == 9 and
                set(initial) == set(range(9))):
                return initial
            else:
                print("Invalid input. Please enter exactly 9 unique numbers from 0 to 8.")
        except ValueError:
            print("Invalid input. Please enter numbers only.")



#TODO :: create a bfs search algorithm that returns the node that solves the 8 puzzle
def bfs(problem):
    frontier = deque([Node(problem.initial)])
    explored = set()
    nodes_created = 1  # Initial node is created
    nodes_expanded = 0  # Counter for nodes expanded
    while frontier:
        node = frontier.popleft()
        nodes_expanded += 1
        if problem.is_goal(node.state):
            print(f"Solution path: {get_solution_path(node)}")
            print(f"Number of nodes expanded: {nodes_expanded}")
            print(f"Number of nodes created: {nodes_created}")
            return node
        explored.add(tuple(node.state))
        for child in expand(problem, node):
            if tuple(child.state) not in explored and all(tuple(child.state) != tuple(n.state) for n in frontier):
                frontier.append(child)
                nodes_created += 1
    print(f"Number of nodes expanded: {nodes_expanded}")
    print(f"Number of nodes created: {nodes_created}")
    return None

#TODO:: create IDDFS algorithm that returns the node that solves the 8 puzzle
def ids(problem):
    max_depth = 100
    for depth in range(max_depth):
        result = dls(problem, depth)
        if result is not None:
            return result  # Found the solution or failure

def dls(problem, limit):
    frontier = [Node(problem.initial)]
    nodes_expanded = 0
    explored = set()
    result = None
    while frontier:
        node = frontier.pop()
        nodes_expanded += 1
        explored.add(tuple(node.state))

        if problem.is_goal(node.state):
            print(f"Solution path: {get_solution_path(node)}")
            print(f"Number of nodes expanded: {nodes_expanded}")
            return node
        if node.depth < limit:
            for child in expand(problem, node):
                if tuple(child.state) not in explored:
                    frontier.append(child)
        else:
            result = 'cutoff'
    return None if result == 'cutoff' else result


#TODO :: crate a GBFS algorithm that returns the node that solves the 8 puzzle
def misplaced_tiles(state, goal):
    return sum(1 for i in range(9) if state[i] != goal[i] and state[i] != 0)

def gbfs(problem):
    """Greedy Best-First Search using Misplaced Tiles heuristic."""
    frontier = []
    heapq.heappush(frontier, (misplaced_tiles(problem.initial, problem.goal), Node(problem.initial)))
    explored = set()
    nodes_created = 1
    nodes_expanded = 0

    while frontier:
        _, node = heapq.heappop(frontier)
        nodes_expanded += 1

        if problem.is_goal(node.state):
            print(f"Solution path: {get_solution_path(node)}")
            print(f"Number of nodes expanded: {nodes_expanded}")
            print(f"Number of nodes created: {nodes_created}")
            return node

        explored.add(tuple(node.state))

        for child in expand(problem, node):
            if tuple(child.state) not in explored and all(tuple(child.state) != tuple(n.state) for _, n in frontier):
                heapq.heappush(frontier, (misplaced_tiles(child.state, problem.goal), child))
                explored.add(tuple(child.state))
                nodes_created += 1

    print(f"Number of nodes expanded: {nodes_expanded}")
    print(f"Number of nodes created: {nodes_created}")
    return None


#TODO :: create a A* algorithm that returns the node that solves the 8 puzzle
def a_star(problem):
    """A* Search Algorithm."""
    # Use priority queue with (f(n), node)
    frontier = []
    heapq.heappush(frontier, (0 + misplaced_tiles(problem.initial, problem.goal), Node(problem.initial)))
    explored = {}  # Dictionary to store the lowest cost for each state
    nodes_created = 1
    nodes_expanded = 0

    while frontier:
        f_cost, node = heapq.heappop(frontier)
        nodes_expanded += 1

        if problem.is_goal(node.state):
            print(f"Solution path: {get_solution_path(node)}")
            print(f"Number of nodes expanded: {nodes_expanded}")
            print(f"Number of nodes created: {nodes_created}")
            return node

        # Only add to explored if the cost is lower or the state is new
        if tuple(node.state) not in explored or node.path_cost < explored[tuple(node.state)]:
            explored[tuple(node.state)] = node.path_cost

            for child in expand(problem, node):
                g_cost = child.path_cost
                h_cost = misplaced_tiles(child.state, problem.goal)
                f_cost = g_cost + h_cost

                if tuple(child.state) not in explored or g_cost < explored[tuple(child.state)]:
                    heapq.heappush(frontier, (f_cost, child))
                    nodes_created += 1

    print(f"No solution found.")
    print(f"Number of nodes expanded: {nodes_expanded}")
    print(f"Number of nodes created: {nodes_created}")
    return None

def print_separator():
    print("\n" + "-" * 100 + "\n")

def start_bfs():
    print_separator()
    # Measure the time taken to solve the problem using BFS
    start_time_bfs = time.time()
    print("Solving the 8-puzzle using BFS:\n")
    solution_node_bfs = bfs(problem)
    end_time_bfs = time.time()
    print(f"Time taken BFS: {end_time_bfs - start_time_bfs:.4f} seconds")
    print_separator()

def start_iidfs():
    start_time_iddfs = time.time()
    print("\n Solving the 8-puzzle using IDDFS:\n")
    solution_node_iddfs = ids(problem)
    end_time_iddfs = time.time()
    print(f"Time taken IDDFS: {end_time_iddfs - start_time_iddfs:.4f} seconds")
    print_separator()

def start_gbfs():
    start_time_gbfs = time.time()
    print("\nSolving the 8-puzzle using GBFS (Misplaced Tiles Heuristic):\n")
    solution_node_gbfs = gbfs(problem)
    end_time_gbfs = time.time()
    print(f"Time taken GBFS (Misplaced Tiles): {end_time_gbfs - start_time_gbfs:.4f} seconds")
    print_separator()

def start_astar():
    start_time_astar = time.time()
    print("\nSolving the 8-puzzle using A* (Misplaced Tiles Heuristic):\n")
    solution_node_astar = a_star(problem)
    end_time_astar = time.time()
    print(f"Time taken A*: {end_time_astar - start_time_astar:.4f} seconds")
    print_separator()

if __name__ == '__main__':
    # Get user input for the initial state
    initial_state = get_user_input()

    # Define the goal state
    goal_state = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    # Check if the puzzle is solvable
    if not is_solvable(initial_state):
        print("The given puzzle is not solvable.")
    else:
        # Create a problem instance
        problem = Problem(initial_state, goal_state)
        start_bfs()
        # Solve using iddfs
        start_gbfs()
        # Solve using GBFS with Misplaced Tiles
        start_iidfs()
        # Solve using A* Search
        start_astar()






