from llmcoder import LLMCoder
from dynaconf import Dynaconf

from llmcoder.eval.evaluate import Evaluation
from llmcoder.utils import get_config_dir


# AITools' second approach for Tree of Thought
"""
MCTS for MDP = (S=states, A=actions, P=transition probabilities, R=reward) in four steps: 
    1. selection: traverse the tre.
    2. expansion: if a node is not fully expanded, add actions and childs. Transition probabilities are not stored.
    3. simulation: generate + evaluate path. 
    4. backpropagation: update parent's scores.

"""
# Minimal score we try to reach
MIN_SCORE = 6
# Maximal depth of the tree
MAX_ITER = 10
# Degree of the tree
DEG = 3
class NodeCompletion:
    def __init__(self, state):
        # Each node represets a stae
        self.state = state
        self.children = []
        # Number of times the node has been visited with MCTS (simulated)
        self.visits = 0
        self.score = 0


def mcts(self, root, iterations):
    for _ in range(iterations):
        # Selection
        selected_node = self._select(root)

        # Expansion
        if not self._is_terminal(selected_node):
            self._expand(selected_node)

        # Simulation
        result = self._simulate(selected_node)

        # Backpropagation
        self._backpropagate(selected_node, result)

    # Choose the best action based on the root's children
    best_action = self._best_child(root).action
    return best_action


def _select(self, node):
    while len(node.children) < DEG and not self._is_terminal(node):
        node = self._select_child(node)
    return node


def _expand(self, node):
    # Generate possible code completions based on the current state (node)
    
    # Initialize child nodes

    # Create possible completions
    possible_completions = self._generate_possible_completions(node.code)

    # Create child nodes for each possible completion
    for completion in possible_completions:
        child_node = NodeCompletion(state=completion)
        node.children.append(child_node)


def _generate_possible_completions(self, node.code):
     # Generate a code completion suggestion based on the current state (node)
    for child in node.childs:
        simulated_code = self._generate_code_completion(node.code)
        child.code = simulated_code

def _generate_code_completion(self, code):
    return llmcoder.complete(code)


def _simulate(self, node):
    simulation_result = 0
    # Evaluate the quality of the simulated code
    # config = Dynaconf(settings_files=[os.path.join(get_config_dir(), args.config)])
    # eval = Evaluation(config)

    # simulation_result = evaluate.evaluate_code(node.code)

    return simulation_result


def _backpropagate(self, node, result):
    # Update the node's visit count and value, then propagate the result back up to the root
    while node is not None:
        node.visits += 1
        node.value += result
        node = node.parent

def _best_child(self, node):
    # Choose the child node with the highest value
    return max(node.children, key=lambda x: x.value / x.visits)

# Helper functions
def _is_terminal(self, node):
    # Check if the current state is a terminal state
    # return iter >= (MAX_ITER - 1) or node.score > min_score

    return node.state.is_terminal()

def _select_child(self, node):
    # Use a selection policy (UCB1) to choose a child node based on its value and exploration potential
    # UCB1 (Upper Confidence Bound 1) for balancing exploration and exploitation in MCTS
    exploration_constant = 1.0
    return max(node.children, key=lambda x: (x.value / x.visits) + exploration_constant * math.sqrt(math.log(node.visits) / x.visits))
