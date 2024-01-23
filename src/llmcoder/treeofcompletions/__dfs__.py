# AITools first approach for tree of thought
class NodeCompletion:
    def __init__(self, state, score):
        # Each node represets a state
        self.state = state
        self.children = []
        self.score = score

def _generate_possible_completions(self, node.code):
     # Generate a code completion suggestion based on the current state (node)
    for child in node.childs:
        simulated_code = self._generate_code_completion(node.code)
        child.code = simulated_code

def _generate_code_completion(self, node.code):

# Depth-First algorithm + Backtrack -> Pruning
    # -consider the possibility the childs make the score of the path worse, backtrack
    # -new vesion: parent will be pruned from the tree
def dfs(self, node):
    # Base case: if the node has no children, return its information
    if not node.children:
        return [(node.code, node.score)]

    best_completions = []

    # Explore each child and backtrack
    for child in node.children:

        # Recursively backtrack on the child
        child_completions = dfs_backtrack(child)

        # Use lambda python algorithm for max
        best_child_completion = max(child_completions, key=lambda x: x[1])

        best_completions.append(best_child_completion)

    return best_completions
