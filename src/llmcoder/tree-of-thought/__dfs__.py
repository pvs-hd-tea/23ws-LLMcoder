# AITools first approach for tree of thought
class NodeCompletion:
    def __init__(self, state, score):
        # Each node represets a stae
        self.state = state
        self.children = []
        # Number of times the node has been visited with MCTS (simulated)
        self.visits = 0
        self.score = score

def dfs_backtrack(node):
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
