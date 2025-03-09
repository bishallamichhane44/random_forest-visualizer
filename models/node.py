class Node:
    def __init__(self, depth=0):
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.depth = depth
        self.samples = None
        self.pos = None  # For visualization (x, y)
        self.radius = None
        
    def is_leaf_node(self):
        return self.value is not None