# %%
#### imports
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.random import randint, random, uniform, normal, choice
    

# %%
#### constants
max_tree_size = 8
max_leaf_value = 100

operators = {
    "sum": lambda x,y : x+y,
    "sub": lambda x,y: x-y,
    "mul": lambda x,y: x*y,
    "div": lambda x,y: x/y if y != 0 else np.inf,
    "pow": lambda x,y: x**y,
    "sin": np.sin,
    "cos": np.cos,
    "id": lambda x: x
}

#### structures

@dataclass
class Node:
    value: tuple
    left: "Node" = None
    right: "Node" = None

class ValType(Enum):
    const = 0   # operand argument
    unary = 1   # one operand operator
    binary = 2  # two operand operator
    var = 3    # expression variable

#### methods

def get_operator_type(op):
    if op in ["sum", "sub", "mul", "div", "pow"]:
        return ValType.binary
    if op in ["sin", "cos", "id"]:
        return ValType.unary
    return None

def evaluate_tree(node: Node, args = None):
    if not node:
        return 0

    value_type, value = node.value
    if value_type == ValType.const:
        return value

    if value_type == ValType.var:
        if args is None:
            return 0
        if value >= len(args):
            return 0
        return args[value]

    # convert to proper operator name
    value = list(operators.keys())[int(value)]

    left = evaluate_tree(node.left)
    if value_type == ValType.unary:
        return operators[value](left)

    right = evaluate_tree(node.right)
    if value_type == ValType.binary:
        return operators[value](left, right)
    
    # or else, error
    print("invalid operator type!")


# %%
#### tree generation

def create_tree(tree_size, node_values, value_types):
    if tree_size == 0:
        return node_values, value_types, None
    
    node_vt, node_v = value_types[0], node_values[0]
    node_values, value_types = node_values[1:], value_types[1:]
    if node_vt == ValType.var:
        node_v = int(node_v)
    
    if tree_size == 1:
        return node_values, value_types, Node((node_vt, node_v))


    rts = (tree_size-1)//2      # right tree size
    lts = (tree_size-1) - rts   # left tree size
   
    node_values, value_types, left = create_tree(lts, node_values, value_types)
    node_values, value_types, right = create_tree(rts, node_values, value_types)
    
    return node_values, value_types, Node((node_vt, node_v), left, right)

def create_trees(n_trees, max_tree_size, max_leaf_value):
    tree_sizes = randint(1, max_tree_size+1, size= n_trees)     # size of each tree
    
    n_leaves = ((tree_sizes+np.ones(n_trees))//2).astype(int)   # leaf number of each tree
    leaf_vals = np.zeros(np.sum(n_leaves))                      # leaf values    
    
    is_var_p = np.random.normal(0.3, 0.1)
    lfs = np.sum(n_leaves)                                      # number of all leaf values
    is_var = choice([1, 0], size=lfs, p=[is_var_p, 1-is_var_p])
    
    n_var_leaves = np.count_nonzero(is_var)                     # number of all leaves containing variable
    n_vars = max(1, int(np.random.exponential(scale=2)))
    leaf_vals[is_var==1] = randint(0, n_vars, size= n_var_leaves)
    
    n_const_leaves = lfs - n_var_leaves
    leaf_vals[is_var==0] = uniform(low=0, high= max_leaf_value, size= n_const_leaves)              
    
    n_operators = tree_sizes - n_leaves                         # operator number of each tree
    op_vals = randint(0, len(operators.keys()), size= np.sum(n_operators)) # operator types


    # set all node vals
    node_vals = np.zeros(np.sum(tree_sizes))
    no = 0  # nodes offset
    lo = 0  # leaves offset
    oo = 0  # operators offset

    value_types = []
    leaf_types = np.where(is_var==0, ValType.const, ValType.var)
    op_keys = list(operators.keys())

    for i in range(n_trees):
        window = node_vals[no:no+tree_sizes[i]] 
        window[0:n_operators[i]] = op_vals[oo:oo+n_operators[i]]
        window[n_operators[i]:tree_sizes[i]] = leaf_vals[lo:lo+n_leaves[i]]

        op_strs = [op_keys[op_vals[i]] for i in range(oo, oo+n_operators[i])]
        value_types += list(map(lambda op: get_operator_type(op), op_strs))
        
        value_types += list(leaf_types[lo:lo+n_leaves[i]])

        no += tree_sizes[i]
        oo += n_operators[i]
        lo += n_leaves[i]
    

    # create trees
    trees = []
    for size in tree_sizes:
        node_vals, value_types, tree = create_tree(size, node_vals, value_types)
        trees.append(tree)

    # print(node_vals, value_types)

    return trees


# %%
#### debugging tools

def tree_to_expression(node: Node, use_common_var_names = True, limit_floating_point = True):
    if node is None:
        return "0"

    val_type, value = node.value
    if val_type == ValType.const:
        if not value:
            return "0"
        if limit_floating_point and isinstance(value, float):
            return "%.2f" % value
        return str(value)

    if val_type == ValType.var:
        if not use_common_var_names:
            return "x"+ str(int(value))
        
        i = int(value)
        if i < 13:
            return ['x','y','z', 'w', 't', 'u', 'v', 'm', 'n', 'p', 'q', 'r', 's'][i]
        
        if i < 26:
            return chr(97+i-13)

        return "x"+ str(i)
        

    # fetch name of operator
    value = str(list(operators.keys())[int(value)])

    if val_type == ValType.unary:
        return value + "("+ tree_to_expression(node.left) + ")"
    
    if val_type == ValType.binary:
        return (
            value + "(" + tree_to_expression(node.left) + ", "
            + tree_to_expression(node.right) + ")"
        )

    print("Unknown value type: " + str(val_type))


# %%

def simulate_one_round(population, ):
    trees = create_trees(population, max_tree_size, max_leaf_value)
    scores = evaluate_tree()

# %%
### DATA GENERATION

def generate_random_data():
    data_size = int(normal(1000, 200))
    max_x = normal(100,20)
    x = uniform(low= -max_x, high= max_x, size= data_size)

    max_value = normal(0, 1000)
    values = uniform(low= -max_value, high= max_value, size= data_size)

    return x, values

def generate_points(f, dsize, interval= (-100,100)):
# generates points for function f = f(x)
    start, end = interval
    x = uniform(low= start, high= end, size= dsize)
    return x,  np.array(list(map(f,x)))

### PERFORMANCE EVALUATION

def split_data(x,y, train_ratio):
    n = int(train_ratio * len(x))
    train_x, train_y, score_x, score_y = x[:n], y[:n], x[n:], y[n:]
    return train_x, train_y, score_x, score_y

# %%
### EVOLUTION
def replicate(node: Node):
    if node is None:
        return None
    return Node(node.value, replicate(node.left), replicate(node.right))

def crossover(xx: Node, xy: Node, p: float = 0.5):
    if xx is None or xy is None:
        return

    x = np.random.random()
    go_right = x > 0.5
    
    if x > p:
    # then don't crossever in root

        # increase the chance of next crossover
        p = 2 * p * np.random.rand()
        if go_right:
            crossover(xx.right, xy.right, p)
        else:
            crossover(xx.left, xx.left, p)
        return

    # time to crossover
    xx.value, xy.value = xy.value, xx.value
    if go_right:
        xx.right, xy.right = xy.right, xx.right  
    else:
        xx.left, xy.left = xy.left, xx.left


def random_operation():
    ops = list(operators.keys())
    x = randint(len(ops))
    op = ops[x]
    return get_operator_type(op), x

def random_of_type(val_type: ValType):
    if val_type is ValType.const:
        return np.random.random()* 100
    if val_type is ValType.var:
        return randint(26)
    if val_type in [ValType.binary, ValType.unary]:
        typ, val = random_operation()
        while typ != val_type:
            typ, val = random_operation()
        return val

def mutate(n: Node, p):
    if n is None:
        return False

    x = np.random.random()
    go_right = x > 0.5
    if x > p:
        # making sure mutation happens!
        p = 1.4 * p

        if go_right:
            return mutate(n.right, p)
        else:
            return mutate(n.left, p)
    
    typ, val = n.value

    y = np.random.random()
    if y<p:
    # mutate the typ itself
        typ = ValType(randint(len(ValType)))
        val = random_of_type(typ)
    else:
        if typ is ValType.const:
            val = val * np.random.normal(0, 3)
        elif typ is ValType.var:
            if go_right:
                val += 1
            else:
                if val > 0:
                    val -= 1
                else:
                    val += 2
        else:
        # if operation
            _typ = typ
            while _typ == typ:
                _typ, _val = random_operation()
            typ, val = _typ, _val

    n.value = typ, val
    return True


# %%
# x,y = generate_points(lambda x : x*2, 10)
# print(x)
# print(y)

# %%
#### tests

def test_mutation():
    # testintg mutation
    trees = create_trees(20, 9, 10)
    for t in trees:
        before = tree_to_expression(t)
        is_mutated = mutate(t, 0.3)
        if not is_mutated:
            continue
        after = tree_to_expression(t)
        
        print("from: " + before)
        print("to: " + after)
        # x = evaluate_tree(t, [1,1,1,1,1])
        # print("%.2f" % x)
        print()

def crossover_test():
    trees = create_trees(2, 20, 10)
    t,u = trees
    print(tree_to_expression(t), "\n",tree_to_expression(u))
    print("crossing over..")
    crossover(t,u, 0.5)
    print(tree_to_expression(t), "\n", tree_to_expression(u))

crossover_test()


# %%

# %%

# %%
