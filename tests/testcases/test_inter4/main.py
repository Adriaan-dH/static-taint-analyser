def select_first(a, b):
    return a

def main(x):
    # Pass tainted x as second argument; function returns the first argument (constant 0)
    y = select_first(0, x)
    sink(y)
