def identity(v):
    return v

def choose(dummy):
    return identity

def main(x):
    f = choose(0)
    y = f(x)
    sink(y)
