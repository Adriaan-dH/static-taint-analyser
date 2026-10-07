def identity(v):
    return v

def main(x):
    f = identity
    a = f(x)
    b = f(0)
    sink(b)
