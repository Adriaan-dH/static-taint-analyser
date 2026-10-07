def apply(f, v):
    return f(v)

def identity(v):
    return v

def main(x):
    y = apply(identity, x)
    sink(y)
