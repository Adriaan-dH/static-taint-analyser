def identity(v):
    return v


def main(x):
    f = identity
    y = f(x)
    sink(y)
