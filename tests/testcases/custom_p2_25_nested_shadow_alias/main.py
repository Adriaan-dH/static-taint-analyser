def f(v):
    return v

def main(x):
    def f(v):
        return 0

    g = f
    y = g(x)
    sink(y)
