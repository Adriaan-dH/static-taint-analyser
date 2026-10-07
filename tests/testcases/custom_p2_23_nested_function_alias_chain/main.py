def main(x):
    def inner(v):
        return v

    f = inner
    g = f
    y = g(x)
    sink(y)
