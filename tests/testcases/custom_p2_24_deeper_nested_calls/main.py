def main(x):
    def outer(v):
        def inner(w):
            return w

        return inner(v)

    y = outer(x)
    sink(y)
