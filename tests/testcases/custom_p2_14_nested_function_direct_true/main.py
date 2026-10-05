def main(x):
    def inner(v):
        return v

    y = inner(x)
    sink(y)
