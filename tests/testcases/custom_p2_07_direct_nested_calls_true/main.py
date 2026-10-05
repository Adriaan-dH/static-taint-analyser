def second(v):
    return v


def first(v):
    return second(v)


def main(x):
    y = first(x)
    sink(y)
