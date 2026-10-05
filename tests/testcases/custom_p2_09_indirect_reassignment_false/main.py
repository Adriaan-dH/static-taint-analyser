def tainted(v):
    return v


def clean(v):
    return 0


def main(x):
    f = tainted
    f = clean
    y = f(x)
    sink(y)
