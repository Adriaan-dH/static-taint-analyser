def identity(v):
    return v


def clean(v):
    return 0


def main(x, c):
    if c:
        f = identity
    else:
        f = clean
    y = f(x)
    sink(y)
