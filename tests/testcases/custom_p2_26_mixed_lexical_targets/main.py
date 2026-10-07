def identity(v):
    return v

def main(x, c):
    def clean(v):
        return 0

    if c:
        f = identity
    else:
        f = clean
    y = f(x)
    sink(y)
