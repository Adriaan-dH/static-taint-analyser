def identity(v):
    return v

def clean(v):
    return 0

def consume(v):
    sink(v)

def main(x, c):
    if c:
        f = identity
    else:
        f = clean
    consume(f(x))
