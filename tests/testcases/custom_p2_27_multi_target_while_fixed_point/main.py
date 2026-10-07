def identity(v):
    return v

def clean(v):
    return 0

def main(x, c):
    if c:
        f = identity
    else:
        f = clean
    y = 0
    z = 0
    while c:
        z = f(y)
        y = x
    sink(z)
