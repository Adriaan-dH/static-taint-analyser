def clean_a(v):
    return 0

def clean_b(v):
    return 1

def main(x, c):
    if c:
        f = clean_a
    else:
        f = clean_b
    y = f(x)
    sink(y)
