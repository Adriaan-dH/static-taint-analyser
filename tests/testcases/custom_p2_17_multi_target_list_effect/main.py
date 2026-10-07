def taint_first(a, v):
    a[0] = v

def clean_first(a, v):
    a[0] = 0

def main(x, c):
    a = [0]
    if c:
        f = taint_first
    else:
        f = clean_first
    f(a, x)
    sink(a[0])
