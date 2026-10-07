def clean_first(a):
    a[0] = 0
    return 0

def read_first(a):
    return a[0]

def main(x, c):
    a = [x]
    if c:
        f = clean_first
    else:
        f = read_first
    y = f(a)
    sink(y)
