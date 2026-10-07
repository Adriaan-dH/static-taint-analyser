def same(a):
    return a

def main(x):
    f = same
    a = [0]
    b = f(a)
    b[0] = x
    sink(a[0])
