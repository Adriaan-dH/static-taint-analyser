def f(v):
    return v

def main(x):
    a = f(x)

    def f(v):
        return 0

    b = f(x)
    sink(a)
