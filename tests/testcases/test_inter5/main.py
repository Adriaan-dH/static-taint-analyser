def foo(z):
    sink(z)

def main(x):
    f = foo
    g = f
    g(x)
