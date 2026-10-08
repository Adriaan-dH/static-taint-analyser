def f(v):
    return v

def main(x):
    if x:
        def f(v):
            return 0
    else:
        pass

    result = f(x)
    sink(result)
