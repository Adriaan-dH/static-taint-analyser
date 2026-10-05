def mutate(a, v):
    a[0] = v


def main(x):
    a = [0]
    mutate(a, x)
    sink(a[0])
