def clean_first(a):
    a[0] = 0


def main(x):
    a = [x]
    clean_first(a)
    sink(a[0])
