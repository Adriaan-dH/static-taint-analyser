def main(x, c):
    a = [x]
    b = [x]
    if c:
        target = a
    else:
        target = b
    target[0] = 0
    sink(a[0])
