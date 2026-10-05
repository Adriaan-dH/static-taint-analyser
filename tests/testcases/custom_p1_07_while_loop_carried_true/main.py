def main(x, c):
    z = 0
    y = 0
    while c:
        z = y
        y = x
    sink(z)
