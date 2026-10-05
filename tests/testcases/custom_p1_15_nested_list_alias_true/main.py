def main(x):
    inner = [0]
    outer = [inner]
    alias = outer[0]
    alias[0] = x
    sink(inner[0])
