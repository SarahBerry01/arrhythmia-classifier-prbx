import matplotlib.pyplot as plt
import numpy as np
import sys

def main(filename):
    with open(filename, 'r') as fp:
        lines = fp.readlines()
        acc = []
        for line in lines:
            if line.startswith("Accuracy :"):
                acc.append(float(line.split()[-1]))
    x = np.arange(0,662, 662/len(acc))
    print(x)
    fig, ax = plt.subplots(1)
    plt.plot(x, acc, color="blue", label="optimized")
    plt.grid(True, which="minor", axis="y", alpha=0.1)
    plt.grid(True, which="major", axis="y", alpha=1)
    plt.grid(True, which="major", axis="x", alpha=1)
    plt.grid(True, which="minor", axis="x", alpha=0.1)

    plt.minorticks_on()
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Time (s)")

    plt.show()

if __name__ == '__main__':
    main(sys.argv[1])

