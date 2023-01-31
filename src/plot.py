import sys
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 2:
        print("pass file to plot as argument to the program")
        exit(-1)
    with open(sys.argv[1]) as f:
        vals = [tuple(map(int, line.strip().split()))
                for line in f.readlines()]
        wins = [a for (a, b, c) in vals]
        ties = [b for (a, b, c) in vals]
        losses = [c for (a, b, c) in vals]

        plt.clf()
        plt.cla()
        plt.plot([10 * i for i in range(len(vals))], wins)
        plt.xlabel("generation")
        plt.ylabel("win %")
        plt.savefig("wins.png")

        plt.clf()
        plt.cla()
        plt.plot([10 * i for i in range(len(vals))], ties)
        plt.xlabel("generation")
        plt.ylabel("tie %")
        plt.savefig("ties.png")

        plt.clf()
        plt.cla()
        plt.plot([10 * i for i in range(len(vals))], losses)
        plt.xlabel("generation")
        plt.ylabel("loss %")
        plt.savefig("losses.png")


if __name__ == "__main__":
    main()
