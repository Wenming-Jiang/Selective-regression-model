import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_testset(statfile, sita, stop_index, output_dir):
    stat = np.load(statfile)
    x = np.arange(0, 360, sita)
    for i, line in enumerate(stat):
        if i >= stop_index:
            break
        mean = line[-3]
        var = line[-2]
        real = line[-1]
        fig = plt.figure(figsize=(4, 3))
        plt.plot(x, line[:-3])
        plt.axhline(y=real, color='gray', linestyle='dashed')
        plt.axhline(y=mean, color='black', linestyle='solid')
        plt.title(str(i)+"th image, mean:"+str(mean//0.1/10)+"var: "+str(var//0.1/10))
        plt.xlabel("Rotated Sita")
        plt.ylabel("TC Intensity")
        
        plt.savefig(output_dir+str(i)+".png")


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--filename", default="./eval_output/ylist36_mean1_var1_dy1.npy")
    parser.add_argument("-sita", "--sita", default=10)
    parser.add_argument("-stop", "--stop_index", default=100)
    parser.add_argument("-o", "--output", default="./output_figure/")
    args = parser.parse_args()

    plot_testset(args.filename, args.sita, args.stop_index, args.output)

