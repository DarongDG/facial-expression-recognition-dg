import csv
import os

import ipdb
import matplotlib.pyplot as plt
#

def plot_train_loss(train,val,save_path):
    plt.clf()
    plt.plot(*zip(*train), label = "train")
    plt.plot(*zip(*val), label = "val")
    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("Training loss")
    if train :
        list = [{"train_loss": train}, {"val_loss": val}]
        for i in list:
            with open(os.path.join(save_path, f'{next(iter(i.keys()))}.csv'), 'w') as out:
                csv_out = csv.writer(out)
                csv_out.writerow(['iteration', 'loss'])
                for row in next(iter(i.values())):
                    csv_out.writerow(row)


    plt.savefig(os.path.join(save_path, "loss vs iterations.png"))


def plot_acc(val_acc, save_path):
    plt.clf()
    plt.plot(*zip(*val_acc), label = "val")
    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel("accuracy")
    plt.title("Accuracy")
    if val_acc :
        val = {"val_acc": val_acc}
        with open(os.path.join(save_path, f'{next(iter(val.keys()))}.csv'), 'w') as out:
            csv_out = csv.writer(out)
            csv_out.writerow(['iteration', 'acc'])
            for row in next(iter(val.values())):
                csv_out.writerow(row)

    plt.savefig(os.path.join(save_path, "acc vs iterations.png"))


