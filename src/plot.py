import matplotlib.pyplot as plt


def plot_acc(ax, dist, acc, name):
    '''
    acc: (K, num)
    dist: (K, )
    name: (K, )
    '''
    assert(acc.shape[0] == len(name))

    for i in range(len(name)):
        ax.plot(dist, acc[i], label=name[i])

    ax.legend()

    ax.set_xlabel('Maximum allowed distance to GT (mm)', fontsize=16)
    ax.set_ylabel('Fraction of samples within distance', fontsize=16)


def plot_mean_err(ax, mean_err, name):
    '''
    mean_err: (K, )
    name: (K, )
    '''
    name = [x.replace('_', '\n') for x in name]
    ax.bar(name, mean_err)
    ax.set_xlabel('Joints in the MSRA Hand Dataset', fontsize=16)
    ax.set_ylabel('Mean error per joint in prediction', fontsize=16)
