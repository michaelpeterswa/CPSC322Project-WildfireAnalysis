import matplotlib.pyplot as plt

def get_counts(col):
    unique_keys = []
    unique_keys_counts = []

    for item in col:
        if item not in unique_keys:
            unique_keys.append(item)
    for item in unique_keys:
        unique_keys_counts.append(col.count(item))
    return unique_keys, unique_keys_counts

def plot_hbar(x, y, xlabel, ylabel, title):
    x_pos = [i for i, _ in enumerate(x)]

    plt.figure(figsize=(5, 10))
    plt.barh(x_pos, y, height=0.9)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.yticks(x_pos, x, rotation='horizontal')
    plt.show()

def get_totals(col):
    return sum(col)

def get_max_min(lst):
    return max(lst), min(lst)

def plot_default_hist(lst, xlabel, ylabel, title, bins=10):
    plt.hist(lst, bins)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()

def plot_scatter(x, y, xlabel, ylabel, title):
    plt.scatter(x, y, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_scatter_regress(x, y, xlabel, ylabel, title, m, b, r, cov):
    title = title + ' (r: %f, cov: %f)' % (r,cov)
    plt.scatter(x, y, alpha=0.5)
    plt.plot([min(x), max(x)], [m * min(x) + b, m * max(x) + b], c="r", lw=5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()