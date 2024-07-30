import matplotlib.pyplot as plt
from sklearn import manifold
from tsnecuda import TSNE

def do_tsne(X, y):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    # X_tsne = TSNE(n_components=2, perplexity=20, early_exaggeration=2, learning_rate=200).fit_transform(X)

    print("Origin data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure(figsize=(10, 10), dpi=500)

    # colors = ['darksalmon', 'goldenrod', 'olivedrab', 'darkturquoise', 'cadetblue', 'slategray', 'thistle']
    colors = ['#d474ac', '#14bccc', '#f47c24', '#d42424', '#8c4c3c', '#1c9c4c', '#bcbc34']

    for i in range(X_norm.shape[0]):
        plt.scatter(X_norm[i, 0], X_norm[i, 1], c=colors[y[i]])

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.show()

    plt.savefig('3.png', bbox_inches='tight', pad_inches=0.0)

    print("tsne done")
