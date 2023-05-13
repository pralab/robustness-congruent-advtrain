import pickle
import numpy as np
import matplotlib.pyplot as plt

# def scatter_distances(d1, d2):
#
#
#     print("")

def main():
    file_path = 'results/distance_results/base_distances/base_distances.gz'
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # for k, v in data.items():
    #     print(f"{k}: {len(v) if isinstance(v, list) else v.shape}")



    # scatter_distances(distances[1], distances[4])

    old_ids = [1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 5, 5, 6]
    new_ids = [4, 7, 4, 5, 7, 2, 4, 5, 6, 7, 7, 4, 7, 7]

    nrows, ncols = 3, 5
    fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))

    for plot_i, (old_id, new_id) in enumerate(zip(old_ids, new_ids)):
        # old_id = 1
        # new_id = 4

        i, j = plot_i // ncols, plot_i % ncols
        try:
            ax = axs[i, j]
        except:
            print("")

        dold, dnew = data['distances'][old_id - 1], data['distances'][new_id - 1]
        m_old_name = data['model_names'][old_id - 1]
        m_new_name = data['model_names'][new_id - 1]

        ax.scatter(dold, dnew)
        ax.set_xlabel(m_old_name)
        ax.set_ylabel(m_new_name)

    fig.show()

    print("")

if __name__ == '__main__':
    main()