import os
import threading
import sys
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import tqdm
import igraph

from .textreeCreate import create_tax_tree


def save_2d(otu, name, path):
    otu.dump(f"{path}/{name}.npy")


def find_root(G, child):
    parent = list(G.predecessors(child))
    if len(parent) == 0:
        print(f"found root: {child}")
        return child
    else:
        return find_root(G, parent[0])


def dfs_rec(tree, node, added, depth, m, N, counter):
    c = counter[0]
    added[node] = True
    neighbors = tree.neighbors(node)
    sum = 0
    num_of_sons = 0
    num_of_descendants = 0
    for neighbor in neighbors:
        if added[neighbor] == True:
            continue
        val, m, descendants, N, name = dfs_rec(tree, neighbor, added, depth + 1, m, N, counter)
        sum += val
        num_of_sons += 1
        num_of_descendants += descendants

    if num_of_sons == 0:
        value = tree.vs[node]["_nx_name"][1]  # the value
        m[depth, c] = value

        name = ";".join(tree.vs[node]["_nx_name"][0])
        N[depth, c] = name
        counter[0] += 1

        return value, m, 1, N, name

    avg = sum / num_of_sons
    name = ";".join(name.split(";")[:-1])
    for j in range(num_of_descendants):
        m[depth][c + j] = avg
        N[depth][c + j] = name

    return avg, m, num_of_descendants, N, name


def dfs_(tree, m, N):
    nv = tree.vcount()
    added = [False for v in range(nv)]
    counter = [0]
    _, m, _, N, _ = dfs_rec(tree, 0, added, 0, m, N, counter)
    return np.nan_to_num(m, 0), N


def get_map(tree, nettree):
    order, layers, ance = tree.bfs(0)
    height = len(layers) - 1

    leafs_num = len([i for i in nettree if len(nettree.succ[i]) == 0])

    m = np.zeros((height, leafs_num)) / 0
    str_size = max([len(str(i)) for i in nettree]) + 16
    N = np.zeros((height, leafs_num)).astype(f"U{str_size}")
    m, N = dfs_(tree, m, N)

    return m, N


def otu22d(df, save=False, with_names=False):
    M = []
    for subj in tqdm.tqdm(df.iloc, total=len(df)):
        nettree = create_tax_tree(subj)
        tree = igraph.Graph.from_networkx(nettree)
        m, N = get_map(tree, nettree)
        M.append(m)
        if save is not False:
            if with_names is not None:
                save_2d(N, "bact_names", save)
            save_2d(m, subj.name, save)

    if with_names:
        return np.array(M), N
    else:
        return np.array(M)


def rec(otu, bacteria_names_order, N=None):
    first_row = None
    for i in range(otu.shape[1]):
        if 2 < len(np.unique(otu[0, i, :])):
            first_row = i
            break
    if first_row is None:
        return
    X = otu[:, first_row, :]
    Y = linkage(X.T)

    sys.setrecursionlimit(13000)
    MB = 2 ** 20
    threading.stack_size(MB * 64)
    tpe = ThreadPoolExecutor(1)

    f = tpe.submit(dendrogram, Y, orientation='left', no_plot=True)
    Z1 = f.result()

    idx = Z1['leaves']
    otu[:, :, :] = otu[:, :, idx]
    if N is not None:
        N[:, :] = N[:, idx]

    bacteria_names_order = bacteria_names_order[idx]

    if first_row == (otu.shape[1] - 1):
        return

    unique_index = sorted(np.unique(otu[:, first_row, :][0], return_index=True)[1])

    S = []
    for i in range(len(unique_index) - 1):
        S.append((otu[:, first_row:, unique_index[i]:unique_index[i + 1]],
                  bacteria_names_order[unique_index[i]:unique_index[i + 1]],
                  None if N is None else N[first_row:, unique_index[i]:unique_index[i + 1]]))
    S.append((otu[:, first_row:, unique_index[-1]:], bacteria_names_order[unique_index[-1]:],
              None if N is None else N[first_row:, unique_index[-1]:]))

    for s in S:
        rec(s[0], s[1], s[2])


def dendogram_ordering(otu, df, folder, save=False, N=None, with_dend=True):
    names = np.array(list(df.columns))
    if with_dend == False:
        df = df
    else:
        rec(otu, names, N)
        df = df[names]

    if not os.path.exists(folder) and save is True:
        os.makedirs(folder)
    if save is not False:

        if with_dend:
            df.to_csv(
                f"{folder}/0_fixed_ordered_n_all_otu_sub_pca_log_tax_7.csv")
        else:
            df.to_csv(
                f"{folder}/0_fixed_ordered_n_all_otu_sub_pca_log_tax_7.csv")

    M = []
    if save is not False:
        if N is not None:
            save_2d(N, "bact_names", folder)
        for m, index in zip(otu, df.index):

            if with_dend:
                m.dump(f"{folder}/{index}.npy")
            else:
                m.dump(f"{folder}/{index}.npy")
            M.append(m)
            save_2d(m, index, folder)
    else:
        for m in otu:
            M.append(m)
    return np.array(M),N,df


def tree_to_newick(g, root=None):
    if root is None:
        roots = list(filter(lambda p: p[1] == 0, g.in_degree()))
        assert 1 == len(roots)
        root = roots[0][0]
    subgs = []
    for child in g[root]:
        if len(g[child]) > 0:
            subgs.append(tree_to_newick(g, root=child))
        else:
            subgs.append(str((child[0][-1], child[1])))
    return "(" + ','.join(subgs) + ")"


def micro2matrix(df, folder, save):
    otus2d, names = otu22d(df, save=False, with_names=True)
    array_of_imgs,bact_names, ordered_df = dendogram_ordering(otus2d, df, folder, save=save, N=names, with_dend=True)
    return array_of_imgs,bact_names, ordered_df
