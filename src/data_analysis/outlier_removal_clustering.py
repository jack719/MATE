import numpy as np
import os
import pandas as pd
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def MoG_clustering(df_feat, relevant_cols, min_comp=21, max_comp=100, step=2, selection_method="BIC"):
    if selection_method not in ["BIC", "silhouette", "both"]:
        raise ValueError(f"selection_method {selection_method} not recognized, must be one of 'BIC', 'silhouette', 'both'")

    cols = relevant_cols["chem_val_cols"]
    feat_scaled = df_feat[cols].copy().fillna(0)
    feat_scaled = (feat_scaled - feat_scaled.mean()) / (feat_scaled.std() + 1e-7)

    # first downproject
    np.random.seed(42)
    tsne = TSNE(n_components=2, random_state=42)
    x_tsne = tsne.fit_transform(feat_scaled)

    np.random.seed(42)
    n_components = np.arange(min_comp, max_comp, step)
    bic = np.zeros(len(n_components))
    sil_scores = np.zeros(len(n_components))
    for i, n in enumerate(n_components):
        gmm = GaussianMixture(n_components=n)
        gmm.fit(x_tsne)
        bic[i] = gmm.bic(x_tsne)
        sil_scores[i] = silhouette_score(x_tsne, gmm.predict(x_tsne))

    bic_scaled = (bic - bic.min()) / (bic.max() - bic.min())
    sil_scores_scaled = (sil_scores - sil_scores.min()) / (sil_scores.max() - sil_scores.min())
    if selection_method == "BIC":
        best_n = n_components[np.argmin(bic_scaled)]
    elif selection_method == "silhouette":
        best_n = n_components[np.argmax(sil_scores_scaled)]
    elif selection_method == "both":
        best_n = n_components[np.argmin(bic_scaled + sil_scores_scaled)]
    else:
        raise ValueError(f"selection_method {selection_method} not recognized")

    print(f"best number of components: {best_n}")

    gmm = GaussianMixture(n_components=best_n)
    labels = gmm.fit_predict(x_tsne)
    print(f"number of clusters: {len(np.unique(labels))}")

    result = {
        "x_tsne": x_tsne,
        "labels": labels,
        "gmm": gmm,
        "best_n": best_n
    }
    
    return result

def plot_results(X, Y, means, covariances, title, idx_to_color_red=None, enable_plots=False, img_dir="./publication/images"):
    color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])
    fig, ax = plt.subplots(1,1,figsize=(15, 12))
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue

        plt.scatter(X[Y == i, 0], X[Y == i, 1], 4, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

    if idx_to_color_red is not None:
        plt.scatter(X[idx_to_color_red,0], X[idx_to_color_red,1], 0.5, c="red", marker="x")

    if title is not None:
        plt.title(title)
    plt.xlabel("t-SNE Dimension 1", fontsize=18)
    plt.ylabel("t-SNE Dimension 2", fontsize=18)
    plt.xticks(())
    plt.yticks(())

    if enable_plots:
        plt.show()
    else:
        img_path = os.path.join(img_dir, "mog_clusters.png")
        plt.savefig(img_path)
        plt.close()
        print(f"saved plot to {img_path}")


def plot_results_per_major_alloying_element(tsne_feat, tsne_targets, chem_val_cols, X, Y, means, covariances, title, idx_to_color_red=None, enable_plots=False, img_dir="./publication/images"):
    def _max_alloying_el(x):
        filtered = x.drop("Al").dropna()
        if not len(filtered):
            return "Al"

        names = filtered[filtered == filtered.max()].index
        if len(names) > 1:
            #return "/".join(names)
            return names[0]

        return names[0]

    # give each element (other than Al) a color
    els = [el for el in chem_val_cols if el != "Al"]
    colormap = plt.cm.get_cmap("tab20", len(els))

    color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])
    fig, ax = plt.subplots(1,1,figsize=(10, 10))
    actually_occured = set()
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        # instead of the color, use the colormap
        major_alloying_els = tsne_feat[chem_val_cols].loc[Y==i].apply(_max_alloying_el, axis=1).value_counts()
        major_el = major_alloying_els.index[0]
        actually_occured.add(major_el)
        color = colormap(els.index(major_el))


        #if np.isclose(color, np.array([0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1.0])).all(): # skip reds
        #    print("skipping red")
        #    color = colormap(len(els) + 2)


        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue

        plt.scatter(X[Y == i, 0], X[Y == i, 1], 4, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

        # for each cluster, get the mean x and y coordinates, and at these coordinates place a text which displays the mean and standard deviation of the YS values (Y)
        mean_x, mean_y = mean
        mean_ys = tsne_targets.loc[Y == i].mean() 
        std_ys = tsne_targets.loc[Y == i].std()
        #chem_means = tsne_feat[chem_val_cols].loc[Y==i].mean().dropna().drop("Al")
        #els = chem_means.index
        #stats = tsne_feat[chem_val_cols].loc[Y==i, els].apply(lambda x: f"{x.mean():.1f}±{x.std():.1f}")
        #str_ = ", ".join([f"{el}: {stat}" for el, stat in stats.items()])

        major_alloying_els_str = [f"{el} ({N})" for el, N in major_alloying_els.items()]
        str_ = ", ".join(major_alloying_els_str)

        if len(major_alloying_els) > 2:
            n = str(major_alloying_els.iloc[2:].sum())
            str_ = ", ".join(major_alloying_els_str[:2]) + f", ...({n})"

        #if len(major_alloying_els) > 1:
        #    plt.text(mean_x, mean_y, str_, fontsize=8, color="black")
        #plt.text(mean_x, mean_y, str_, fontsize=8, color="black")


    if idx_to_color_red is not None:
        plt.scatter(X[idx_to_color_red,0], X[idx_to_color_red,1], 0.5, c="red", marker="x")

    if title is not None:
        plt.title(title)
    plt.xlabel("t-SNE Axis 1", fontsize=12)
    plt.ylabel("t-SNE Axis 2", fontsize=12)
    plt.xticks(())
    plt.yticks(())

    def get_handles():
        handles = []
        for el in actually_occured:
            color = colormap(els.index(el))

            #if np.isclose(color, np.array([0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1.0])).all():
            #    print("skipping red")
            #    color = colormap(len(els) + 2)

            handles.append(mpl.patches.Patch(color=color, label=el))
        return handles

    #plt.legend(handles=[mpl.patches.Patch(color=colormap(els.index(el)), label=el) for el in actually_occured])
    plt.legend(handles=get_handles())

    if enable_plots:
        plt.show()
    else:
        img_path = os.path.join(img_dir, "mog_clusters.pdf")
        plt.savefig(img_path)
        plt.close()
        print(f"saved plot to {img_path}")


def remove_outliers_in_clusters(df_feat, targets, labels, relevant_cols, thr_rel=0.25, thr_abs=100):
    feat_removed = df_feat.copy()
    chem_val_cols = relevant_cols["chem_val_cols"]

    # now go through each cluster, compute the mean ys, remove samples which deviate by more than 25% from the mean or by more than 100 MPa, whichever is larger
    n_removed = 0
    stds_before = []
    stds_after = []
    for label in np.unique(labels):
        if label == -1:
            continue

        cluster_idx = labels == label
        ys = targets[cluster_idx]
        mean_ys = ys.mean()
        std_ys = ys.std()
        stds_before.append(std_ys)
        print(f"cluster {label} - mean ys: {mean_ys:.2f}, std ys: {std_ys:.2f}")

        # remove samples which deviate by more than 25% from the mean or by more than 100 MPa, whichever is larger
        remove_idx = (ys > mean_ys + thr_rel*mean_ys) | (ys < mean_ys - thr_rel*mean_ys) | (ys > mean_ys + thr_abs) | (ys < mean_ys - thr_abs)
        n_removed += remove_idx.sum()
        print(f"removing {remove_idx.sum()} samples")
        feat_removed.loc[targets[cluster_idx][remove_idx].index, chem_val_cols] = np.nan
        
        ys_after = targets[cluster_idx][~remove_idx]
        std_ys_after = ys_after.std()
        stds_after.append(std_ys_after)
        
    feat_removed = feat_removed[~feat_removed[chem_val_cols].isna().all(1)]
    targets_removed = targets.loc[feat_removed.index]
    print(f"removed {n_removed} samples")
    print(f"mean std per cluster before: {np.nanmean(stds_before)}")
    print(f"mean std per cluster after: {np.nanmean(stds_after)}")
        
    return feat_removed.reset_index(drop=True), targets_removed.reset_index(drop=True)

