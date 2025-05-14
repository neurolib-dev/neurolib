import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import logging
import tqdm
import networkx as nx
import nibabel as nib
import pathlib

from IPython.display import clear_output
from neurolib.utils import atlases


class Brainplot:
    def __init__(self, Cmat, data, nframes=None, dt=0.1, fps=25, labels=False, darkmode=True):
        self.sc = Cmat
        self.n = self.sc.shape[0]

        self.data = data
        self.darkmode = darkmode

        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.n))

        coords = {}
        atlas = atlases.AutomatedAnatomicalParcellation2()
        for i, c in enumerate(atlas.coords()):
            coords[i] = [c[0], c[1]]
        self.position = coords

        self.edge_threshold = 0.01

        self.fps = fps
        self.dt = dt

        nframes = nframes or int((data.shape[1] * self.dt / 1000) * self.fps)  # 20 fps default
        logging.info(f"Defaulting to {nframes} frames at {self.fps} fp/s")
        self.nframes = nframes

        self.frame_interval = self.data.shape[1] // self.nframes

        self.interval = int(self.frame_interval * self.dt)

        self.draw_labels = labels

        for t in range(self.n):
            # print t
            for s in range(t):
                # print( n, t, s)
                if self.sc[t, s] > self.edge_threshold:
                    # print( 'edge', t, s, self.sc[t,s])
                    self.G.add_edge(t, s)

        # node color map
        self.cmap = plt.get_cmap("plasma")  # mpl.cm.cool

        # default style

        self.imagealpha = 0.5

        self.edgecolor = "k"
        self.edgealpha = 0.8
        self.edgeweight = 1.0

        self.nodesize = 50
        self.nodealpha = 0.8
        self.vmin = 0
        self.vmax = 50

        self.lw = 0.5

        if self.darkmode:
            plt.style.use("dark")
            # let's choose a cyberpunk style for the dark theme
            self.edgecolor = "#37f522"
            self.edgeweight = 0.5
            self.edgealpha = 0.6

            self.nodesize = 40
            self.nodealpha = 0.8
            self.vmin = 0
            self.vmax = 30
            self.cmap = plt.get_cmap("cool")  # mpl.cm.cool

            self.imagealpha = 0.5

            self.lw = 1

            # fname = os.path.join("neurolib", "data", "resources", "clean_brain_white.png")
            fname = os.path.join(
                pathlib.Path(__file__).parent.absolute(), "..", "data", "resources", "clean_brain_white.png"
            )
        else:
            # plt.style.use("light")
            # fname = os.path.join("neurolib", "data", "resources", "clean_brain.png")
            fname = os.path.join(pathlib.Path(__file__).parent.absolute(), "..", "data", "resources", "clean_brain.png")

        print(fname)
        self.imgTopView = mpl.image.imread(fname)

        self.pbar = tqdm.tqdm(total=self.nframes)

    def update(self, i, ax, ax_rates=None, node_color=None, node_size=None, node_alpha=None, clear=True):
        frame = int(i * self.frame_interval)

        node_color = node_color or self.data[:, frame]
        node_size = node_size or self.nodesize
        node_alpha = node_alpha or self.nodealpha
        if clear:
            ax.cla()
        im = ax.imshow(self.imgTopView, alpha=self.imagealpha, origin="upper", extent=[40, 202, 28, 240])
        ns = nx.draw_networkx_nodes(
            self.G,
            pos=self.position,
            node_color=node_color,
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            node_size=node_size,
            alpha=node_alpha,
            ax=ax,
            edgecolors="k",
        )
        es = nx.draw_networkx_edges(
            self.G, pos=self.position, alpha=self.edgealpha, edge_color=self.edgecolor, ax=ax, width=self.edgeweight
        )

        labels = {}
        for ni in range(self.n):
            labels[ni] = str(ni)

        if self.draw_labels:
            nx.draw_networkx_labels(self.G, self.position, labels, font_size=8)

        ax.set_axis_off()
        ax.set_xlim(20, 222)
        ax.set_ylim(25, 245)

        # timeseries
        if ax_rates:
            ax_rates.cla()
            ax_rates.set_xticks([])
            ax_rates.set_yticks([])
            ax_rates.set_ylabel("Brain activity", fontsize=8)

            t = np.linspace(0, frame * self.dt, frame)
            ax_rates.plot(t, np.mean(self.data[:, :frame], axis=0).T, lw=self.lw)

            t_total = self.data.shape[1] * self.dt
            ax_rates.set_xlim(0, t_total)

        self.pbar.update(1)
        plt.tight_layout()
        if clear:
            clear_output(wait=True)


def plot_rates(model):
    plt.figure(figsize=(4, 1))
    plt_until = 10 * 1000
    plt.plot(model.t[model.t < plt_until], model.output[:, model.t < plt_until].T, lw=0.5)


def plot_brain(
    model, ds, color=None, size=None, title=None, cbar=True, cmap="RdBu", clim=None, cbarticks=None, cbarticklabels=None
):
    """Dump and easy wrapper around the brain plotting function.

    :param color: colors of nodes, defaults to None
    :type color: numpy.ndarray, optional
    :param size: size of the nodes, defaults to None
    :type size: numpy.ndarray, optional
    :raises ValueError: Raises error if node size is too big.
    """
    plot_data = model.output
    s = Brainplot(ds.Cmat, model.output, fps=10, darkmode=False)
    s.cmap = plt.get_cmap(cmap)

    if color is None:
        color = np.ones(ds.Cmat.shape[0])

    dpi = 300
    fig = plt.figure(dpi=dpi)
    ax = plt.gca()
    if title:
        ax.set_title(title, fontsize=26)

    if clim is None:
        s.vmin, s.vmax = np.min(color), np.max(color)
    else:
        s.vmin, s.vmax = clim[0], clim[1]

    if size is not None:
        node_size = size
    else:
        # some weird scaling of the color to a size
        def norm(what):
            what = np.asarray(what.copy())
            if np.min(what) < np.max(what):
                what -= np.min(what)
                what = np.divide(what, np.max(what))
            return what

        node_size = list(np.exp((norm(color) + 2) * 2))

    if isinstance(color, np.ndarray):
        color = list(color)
    if isinstance(node_size, np.ndarray):
        node_size = list(node_size)

    if np.max(node_size) > 2000:
        raise ValueError(f"node_size too big: {np.max(node_size)}")
    s.update(0, ax, node_color=color, node_size=node_size, clear=False)
    if cbar:
        # cbaxes = fig.add_axes([0.68, 0.1, 0.015, 0.7])
        cbaxes = fig.add_axes([0.75, 0.1, 0.015, 0.7])
        sm = plt.cm.ScalarMappable(cmap=s.cmap, norm=plt.Normalize(vmin=s.vmin, vmax=s.vmax))
        cbar = plt.colorbar(sm, cbaxes, ticks=cbarticks)
        cbar.ax.tick_params(labelsize=16)
        if cbarticklabels:
            cbar.ax.set_yticklabels(cbarticklabels)