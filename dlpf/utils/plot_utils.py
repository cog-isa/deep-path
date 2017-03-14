import itertools

import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas

matplotlib.use('Agg')

LINES = '- -- -. : . , 0 v ^ < > 1 2 3 4 s p * h H + x D d | - _'.split(' ')
MARKERS = '. , o v ^ < > 1 2 3 4 8 s p * h H + x D d | _'.split(' ')
# STYLES_GEN = itertools.cycle([''.join(s) for s in itertools.product(LINES, MARKERS)])
STYLES_GEN = itertools.cycle(['-'])
MARKERS_GEN = itertools.cycle(MARKERS)


def basic_plot(title_data_tuples, out_file=None):
    fig, ax = plt.subplots()
    fig.set_size_inches((20, 10))
    for (name, x, y), style in itertools.izip(title_data_tuples,
                                              STYLES_GEN):
        ax.plot(x, y, style, label=name)
    ax.legend()
    if not out_file is None:
        fig.savefig(out_file)
    return fig, ax


def basic_plot_from_df(df, out_file=None, need_get_dummies=True, ignore=()):
    if need_get_dummies and df.shape[1] > 0:
        df = pandas.get_dummies(df)
    cols_to_ignore = set(ignore) & set(df.columns)
    if len(cols_to_ignore) > 0:
        df.drop(cols_to_ignore, axis=1, inplace=True)

    while df.shape[0] > 0:
        try:
            return basic_plot(((col, df.index, df[col].values) for col in df.columns),
                              out_file=out_file)
        except:
            df = df.sample(frac=0.8)


def basic_plot_from_df_rolling_mean(df, window=None, smooth_factor=50.0, out_file=None, ignore=()):
    if df.shape[1] > 0:
        df = pandas.get_dummies(df)
    if window is None:
        window = max(int(float(df.shape[0]) / smooth_factor), 10)
    df = pandas.rolling_mean(df, window)
    return basic_plot_from_df(df,
                              out_file=out_file,
                              need_get_dummies=False,
                              ignore=ignore)


def basic_plot_via_df(raw_data, out_file=None):
    return basic_plot_from_df(pandas.DataFrame(raw_data),
                              out_file=out_file)


def scatter_plot(config, invert_y=True, y_lim=None, x_lim=None, default_scale=100, grid=True, int_ticks=True,
                 offset=None, size=(10, 10), out_file=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(size)

    if y_lim:
        ax.set_ylim(y_lim)
    if x_lim:
        ax.set_xlim(x_lim)
    if grid:
        ax.grid(True)

    if invert_y:
        ax.set_ylim(ax.get_ylim()[::-1])

    if int_ticks:
        xlim = ax.get_xlim()
        ax.set_xticks(range(int(xlim[0]),
                            int(xlim[1]),
                            1 if xlim[1] > xlim[0] else -1))
        ylim = ax.get_ylim()
        ax.set_yticks(range(int(ylim[0]),
                            int(ylim[1]),
                            1 if ylim[1] > ylim[0] else -1))

    x_off, y_off = offset or (0, 0)
    for i, (series_info, marker) in enumerate(itertools.izip(config, itertools.cycle(MARKERS))):
        x, y = map(numpy.asarray, zip(*series_info['data']))
        ax.scatter(x + x_off, y + y_off,
                   c=series_info.get('color'),
                   s=series_info.get('scale', default_scale),
                   label=series_info.get('label', str(i)),
                   marker=series_info.get('marker', marker))

    ax.legend(loc='upper center',
              bbox_to_anchor=(1.05, 1),
              fancybox=True,
              shadow=True)

    if not out_file is None:
        fig.savefig(out_file)
    return fig, ax
