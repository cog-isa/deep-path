import pandas, itertools

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

LINES = '- -- -. : . , 0 v ^ < > 1 2 3 4 s p * h H + x D d | - _'.split(' ')
MARKERS = '. , o v ^ < > 1 2 3 4 8 s p * h H + x D d | _'.split(' ')
STYLES_GEN = itertools.cycle([''.join(s) for s in itertools.product(LINES, MARKERS)])


def basic_plot(title_data_tuples, out_file = None):
    fig, ax = plt.subplots()
    fig.set_size_inches((10, 6))
    for (name, x, y), style in itertools.izip(title_data_tuples,
                                              STYLES_GEN):
        ax.plot(x, y, style, label = name)
    ax.legend()
    if not out_file is None:
        fig.savefig(out_file)
    return fig, ax


def basic_plot_from_df(df, out_file = None):
    if sum(df.shape) > 0:
        df = pandas.get_dummies(df)
    return basic_plot(((col, df.index, df[col].values) for col in df.columns),
                      out_file = out_file)


def basic_plot_via_df(raw_data, out_file = None):
    basic_plot_from_df(pandas.DataFrame(raw_data),
                       out_file = out_file)
