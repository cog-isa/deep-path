import pandas
import matplotlib.pyplot as plt


def basic_plot(title_data_tuples, out_file = None):
    fig, ax = plt.subplots()
    fig.set_size_inches((10, 6))
    for name, x, y in title_data_tuples:
        ax.plot(x, y, label = name)
    ax.legend()
    if not out_file is None:
        fig.savefig(out_file)
    return fig, ax


def basic_plot_df(raw_data, out_file = None):
    df = pandas.DataFrame(raw_data)
    return basic_plot(((col, df.index, df[col]) for col in df.columns),
                      out_file = out_file)
