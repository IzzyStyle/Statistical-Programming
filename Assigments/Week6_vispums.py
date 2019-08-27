#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
  DataStats.py
  Jeff Holmes
  08/01/2019
  CPSC-51100-003 Summer 2019
  Week 6 Assignment

  References:
    https://pixelcalculator.com/
    https://stackoverflow.com/questions/47850202/plotting-a-histogram-on-a-log-scale-with-matplotlib
    https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0
    https://www.machinelearningplus.com/plots/matplotlib-histogram-python-examples/
    https://medium.com/better-programming/how-to-use-colormaps-with-matplotlib-to-create-colorful-plots-in-python-969b5a892f0c
    https://matplotlib.org/3.1.1/tutorials/colors/colormapnorms.html
    http://www.datasciencemadesimple.com/scaling-normalizing-column-pandas-dataframe-python/
"""

# imports

import numpy as np
import pandas as pd
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mpt
import seaborn as sns

from numpy import nan as NA

# constants
PLOT_FILE = 'pums.png'
AXES_FONT_SIZE = 12
TITLE_FONT_SIZE = 14
LEGEND_FONT_SIZE=12


def trim_all_columns(series):
    """
    Trim whitespace from ends of each value across all series (columns) in DataFrame

    :param DataFrame frame: dataframe to be trimmed
    :return:
    """
    trim_strings = lambda x: x.strip() if isinstance(x, str) else x

    return series.applymap(trim_strings)

def plot_pie_chart(ax, series):
    """
    Plot a pie chart
    :param Axes (AxesSubplot) ax: axes of subplot for plot
    :param Series series: data to plot
    :return: None
    """

    languages = ['English only', 'Spanish', 'Other Indo_European', 'Asian and Pacific Island Languages', 'Other']

    # portion covered by each label
    # data = [3, 7, 8, 6]

    # color for each label
    # colors = ['b', 'tab:orange', 'g', 'r', 'm']

    # plotting the pie chart
    # axis.pie(slices, labels=activities, colors=colors,
    #          startangle=90, shadow=True, explode=(0, 0, 0.1, 0),
    #          radius=1.2, autopct='%1.1f%%')

    # wedges, texts = ax.pie(series, labels=languages, startangle=-118, radius=1.0, labeldistance=None)
    wedges, texts = ax.pie(series, labels=['', '', '', '', ''], startangle=-118, radius=1.0)

    ax.legend(wedges, languages, loc=1, fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(0.04, 0.475, 0.5, 0.5))

    ax.set_ylabel('HHL', fontdict={'fontsize': AXES_FONT_SIZE})
    ax.yaxis.set_label_coords(-0.55, 0.475)

    ax.set_title('Household Languages', fontdict={'fontsize': TITLE_FONT_SIZE})


def plot_histogram(ax, series):
    """
    Plot a histogram
    :param Axes (AxesSubplot) ax: axes of subplot for plot
    :param Series series: data to plot
    :return: None
    """

    # Generate x values
    bins = np.logspace(np.log10(series.min()), np.log10(series.max()), 100)
    # bins = np.logspace(1, 7, 100)

    # ax.hist(data, bins=bins, density=True, color='green')
    # ax.hist(data, bins=bins, density=False, alpha=0.5, color='steelblue', edgecolor='none');
    # ax.hist(ages, bins, range, color='green', histtype='bar', rwidth=0.8)

    # Plot histogram with log scale on x-axis
    n, bins, _ = ax.hist(series, bins=bins, density=True, histtype='bar', color='green', alpha=0.5)

    # Add kde plot using pandas
    series.plot.kde(ax=ax, legend=False, linestyle='--', color='k')

    # series.hist(ax=ax, bins=100, grid=False)
    # df.column_name.plot.kde(ax=ax, legend=False, secondary_y=True)

    # Set axis labels
    ax.set_xlabel('Household income ($) - Log Scaled', fontdict={'fontsize': AXES_FONT_SIZE})
    ax.set_ylabel('Density', fontdict={'fontsize': AXES_FONT_SIZE})
    ax.set_xscale('log')

    # Set limits on axes
    ax.set_xlim(7.5, 1.75*10**7)
    ax.set_xticks([10, 10**2, 10**3, 10**4, 10**5, 10**6, 10**7])
    ax.set_ylim(0, 25*10**-6)
    ax.set_yticks([0, 5*10**-6, 10*10**-6, 15*10**-6, 20*10**-6])

    # Add kde plot using matplotlib
    # kde = stats.gaussian_kde(x)
    # plt, ax = plt.subplots(figsize=(8, 6))
    # ax.hist(x, normed=True, bins=bins, alpha=0.3)
    # ax.plot(xx, kde(xx))

    # Set plot title
    ax.set_title('Distribution of Household Income', fontdict={'fontsize': TITLE_FONT_SIZE})


def plot_bar_chart(ax, series):
    """
    Plot a pie chart
    :param Axes (AxesSubplot) ax: axes of subplot for plot
    :param Series series: data to plot
    :return: None
    """

    # labels for bars
    # tick_label = ['one', 'two', 'three', 'four', 'five']

    # Plot bar chart
    ax.bar(series.index, series.values, width=0.8, color='red')
    # ax.bar(left, height, tick_label=tick_label, width=0.8, color=['red', 'green'])

    # Set axes labels
    ax.set_xlabel('# of Vehicles', fontdict={'fontsize': AXES_FONT_SIZE})
    ax.set_ylabel('Thousands of Households', fontdict={'fontsize': AXES_FONT_SIZE})

    # Set limits on axes
    ax.set_ylim(20, 1900)
    ax.set_yticks([250, 500, 750, 1000, 1250, 1500, 1750])

    ax.margins(x=0)

    # Set plot title
    ax.set_title('Vehicles Available in Households', fontdict={'fontsize': TITLE_FONT_SIZE})


def plot_scatter(ax, frame):
    """
    Plot a pie chart
    :param Axes (AxesSubplot) ax: axes of subplot for plot
    :param DatFrame frame: data to plot
    :return: None
    """

    # Plotting points as a scatter plot
    # ax.scatter(data['VALP'], data['TAXP'], color='green', marker='o', s=30)

    # Create instance of class used to normalize color values
    norm = mpl.colors.Normalize(vmin=-1., vmax=1.)

    # The scatter function can only do one kind of marker at a time,
    # so we have to plot the different types separately.
    marker_types = frame['WGTP'].unique()
    for size in marker_types:
        d = frame[frame['WGTP'] == size]
        plt.scatter(x=d.VALP, y=d.TAXP_VALUE, s=d.WGTP, c=d.MRGP, norm=norm,
                    marker='o', alpha=0.3, cmap=plt.get_cmap('bwr'))

    # Display color bar
    cmin, cmax = frame.MRGP.min(), frame.MRGP.max()
    plt.clim(cmin, cmax)
    cbar = plt.colorbar(ticks=[1250, 2500, 3750, 5000], fraction=0.12, pad=0.05)
    # cbar = plt.colorbar(ticks=[1250, 2500, 3750, 5000], fraction=0.05, pad=0.05)
    # cbar.ax.set_yticklabels([1250, 2500, 3750, 5000])
    cbar.set_label('First Mortgage Payment (Monthly $)')

    ax.set_xticks([0, 2*10**5, 4*10**5, 6*10**5, 8*10**5, 10**6, 1.2*10**6])
    ax.set_xlim(0, 1.2*10**6)
    ax.set_ylim(0, 11000)

    # Set axis labels
    ax.set_xlabel('Property Values ($)', fontdict={'fontsize': AXES_FONT_SIZE})
    ax.set_ylabel('Taxes ($)', fontdict={'fontsize': AXES_FONT_SIZE})

    # Set plot title
    ax.set_title('Property Taxes vs. Property Values', fontdict={'fontsize': TITLE_FONT_SIZE})


if __name__ == "__main__":
    """
    Main program

    The program assumes the input/output files will be located 
    in the current directory where the python .py file is located.
    """

    # Print required output header
    print('CPSC-51100, Summer 2019')
    print('NAME: Jeff Holmes & Israel Nolazco')
    print('PROGRAMMING ASSIGNMENT #6\n')

    print('matplotlib: {}'.format(mpl.__version__))

    # These are the bins used for the scatter plot.
    # We created a CSV file in Sublime text editor using the text from the
    # PUMS data dictionary mapping the TAXP category values to dollar amounts.
    bins_taxp = [0.0, 1.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0,
                 700.0, 750.0, 800.0, 850.0, 900.0, 950.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0,
                 1700.0, 1800.0, 1900.0, 2000.0, 2100.0, 2200.0, 2300.0, 2400.0, 2500.0, 2600.0, 2700.0, 2800.0, 2900.0,
                 3000.0, 3100.0, 3200.0, 3300.0, 3400.0, 3500.0, 3600.0, 3700.0, 3800.0, 3900.0, 4000.0, 4100.0, 4200.0,
                 4300.0, 4400.0, 4500.0, 4600.0, 4700.0, 4800.0, 4900.0, 5000.0, 5500.0, 6000.0, 7000.0, 8000.0, 9000.0,
                 10000.0]


        # Create dataframe from csv file
    data = pd.read_csv('ss13hil.csv')

    # Simple method of creating figure and subplots
    # fig = plt.figure()
    # plot1 = fig.add_subplot(2, 2, 1)
    # plot2 = fig.add_subplot(2, 2, 2)
    # plot3 = fig.add_subplot(2, 2, 3)

    # Better method of creating figure and subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.set_size_inches(17, 9)

    # Pie Chart
    plot_pie_chart(ax1, data['HHL'].value_counts())

    # Histogram
    # Get data needed for histogram
    df_hist = pd.DataFrame(data, columns=['HINCP', 'ADJINC'])

    # We are dropping NaN values
    df_hist.dropna(inplace=True)

    # Convert column to int
    # df_hist['HINCP'] = df_hist['HINCP'].astype(int)

    # Apply adjustment factor ADJINC = 1.007549
    s_hist = df_hist['HINCP'] * 1.007549
    s_hist = s_hist[s_hist > 0]

#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
  DataStats.py
  Jeff Holmes
  08/01/2019
  CPSC-51100-003 Summer 2019
  Week 6 Assignment

  References:
    https://pixelcalculator.com/
    https://stackoverflow.com/questions/47850202/plotting-a-histogram-on-a-log-scale-with-matplotlib
    https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0
    https://www.machinelearningplus.com/plots/matplotlib-histogram-python-examples/
    https://medium.com/better-programming/how-to-use-colormaps-with-matplotlib-to-create-colorful-plots-in-python-969b5a892f0c
    https://matplotlib.org/3.1.1/tutorials/colors/colormapnorms.html
    http://www.datasciencemadesimple.com/scaling-normalizing-column-pandas-dataframe-python/
"""

# imports
import scipy.special as special
import numpy as np
import pandas as pd
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mpt
import seaborn as sns

from numpy import nan as NA

# constants
PLOT_FILE = 'pums.png'
AXES_FONT_SIZE = 12
TITLE_FONT_SIZE = 14
LEGEND_FONT_SIZE=12


def trim_all_columns(series):
    """
    Trim whitespace from ends of each value across all series (columns) in DataFrame

    :param DataFrame frame: dataframe to be trimmed
    :return:
    """
    trim_strings = lambda x: x.strip() if isinstance(x, str) else x

    return series.applymap(trim_strings)

def plot_pie_chart(ax, series):
    """
    Plot a pie chart
    :param Axes (AxesSubplot) ax: axes of subplot for plot
    :param Series series: data to plot
    :return: None
    """

    languages = ['English only', 'Spanish', 'Other Indo_European', 'Asian and Pacific Island Languages', 'Other']

    # portion covered by each label
    # data = [3, 7, 8, 6]

    # color for each label
    # colors = ['b', 'tab:orange', 'g', 'r', 'm']

    # plotting the pie chart
    # axis.pie(slices, labels=activities, colors=colors,
    #          startangle=90, shadow=True, explode=(0, 0, 0.1, 0),
    #          radius=1.2, autopct='%1.1f%%')

    # wedges, texts = ax.pie(series, labels=languages, startangle=-118, radius=1.0, labeldistance=None)
    wedges, texts = ax.pie(series, labels=['', '', '', '', ''], startangle=-118, radius=1.0)

    ax.legend(wedges, languages, loc=1, fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(0.04, 0.475, 0.5, 0.5))

    ax.set_ylabel('HHL', fontdict={'fontsize': AXES_FONT_SIZE})
    ax.yaxis.set_label_coords(-0.55, 0.475)

    ax.set_title('Household Languages', fontdict={'fontsize': TITLE_FONT_SIZE})


def plot_histogram(ax, series):
    """
    Plot a histogram
    :param Axes (AxesSubplot) ax: axes of subplot for plot
    :param Series series: data to plot
    :return: None
    """

    # Generate x values
    bins = np.logspace(np.log10(series.min()), np.log10(series.max()), 100)
    # bins = np.logspace(1, 7, 100)

    # ax.hist(data, bins=bins, density=True, color='green')
    # ax.hist(data, bins=bins, density=False, alpha=0.5, color='steelblue', edgecolor='none');
    # ax.hist(ages, bins, range, color='green', histtype='bar', rwidth=0.8)

    # Plot histogram with log scale on x-axis
    n, bins, _ = ax.hist(series, bins=bins, density=True, histtype='bar', color='green', alpha=0.5)

    # Add kde plot using pandas
    series.plot.kde(ax=ax, legend=False, linestyle='--', color='k')

    # series.hist(ax=ax, bins=100, grid=False)
    # df.column_name.plot.kde(ax=ax, legend=False, secondary_y=True)

    # Set axis labels
    ax.set_xlabel('Household income ($) - Log Scaled', fontdict={'fontsize': AXES_FONT_SIZE})
    ax.set_ylabel('Density', fontdict={'fontsize': AXES_FONT_SIZE})
    ax.set_xscale('log')

    # Set limits on axes
    ax.set_xlim(7.5, 1.75*10**7)
    ax.set_xticks([10, 10**2, 10**3, 10**4, 10**5, 10**6, 10**7])
    ax.set_ylim(0, 25*10**-6)
    ax.set_yticks([0, 5*10**-6, 10*10**-6, 15*10**-6, 20*10**-6])

    # Add kde plot using matplotlib
    # kde = stats.gaussian_kde(x)
    # plt, ax = plt.subplots(figsize=(8, 6))
    # ax.hist(x, normed=True, bins=bins, alpha=0.3)
    # ax.plot(xx, kde(xx))

    # Set plot title
    ax.set_title('Distribution of Household Income', fontdict={'fontsize': TITLE_FONT_SIZE})


def plot_bar_chart(ax, series):
    """
    Plot a pie chart
    :param Axes (AxesSubplot) ax: axes of subplot for plot
    :param Series series: data to plot
    :return: None
    """

    # labels for bars
    # tick_label = ['one', 'two', 'three', 'four', 'five']

    # Plot bar chart
    ax.bar(series.index, series.values, width=0.8, color='red')
    # ax.bar(left, height, tick_label=tick_label, width=0.8, color=['red', 'green'])

    # Set axes labels
    ax.set_xlabel('# of Vehicles', fontdict={'fontsize': AXES_FONT_SIZE})
    ax.set_ylabel('Thousands of Households', fontdict={'fontsize': AXES_FONT_SIZE})

    # Set limits on axes
    ax.set_ylim(20, 1900)
    ax.set_yticks([250, 500, 750, 1000, 1250, 1500, 1750])

    ax.margins(x=0)

    # Set plot title
    ax.set_title('Vehicles Available in Households', fontdict={'fontsize': TITLE_FONT_SIZE})


def plot_scatter(ax, frame):
    """
    Plot a pie chart
    :param Axes (AxesSubplot) ax: axes of subplot for plot
    :param DatFrame frame: data to plot
    :return: None
    """

    # Plotting points as a scatter plot
    # ax.scatter(data['VALP'], data['TAXP'], color='green', marker='o', s=30)

    # Create instance of class used to normalize color values
    norm = mpl.colors.Normalize(vmin=-1., vmax=1.)

    # The scatter function can only do one kind of marker at a time,
    # so we have to plot the different types separately.
    marker_types = frame['WGTP'].unique()
    for size in marker_types:
        d = frame[frame['WGTP'] == size]
        plt.scatter(x=d.VALP, y=d.TAXP_VALUE, s=d.WGTP, c=d.MRGP, norm=norm,
                    marker='o', alpha=0.3, cmap=plt.get_cmap('bwr'))

    # Display color bar
    cmin, cmax = frame.MRGP.min(), frame.MRGP.max()
    plt.clim(cmin, cmax)
    cbar = plt.colorbar(ticks=[1250, 2500, 3750, 5000], fraction=0.12, pad=0.05)
    # cbar = plt.colorbar(ticks=[1250, 2500, 3750, 5000], fraction=0.05, pad=0.05)
    # cbar.ax.set_yticklabels([1250, 2500, 3750, 5000])
    cbar.set_label('First Mortgage Payment (Monthly $)')

    ax.set_xticks([0, 2*10**5, 4*10**5, 6*10**5, 8*10**5, 10**6, 1.2*10**6])
    ax.set_xlim(0, 1.2*10**6)
    ax.set_ylim(0, 11000)

    # Set axis labels
    ax.set_xlabel('Property Values ($)', fontdict={'fontsize': AXES_FONT_SIZE})
    ax.set_ylabel('Taxes ($)', fontdict={'fontsize': AXES_FONT_SIZE})

    # Set plot title
    ax.set_title('Property Taxes vs. Property Values', fontdict={'fontsize': TITLE_FONT_SIZE})


if __name__ == "__main__":
    """
    Main program

    The program assumes the input/output files will be located 
    in the current directory where the python .py file is located.
    """

    # Print required output header
    print('CPSC-51100, Summer 2019')
    print('NAME: Jeff Holmes & Israel Nolazco')
    print('PROGRAMMING ASSIGNMENT #6\n')

    print('matplotlib: {}'.format(mpl.__version__))

    # These are the bins used for the scatter plot.
    # We created a CSV file in Sublime text editor using the text from the
    # PUMS data dictionary mapping the TAXP category values to dollar amounts.
    bins_taxp = [0.0, 1.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0,
                 700.0, 750.0, 800.0, 850.0, 900.0, 950.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0,
                 1700.0, 1800.0, 1900.0, 2000.0, 2100.0, 2200.0, 2300.0, 2400.0, 2500.0, 2600.0, 2700.0, 2800.0, 2900.0,
                 3000.0, 3100.0, 3200.0, 3300.0, 3400.0, 3500.0, 3600.0, 3700.0, 3800.0, 3900.0, 4000.0, 4100.0, 4200.0,
                 4300.0, 4400.0, 4500.0, 4600.0, 4700.0, 4800.0, 4900.0, 5000.0, 5500.0, 6000.0, 7000.0, 8000.0, 9000.0,
                 10000.0]


        # Create dataframe from csv file
    data = pd.read_csv('ss13hil.csv')

    # Simple method of creating figure and subplots
    # fig = plt.figure()
    # plot1 = fig.add_subplot(2, 2, 1)
    # plot2 = fig.add_subplot(2, 2, 2)
    # plot3 = fig.add_subplot(2, 2, 3)

    # Better method of creating figure and subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.set_size_inches(17, 9)

    # Pie Chart
    plot_pie_chart(ax1, data['HHL'].value_counts())

    # Histogram
    # Get data needed for histogram
    df_hist = pd.DataFrame(data, columns=['HINCP', 'ADJINC'])

    # We are dropping NaN values
    df_hist.dropna(inplace=True)

    # Convert column to int
    # df_hist['HINCP'] = df_hist['HINCP'].astype(int)

    # Apply adjustment factor ADJINC = 1.007549
    s_hist = df_hist['HINCP'] * 1.007549
    s_hist = s_hist[s_hist > 0]

    plot_histogram(ax2, s_hist)

    # Bar Chart
    # Get data for bar chart
    df_bar = pd.DataFrame(data, columns=['VEH', 'WGTP', 'NP'])
    df_bar['VEH_WGTP'] = df_bar.VEH * df_bar.WGTP

    # We are dropping VEH NaN values (there are none in WGTP)
    df_bar.dropna(inplace=True)

    # Convert column to int
    df_bar['VEH'] = df_bar['VEH'].astype(int)

    # Populate series with vehicle counts using WGTP value
    s_bar = pd.Series([])
    for x in range(7):
        s_bar[x] = df_bar['WGTP'][df_bar['VEH'] == x].sum()

    # Convert thousands of households
    s_bar = s_bar / 1000

    plot_bar_chart(ax3, s_bar)

    # Scatter Plot
    # Creat a series from the bins list to be used below
    s_taxp = pd.Series(bins_taxp)
    s_taxp.index += 1

    # Get data needed for scatter plot
    df_scatter = pd.DataFrame(data, columns=['TAXP', 'VALP', 'WGTP', 'MRGP'])

    # We are dropping nan values
    df_scatter.dropna(inplace=True)

    # Convert column to int
    df_scatter['TAXP'] = df_scatter['TAXP'].astype(int)

    # Copy column to be used below
    df_scatter['TAXP_VALUE'] = df_scatter['TAXP'].astype(int)

    # Convert column to int
    df_scatter['MRGP'] = df_scatter['MRGP'].astype(int)

    # Reindex the columns
    df_scatter = df_scatter.reindex(columns=['TAXP', 'TAXP_CAT', 'VALP', 'WGTP', 'MRGP'])

    # Normalize color values (0 - 255)
    norm = mpl.colors.Normalize(vmin=-1., vmax=1.)
    df_scatter['MRGP_NORM'] = df_scatter['MRGP'].apply(norm)

    # Convert column using lower bound of interval
    for index, row in df_scatter.iterrows():
        interval = s_taxp[row['TAXP']]
        df_scatter.at[index, 'TAXP_VALUE'] = interval

    plot_scatter(ax4, df_scatter)

    # Hide unused subplots
    # axes[1, 0].set_visible(False)
    # axes[1, 1].set_visible(False)

    plt.tight_layout(pad=1, h_pad=4, w_pad=3)

    # plt.subplots_adjust(left=0.1, right=0.95, top=0.8, bottom=0.2, wspace=0.2, hspace=0.2)
    # plt.subplots_adjust(left=0.125, right=0.9, top=0.8, bottom=0.2, wspace=0.2, hspace=0.2)

    box = mpt.Bbox.from_bounds(-1, -0.5, 19.0, 10.0)

    plt.savefig(PLOT_FILE, format='png', dpi=85, bbox_inches=box, pad_inches=1)
    # plt.savefig(PLOT_FILE, format='png', dpi=85, bbox_inches='tight', pad_inches=1)

    plt.show()