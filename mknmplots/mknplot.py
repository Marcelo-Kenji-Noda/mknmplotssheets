import matplotlib.pyplot as plt
from matplotlib import rc_context
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.container import BarContainer

df = pd.read_csv("static/Pokemon.csv")

pixel = int
axsize = float

MPL_CONTEXT = {
    'axes.facecolor':'fffbe8',
    'axes.grid':True,
    'axes.labelsize' : 6,
    'xtick.major.size':1.0,
    'xtick.minor.size':0.5,
    'xtick.major.width':0.7,
    'xtick.minor.width':0.5,
    'xtick.labelsize':6,
    'ytick.major.size':1.0,
    'ytick.minor.size':0.5,
    'ytick.major.width':0.7,
    'ytick.minor.width':0.5,
    'ytick.labelsize':6,
    
    'grid.color':"#b0b0b0",
    'grid.linewidth': 0.5,
    'grid.alpha':0.2,

    'legend.frameon':True,
    'legend.facecolor':'fffbe8',
    'legend.edgecolor':'0A0A0A',
    'legend.fontsize':4,

    'figure.facecolor':'fffbe8',
    
    'lines.linewidth':0.5,
    'lines.markersize':3,
}

def fontsize_to_pixel_size(fontsize:float| int, dpi: float| int) -> float:
    """
    Converts fontsize into pixel size, according to the dpi
    """
    return (fontsize / 72) * dpi

def pixel_size_to_axes_size(ax: plt.Axes):
    """
    Converts pixel size into the axes size (height)
    """
    return 1 / ax.get_window_extent().height

def get_spacing_in_transaxes(spacing: pixel, ax:plt.Axes) -> axsize:
    """
    tile_spacing in pixels
    """
    return pixel_size_to_axes_size(ax) * spacing

def set_title_subtitle(ax: plt.Axes, title: str, subtitle:str = None, dpi:float| int = 150, title_spacing: pixel = 10, subtitle_spacing:pixel=7,  title_fontsize: int = 8, subtitle_fontisze: int = 6):
    """
    Add and format title and subtitle to ax
    
    Args:
        ax (plt.Axes): Axes to add title to
        title (str): Title content
        subtitle (str, optional): Subtitle content. Defaults to None.
        dpi (float | int, optional): figure dpi. Defaults to 150.
        title_spacing (pixel, optional): Title spacing from axes to text. Defaults to 10.
        subtitle_spacing (pixel, optional): Subtitle spacing from title to text. Defaults to 7.
        title_fontsize (int, optional): Title fontsize. Defaults to 8.
        subtitle_fontisze (int, optional): Subtitle fontsize. Defaults to 6.
    """
    title_spacing_ax: axsize = get_spacing_in_transaxes(spacing=title_spacing,ax = ax)
    subtitle_spacing_ax: axsize = get_spacing_in_transaxes(spacing=subtitle_spacing, ax=ax)
    title_height: axsize = get_spacing_in_transaxes(fontsize_to_pixel_size(subtitle_fontisze, dpi=dpi), ax=ax)
     
    if subtitle:
        ax.text(
            0, 
            1 + title_spacing_ax + title_height + subtitle_spacing_ax,
            title,
            transform=ax.transAxes,
            fontsize = title_fontsize,
            fontweight='bold'
            )
        ax.text(
            0,
            1 + title_spacing_ax,
            subtitle,
            transform=ax.transAxes,
            fontsize=subtitle_fontisze
            )
    else:
        ax.text(
            0,
            1+title_spacing_ax,
            title,
            transform=ax.transAxes, 
            fontsize = title_fontsize, 
            fontweight='bold'
        )
    return

def set_xlabel_ylabel(ax: plt.Axes, xlabel:str = None, ylabel:str = None, label_fontsize:int = 6):
    """

    Args:
        ax (plt.Axes): _description_
        xlabel (str, optional): _description_. Defaults to None.
        ylabel (str, optional): _description_. Defaults to None.
        label_fontsize (int, optional): _description_. Defaults to 6.

    Returns:
        _type_: _description_
    """
    if xlabel:
        ax.set_xlabel(xlabel)
        ax.xaxis.label.set_fontsize(label_fontsize)
    if ylabel:
        ax.set_ylabel(ylabel)
        ax.yaxis.label.set_fontsize(label_fontsize)
    return ax

def set_spines_off(ax: plt.Axes | list[plt.Axes], spines: list[str] = ["right", "top"]) -> plt.Axes:
    if type(ax) == list:
        for a in ax:
            for s in spines:
                a.spines[s].set_visible(False)
    else:
        for s in spines:
            ax.spines[s].set_visible(False)
    return ax

def set_text_to_hbars(ax: plt.Axes, spacing: float= 0.01)-> plt.Axes:
    x_min, x_max = ax.get_xlim()  # Get x-axis limits

    xsize = x_max - x_min
    
    for idx, p in enumerate(ax.patches):
        value = f'{p.get_width():.2f}'
        x = p.get_x() + p.get_width() + spacing * xsize
        y = p.get_y() + p.get_height() / 2 
        ax.text(x, y, value, ha='left', va='center', fontsize=6)
    return ax

def set_text_to_vbars(ax: plt.Axes, spacing: float= 0.01)-> plt.Axes:
    y_min, y_max = ax.get_ylim()  # Get x-axis limits

    ysize = y_max - y_min
    
    for idx, p in enumerate(ax.patches):
        value = f'{p.get_height():.2f}'
        x = p.get_x() + p.get_width() / 2
        y = p.get_y() + p.get_height() + spacing * ysize 
        ax.text(x, y, value, ha='center', va='bottom', fontsize=4) 
        
def get_fig_gs(nrows:int=1, ncols:int=1, dpi: int = 150) -> tuple[Figure, GridSpec]:
    fig = plt.figure(figsize=(7,4), dpi=200)
    gs = fig.add_gridspec(nrows, ncols)
    return fig, gs

def set_legend(ax: plt.Axes):
    legend = ax.legend()
    legend.get_frame().set_linewidth(0.5)  # Set border linewidth
    legend.get_frame().set_boxstyle("Square")  # Disable rounded corners
    return legend

def _add_text_to_center_of_bar(ax:plt.Axes, text:str, p: BarContainer):
    for _, p in enumerate(p.patches):
        x = p.get_x() + p.get_width() / 2
        y = p.get_y() + p.get_height() / 2
        ax.text(x, y, text, ha='center', va='center', fontsize=6, color='white')
    return ax

def _create_nan_legend(ax:plt.Axes,CONTEXT: dict):
    with rc_context(CONTEXT):
        legend= ax.legend(handles=[mpatches.Patch(color='grey', label='NaN')])
        legend.get_frame().set_linewidth(0.5)  # Set border linewidth
        legend.get_frame().set_boxstyle("Square")  # Disable rounded corners
    return legend

def plot_nan_dtype_dataframe(df: pd.DataFrame, CONTEXT:dict):
    TOTAL_ROWS: int = df.shape[0]
    
    dftypes: pd.DataFrame = df.dtypes.copy()
    
    yticklabels = []
    yticks = np.arange(0.5,dftypes.shape[0] + 0.5)
    
    bottom = 0
    counter = 0
    
    # Prep
    with rc_context(CONTEXT):
        fig, gs = get_fig_gs(1,6)
        ax = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1:])
    
    ## Plotting the data
    for df_dtype in dftypes.value_counts().index:
        text = str(df_dtype)
        height = dftypes[dftypes == df_dtype].shape[0]
        yticklabels.extend(list(dftypes[dftypes == df_dtype].index))
        
        p = ax.bar("Data types", height, 0.5, label=text, bottom=bottom, linewidth=0.5, linestyle='dashed', edgecolor='black')
        _add_text_to_center_of_bar(ax, text, p)

            
        color = p.patches[0].get_facecolor()
        bottom += height
        
        for col in dftypes[dftypes == df_dtype].index:
            nan_shape = df[col].isna().sum() * 100/TOTAL_ROWS
            ax2.barh(yticks[counter],100, color=color, height=1,linewidth=0.5, linestyle='dashed', edgecolor='black')
            ax2.barh(yticks[counter],nan_shape, height=1,color='grey',linewidth=0.5, linestyle='dashed', edgecolor='black')
            
            counter += 1

    _create_nan_legend(ax2, CONTEXT=CONTEXT)
    
    
    ## FIGURE CONFIGS
    set_title_subtitle(ax, title="Data types and NaN Columns", dpi=150, title_spacing=0)
    set_xlabel_ylabel(ax2,xlabel='Not NaN Cols percentage')

    ax2.set_ylim(ax.get_ylim())
    
    ax.tick_params(axis='both', which='both', length=0)
    ax2.tick_params(axis='y', which='both', length=0)
    
    ax.set_yticks(yticks)
    ax2.set_yticks(yticks)
    
    ax.set_yticklabels(yticklabels)
    ax2.set_yticklabels([])

    set_spines_off([ax,ax2],["right","top","left","bottom"])
    
    ax2.grid(False)
    ax.grid(False)
    plt.show()
    return

if __name__ == '__main__':
    
    collisions = pd.read_csv("https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv")
    
    plot_nan_dtype_dataframe(df, MPL_CONTEXT)
    