import matplotlib.pyplot as plt
plt.switch_backend('agg')

# Note: Unless you share your rc file (or move this code into
# distributed source code), your display results will not be easily
# reproducible by other people!
from holoviews.plotting.mpl import MPLPlot
from holoviews import plotting       # This import defines package defaults
from holoviews import Options, Store
import holoviews as hv

options = Store.options(backend='matplotlib')
options.BlochSphere = Options('plot', show_grid=False, show_title=False)
options.BlochSphere = Options('plot', yaxis=None, xaxis=None)
options.Contours = Options('style', linewidth=0.5, color='k')
options.Contours = Options('plot', aspect='square')
options.Image = Options('style', cmap='gist_heat_r')
options.Image = Options('plot', title_format='{label}')
options.Path = Options('style', linewidth=0.5, color='k')
options.Path = Options('plot', aspect='square', title_format='{label}')
options.Scatter = Options('style', marker='o', s=100)
options.Overlay = Options('plot', show_legend=False, title_format='{label}')


MPLPlot.fig_rcparams['text.usetex'] = True
latex_packs = [r'\usepackage{amsmath}',
               r'\usepackage{amssymb}'
               r'\usepackage{bm}']
MPLPlot.fig_rcparams['text.latex.preamble'] = latex_packs
MPLPlot.fig_rcparams['font.size'] = 10
MPLPlot.fig_rcparams['axes.linewidth'] = 0.5

hv.plotting.mpl.SideHistogramPlot.show_xlabel=True
hv.plotting.mpl.SideHistogramPlot.offset = 0
hv.plotting.mpl.SideHistogramPlot.border_size = 0.3
