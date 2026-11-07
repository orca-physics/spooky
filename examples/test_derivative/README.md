# Spectral derivative check

`test_derivative.py` differentiates `cos(2 pi x / Lx)` spectrally on a 1D grid (`Lx = 22`, 128
points) and a 2D grid (`4 pi x 4 pi`, 64×32). It plots the numerical derivative on top of the
analytic one; the dashed curve should lie exactly on the solid one. It also shows the 2D
dealiasing mask (`grid.dealias_modes`, the modes zeroed by the 2/3 rule). It opens three figure
windows (`plt.show()`) and writes no files.

The quantitative checks of the grids are in `tests/test_grid.py`.
