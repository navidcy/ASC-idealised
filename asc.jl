using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.Models.HydrostaticFreeSurfaceModels: FFTImplicitFreeSurfaceSolver
using Printf

architecture = CPU()

grid = RectilinearGrid(architecture,
                       topology = (Periodic, Bounded, Bounded), 
                       size = (128, 128, 16),
                       x = (-500kilometers, 500kilometers),
                       y = (-500kilometers, 500kilometers),
                       z = (-3kilometers, 0),
                       halo = (3, 3, 3))

H_deep = H = grid.Lz
H_shelf = h = 500meters
width_shelf = 150kilometers

shelf(x, y) = -(H + h)/2 - (H - h)/2 * tanh(y / width_shelf)

bump_amplitude = 50
width_bump = 10kilometers

x_bump, y_bump = 0, 200kilometers
bump(x, y) = bump_amplitude * exp(-((x - x_bump)^2 + (y - y_bump)^2) / 2width_bump^2)

bathymetry(x, y) = shelf(x, y) + bump(x, y)

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bathymetry))
