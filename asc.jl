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

Lx, Ly = grid.Lx, grid.Ly

H_deep = H = Lz = grid.Lz
H_shelf = h = 500meters
width_shelf = 150kilometers

shelf(x, y) = -(H + h)/2 - (H - h)/2 * tanh(y / width_shelf)

bump_amplitude = 50
width_bump = 10kilometers

x_bump, y_bump = 0, 200kilometers
bump(x, y) = bump_amplitude * exp(-((x - x_bump)^2 + (y - y_bump)^2) / 2width_bump^2)

bathymetry(x, y) = shelf(x, y) + bump(x, y)

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bathymetry))

@info "Built a grid: $grid."

#free_surface = ImplicitFreeSurface()
# fft_preconditioner = FFTImplicitFreeSurfaceSolver(grid.underlying_grid)
# free_surface = ImplicitFreeSurface(solver_method=:PreconditionedConjugateGradient, preconditioner=fft_preconditioner)
free_surface = ImplicitFreeSurface(solver_method=:HeptadiagonalIterativeSolver)

# Physics
Δx = grid.Lx / grid.Nx
κ₄h = Δx^4 / 1day
κz = 1e-2

diffusive_closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=κz, κ=κz)
horizontal_closure = HorizontalScalarBiharmonicDiffusivity(ν=κ₄h, κ=κ₄h)
convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0,
                                                                convective_νz = 0.0)

#####
##### Boundary conditions
#####

α  = 2e-4     # [K⁻¹] thermal expansion coefficient 
g  = 9.8061   # [m s⁻²] gravitational constant
cᵖ = 3994.0   # [J K⁻¹] heat capacity
ρ  = 1024.0   # [kg m⁻³] reference density

parameters = (Ly = Ly,  
              Lz = Lz,    
              Qᵇ = 10 / (ρ * cᵖ) * α * g,          # buoyancy flux magnitude [m² s⁻³]    
              y_shutoff = 5/6 * Ly,                # shutoff location for buoyancy flux [m]
              τ = 0.2/ρ,                           # surface kinematic wind stress [m² s⁻²]
              μ = 1 / 30days,                      # bottom drag damping time-scale [s⁻¹]
              ΔB = 8 * α * g,                      # surface vertical buoyancy gradient [s⁻²]
              H = Lz,                              # domain depth [m]
              h = 1000.0,                          # exponential decay scale of stable stratification [m]
              y_sponge = 19/20 * Ly,               # southern boundary of sponge layer [m]
              λt = 7.0days                         # relaxation time scale [s]
)


@inline function buoyancy_flux(i, j, grid, clock, model_fields, p)
    y = ynode(Center(), j, grid)
    return ifelse(y < p.y_shutoff, p.Qᵇ * cos(3π * y / p.Ly), 0.0)
end

buoyancy_flux_bc = FluxBoundaryCondition(buoyancy_flux, discrete_form=true, parameters=parameters)

@inline function u_stress(i, j, grid, clock, model_fields, p)
    y = ynode(Center(), j, grid)
    return - p.τ * sin(π * y / p.Ly)
end

u_stress_bc = FluxBoundaryCondition(u_stress, discrete_form=true, parameters=parameters)

@inline u_drag(i, j, grid, clock, model_fields, p) = @inbounds - p.μ * p.Lz * model_fields.u[i, j, 1] 
@inline v_drag(i, j, grid, clock, model_fields, p) = @inbounds - p.μ * p.Lz * model_fields.v[i, j, 1]

u_drag_bc = FluxBoundaryCondition(u_drag, discrete_form=true, parameters=parameters)
v_drag_bc = FluxBoundaryCondition(v_drag, discrete_form=true, parameters=parameters)

b_bcs = FieldBoundaryConditions(top = buoyancy_flux_bc)

u_bcs = FieldBoundaryConditions(top = u_stress_bc, bottom = u_drag_bc)
v_bcs = FieldBoundaryConditions(bottom = v_drag_bc)

#####
##### Coriolis
#####

const f = -1e-4     # [s⁻¹]
const β =  1e-11    # [m⁻¹ s⁻¹]
coriolis = BetaPlane(f₀ = f, β = β)


#####
##### Model building
#####

@info "Building a model..."

model = HydrostaticFreeSurfaceModel(; grid, free_surface, coriolis,
                                    buoyancy = BuoyancyTracer(),
                                    closure = (diffusive_closure, horizontal_closure, convective_adjustment),
                                    tracers = (:b, :c),
                                    momentum_advection = WENO5(),
                                    tracer_advection = WENO5())

@info "Built $model."
