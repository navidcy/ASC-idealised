using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: xnode, ynode, znode
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.Models.HydrostaticFreeSurfaceModels: FFTImplicitFreeSurfaceSolver
using Printf

architecture = CPU()

save_fields_interval = 7days
stop_time = 2years
Δt₀ = 5minutes

filename = "asc_channel"

Lx, Ly, Lz = 500kilometers, 600kilometers, 3kilometers

grid = RectilinearGrid(architecture,
                       topology = (Periodic, Bounded, Bounded), 
                       size = (128, 128, 16),
                       x = (-Lx/2, Lx/2),
                       y = (-Ly/2, Ly/2),
                       z = (-Lz, 0),
                       halo = (3, 3, 3))

H_deep = H = grid.Lz
H_shelf = h = 500meters
width_shelf = 150kilometers

shelf(x, y) = -(H + h)/2 - (H - h)/2 * tanh(y / width_shelf)

# We can add a bump to break the homogeneity in zonal direction
bump_amplitude = 50
width_bump = 10kilometers

x_bump, y_bump = 0, 200kilometers
bump(x, y) = bump_amplitude * exp(-((x - x_bump)^2 + (y - y_bump)^2) / 2width_bump^2)

# We don't add a bump here because we add noise as an initial condition
bathymetry(x, y) = shelf(x, y)

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bathymetry))

@info "Built a grid: $grid."

## Check the biharmonic diffusivity used by Thompson and Stewart, 2014.
## Keeping these terms the default pending feedback from Andrew Thompson.

# Physics
Δx = grid.Lx / grid.Nx
κ₄h = Δx^4 / 1day

ν  = 12          # [m² s⁻¹]
νz = 3e-4        # [m² s⁻¹]
κz = 5e-6        # [m² s⁻¹]

vertical_diffusivities = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=νz, κ=κz)
horizontal_viscosity = HorizontalScalarDiffusivity(; ν)
horizontal_biharmonic = HorizontalScalarBiharmonicDiffusivity(ν=κ₄h, κ=κ₄h)
convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0,
                                                                convective_νz = 0.0)
#####
##### Boundary conditions
#####

α  = 2e-4     # [K⁻¹] thermal expansion coefficient 
g  = 9.8061   # [m s⁻²] gravitational constant
cᵖ = 3994.0   # [J K⁻¹] heat capacity
ρ  = 1024.0   # [kg m⁻³] reference density

polynya_width = 50kilometers
sponge_width = 100kilometers
parameters = (Ly = Ly,
              Lz = Lz,
              polynya_width = polynya_width,
              y_salt_shutoff = - (Ly/2 - polynya_width),  # shutoff location for salt flux [km]
              Qsalt = 2.5e-3,                             # salt input (into the domain) [gm⁻² s⁻¹]
              τ = 0.075 / ρ,                              # surface kinematic wind stress [m² s⁻²]
              Qᵇ = 10 / (ρ * cᵖ) * α * g,                 # buoyancy flux magnitude [m² s⁻³]    
              y_shutoff = 5/6 * Ly,                       # shutoff location for buoyancy flux [m]
	          μ = 1 / 30days,                             # bottom drag damping time-scale [s⁻¹]
              ΔT = 5,                                     # surface temperature gradient [K]
              ΔS = 0.5,                                   # surface salinity gradient [K]
              H = Lz,                                     # domain depth [m]
              h = 1000.0,                                 # exponential decay scale of stable stratification [m]
              y_sponge =  Ly/2-sponge_width,              # northern boundary of sponge layer [km]
              λT = 56.0days,                              # relaxation time scale for T, S  [s]
              λu = 26.0days,                              # relaxation time scale for u, v, and w [s]
	      )

@inline function u_stress(i, j, grid, clock, model_fields, p)
    y = ynode(Center(), j, grid)
    return ifelse(y > p.y_salt_shutoff, p.τ * sin(π * (y - p.y_salt_shutoff) / (p.Ly - p.polynya_width)), 0.0)
end

# Zonal wind stress
u_stress_bc = FluxBoundaryCondition(u_stress, discrete_form=true, parameters=parameters)

# Bottom drag
## Is this the same as having a linear drag coefficient? ##
@inline u_drag(i, j, grid, clock, model_fields, p) = @inbounds - p.μ * p.Lz * model_fields.u[i, j, 1] 
@inline v_drag(i, j, grid, clock, model_fields, p) = @inbounds - p.μ * p.Lz * model_fields.v[i, j, 1]

u_drag_bc = FluxBoundaryCondition(u_drag, discrete_form=true; parameters)
v_drag_bc = FluxBoundaryCondition(v_drag, discrete_form=true; parameters)

@inline u_immersed_drag(i, j, k, grid, clock, model_fields, p) = @inbounds - p.μ * p.Lz * model_fields.u[i, j, k] @inline v_immersed_drag(i, j, k, grid, clock, model_fields, p) = @inbounds - p.μ * p.Lz * model_fields.v[i, j, k]    
u_immersed_drag_bc = FluxBoundaryCondition(u_immersed_drag, discrete_form=true, parameters=parameters)
v_immersed_drag_bc = FluxBoundaryCondition(u_immersed_drag, discrete_form=true, parameters=parameters)

u_bcs = FieldBoundaryConditions(top = u_stress_bc, bottom = u_drag_bc, immersed = u_immersed_drag_bc)
v_bcs = FieldBoundaryConditions(bottom = v_drag_bc, immersed = v_immersed_drag_bc)

@inline function salf_flux(i, j, grid, clock, model_fields, p)
    y = ynode(Center(), j, grid)
    return ifelse(y < p.y_salt_shutoff, - p.Qsalt, 0.0)
end

salt_flux_bc = FluxBoundaryCondition(salf_flux, discrete_form=true, parameters=parameters)

S_bcs = FieldBoundaryConditions(top = salt_flux_bc)

@inline initial_temperature(y, z, p) = p.ΔT * (exp(z / p.h) - 1) + p.ΔT * y / p.Ly
@inline initial_salinity(y, z, p) = p.ΔS * (exp(z / p.h) - 1) + p.ΔS * y / p.Ly

@inline mask(y, p) = max(0.0, y - p.y_sponge) / (p.Ly/2 - p.y_sponge)

@inline function temperature_relaxation(i, j, k, grid, clock, model_fields, p)
    timescale = p.λT
    y = ynode(Center(), j, grid)
    z = znode(Center(), k, grid)
    target_T = initial_temperature(y, z, p)
    T = @inbounds model_fields.T[i, j, k]
    return -1 / timescale * mask(y, p) * (T - target_T)
end

@inline function salt_relaxation(i, j, k, grid, clock, model_fields, p)
    timescale = p.λT
    y = ynode(Center(), j, grid)
    z = znode(Center(), k, grid)
    target_S = initial_salinity(y, z, p)
    S = @inbounds model_fields.S[i, j, k]
    return -1 / timescale * mask(y, p) * (S - target_S)
end


@inline function u_relaxation(i, j, k, grid, clock, model_fields, p)
    timescale = p.λu
    y = ynode(Center(), j, grid)
    z = znode(Center(), k, grid)
    u = @inbounds model_fields.u[i, j, k]
    return - 1 / timescale * mask(y, p) * u
end

@inline function v_relaxation(i, j, k, grid, clock, model_fields, p)
    timescale = p.λu
    y = ynode(Face(), j, grid)
    z = znode(Center(), k, grid)
    v = @inbounds model_fields.v[i, j, k]
    return - 1 / timescale * mask(y, p) * v
end


#####
##### Coriolis
#####

const f₀ = -1.31e-4     # [s⁻¹]
const β =  1e-11    # [m⁻¹ s⁻¹]
coriolis = BetaPlane(; f₀, β)


#####
##### Forcing and initial condition
#####

temperature_forcing = Forcing(temperature_relaxation, discrete_form = true, parameters = parameters)
salt_forcing = Forcing(salinity_relaxation, discrete_form = true, parameters = parameters)
u_forcing = Forcing(u_relaxation, discrete_form = true, parameters = parameters)
v_forcing = Forcing(v_relaxation, discrete_form = true, parameters = parameters)

#####
##### Buoyancy model
#####

## We need the 25 term EOS from TS14 ##
buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion=2e-3, haline_contraction=5e-4))


#####
##### Model building
#####

@info "Building a model..."

fft_preconditioner = FFTImplicitFreeSurfaceSolver(grid.underlying_grid)
free_surface = ImplicitFreeSurface(solver_method=:PreconditionedConjugateGradient, preconditioner=fft_preconditioner)
# free_surface = ImplicitFreeSurface(solver_method=:HeptadiagonalIterativeSolver)

## I assume this is where the sponge is implemented? ##

model = HydrostaticFreeSurfaceModel(; grid,
                                    free_surface,
                                    coriolis,
                                    buoyancy,
                                    closure = (vertical_diffusivities,
                                               horizontal_viscosity,
                                               horizontal_biharmonic,
                                               convective_adjustment),
                                    tracers = (:T, :S, :c),
                                    momentum_advection = WENO5(),
                                    tracer_advection = WENO5(),
                                    boundary_conditions = (S=S_bcs, u=u_bcs, v=v_bcs),
                                    forcing = (; T = temperature_forcing, S = salinity_forcing, v = v_forcing, u = u_forcing)
                                    )

@info "Built $model."

#####
##### Initial conditions
#####

# resting initial condition
ε(σ) = σ * randn()

uᵢ(x, y, z) = ε(1e-8)
vᵢ(x, y, z) = ε(1e-8)
wᵢ(x, y, z) = ε(1e-8)

Tᵢ(x, y, z) = initial_temperature(y, z, parameters) + ε(1e-8)
Sᵢ(x, y, z) = initial_salinity(y, z, parameters) + ε(1e-8)

# horizontal and vertical width of passive tracer initial distribution
Δz = 100               # [m]
Δc = 100kilometers     # [m]
cᵢ(x, y, z) = exp(-y^2 / 2Δc^2) * exp(-(z + Lz/4)^2 / 2Δz^2)

set!(model, S=Sᵢ, T=Tᵢ, u=uᵢ, v=vᵢ, w=wᵢ, c=cᵢ)


#####
##### Simulation building
#####

simulation = Simulation(model, Δt=Δt₀, stop_time=stop_time)

# add timestep wizard callback
wizard = TimeStepWizard(cfl=0.2, max_change=1.1, max_Δt=20minutes)

simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(20))

# add progress callback
# Also, we add a callback to print a message about how the simulation is going,

using Printf

wall_clock = [time_ns()]

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, next Δt: %s\n",
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w),
            prettytime(sim.Δt))

    wall_clock[1] = time_ns()
    
    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(100))


#####
##### Diagnostics
#####

u, v, w = model.velocities
T, S, c = model.tracers.T, model.tracers.S, model.tracers.c

#=
ζ = Field(∂x(v) - ∂y(u))

B = Field(Average(b, dims=1))
U = Field(Average(u, dims=1))
V = Field(Average(v, dims=1))
W = Field(Average(w, dims=1))

b′ = b - B
v′ = v - V
w′ = w - W

v′b′ = Field(Average(v′ * b′, dims=1))
w′b′ = Field(Average(w′ * b′, dims=1))

outputs = (; b, ζ, u)

averaged_outputs = (; v′b′, w′b′, B, U)
=#

#####
##### Build checkpointer and output writer
#####

simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule = TimeInterval(1years),
                                                        prefix = filename,
                                                        overwrite_existing = true)

velocities_filename = joinpath(@__DIR__, "ASC_outputs", filename * "_velocities" * ".nc")
simulation.output_writers[:velocities] = NetCDFOutputWriter(model, (; u, v, w);
                                                            filename = velocities_filename,
                                                            schedule = TimeInterval(save_fields_interval))

tracers_filename = joinpath(@__DIR__, "ASC_outputs", filename * "_tracers" * ".nc")
simulation.output_writers[:tracers] = NetCDFOutputWriter(model, (; T, S, c);
                                                         filename = tracers_filename,
                                                         schedule = TimeInterval(save_fields_interval))
#=
slicers = (west = (1, :, :),
           east = (grid.Nx, :, :),
           south = (:, 1, :),
           north = (:, grid.Ny, :),
           bottom = (:, :, 1),
           top = (:, :, grid.Nz))

for side in keys(slicers)
    indices = slicers[side]

    simulation.output_writers[side] = JLD2OutputWriter(model, outputs;
                                                       filename = filename * "_$(side)_slice",
                                                       schedule = TimeInterval(save_fields_interval),
                                                       indices)
end

simulation.output_writers[:zonal] = JLD2OutputWriter(model, (b=B, u=U),#, v=V, w=W, vb=v′b′, wb=w′b′),
                                                     schedule = TimeInterval(save_fields_interval),
                                                     filename = filename * "_zonal_average",
                                                     overwrite_existing = true)

simulation.output_writers[:averages] = JLD2OutputWriter(model, averaged_outputs,
                                                        schedule = AveragedTimeInterval(1days, window=1days, stride=1),
                                                        filename = filename * "_averages",
                                                        verbose = true,
                                                        overwrite_existing = true)
=#

@info "Running the simulation..."

run!(simulation)

@info "Simulation completed in " * prettytime(simulation.run_wall_time)
