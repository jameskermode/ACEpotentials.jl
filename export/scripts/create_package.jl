#=
Create Redistributable ACE Potential Package
=============================================

Creates a pip-installable Python package from an ACE model by compiling
the native ACEpotentials evaluation path using Reactant.

Usage:
    julia +1.11 --project=.. scripts/create_package.jl --test-model --name mypot --output ./dist
    julia +1.11 --project=.. scripts/create_package.jl model.json --name mypot --output ./dist
=#

using Pkg
Pkg.activate(dirname(@__DIR__))

using Printf
using Random
using NPZ
using JSON3

using Reactant
using ACEpotentials
using AtomsBase
using AtomsBase: ChemicalSpecies
using AtomsCalculators
using Lux
using LuxCore
using StaticArrays
using Unitful
using DecoratedParticles: PState
using Base.Cartesian: @nexprs, @ntuple

M = ACEpotentials.Models
ETM = ACEpotentials.ETModels

import EquivariantTensors as ET

## ============================================================================
## Package Templates
## ============================================================================

const PYPROJECT_TEMPLATE = """
[project]
name = "{name}"
version = "{version}"
description = "ACE interatomic potential"
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.20",
    "ase>=3.22",
    "iree-base-runtime>=3.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{name}"]
"""

const INIT_TEMPLATE = """
\"""
{name} - ACE Interatomic Potential
\"""
from .calculator import ACECalculator
__version__ = "{version}"
__all__ = ["ACECalculator"]
"""

const CALCULATOR_TEMPLATE = """
\"""ASE Calculator interface for compiled ACE model.\"""

import numpy as np
from pathlib import Path
from ase.calculators.calculator import Calculator, all_changes

try:
    import iree.runtime as iree_rt
    HAS_IREE = True
except ImportError:
    HAS_IREE = False


class ACECalculator(Calculator):
    \"""ASE Calculator for compiled ACE potential.\"""

    implemented_properties = ['energy', 'forces']

    def __init__(self, device='cpu', **kwargs):
        super().__init__(**kwargs)

        if not HAS_IREE:
            raise ImportError("iree-runtime not installed")

        # Load compiled model
        models_dir = Path(__file__).parent / "models"
        vmfb_path = models_dir / f"model_{device}.vmfb"

        if not vmfb_path.exists():
            raise FileNotFoundError(f"Model not found: {vmfb_path}")

        # Load metadata
        with open(models_dir / "metadata.json") as f:
            import json
            self.metadata = json.load(f)

        self.rcut = self.metadata['rcut']

        # Initialize IREE runtime
        config = iree_rt.Config(device)
        self.module = iree_rt.load_vm_flatbuffer_file(str(vmfb_path), config)

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        # Build neighbor list and call IREE module
        # Implementation depends on VMFB interface
        raise NotImplementedError("Calculator implementation pending VMFB interface")
"""

## ============================================================================
## Model Creation
## ============================================================================

"""Create a minimal test ACE model."""
function create_test_model(; elements=(:Si,), order=2, max_level=6, rcut=5.5)
    @info "Creating test model..." elements order max_level

    rng = Random.MersenneTwister(42)

    rin0cuts = M._default_rin0cuts(elements)
    rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = rcut)).(rin0cuts)

    model = M.ace_model(;
        elements = elements,
        order = order,
        Ytype = :solid,
        level = M.TotalDegree(),
        max_level = max_level,
        maxl = 2,
        pair_maxn = max_level,
        rin0cuts = rin0cuts,
        pair_learnable = true,
        init_WB = :glorot_normal,
        init_Wpair = :glorot_normal
    )

    ps, st = Lux.setup(rng, model)
    calc = ETM.convert2et_full(model, ps, st; rng=rng)

    return calc, elements, rcut
end

"""Create a reference AtomsBase system for compilation."""
function create_reference_system(elements, rcut; n_atoms=8)
    # Create a simple cubic structure
    a = rcut / 2  # Lattice constant ensuring neighbors
    positions = SVector{3,Float64}[]

    for i in 0:1, j in 0:1, k in 0:1
        push!(positions, SVector(i*a, j*a, k*a))
    end

    box = (SVector(2a, 0.0, 0.0), SVector(0.0, 2a, 0.0), SVector(0.0, 0.0, 2a))
    Z = first(elements)

    atoms = [Atom(Z, pos * u"Å") for pos in positions]
    return FlexibleSystem(atoms; cell_vectors=box .* u"Å", periodicity=(true, true, true))
end

## ============================================================================
## Export Functions
## ============================================================================

"""
Extract arrays from ETGraph for tracing.

Returns: (rij, ii, jj, zi, zj, node_positions, node_species)
- rij: [n_edges, 3] relative position vectors
- ii, jj: [n_edges] neighbor indices (1-based)
- zi, zj: [n_edges] species indices for each edge
- node_positions: [n_atoms, 3] atomic positions
- node_species: [n_atoms] species indices
"""
function extract_graph_arrays(G::ET.ETGraph, species_list)
    n_edges = length(G.ii)
    n_atoms = length(G.node_data)

    # Species to index mapping
    species_to_idx = Dict(s => i for (i, s) in enumerate(species_list))

    # Edge arrays
    rij = zeros(Float32, n_edges, 3)
    ii = Vector{Int32}(G.ii)
    jj = Vector{Int32}(G.jj)
    zi = zeros(Int32, n_edges)
    zj = zeros(Int32, n_edges)

    for (e, edge) in enumerate(G.edge_data)
        rij[e, :] .= Float32.(edge.𝐫)
        zi[e] = species_to_idx[edge.z0]
        zj[e] = species_to_idx[edge.z1]
    end

    # Node arrays
    node_positions = zeros(Float32, n_atoms, 3)
    node_species = zeros(Int32, n_atoms)

    for (i, node) in enumerate(G.node_data)
        node_positions[i, :] .= Float32.(node.𝐫)
        node_species[i] = species_to_idx[node.z]
    end

    return (; rij, ii, jj, zi, zj, node_positions, node_species)
end

"""
Build ETGraph from arrays (inverse of extract_graph_arrays).
Used inside the traced function.
"""
function build_graph_from_arrays(rij, ii, jj, zi, zj, node_positions, node_species,
                                  species_list, graph_data)
    n_edges = length(ii)
    n_atoms = size(node_positions, 1)

    # Build edge_data
    edge_data = [
        PState(𝐫 = SVector{3,Float64}(rij[e, 1], rij[e, 2], rij[e, 3]),
               z0 = species_list[zi[e]],
               z1 = species_list[zj[e]],
               𝐒 = SVector{3,Int64}(0, 0, 0))  # Shift not needed for energy
        for e in 1:n_edges
    ]

    # Build node_data
    node_data = [
        PState(𝐫 = SVector{3,Float64}(node_positions[i, 1], node_positions[i, 2], node_positions[i, 3]),
               z = species_list[node_species[i]])
        for i in 1:n_atoms
    ]

    # Build first array (cumulative edge count per node)
    first = zeros(Int64, n_atoms + 1)
    first[1] = 1
    for e in 1:n_edges
        first[ii[e] + 1] += 1
    end
    cumsum!(first, first)

    # Compute maxneigs
    maxneigs = maximum(first[i+1] - first[i] for i in 1:n_atoms)

    return ET.ETGraph(Vector{Int64}(ii), Vector{Int64}(jj);
                      edge_data=edge_data, node_data=node_data,
                      graph_data=graph_data)
end

"""
    stacked_energy_from_graph(G, calc)

Compile-time unrolled energy evaluation for Reactant tracing.
Takes an ETGraph and evaluates all models in the StackedCalculator.

Uses @generated to unroll the loop over calculators at compile time,
ensuring Reactant sees a static call graph with no dynamic dispatch.
"""
@generated function stacked_energy_from_graph(
    G::ET.ETGraph,
    calc::ETM.StackedCalculator{N}
) where {N}
    quote
        @nexprs $N i -> begin
            Ei_i, _ = calc.calcs[i].model(G, calc.calcs[i].ps, calc.calcs[i].st)
            E_i = sum(Ei_i)
        end
        return sum(@ntuple $N E)
    end
end

"""
    export_mlir(calc, sys, output_dir; name="model")

Export model to StableHLO MLIR format.
Traces at the graph level with explicit array inputs matching Python/LAMMPS API.
"""
function export_mlir(calc::ETM.StackedCalculator, sys, output_dir::String;
                     name::String="model", elements=(:Si,))
    mkpath(output_dir)

    @info "Exporting to MLIR..." output_dir

    # Build reference graph
    rcut = maximum(c.rcut for c in calc.calcs if hasproperty(c, :rcut))
    G = ET.Atoms.interaction_graph(sys, rcut * u"Å")

    # Extract arrays
    species_list = [ChemicalSpecies(e) for e in elements]
    arrays = extract_graph_arrays(G, species_list)

    @info "Graph size" n_atoms=size(arrays.node_positions, 1) n_edges=length(arrays.ii)

    # Create energy function that takes arrays as input
    # This closure captures calc, species_list, and graph_data
    graph_data = G.graph_data

    function energy_from_arrays(rij, ii, jj, zi, zj, node_pos, node_spec)
        # Rebuild graph from arrays
        G_rebuilt = build_graph_from_arrays(rij, ii, jj, zi, zj, node_pos, node_spec,
                                            species_list, graph_data)
        # Use @generated function for compile-time unrolled evaluation
        return stacked_energy_from_graph(G_rebuilt, calc)
    end

    # Convert arrays to Reactant arrays for proper tracing
    rij_r = Reactant.to_rarray(arrays.rij)
    ii_r = Reactant.to_rarray(arrays.ii)
    jj_r = Reactant.to_rarray(arrays.jj)
    zi_r = Reactant.to_rarray(arrays.zi)
    zj_r = Reactant.to_rarray(arrays.zj)
    node_pos_r = Reactant.to_rarray(arrays.node_positions)
    node_spec_r = Reactant.to_rarray(arrays.node_species)

    # Export using Reactant's serialization
    Reactant.Serialization.export_to_enzymejax(
        energy_from_arrays,
        rij_r, ii_r, jj_r, zi_r, zj_r, node_pos_r, node_spec_r;
        output_dir=output_dir,
        function_name="$(name)_energy"
    )

    # Find generated MLIR file
    mlir_files = filter(f -> endswith(f, ".mlir"), readdir(output_dir))
    if !isempty(mlir_files)
        return joinpath(output_dir, first(mlir_files))
    end
    return nothing
end

"""
    compile_vmfb(mlir_path, vmfb_path; backend=:cpu)

Compile MLIR to IREE VMFB.
"""
function compile_vmfb(mlir_path::String, vmfb_path::String; backend::Symbol=:cpu)
    iree_compile = get(ENV, "IREE_COMPILE", nothing)
    if isnothing(iree_compile) || !isfile(iree_compile)
        iree_compile = Sys.which("iree-compile")
    end

    if isnothing(iree_compile)
        @warn "iree-compile not found - skipping VMFB"
        return nothing
    end

    iree_backend = backend == :cuda ? "cuda" : "llvm-cpu"

    try
        run(`$iree_compile
            --iree-input-type=stablehlo
            --iree-hal-target-backends=$iree_backend
            $mlir_path -o $vmfb_path`)
        @info "Generated: $vmfb_path"
        return vmfb_path
    catch e
        @warn "VMFB compilation failed" exception=e
        return nothing
    end
end

## ============================================================================
## Force Computation Support
## ============================================================================

"""
    compute_embedding_jacobians(calc, G, elements)

Compute embedding Jacobians dRnl/drij and dYlm/drij for force computation.

The chain rule for forces is:
    ∂E/∂rij = ∂E/∂Rnl @ dRnl/drij + ∂E/∂Ylm @ dYlm/drij

Returns:
- Rnl: (nedges, n_Rnl) - radial embeddings
- dRnl: (nedges, n_Rnl, 3) - Jacobian of Rnl w.r.t. rij
- Ylm: (nedges, n_Ylm) - angular embeddings
- dYlm: (nedges, n_Ylm, 3) - Jacobian of Ylm w.r.t. rij
"""
function compute_embedding_jacobians(calc::ETM.StackedCalculator, G::ET.ETGraph, elements)
    # Get ETACE model (index 3 in StackedCalculator)
    etace = calc.calcs[3]
    etace_model = etace.model
    etace_ps = etace.ps
    etace_st = etace.st

    # Use evaluate_ed to get embeddings and their Jacobians
    # This computes both P and dP/dX in forward mode
    (Rnl, dRnl), _ = ET.evaluate_ed(etace_model.rembed.layer, G.edge_data, etace_ps.rembed, etace_st.rembed)
    (Ylm, dYlm), _ = ET.evaluate_ed(etace_model.yembed.layer, G.edge_data, etace_ps.yembed, etace_st.yembed)

    return (; Rnl, dRnl, Ylm, dYlm)
end

"""
    export_energy_and_jacobians(calc, sys, output_dir; elements, name)

Export energy function (scatter-free) plus embedding Jacobian computation.

The exported VMFB computes: (Rnl_3d, Ylm_3d, species, W) → E
Forces are computed via chain rule in Python using Jacobians.
"""
function export_energy_and_jacobians(calc::ETM.StackedCalculator, sys, output_dir::String;
                                      name::String="model", elements=(:Si,))
    mkpath(output_dir)

    rcut = maximum(c.rcut for c in calc.calcs if hasproperty(c, :rcut))
    G = ET.Atoms.interaction_graph(sys, rcut * u"Å")

    # Get ETACE model
    etace = calc.calcs[3]
    etace_model = etace.model
    etace_ps = etace.ps
    etace_st = etace.st

    # Compute embeddings (2D edge format)
    Rnl, _ = etace_model.rembed(G, etace_ps.rembed, etace_st.rembed)
    Ylm, _ = etace_model.yembed(G, etace_ps.yembed, etace_st.yembed)

    # Species indices
    zlist = [ChemicalSpecies(e) for e in elements]
    node_species = Int64[ET.cat2idx(zlist, nd.z) for nd in G.node_data]
    W = etace_ps.readout.W

    # Reshape to 3D for Reactant (use reshape_embedding which works with Julia arrays)
    Rnl_3d = ET.reshape_embedding(Float32.(Rnl), G)
    Ylm_3d = ET.reshape_embedding(Float32.(Ylm), G)

    basis_st = etace_st.basis

    @info "Shapes" Rnl_3d=size(Rnl_3d) Ylm_3d=size(Ylm_3d) W=size(W)

    # Energy function (scatter-free in forward pass)
    function energy_from_3d(Rnl_in, Ylm_in, species_in, W_in)
        (BB,), _ = etace_model.basis((Rnl_in, Ylm_in), etace_ps.basis, basis_st)
        phi = ET._reactant_apply_selectlinl(etace_model.readout, BB, species_in, W_in)
        return sum(phi)
    end

    # Convert to Reactant arrays
    Rnl_r = Reactant.to_rarray(Rnl_3d)
    Ylm_r = Reactant.to_rarray(Ylm_3d)
    species_r = Reactant.to_rarray(node_species)
    W_r = Reactant.to_rarray(Float32.(W))

    @info "Exporting energy function to MLIR..."

    # Export energy function
    Reactant.Serialization.export_to_enzymejax(
        energy_from_3d, Rnl_r, Ylm_r, species_r, W_r;
        output_dir=output_dir,
        function_name="$(name)_energy"
    )

    # Find and verify MLIR file
    mlir_files = filter(f -> endswith(f, ".mlir"), readdir(output_dir))
    if !isempty(mlir_files)
        mlir_path = joinpath(output_dir, first(mlir_files))
        content = read(mlir_path, String)
        has_scatter = occursin("stablehlo.scatter", content)
        @info "MLIR exported" path=mlir_path lines=count('\n', content) has_scatter=has_scatter
        if has_scatter
            @warn "MLIR contains scatter operations - this is unexpected for energy-only"
        end
        return mlir_path
    end
    return nothing
end

## ============================================================================
## Export energy_from_rij (End-to-End VMFB for JAX.grad)
## ============================================================================

"""
    splinify_radial_embedding(rembed, ps_rembed, st_rembed; nspl=100, yrange=(-1.0, 1.0))

Convert a polynomial-basis radial embedding to spline-based for Reactant export.

Returns: (splined_layer, splined_ps, splined_st) or (nothing, nothing, nothing) if already splinified.
"""
function splinify_radial_embedding(rembed, ps_rembed, st_rembed; nspl=100, yrange=(-1.0, 1.0))
    layer = rembed.layer

    # Check if already splinified
    if layer isa ET.TransSelSplines
        @info "Radial embedding already uses TransSelSplines"
        return nothing, nothing, nothing
    elseif layer isa ET.EmbedDP && hasfield(typeof(layer), :basis) && layer.basis isa ET.TransSelSplines
        @info "Radial embedding already uses EmbedDP with TransSelSplines basis"
        return nothing, nothing, nothing
    end

    # Must be EmbedDP with polynomial basis - convert to splines
    if !(layer isa ET.EmbedDP)
        error("Cannot splinify radial embedding: expected EmbedDP, got $(typeof(layer))")
    end

    @info "Converting polynomial basis to splines" nspl=nspl yrange=yrange

    # Use EquivariantTensors trans_splines function
    # This converts EmbedDP with polynomial basis to TransSelSplines
    # Note: extract_envelope=false is used because ACEpotentials uses
    # WrappedFunction for envelope instead of TransformST
    splined_layer = ET.trans_splines(layer, ps_rembed, st_rembed;
                                      yrange=yrange, nspl=nspl,
                                      extract_envelope=false)

    # Create new EdgeEmbed with splined layer
    new_rembed = ET.EdgeEmbed(splined_layer)

    # Setup new parameters and state
    rng = Random.MersenneTwister(42)
    new_ps_rembed, new_st_rembed = LuxCore.setup(rng, new_rembed)

    @info "Splinification complete" layer_type=typeof(splined_layer)

    return new_rembed, new_ps_rembed, new_st_rembed
end

"""
    has_spline_radial(rembed)

Check if radial embedding is spline-based (required for vectorized Reactant export).
"""
function has_spline_radial(rembed)
    layer = rembed.layer
    # Case 1: Direct TransSelSplines
    if layer isa ET.TransSelSplines
        return true
    end
    # Case 2: EmbedDP with TransSelSplines basis
    if layer isa ET.EmbedDP && hasfield(typeof(layer), :basis) && layer.basis isa ET.TransSelSplines
        return true
    end
    return false
end

"""
    export_energy_from_rij(calc, sys, output_dir; name, elements, auto_splinify=true)

Export a single VMFB that computes energy directly from rij displacement vectors.

This enables JAX.grad(energy_from_rij)(rij) to compute pair forces directly,
without needing to implement embeddings in Python.

The graph structure (ii, jj, species) is captured at trace time.
Only rij varies at runtime, which enables efficient forward-mode differentiation
through the embeddings.

Uses vectorized embedding functions that avoid scalar indexing, enabling
full Reactant tracing without "Scalar indexing is disallowed" errors.

If `auto_splinify=true` (default), polynomial-basis models are automatically
converted to spline-based for Reactant compatibility.

Input:  rij (n_edges, 3) - displacement vectors
Output: E (scalar) - total potential energy
"""
function export_energy_from_rij(calc::ETM.StackedCalculator, sys, output_dir::String;
                                 name::String="energy_from_rij", elements=(:Si,),
                                 auto_splinify::Bool=true)
    mkpath(output_dir)

    rcut = maximum(c.rcut for c in calc.calcs if hasproperty(c, :rcut))
    G = ET.Atoms.interaction_graph(sys, rcut * u"Å")

    # Extract ETACE components (captured at trace time)
    etace = calc.calcs[3]
    etace_model = etace.model
    etace_ps = etace.ps
    etace_st = etace.st

    species_list = [ChemicalSpecies(e) for e in elements]
    zlist = [ChemicalSpecies(e) for e in elements]
    node_species = Int64[ET.cat2idx(zlist, nd.z) for nd in G.node_data]
    W = Float32.(etace_ps.readout.W)
    basis_st = etace_st.basis

    # Extract reference arrays
    arrays = extract_graph_arrays(G, species_list)
    n_edges = length(G.ii)

    @info "Exporting energy_from_rij" n_edges=n_edges n_atoms=length(G.node_data)

    # Check if radial embedding needs splinification
    rembed = etace_model.rembed
    ps_rembed = etace_ps.rembed
    st_rembed = etace_st.rembed

    if !has_spline_radial(rembed)
        if auto_splinify
            @info "Model uses polynomial basis - auto-converting to splines for Reactant export"
            new_rembed, new_ps_rembed, new_st_rembed = splinify_radial_embedding(
                rembed, ps_rembed, st_rembed; nspl=100, yrange=(-1.0, 1.0))
            if new_rembed !== nothing
                rembed = new_rembed
                ps_rembed = new_ps_rembed
                st_rembed = new_st_rembed
            end
        else
            error("""
            Vectorized export requires spline-based radial embedding.

            The model's radial embedding is: $(typeof(etace_model.rembed.layer))

            Options:
            1. Use auto_splinify=true (default) to auto-convert
            2. Manually convert to splines before export:
                using EquivariantTensors: trans_splines, EdgeEmbed
                spline_layer = trans_splines(etace_model.rembed.layer, etace_ps.rembed, etace_st.rembed)
                new_rembed = EdgeEmbed(spline_layer)
            3. Use export_energy_and_jacobians for non-splinified models
            """)
        end
    end

    # Capture graph topology for vectorized evaluation
    # These are constant during tracing (not TracedRArrays)
    first_arr_captured = Int64.(copy(G.first))
    maxn_captured = G.maxneigs
    nn_captured = ET.nnodes(G)
    zi_captured = Int64.(arrays.zi)
    zj_captured = Int64.(arrays.zj)
    node_species_captured = Int64.(arrays.node_species)

    # Get the ReactantExt module for vectorized functions
    ReactantExt = Base.get_extension(ET, :ReactantExt)

    # Extract spline parameters using ReactantExt helper
    radial_params = ReactantExt.extract_radial_params(rembed, st_rembed)
    trans_params = radial_params.trans_params
    trans_zlist = radial_params.zlist
    spline_st = radial_params.spline_st
    n_basis = radial_params.n_basis

    # Angular embedding parameters (extract maxl)
    angular_layer = etace_model.yembed.layer
    if hasfield(typeof(angular_layer), :basis) && hasfield(typeof(angular_layer.basis), :maxL)
        maxl_captured = angular_layer.basis.maxL
    else
        # Try to infer from output dimension
        # n_ylm = (maxl+1)^2
        test_rij = zeros(Float32, 1, 3)
        test_rij[1, :] = [1.0, 0.0, 0.0]
        test_Ylm = ReactantExt.reactant_apply_embeddp_angular(test_rij, 2)
        n_ylm = size(test_Ylm, 2)
        maxl_captured = Int(sqrt(n_ylm)) - 1
        @info "Inferred maxl from spherical harmonics output" maxl=maxl_captured n_ylm=n_ylm
    end

    # Define energy_from_rij function using vectorized paths
    # Graph structure is fixed at trace time; only rij varies
    function energy_from_rij_fn(rij)
        # Compute radial embeddings using vectorized path (no scalar indexing)
        Rnl_3d_raw = ReactantExt.reactant_edge_embed_radial(
            rij, zi_captured, zj_captured,
            first_arr_captured, maxn_captured, nn_captured, n_edges,
            trans_params, trans_zlist, spline_st, nothing, n_basis
        )

        # Compute angular embeddings using vectorized path
        Ylm_3d = ReactantExt.reactant_edge_embed_angular(
            rij, first_arr_captured, maxn_captured, nn_captured, n_edges,
            maxl_captured
        )

        # Ensure type consistency - splines may return Float64, Ylm is Float32
        # ACE basis requires matching element types
        Rnl_3d = Float32.(Rnl_3d_raw)

        # ACE basis + readout (already Reactant-compatible with 3D inputs)
        (BB,), _ = etace_model.basis((Rnl_3d, Ylm_3d), etace_ps.basis, basis_st)
        phi = ET._reactant_apply_selectlinl(etace_model.readout, BB, node_species_captured, W)

        return sum(phi)
    end

    # Convert rij to Reactant array
    rij_r = Reactant.to_rarray(arrays.rij)

    @info "Exporting energy_from_rij to MLIR..." input_shape=size(arrays.rij)

    try
        Reactant.Serialization.export_to_enzymejax(
            energy_from_rij_fn, rij_r;
            output_dir=output_dir, function_name=name
        )
    catch e
        @warn "export_to_enzymejax failed" exception=e
        return nothing
    end

    # Return path to MLIR file
    mlir_files = filter(f -> endswith(f, ".mlir"), readdir(output_dir))
    if isempty(mlir_files)
        @warn "No MLIR files generated"
        return nothing
    end

    mlir_path = joinpath(output_dir, first(mlir_files))

    # Analyze MLIR for scatter operations
    content = read(mlir_path, String)
    has_scatter = occursin("stablehlo.scatter", content)
    n_lines = count('\n', content)

    @info "MLIR exported" path=mlir_path lines=n_lines has_scatter=has_scatter

    if has_scatter
        @warn "MLIR contains scatter - this is expected for energy_from_rij"
        @warn "JAX reverse-mode grad will work (JAX handles scatter), but IREE won't compile"
    end

    # Save metadata for Python
    meta_path = joinpath(output_dir, "$(name)_metadata.json")
    open(meta_path, "w") do f
        JSON3.pretty(f, Dict(
            "name" => name,
            "input_shape" => [n_edges, 3],
            "n_edges" => n_edges,
            "n_atoms" => length(G.node_data),
            "elements" => String.(elements),
            "rcut" => rcut,
            "has_scatter" => has_scatter,
        ))
    end

    # Save reference rij and ii, jj arrays for Python
    npz_path = joinpath(output_dir, "$(name)_inputs.npz")
    NPZ.npzwrite(npz_path,
        rij = arrays.rij,
        ii = arrays.ii,
        jj = arrays.jj,
        zi = arrays.zi,
        zj = arrays.zj,
    )

    return mlir_path
end

## ============================================================================
## Package Creation
## ============================================================================

"""
    create_package(calc, name, output_dir; kwargs...)

Create a redistributable Python package from an ACE calculator.
"""
function create_package(calc, name::String, output_dir::String;
                        elements=(:Si,),
                        rcut::Float64=5.5,
                        version::String="1.0.0",
                        backends::Vector{Symbol}=[:cpu])

    @info "Creating package: $name"

    # Create directory structure
    pkg_dir = joinpath(output_dir, name)
    src_dir = joinpath(pkg_dir, "src", name)
    models_dir = joinpath(src_dir, "models")
    build_dir = joinpath(output_dir, ".build")

    mkpath(src_dir)
    mkpath(models_dir)
    mkpath(build_dir)

    # Create reference system for compilation
    sys = create_reference_system(elements, rcut)

    # Export MLIR
    @info "Step 1: Exporting to MLIR..."
    mlir_path = export_mlir(calc, sys, build_dir; elements=elements)

    if mlir_path === nothing
        @warn "MLIR export failed"
    else
        # Copy MLIR to models dir
        cp(mlir_path, joinpath(models_dir, "model.mlir"); force=true)

        # Compile to VMFB for each backend
        @info "Step 2: Compiling to VMFB..."
        for backend in backends
            vmfb_path = joinpath(models_dir, "model_$(backend).vmfb")
            compile_vmfb(mlir_path, vmfb_path; backend=backend)
        end
    end

    # Export metadata
    @info "Step 3: Creating metadata..."
    metadata = Dict(
        "name" => name,
        "version" => version,
        "elements" => String.(elements),
        "rcut" => rcut,
        "backends" => String.(backends),
    )
    open(joinpath(models_dir, "metadata.json"), "w") do f
        JSON3.pretty(f, metadata)
    end

    # Create Python package files
    @info "Step 4: Creating Python package..."

    # pyproject.toml
    pyproject = replace(PYPROJECT_TEMPLATE, "{name}" => name, "{version}" => version)
    write(joinpath(pkg_dir, "pyproject.toml"), pyproject)

    # __init__.py
    init_content = replace(INIT_TEMPLATE, "{name}" => name, "{version}" => version)
    write(joinpath(src_dir, "__init__.py"), init_content)

    # calculator.py
    write(joinpath(src_dir, "calculator.py"), CALCULATOR_TEMPLATE)

    @info "Package created: $pkg_dir"

    # Summary
    println("\n" * "="^50)
    println("Package: $name")
    println("Location: $pkg_dir")
    println("Elements: $(join(String.(elements), ", "))")
    println("Cutoff: $rcut Å")
    println("="^50)
    println("\nTo install:")
    println("  pip install $pkg_dir")

    return pkg_dir
end

## ============================================================================
## Main
## ============================================================================

function main()
    args = ARGS

    # Defaults
    model_path = nothing
    test_model = false
    name = "acepotential"
    output_dir = "dist"
    backends = [:cpu]
    elements = (:Si,)
    version = "1.0.0"

    # Parse args
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--test-model"
            test_model = true
        elseif arg == "--name" && i < length(args)
            i += 1; name = args[i]
        elseif arg == "--output" && i < length(args)
            i += 1; output_dir = args[i]
        elseif arg == "--backends" && i < length(args)
            i += 1; backends = [Symbol(b) for b in split(args[i], ",")]
        elseif arg == "--help" || arg == "-h"
            println("""
Usage: julia create_package.jl [OPTIONS] [MODEL_PATH]

Options:
  --test-model      Create minimal test model
  --name NAME       Package name (default: acepotential)
  --output DIR      Output directory (default: dist)
  --backends LIST   Comma-separated: cpu,cuda (default: cpu)
  --help            Show this help
""")
            return
        elseif !startswith(arg, "-")
            model_path = arg
        end
        i += 1
    end

    # Create or load model
    local calc, model_elements, rcut

    if test_model
        calc, model_elements, rcut = create_test_model(; elements=elements)
    elseif model_path !== nothing
        calc = ACEpotentials.load_potential(model_path)
        model_elements = elements
        rcut = 5.5
    else
        @error "Provide --test-model or a model path"
        return
    end

    create_package(calc, name, output_dir;
                   elements=model_elements, rcut=rcut, version=version, backends=backends)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
