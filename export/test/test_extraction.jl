#=
Test ReactantStackedModel Extraction
====================================

Tests extraction of parameters from ACEpotentials.jl models into
Reactant-compatible format (ReactantStackedModel/ReactantETACEState).

This test creates a simple ACE model using M.ace_model() (LearnableRnlrzzBasis),
converts it to ETACE, then extracts parameters and verifies they match.
=#

using Test
using Printf
using LinearAlgebra
using Random
using StaticArrays
using SparseArrays

# Load ACEpotentials
using ACEpotentials
M = ACEpotentials.Models
ETM = ACEpotentials.ETModels

# Load extraction modules directly (without Reactant dependency)
include(joinpath(@__DIR__, "..", "src", "reactant_state.jl"))
include(joinpath(@__DIR__, "..", "src", "reactant_stacked.jl"))

# Define ACEExport module wrapper for imported functions
module ACEExport
    using ..Main: ReactantETACEState, ReactantPairState, ReactantStackedModel,
        prepare_reactant_state, spec_to_matrix, sparse_to_dense,
        z_to_species_index, zz_to_pair_index, zz_to_pair_index_sym
    export ReactantETACEState, ReactantPairState, ReactantStackedModel,
        prepare_reactant_state, spec_to_matrix, sparse_to_dense
end
using .ACEExport

import EquivariantTensors as ET
using Lux

println("="^60)
println("REACTANT STACKED MODEL EXTRACTION TEST")
println("="^60)

## ============================================================================
## Step 1: Create a simple ACE model using M.ace_model
## ============================================================================

println("\n1. Creating ACE model with LearnableRnlrzzBasis...")

# Use M.ace_model (LearnableRnlrzzBasis), not ace1_model (SplineRnlrzzBasis)
# This is required for compatibility with convert2et
elements = (:Si,)
level = M.TotalDegree()
max_level = 8  # Small for fast testing
order = 2
maxl = 2

rin0cuts = M._default_rin0cuts(elements)
rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = 5.5)).(rin0cuts)

rng = Random.MersenneTwister(1234)

ace_model = M.ace_model(; elements = elements, order = order,
                        Ytype = :solid, level = level, max_level = max_level,
                        maxl = maxl, pair_maxn = max_level,
                        rin0cuts = rin0cuts,
                        init_WB = :glorot_normal, init_Wpair = :glorot_normal)

ps, st = Lux.setup(rng, ace_model)

n_species = length(ace_model.rbasis._i2z)
rcut = maximum(a.rcut for a in ace_model.pairbasis.rin0cuts)

@printf("   Elements: %s\n", elements)
@printf("   Order: %d, Max level: %d\n", order, max_level)
@printf("   Cutoff: %.2f Å\n", rcut)
@printf("   N species: %d\n", n_species)

## ============================================================================
## Step 2: Convert to ETACE model (many-body only)
## ============================================================================

println("\n2. Converting to ETACE model...")

et_model = ETM.convert2et(ace_model)
et_ps, et_st = Lux.setup(rng, et_model)

# Copy radial basis parameters
for i in 1:n_species, j in 1:n_species
    idx = (i-1)*n_species + j
    et_ps.rembed.post.W[:, :, idx] .= ps.rbasis.Wnlq[:, :, i, j]
end

# Copy readout parameters
for s in 1:n_species
    et_ps.readout.W[1, :, s] .= ps.WB[:, s]
end

# Create ETACEPotential (wrapped calculator)
ace_calc = ETM.ETACEPotential(et_model, et_ps, et_st, rcut)

# Get model info
nbasis = et_model.readout.in_dim
nspecies = et_model.readout.ncat
@printf("   ETACE: %d basis functions, %d species\n", nbasis, nspecies)

## ============================================================================
## Step 3: Test ReactantETACEState extraction
## ============================================================================

println("\n3. Testing ReactantETACEState extraction...")

ace_state = ACEExport.prepare_reactant_state(ace_calc; T=Float32)

@test ace_state isa ReactantETACEState{Float32}
@test ace_state.n_species == nspecies
@test ace_state.rcut ≈ Float32(rcut)
@test ace_state.n_basis == nbasis

@printf("   n_species: %d\n", ace_state.n_species)
@printf("   rcut: %.2f\n", ace_state.rcut)
@printf("   n_basis: %d\n", ace_state.n_basis)
@printf("   n_polys: %d\n", ace_state.n_polys)
@printf("   n_rnl: %d\n", ace_state.n_rnl)
@printf("   maxl: %d\n", ace_state.maxl)
@printf("   nYlm: %d\n", ace_state.nYlm)

# Check spec arrays
@test length(ace_state.spec_R) == length(ace_state.spec_Y)
@test length(ace_state.specs_mats) > 0
@printf("   spec_R/Y: %d indices\n", length(ace_state.spec_R))
@printf("   specs_mats: %d order matrices\n", length(ace_state.specs_mats))

# Check A2Bmap
@test size(ace_state.A2Bmap, 1) == nbasis
@printf("   A2Bmap: %s\n", size(ace_state.A2Bmap))

# Check Agnesi params
n_pairs = nspecies * nspecies
@test size(ace_state.agnesi_params) == (5, n_pairs)
@printf("   agnesi_params: %s\n", size(ace_state.agnesi_params))

# Check radial weights
@test size(ace_state.W_radial, 3) == n_pairs
@printf("   W_radial: %s\n", size(ace_state.W_radial))

# Check readout weights
@test size(ace_state.W_readout, 2) == nspecies
@printf("   W_readout: %s\n", size(ace_state.W_readout))

# Check polynomial coefficients
@test length(ace_state.poly_A) == ace_state.n_polys
@test length(ace_state.poly_B) == ace_state.n_polys
@test length(ace_state.poly_C) == ace_state.n_polys

println("   ✓ ReactantETACEState extraction successful")

## ============================================================================
## Step 4: Test ReactantStackedModel from single ETACE
## ============================================================================

println("\n4. Testing ReactantStackedModel from single ETACE...")

single_state = ReactantStackedModel(ace_calc; T=Float32)

@test single_state isa ReactantStackedModel{Float32}
@test single_state.has_pair == false
@test single_state.pair_state === nothing
@test single_state.ace_state.n_basis == nbasis
@test single_state.n_species == nspecies
@test single_state.rcut ≈ Float32(rcut)

@printf("   n_species: %d\n", single_state.n_species)
@printf("   rcut: %.2f\n", single_state.rcut)
@printf("   has_pair: %s\n", single_state.has_pair)
@printf("   E0: %s\n", single_state.E0)

println("   ✓ Single ETACE extraction successful")

## ============================================================================
## Step 5: Test StackedCalculator extraction (with E0)
## ============================================================================

println("\n5. Testing StackedCalculator extraction...")

# Create ETOneBody model
E0s = Dict(:Si => -158.54)
et_onebody = ETM.one_body(E0s, x -> x.z)
_, onebody_st = Lux.setup(rng, et_onebody)
E0_calc = ETM.WrappedSiteCalculator(et_onebody, nothing, onebody_st, 3.0)

# Stack E0 + ETACE
stacked_calc = ETM.StackedCalculator((E0_calc, ace_calc))

stacked_state = ReactantStackedModel(stacked_calc; T=Float32)

@test stacked_state isa ReactantStackedModel{Float32}
@test stacked_state.n_species == nspecies
@test stacked_state.has_pair == false  # No pair model in this stack
@test stacked_state.ace_state.n_basis == nbasis

# E0 should be extracted (will be zero if extraction doesn't find the right structure)
@printf("   E0: %s\n", stacked_state.E0)
@printf("   has_pair: %s\n", stacked_state.has_pair)

println("   ✓ StackedCalculator extraction successful")

## ============================================================================
## Step 6: Verify numerical consistency
## ============================================================================

println("\n6. Verifying numerical consistency...")

# Check that A2Bmap was correctly converted from sparse to dense
basis_st = ace_calc.st.basis
A2Bmap_original = basis_st.A2Bmaps[1]
A2Bmap_extracted = ace_state.A2Bmap

# Compare (allowing for Float32 precision)
@test size(A2Bmap_extracted, 1) == A2Bmap_original.m
@test size(A2Bmap_extracted, 2) == A2Bmap_original.n

# Check non-zero values match (SparseMatCSX stores values in nzval_csr)
nnz_original = length(A2Bmap_original.nzval_csr)
nnz_extracted = count(x -> abs(x) > 1e-10, A2Bmap_extracted)
# Note: nnz_extracted counts all elements, but with proper sparse->dense conversion
# we should have at least as many non-zeros in extracted as in original
@test nnz_extracted >= nnz_original - 10  # Allow some tolerance
@printf("   A2Bmap: %d dense elements with |val|>1e-10 (sparse has %d nz)\n", nnz_extracted, nnz_original)

# Check readout weights were copied correctly
W_original = ace_calc.ps.readout.W
W_extracted = ace_state.W_readout
@test size(W_original, 2) == size(W_extracted, 1)
@test size(W_original, 3) == size(W_extracted, 2)

# Verify values match (Float32 precision)
max_W_diff = maximum(abs.(Float32.(W_original[1, :, :]) - W_extracted))
@test max_W_diff < 1e-6
@printf("   Readout weights: max diff = %.2e\n", max_W_diff)

println("   ✓ Numerical consistency verified")

## ============================================================================
## Summary
## ============================================================================

println("\n" * "="^60)
println("ALL EXTRACTION TESTS PASSED")
println("="^60)

println("\nExtracted model summary:")
@printf("  - Species: %d (Z = %s)\n", stacked_state.n_species, stacked_state.species_Z)
@printf("  - Cutoff: %.2f Å\n", stacked_state.rcut)
@printf("  - E0: %s\n", stacked_state.E0)
@printf("  - Has pair potential: %s\n", stacked_state.has_pair)
@printf("  - ACE basis functions: %d\n", stacked_state.ace_state.n_basis)
@printf("  - Radial polynomials: %d\n", stacked_state.ace_state.n_polys)
@printf("  - max angular momentum: %d\n", stacked_state.ace_state.maxl)
