using ACEpotentials, LinearAlgebra
ET = ACEpotentials.Models.EquivariantTensors
for ll in [(0,), (1,1), (0,0,0), (1,1,2), (2,1,1), (1,2,1), (2,2,2), (1,1,1,1)]
   U, MM = ET.O3.coupling_coeffs(0, ll, collect(1:length(ll)); PI = false, basis = real)
   println("ll=$ll  size U=$(size(U))  nMM=$(length(MM))  rank=$(rank(U))  MM[1]=$(MM[1]) type $(typeof(MM[1]))")
   Up, MMp = ET.O3.coupling_coeffs(0, ll, fill(1, length(ll)); PI = true, basis = real)
   println("     PI, all n equal: size U=$(size(Up)) nMM=$(length(MMp))")
end
