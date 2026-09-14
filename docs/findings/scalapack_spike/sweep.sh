cd /tmp/essswb-scalapack-spike/sp
F=sweep.log; : > $F
flt() { grep -v "PCI\|32bits\|ABI\|Warning\|soft scope\|└" ; }
for fam in scaled mixed; do
 for cnd in 1e8 1e12 1e16 1e21; do
  echo "=================== family=$fam cond=$cnd  m=20000 n=500 ranks=8" >> $F
  bash run.sh 8 mpi_tsqr.jl      m=20000 n=500 cond=$cnd family=$fam 2>&1 | flt >> $F
  bash run.sh 8 mpi_scalapack.jl m=20000 n=500 cond=$cnd family=$fam grid=1d 2>&1 | flt >> $F
  bash run.sh 8 mpi_scalapack.jl m=20000 n=500 cond=$cnd family=$fam grid=2d 2>&1 | flt >> $F
  bash run.sh 8 mpi_lsqr.jl      m=20000 n=500 cond=$cnd family=$fam precond=1 2>&1 | flt >> $F
  bash run.sh 8 mpi_lsqr.jl      m=20000 n=500 cond=$cnd family=$fam precond=1 damp=5e-3 2>&1 | flt >> $F
 done
done
echo "SWEEP DONE" >> $F
