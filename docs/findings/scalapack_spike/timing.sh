cd /tmp/essswb-scalapack-spike/sp
# wait for sweep
while ! grep -q "SWEEP DONE" sweep.log; do sleep 10; done
F=timing.log; : > $F
flt() { grep -v "PCI\|32bits\|ABI\|Warning\|soft scope\|└" ; }
M=120000; N=2000
echo "=================== TIMING m=$M n=$N (1.92 GB) family=scaled cond=1e16 ranks=8" >> $F
for s in "mpi_tsqr.jl" "mpi_scalapack.jl grid=1d" "mpi_scalapack.jl grid=2d" "mpi_lsqr.jl precond=1"; do
  sed "s/ulimit -v 6000000/ulimit -v 14000000/" run.sh > run_big.sh
  bash run_big.sh 8 $s m=$M n=$N cond=1e16 family=scaled 2>&1 | flt >> $F
  free -g | head -2 >> $F
done
sed "s/ulimit -v 6000000/ulimit -v 14000000/" run_elem.sh > run_elem_big.sh
bash run_elem_big.sh 8 mpi_elemental.jl m=$M n=$N cond=1e16 family=scaled 2>&1 | flt >> $F
# Elemental + mixed family at 8 ranks, small, for the agreement table
for cnd in 1e12 1e16; do for fam in scaled mixed; do
  bash run_elem.sh 8 mpi_elemental.jl m=20000 n=500 cond=$cnd family=$fam 2>&1 | flt >> $F
done; done
echo "TIMING DONE" >> $F
