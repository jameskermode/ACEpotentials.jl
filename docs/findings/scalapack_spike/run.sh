# usage: run.sh <nranks> <script> [args...]
cd /tmp/essswb-scalapack-spike/sp
MPIEXEC=$(julia +1.11 --project=../proj -e 'using MPI; print(MPI.mpiexec().exec[1])' 2>/dev/null)
NR=$1; shift; S=$1; shift
ulimit -v 6000000    # 6 GB virtual per process
export OPENBLAS_NUM_THREADS=2
timeout 500 $MPIEXEC -n $NR julia +1.11 --project=../proj $S "$@"
echo "EXIT=$?"
