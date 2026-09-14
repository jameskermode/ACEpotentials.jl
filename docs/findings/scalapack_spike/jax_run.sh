cd /tmp/essswb-scalapack-spike/sp
V=/storage/eng/essswb/macejax-gpu/venv/bin/python
ulimit -v 40000000
export XLA_PYTHON_CLIENT_PREALLOCATE=false
$V jax_arm.py cases acc > jax_acc.log 2>&1; echo "ACC EXIT=$?" >> jax_acc.log
$V jax_tsqr_shard.py cases > jax_shard.log 2>&1; echo "SHARD EXIT=$?" >> jax_shard.log
echo "JAX SMALL DONE" > jax_small.done
