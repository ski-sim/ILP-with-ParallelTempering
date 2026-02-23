sampler=lbp
graph_type=mis
max_num_nodes=3000
max_num_constraints=13000
penalty_weight=500
t_schedule=exp_decay
init_temperature=50.0
t_min=50.0
t_max=100.0
decay_rate=0.5

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=.96
# export XLA_FLAGS='--xla_force_host_platform_device_count=4'
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

# graph_type should be one of [mis, ca, sc]
if [ "$graph_type" != "mis" ] && [ "$graph_type" != "ca" ] && [ "$graph_type" != "sc" ] 
then
   echo "graph_type should be one of [mis, ca, sc]"
   exit 1
fi

echo "ilp"
echo "$sampler"
echo "$graph_type"

python -m discs.experiment.main_sampling \
   --sampler_config="discs/samplers/configs/${sampler?}_config.py" \
   --run_local=True --save_root=./discs/results --model=ilp \
   --graph_type=${graph_type} --max_num_nodes=${max_num_nodes} --max_num_constraints=${max_num_constraints} \
   --penalty_weight=${penalty_weight} --formulation=${formulation} --reweight=None \
   --num_instances=20 --num_models=1 --batch_size=15 --chain_length=100000 \
   --t_schedule=${t_schedule} --init_temperature=${init_temperature} --decay_rate=${decay_rate} \
   --pt=deo --pt_interval=200 --t_min=${t_min} --t_max=${t_max} --log_every_steps=100 --mode=test_step --step_limit=5000
