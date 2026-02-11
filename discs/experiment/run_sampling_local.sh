sampler=lbp
graph_type=ca
max_num_nodes=4000
max_num_constraints=2800
penalty_weight=1000
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=.48
# export XLA_FLAGS='--xla_force_host_platform_device_count=4'

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
   --penalty_weight=${penalty_weight} --formulation=max_linear --proposal_type=obj --reweight=None \
   --num_instances=20 --num_models=1 --batch_size=15 --chain_length=100000 \
   --t_schedule=pt_exp_decay --init_temperature=100 --decay_rate=0.1 --final_temperature=0.0 \
   --pt=deo --pt_interval=1000 --t_min=100 --t_max=200
