model=ilp
graph_type=mis
sampler=lbp
export CUDA_VISIBLE_DEVICES=3
# export XLA_FLAGS='--xla_force_host_platform_device_count=4'

default="default_value"
graph_type=${graph_type:-$default}
echo "$model"
echo "$sampler"
echo "$graph_type"


if [ "$graph_type" == "$default" ]
then
   exp_config="discs/common/configs.py"
else
   exp_config="discs/experiment/configs/${model?}/${graph_type:-$default}.py" 
fi

if [ "$model" == "text_infilling" ]
then
   exp_config="discs/experiment/configs/lm_experiment.py"
fi

echo $exp_config


python -m discs.experiment.main_sampling \
  --model_config="discs/models/configs/${model?}_config.py" \
  --sampler_config="discs/samplers/configs/${sampler?}_config.py" \
  --config=$exp_config \
  --run_local=True \
  --model=ilp --graph_type=mis --penalty_weight=10 --num_models=1 --max_num_nodes=3000 --max_num_constraints=10000 --num_instances=20 \
  --formulation=max_linear_square --proposal_type=obj --batch_size=20 --t_schedule=exp_decay --chain_length=100000 --decay_rate=0.01 \
  --save_root=./discs/results --pt=deo --pt_interval=1000 --t_min=0.1 --t_max=5.0 --init_temperature=1 --final_temperature=0.0001 --reweight=None

