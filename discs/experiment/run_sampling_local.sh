model=ilp
graph_type=mis
sampler=lbp
# export CUDA_VISIBLE_DEVICES=1
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


CUDA_VISIBLE_DEVICES=4 python -m discs.experiment.main_sampling \
  --model_config="discs/models/configs/${model?}_config.py" \
  --sampler_config="discs/samplers/configs/${sampler?}_config.py" \
  --config=$exp_config \
  --run_local=True \
  --model=ilp --graph_type=ca --penalty_weight=1000 --num_models=1 --max_num_nodes=2000 --max_num_constraints=10000 --num_instances=20 \
  --formulation=max_linear --proposal_type=obj --batch_size=20 --t_schedule=exp_decay --chain_length=100000 --decay_rate=0.01 \
  --save_root=./discs/results --pt=deo --pt_interval=1000 --t_min=0.1 --t_max=5.0 --init_temperature=100 --final_temperature=10 --reweight=None&

CUDA_VISIBLE_DEVICES=5 python -m discs.experiment.main_sampling \
  --model_config="discs/models/configs/${model?}_config.py" \
  --sampler_config="discs/samplers/configs/${sampler?}_config.py" \
  --config=$exp_config \
  --run_local=True \
  --model=ilp --graph_type=ca --penalty_weight=1000 --num_models=1 --max_num_nodes=2000 --max_num_constraints=10000 --num_instances=20 \
  --formulation=max_linear --proposal_type=obj --batch_size=20 --t_schedule=exp_decay --chain_length=100000 --decay_rate=0.01 \
  --save_root=./discs/results --pt=deo --pt_interval=1000 --t_min=0.1 --t_max=5.0 --init_temperature=100 --final_temperature=1 --reweight=None&

CUDA_VISIBLE_DEVICES=6 python -m discs.experiment.main_sampling \
  --model_config="discs/models/configs/${model?}_config.py" \
  --sampler_config="discs/samplers/configs/${sampler?}_config.py" \
  --config=$exp_config \
  --run_local=True \
  --model=ilp --graph_type=ca --penalty_weight=1000 --num_models=1 --max_num_nodes=2000 --max_num_constraints=10000 --num_instances=20 \
  --formulation=max_linear --proposal_type=obj --batch_size=20 --t_schedule=exp_decay --chain_length=100000 --decay_rate=0.01 \
  --save_root=./discs/results --pt=deo --pt_interval=1000 --t_min=0.1 --t_max=5.0 --init_temperature=1000 --final_temperature=10 --reweight=None&
