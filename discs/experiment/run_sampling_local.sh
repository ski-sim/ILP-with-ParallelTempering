graph_type=$1  # mis, ca, sc
max_num_nodes=$2
t_schedule=$3  # exp_decay, pt_exp_decay
sampler=${4:-lbp}
formulation=${5:-max_linear_square}

# MIS
if [ "$graph_type" == "mis" ]; then
   penalty_weight=2
   init_temperature=0.2
   t_min=0.2
   t_max=0.4
   if [ "$max_num_nodes" == 1500 ]; then
      max_num_constraints=7000
      if [ "$t_schedule" == "exp_decay" ]; then
         step_limit=90000
      elif [ "$t_schedule" == "pt_exp_decay" ]; then
         step_limit=85000
      else
         echo "t_schedule should be one of [exp_decay, pt_exp_decay]"
         exit 1
      fi
   elif [ "$max_num_nodes" == 3000 ]; then
      max_num_constraints=15000
      if [ "$t_schedule" == "exp_decay" ]; then
         step_limit=55000
      elif [ "$t_schedule" == "pt_exp_decay" ]; then
         step_limit=52000
      else
         echo "t_schedule should be one of [exp_decay, pt_exp_decay]"
         exit 1
      fi
   else
      echo "max_num_nodes should be 1500 or 3000"
      exit 1
   fi

# CA
elif [ "$graph_type" == "ca" ]; then
   penalty_weight=500
   init_temperature=50
   t_min=50
   t_max=100

   if [ "$max_num_nodes" == 2000 ]; then
      max_num_constraints=1400
      if [ "$t_schedule" == "exp_decay" ]; then
         step_limit=90000
      elif [ "$t_schedule" == "pt_exp_decay" ]; then
         step_limit=85000
      else
         echo "t_schedule should be one of [exp_decay, pt_exp_decay]"
         exit 1
      fi
   elif [ "$max_num_nodes" == 4000 ]; then
      max_num_constraints=2800
      if [ "$t_schedule" == "exp_decay" ]; then
         step_limit=88000
      elif [ "$t_schedule" == "pt_exp_decay" ]; then
         step_limit=84000
      else
         echo "t_schedule should be one of [exp_decay, pt_exp_decay]"
         exit 1
      fi
   else
      echo "max_num_nodes should be 2000 or 4000"
      exit 1
   fi

# SC
elif [ "$graph_type" == "sc" ]; then
   penalty_weight=5
   init_temperature=0.5
   t_min=0.5
   t_max=1.0
   if [ "$max_num_nodes" == 2000 ]; then
      max_num_constraints=5000
      if [ "$t_schedule" == "exp_decay" ]; then
      step_limit=90000
      elif [ "$t_schedule" == "pt_exp_decay" ]; then
      step_limit=85000
      else
      echo "t_schedule should be one of [exp_decay, pt_exp_decay]"
      exit 1
      fi
   elif [ "$max_num_nodes" == 4000 ]; then
      max_num_constraints=5000
      if [ "$t_schedule" == "exp_decay" ]; then
         step_limit=78000
      elif [ "$t_schedule" == "pt_exp_decay" ]; then
         step_limit=74000
      else
         echo "t_schedule should be one of [exp_decay, pt_exp_decay]"
         exit 1
      fi
   else
      echo "max_num_nodes should be 2000 or 4000"
      exit 1
   fi

else
   echo "graph_type should be one of [mis, ca, sc]"
   exit 1
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION=.96
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

python -m discs.experiment.main_sampling \
   --sampler_config="discs/samplers/configs/${sampler?}_config.py" \
   --run_local=True --save_root=./discs/results --model=ilp \
   --graph_type=${graph_type} --max_num_nodes=${max_num_nodes} --max_num_constraints=${max_num_constraints} \
   --penalty_weight=${penalty_weight} --formulation=${formulation} --reweight=None \
   --num_instances=20 --num_models=1 --batch_size=15 --chain_length=100000 \
   --t_schedule=${t_schedule} --init_temperature=${init_temperature} --decay_rate=0.5 \
   --pt=deo --pt_interval=200 --t_min=${t_min} --t_max=${t_max} --log_every_steps=100 --mode=test_step --step_limit=${step_limit}
