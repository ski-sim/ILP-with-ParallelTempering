instance_name=$1  # mis, ca, sc, mvc
max_num_vars=$2
t_schedule=$3      # exp_decay | pt | pt_exp_decay | pen_pt | pen_pt_exp_decay
sampler=${4:-lbp}
num_flips=${5:-}    # optional; overrides sampler_config.num_flips if set
adaptive=${6:-}     # optional; overrides sampler_config.adaptive (true/false) if set
formulation=${7:-max_linear}

valid_schedules="[exp_decay, pt, pt_exp_decay, pen_pt, pen_pt_exp_decay]"

# Minimum Vertex Cover
if [ "$instance_name" == "mvc" ]; then
   penalty_weight=1
   init_temperature=0.2
   if [ "$max_num_vars" == 1000 ]; then
      max_num_cons=65100
      case "$t_schedule" in
         exp_decay)
            step_limit=30000
            ;;
         pt|pt_exp_decay)
            init_temperature=0.1
            step_limit=30000
            ;;
         pen_pt|pen_pt_exp_decay)
            step_limit=30000
            ;;
         *)
            echo "t_schedule should be one of ${valid_schedules}"
            exit 1
            ;;
      esac
   elif [ "$max_num_vars" == 2000 ]; then
      max_num_cons=15000
      case "$t_schedule" in
         exp_decay)
            step_limit=10000
            ;;
         pt|pt_exp_decay)
            init_temperature=0.1
            step_limit=10000
            ;;
         pen_pt|pen_pt_exp_decay)
            step_limit=10000
            ;;
         *)
            echo "t_schedule should be one of ${valid_schedules}"
            exit 1
            ;;
      esac
   else
      echo "max_num_vars should be 1000 or 2000"
      exit 1
   fi


# Maximum Independent Set
elif [ "$instance_name" == "mis" ]; then
   penalty_weight=2
   init_temperature=0.2
   if [ "$max_num_vars" == 1500 ]; then
      max_num_cons=7000
      case "$t_schedule" in
         exp_decay)
            step_limit=90000
            ;;
         pt|pt_exp_decay)
            step_limit=85000
            ;;
         pen_pt|pen_pt_exp_decay)
            step_limit=85000
            ;;
         *)
            echo "t_schedule should be one of ${valid_schedules}"
            exit 1
            ;;
      esac
   elif [ "$max_num_vars" == 3000 ]; then
      max_num_cons=15000
      case "$t_schedule" in
         exp_decay)
            step_limit=55000
            ;;
         pt|pt_exp_decay)
            step_limit=52000
            ;;
         pen_pt|pen_pt_exp_decay)
            step_limit=52000
            ;;
         *)
            echo "t_schedule should be one of ${valid_schedules}"
            exit 1
            ;;
      esac
   else
      echo "max_num_vars should be 1500 or 3000"
      exit 1
   fi


# Combinatorial Auction
elif [ "$instance_name" == "ca" ]; then
   penalty_weight=300
   init_temperature=50
   if [ "$max_num_vars" == 2000 ]; then
      max_num_cons=1400
      case "$t_schedule" in
         exp_decay)
            step_limit=90000
            ;;
         pt|pt_exp_decay)
            step_limit=85000
            ;;
         pen_pt|pen_pt_exp_decay)
            penalty_weight=400
            step_limit=85000
            ;;
         *)
            echo "t_schedule should be one of ${valid_schedules}"
            exit 1
            ;;
      esac
   elif [ "$max_num_vars" == 4000 ]; then
      max_num_cons=2800
      case "$t_schedule" in
         exp_decay)
            step_limit=88000
            ;;
         pt|pt_exp_decay)
            step_limit=87500
            ;;
         pen_pt|pen_pt_exp_decay)
            penalty_weight=400
            step_limit=84000
            ;;
         *)
            echo "t_schedule should be one of ${valid_schedules}"
            exit 1
            ;;
      esac
   else
      echo "max_num_vars should be 2000 or 4000"
      exit 1
   fi

# Set Covering
elif [ "$instance_name" == "sc" ]; then
   penalty_weight=5
   init_temperature=1.0
   if [ "$max_num_vars" == 2000 ]; then
      max_num_cons=5000
      case "$t_schedule" in
         exp_decay)
            step_limit=90000
            ;;
         pt|pt_exp_decay)
            step_limit=85000
            ;;
         pen_pt|pen_pt_exp_decay)
            step_limit=85000
            ;;
         *)
            echo "t_schedule should be one of ${valid_schedules}"
            exit 1
            ;;
      esac
   elif [ "$max_num_vars" == 4000 ]; then
      max_num_cons=5000
      case "$t_schedule" in
         exp_decay)
            step_limit=78000
            ;;
         pt|pt_exp_decay)
            step_limit=74000
            ;;
         pen_pt|pen_pt_exp_decay)
            step_limit=74000
            ;;
         *)
            echo "t_schedule should be one of ${valid_schedules}"
            exit 1
            ;;
      esac
   else
      echo "max_num_vars should be 2000 or 4000"
      exit 1
   fi


else
   echo "instance_name should be one of [mis, ca, sc, mvc]"
   exit 1
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION=.96
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export TF_CPP_MIN_LOG_LEVEL=3
export GRPC_VERBOSITY=ERROR
export GLOG_minloglevel=3
export PYTHONWARNINGS="ignore"
export TF_ENABLE_ONEDNN_OPTS=0

# Build optional sampler_config overrides
sampler_overrides=()
[ -n "$num_flips" ] && sampler_overrides+=(--sampler_config.num_flips=${num_flips})
[ -n "$adaptive" ] && sampler_overrides+=(--sampler_config.adaptive=${adaptive})

python -W ignore -m PT_ILP.main_sampling \
   --sampler_config="PT_ILP/samplers/configs/${sampler?}_config.py" \
   "${sampler_overrides[@]}" \
   --save_root=./PT_ILP/results --model=ilp \
   --instance_name=$instance_name --max_num_vars=${max_num_vars} --max_num_cons=${max_num_cons} \
   --penalty_weight=${penalty_weight} --formulation=${formulation}  \
   --num_instances=100 --num_models=1 --batch_size=15 --chain_length=100000 --l_min=$(echo "${penalty_weight}/2" | bc -l) --l_max=${penalty_weight} \
   --t_schedule=${t_schedule} --init_temperature=${init_temperature} --decay_rate=0.5 \
   --pt=deo --pt_interval=200 --t_min=${init_temperature} --t_max=$(echo "2*${init_temperature}" | bc -l) --log_every_steps=100 --mode=steplimit --step_limit=${step_limit}
wait
