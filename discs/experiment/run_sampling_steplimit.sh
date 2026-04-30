graph_type=$1  # mis, ca, sc
max_num_nodes=$2
t_schedule=$3  # exp_decay, pt_exp_decay
sampler=${4:-lbp}
formulation=${5:-max_linear}


if [ "$graph_type" == "mvc" ]; then
   penalty_weight=1
   init_temperature=0.2
   if [ "$max_num_nodes" == 1000 ]; then
      max_num_constraints=65100
      if [ "$t_schedule" == "exp_decay" ]; then
         step_limit=30000
      elif [ "$t_schedule" == "pt_exp_decay" ]; then
         init_temperature=0.1
         step_limit=30000
      elif [ "$t_schedule" == "pen_pt_exp_decay" ]; then
         step_limit=30000
      else
         echo "t_schedule should be one of [exp_decay, pt_exp_decay]"
         exit 1
      fi
   elif [ "$max_num_nodes" == 2000 ]; then
      max_num_constraints=15000
      if [ "$t_schedule" == "exp_decay" ]; then
         step_limit=10000
      elif [ "$t_schedule" == "pt_exp_decay" ]; then
         init_temperature=0.1
         step_limit=10000
      elif [ "$t_schedule" == "pen_pt_exp_decay" ]; then
         step_limit=10000
      else
         echo "t_schedule should be one of [exp_decay, pt_exp_decay]"
         exit 1
      fi
   else
      echo "max_num_nodes should be 1500 or 3000"
      exit 1
   fi


elif [ "$graph_type" == "mvc_long" ]; then
   penalty_weight=1
   init_temperature=0.2
   if [ "$max_num_nodes" == 1000 ]; then
      max_num_constraints=65100
      if [ "$t_schedule" == "exp_decay" ]; then
         step_limit=150000
      elif [ "$t_schedule" == "pt_exp_decay" ]; then
         step_limit=150000
      elif [ "$t_schedule" == "pen_pt_exp_decay" ]; then
         step_limit=150000
      else
         echo "t_schedule should be one of [exp_decay, pt_exp_decay]"
         exit 1
      fi
   elif [ "$max_num_nodes" == 2000 ]; then
      max_num_constraints=15000
      if [ "$t_schedule" == "exp_decay" ]; then
         step_limit=50000
      elif [ "$t_schedule" == "pt_exp_decay" ]; then
         step_limit=50000
      elif [ "$t_schedule" == "pen_pt_exp_decay" ]; then
         step_limit=50000
      else
         echo "t_schedule should be one of [exp_decay, pt_exp_decay]"
         exit 1
      fi
   else
      echo "max_num_nodes should be 1500 or 3000"
      exit 1
   fi

# MIS
elif [ "$graph_type" == "mis" ]; then
   penalty_weight=2
   init_temperature=0.2
   if [ "$max_num_nodes" == 1500 ]; then
      max_num_constraints=7000
      if [ "$t_schedule" == "exp_decay" ]; then
         step_limit=90000
      elif [ "$t_schedule" == "pt_exp_decay" ]; then
         step_limit=85000
      elif [ "$t_schedule" == "pen_pt_exp_decay" ]; then
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
      elif [ "$t_schedule" == "pen_pt_exp_decay" ]; then
         step_limit=52000
      else
         echo "t_schedule should be one of [exp_decay, pt_exp_decay]"
         exit 1
      fi
   else
      echo "max_num_nodes should be 1500 or 3000"
      exit 1
   fi

# MIS
elif [ "$graph_type" == "mis_long" ]; then
   penalty_weight=2
   init_temperature=0.2
   if [ "$max_num_nodes" == 1500 ]; then
      max_num_constraints=7000
      if [ "$t_schedule" == "exp_decay" ]; then
         step_limit=450000
      elif [ "$t_schedule" == "pt_exp_decay" ]; then
         step_limit=450000
      elif [ "$t_schedule" == "pen_pt_exp_decay" ]; then
         step_limit=450000
      else
         echo "t_schedule should be one of [exp_decay, pt_exp_decay]"
         exit 1
      fi
   elif [ "$max_num_nodes" == 3000 ]; then
      max_num_constraints=15000
      if [ "$t_schedule" == "exp_decay" ]; then
         step_limit=260000
      elif [ "$t_schedule" == "pt_exp_decay" ]; then
         step_limit=260000
      elif [ "$t_schedule" == "pen_pt_exp_decay" ]; then
         step_limit=260000
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
   penalty_weight=300
   init_temperature=50
   if [ "$max_num_nodes" == 2000 ]; then
      max_num_constraints=1400
      if [ "$t_schedule" == "exp_decay" ]; then
         step_limit=90000
      elif [ "$t_schedule" == "pt_exp_decay" ]; then
         step_limit=85000
      elif [ "$t_schedule" == "pen_pt_exp_decay" ]; then
         penalty_weight=400
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
         step_limit=87500
      elif [ "$t_schedule" == "pen_pt_exp_decay" ]; then
         penalty_weight=400
         step_limit=84000
      else
         echo "t_schedule should be one of [exp_decay, pt_exp_decay]"
         exit 1
      fi
   else
      echo "max_num_nodes should be 2000 or 4000"
      exit 1
   fi

# CA
elif [ "$graph_type" == "ca_long" ]; then
   penalty_weight=300
   init_temperature=50
   if [ "$max_num_nodes" == 2000 ]; then
      max_num_constraints=1400
      if [ "$t_schedule" == "exp_decay" ]; then
         step_limit=450000
      elif [ "$t_schedule" == "pt_exp_decay" ]; then
         step_limit=450000
      elif [ "$t_schedule" == "pen_pt_exp_decay" ]; then
         step_limit=450000
      else
         echo "t_schedule should be one of [exp_decay, pt_exp_decay]"
         exit 1
      fi
   elif [ "$max_num_nodes" == 4000 ]; then
      max_num_constraints=2800
      if [ "$t_schedule" == "exp_decay" ]; then
         step_limit=420000
      elif [ "$t_schedule" == "pt_exp_decay" ]; then
         step_limit=420000
      elif [ "$t_schedule" == "pen_pt_exp_decay" ]; then
         penalty_weight=400
         step_limit=420000
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
   init_temperature=1.0
   if [ "$max_num_nodes" == 2000 ]; then
      max_num_constraints=5000
      if [ "$t_schedule" == "exp_decay" ]; then
         step_limit=90000
      elif [ "$t_schedule" == "pt_exp_decay" ]; then
         step_limit=85000
      elif [ "$t_schedule" == "pen_pt_exp_decay" ]; then
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
      elif [ "$t_schedule" == "pen_pt_exp_decay" ]; then
         step_limit=74000
      else
         echo "t_schedule should be one of [exp_decay, pt_exp_decay]"
         exit 1
      fi
   else
      echo "max_num_nodes should be 2000 or 4000"
      exit 1
   fi


# SC
elif [ "$graph_type" == "sc_long" ]; then
   penalty_weight=5
   init_temperature=1.0
   if [ "$max_num_nodes" == 2000 ]; then
      max_num_constraints=5000
      if [ "$t_schedule" == "exp_decay" ]; then
         step_limit=450000
      elif [ "$t_schedule" == "pt_exp_decay" ]; then
         step_limit=450000
      elif [ "$t_schedule" == "pen_pt_exp_decay" ]; then
         step_limit=450000
      else
         echo "t_schedule should be one of [exp_decay, pt_exp_decay]"
         exit 1
      fi
   elif [ "$max_num_nodes" == 4000 ]; then
      max_num_constraints=5000
      if [ "$t_schedule" == "exp_decay" ]; then
         step_limit=37000
      elif [ "$t_schedule" == "pt_exp_decay" ]; then
         step_limit=37000
      elif [ "$t_schedule" == "pen_pt_exp_decay" ]; then
         step_limit=37000
      else
         echo "t_schedule should be one of [exp_decay, pt_exp_decay]"
         exit 1
      fi
   else
      echo "max_num_nodes should be 2000 or 4000"
      exit 1
   fi
# SC
elif [ "$graph_type" == "item" ]; then
   penalty_weight=5
   init_temperature=0.5
   if [ "$max_num_nodes" == 1083 ]; then
      max_num_constraints=5000
      if [ "$t_schedule" == "exp_decay" ]; then
         step_limit=50000
      elif [ "$t_schedule" == "pt_exp_decay" ]; then
         step_limit=55000
      elif [ "$t_schedule" == "pen_pt_exp_decay" ]; then
         step_limit=55000
      else
         echo "t_schedule should be one of [exp_decay, pt_exp_decay]"
         exit 1
      fi
   fi
elif [ "$graph_type" == "anonymous" ]; then
   penalty_weight=10
   init_temperature=1.0
   if [ "$max_num_nodes" == 1083 ]; then
      max_num_constraints=5000
      if [ "$t_schedule" == "exp_decay" ]; then
         step_limit=50000
      elif [ "$t_schedule" == "pt_exp_decay" ]; then
         step_limit=55000
      elif [ "$t_schedule" == "pen_pt_exp_decay" ]; then
         step_limit=55000
      else
         echo "t_schedule should be one of [exp_decay, pt_exp_decay]"
         exit 1
      fi
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
   --penalty_weight=${penalty_weight} --formulation=${formulation}  \
   --num_instances=100 --num_models=1 --batch_size=15 --chain_length=100000 --l_min=$(echo "${penalty_weight}/2" | bc -l) --l_max=${penalty_weight} \
   --t_schedule=${t_schedule} --init_temperature=${init_temperature} --decay_rate=0.5 --reweight=None \
   --pt=deo --pt_interval=200 --t_min=${init_temperature} --t_max=$(echo "2*${init_temperature}" | bc -l) --log_every_steps=100 --mode=test_step --step_limit=${step_limit}
wait