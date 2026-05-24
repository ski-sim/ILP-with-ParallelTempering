instance_name=$1  # mis, ca, sc
max_num_vars=$2
t_schedule=$3  # exp_decay, pt_exp_decay
sampler=${4:-lbp}
formulation=${5:-max_linear}


if [ "$instance_name" == "mvc" ]; then
   penalty_weight=1
   init_temperature=0.2
   if [ "$max_num_vars" == 1000 ]; then
      max_num_cons=65100
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
   elif [ "$max_num_vars" == 2000 ]; then
      max_num_cons=15000
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
      echo "max_num_vars should be 1500 or 3000"
      exit 1
   fi


elif [ "$instance_name" == "mvc_long" ]; then
   penalty_weight=1
   init_temperature=0.2
   if [ "$max_num_vars" == 1000 ]; then
      max_num_cons=65100
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
   elif [ "$max_num_vars" == 2000 ]; then
      max_num_cons=15000
      if [ "$t_schedule" == "exp_decay" ]; then
         step_limit=25000
      elif [ "$t_schedule" == "pt_exp_decay" ]; then
         step_limit=25000
      elif [ "$t_schedule" == "pen_pt_exp_decay" ]; then
         step_limit=25000
      else
         echo "t_schedule should be one of [exp_decay, pt_exp_decay]"
         exit 1
      fi
   else
      echo "max_num_vars should be 1500 or 3000"
      exit 1
   fi

# MIS
elif [ "$instance_name" == "mis" ]; then
   penalty_weight=2
   init_temperature=0.2
   if [ "$max_num_vars" == 1500 ]; then
      max_num_cons=7000
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
   elif [ "$max_num_vars" == 3000 ]; then
      max_num_cons=15000
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
      echo "max_num_vars should be 1500 or 3000"
      exit 1
   fi

# MIS
elif [ "$instance_name" == "mis_long" ]; then
   penalty_weight=2
   init_temperature=0.2
   if [ "$max_num_vars" == 1500 ]; then
      max_num_cons=7000
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
   elif [ "$max_num_vars" == 3000 ]; then
      max_num_cons=15000
      if [ "$t_schedule" == "exp_decay" ]; then
         step_limit=130000
      elif [ "$t_schedule" == "pt_exp_decay" ]; then
         step_limit=130000
      elif [ "$t_schedule" == "pen_pt_exp_decay" ]; then
         step_limit=130000
      else
         echo "t_schedule should be one of [exp_decay, pt_exp_decay]"
         exit 1
      fi
   else
      echo "max_num_vars should be 1500 or 3000"
      exit 1
   fi

# CA
elif [ "$instance_name" == "ca" ]; then
   penalty_weight=300
   init_temperature=50
   if [ "$max_num_vars" == 2000 ]; then
      max_num_cons=1400
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
   elif [ "$max_num_vars" == 4000 ]; then
      max_num_cons=2800
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
      echo "max_num_vars should be 2000 or 4000"
      exit 1
   fi

# CA
elif [ "$instance_name" == "ca_long" ]; then
   penalty_weight=300
   init_temperature=50
   if [ "$max_num_vars" == 2000 ]; then
      max_num_cons=1400
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
   elif [ "$max_num_vars" == 4000 ]; then
      max_num_cons=2800
      if [ "$t_schedule" == "exp_decay" ]; then
         step_limit=210000
      elif [ "$t_schedule" == "pt_exp_decay" ]; then
         step_limit=210000
      elif [ "$t_schedule" == "pen_pt_exp_decay" ]; then
         penalty_weight=400
         step_limit=210000
      else
         echo "t_schedule should be one of [exp_decay, pt_exp_decay]"
         exit 1
      fi
   else
      echo "max_num_vars should be 2000 or 4000"
      exit 1
   fi
# SC
elif [ "$instance_name" == "sc" ]; then
   penalty_weight=5
   init_temperature=1.0
   if [ "$max_num_vars" == 2000 ]; then
      max_num_cons=5000
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
   elif [ "$max_num_vars" == 4000 ]; then
      max_num_cons=5000
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
      echo "max_num_vars should be 2000 or 4000"
      exit 1
   fi


# SC
elif [ "$instance_name" == "sc_long" ]; then
   penalty_weight=5
   init_temperature=1.0
   if [ "$max_num_vars" == 2000 ]; then
      max_num_cons=5000
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
   elif [ "$max_num_vars" == 4000 ]; then
      max_num_cons=5000
      if [ "$t_schedule" == "exp_decay" ]; then
         step_limit=185000
      elif [ "$t_schedule" == "pt_exp_decay" ]; then
         step_limit=185000
      elif [ "$t_schedule" == "pen_pt_exp_decay" ]; then
         step_limit=185000
      else
         echo "t_schedule should be one of [exp_decay, pt_exp_decay]"
         exit 1
      fi
   else
      echo "max_num_vars should be 2000 or 4000"
      exit 1
   fi
# SC
elif [ "$instance_name" == "item" ]; then
   penalty_weight=10
   init_temperature=0.1
   if [ "$max_num_vars" == 1083 ]; then
      max_num_cons=5000
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
elif [ "$instance_name" == "anonymous" ]; then
   penalty_weight=10
   init_temperature=1.0
   if [ "$max_num_vars" == 1000 ]; then
      max_num_cons=5000
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

elif [ "$instance_name" == "loadBalancing" ]; then
   penalty_weight=10
   init_temperature=1.0
   if [ "$max_num_vars" == 61000 ]; then
      max_num_cons=65000
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
   echo "instance_name should be one of [mis, ca, sc, loadBalancing]"
   exit 1
fi
export XLA_PYTHON_CLIENT_MEM_FRACTION=.96
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
python -m PT_ILP.main_sampling \
   --sampler_config="PT_ILP/samplers/configs/${sampler?}_config.py" \
   --save_root=./PT_ILP/results --model=ilp \
   --instance_name=$instance_name --max_num_vars=${max_num_vars} --max_num_cons=${max_num_cons} \
   --penalty_weight=${penalty_weight} --formulation=${formulation}  \
   --num_instances=100 --num_models=1 --batch_size=15 --chain_length=100000 --l_min=$(echo "${penalty_weight}/2" | bc -l) --l_max=${penalty_weight} \
   --t_schedule=${t_schedule} --init_temperature=${init_temperature} --decay_rate=0.5 --reweight=None \
   --pt=deo --pt_interval=200 --t_min=${init_temperature} --t_max=$(echo "2*${init_temperature}" | bc -l) --log_every_steps=100 --mode=test_step --step_limit=${step_limit}
wait