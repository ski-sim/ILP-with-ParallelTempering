"""Main Config Structure."""

from ml_collections import config_dict


def get_config(FLAGS):
    """Get common config sketch."""
    general_config = dict(
        model=dict(
            name=FLAGS.model,
            graph_type=FLAGS.graph_type,
            data_root=FLAGS.data_root,
            penalty=FLAGS.penalty_weight,
            save_dir_name=FLAGS.save_dir_name,
            num_models=FLAGS.num_models,
            max_num_vars=FLAGS.max_num_vars,
            max_num_cons=FLAGS.max_num_cons,
            num_instances=FLAGS.num_instances,
            num_categories=FLAGS.num_categories,
            shape=(0,),
            formulation=FLAGS.formulation,  # indicator
            mode=FLAGS.mode,
            max_runtime=FLAGS.max_runtime,
            step_limit=FLAGS.step_limit,
        ),
        sampler=dict(
            name="",
        ),
        experiment=dict(
            num_models=FLAGS.num_models,
            run_parallel=True,
            get_additional_metrics=False,
            save_samples=False,
            use_tqdm=False,
            co_opt_prob=False,
            batch_size=FLAGS.batch_size,
            t_schedule=FLAGS.t_schedule,
            chain_length=FLAGS.chain_length,
            log_every_steps=FLAGS.log_every_steps,
            init_temperature=FLAGS.init_temperature,
            decay_rate=FLAGS.decay_rate,
            final_temperature=FLAGS.final_temperature,
            save_root=FLAGS.save_root,
            pt=FLAGS.pt,
            pt_interval=FLAGS.pt_interval,
            t_min=FLAGS.t_min,
            t_max=FLAGS.t_max,
            l_min=FLAGS.l_min,
            l_max=FLAGS.l_max,
            reweight=FLAGS.reweight,
            lp_interval=FLAGS.lp_interval,
        ),
    )
    return config_dict.ConfigDict(general_config)
