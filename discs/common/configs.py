"""Main Config Structure."""

from ml_collections import config_dict


def get_config(FLAGS):
    """Get common config sketch."""
    general_config = dict(
        model=dict(
            name=FLAGS.model,
            graph_type=FLAGS.graph_type,
            cfg_str=FLAGS.cfg_str,
            data_root=FLAGS.data_root,
            penalty=FLAGS.penalty_weight,
            save_dir_name=FLAGS.save_dir_name,
            num_models=FLAGS.num_models,
            max_num_nodes=FLAGS.max_num_nodes,
            max_num_constraints=FLAGS.max_num_constraints,
            num_instances=FLAGS.num_instances,
            num_categories=FLAGS.num_categories,
            shape=(0,),
            rand_type=FLAGS.rand_type,
            formulation=FLAGS.formulation,  # indicator
            mode=FLAGS.mode,
            max_runtime=FLAGS.max_runtime,
        ),
        sampler=dict(
            name="",
        ),
        experiment=dict(
            name="CO_Experiment",
            evaluator="co_eval",
            num_models=FLAGS.num_models,
            ess_ratio=0.5,
            run_parallel=True,
            get_additional_metrics=False,
            shuffle_buffer_size=0,
            plot_every_steps=10,
            fig_folder="",
            save_samples=False,
            get_estimation_error=False,
            use_tqdm=False,
            co_opt_prob=False,
            window_size=10,
            window_stride=10,
            batch_size=FLAGS.batch_size,
            t_schedule=FLAGS.t_schedule,
            chain_length=FLAGS.chain_length,
            log_every_steps=FLAGS.log_every_steps,
            save_every_steps=FLAGS.save_every_steps,
            init_temperature=FLAGS.init_temperature,
            decay_rate=FLAGS.decay_rate,
            final_temperature=FLAGS.final_temperature,
            save_root=FLAGS.save_root,
            pt=FLAGS.pt,
            pt_interval=FLAGS.pt_interval,
            t_min=FLAGS.t_min,
            t_max=FLAGS.t_max,
            reweight=FLAGS.reweight,
        ),
    )
    return config_dict.ConfigDict(general_config)
