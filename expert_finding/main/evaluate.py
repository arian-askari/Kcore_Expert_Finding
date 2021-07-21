# encoding: utf-8
import expert_finding.data.config
import os
import expert_finding.data.sets
import expert_finding.models
import expert_finding.evaluation.batch
import expert_finding.evaluation.visual
import expert_finding.models.voting_model as voter

import pkg_resources


def run(parameters):
    # Load config file
    config_path = pkg_resources.resource_filename("expert_finding", 'conf.yml')
    config = expert_finding.data.config.load_from_yaml(config_path)

    #  Load parameters
    working_dir = parameters["output_dir"]
    #  Load dataset "aminer"
    dataset_path = parameters["input_dir"]

    dataset = expert_finding.data.sets.DataSet("acm")
    dataset.load_data(dataset_path)
    eval_batch = expert_finding.evaluation.batch.EvalBatch(dataset, dump_dir=working_dir,
                                                                        max_queries=parameters["max_queries"])
    models_dict = {
        "vote": voter.VotingModel,
    }
    gathered_evaluations = dict()

    #  Get individual evaluations
    model_name = parameters["algorithm"]
    model = models_dict[model_name](config, **parameters)
    individual_evaluations = eval_batch.run_individual_evaluations(model, parameters=parameters)

    gathered_evaluations[model_name] = eval_batch.merge_evaluations(individual_evaluations)

    # Plot prec/rec and ROC curves and other metrics for all evaluation
    expert_finding.evaluation.visual.plot_evaluation(gathered_evaluations[model_name]["all"],
                                                                  prefix=model_name,
                                                                  path_visuals=working_dir, parameters=parameters)

