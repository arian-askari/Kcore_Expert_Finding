import expert_finding.data.config
import os
import expert_finding.data.sets
import expert_finding.models
import expert_finding.evaluation.batch
import expert_finding.evaluation.visual

import pkg_resources

def run(parameters):
    # Load config file
    config_path = pkg_resources.resource_filename("expert_finding", 'conf.yml')
    config = expert_finding.data.config.load_from_yaml(config_path)

    # Load parameters
    working_dir = parameters["output_dir"]

    # Load dataset "aminer"
    dataset_path = parameters["input_dir"]
    dataset = expert_finding.data.sets.DataSet("acm")
    dataset.load_data(dataset_path)

    eval_batch = expert_finding.evaluation.batch.EvalBatch(
       dataset,
       dump_dir = working_dir,
       topics_as_queries = True
       )

    models_dict = {
        "vote": expert_finding.models.voting_model_LExR.VotingModel,
    }

    gathered_evaluations = dict()

    #  Get individual evaluations
    model_name = parameters["algorithm"]
    model = models_dict[model_name](config,**parameters)
    individual_evaluations = eval_batch.run_individual_evaluations(model,parameters=parameters)

    # Gather evaluations by all/topics
    gathered_evaluations[model_name] = eval_batch.merge_evaluations(individual_evaluations)

    prefix = model_name + "_topic_queries"

    # Plot prec/rec and ROC curves and other metrics for all evaluation
    expert_finding.evaluation.visual.plot_evaluation(gathered_evaluations[model_name]["all"],
                                                                  prefix=prefix,
                                                                  path_visuals=working_dir, parameters=parameters)

    # # Plot ROC curves for each topic
    # expert_finding.evaluation.visual.plot_ROC_topics(gathered_evaluations[model_name]["topics"], prefix=prefix,
    #                                                  path_visuals=working_dir)
    #
    # # Plot pre rec curves for each topic
    # expert_finding.evaluation.visual.plot_PreRec_topics(gathered_evaluations[model_name]["topics"], prefix=prefix,
    #                                                     path_visuals=working_dir)

