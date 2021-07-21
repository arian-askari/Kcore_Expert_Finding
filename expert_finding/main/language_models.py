import expert_finding.data.config
import expert_finding.language_models.wrapper
import expert_finding.data.sets
import expert_finding.data.io
import expert_finding.tools.graphs
import expert_finding.evaluation.visual


def run(parameters):
    output_dir = parameters["output_dir"]
    input_dir = parameters["input_dir"]
    expert_finding.data.io.check_and_create_dir(output_dir)
    expert_finding.data.io.check_and_create_dir(input_dir)

    print("Building language model")
    dataset = expert_finding.data.sets.DataSet()
    dataset.load_data(input_dir)
    expert_finding.language_models.wrapper.build_all(dataset.ds.documents, output_dir)
