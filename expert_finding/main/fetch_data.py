import expert_finding.data.config
import expert_finding.data.io
import expert_finding.data.sets
import expert_finding.evaluation.visual
import expert_finding.tools.graphs

import os
import patoolib
import zipfile
import pkg_resources
import urllib.request

"""
parameters = {
            'output_dir': "/ddisk/lj/DBLP/databak",
            'dump_dir': "/ddisk/lj/DBLP/data_info",
            'type': type,
        }
"""


def run(parameters):
    config_path = pkg_resources.resource_filename("expert_finding", 'conf.yml')
    config = expert_finding.data.config.load_from_yaml(config_path)
    # build_v1(config, parameters)
    build_v1(config, parameters)


def build_v1(config, parameters):
    #  Dataset V1
    print("Downloading and build dataset V1... (may take a few minutes)")
    data_folder = os.path.join(parameters["output_dir"], "V1")
    expert_finding.data.io.check_and_create_dir(data_folder)
    dest_file = os.path.join(data_folder, config["data_citation_network_rar_file_name_v1"])
    text_file = os.path.join(data_folder, config["data_citation_network_text_file_name_v1"])

    full_path = os.path.join(data_folder, "dataset_full")
    full_stats_path = os.path.join(parameters["dump_dir"], "dataset_v1_stats_full")
    expert_finding.data.io.check_and_create_dir(full_path)
    expert_finding.data.io.check_and_create_dir(full_stats_path)
    input_file = os.path.join(data_folder, config["data_citation_network_text_file_name_v1"])
    dataset = expert_finding.data.sets.DataSet("aminer",
                                                            input_file,
                                                            config["data_experts_folder"],
                                                            version="V1", type=parameters['type'])

    expert_finding.evaluation.visual.plot_stats(dataset,
                                                             full_stats_path,
                                                             min_documents=100,
                                                             min_in_citations=100,
                                                             min_out_citations=100)
    dataset.save(full_path)

    print("Creating cleaned dataset")
    cleaned_path = os.path.join(data_folder, "dataset_cleaned")
    cleaned_stats_path = os.path.join(parameters["dump_dir"], "dataset_v1_stats_cleaned")
    expert_finding.data.io.check_and_create_dir(cleaned_path)
    expert_finding.data.io.check_and_create_dir(cleaned_stats_path)
    dataset.clean_associations(max_documents_per_candidates=100, min_documents_per_candidates=1)
    expert_finding.evaluation.visual.plot_stats(dataset,
                                                             cleaned_stats_path,
                                                             min_documents=10,
                                                             min_in_citations=10,
                                                             min_out_citations=10
                                                             )
    dataset.save(cleaned_path)

    # Build subgraph for associations
    print("Creating associations sub graph dataset")
    associations_path = os.path.join(data_folder, "dataset_associations")
    associations_stats_path = os.path.join(parameters["dump_dir"], "dataset_v1_stats_associations")
    expert_finding.data.io.check_and_create_dir(associations_path)
    expert_finding.data.io.check_and_create_dir(associations_stats_path)
    dataset.clean_associations(max_documents_per_candidates=100, min_documents_per_candidates=1)
    documents_set, candidates_set = expert_finding.tools.graphs.extract_experts_associations_subgraph(
        dataset,
        length_walks=5,
        number_of_walks=50
    )
    dataset.reduce(documents_set, candidates_set)
    expert_finding.evaluation.visual.plot_stats(dataset,
                                                             associations_stats_path,
                                                             min_documents=10,
                                                             min_in_citations=10,
                                                             min_out_citations=10
                                                             )
    dataset.save(associations_path)


def build_v2(config, parameters):
    # #  Dataset V2

    print("Downloading and build dataset V2... (may take a few minutes)")
    data_folder = os.path.join(parameters["output_dir"], "V2")
    expert_finding.data.io.check_and_create_dir(data_folder)
    dest_file = os.path.join(data_folder, config["data_citation_network_rar_file_name_v2"])
    text_file = os.path.join(data_folder, config["data_citation_network_text_file_name_v2"])

    try:
        os.remove(dest_file)
        os.remove(text_file)
    except OSError:
        pass
    urllib.request.urlretrieve(config["data_citation_network_url_v2"], dest_file)
    zip_ref = zipfile.ZipFile(dest_file, 'r')
    zip_ref.extractall(data_folder)
    zip_ref.close()

    full_path = os.path.join(data_folder, "dataset_full")
    full_stats_path = os.path.join(parameters["dump_dir"], "dataset_v2_stats_full")
    expert_finding.data.io.check_and_create_dir(full_path)
    expert_finding.data.io.check_and_create_dir(full_stats_path)
    input_file = os.path.join(data_folder, config["data_citation_network_text_file_name_v2"])
    dataset = expert_finding.data.sets.DataSet("aminer", input_file, config["data_experts_folder"],
                                                            version="V2")
    expert_finding.evaluation.visual.plot_stats(dataset,
                                                             full_stats_path,
                                                             min_documents=100,
                                                             min_in_citations=100,
                                                             min_out_citations=100)
    dataset.save(full_path)

    print("Creating cleaned dataset")
    cleaned_path = os.path.join(data_folder, "dataset_cleaned")
    cleaned_stats_path = os.path.join(parameters["dump_dir"], "dataset_v2_stats_cleaned")
    expert_finding.data.io.check_and_create_dir(cleaned_path)
    expert_finding.data.io.check_and_create_dir(cleaned_stats_path)
    dataset.clean_associations(max_documents_per_candidates=100, min_documents_per_candidates=1)
    expert_finding.evaluation.visual.plot_stats(dataset,
                                                             cleaned_stats_path,
                                                             min_documents=5,
                                                             min_in_citations=5,
                                                             min_out_citations=5
                                                             )
    dataset.save(cleaned_path)

    # Build subgraph for associations
    print("Creating associations sub graph dataset")
    associations_path = os.path.join(data_folder, "dataset_associations")
    associations_stats_path = os.path.join(parameters["dump_dir"], "dataset_v2_stats_associations")
    expert_finding.data.io.check_and_create_dir(associations_path)
    expert_finding.data.io.check_and_create_dir(associations_stats_path)
    dataset.clean_associations(max_documents_per_candidates=100, min_documents_per_candidates=1)
    documents_set, candidates_set = expert_finding.tools.graphs.extract_experts_associations_subgraph(
        dataset,
        length_walks=5,
        number_of_walks=50
    )
    dataset.reduce(documents_set, candidates_set)
    expert_finding.evaluation.visual.plot_stats(dataset,
                                                             associations_stats_path,
                                                             min_documents=10,
                                                             min_in_citations=10,
                                                             min_out_citations=10
                                                             )
    dataset.save(associations_path)


def build_DBLP_v4(config, parameters):
    print("Downloading and build dataset V4... (may take a few minutes)")
    data_folder = os.path.join(parameters["output_dir"], "V4")
    expert_finding.data.io.check_and_create_dir(data_folder)
    # dest_file = os.path.join(data_folder, config["data_citation_network_rar_file_name_v4"])
    # text_file = os.path.join(data_folder, config["data_citation_network_text_file_name_v4"])

    full_path = os.path.join(data_folder, "dataset_full")
    full_stats_path = os.path.join(parameters["dump_dir"], "dataset_v4_stats_full")
    expert_finding.data.io.check_and_create_dir(full_path)
    expert_finding.data.io.check_and_create_dir(full_stats_path)
    input_file = os.path.join(data_folder, config["data_citation_network_text_file_name_v4"])
    dataset = expert_finding.data.sets.DataSet("aminer",
                                                            input_file,
                                                            config["data_experts_folder"],
                                                            version="V4", type=parameters['type'])

    expert_finding.evaluation.visual.plot_stats(dataset,
                                                             full_stats_path,
                                                             min_documents=100,
                                                             min_in_citations=100,
                                                             min_out_citations=100)
    dataset.save(full_path)

    print("Creating cleaned dataset")
    cleaned_path = os.path.join(data_folder, "dataset_cleaned")
    cleaned_stats_path = os.path.join(parameters["dump_dir"], "dataset_v4_stats_cleaned")
    expert_finding.data.io.check_and_create_dir(cleaned_path)
    expert_finding.data.io.check_and_create_dir(cleaned_stats_path)
    dataset.clean_associations(max_documents_per_candidates=100, min_documents_per_candidates=1)
    expert_finding.evaluation.visual.plot_stats(dataset,
                                                             cleaned_stats_path,
                                                             min_documents=10,
                                                             min_in_citations=10,
                                                             min_out_citations=10
                                                             )
    dataset.save(cleaned_path)

    # Build subgraph for associations
    print("Creating associations sub graph dataset")
    associations_path = os.path.join(data_folder, "dataset_associations")
    associations_stats_path = os.path.join(parameters["dump_dir"], "dataset_v4_stats_associations")
    expert_finding.data.io.check_and_create_dir(associations_path)
    expert_finding.data.io.check_and_create_dir(associations_stats_path)
    dataset.clean_associations(max_documents_per_candidates=100, min_documents_per_candidates=1)
    documents_set, candidates_set = expert_finding.tools.graphs.extract_experts_associations_subgraph(
        dataset,
        length_walks=5,
        number_of_walks=50
    )
    dataset.reduce(documents_set, candidates_set)
    expert_finding.evaluation.visual.plot_stats(dataset,
                                                             associations_stats_path,
                                                             min_documents=10,
                                                             min_in_citations=10,
                                                             min_out_citations=10
                                                             )
    dataset.save(associations_path)

