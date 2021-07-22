import expert_finding.data.io
import expert_finding.main.fetch_data
# import expert_finding.main.fetch_data_mag
# import expert_finding.main.fetch_data_acm
from sklearn.preprocessing import normalize
import pickle
import expert_finding.main.evaluate
import expert_finding.main.topics
import expert_finding.data.io
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models, util
from sentence_transformers.readers import STSBenchmarkDataReader, InputExample
import csv

import logging
import expert_finding.main.language_models
import expert_finding.data.config
import os
import expert_finding.data.sets

current_folder = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_folder, 'output/', 'data/')
dump_dir = os.path.join(current_folder, 'output/', 'data_info')


def fetch(type=None, data_dir=None):
    parameters = {
        'output_dir': output_dir,
        'dump_dir': dump_dir,
        'type': type,
    }
    expert_finding.main.fetch_data.run(parameters)


def representation():
    datasets_versions = ["V1", "V2", "V3"]
    datasets_types = ["dataset_full", "dataset_cleaned", "dataset_associations"]
    for dv in datasets_versions:
        for dt in datasets_types:
            input_dir = os.path.join(current_folder, 'output/', 'data/', dv, dt)
            output_dir = os.path.join(input_dir, 'documents_representations')
            parameters = {
                'output_dir': output_dir,
                'input_dir': input_dir,
            }
            expert_finding.main.language_models.run(parameters)


fetch()
representation()
