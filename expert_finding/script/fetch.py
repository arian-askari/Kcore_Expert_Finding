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

def fetch(type=None):

    parameters = {
        'output_dir': "/ddisk/lj/DBLP/data",
        'dump_dir': "/ddisk/lj/DBLP/data_info",
        'type': type,
    }
    expert_finding.main.fetch_data.run(parameters)

def representation(data_name, data_type):
    datasets_versions = ["/V1"]
    datasets_types = ["/dataset_associations"]
    for dv in datasets_versions:
        for dt in datasets_types:
            input_dir = "/ddisk/lj/DBLP/data/" + dv + dt
            output_dir = os.path.join(input_dir, 'documents_representations')
            parameters = {
                'output_dir': output_dir,
                'input_dir': input_dir,
            }
            expert_finding.main.language_models.run(parameters)






