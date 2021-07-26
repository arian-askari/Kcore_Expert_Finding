import expert_finding.script.finding_pipline as expert
import expert_finding.finetuning.network as net_model
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
work_dir = os.path.join(current_dir, "output/data")
communities_path = os.path.join(work_dir, "triples")

data_type = "dataset_cleaned"
path = "PAP"
kcore = [4, 5, 5]
rate = [0.3, 0.3, 0.3]
version = ["V1", "V2", "V3"]

for i, v in enumerate(version):
    finder = expert.ExpertFind(version=v,
                               work_dir=work_dir,
                               communities_path=communities_path,
                               data_type=data_type,
                               sample_rate=rate[i],
                               meta_path=path,
                               kcore=rate[i])
    finder.start(m=1000)
