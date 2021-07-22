import expert_finding.script.finding_pipline as expert
import expert_finding.finetuning.network as net_model

data_version = ["V1", "V2", "V3"]
mate_paths = ["PAP"]
sample_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
kcore_sizes = [2, 3, 4, 5, 6, 7, 8, 9]
for version in data_version:
    for sample_rate in sample_rates:
        expert.ExpertFind(sample_rate=sample_rate)

    for k_size in kcore_sizes:
        expert.ExpertFind(kcore=k_size)

