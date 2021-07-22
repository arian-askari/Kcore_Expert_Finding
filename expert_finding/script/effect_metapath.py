import expert_finding.script.finding_pipline as expert
import expert_finding.finetuning.network as net_model

mate_paths = ["PAP", "PTP", "PCP", "PAPPTP", "PAPPCP", "PCPPTP", "PAPPCPPTP"]
for path in mate_paths:
    expert.ExpertFind(meta_path=path)

