import expert_finding.script.finding_pipline as expert
import expert_finding.finetuning.network as net_model

current_dir = os.path.dirname(os.path.abspath(__file__))
work_dir = os.path.join(current_dir, "output/data")
communities_path = os.path.join(work_dir, "triples")
mate_paths = ["PAP", "PTP", "PCP", "PAPPTP", "PAPPCP", "PCPPTP", "PAPPCPPTP"]
for path in mate_paths:
    expert.ExpertFind(meta_path=path)
