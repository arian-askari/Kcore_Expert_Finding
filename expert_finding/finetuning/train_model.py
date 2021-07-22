import os
import expert_finding.finetuning.train as trainer

current_dir = os.path.dirname(os.path.abspath(__file__))

work_dir = os.path.join(current_dir, "output/data")
communities_path = os.path.join(work_dir, "triples")


trainer.train(config=None, work_dir=work_dir, triples_dir="", triples_name="",
              version="",
              encoder_name="")
