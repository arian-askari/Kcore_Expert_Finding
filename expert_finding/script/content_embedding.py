import expert_finding.script.finding_pipline as expert
import expert_finding.finetuning.network as net_model
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
work_dir = os.path.join(current_dir, "output/data")


def content_embedding_finding():
    models = ["tfidf", "GloVe", "sci_bert_nil_sts"]
    for model in models:
        finder = ExpertFind("V1", work_dir, communities_path, "0", model)
        finder.preprocess()
        finder.offline_embedding(encoder_name="GloVe")
        finder.evalutation(index=True, m=1000, encoder_name=model)

    finder = ExpertFind("V2", "/ddisk/lj/DBLP/data/", "/ddisk/lj/triples", "0", "GloVe")
    finder.preprocess()
    # finder.offline_embedding(encoder_name="GloVe")
    finder.evalutation(index=True, k=1000, encoder_name="GloVe")
    #
    finder = ExpertFind("V3", "/ddisk/lj/DBLP/data/", "/ddisk/lj/triples", "0", "sci_bert_nil_sts")
    finder.preprocess()
    # finder.offline_embedding(encoder_name="sci_bert_nil_sts")
    finder.evalutation(index=True, k=1000, encoder_name="sci_bert_nil_sts")
