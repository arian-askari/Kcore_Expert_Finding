# encoding: utf-8
import sys
import os

root_path = "/ddisk/lj/tmp/pycharm_project_109"
sys.path.append(root_path)

import expert_finding.data.sets as  sets
from expert_finding.finetuning.generate import Generator
import expert_finding.finetuning.train as trainer
import expert_finding.script.document_embedding as saver
import expert_finding.script.experiment as evaluator
import expert_finding.data.io as io
import expert_finding.script.fetch as fetcher
import numpy as np
import faiss
import expert_finding.script.document_embedding  as embedded
import time

# parameter = {
#     'model_dir': "/ddisk/lj/DBLP/data/V2/dataset_associations/embedding",
#     'model_name': "V2_0.4_PAP"
# }

parameters = {
    'output_dir': "/ddisk/lj/DBLP/data",
    'dump_dir': "/ddisk/lj/DBLP/data_info"
}

# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class ExpertFind:
    def __init__(self, version, work_dir, communities_path, sample_rate, meta_path, data_type, k=5):
        self.stage = 0
        self.core_k = k
        self.data_type = data_type
        self.work_dir = work_dir
        self.dataset = None
        self.version = version
        self.communities_dir = communities_path
        self.triples_dir = communities_path
        self.sample_rate = sample_rate
        self.meta_path = meta_path

        self.communities_name = self.version + "_" + str(sample_rate) + "_" + meta_path + "_" + str(k) + "_commities"
        self.triples_name = self.version + "_" + str(sample_rate) + "_" + meta_path + "_" + str(k) + "_triples"
        self.encoder_name = self.version + "_" + str(sample_rate) + "_" + meta_path + "_" + str(k) + "_core"

        self.model_path = os.path.join(self.work_dir, self.version, self.data_type, "output")
        self.embedding_path = os.path.join(self.work_dir, self.version, self.data_type, "embedding")
        self.generator = None

    def preprocess(self, type=None):
        dataset = sets.DataSet(type=type)
        data_path = os.path.join(self.work_dir, self.version, self.data_type)
        dataset.load_data(data_path)
        self.dataset = dataset
        self.generator = Generator(self.version, self.work_dir, self.communities_dir, self.sample_rate, self.meta_path,
                                   self.dataset)

    def init_triples(self):
        """
        experts = dataset.ds.candidates
        A_dd = dataset.ds.citations
        A_da = dataset.ds.associations
        # A_da_weight = dataset.ds.associations_weight
        A_ad = A_da.transpose()
        T = dataset.ds.documents
        G_ta = dataset.gt.associations
        :return:
        """
        T = self.dataset.ds.documents
        A_da = self.dataset.ds.associations
        A_ad = A_da.transpose()
        seed_papers = self.generator.get_seed_paper(self.sample_rate)
        communities, nears = self.generator.get_commities_muti(seed_papers=seed_papers, k=4,
                                                               communities_dir=self.communities_dir,
                                                               communities_name=self.communities_name,
                                                               meta_path=self.meta_path)
        self.generator.sample_triples_random(seed_papers=seed_papers, comunities=communities, T=T,
                                             triples_dir=self.triples_dir,
                                             triples_name=self.triples_name, core_k=self.core_k)

    def model_train(self):
        trainer.train(config=None, work_dir=self.work_dir, triples_dir=self.triples_dir, triples_name=self.triples_name,
                      version=self.version,
                      encoder_name=self.encoder_name)

    def offline_embedding(self, encoder_name=None):
        """
        :param encoder_name: 根据encoder姓名. 判断是否需要重新训练.
        :return:
        """
        if encoder_name is None:
            saver.save_embedding(save_dir=self.embedding_path, model_dir=self.model_path, model_name=self.encoder_name,
                                 dataset=self.dataset)
        else:
            saver.save_embedding(save_dir=self.embedding_path, model_dir=self.model_path, model_name=encoder_name,
                                 dataset=self.dataset)

    def get_embedding(self, encoder_name=None):
        self.preprocess()
        # embeddings = saver.get_embedding(save_dir=self.embedding_path, model_dir=self.model_path, model_name=self.encoder_name,
        #                          dataset=self.dataset)
        # saver.build_NSG_index(save_dir=self.embedding_path, model_dir=self.model_path, model_name=self.encoder_name,
        #                       dataset=self.dataset)
        saver.query_time(save_dir=self.embedding_path, model_dir=self.model_path, model_name=self.encoder_name,
                         dataset=self.dataset)

    def evalutation(self, index, k, encoder_name=None):
        if encoder_name is None:
            evaluator.run(version=self.version, model_dir=self.embedding_path, model_name=self.encoder_name,
                          work_dir=self.work_dir, index=index, k=k)
        else:
            evaluator.run(version=self.version, model_dir=self.embedding_path, model_name=encoder_name,
                          work_dir=self.work_dir, index=index, k=k)

    def start(self, k):
        self.preprocess()
        if not os.path.exists(os.path.join(self.triples_dir, self.triples_name)):  # True/False
            self.init_triples()
        if not os.path.exists(os.path.join(self.model_path, self.encoder_name)):
            self.model_train()
        if not os.path.exists(os.path.join(self.embedding_path, self.encoder_name)):
            self.offline_embedding()
        self.evalutation(index=True, k=k)

    def meta_path_start(self):
        self.preprocess()
        print(self.triples_name)
        if not os.path.exists(os.path.join(self.triples_dir, self.triples_name)):  # True/False
            print("sampling...", self.triples_name)
            if not os.path.exists(os.path.join(self.communities_dir, "muti_seed")):
                seed_papers = self.generator.get_seed_paper(self.sample_rate, self.dataset.ds.documents)
                io.save_as_json(self.communities_dir, "muti_seed", seed_papers)

            seed_papers = io.load_as_json(self.communities_dir, "muti_seed")
            communities = self.generator.get_commities_muti(seed_papers=seed_papers, k=self.core_k,
                                                            dataset=self.dataset,
                                                            communities_dir=self.communities_dir,
                                                            communities_name=self.communities_name,
                                                            meta_path=self.meta_path)

            self.generator.sample_triples_near_negatives(seed_papers=seed_papers, comunities=communities,
                                                         T=self.dataset.ds.documents,
                                                         triples_dir=self.triples_dir,
                                                         triples_name=self.triples_name, core_k=self.core_k)

        if not os.path.exists(os.path.join(self.model_path, self.encoder_name)):
            self.model_train()
        if not os.path.exists(os.path.join(self.embedding_path, self.encoder_name)):
            self.offline_embedding()
        self.evalutation(index=True, k=1000)

    def pg_index_start(self):
        self.preprocess()
        if not os.path.exists(os.path.join(self.triples_dir, self.triples_name)):  # True/False
            self.init_triples()
        if not os.path.exists(os.path.join(self.model_path, self.encoder_name)):
            self.model_train()
        if not os.path.exists(os.path.join(self.embedding_path, self.encoder_name + "PG")):
            self.offline_embedding()
            self.evalutation(index=True)
        self.evalutation(index=False)

    def force_start(self):

        self.preprocess()
        self.init_triples()
        self.model_train()
        self.offline_embedding()
        self.evalutation(index=True, k=1000)


def sample_rate():
    finder = ExpertFind(version="V2", work_dir="/ddisk/lj/DBLP/data/", communities_path="/ddisk/lj/triples",
                        sample_rate=0.3, meta_path="PAP")
    finder.pg_index_start()


# sample_rate()
# #
# rate = [0.1, 0.2, 0.3, 0.4, 0.5]
# for i in range(3, 5):
#     finder = ExpertFind(version="V2", work_dir="/ddisk/lj/DBLP/wdata/", communities_path="/ddisk/lj/wtriples",
#                         sample_rate=rate[i],
#                         meta_path="PAP")
#     finder.weight_start()


parameters = {
    'output_dir': "/ddisk/lj/DBLP/data",
    'dump_dir': "/ddisk/lj/DBLP/data_info"
}

work_dir = "/ddisk/lj/DBLP/data/"
communities_path = "/ddisk/lj/tripes"

rate = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
version = ["V1"]
kcore = [2, 3, 4, 5, 6, 7, 8, 9, 10]

for v in version:
    finder = ExpertFind(version=v,
                        work_dir=work_dir,
                        communities_path=communities_path,
                        sample_rate=0.1,
                        meta_path="PAP",
                        k=5)
    finder.start(k=1000)
    # finder.force_start()
    # finder.preprocess()
    # finder.offline_embedding()
    # for d in [50, 100, 200, 500, 1000]:
    # finder.start(k=1000)
    # query_1864 = finder.dataset.ds.documents[1864]
    # print(query_1864)
    # query_1744 = finder.dataset.ds.documents[1744]
    # print(query_1744)
    # # d_a = A_da.toarray()
    # d_a = finder.dataset.ds.associations.toarray()
    # a_d = d_a.T
    # print(np.flatnonzero(a_d[164])) # 164 专家的所有参与的文章.
    # for document in np.flatnonzero(a_d[164]): # 其参与的文章中的其他作者.
    #     print("co_author's documets: ", np.flatnonzero(d_a[document]))

    # print(co_authors)

    # authors_index = np.flatnonzero(d_a[doc])

    # net
    # print("38", finder.dataset.gt.candidates[32]) #[ 41 174  11   5  21]
    # print("68", finder.dataset.gt.candidates[203]) # [205  38  68  44 164]

    # print(finder.dataset.gt.associations[2]) # bingo 4 correct [ 38  44  68 105  19]


# # #

# finder.preprocess()
# finder.init_triples()
################################overhead############################
# finder = ExpertFind("V4", "/ddisk/lj/DBLP/data/", "/ddisk/lj/triples", 1.0, "PAP", k=15)
# finder.start()

# finder.pg_index_start()
def query_time():
    finder = ExpertFind("V3", "/ddisk/lj/DBLP/data/", "/ddisk/lj/triples", 1.0, "PAP", k=15)
    finder.preprocess()
    # finder.start()
    finder.get_embedding()


# query_time()

### ============================other model.=================================
def other():
    models = ["tfidf", "GloVe", "sci_bert_nil_sts"]
    for model in models:
        finder = ExpertFind("V1", "/ddisk/lj/DBLP/data/", "/ddisk/lj/triples", "0", model)
        finder.preprocess()
        # finder.offline_embedding(encoder_name="GloVe")
        finder.evalutation(index=True, k=1000, encoder_name=model)

    finder = ExpertFind("V4", "/ddisk/lj/DBLP/data/", "/ddisk/lj/triples", "0", "GloVe")
    finder.preprocess()
    # finder.offline_embedding(encoder_name="GloVe")
    finder.evalutation(index=True, k=1000, encoder_name="GloVe")
    #
    finder = ExpertFind("V1", "/ddisk/lj/DBLP/data/", "/ddisk/lj/triples", "0", "sci_bert_nil_sts")
    finder.preprocess()
    # finder.offline_embedding(encoder_name="sci_bert_nil_sts")
    finder.evalutation(index=True, k=1000, encoder_name="sci_bert_nil_sts")


## different  meta-path
def print_stats():
    # fetcher.fetch()
    dataset = sets.DataSet("aminer")
    data_path = os.path.join("/ddisk/lj/DBLP/data/", "V1", "dataset_full")
    dataset.load(data_path)
    dataset.print_stats()
