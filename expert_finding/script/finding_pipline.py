# encoding: utf-8
import sys
import os

import expert_finding.data.sets as  sets
from expert_finding.finetuning.generate import Generator
import expert_finding.finetuning.train as trainer
import expert_finding.script.document_embedding as saver
import expert_finding.script.experiment as evaluator
import expert_finding.data.io as io
import expert_finding.finetuning.network as net_model

# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class ExpertFind:
    def __init__(self, work_dir, communities_path , sample_rate=0.3, meta_path="PAP", data_type="datast_associations",
                 kcore=5, version="V1"):
        self.stage = 0
        self.core_k = kcore
        self.data_type = data_type
        self.work_dir = work_dir
        self.dataset = None
        self.version = version
        self.communities_dir = communities_path
        self.triples_dir = communities_path
        self.sample_rate = sample_rate
        self.meta_path = meta_path

        self.communities_name = self.version + "_" + str(sample_rate) + "_" + meta_path + "_" + str(
            kcore) + "_commities"
        self.triples_name = self.version + "_" + str(sample_rate) + "_" + meta_path + "_" + str(kcore) + "_triples"
        self.encoder_name = self.version + "_" + str(sample_rate) + "_" + meta_path + "_" + str(kcore) + "_core"

        self.model_path = os.path.join(self.work_dir, self.version, self.data_type, "output")
        self.embedding_path = os.path.join(self.work_dir, self.version, self.data_type, "embedding")
        self.generator = None

    def preprocess(self, type=None):
        dataset = sets.DataSet(type=type, name="data")
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
        print("Sampling and construct triplets (may take a few minutes)")
        seed_papers = self.generator.get_seed_paper()
        communities = self.generator.get_commities_muti(seed_papers=seed_papers, k=self.core_k,
                                                        communities_dir=self.communities_dir,
                                                        communities_name=self.communities_name,
                                                        meta_path=self.meta_path)
        self.generator.sample_triples_random(seed_papers=seed_papers, comunities=communities, T=T,
                                             triples_dir=self.triples_dir,
                                             triples_name=self.triples_name)

    def model_train(self):
        trainer.train(config=None, work_dir=self.work_dir, triples_dir=self.triples_dir, triples_name=self.triples_name,
                      version=self.version,
                      encoder_name=self.encoder_name)

    def offline_embedding(self, encoder_name=None):
        if encoder_name is None:
            saver.save_embedding(save_dir=self.embedding_path, model_dir=self.model_path, model_name=self.encoder_name,
                                 dataset=self.dataset)
        else:
            saver.save_embedding(save_dir=self.embedding_path, model_dir=self.model_path, model_name=encoder_name,
                                 dataset=self.dataset)

    def get_embedding(self, encoder_name=None):
        self.preprocess()
        saver.query_time(save_dir=self.embedding_path, model_dir=self.model_path, model_name=self.encoder_name,
                         dataset=self.dataset)

    def evalutation(self, index, m, encoder_name=None):
        if encoder_name is None:
            evaluator.run(version=self.version, model_dir=self.embedding_path, model_name=self.encoder_name,
                          work_dir=self.work_dir, index=index, m=m, dataset_type=data_type)
        else:
            evaluator.run(version=self.version, model_dir=self.embedding_path, model_name=encoder_name,
                          work_dir=self.work_dir, index=index, m=m)

    def start(self, m):
        self.preprocess()
        if not os.path.exists(os.path.join(self.triples_dir, self.triples_name)):  # True/False
            self.init_triples()
        if not os.path.exists(os.path.join(self.model_path, self.encoder_name)):
            self.model_train()
        if not os.path.exists(os.path.join(self.embedding_path, self.encoder_name)):
            self.offline_embedding()
        self.evalutation(index=True, m=m)

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
        self.evalutation(index=True, m=1000)

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
        self.evalutation(index=True, m=1000)


def sample_rate():
    finder = ExpertFind(version="V2", work_dir="/ddisk/lj/DBLP/data/", communities_path="/ddisk/lj/triples",
                        sample_rate=0.3, meta_path="PAP")
    finder.pg_index_start()



current_dir = os.path.dirname(os.path.abspath(__file__))
work_dir = os.path.join(current_dir, "output/data")
communities_path = os.path.join(work_dir, "triples")




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
    finder = ExpertFind("V3", work_dir, communities_path, 0.3, "PAP", kcore=15)

    finder.preprocess()
    # finder.start()
    finder.get_embedding()


### ============================other model.=================================



def kcorebase_embeding_finding():
    finder = ExpertFind("V3", "/ddisk/lj/DBLP/data/", "/ddisk/lj/triples", 1.0, "PAP", k=15)
    finder.force_start()


def network_embedding():
    models = ["TADW", "GVNRT", "IDNE", "G2G"]
    finder = net_model.Network("V3", "/ddisk/lj/DBLP/data/", "/ddisk/lj/triples", 1.0, "PAP", k=15)
    finder.start()


## different  meta-path
def print_stats():
    # fetcher.fetch()
    dataset = sets.DataSet()
    data_path = os.path.join("/ddisk/lj/DBLP/data/", "V1", "dataset_full")
    dataset.load_data(data_path)
    dataset.print_stats()
