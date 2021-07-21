# encoding: utf-8
import expert_finding.data.sets as  sets
from expert_finding.finetuning.generate import Generator
import expert_finding.finetuning.train as trainer
import expert_finding.script.document_embedding as saver
import expert_finding.models.tadw as tadw
import expert_finding.models.gvnrt as  gvnrt
# import expert_finding.models.post_ane_model as post
import expert_finding.models.graph2gauss as g2g
import expert_finding.models.idne as idne
import expert_finding.script.experiment as evaluator

import expert_finding.data.io as io

import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class Network:
    def __init__(self, version, work_dir, model_name, data_type):
        self.stage = 0
        self.data_type = data_type
        self.work_dir = work_dir
        self.dataset = None
        self.version = version
        self.model_name = model_name
        self.model_path = os.path.join(self.work_dir, self.version, self.data_type, "output")
        self.embedding_path = os.path.join(self.work_dir, self.version, self.data_type, "embedding")
        self.model_save = self.version + "_" + model_name

    def preprocess(self):
        """
        load the dataset.
        :return:
        """
        dataset = sets.DataSet()
        data_path = os.path.join(self.work_dir, self.version, self.data_type)
        dataset.load_data(data_path)
        self.dataset = dataset

    def print_stats(self):
        self.dataset.print_stats()

    def network_train(self):
        A_da = self.dataset.ds.associations
        A_dd = self.dataset.ds.citations

        documents = self.dataset.ds.documents
        dd = A_da @ A_da.T  # 矩阵乘法. ->  i,j = i 行 * j 列
        dd.setdiag(0)  ## PAP. dd之间的关系.
        network = dd + A_dd

        print("init_network_embedding_model: ", self.model_name)
        if self.model_name is "GVNRT":
            model = g2g.Model()
            model.fit(network, documents)
            io.save_as_json(self.embedding_path, self.model_save, model.get_embeddings())
        elif self.model_name is "TADW":
            model = gvnrt.Model()
            model.fit(network, documents)
            io.save_as_json(self.embedding_path, self.model_save, model.get_embeddings())
        elif self.model_name is "IDNE":
            model = tadw.Model()
            model.fit(network, documents)
            io.save_as_json(self.embedding_path, self.model_save, model.get_embeddings())
        else:
            model = idne.Model()
            model.fit(network, documents)
            io.save_as_json(self.embedding_path, self.model_save, model.get_embeddings())

        saver.build_PG_index(save_dir=self.embedding_path, model_name=self.model_save)

    def evalutation(self, index, encoder_name=None):
        evaluator.run(version=self.version, model_dir=self.embedding_path, model_name=self.model_save,
                      work_dir=self.work_dir, index=index, k=1000)

    def start(self):
        self.preprocess()
        if not os.path.exists(os.path.join(self.embedding_path, self.model_save + "PG")):
            self.network_train()
        self.evalutation(index=True)

