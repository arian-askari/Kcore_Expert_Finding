# encoding: utf-8
import expert_finding.data.io_aminer
import expert_finding.data.io
import numpy as np
import scipy.sparse
import os


class DataFrame:
    def __init__(self, name):
        self._name = name  # name related to the origin of the data
        self.documents = list()  # list that contains the documents' contents (title+abstract)
        self.candidates = list()  # list that contains the candidates' identifications
        self.venues = list()
        self.publish = None
        self.associations = None  # sparse associations between documents (rows) and candidates (cols)
        self.citations = None  # sparse citations graph between documents (rows, cols)
        self.participate = list()


class GroundTruth:
    def __init__(self, name):
        self._name = name  # name related to the origin of the data
        self.topics = list()  #  list of topics
        self.candidates = list()  # list of candidates identifications # 
        self.experts_mask = list()  # mask to extract experts from associations [indices]
        self.associations = None  # sparse associations between topics (rows) and candidates(cols)


class DataSet:
    def __init__(self, name, documents_file_path=None, experts_folder_path=None, version=None, type=None):
        self._name = name
        self.ds = DataFrame(name)
        self.gt = GroundTruth(name)
        self.version = version
        if documents_file_path is not None and experts_folder_path is not None:
            self.getDataset(documents_file_path, experts_folder_path, self.version, type)

    def reduce(self, documents_set, candidates_set):
        print("reducing with documents_set", len(documents_set), "candidates_set", len(candidates_set))
        docs_mask = documents_set
        cands_mask = candidates_set
        cands_boolean_mask = np.zeros(len(self.ds.candidates), dtype=np.bool_)
        cands_boolean_mask[cands_mask] = True

        #  Update experts
        experts_boolean_mask = np.zeros(len(cands_boolean_mask), dtype=np.bool_)
        experts_boolean_mask[self.gt.experts_mask] = True
        experts_boolean_mask = experts_boolean_mask[experts_boolean_mask]
        self.gt.experts_mask = experts_boolean_mask.nonzero()[0]

        # Update associations
        self.ds.associations = self.ds.associations[docs_mask][:, cands_mask]
        self.gt.associations = self.gt.associations[:, cands_mask]
        topics_mask = (np.squeeze(np.asarray(self.gt.associations.sum(axis=1))) > 0).nonzero()[0]
        self.gt.associations = self.gt.associations[topics_mask]
        self.ds.citations = self.ds.citations[docs_mask][:, docs_mask]
        self.ds.citations.eliminate_zeros()
        self.ds.associations.eliminate_zeros()
        self.gt.associations.eliminate_zeros()

        # Updates attributes
        mask = set(topics_mask)
        self.gt.topics = [e for i, e in enumerate(self.gt.topics) if i in mask]
        mask = set(cands_mask)
        self.gt.candidates = [e for i, e in enumerate(self.gt.candidates) if i in mask]
        mask = set(docs_mask)
        self.ds.documents = [e for i, e in enumerate(self.ds.documents) if i in mask]
        mask = set(cands_mask)
        self.ds.candidates = [e for i, e in enumerate(self.ds.candidates) if i in mask]
        self.print_stats()


    def clean_associations(self, max_documents_per_candidates=None, min_documents_per_candidates=None):
        num_docs_per_can = np.squeeze(np.asarray(self.ds.associations.sum(axis=0)))

        if max_documents_per_candidates is None:
            max_documents_per_candidates = num_docs_per_can.max()
        if min_documents_per_candidates is None:
            min_documents_per_candidates = 1
        print('Min_docs_per_can:', min_documents_per_candidates)
        print('Max_docs_per_can:', max_documents_per_candidates)

        candidates_boolean_mask = np.logical_and(
            num_docs_per_can <= max_documents_per_candidates,
            num_docs_per_can >= min_documents_per_candidates
        )
        candidates_index_mask = np.sort(np.unique(candidates_boolean_mask.nonzero()[0]))

        # take care of expert_mask
        experts_boolean_mask = np.zeros(len(candidates_boolean_mask), dtype=np.bool_)
        experts_boolean_mask[self.gt.experts_mask] = True
        experts_boolean_mask = experts_boolean_mask[candidates_index_mask]
        self.gt.experts_mask = experts_boolean_mask.nonzero()[0]

        # Update associations
        self.ds.associations = self.ds.associations[:, candidates_index_mask]
        self.gt.associations = self.gt.associations[:, candidates_index_mask]

        docs_mask = (np.squeeze(np.asarray(self.ds.associations.sum(axis=1))) > 0).nonzero()[0]
        topics_mask = (np.squeeze(np.asarray(self.gt.associations.sum(axis=1))) > 0).nonzero()[0]

        self.ds.associations = self.ds.associations[docs_mask]
        # self.ds.participate = self.ds.participate[docs_mask]

        self.gt.associations = self.gt.associations[topics_mask]
        self.ds.citations = self.ds.citations[docs_mask][:, docs_mask]
        self.ds.citations.eliminate_zeros()
        self.ds.associations.eliminate_zeros()
        self.gt.associations.eliminate_zeros()

        # Updates attributes
        mask = set(topics_mask)
        self.gt.topics = [e for i, e in enumerate(self.gt.topics) if i in mask]
        mask = set(candidates_index_mask)
        self.gt.candidates = [e for i, e in enumerate(self.gt.candidates) if i in mask]
        mask = set(docs_mask)
        self.ds.documents = [e for i, e in enumerate(self.ds.documents) if i in mask]

        self.ds.index2doc = {d: i for i, d in enumerate(self.ds.documents)}
        mask = set(candidates_index_mask)
        self.ds.candidates = [e for i, e in enumerate(self.ds.candidates) if i in mask]

        self.print_stats()



    def save(self, dir_path):
        expert_finding.data.io.save_as_json(dir_path, "df_documents", self.ds.documents)
        expert_finding.data.io.save_as_json(dir_path, "df_candidates", self.ds.candidates)
        expert_finding.data.io.save_as_json(dir_path, "df_venues", self.ds.venues)
        scipy.sparse.save_npz(os.path.join(dir_path, "df_associations"), self.ds.associations)
        scipy.sparse.save_npz(os.path.join(dir_path, "df_publish"), self.ds.publish)
        scipy.sparse.save_npz(os.path.join(dir_path, "df_citations"), self.ds.citations)
        expert_finding.data.io.save_as_json(dir_path, "gt_topics", self.gt.topics)
        expert_finding.data.io.save_as_json(dir_path, "gt_candidates", self.gt.candidates)
        expert_finding.data.io.save_as_json(dir_path, "gt_experts_mask", self.gt.experts_mask.tolist())
        scipy.sparse.save_npz(os.path.join(dir_path, "gt_associations"), self.gt.associations)


    def load_data(self, dir_path):
        self.ds.documents = expert_finding.data.io.load_as_json(dir_path, "df_documents")
        self.ds.candidates = expert_finding.data.io.load_as_json(dir_path, "df_candidates")
        self.ds.venues = expert_finding.data.io.load_as_json(dir_path, "df_venues")
        self.ds.associations = scipy.sparse.load_npz(os.path.join(dir_path, "df_associations.npz"))
        self.ds.publish = scipy.sparse.load_npz(os.path.join(dir_path, "df_publish.npz"))
        self.ds.citations = scipy.sparse.load_npz(os.path.join(dir_path, "df_citations.npz"))
        self.gt.topics = expert_finding.data.io.load_as_json(dir_path, "gt_topics")
        self.gt.candidates = expert_finding.data.io.load_as_json(dir_path, "gt_candidates")
        self.gt.experts_mask = np.array(expert_finding.data.io.load_as_json(dir_path, "gt_experts_mask"),
                                        dtype=np.int_)
        self.gt.associations = scipy.sparse.load_npz(os.path.join(dir_path, "gt_associations.npz"))

    def getDataset(self, documents_file_path, experts_folder_path, version, type=None):
        # Load data from disk
        print("Loading documents from aminer dataset...")
        documents = expert_finding.data.io_aminer.load_papers(documents_file_path)
        print("Loading experts from aminer dataset...")
        experts = expert_finding.data.io_aminer.load_experts(experts_folder_path, version)  # version= version

        # Build dataFrame
        print("Building candidates-documents associations...")
        for d in documents:
            # Add documents and candidates (duplicated)
            self.ds.documents.append("{0} {1}".format(d['title'], d['abstract']))
            for c in d['authors']:
                self.ds.candidates.append(c)

        # reduces candidates get all candidates
        candidates_set = set(self.ds.candidates)
        self.ds.candidates = list(candidates_set)

        M = len(self.ds.documents)
        N = len(self.ds.candidates)

        dat = list()
        row_ind = list()
        col_ind = list()
        # {index : author};
        candidates_index_dict = {n: i for i, n in enumerate(self.ds.candidates)}

        # get the authors in int the citation data set
        # construct the map like this
        '''
            a1  a2  a3  a4 ..
        d1   1   0   1   0
        d2   0   0   1   0
        ..
        '''
        if type is False:
            print("type", "weight")
            for i, d in enumerate(documents):
                n = len(d['authors'])
                dem = 0.0
                for i in range(1, n + 1):
                    dem = dem + (1.0 / i)
                for j, c in enumerate(d['authors']):
                    weight = self.author_weight(j, dem);

                    dat.append(weight)
                    row_ind.append(i)
                    col_ind.append(candidates_index_dict[c.strip()])
        else:
            print("type", "with out weight")
            for i, d in enumerate(documents):
                for j, c in enumerate(d['authors']):
                    dat.append(1)
                    row_ind.append(i)
                    col_ind.append(candidates_index_dict[c.strip()])
        self.ds.associations = scipy.sparse.csr_matrix((np.array(dat), (np.array(row_ind), np.array(col_ind))),
                                                       shape=(M, N))

        print("Building documents citations...")
        documents_idx = {d["idx"]: i for i, d in enumerate(documents)}
        # index2doc = {d: i for i, d in enumerate(documents)}
        M = len(self.ds.documents)
        N = len(self.ds.documents)
        dat = list()
        row_ind = list()
        col_ind = list()
        count = 0
        for i, d in enumerate(documents):
            for r in d['refs']:
                if r in documents_idx:
                    dat.append(1)
                    row_ind.append(i)
                    col_ind.append(documents_idx[r])
                else:
                    count += 1
        print("There were ", count, "unknown referenced documents on ", count + len(dat), "citations. (",
              count * 100 / (count + len(dat)), "%)")
        self.ds.citations = scipy.sparse.csr_matrix((np.array(dat), (np.array(row_ind), np.array(col_ind))),
                                                    shape=(M, N))
        print("Building experts-topics associations...")
        # create list of topics
        self.gt.topics = list(experts.keys())
        self.gt.candidates = self.ds.candidates
        # create experts-topics associations
        M = len(self.gt.topics)
        N = len(self.gt.candidates)
        dat = list()
        row_ind = list()
        col_ind = list()

        for i, t in enumerate(self.gt.topics):
            for c in experts[t]:
                if c in candidates_set:
                    dat.append(1)
                    row_ind.append(i)
                    col_ind.append(candidates_index_dict[c])
        print(len(dat), len(row_ind), len(col_ind))
        print(max(dat), max(row_ind), max(col_ind))
        print((M, N))
        self.gt.associations = scipy.sparse.csr_matrix((np.array(dat), (np.array(row_ind), np.array(col_ind))),
                                                       shape=(M, N))
        self.gt.experts_mask = np.sort(np.unique(self.gt.associations.nonzero()[1]))
        # self.print_stats()

        print("Building document-venues  associations...")
        # create list of venues
        for d in documents:
            self.ds.venues.append(d['venue'])
        venues_set = set(self.ds.venues)
        self.ds.venues = list(venues_set)
        M = len(self.ds.documents)
        N = len(self.ds.venues)

        dat = list()
        row_ind = list()
        col_ind = list()
        venues_index_dict = {n: i for i, n in enumerate(self.ds.venues)}

        for i, d in enumerate(documents):
            # for j, c in enumerate(d['venue']):
            dat.append(1)
            row_ind.append(i)
            col_ind.append(venues_index_dict[d['venue']])

        self.ds.publish = scipy.sparse.csr_matrix((np.array(dat), (np.array(row_ind), np.array(col_ind))),
                                                  shape=(M, N))
        self.print_stats()


    def author_weight(self, index, dem):
        return 1.0 / ((index + 1) * dem)

    def print_stats(self):
        print("--STATS--")
        print("Number of candidates: ", len(self.ds.candidates))
        print("Number of documents: ", len(self.ds.documents))
        print("Number of venues: ", len(self.ds.venues))

        print("Number of topics: ", len(self.gt.topics))
        print("Number of experts: ", len(self.gt.experts_mask))

        print("Shape of citations: ", self.ds.citations.shape)
        print("Shape of candidates-documents: ", self.ds.associations.shape)
        print("Shape of candidates-topics: ", self.gt.associations.shape)
        # print("Shape of venues-topics: ", self.ds.publish.shape)

        print("Number of citations_num: ", self.ds.citations.sum())
        print("Number of candidates-topics_num: ", self.gt.associations.sum())
        print("Number of candidates-documents links: ", self.ds.associations.sum())
        # print("Number of venues-documents links: ", self.ds.publish.sum())

        print("Number of all link num",
              self.ds.citations.sum() + self.ds.associations.sum() + self.gt.associations.sum())
        #       + self.ds.publish.sum())
        numb_docs = np.squeeze(np.asarray(self.ds.associations.sum(axis=0)))
        print("Documents per candidates stats: ", "min", numb_docs.min(), "max", numb_docs.max(),
              "mean", numb_docs.mean(), "std", numb_docs.std())
        numb_cand = np.squeeze(np.asarray(self.ds.associations.sum(axis=1)))
        print("Candidates per documents stats: ", "min", numb_cand.min(), "max", numb_cand.max(), "mean",
              numb_cand.mean(), "std", numb_cand.std())

        print("Number of citations links: ", self.ds.citations.sum())
        numb_docs = np.squeeze(np.asarray(self.ds.citations.sum(axis=0)))
        print("Documents in-citations stats: ", "min", numb_docs.min(), "max", numb_docs.max(),
              "mean", numb_docs.mean(), "std", numb_docs.std())
        numb_cand = np.squeeze(np.asarray(self.ds.citations.sum(axis=1)))
        print("Documents out-citations stats: ", "min", numb_cand.min(), "max", numb_cand.max(), "mean",
              numb_cand.mean(), "std", numb_cand.std())

        numb_docs = np.squeeze(np.asarray(self.ds.associations[:, self.gt.experts_mask].sum(axis=0)))
        print("Documents per experts stats: ", "min", numb_docs.min(), "max", numb_docs.max(),
              "mean", numb_docs.mean(), "std", numb_docs.std())
        numb_cand = np.squeeze(np.asarray(self.ds.associations[:, self.gt.experts_mask].sum(axis=1)))
        print("Experts per documents stats: ", "min", numb_cand.min(), "max", numb_cand.max(), "mean",
              numb_cand.mean(), "std", numb_cand.std())
        numb_tops = np.squeeze(np.asarray(self.gt.associations[:, self.gt.experts_mask].sum(axis=1)))
        print("Experts per topics stats: ", "min", numb_tops.min(), "max", numb_tops.max(),
              "mean", numb_tops.mean(), "std", numb_tops.std())
        print("--STATS--")
        print()
