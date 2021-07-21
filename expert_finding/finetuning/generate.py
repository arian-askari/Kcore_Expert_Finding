# encoding: utf-8
import expert_finding.data.sets as  sets
import expert_finding.data.io as io
import expert_finding.data.io_aminer as aminer
import expert_finding.data.config
import expert_finding.data.sets
import expert_finding.evaluation.visual
import expert_finding.tools.graphs
import expert_finding.finetuning.train
import expert_finding.script.document_embedding
import expert_finding.script.experiment

import os
import patoolib
import zipfile
import pkg_resources
import urllib.request
import random
import numpy as np


# from Queue import Queue


class Generator:
    def __init__(self, version, work_dir, communities_path, sample_rate, meta_path, dataset):
        self.version = version
        self.work_dir = work_dir
        self.k = 5
        self.strategy = []
        self.communities_path = communities_path
        self.sample_rate = sample_rate
        self.meta_path = meta_path
        self.positive = []
        self.seed_topic = list()
        self.dataset = dataset
        self.seed_paper = list()

    def get_seed_paper(self):
        self.positive = [[] for i in range(self.dataset.ds.associations.shape[-1])]
        for i, t in enumerate(self.dataset.gt.topics):
            experts_indices = self.dataset.gt.associations[i, self.dataset.gt.experts_mask].nonzero()[1]
            documents_indices = np.unique(
                self.dataset.ds.associations[:, self.dataset.gt.experts_mask[experts_indices]].nonzero()[0])
            maxq = len(documents_indices)
            self.positive[i] = documents_indices
            for j in range(int(maxq * self.sample_rate)):
                d = documents_indices[j]
                self.seed_paper.append(d)
                self.seed_topic.append(i)
        return self.seed_paper

    def get_k_core_commity_PAP(self, seed_paper, k, A_da, A_ad):
        Tque = set()
        Qque = set()
        QRem = set()
        N = set()
        S = set()
        vis = A_da.shape[0] * [0]
        Phi = [set() for i in range(A_da.shape[0])]

        Tque.add(seed_paper)
        S.add(seed_paper)
        vis[seed_paper] = 1
        while len(Tque) > 0:
            doc = Tque.pop()
            for author in np.nonzero(A_da[doc].toarray()[0])[0]:
                for a_doc in np.nonzero(A_ad[author].toarray()[0])[0]:
                    if vis[a_doc] == 0:
                        vis[a_doc] = 1
                        Phi[doc].add(a_doc)
                        S.add(a_doc)

            if len(Phi[doc]) >= k:
                Tque = Tque.union(Phi[doc])
            else:
                Qque.add(doc)

        while len(Qque) != 0:
            v = Qque.pop()
            S.remove(v)
            for u in Phi[v].copy():
                if (u not in S): continue
                if (v in Phi[u]): Phi[u].remove(v)
                if len(Phi[u]) < k:
                    Qque.add(u)

        for author in np.nonzero(A_da[seed_paper].toarray()[0])[0]:
            for a_doc in np.nonzero(A_ad[author].toarray()[0])[0]:
                N.add(a_doc)

        commity = S.union(N)
        return commity, QRem

    def get_k_core_commity_PTP(self, seed_paper, k, G_ta, A_ad):
        Tque = set()
        Qque = set()
        QRem = set()
        N = set()
        S = set()
        A_td = G_ta.dot(A_ad)
        A_dt = A_td.transpose()
        vis = A_dt.shape[0] * [0]
        Phi = [set() for i in range(A_dt.shape[0])]
        Tque.add(seed_paper)
        S.add(seed_paper)
        vis[seed_paper] = 1
        while len(Tque) > 0:
            doc = Tque.pop()
            for topic in np.nonzero(A_dt[doc].toarray()[0])[0]:
                for t_doc in np.nonzero(A_td[topic].toarray()[0])[0]:
                    if vis[t_doc] == 0:
                        vis[t_doc] = 1
                        Phi[doc].add(t_doc)
                        S.add(t_doc)

            if len(Phi[doc]) >= k:
                Tque.union(Phi[doc])
            else:
                Qque.add(doc)

        while len(Qque) != 0:
            v = Qque.pop()
            S.remove(v)
            for u in Phi[v].copy():
                if (u not in S): continue
                if (v in Phi[u]): Phi[u].remove(v)
                if len(Phi[u]) < k:
                    Qque.add(u)

        for topic in np.nonzero(A_dt[seed_paper].toarray()[0])[0]:
            for t_doc in np.nonzero(A_td[topic].toarray()[0])[0]:
                N.add(t_doc)

        commity = S.union(N)
        return commity, QRem

    def get_k_core_commity_PCP(self, seed_paper, k, A_dd):
        Tque = set()
        Qque = set()
        QRem = set()
        N = set()
        S = set()

        vis = A_dd.shape[0] * [0]
        Phi = [set() for i in range(A_dd.shape[0])]

        Tque.add(seed_paper)
        S.add(seed_paper)
        vis[seed_paper] = 1
        while len(Tque) > 0:
            doc = Tque.pop()
            for c_doc in np.nonzero(A_dd[doc].toarray()[0])[0]:
                if vis[c_doc] == 0:
                    vis[c_doc] = 1
                    Phi[doc].add(c_doc)
                    S.add(c_doc)

            if len(Phi[doc]) >= k:
                Tque.union(Phi[doc])
            else:
                Qque.add(doc)

        while len(Qque) != 0:
            v = Qque.pop()
            QRem.add(v)
            S.remove(v)
            for u in Phi[v].copy():
                if (u not in S): continue
                if (v in Phi[u]): Phi[u].remove(v)
                if len(Phi[u]) < k:
                    Qque.add(u)

        for c_doc in np.nonzero(A_dd[seed_paper].toarray()[0])[0]:
            N.add(c_doc)
        commity = S.union(N)
        return commity, QRem

    def get_commities(self, seed_papers, k, A_da, A_ad, communities_dir, communities_name, meta_path):
        commities = list()
        nears = list()
        for i, seed in enumerate(seed_papers):
            if meta_path is "PAP":
                commity, near = self.get_k_core_commity_PAP(seed, k, A_da, A_ad)
                commities.append(list(commity))
                commities.append(list(near))
            if meta_path is "PTP":
                commity, near = self.get_k_core_commity_PTP(seed, k, A_da, A_ad)
                commities.append(list(commity))
                commities.append(list(near))
            if meta_path is "PCP":
                commity, near = self.get_k_core_commity_PCP(seed, k, A_ad)
                commities.append(list(commity))
                commities.append(list(near))
        print("commities num:", len(commities))
        io.save_as_json(communities_dir, communities_name, commities)
        io.save_as_json(communities_dir, communities_name + "nears", nears)
        return commities, nears

    def get_commities_muti(self, seed_papers, k, communities_dir, communities_name, meta_path):
        dataset = self.dataset
        A_da = dataset.ds.associations
        A_ad = A_da.transpose()
        A_dd = dataset.ds.citations
        G_ta = dataset.gt.associations
        commities = list()
        n = str(len(seed_papers))
        if meta_path is "PAP":
            commities, nears = self.get_commities(seed_papers, k, A_da, A_ad, communities_dir, communities_name,
                                                  meta_path)
        if meta_path is "PTP":
            commities, nears = self.get_commities(seed_papers, k, G_ta, A_ad, communities_dir, communities_name,
                                                  meta_path)
        if meta_path is "PCP":
            commities, nears = self.get_commities(seed_papers, k, A_dd, A_dd, communities_dir, communities_name,
                                                  meta_path)
        if meta_path is "PAPPTP":
            for i, seed in enumerate(seed_papers):
                commity_pap = self.get_k_core_commity_PAP(seed, k, A_da, A_ad)
                commity_ptp = self.get_k_core_commity_PTP(seed, k, G_ta, A_ad)
                commity = commity_pap & commity_ptp
                commities.append(list(commity))
        if meta_path is "PCPPTP":
            for i, seed in enumerate(seed_papers):
                commity_pcp = self.get_k_core_commity_PCP(seed, k, A_dd)
                commity_ptp = self.get_k_core_commity_PTP(seed, k, G_ta, A_ad)
                commity = commity_pcp & commity_ptp
                commities.append(list(commity))
        if meta_path is "PAPPCP":
            for i, seed in enumerate(seed_papers):
                commity_pap = self.get_k_core_commity_PAP(seed, k, A_da, A_ad)
                commity_pcp = self.get_k_core_commity_PCP(seed, k, A_dd)
                commity = commity_pap & commity_pcp
                commities.append(list(commity))
        if meta_path is "PAPPCPPTP":
            for i, seed in enumerate(seed_papers):
                commity_pap = self.get_k_core_commity_PAP(seed, k, A_da, A_ad)
                commity_pcp = self.get_k_core_commity_PCP(seed, k, A_dd)
                commity_ptp = self.get_k_core_commity_PTP(seed, k, G_ta, A_ad)
                commity = commity_pap & commity_pcp & commity_ptp
                commities.append(list(commity))
        return list(commities)

    def sample_triples_random(self, seed_papers, comunities, T, triples_dir, triples_name):
        triplets = list()
        all = range(0, len(T) - 1)
        for i, seed in enumerate(seed_papers):
            negative = list(set(all) - set(comunities[i]))
            for pos in comunities[i]:
                neg2 = np.random.choice(negative)
                triplets.append([pos, seed, neg2])
        print("triples sample num: ", len(triplets))
        io.save_as_json(triples_dir, triples_name, triplets)
        return triplets

    def sample_triples_near(self, seed_papers, comunities, T, triples_dir, triples_name, near):
        triplets = list()
        all = range(0, len(T) - 1)
        for i, seed in enumerate(seed_papers):
            negative = list(comunities)
            for pos in comunities[i]:
                neg = np.random.choice(negative)
                triplets.append([seed, pos, neg])
        print("triples sample num: ", len(triplets))
        io.save_as_json(triples_dir, triples_name, triplets)
        return triplets
