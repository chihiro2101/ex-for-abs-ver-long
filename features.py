import preprocess
import math
import nltk
import numpy as np


class METRIC(object):
    def __init__(self, title, sentences, agent, simWithTitle, simWithDoc, sim2sents, number_of_nouns):
        self.title = title
        self.sentences = sentences
        self.n = len(sentences)
        self.values = agent
        self.simWithTitle = simWithTitle
        self.simWithDoc = simWithDoc
        self.sim2sents = sim2sents
        self.number_of_nouns = number_of_nouns
        self.sum_nouns = sum(number_of_nouns)

    # number of sentences in summary
    def O(self):
        return np.sum(self.values)

    def position(self):
        p = 0
        for i in range(self.n):
            if self.values[i] == 1:
                p = p + math.sqrt(1 / (i + 1))
        return p

    # def scale_noun(self):
    #     scale = 0
    #     for i in range(self.n):
    #         if self.values[i] == 1:
    #             scale += self.number_of_nouns[i]
    #     return scale / self.sum_nouns

    def relationT(self):
        rt = 0
        for i in range(self.n):
            if self.values[i] == 1:
                rt += self.simWithTitle[i]/(self.O())
        return rt

    def cohesion(self):
        Ns = (self.O())*(self.O() - 1)/2.0
        simcos_sents_in_summary = []
        for i in range(self.n - 1):
            if self.values[i] == 1:
                for j in range(i+1, self.n):
                    if self.values[j] == 1:
                        simcos_sents_in_summary.append(self.sim2sents[i][j])

        Cs = np.sum(simcos_sents_in_summary)/Ns
        try:
            M = max(simcos_sents_in_summary)
        except:
            M = 0
        if M == 0:
            CoH = 0
        else:
            CoH = (math.log(Cs*9.0+1.0))/(math.log(M*9.0+1.0))
        return CoH

    def Cov(self):
        cov = 0
        for i in range(self.n - 1):
            if self.values[i] == 1:
                # for j in range(i+1, len(self.sentences)):
                #     if self.values[j] == 1:
                #         cov += self.simWithDoc[i] + self.simWithDoc[j]
                cov += self.simWithDoc[i]
        return cov

    def leng(self):
        length = {}
        for i in range(self.n):
            if self.values[i] == 1:
                length[i] = self.words_count(self.sentences[i])

        length_value = list(length.values())
        std = np.std(length_value)
        avgl = np.mean(length_value)

        le = 0
        for i in range(len(length_value)):
            if std == 0:
                sigmoid = 0.98
            else:
                sigmoid = np.exp((-length_value[i] - avgl) / std)
            le += (1 - sigmoid) / (1 + sigmoid)
        return le

    def words_count(self, sentences):
        words = nltk.word_tokenize(sentences)
        return len(words)

    def fitness(self):
        rel = 0.35
        cover = 0.3
        le  = 0.35
        fit = rel*self.relationT() + cover*self.Cov() + le*self.leng()
        return fit

    # def GLS(self):
    #     sim_sent = []
    #     sim_sent = self.simWithTitle
    #     c = []
    #     d = []
    #     p = [0]*self.n
    #     max_sim = max(sim_sent)
    #     for i in range(self.n):
    #         c.append(math.sqrt(1 / (i + 1)) + sim_sent[i]/max_sim)
    #         d.append(c[i]/(1+p[i]))

    #     gls = 0
    #     for i in range(self.n):
    #         if d[i] == min(d):
    #             p[i] += 1
    #         gls += 0.5*p[i]*self.values[i]
    #     return self.fitness() - gls


def compute_fitness(title, sentences, agent, simWithTitle, simWithDoc, sim2sents, number_of_nouns):
    metric = METRIC(title, sentences, agent, simWithTitle,
                    simWithDoc, sim2sents, number_of_nouns)
    # return metric.GLS()
    return metric.fitness()
