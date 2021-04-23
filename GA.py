import random
from features import compute_fitness
from preprocess import preprocess_raw_sent
from preprocess import sim_with_title
from preprocess import sim_with_doc
from preprocess import sim_2_sent
from preprocess import count_noun
from copy import copy
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import nltk
import os.path
import statistics as sta
from rouge import Rouge
import re
import time
import os
import glob
from shutil import copyfile
import pandas as pd
import math
import multiprocessing
from sklearn.feature_extraction.text import TfidfVectorizer


class Summerizer(object):
    def __init__(self, title, sentences, raw_sentences, population_size, max_generation, crossover_rate, mutation_rate, num_picked_sents, simWithTitle, simWithDoc, sim2sents, number_of_nouns, order_params):
        self.title = title
        self.raw_sentences = raw_sentences
        self.sentences = sentences
        self.num_objects = len(sentences)
        self.population_size = population_size
        self.max_generation = max_generation
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.num_picked_sents = num_picked_sents
        self.simWithTitle = simWithTitle
        self.simWithDoc = simWithDoc
        self.sim2sents = sim2sents
        self.number_of_nouns = number_of_nouns
        self.order_params = order_params


    def generate_population(self, amount):
        population = []
        for i in range(amount):
            agent = np.zeros(self.num_objects)
            try:
                agent[np.random.choice(list(range(self.num_objects)), self.num_picked_sents, replace=False)] = 1
            except:
                agent[np.random.choice(list(range(self.num_objects)), self.num_picked_sents, replace=True)] = 1
            agent = agent.tolist()
            fitness = compute_fitness(self.title, self.sentences, agent, self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns, self.order_params)
            population.append((agent, fitness))
        return population 


    def roulette_select(self, total_fitness, population):
        fitness_slice = np.random.rand() * total_fitness
        fitness_so_far = 0.0
        for phenotype in population:
            fitness_so_far += phenotype[1]
            if fitness_so_far >= fitness_slice:
                return phenotype
        return None


    def rank_select(self, population):
        ps = len(population)
        population = sorted(population, key=lambda x: x[1], reverse=True)
        fitness_value = []
        for individual in population:
            fitness_value.append(individual[1])

        fittest_individual = max(fitness_value)
        medium_individual = sta.median(fitness_value)
        selective_pressure = fittest_individual - medium_individual
        j_value = 1
        a_value = np.random.rand()   
        for agent in population:
            if ps == 0:
                return None
            elif ps == 1:
                return agent
            else:
                range_value = selective_pressure - (2*(selective_pressure - 1)*(j_value - 1))/( ps - 1) 
                prb = range_value/ps
                if prb > a_value:
                    return agent
            j_value +=1

                
    def crossover(self, individual_1, individual_2, max_sent):
        if self.num_objects < 2 or random.random() >= self.crossover_rate:
            return individual_1[:], individual_2[:]
        crossover_point = 1 + random.randint(0, self.num_objects - 2)
        agent_1 = individual_1[0][:crossover_point] + individual_2[0][crossover_point:]
        fitness_1 = compute_fitness(self.title, self.sentences, agent_1, self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns, self.order_params)
        child_1 = (agent_1, fitness_1)
        sum_sent_in_summary = sum(child_1[0])
        if sum_sent_in_summary > max_sent:
            while(sum_sent_in_summary > max_sent):
                remove_point = 1 + random.randint(0, self.num_objects - 2)
                if agent_1[remove_point] == 1:
                    agent_1[remove_point] = 0
                    sent = self.sentences[remove_point]
                    sum_sent_in_summary -=1            
            fitness_1 = compute_fitness(self.title, self.sentences, agent_1, self.simWithTitle, self.simWithDoc,self.sim2sents, self.number_of_nouns, self.order_params)
            child_1 = (agent_1, fitness_1)

        crossover_point_2 = 1 + random.randint(0, self.num_objects - 2)
        agent_2 = individual_2[0][:crossover_point_2] + individual_1[0][crossover_point_2:]
        fitness_2 = compute_fitness(self.title, self.sentences, agent_2, self.simWithTitle, self.simWithDoc,self.sim2sents, self.number_of_nouns, self.order_params)
        child_2 = (agent_2, fitness_2)
        sum_sent_in_summary_2 = sum(child_2[0])
        if sum_sent_in_summary_2 > max_sent:
            while(sum_sent_in_summary_2 > max_sent):
                remove_point = 1 + random.randint(0, self.num_objects - 2)
                if agent_2[remove_point] == 1:
                    agent_2[remove_point] = 0
                    sent = self.sentences[remove_point]
                    sum_sent_in_summary_2 -= 1
            fitness_2 = compute_fitness(self.title, self.sentences, agent_2, self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns, self.order_params)
            child_2 = (agent_2, fitness_2)
        return child_1, child_2
    

    def mutate(self, individual, max_sent):
        sum_sent_in_summary = sum(individual[0])
        agent = individual[0][:]
        for i in range(len(agent)):
            if random.random() < self.mutation_rate and sum_sent_in_summary < max_sent :
                if agent[i] == 0 :
                   agent[i] = 1
                   sum_sent_in_summary +=1
                else:
                   agent[i] = 0
                   sum_sent_in_summary -=1
        
        fitness = compute_fitness(self.title, self.sentences, agent, self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns, self.order_params)
        return (agent, fitness)

    def compare(self, lst1, lst2):
        for i in range(self.num_objects):
            if lst1[i] != lst2[i]:
                return False
        return True

    def selection(self, population):
        if len(population) == 0:
            population = self.generate_population(self.population_size)

        max_sent = int(len(self.sentences)*0.2)
        if len(self.sentences) < max_sent:
            max_sent = len(self.sentences)

        population = sorted(population, key=lambda x: x[1], reverse=True)

        chosen_agents = int(0.1*len(population))
        
        new_population = population[: chosen_agents]

        population = population[chosen_agents : ]
        
        total_fitness = 0
        for indivi in population:
            total_fitness = total_fitness + indivi[1]  
        population_size = len(population)
        cpop = 0.0

        while cpop <= population_size:
            population = sorted(population, key=lambda x: x[1], reverse=True)
            parent_1 = None
            while parent_1 == None:
                parent_1 = self.rank_select(population)

            parent_2 = None
            while parent_2 == None :
                parent_2 = self.roulette_select(total_fitness, population)
                if parent_2 != None:
                    if self.compare(parent_2[0], parent_1[0]) :
                        parent_2 = self.roulette_select(total_fitness, population)

            
            parent_1, parent_2 = copy(parent_1), copy(parent_2)
            child_1, child_2 = self.crossover(parent_1, parent_2, max_sent)

            # child_1
            individual_X = self.mutate(child_1, max_sent)
            check1 = 0
            check2 = 0
            if len(population) > 4 :
                competing = random.sample(population, 4)
                lowest_individual = min(competing , key = lambda x: x[1])
                if individual_X[1] > lowest_individual[1]:
                    new_population.append(individual_X)
                    check1 = 1
                elif sum(lowest_individual[0]) <= max_sent:
                    new_population.append(lowest_individual)
                    check1 = 1

            # child_2
            individual_Y = self.mutate(child_2, max_sent)
            if len(population) > 4 :
                competing_2 = random.sample(population, 4)
                lowest_individual_2 = min(competing_2 , key = lambda x: x[1])
                if individual_Y[1] > lowest_individual_2[1]:
                    new_population.append(individual_Y)
                    check2 = 1
                elif sum(lowest_individual_2[0]) <= max_sent:
                    new_population.append(lowest_individual_2)
                    check2 = 1
            
            
            if  check1 + check2 == 0:
                cpop += 0.1
            else:
                cpop += check1 + check2

        fitness_value = []
        for individual in new_population:
            fitness_value.append(individual[1])
        avg_fitness = sta.mean(fitness_value)
        agents_in_Ev = [] 
        for agent in new_population:
            if (agent[1] > 0.95*avg_fitness) and (agent[1] < 1.05*avg_fitness):
                agents_in_Ev.append(agent)
        if len(agents_in_Ev) >= len(new_population)*0.9 :
            new_population = self.generate_population(int(0.7*self.population_size)) 
            agents_in_Ev = sorted(agents_in_Ev, key=lambda x: x[1], reverse=True)
            chosen = self.population_size - len(new_population)
            new_population.extend(agents_in_Ev[: chosen])
        return new_population

    def find_best_individual(self, population):
        if len(population) == 0:
            return None
        best_individual = sorted(population, key=lambda x: x[1], reverse=True)[0]
        return best_individual
 
    def check_best(self, arr):
        if len(arr) > 30:
            reversed_arr = arr[::-1][:20]
            if reversed_arr[0] > reversed_arr[-1]:
                return True
            else: 
                return False
        else:
            return True
   #MASingleDocSum    
    def solve(self):
        population = self.generate_population(self.population_size)
        best_individual = sorted(population, key=lambda x: x[1], reverse=True)[1]
        best_fitness_value = best_individual[1]
        tmp_arr = []
        tmp_arr.append(best_fitness_value)
        count = 0
    
        while count < self.max_generation or self.check_best(tmp_arr) == True:
            population = self.selection(population)
            best_individual = sorted(population, key=lambda x: x[1], reverse=True)[1]
            best_fitness_value = best_individual[1]  
            tmp_arr.append(best_fitness_value)          
            count +=1
        # for i in tqdm(range(self.max_generation)):
        #     population = self.selection(population)
        return self.find_best_individual(population)
    
    
    def show(self, individual,  file):
        index = individual[0]
        f = open(file,'w', encoding='utf-8')
        for i in range(len(index)):
            if index[i] == 1:
                f.write(self.raw_sentences[i] + ' ')
        f.close()

def load_a_doc(filename):
    file = open(filename, encoding='utf-8')
    article_text = file.read()
    file.close()
    return article_text   


def load_docs(directory):
	docs = list()  
	for name in os.listdir(directory):
		filename = directory + '/' + name
		doc = load_a_doc(filename)
		docs.append((doc, name))
	return docs

def clean_text(text):
    cleaned = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", ",", "'", "(", ")")).strip()
    check_text = "".join((item for item in cleaned if not item.isdigit())).strip()
    if len(check_text.split(" ")) < 4:
        return 'None'
    return text

def evaluate_rouge(hyp_path):
    hyp = hyp_path
    raw_ref = 'abstracts'
    FJoin = os.path.join
    files_hyp = [FJoin(hyp, f) for f in os.listdir(hyp)]
    files_raw_ref = [FJoin(raw_ref, f) for f in os.listdir(hyp)]
    
    f_hyp = []
    f_raw_ref = []
    print("number of document: ", len(files_hyp))
    for file in files_hyp:
        f = open(file)
        f_hyp.append(f.read())
        f.close()
    for file in files_raw_ref:
        f = open(file)
        f_raw_ref.append(f.read())
        f.close()
        
    rouge_1_tmp = []
    rouge_2_tmp = []
    rouge_L_tmp = []
    for hyp, ref in zip(f_hyp, f_raw_ref):
        try:
            rouge = Rouge()
            scores = rouge.get_scores(hyp, ref, avg=True)
            rouge_1 = scores["rouge-1"]["f"]
            rouge_2 = scores["rouge-2"]["f"]
            rouge_L = scores["rouge-l"]["f"]
            rouge_1_tmp.append(rouge_1)
            rouge_2_tmp.append(rouge_2)
            rouge_L_tmp.append(rouge_L)
        except Exception:
            pass
        # print(scores)
    rouge_1_avg = sta.mean(rouge_1_tmp)
    rouge_2_avg = sta.mean(rouge_2_tmp)
    rouge_L_avg = sta.mean(rouge_L_tmp)
    print('Rouge-1: ', rouge_1_avg)
    print('Rouge-2: ',rouge_2_avg )
    print('Rouge-L: ', rouge_L_avg)

    # for path in os.listdir(hyp_path):
    #     full_path = os.path.join(hyp_path, path)
    #     os.remove(full_path)

    return rouge_1_avg, rouge_2_avg, rouge_L_avg  

def start_run(processID, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE, sub_stories, save_path, order_params):
   
    for example in sub_stories:
        start_time = time.time()
        raw_sents = re.split("\n\n", example[0])[1].split(' . ')
        title = re.split("\n\n", example[0])[0] 

        #remove too short sentences
        df = pd.DataFrame(raw_sents, columns =['raw'])
        df['preprocess_raw'] = df['raw'].apply(lambda x: clean_text(x))
        newdf = df.loc[(df['preprocess_raw'] != 'None')]
        raw_sentences = newdf['preprocess_raw'].values.tolist()
        if len(raw_sentences) == 0:
            continue

        preprocessed_sentences = []
        for raw_sent in raw_sentences:
            preprocessed_sent = preprocess_raw_sent(raw_sent)
            preprocessed_sentences.append(preprocessed_sent)
            
        #tfidf for sentences 
        bodyandtitle = preprocessed_sentences.copy()
        bodyandtitle.append(preprocess_raw_sent(title.lower()))
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(bodyandtitle)
        feature_names = vectorizer.get_feature_names()
        dense = vectors.todense()
        denselist = dense.tolist()
        df_tfidf = pd.DataFrame(denselist, columns=feature_names)
        title_vector = denselist[-1]

        #tfidf for document
        document = [(" ").join(bodyandtitle)]
        vector_doc = vectorizer.fit_transform(document)
        dense_doc = vector_doc.todense()
        document_vector = dense_doc.tolist()[0]

        list_sentences_frequencies = denselist[:-1]
        # number_of_nouns = count_noun(preprocessed_sentences, option = True)
        number_of_nouns = 0
        simWithTitle = sim_with_title(list_sentences_frequencies, title_vector)
        sim2sents = sim_2_sent(list_sentences_frequencies)
        simWithDoc = sim_with_doc(list_sentences_frequencies, document_vector)
          
        print("Done preprocessing!")
        
        print('time for processing', time.time() - start_time)
        if len(preprocessed_sent) < 4:
            NUM_PICKED_SENTS = len(preprocessed_sentences)
        else:
            NUM_PICKED_SENTS = 4
        # DONE!
        Solver = Summerizer(title, preprocessed_sentences, raw_sentences, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE, NUM_PICKED_SENTS, simWithTitle, simWithDoc, sim2sents, number_of_nouns, order_params)
        best_individual = Solver.solve()
        file_name = os.path.join(save_path, example[1] )    

 
        if best_individual is None:
            print ('No solution.')
        else:
            print(file_name)
            print(best_individual)
            Solver.show(best_individual, file_name)
    rouge1, rouge2, rougel = evaluate_rouge(save_path)
    result_file = '{}.{}'.format(processID, 'txt')
    fp = open(result_file, 'w', encoding='utf-8')
    fp.write('r1: {}, r2: {}, rL: {} '.format(rouge1, rouge2, rougel))
        
    
def multiprocess(num_process, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE, stories, save_path):
    # processes = []
    # n = math.floor(len(stories)/5)
    # set_of_docs = [stories[i:i + n] for i in range(0, len(stories), n)] 
    # for index, sub_stories in enumerate(set_of_docs):
    #     p = multiprocessing.Process(target=start_run, args=(
    #         index, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE,sub_stories, save_path[index]))
    #     processes.append(p)
    #     p.start()      
    # for p in processes:
    #     p.join()

    processes = []
    for index in range(len(save_path)):
        p = multiprocessing.Process(target=start_run, args=(
            index, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE,stories, save_path[index], index))
        processes.append(p)
        p.start()      
    for p in processes:
        p.join()



def main():
    # Setting Variables
    POPU_SIZE = 30
    MAX_GEN = 200
    CROSS_RATE = 0.8
    MUTATE_RATE = 0.4
    # NUM_PICKED_SENTS = 4

    directory = 'documents'
    save_path=['hyp1', 'hyp2', 'hyp3', 'hyp4', 'hyp5']

    if not os.path.exists('hyp1'):
        os.makedirs('hyp1')
    if not os.path.exists('hyp2'):
        os.makedirs('hyp2')
    if not os.path.exists('hyp3'):
        os.makedirs('hyp3')
    if not os.path.exists('hyp4'):
        os.makedirs('hyp4')
    if not os.path.exists('hyp5'):
        os.makedirs('hyp5')


    print("Setting: ")
    print("POPULATION SIZE: {}".format(POPU_SIZE))
    print("MAX NUMBER OF GENERATIONS: {}".format(MAX_GEN))
    print("CROSSING RATE: {}".format(CROSS_RATE))
    print("MUTATION SIZE: {}".format(MUTATE_RATE))

    # list of documents
    stories = load_docs(directory)
    start_time = time.time()
    
    multiprocess(5, POPU_SIZE, MAX_GEN, CROSS_RATE,
                 MUTATE_RATE, stories, save_path)
    # start_run(1, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE, stories, save_path[0], 0)

    print("--- %s mins ---" % ((time.time() - start_time)/(60.0*len(stories))))

if __name__ == '__main__':
    main()  
        
        
     
    


    
    
    
    
        
            
            
         
