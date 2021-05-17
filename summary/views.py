from django.shortcuts import render
#imports for project
import re
import string
from nltk.tokenize import sent_tokenize
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
re_punc = ''.join([re.escape(x) for x in string.punctuation])
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings(action = 'ignore')
from gensim.models import Word2Vec
from gensim.models import keyedvectors
from nltk.stem import WordNetLemmatizer
import numpy as np
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1','rougeL'],use_stemmer=False)
from nltk.translate.bleu_score import sentence_bleu
from tqdm.notebook import trange
#from nltk.translate.bleu_score import corpus_bleu
from sklearn.metrics.pairwise import cosine_similarity

# Create your views here.
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def _create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix
def _create_documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table
import math
def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix
def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix
def _score_sentences(tf_idf_matrix) -> dict:
    """
    score a sentence by its word's TF
    Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue
def _find_average_score(sentenceValue) -> int:
    """
    Find the average score from the sentence value dictionary
    :rtype: int
    """
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]
    if(len(sentenceValue)==0):
        return 0

    # Average value of a sentence from original summary_text
    average = (sumValues / len(sentenceValue))

    return average
def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence_count<=4 and sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary


def home(request):
    
    news=request.POST.get('fulltext',False)
    news=str(news)
    news1=news
    news=news.strip()
    news=news.lower()
    news=decontracted(news)
    tokenized_sentences=sent_tokenize(news)
    for j in range(len(tokenized_sentences)):
        tokenized_sentences[j] = tokenized_sentences[j].replace(re_punc,"")
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()
    
    for sent in tokenized_sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table
    
    tf_matrix=_create_tf_matrix(frequency_matrix)
    count_doc_per_words=_create_documents_per_words(frequency_matrix)
    idf_matrix=_create_idf_matrix(frequency_matrix,count_doc_per_words,total_documents=4396)
    tf_idf_matrix=_create_tf_idf_matrix(tf_matrix,idf_matrix)
    
    sentence_scores = _score_sentences(tf_idf_matrix)
    
    threshold = _find_average_score(sentence_scores)
    summary = _generate_summary(tokenized_sentences, sentence_scores, 1 * threshold)
    print(summary)
    cleaned_sentences=sent_tokenize(news1)
    references=[]
    for i in cleaned_sentences:
        words=word_tokenize(i)
        references.append(words)
    summ_cleaned_texts=sent_tokenize(summary)
    candidates=[]
    for i in summ_cleaned_texts:
        words=word_tokenize(i)
        candidates.append(words)
    tot_score=0
    for candidate in candidates:
        score = sentence_bleu(references, candidate)
        tot_score+=score
    bleu_score=tot_score/len(candidates)
    print(bleu_score)
    rouge = scorer.score(news1, summary)

    print("rougge_score:",rouge)
    X_list = word_tokenize(news1)
    Y_list = word_tokenize(summary)

    # sw contains the list of stopwords
    sw = stopwords.words('english')
    l1 =[];l2 =[]

    # remove stop words from the string
    X_set = {w for w in X_list if not w in sw}
    Y_set = {w for w in Y_list if not w in sw}

    # form a set containing keywords of both strings
    rvector = X_set.union(Y_set)
    for w in rvector:
        if w in X_set: l1.append(1) # create a vector
        else: l1.append(0)
        if w in Y_set: l2.append(1)
        else: l2.append(0)
    c = 0

    # cosine formula
    for i in range(len(rvector)):
            c+= l1[i]*l2[i]
    cosine = c / float((sum(l1)*sum(l2))**0.5)
    print("similarity: ", cosine)




    
    return render(request,'home.html',{'original_text':news1,'summarized_text':summary,'bleu':bleu_score,'rouge':rouge,'cos':cosine})

def h(request):
    return render(request,'home.html')

def _word2vec(request):
    news=request.POST.get('fulltext',False)
    news=str(news)
    news1=news
    news=news.strip()
    news=news.lower()
    data1 = []
    news=decontracted(news)
    
    sentences = sent_tokenize(news)
    sentence_vector = []
    stopWords = set(stopwords.words("english"))
    for j in range(len(sentences)):
        sentences[j] = sentences[j].replace(re_punc,"")
    
    for sentence in sentences:
        temp = []
        # tokenize the sentence into words
        w = word_tokenize(sentence)
        for word in w:
            if word in stopWords:
                continue
            if word.isalnum():
                temp.append(word.lower())                
        data1.append(temp)

    model = Word2Vec(data1, min_count = 1, vector_size = 100, window = 5)
    words = list(model.wv.key_to_index)
    #print(words)
    word_vectors=model.wv
    word_vectors.save_word2vec_format('vecs.txt')
    reloaded_word_vectors = keyedvectors.load_word2vec_format('vecs.txt',binary=False)
    clean_sentence = []
    word_vector = []
    sentence_vector = []
    all_sentences_feature_vec = []  # feature vectors of all sentences in a record
    para_vector = np.zeros((100, ), dtype='float32')
    references = []
        
    for sentence in sentences:
        feature_vec = np.zeros((100, ), dtype='float32')
        w = word_tokenize(sentence)
        n_words = 0
        s=""
        refs_sentence=[]
        for word in w:
            if word in stopWords:
                continue
            if word.isalnum():
                n_words+=1
                vec = reloaded_word_vectors.get_vector(word.lower())
                feature_vec = np.add(feature_vec,vec)
                s+=word.lower()
                s+=" "
                refs_sentence.append(word.lower())
                
        references.append(refs_sentence)
        clean_sentence.append(s.strip())
        
        if n_words>0:
            feature_vec = np.divide(feature_vec,n_words)
            
        all_sentences_feature_vec.append(feature_vec)
        para_vector = np.add(para_vector,feature_vec)
        final_sentence_vec = np.dot(feature_vec,np.ones((100,1)))
        sentence_vector.append(final_sentence_vec[0])
        
    para_vector = np.divide(para_vector, len(sentences))

    sort_idx = np.argsort(sentence_vector)
    largest_indices = sort_idx[::-1][:5]
    #print(largest_indices)
    largest_indices.sort()
    #print(largest_indices)

    summ_vector = np.zeros((100, ), dtype='float32')
    candidates = []
    summarized_text = ""
    if(len(sentences)<5):
        s=""
        for j in range(len(sentences)):
            s+=sentences[j]
            a = clean_sentence[j].split()
            candidates.append(a)
            summ_vector = np.add(summ_vector,all_sentences_feature_vec[j])
        summarized_text+=s
        summ_vector = np.divide(summ_vector, len(sentences))
    else:
        s=""
        for j in largest_indices:
            s+=sentences[j]
            a = clean_sentence[j].split()
            candidates.append(a)
            summ_vector = np.add(summ_vector,all_sentences_feature_vec[j])
        summarized_text+=s
        summ_vector = np.divide(summ_vector, 5)
        
    cos = cosine_similarity(para_vector.reshape(1, 100), summ_vector.reshape(1, 100))[0,0]
        
    tot_score = 0
    for candidate in candidates:
        sc = sentence_bleu(references, candidate)
        tot_score+=sc
    bleu = tot_score/len(candidates)
        
    rouge = scorer.score(news1, summarized_text)
    
    return render(request,'home.html',{'original_text':news1,'summarized_text':summarized_text,'bleu':bleu,'rouge':rouge,'cos':cos})



