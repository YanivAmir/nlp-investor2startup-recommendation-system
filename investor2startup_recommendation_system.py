from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests
from time import sleep
from random import randint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re


def get_notable_investments(df):
    investors, notable_companies, bad_investors = [],[],[]
    for i,investments in enumerate(df['Notable investments']):
        if investments:
            try:
                companies = investments.split(', ')
                if len(companies)>1:
                    investors.append(df['Angel Name'][i])
                    notable_companies.append(companies)
                else:
                    companies = investments.split(',')
                    if len(companies) > 1:
                        investors.append(df['Angel Name'][i])
                        notable_companies.append(companies)
            except:
                bad_investors.append(print(df['Angel Name'][i]))
    data={'Angel':investors, 'Profiling Investments':notable_companies}
    print('Angels w/o parsable profiling companies:')
    print('\t'+bad_investors)
    return(pd.DataFrame.from_dict(data))

def get_company_name(homepage):
    company = homepage.split('.')[-2].split('/')[-1]
    names = [company[1:i] for i in range(4,len(company)+1)]
    names.append(company)
    return(company,names)

def get_text(query,website=False):
    if website:
        result = requests.get('https://www.google.com/search?q='+query)
    else:
        result = requests.get('https://www.google.com/search?q='+query+' company')
    html_page = result.content
    soup = BeautifulSoup(html_page, 'html.parser')
    return(soup.findAll(text=True))

def sort_text(text,black_list):
    text_dict = {}
    text_dict['word #'],text_dict['text'],positive_index,positive_text= [],[],[],[]
    for i, snipett in enumerate(text):
        if not any(word in snipett for word in black_list):
            positive_index.append(i)
            positive_text.append(snipett)
            text_dict['text'].append(snipett)
            text_dict['word #'].append(len(snipett))
    text_sorted = pd.DataFrame.from_dict(text_dict).sort_values(by='word #', axis=0, ascending=False)
    return(text_sorted)

def remove_hebrew(text_list):
    non_hebrew_text = []
    for snippet in text_list:
        non_hebrew = re.sub(r'[^A-Za-z ]+', '', snippet)
        if non_hebrew[:11] != 'functionvar':
            non_hebrew_text.append(non_hebrew)
    return(non_hebrew_text)

def join_text(text_sorted,sentences_to_join):
    return('\n '.join(text_sorted['text'][:sentences_to_join]))

def save_startup_df(texts,names,start_index,index):
    summary = {'Company': names,'Text': texts}
    summary = pd.DataFrame.from_dict(summary)
    summary.to_csv('Startups Texts of Startup '+str(start_index)+'-'+str(index+1))
    print('SAVED : Startups Texts of Startup '+str(start_index)+'-'+str(index+1))
    return(summary)

def startups_text(startups_df, end_index, start_index=0, sentences_to_join=10, sleep_start=5, sleep_end=10,sleep_interval=150):
    print('initiating text mining for startups')
    startups_names, startups_texts = [], []
    for i in range(start_index, end_index):
        startup = startups_df['Company Name'][i]
        html = startups_df['Website'][i]
        print(startup)
        sleep(randint(sleep_start, sleep_end))
        text = get_text(html,website=True)
        non_hebrew_text = remove_hebrew(text)
        text_black_list = ['witter', 'nstagram', 'inkedin', 'html', 'oogle', '@', '=', 'px']
        text_sorted = sort_text(non_hebrew_text, text_black_list)
        startup_text = join_text(text_sorted, sentences_to_join)
        startups_names.append(startup)
        startups_texts.append(startup_text)
        if (i + 1) % 10 == 0:
            print('*******  ', i + 1, ' : intermission   ********')
            sleep(sleep_interval)
            if (i + 1) % 30 == 0:
                iterim_df = save_startup_df(startups_texts, startups_names, start_index, i)
    startups_texts_df = save_startup_df(startups_texts, startups_names, start_index, end_index)
    return (startups_texts_df)

def save_angels_df(angels,texts,company_names,start_index,index):
    summary = {'Angels': angels, 'Company': company_names,'Text': texts}
    summary = pd.DataFrame.from_dict(summary)
    summary.to_csv('Angels Texts of Startup '+str(start_index)+'-'+str(index+1))
    print('SAVED : Angels Texts of Startup '+str(start_index)+'-'+str(index+1))
    return(summary)

def angels_text(angels_df,end_index, start_index=0, sentences_to_join=10, sleep_start=5, sleep_end=10,sleep_interval=150):
    print('initiating text mining for angels profiling companies')
    angels_names,companies_names,companies_texts= [],[],[]
    for j in range(start_index,end_index):
        print(angels_df['Angel'][j])
        for i,company in enumerate(angels_df['Profiling Investments'][j]):
            print('\t',company)
            sleep(randint(sleep_start,sleep_end))
            text = get_text(company, website=False)
            non_hebrew_text = remove_hebrew(text)
            html_black_list = ['witter', 'nstagram', 'inkedin', 'html', 'oogle', '@', '=', 'px']
            text_sorted = sort_text(non_hebrew_text, html_black_list)
            company_text = join_text(text_sorted, sentences_to_join)
            angels_names.append(angels_df['Angel'][j])
            companies_names.append(company)
            companies_texts.append(company_text)
        if (j + 1) % 10 == 0:
            print('*******  ', j + 1, ' : intermission   ********')
            sleep(sleep_interval)
            if (j + 1) % 30 == 0:
                iterim_df = save_angels_df(angels_names,companies_texts, companies_names, start_index, j)
    startups_texts_df = save_angels_df(angels_names, companies_texts, companies_names, start_index, end_index)
    return(startups_texts_df)

def edit_combined_text(df,words_black_list):
    company,final_text,text_length = [],[],[]
    for i in range(len(df)):
        # labels.append(companies_df['Industry Categories'][i]) #[companies_df['Company Name']==df['Company']
        joined_text =  df['Text'][i] #+ companies_df['Description and Metrics'][i]
        joined_text = re.sub(r'[^A-Za-z ]+', '', joined_text).lower().split(' ')
        joined_text =  ' '.join([word for word in joined_text if word not in words_black_list])
        joined_text = joined_text.replace('  ',' ')
        final_text.append(joined_text)
        company.append(df['Company'][i])
        text_length.append(len(joined_text))
    data_text = {'Company':company, 'Text Length': text_length, 'Text': final_text}  #, 'Labels': labels}
    data_text = pd.DataFrame.from_dict(data_text)
    print(data_text['Text Length'].describe())
    return(df)

def word_vectorize(df,ngram_min, ngram_max,max_word_features):
    word_vectorizer = TfidfVectorizer(sublinear_tf=True,strip_accents='unicode',analyzer='word',
        token_pattern=r'\w{1,}', stop_words='english',ngram_range=(ngram_min, ngram_max),max_features=max_word_features)
    print('Done with word vectorizer')
    return(word_vectorizer.fit_transform(df['Text']))

def char_vectorize(df,char_min, char_max,max_char_features):
    char_vectorizer = TfidfVectorizer(sublinear_tf=True,strip_accents='unicode',analyzer='char',
        stop_words='english',ngram_range=(char_min, char_max),max_features=max_char_features)
    print('Done with character vectorizer')
    return(char_vectorizer.fit_transform(df['Text']))

def cosine_scoring(n_startups,n_angels,word_features,char_features,char2word_ratio):
    cosine_scores = np.zeros((n_angels,n_startups))
    for i in range(n_angels):
        cosine_word_scores = linear_kernel(word_features[i], word_features).flatten()
        cosine_char_scores = linear_kernel(char_features[i], char_features).flatten()
        cosine_scores_interim = (char2word_ratio * cosine_char_scores + cosine_word_scores) / (1 + char2word_ratio)
        cosine_scores[i] = cosine_scores_interim[n_angels:]
    return(cosine_scores)

def find_matches(angel_scoring,startups_text_df,num_of_matches):
    j,k = 0,0
    startups,scores = [],[]
    while j < num_of_matches:
        startup = startups_text_df.iloc[angel_scoring['company index'][k]]['Company']
        if startup not in startups:
            # startups_stats.loc[comp, 'number_of_recs'] += 1
            startups.append(startup)
            scores.append(angel_scoring['score'][j])
            j += 1
        k += 1
    return(startups,scores)

def find_max_min(angel_scoring,max_total,min_total):
    max_comp = angel_scoring['score'][0]
    if max_comp > max_total:
        max_total = max_comp
    min_comp = angel_scoring['score'].iloc[-1]
    if min_comp < min_total:
        min_total = min_comp
    return(max_total,min_total)

def normalize_scores(matching,max,min,num_of_matches):
    for i in range(len(matching)):
        norm_scores = [100 * (score - min) / (max - min) for score in matching['Best Scores'][i]]
        matching['Best Scores'][i] = norm_scores
    score_column_names = ['Score '+str(i + 1) for i in range(num_of_matches)]
    matching[score_column_names] = pd.DataFrame(matching['Best Scores'].tolist(), index=matching.index).round(decimals=2)
    matching = matching.drop(['Best Scores'], axis=1)
    pd.set_option('precision', 2)
    return(matching)

def find_startups(startups_text_df,cosine_scores,angel_df,num_of_matches=5,individually=True,
                  angels='all',max_stndrd=0,min_stndrd=0):
    all_angels_names, all_angels_profiling_companies,all_angels_matches,all_angels_matches_scores = [],[],[],[]
    max_total, min_total = 0, 1
    if angels=='all':
        angels = list(set(angel_df.Angel.dropna()))
        angels.sort()
    for angel in angels:
        all_angels_names.append(angel)
        print(angel.replace('\n',''))
        angel_indices = angel_df.index[angel_df['Angel'] == angel].tolist()
        all_angels_profiling_companies.append(angel_df.loc[angel_indices, 'Company'].tolist())
        for index in angel_indices:
            # print('\t', angel_df.loc[index, 'Company'])
            cosine_similarities_sorted = np.sort(cosine_scores[index])[::-1]
            similar_comp_indices_sorted = cosine_scores[index].argsort()[::-1]
            interim = pd.DataFrame([similar_comp_indices_sorted, cosine_similarities_sorted]).T
            interim.columns = ['company index', 'score']
            if index == angel_indices[0]:
                angel_scoring = interim
            if index != angel_indices[0] and individually:
                angel_scoring = pd.concat([angel_scoring, interim])
            if index != angel_indices[0] and not individually:
                angel_scoring = angel_scoring.merge(interim, how='left', on='company index')
        if not individually:
            angel_scoring['score'] = angel_scoring.iloc[:, 1:].mean(axis=1)
        angel_scoring = angel_scoring.sort_values(by='score', ascending=False).reset_index()
        angel_scoring = angel_scoring.astype({"company index": int})
        max_total, min_total = find_max_min(angel_scoring,max_total,min_total)
        angel_matches, matches_scores = find_matches(angel_scoring,startups_text_df,num_of_matches)
        all_angels_matches.append(angel_matches)
        all_angels_matches_scores.append(matches_scores)
    print('max score: ', max_total)
    print('min score:', min_total)
    matching_df = pd.DataFrame.from_dict({'Angel':all_angels_names,'Profiling Companies':all_angels_profiling_companies,
                                  'Best Matches': all_angels_matches, 'Best Scores':all_angels_matches_scores})
    if max_stndrd==0 and min_stndrd==0:
        matching_df = normalize_scores(matching_df,max_total,min_total,num_of_matches)
    else:
        matching_df = normalize_scores(matching_df, max_stndrd,min_stndrd, num_of_matches)
    return(matching_df)

def create_score_matrix_df(matching_df,startups_text_df):
    all_startups = startups_text_df['Company'].tolist()
    all_startups.sort()
    angels2startups = pd.DataFrame(np.zeros((len(matching_df),len(all_startups))), columns=all_startups, index = matching_df['Angel'])
    for i,angel in enumerate(matching_df['Angel']):
        print(angel.replace('\n',''))
        angel_scores = pd.DataFrame([matching_df.loc[i,'Best Matches'],matching_df.iloc[i, 3:].values]).T
        angel_scores.columns=['startups','scores']
        # angel_scores = angel_scores.sort_values(by='startups')
        angel_startups = angel_scores.startups.values
        for j,startup in enumerate(angel_startups):
            angels2startups[startup][i] = angel_scores.loc[angel_scores['startups']==startup,'scores']
    return(angels2startups)

def find_angels(matrix_df):
    all_startups_scoring,all_startups_matchs=[],[]
    for startup in matrix_df.columns:
        print(startup)
        # print(matrix_df[startup].sort_values(ascending=False).tolist())
        all_startups_scoring.append(matrix_df[startup].sort_values(ascending=False).tolist())
        indices = matrix_df[startup].values.argsort().tolist()
        indices = indices[::-1]
        all_startups_matchs.append(matrix_df.index[indices].values.tolist())
    matching_df = pd.DataFrame.from_dict({'Startup':matrix_df.columns,'Best Matches': all_startups_matchs, 'Best Scores':all_startups_scoring})
    score_column_names = ['Score ' + str(i + 1) for i in range(len(matching_df['Best Scores'][0]))]
    matching_df[score_column_names] = pd.DataFrame(matching_df['Best Scores'].tolist(), index=matching_df.index).round(decimals=2)
    matching_df = matching_df.drop(['Best Scores'], axis=1)
    return(matching_df)


def calc_distances(score_mat_df):
    startup_distances = np.zeros((score_mat_df.shape[1],score_mat_df.shape[1]))
    for i in range(score_mat_df.shape[1]):
        for j in range(i):
            startup_distances[i,j]=np.linalg.norm(score_mat_df.iloc[:,i].values - score_mat_df.iloc[:,j].values)
    angel_distances = np.zeros((score_mat_df.shape[0],score_mat_df.shape[0]))
    for i in range(score_mat_df.shape[0]):
        for j in range(i):
            startup_distances[i,j]=np.linalg.norm(score_mat_df.iloc[i,:].values - score_mat_df.iloc[j,:].values)
    return(startup_distances,angel_distances)



'''Parameters'''
sentences_to_join = 10
start_index=0
sleep_start=5
sleep_end=10
sleep_interval=150
recommendations_per_investor = 10
words_black_list = ['rasing','company','series','revenue','usd','seed','round','followers','linkedin','twitter','contact','employees','it','location','view','is','in','a','an','that','this','these','and','if','or','to','by','are','the','of','with','for','etc','at','via']
ngram_min = 1
ngram_max = 4
max_word_features = 20000
char_min = 3
char_max = 7
max_char_features = 50000
char_vs_word_score_ratio = 3

"""Main"""
'''text mining subroutines'''
startups_df = pd.read_csv('/Users/yanivamir/Documents/Machine Learning/GVI Kobi Kalderon/Funding Seeking Startups.csv')
end_index=len(startups_df)
startups_texts_df = startups_text(startups_df,end_index,start_index,sentences_to_join,sleep_start,sleep_end,sleep_interval)
startups_texts_df = startups_texts_df.drop_duplicates(subset='Company').reset_index()
angels_df = pd.read_csv("/Users/yanivamir/Documents/Machine Learning/GVI Kobi Kalderon/Flashpoint's list of Active Angels in Israel/Flashpoints list of Active Ange-Table 1.csv")
angels_df = get_notable_investments(angels_df)
angels_text_df = angels_text(angels_df,end_index,start_index,sentences_to_join,sleep_start,sleep_end,sleep_interval)
angels_text_df = angels_text_df.dropna().reset_index()
combined_df = pd.concat([angels_text_df,startups_texts_df]).reset_index()
combined_df = edit_combined_text(combined_df,words_black_list)
n_startups = len(startups_texts_df)
n_angels = len(angels_text_df)
assert n_angels + n_startups == len(combined_df), "oh no!"
print('mining text is complete')

'''text vectorisation and cosine similarities'''
word_features = word_vectorize(combined_df,ngram_min,ngram_max,max_word_features)
char_features = char_vectorize(combined_df,char_min,char_max,max_char_features)
cosine_matrix = cosine_scoring(n_startups,n_angels,word_features,char_features,char_vs_word_score_ratio)
print(cosine_matrix.shape)
np.save('cosine_matrix', cosine_matrix)
cosine_matrix = np.load('/Users/yanivamir/Documents/Machine Learning/GVI Kobi Kalderon/pythonProject/cosine_matrix.npy')
print('cosine matrix was saved')

'''finding matches based on cosine similarities' matrix'''
matching_angels_to_startups = find_startups(startups_texts_df,cosine_matrix,angels_text_df,len(startups_texts_df),individually=False,angels='all')
matching_angels_to_startups.to_csv('matching_angels_to_startups')
matching_angels_to_startups = pd.read_csv('/Users/yanivamir/Documents/Machine Learning/GVI Kobi Kalderon/pythonProject/matching_angels_to_startups', index_col=0)
score_matrix_df = create_score_matrix_df(matching_angels_to_startups,startups_texts_df)
score_matrix_df.to_csv('score_matrix_df')
matching_startups_to_angels = find_angels(score_matrix_df)
matching_startups_to_angels.to_csv('matching_startups_to_angels')
# matching_angel_to_startups = find_angels(score_matrix_df.T)
# matching_angel_to_startups.to_csv('matching_angel_to_startups')
print('csv summary files were saved')
print('done')

