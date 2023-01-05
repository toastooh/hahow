import pandas as pd
import argparse
import re
from sentence_transformers import SentenceTransformer,util
import torch
from collections import OrderedDict
from operator import itemgetter

def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)
    
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        required=True,
        help="train.csv",
    )

    parser.add_argument(
        "--course_file",
        type=str,
        default=None,
        required=True,
        help="courses.csv",
    )

    parser.add_argument(
        "--user_file",
        type=str,
        default=None,
        required=True,
        help="users.csv",
    )

    parser.add_argument(
        "--sub_group",
        type=str,
        default=None,
        required=True,
        help="subgroups.csv",
    )

    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        required=True,
        help="test_file.csv",
    )

    parser.add_argument(
        "--course_out_file",
        type=str,
        default=None,
        required=True,
        help="the course prediction output file name",
    )
    
    parser.add_argument(
        "--subgroup_outfile",
        type=str,
        default=None,
        required=True,
        help="the user topic prediction output file name",
    )

    parser.add_argument(
        "--threshold",
        action="store_true",
        help="If passed, use threshold.",
    )
    args = parser.parse_args()
    return args

def compute_popularity(train):
    new_df=pd.DataFrame()
    course_lis=[]
    for i in list(train['course_id']):
        for j in i.split(' '):
            course_lis.append(j)
    new_df['course_id']=course_lis

    #popularity computed by train file
    train_pop=new_df.groupby(['course_id'])['course_id'].count()
    
    #normalized popularity
    normalized_pop=(train_pop-train_pop.min())/(train_pop.max()-train_pop.min())

    return normalized_pop

def main():
    args = parse_args()
    train=pd.read_csv(args.train_file,dtype={'course_id':'string'})

    normalized_pop=compute_popularity(train)

    course=pd.read_csv(args.course_file,dtype={'course_name':'string', 'description':'string', 'sub_groups':'string','topics':'string','target_group':'string','will_learn':'string'})
    user=pd.read_csv(args.user_file,dtype={'interests':'string','recreation_names':'string','occupation_titles':'string'})
    group_df=pd.read_csv(args.sub_group,dtype={'subgroup':'string'})
    
    test=pd.DataFrame()
    if args.test_file != None:
        test=pd.read_csv(args.test_file)

    desc=list(course['description'])
    course_name=list(course['course_name'])
    sub_group=list(course['sub_groups'])
    topic=list(course['topics'])
    course_id=list(course['course_id'])
    target_group=list(course['target_group'])

    #course information
    doc_lis=[]
    for i in range(0,len(desc)):
        doc=''
        doc+=course_name[i]
        if not pd.isnull(sub_group[i]):
            doc+=' '+sub_group[i]
        if not pd.isnull(topic[i]):
            doc+=' '+topic[i]
        if not pd.isnull(target_group[i]):
            doc+=' '+target_group[i]
        doc+=' '+remove_html_tags(desc[i])
        doc = re.sub(r'[^\w\s]', ' ', doc)
        doc_lis.append(doc)

    #user information
    interest=user['interests']
    recreation=user['recreation_names']
    occupation=user['occupation_titles']

    u_doc_lis=[]
    for i in range(0,len(interest)):
        doc=''
        if not pd.isnull(interest[i]):
            doc+=' '+interest[i]
        if not pd.isnull(recreation[i]):
            doc+=' '+recreation[i]
        if not pd.isnull(occupation[i]):
            doc+=' '+occupation[i]
        doc = re.sub(r'[^\w\s]', ' ', doc)
        u_doc_lis.append(doc)

    us_doc=[]
    usr=list(user['user_id'])

    for i in list(test['user_id']):
        idx=usr.index(i)
        us_doc.append(u_doc_lis[idx])
    
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    #encode course
    sentence_embeddings = model.encode(doc_lis)

    #encdoe user
    query_embeddings = model.encode(us_doc)

    #Compute cosine-similarities
    cosine_scores = util.cos_sim(query_embeddings, sentence_embeddings)

    #course popularity
    popularity=[]

    for i in course_id:
        if i in normalized_pop.keys():
            popularity.append(normalized_pop[i])
        else:
            popularity.append(0)
    
    popularity= torch.FloatTensor(popularity)

    #popularity mix with cosine scores
    for i in range(0,len(cosine_scores)):
        cosine_scores[i]=cosine_scores[i]*0.7+popularity*0.3
    
    id=list(course['course_id'])
    pred_lis=[]

    for i in range(0,len(cosine_scores)):
        pred=''
        idx=sorted(range(len(cosine_scores[i])), key=lambda k: cosine_scores[i][k],reverse=True)[:50]
        for j in idx:
            pred+=id[j]+' '
        pred_lis.append(pred[:-1])
    
    test['course_id']=pred_lis

    #save user topic prediction csv file
    test.to_csv(args.course_out_file,index=False)

        #construct subgroup dictionary
    dic=dict()
    idx=0
    name=list(group_df['subgroup_id'])
    for i in list(group_df['subgroup_name']):
        dic[i]=name[idx]
        idx+=1

    c_dic=dict()
    
    idx=0
    for i in list(course['sub_groups']):
        sub=[]
        if not pd.isnull(i):
            for j in i.split(','):
                sub.append(dic[j])
            c_dic[id[idx]]=sub
        idx+=1

    g_dic=dict()
    sub_pred=[]
    for i in list(test['course_id']):
    
        for x in range(1,92):
            g_dic[x]=0
        for j in i.split(' '):
            if j in c_dic.keys():
                for t in c_dic[j]:
                    g_dic[t]+=1
        sort_group=sorted(g_dic.items(), key=itemgetter(1),reverse=True)
        sort_=''
        for ff in sort_group:
            if ff[1]!=0:
                sort_+=str(ff[0])+' '
        sub_pred.append(sort_[:-1])
    
    sub_df=pd.DataFrame()
    sub_df['user_id']=test['user_id']
    sub_df['subgroup']=sub_pred

    #output user topic domain prediction
    sub_df.to_csv(args.subgroup_outfile,index=False)
 
main()