"""
IR Assignment-1
"Songle" - A Song Based Information Retrieval System 
Authors: Dhruv Shah
        Vedant Goyal
        Ashish Gupta
"""

"""
All the necessary libraries and files are imported here such as numpy, os, operator etc.
"""
import numpy as np
import operator
from numpy import linalg as la

""" This is a python program for a text based information retrieval system.
Here the user will enter a query for searching a song and will receive the top 10 song results
pertaining to that query and will also receive top 10 most searched song recommendations. 
"""

""" This para opens the words text file in read only format containing all the 
words present in the song dataset. 
The split() function converts all the words into a elements of the list tokens. 
"""
file = open("words.txt",'r')
dataset = file.read()
tokens=dataset.split(',')

"""
This para opens the tracks text file in read only format containing the trackID
and the occurences of that word in the track.  
"""
tracks= open("tracks.txt",'r')
trackset=tracks.read()
trackset=trackset.split()

"""num_of_tracks stores the total number of documents present in the trackset
top_songs is a list containing the top most searched songs  
"""
num_of_tracks= len(trackset)
#-----------
top_songs={}
for i in range(0,210519):
    top_songs[i]=0
    
""" main_dict is a dictionary of dictionary which stores all the words as keys and 
the docIDs and the occurences of the words in those docIDs 

This is similar to an inverted index containing docIDs in primary posting list and 
the occurences of the word in that doc as a secondary posting list.
"""    
total_words_in_a_doc=[]
main_dict={}
for i in range(0,5001):
    main_dict[i]={}
  
"""doc contains the info of a single track.

track_name contains the name of the track.

word_id contains the id of the word and count stores the occurences of that word for 
a particular track

sum is the total number of all word occurences in a track

sum1 is the total number of all word occurences in the whole dictionary

This loop stores the total words in the docs in a list form and finds their counts for
storing the term frequency.
"""
track_name=[]
sum1=0
for j in range(0,210519):
    doc= trackset[j].split(',')
    docsize= len(doc)
    sum=0
    
    track_name.append(doc[0])
    for i in range(2,docsize):
        splitdoc=doc[i].split(':')
        word_id=int(splitdoc[0])
        count=int(splitdoc[1])
        sum+=count
        main_dict[word_id][j]=count
    sum1+=sum
    total_words_in_a_doc.append(sum)
"""This function returns the track name corresponding to the track id
"""
def get_track_name(id):
        return track_name[id]
    
"""This function prints all the output songs based on the query according 
to their trackIds which are passed to the function
"""    
def get_songs_based_on_query(track):
    for x,y in track.items():
        print(get_track_name(x))

"""This function prints the most searched top 10 songs.
"""
def get_top_songs(track):
    for (x,y) in track:
        print(get_track_name(x))

#-----------------------------------

#-----------------------------------
"""This function accepts a song query and creates a list str of all the words present
in the query .

store is a dictionary which stores the words from the query as keys and the corresponding
occurences of those words from the query.

flag integer tells if the query word is present in the dictionary or not.
flag is zero if all words of the query not present in the dictionary.

query stores the words of the query whereas query_list stores the occurences of those words.

vector_list is a matrix formed according to the Vector Space Model.
It stores only those words as columns which are present in the query and the 
rows as docIds and the tf-idfs of those words in the cells corresponding to each document.

the for loop calculates the idf for each document and tf for each term and then 
multiply both to calculate the tf-idf which will then be stored in the corresponding 
cells of the vector_list matrix.

Now the cosine similarity is calculated between the query and the vectors (rows in vector_list matrix)
to find which documents/tracks are the most nearest to the given query.
This is now sorted and stored in the sorted_final_dict which contains the keys as trackIds 
and values as the cosine similarity of the vector with query words.

Now the top 10 query related songs are returned and presented to the user.

Finally, the top most 10 searched songs are also returned to the user as an addition
to the previous result.
"""
def get_songs_given_query():
    str= input("Enter Song Query")
    str = str.split()
    query_size=len(str)
    query=[]
    store={}
    flag=0
    for i in range(0,query_size):
        if str[i] in dataset:
            flag=1
            if str[i] in store:
                store[str[i]]+=1
            else:
                store[str[i]]=1
    
    
    
    if flag==0:
        print("sorry!No songs found")
    else:
        query_list=[]        
        for x,y in store.items():
            query.append(x)
            query_list.append(y)
        query_size=len(query)
        
        vector_list=np.zeros( (210519, query_size) )
        count=0
        
        for qword in query:
            qword_id=tokens.index(qword)+1
            num_of_docs_having_qword=len(main_dict[qword_id])
            
            idf=np.log(210519/(num_of_docs_having_qword+1))
            
            for doc_id in main_dict[qword_id]:
                tf_for_a_doc=main_dict[qword_id][doc_id]/total_words_in_a_doc[doc_id]
                #tf_idf_for_docs_having_qword[doc_id]=(tf_for_a_doc*idf)
                vector_list[doc_id][count]=(tf_for_a_doc*idf)
            count+=1
                
        query_list=np.array(query_list)
        
        final_dict={}
        for i in range(0,210519):
            temp=la.norm(vector_list[i])*la.norm(query_list)
            if(temp==0):
                temp=1
            final_dict[i]=np.dot(vector_list[i], query_list)/temp
        
        sorted_final_dict = sorted(final_dict.items(),key=operator.itemgetter(1),reverse=True)
        final_ans=sorted_final_dict[0:10]
        final_ans=dict(final_ans)
        
        
        #----------------------------------------------
        global top_songs
        top_songs=dict(top_songs)
        for i,j in final_ans.items():
            top_songs[i]+=1
        top_songs=sorted(top_songs.items(),key=operator.itemgetter(1),reverse=True)
        
        
        
        
        print("Here are the top 10 songs according to the query")
        get_songs_based_on_query(final_ans)
        print("\n\n\n\n")
        
    print("Here are the top 10 searches by the users")
    get_top_songs(top_songs[0:10])











