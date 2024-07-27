
import pandas as pd
import numpy as np
import sqlite3

conn = sqlite3.connect('../creig-server-1/prisma/test.db')

jobs = pd.read_sql('SELECT InstantJobs.*, JobType.name as category, Location.name as location FROM InstantJobs JOIN JobType ON InstantJobs.typeId = JobType.id JOIN Location ON InstantJobs.locationId = Location.id',conn)
jobs.head()

jobs["description"] = jobs["description"].apply(lambda x:x.split())

jobs['category'] = jobs['category'].apply(lambda x: ["".join(x)])

jobs['location'] = jobs['location'].apply(lambda x: ["".join(x)])

jobs["tags"] = jobs["description"] + jobs["category"] + jobs["location"]

new_jobs = jobs[["id","title","tags"]]
new_jobs.dropna(inplace=True)
new_jobs.info()

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 100, stop_words="english")

new_jobs["tags"] = new_jobs['tags'].apply(lambda x:' '.join(x))

cv.fit_transform(new_jobs["tags"]).toarray().shape

vectors = cv.fit_transform(new_jobs['tags']).toarray()

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
  y=[]
  for i in text.split():
    y.append(ps.stem(i))
  return " ".join(y)

new_jobs['tags'] = new_jobs["tags"].apply(stem)

from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity(vectors)

similarity = cosine_similarity(vectors)


def recommend(job):
    # Check if the job exists in the new_jobs DataFrame
    print(jobs)
    if job not in new_jobs['id'].values:
        print("Job ID not found.")
        return

    # Get the index of the job
    job_index = new_jobs[new_jobs['id'] == job].index[0]

    # Calculate distances
    distances = similarity[job_index]

    # Sort jobs based on distance and get top 5 similar jobs
    jobs_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    # recommended_job_ids = [new_jobs.iloc[i[0]].id for i in jobs_list]
    recommended_jobs = [{'id': new_jobs.iloc[i[0]].id, 'percentage_matched': round(i[1] * 100, 2)} for i in jobs_list]

    return recommended_jobs
    # Print job IDs of the recommended jobs
    # for i in jobs_list:
    #     print(new_jobs.iloc[i[0]].id)
    


def for_you(user_history):
    # Dictionary to store aggregated recommendations
    aggregated_recommendations = {}

    for job_id in user_history:
        recommended_jobs = recommend(job_id)
        for job in recommended_jobs:
            job_id = job['id']
            if job_id in user_history:  # Skip if the job is already in user history
                continue
            match_score = job['percentage_matched']
            if job_id in aggregated_recommendations:
                aggregated_recommendations[job_id] += match_score
            else:
                aggregated_recommendations[job_id] = match_score

    # Sort aggregated recommendations by match score in descending order
    sorted_recommendations = sorted(aggregated_recommendations.items(), key=lambda x: x[1], reverse=True)

    # Convert sorted recommendations to a list of dictionaries
    final_recommendations = [{'id': job_id, 'percentage_matched': round(score, 2)} for job_id, score in sorted_recommendations]

    return final_recommendations