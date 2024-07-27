import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import sqlite3
ps = PorterStemmer()

conn = sqlite3.connect('./test.db')
# jobs = pd.read_json("fulltime.json")
jobs = pd.read_sql('SELECT FulltimeJob.*, JobType.name as category, Location.name as location FROM FulltimeJob JOIN JobType ON FulltimeJob.jobTypeId = JobType.id JOIN Location ON FulltimeJob.locationId = Location.id',conn)

def convert_to_text(text):
  # Assuming the description is stored in a variable called 'description_html'
  description_text = BeautifulSoup(text, 'html.parser').get_text()
  return description_text

def stem(text):
  y=[]
  for i in text.split():
    y.append(ps.stem(i))
  return " ".join(y)


jobs["description"] = jobs["description"].apply(convert_to_text)
jobs["excerpt"] = jobs["excerpt"].apply(lambda x:x.split())
jobs["description"] = jobs["description"].apply(lambda x:x.split())
jobs["experience"] = jobs["experience"].apply(lambda x:x.split())
jobs["educationLevel"] = jobs["educationLevel"].apply(lambda x: x.split() if x is not None else None)
jobs['category'] = jobs['category'].apply(lambda x: ["".join(x)])
jobs["level"] = jobs["level"].apply(lambda x: [x] if x is not None else [])
# jobs["skills"] = [skill.split(', ') for skill in jobs["skills"]]
jobs["skills"] = jobs["skills"].apply(lambda x: [' '.join(x)])
jobs["tags"] = jobs["excerpt"] + jobs['description'] + jobs["educationLevel"] + jobs["experience"] + jobs["skills"] + jobs["category"] + jobs["level"]
new_jobs = jobs[["id","title","tags"]]
new_jobs.dropna(inplace=True)
new_jobs["tags"] = new_jobs['tags'].apply(lambda x:' '.join(x))
new_jobs["tags"] = new_jobs["tags"].apply(lambda x: x.lower())

cv = CountVectorizer(max_features = 100, stop_words="english")
vectors = cv.fit_transform(new_jobs['tags']).toarray()
new_jobs['tags'] = new_jobs["tags"].apply(stem)
similarity = cosine_similarity(vectors)

def recommend(job):
    # Check if the job exists in the new_jobs DataFrame
    if job not in new_jobs['id'].values:
        print("Job ID not found.")
        return []

    # Get the index of the job
    job_index = new_jobs[new_jobs['id'] == job].index[0]

    # Calculate distances
    distances = similarity[job_index]

    # Sort jobs based on distance and get top 5 similar jobs
    jobs_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    # Create a list of dictionaries with job id and percentage matched
    recommended_jobs = [{'id': new_jobs.iloc[i[0]].id, 'percentage_matched': round(i[1] * 100, 2)} for i in jobs_list]

    return recommended_jobs
    # Print job IDs of the recommended jobs
    # for i in jobs_list:
    #     print(new_jobs.iloc[i[0]].id)

def for_you(user_history):
    # Dictionary to store aggregated recommendations
    aggregated_recommendations = {}
    print(user_history)

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