from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
from jobs import recommend as fulltime_recommend, for_you as fulltime_for_you
from instant import recommend as instant_recommend, for_you as instant_for_you
from gigs import recommend as gig_recommend, for_you as gigs_for_you
from typing import List
# import jobs
app = FastAPI()

class JobHistoryRequest(BaseModel):
    ids: List[str]

@app.get("/jobs")
def read_root(q: Union[str, None] = None):
    jobs = fulltime_recommend(q)
    return jobs

@app.post("/jobs/for-you")
def read_root(request: JobHistoryRequest):
    jobs = fulltime_for_you(request.ids)
    return jobs

@app.get("/instant-jobs")
def read_root_2(q: Union[str, None] = None):
    jobs = instant_recommend(q)
    return jobs

@app.post("/instant-jobs/for-you")
def read_root(request: JobHistoryRequest):
    jobs = instant_for_you(request.ids)
    return jobs


@app.get("/gigs")
def read_root_3(q: Union[str, None] = None):
    jobs = gig_recommend(q)
    return jobs

@app.post("/gigs/for-you")
def read_root(request: JobHistoryRequest):
    jobs = gigs_for_you(request.ids)
    return jobs

    