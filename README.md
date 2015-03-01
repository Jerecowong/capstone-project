# CourseBridge: a Tailored Coursera Course Recommender Based on Your Profile and a Job Posting

## Summary
The NLP based web application is aimed to help people who are looking for jobs. In addition to helping them to identify their most matched qualifications for a job posting they are interested in, the application can also help them to be better prepared for this job by offering a list of Coursera courses based on their missing skill sets for the very same job.

I might include edX courses if I have enough time.

## Motivation
Checking a job posting, matching the qualifications, identifying missing skills and figuring out a resources to makeup the needed skills are very typical procedures of job hunters.

The app is meant to make the process a little more efficient and less tedious, which can be useful to direct people to find resources on Coursera and to help them to sharpen the skills needed for their desired position.

## Data Sources
Getting Coursera course feature data is easy, I scraped the data using their API.

I have the data stored in MongoDB. Each doc has course name and a short description for the course.

```python
{
"_id" : ObjectId("54e95b788d804728dd351bb4"),
"language" : "en",
"links" : {
},
"shortName" : "perceptivehunting",
"shortDescription" : "The rich history of wildlife management and recreational hunting plays an important role in the evolving face of conservation. This course will explore the ethics, science, and democracy of conservation, hunting, and The Land Ethic in North America.",
"id" : 2163,
"name" : "The Land Ethic Reclaimed: Perceptive Hunting, Aldo Leopold, and Conservation"
}
```

Description will be used for similarity check.

Name and shortName will be used to to construct the hot links for the recommendation page, it has a pattern of
baseurl "www.coursera.org/course"  + shortName. For exmple: [https://www.coursera.org/course/perceptivehunting] (https://www.coursera.org/course/perceptivehunting)

## Process
There are 2 webpages for the app:

* Input page

	2 text boxes for pasting the resume and job posting. Urls pointing to resume and job posting are also acceptable.
The background will be a heat map for the courses based on their subjects, which shows some insight
of what are taught currently.

* Output page

	The top qualifications you have for the job and a recommender of courses based on your missing skills.

* Implementation plan

	1. Build a TfidfVectorizer using the Coursera course shortDescription.
	2. Parsing the job posting and form the requirements into a list of requirement. Build tfidf for the list.
	3. Build the tfidf for the resume.
	4. Run similarity check against the resume tfidf for each requirement.
	5. Display top 2 qualifications.
	6. The missing set are the ones with very low similarity. Those will be used to form a list for querys for Coursera data.
	7. Search the Coursera data with the queries and return list of items with highest similarity.