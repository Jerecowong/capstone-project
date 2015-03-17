# CourseBridge: a Tailored Coursera Course Recommender Based on Your Resume and a Job Posting

## Summary
The NLP based web application is aimed to help people who are looking for jobs. In addition to identifying their missing skill sets for the job they are interested in, the app will offer a list of Coursera courses based on each missing requirement, which ensure them to be better prepared to meet all the requirements for the job.

## Motivation
Checking a job posting, matching the qualifications, identifying missing skills and then figuring out a resources to cover up the needed skills is a very typical procedure for job hunters.

The app is designed to make the process a little more efficient and less tedious. It can be useful to direct people to  resources on Coursera so that they can sharpen the skills needed for their desired position by taking the right courses.

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
"shortDescription" : "The rich history of wildlife management and recreational hunting plays 
 an important role in the evolving face of conservation. This course will explore the ethics, 
 science, and democracy of conservation, hunting, and The Land Ethic in North America.",
"id" : 2163,
"name" : "The Land Ethic Reclaimed: Perceptive Hunting, Aldo Leopold, and Conservation"
}
```

Description will be used for similarity check.

Name and shortName will be used to to construct the hot links for the recommendation page. The pattern is
baseurl "www.coursera.org/course/"  + shortName. 
For example: [https://www.coursera.org/course/perceptivehunting] (https://www.coursera.org/course/perceptivehunting)

## Process
There are 2 web pages for the app:

* Input page

	2 text boxes for pasting the resume and job requirements. 

* Output page

	The missing qualifications and a recommender of courses for each missing requirement.


* Implementation details

	1. Build a TfidfVectorizer using the Coursera course shortDescription.
	2. Parse the job posting and split the requirements into a list of requirement. Build tf-idf for the list.
	3. Build the tf-idf for the resume.
	4. Run similarity check against the resume tf-idf for each requirement.
	5. Form a list of queries for Coursera course data using the set of missing requirements, which have very low similarity as the resume. 
	6. Search the Coursera data with the queries and return a list of courses with highest similarity scores for each query.


## Files contained in this repo:

* [code](https://https://github.com/Jerecowong/capstone-project/tree/master/code)
    contains the following:

	app.py: flask app

	CourseraTokenizer.py: CourseraTokenizer class

	experiment.py: text based application for experiment convenience

	coursera_desc.py:  Python script to scrape the Coursera course data

	recommender.py: recommender class, the core part for the app


* [data](https://github.com/Jerecowong/capstone-project/tree/master/data) 
	contains the Coursera course data and testing data. 


* [test](https://github.com/Jerecowong/capstone-project/tree/master/test)
	contains the unit test:  tests.py


* [webapp](https://github.com/Jerecowong/capstone-project/tree/master/webapp)
	contains Python files, html templates, static style sheets, and their dependencies.


## Possible future work
	1. Extend the course recommendations to include edX and Udacity
	2. Alter to help HR and recruiters with screening the candidates
	3. Find a better approach to identify the real missing skills, maybe by adding a mini classifier to classify each word in the requirement to ensure focus on “real skills and technologies”

## Web App (can be found in webapp/*)
The final product can be found at:  
http://jerecowong.pythonanywhere.com/

## Acknowledgements
This project uses the Python packages including sklearn, nltk, TextBlob, pandas and flask; And most of all, support and guidance from Zipfian Academy.