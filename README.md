# CourseBridge: a Tailored Coursera Course Recommender Based on Your Resume and a Job Posting

## Summary
The NLP based web application is aimed to help people who are looking for jobs. In addition to identifying their missing skill sets for the job they are interested in, the app will offer a list of Coursera courses based on each missing requirement, which ensure them to be better prepared to meet all the requirements for the job.

## Motivation
Checking a job posting, matching the qualifications, identifying missing skills and then figuring out a resources to cover up the needed skills is a very typical procedure for job hunters.

The app is designed to make the process a little more efficient and less tedious. It can be useful to direct people to  resources on Coursera so that they can sharpen the skills needed for their desired position by thaking the right courses.

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
baseurl "www.coursera.org/course/"  + shortName. For exmple: [https://www.coursera.org/course/perceptivehunting] (https://www.coursera.org/course/perceptivehunting)

## Process
There are 2 webpages for the app:

* Input page

	2 text boxes for pasting the resume and job requirements. 

* Output page

	The missing qualifications and a recommender of courses for each missing requirement.


* Implementation detail

	1. Build a TfidfVectorizer using the Coursera course shortDescription.
	2. Parsing the job posting and form the requirements into a list of requirement. Build tfidf for the list.
	3. Build the tfidf for the resume.
	4. Run similarity check against the resume tfidf for each requirement.
	5. The missing set of requirements are the ones with very low similarity score. Those will be used to form a list of querys for Coursera data.
	6. Search the Coursera data with the queries and return list of courses with highest similarity for each query.


## Files contained in this repo:

* [code](https://https://github.com/Jerecowong/capstone-project/tree/master/code)
	app.py
	CourseraTokenizer.py
	experiment.py
	recommend.py
	coursera_desc.py  
	CourseraTokenizer.py

* [data](https://github.com/Jerecowong/capstone-project/tree/master/data)
	Contains the Coursera course data and testing data. 
* [test](https://github.com/Jerecowong/capstone-project/tree/master/test)
	Contains the unit test.
* [webapp](https://github.com/Jerecowong/capstone-project/tree/master/webapp)
	Contains the dependency of the Python files, html templates and statics style sheets.

## Possible future work
	1. Extend the course recommendation to include edX and Udacity
	2. Alter to help HR and recruiter with screening the candidates
	3. Find a better approach to identify the real missing skills, maybe by adding a mini classifier to classify each word in the requirement to ensure to focus on “real skills and technologies”

## Web App (can be found in webapp/*)
The final product can be found at:  
http://jerecowong.pythonanywhere.com/

## Acknowledgements
This project uses the Python packages including sklearn, nltk, TextBlob, pandas and flask; And most of all, support and guidance from Zipfian Academy.