from recommend import Recommender

def get_recommendations(resume_file, requirement_file, use_ngram=False, use_tagger=False):
    with open(resume_file, 'r') as handle:
       resume = handle.read()
    with open(requirement_file, 'r') as handle:
       requirements = handle.read()
    recommender = Recommender()
    #recommender.initialize_attributes(resume, requirements, coursera_vectorizer, coursera_vectors)
    recommender.initialize_attributes(resume, requirements)
    recommender.vectorize_resume()
    recommender.vectorize_requirements()
    missing_requirements = recommender.find_missing_skills()
    course_recommedations = recommender.recommend()
    return missing_requirements, course_recommedations

if __name__ == '__main__':
    resume_files = ['../data/resume_cs.txt', '../data/resume_sci.txt', '../data/resume_ds.txt']
    req_files = ['../data/req_cs.txt', '../data/req_ds.txt', '../data/req_ml.txt']
    for resume_file in resume_files:
        for req_file in req_files:
            print "%s    %s" %(resume_file, req_file)
            missing_requirements, course_recommedations = get_recommendations(resume_file, req_file)
            print "Missing skills:" 
            print missing_requirements
            print 
            print "Course course_recommedations:"
            print course_recommedations
