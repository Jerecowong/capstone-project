from recommend import Recommender

def get_recommendations(resume_file, requirement_file, ngram_range=(1,1), use_tagger=False):
    with open(resume_file, 'r') as handle:
       resume = handle.read()
    with open(requirement_file, 'r') as handle:
       requirements = handle.read()
    recommender = Recommender(ngram_range=ngram_range, use_tagger=use_tagger)
    #recommender.initialize_attributes(resume, requirements, coursera_vectorizer, coursera_vectors)
    recommender.initialize_attributes(resume, requirements)
    recommender.vectorize_resume()
    recommender.vectorize_requirements()
    missing_requirements = recommender.find_missing_skills()
    print "Requirements:"
    print recommender.requirements
    print "preprocessed_requirements:"
    print recommender.preprocessed_requirements
    print "recommender.missing_requirements"
    print recommender.missing_requirements
    course_recommedations = recommender.recommend()
    return missing_requirements, course_recommedations

if __name__ == '__main__':
    #resume_files = ['../data/resume_cs.txt', '../data/resume_sci.txt', '../data/resume_ds.txt']
    #req_files = ['../data/req_cs.txt', '../data/req_ds.txt', '../data/req_ml.txt']
    resume_files = ['../data/resume_h.txt']
    req_files = ['../data/req_h.txt']
    for resume_file in resume_files:
        for req_file in req_files:
            print "%s    %s" %(resume_file, req_file)
            missing_requirements, course_recommedations = get_recommendations(resume_file, req_file,\
                ngram_range=(1,1), use_tagger=True)
            print "Missing skills:" 
            print missing_requirements
            print 
            print "Course course_recommedations:"
            print course_recommedations

