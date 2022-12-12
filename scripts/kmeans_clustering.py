import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
from sklearn import preprocessing

from fairmotion.data import bvh
import motionspan
import motiondensity

# TODO : make csv / excel file 
def get_motion_file_names():
    path_dir = 'motion_trimmed/'
    file_list = os.listdir(path_dir) 
    return file_list

def get_postural_features(frame_start, frame_end):
    path_dir = 'motion_trimmed/'
    file_list = os.listdir(path_dir)
    
    motion_spans = []
    motion_densities = []
    
    for i in range(len(file_list)):
        print("[" ,i, "] starts")
        
        BVH_FILE = os.path.join(path_dir, file_list[i])
        motion = bvh.load(BVH_FILE)
        motion_span = motionspan.extract_motion_span(motion, frame_start, frame_end, True)
        motion_spans.append(motion_span)

        print("[" ,i, "] span finished")
        
        motion_density = motiondensity.extract_motion_density(motion, frame_start, frame_end, is_local=False, is_sum_all=True)
        motion_densities.append(motion_density)
        
        print("[" ,i, "] density finished")
        
    return motion_spans, motion_densities
      
def run_kmeans():
    
    data = pd.read_csv('PosturalFeatures.csv')
    data.head()
    
    # copy original data to process 
    processed_data = data.copy()
    
    # data preprocessing (normalization)
    scaler = preprocessing.MinMaxScaler()
    processed_data[['MotionSpan', 'MotionDensity']] = scaler.fit_transform(processed_data[['MotionSpan', 'MotionDensity']])
    
    # Tests by increasing K 
    if False:
        for i in range(6,11):
            estimator = KMeans(n_clusters=i)
            ids = estimator.fit_predict(processed_data[['MotionSpan', 'MotionDensity']])
            
            plt.subplot(3, 2, i-5)
            plt.tight_layout()
            plt.title("K value = {}".format(i))
            plt.scatter(processed_data['MotionSpan'], processed_data['MotionDensity'], c=ids)
    
    # Set K with fixed value
    if True:
        # create clusterings
        estimator = KMeans(n_clusters=7)
        cluster_ids = estimator.fit_predict(processed_data[['MotionSpan', 'MotionDensity']])

        labels = estimator.labels_
        centroids = estimator.cluster_centers_
        
        # create a scatter plot 
        plt.scatter(processed_data['MotionSpan'], processed_data['MotionDensity'], c=cluster_ids)
        plt.scatter(centroids[:,0], centroids[:,1], s=30, color='magenta')
        
        # make label(motion_file_name) on data 
        if True:
            for idx, motion_files, motion_span, motion_density in processed_data.itertuples():
                txt = motion_files.split('_')[4]
                txt = txt[1:2]
                plt.annotate(text=txt, xy=(motion_span, motion_density))
    
    
    plt.xlabel('MotionSpan')
    plt.ylabel('MotionDensity')
    plt.show()

def main():
    # Create DataFrame
    if True:
        FRAME_START = 0
        FRAME_END = 1800
        
        motion_file_names = get_motion_file_names()
        motion_spans, motion_densities = get_postural_features(FRAME_START, FRAME_END)
        
        df = pd.DataFrame(motion_file_names, columns=['MotionFiles'])        
        df['MotionSpan'] = motion_spans
        df['MotionDensity'] = motion_densities
        
        # Save DataFrame in csv file
        df.to_csv("TEST.csv", index = False)

    if False:
        run_kmeans()
    
if __name__ == "__main__":
    main()