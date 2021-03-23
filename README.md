
# Simulated Annealing
Simulated annealing algorithm with KNN Classifier.
## Dataset
Forest Cover Type Data From UCI dataset is used in this repo. The data set consists of 54 columns (features) in total. Since the data set consists of 581012 lines in total during the application, only 2000 of these lines were taken into account when classifying. The reason for this is that there is no equipment to classify with such large data.

The Cover_Type feature was used as a target for classification application. It contains 7 types; 
1. Spruce/Fir 
2. Lodgepole Pine 
3. Ponderosa Pine 
4.  Cottonwood/Willow 
5. Aspen 
6. Douglas-fir 
7. Krummholz List 

Here, only Spruce / Fir and Lodgepole Pine classes were taken and binary classes to be used in classification were obtained.

## Algorithm
First, it was determined how to lower the temperature at each iteration. This work started with a temperature of 90 degrees and the current temperature was reduced by 0.01 step until it reached a final temperature of 0.1 degrees. Later, the initial_state was determined and assigned to the solution. In the loop, this process is repeated until current_temp is lower than final_temp. For each iteration, a random neighbor of current_state is received (new dataframe consisting of 0 and 1). Then the differences between neighbor and current_state are calculated. If the new solution is better, it is accepted. The new solution was still accepted if the temperature was high, if not better. Because according to the logic of the algorithm, the worst solution is also used to avoid getting stuck at the local minimum. With the probability equation, a neighbor that is not slightly worse than current_state is taken. The next step is reducing the current temperature according to the alpha value. So finally, the current_state is returned.
