# NeuroFlow

## Converting Event Streams to Input Data for Networks

1. Regardless of source, all event streams start as h5 files
2. Convert the h5 file to a csv file
3. Split the csv file into groups of an equal number of events
4. Convert split csv file into image data for network
5. (Optional) Log data for simplifying class construction

**See:** [Event_Funcs Readme for More Info](Event_Funcs/Event_Streams_To_Input_Data.md)

## Model Documentation

1. The CNN is split into Optical and Event versions. There is also a Unified Event version and a Reduced version for Paper Ablation
2. ConvNeXtNet contains Event and Optical versions for both ConvNeXt and AlexNet
3. The LSTM is only an Event version
4. The SNN only has the Event and Unified Event versions
5. The FT Method provides a method to create data tables for the KNN method. Then KNN can be run to evaluate and create confusion matrices

**See:** [Models Readme for More Info](Models/Directions_For_Models.md)

## Evaluating Models and Creating Confusion Matrices

1. Move PTH files into this directory
2. Update each file with appropriate PTH file
3. Run code

**See:** [PTHTester Readme for More Info](PTHTester/Confusion_Matrix_Evaluation.md)

## See Misc Experimental Settings for Experimental Information