# Converting Event Streams to Input Data for Networks

1. Regardless of source, all event streams start as h5 files
2. Convert the h5 file to a csv file
3. Split the csv file into groups of an equal number of events
4. Convert split csv file into image data for network
5. (Optional) Log data for simplifying class construction

## Convert h5 Event Stream to Full CSV Event Stream

Regardless of event stream source (Prophesee Event Camera or V2E Event Reconstruction) there are 4 columns: x-coordinate, y-coordinate, polarity (0,1), and time ($\mu s$). Prophesee cameras use the order `t,x,y,p`. V2E streams use the order `x,y,p,t`.

We recommend storing raw h5 files, and processed csv files in separate folders for backup.

**See:** [h5EventFileToCSV.py](h5EventFileToCSV.py) for conversion process

## Split Full CSV Event Stream to Equal Sized Chunks

Each large event stream is then split into chunks of an equal number of events. This number is the *accumulator*. This number is variable, although 50,000 events is the preferred accumulator. This can be further split with proper attention to final image data.

Each new csv file is renamed according to the iterator value and the original name of the file, to make for easy sorting. Each of these groups of new csv files can be stored in separate folders named according to the original event streams.

**See:** [splitFullCSVToEqualFrames.py](splitFullCSVToEqualFrames.py) for splitting process

*Note: The final file will be shorter than 50,000 events. To accommodate this, the file [removeSmaller.sh](removeSmaller.sh) will remove the offending file.*

## Convert Equal Sized Chunks to Image Input Data

At this point there will be a folder of equal sized csv files. This is the input folder. Create an output folder for the images that will be generated from this process. Prophesee streams are converted to raw images of dimension (720 x  1280). V2E streams are converted to raw images of dimension (768 x 1024).

**See:** [csvToImg.py](csvToImg.py) for this creation process.

## (Optional) Logging Class Data for Class Construction

Depending on the source and the experiment characteristics, an event stream may contain more than 1 regime. In this case, scrub through image data, classifying data according to nearest regime. Log information to ensure completion and avoid redundancy.

**See:** [Experiment_Log.xlsx](Experiment_Log.xlsx) for an example

## Splitting Complete Class Dataset into Test and Train Data

Once a directory of folders, each containing only one class, is constructed, the PyPI package [split-folders](https://pypi.org/project/split-folders/) can be used to create the final dataset.
