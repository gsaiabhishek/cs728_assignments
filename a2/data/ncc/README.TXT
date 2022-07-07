th = Tratz-Hovy data set
th/th.csv lines look like this:
earthquake,zone,0,CONTAIN,CONTAIN.area
The columns are (most likely) in the format
first word, second word, (unknown), coarse label, fine label
There are 37 coarse labels and 120 fine labels.

f = Farahmand data set
f/f.csv lines look like this:
action,film,0.0
blood,bath,1.0
This means "action film" is fully compositional 
whereas "blood bath" is not at all compositional.

r1, r2 = Reddy data sets
r1/r1.csv and r2/r2.csv have lines that look like this:
silver,spoon,0.6962962
snake,oil,0.85
graduate,student,0.02
engine,room,0.0137932
This means "silver spoon" and "snake oil" are not compositional,
whereas "graduate student" and "engine room" are.

The difference between Farahmand and Reddy data is that
Farahmand gives an extreme 0/1 judgment while Reddy is graded.

https://aclanthology.org/P10-1070/
https://aclanthology.org/W15-0904/
http://www.aclweb.org/anthology/I11-1024
