# learn-to-tensorflow
Let's learn to machine-learning, deep-learning with tensorflow

Anaconda 2018.12 (python 3.7)

Visual Studio Code

lectures: [모두를 위한 머신러닝/딥러닝 강의](http://hunkim.github.io/ml/)


- - -

 ### Local CloudML Test

```
$ gcloud ml-engine local train ^
    --module-name train.1-multiply ^    
    --package-path train/
```

### Cloud Environment CloudML Test

```
$ gcloud ml-engine jobs submit training $JOB_NAME ^
    --job-dir $OUTPUT_PATH ^
    --module-name train.1-multiply ^
    --package-path train/ ^
    --region $REGION ^
    --runtime-version 1.2
```
