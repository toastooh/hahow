## Environment
```bash
pip install -r requirements.txt

```
## Run seen domain course prediction and user topic prediction
This file can be executed for seen domain validation and test prediction output.
```bash
python3 pred_seen.py 
    --train_file ${1}  --course_file ${2} --sub_group ${3} 
    --test_file ${4} --course_out_file ${5} --subgroup_outfile ${6} --user_file ${7} --threshold

```
train_file: Path to the train file(e.g. train.csv)

course_file: Path to the courses file(e.g. courses.csv)

sub_group: Path to the subgroups file(e.g. subgroups.csv)

test_file: Path to the test file(e.g. test_seen.csv)

course_out_file: Course prediction file path

subgroup_outfile: User Topic prediction file path

user_file: Path to the user file(e.g. users.csv)

threshold: If passed, use threshold on cosine similarity.

## Run unseen domain course prediction and user topic prediction
This file can be executed for unseen domain validation and test prediction output.
```bash
python3 pred_unseen.py 
    --train_file ${1}  --course_file ${2} --sub_group ${3} 
    --test_file ${4} --course_out_file ${5} --subgroup_outfile ${6} --user_file ${7}

```
train_file: Path to the train file(e.g. train.csv)

course_file: Path to the courses file(e.g. courses.csv)

sub_group: Path to the subgroups file(e.g. subgroups.csv)

test_file: Path to the test file(e.g. test_seen.csv)

course_out_file: Course prediction file path

subgroup_outfile: User Topic prediction file path

user_file: Path to the user file(e.g. users.csv)

## Reproduce seen domain course/topic prediction
```bash
bash ./run_seen.sh ./path/to/train.csv ./path/to/courses.csv ./path/to/subgroups.csv ./path/to/test_seen.csv ./path/to/course_prediction.csv ./path/to/topic_prediction.csv ./path/to/users.csv
```

## Reproduce unseen domain course/topic prediction
```bash
bash ./run_unseen.sh ./path/to/train.csv ./path/to/courses.csv ./path/to/subgroups.csv ./path/to/test_seen.csv ./path/to/course_prediction.csv ./path/to/topic_prediction.csv ./path/to/users.csv
```
