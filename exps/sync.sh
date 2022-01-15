rsync -avu --include="*/" --include="train_cv_score.csv" --exclude="*" /mnt/takamisato/working/kaggle_petfinder/ hpc3:/pub/takamis/kaggle_petfinder/
#rsync -avu --include="*/" --include="*.py" --exclude="*" /mnt/takamisato/working/kaggle_petfinder/ hpc3:/pub/takamis/kaggle_petfinder/
#rsync -avu --include="*/" --include="*.sh" --exclude="*" /mnt/takamisato/working/kaggle_petfinder/ hpc3:/pub/takamis/kaggle_petfinder/
rsync -avu --include="*/" --include="train_cv_score.csv" --exclude="*" hpc3:/pub/takamis/kaggle_petfinder/exps/ /mnt/takamisato/working/kaggle_petfinder/exps/
