wget 'https://www.dropbox.com/s/de1p5do5sit8qsr/model_190.pth.tar?dl=1'
python3 main.py --mode inference --output_dir $2 --validate_dir $1 \
    --resume model_190.pth.tar?dl=1 --model SC-FEGAN
