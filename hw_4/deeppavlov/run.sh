python -m venv deeppavlov-venv
deeppavlov-venv/bin/pip install deeppavlov corus
file=lenta-ru-news.csv.gz
if [ -e "$file" ]; then
    echo "lenta-ru-news exists"
else 
    wget https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.0/lenta-ru-news.csv.gz
fi 
deeppavlov-venv/bin/python3 main.py

