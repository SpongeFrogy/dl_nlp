from deeppavlov import build_model
from corus import load_lenta
import pickle
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def main():
    logging.info('Loading lenta-ru-news...')
    records = load_lenta('lenta-ru-news.csv.gz')

    logging.info('Building ner_collection3_bert...')
    model = build_model('ner_collection3_bert', download=True, install=True)
    
    texts = [record.title for record in records][:10_000] # берем title так как надо ужаться в max_tokens_len = 512
    tokens, tags = [], []
    batch_size = 100
    for start, end in tqdm(zip(range(0, len(texts), batch_size), range(batch_size, len(texts), batch_size)), total=len(texts)//batch_size, unit='batches', desc='Predicting'): 
        batch = texts[start:end]
        try:
            batch_tokens, batch_tags = model(batch)
            tokens.extend(batch_tokens)
            tags.extend(batch_tags)
        except RuntimeError:
            logging.warning(f'RuntimeError at {start}')
            continue

    
    with open('ner.pkl', 'wb') as f:
        pickle.dump((tokens, tags), f)
    logging.info('Done!')

if __name__ == '__main__':
    main()