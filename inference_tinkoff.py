import pandas as pd
from pathlib import Path
from tqdm import tqdm
from graphrag.cli.query import run_local_search

df = pd.read_excel('D:\\Scripts\\TestGraphRAG\\T-results.xlsx')
df = df.drop(columns=['team', 'Оценка', 'Причина', 'ОтветСервиса', 'ОтветСервисаСсылки'])
df = df.drop_duplicates()
tmp = df['Вопрос'].to_list()

empty_answers = 0
all_answers = []
for el in tqdm(tmp):
    response, _ = run_local_search(config_filepath=None,
                data_dir=None,
                root_dir=Path('./tinkoff_full'),
                community_level=2,
                response_type='str',
                streaming=False,
                query=el)

    all_answers.append(response)

out = pd.DataFrame({'questions': tmp, 'answers': all_answers})
out.to_csv('tinkoff_full_1_answers.csv', index=False)