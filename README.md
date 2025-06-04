# Apollo

![](/Apollo/assets/pref_github.png)

👉 [Прочитать документацию](https://microsoft.github.io/graphrag)<br/>
👉 [GraphRAG Arxiv](https://arxiv.org/pdf/2404.16130)

## Обзор

Apollo представляет из себя систему, ядром которая является модификация реализации архитектуры GraphRAG. В отличие от оригинала, в котором доступны лишь модели от OpenAI, данное решение предоставляет возможность использовать модели с открытым исходным кодом (и некоторые закрытые). Все модели, чьи токенизаторы доступны на площадке HuggingFace, можно использовать в данном проекте.

## Архитектура системы

![](/Apollo/assets/Deployment%20Diagram.png)

Архитектурно система реализована на языке Python. Серверная часть, отвечающая за API и бизнес-логику, построена на фреймворке FastAPI. Взаимодействие с пользователем через Telegram реализовано с помощью библиотеки aiogram. Для хранения истории диалогов и оценок пользователей используется MySQL.

В качестве модели для распознования аудиозапросов была выбрана [GigaAM RNNT 2](https://github.com/salute-developers/GigaAM), т.к. на текущий момент она является SOTA решением для задачи Automatic Speech Recognition.


## Пример запуска
Установка зависимостей
```
git clone https://github.com/salute-developers/GigaAM.git
cd GigaAM
pip install -e .
cd ..
poetry install
```
Добавление документов
```
mkdir -p ./ragtest/input
curl https://www.gutenberg.org/cache/epub/24022/pg24022.txt -o ./ragtest/input/book.txt
graphrag init --root ./ragtest
```
Далее необходимо настроить файлы .env и settings.yaml: 
* .env содержит поле GRAPHRAG_API_KEY
* setting.yaml содержит настройки пайплайна.

Если используется не модели от OpenAI или модели, имеющие токенизатор от них, необходимо убедиться что для LLM настроен правильный токенизатор. Для этого убедитесь, что указана корректная ссылка на HuggingFace модель.

К примеру вы планируете использовать deepseekv3, тогда поле в файле ```settings.yaml``` должно иметь вид: ```encoding_model: deepseek-ai/DeepSeek-V3```

Запуск создания базы знаний
```
graphrag index --root ./ragtest
```
Чтобы проверить, что система работает, предлагается использовать следующую команду
```
graphrag query \
--root ./ragtest \
--method local \
--query "Who is Scrooge and what are his main relationships?"
```

## Конфиденциальность

[Microsoft Privacy Statement](https://privacy.microsoft.com/en-us/privacystatement)
