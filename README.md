Heart Risk Prediction Project

Описание проекта
Проект посвящён задаче предсказания риска сердечного приступа на основе медицинских данных пациентов.

Используются признаки:
- возраст
- уровень холестерина
- давление
- образ жизни (курение, сон, активность)
- биохимические показатели

Цель — классифицировать пациентов на группы высокого и низкого риска.

Стек технологий
- Python
- Pandas, NumPy
- Scikit-learn
- CatBoost
- FastAPI

Этапы работы
1. Анализ данных (EDA)
2. Предобработка:
   - обработка пропусков
   - очистка признаков
3. Обучение модели CatBoost
4. Подбор оптимального порога (threshold)
5. Оценка качества модели

Метрики
- ROC-AUC: ~0.56  
- F1-score: ~0.53  

Структура проекта

heart_risk_project/
- app/ — FastAPI приложение  
- src/ — логика модели (ООП)  
- data/ — данные  
- models/ — сохранённые модели  
- predictions/ — submission.csv  
- notebooks/ — исследования  
- tests/ — тесты  
- README.md  
- requirements.txt  

Запуск проекта

Установка зависимостей:
pip install -r requirements.txt

Запуск API:
uvicorn app.main:app --reload

Использование API

POST запрос на:
http://127.0.0.1:8000/predict

Пример:
```python
import requests

files = {"file": open("data/heart_test.csv", "rb")}
response = requests.post("http://127.0.0.1:8000/predict", files=files)

print(response.json())
