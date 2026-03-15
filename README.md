# Personalized Recommendation Engine

> **Database Management & ML Integration Project** | Data Systems Research

## Overview

A personalized recommendation engine for e-commerce platforms, leveraging collaborative filtering and matrix factorization techniques to deliver relevant product suggestions at scale.

## Key Highlights

- Implemented collaborative filtering and matrix factorization algorithms for e-commerce product recommendations
- Integrated ML models with database management systems for efficient data retrieval and recommendation serving
- Designed scalable data pipelines for processing user interaction data
- Evaluated recommendation quality using standard metrics (Precision@K, Recall@K, RMSE)

## Algorithms Implemented

### Collaborative Filtering
- **User-Based CF**: Finds similar users and recommends items they liked
- **Item-Based CF**: Recommends items similar to those a user has interacted with

### Matrix Factorization
- **SVD (Singular Value Decomposition)**
- **ALS (Alternating Least Squares)**
- **NMF (Non-negative Matrix Factorization)**

## Technologies

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat&logo=mysql&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

- **Languages:** Python, SQL
- **ML Libraries:** Scikit-learn, Surprise, NumPy, Pandas
- **Database:** MySQL / PostgreSQL
- **Frameworks:** FastAPI (serving), SQLAlchemy (ORM)

## Project Structure

```
personalized-recommendation-engine/
├── data/               # Dataset & preprocessing scripts
├── models/             # CF & matrix factorization models
├── api/                # FastAPI recommendation serving
├── database/           # DB schema & migration scripts
├── evaluation/         # Metrics & evaluation scripts
└── README.md
```

## License

MIT
