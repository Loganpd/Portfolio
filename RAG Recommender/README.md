# Project description
This project aims at creation and orchestration of a RAG powered movie recommender system using Python, Airflow 3.0, and Weaviate.
The process for this task is shown in the DAG Diagram image in the same directory as this file, and is also described below.

- Prepare a vector database
- Have a large set of movies alongside their descriptions
- Prepare and create the vector embeddings of the movie descriptions
- Store the vector embeddings into the vector database
- Receive the description of what kind of movie the user wants to watch and embed it to the same embedding space
- Find the closest match to the embedded query in our database of movie descriptions
- Provide the user with the title, release date, and description of the best movie found
