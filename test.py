import mlflow
import openai
import os 
import pandas as pd
import dagshub
from dotenv import load_dotenv

load_dotenv()

dagshub.init(repo_owner='akshay.satasiya', repo_name='mlflow1', mlflow=True)
mlflow.set_tracking_uri("https://github.com/UpforceTechAJS/mlflow.git")


eval_data = pd.DataFrame(
    {
        'inputs':[
            "what is MLflow?",
            "what is Spark?"
        ],

        'ground_truth':[
            "MLflow is an open-source platform designed to streamline and manage the machine learning lifecycle. It provides tools for tracking experiments, packaging code into reproducible runs, and sharing and deploying models. By offering components such as experiment tracking, model management, and project packaging, MLflow simplifies the complexities of developing and maintaining machine learning models, making it easier for data scientists and engineers to collaborate and scale their projects.",
            "Apache Spark is an open-source, distributed computing system designed for fast and efficient processing of large-scale data. It provides a unified analytics engine that supports various data processing tasks, including batch processing, real-time streaming, machine learning, and graph processing. Spark achieves high performance through in-memory computing and is widely used for big data applications due to its scalability, speed, and ease of use compared to traditional MapReduce frameworks."
        ]
    }
)

mlflow.set_experiment("LLM_Evaluation")
with mlflow.start_run() as run:
    system_prompt = "Answer the following question in two sentences"

    # wrap gpt model as mlflow model  
    logged_model_info = mlflow.openai.log_model(
        model = 'gpt-3.5-turbo-0125',
        task = openai.chat.completions,
        artifact_path = "model",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "{question}"},
        ],
    )

results = mlflow.evaluate(
    logged_model_info.model_uri,
    eval_data,
    targets='ground_truth',
    model_type='question-answering',
    extra_metrics=[mlflow.metrics.toxicity(), mlflow.metrics.latency(),mlflow.metrics.genai.answer_similarity()]
)
print(f"See aggregated evaluation results below: \n{results.metrics}")

# Evaluation result for each data record is available in `results.tables`.
eval_table = results.tables["eval_results_table"]
df=pd.DataFrame(eval_table)
df.to_csv('eval.csv')
print(f"See evaluation table below: \n{eval_table}")