#!/usr/bin/env python
# coding: utf-8

# # Evaluation of the agent (VisualAgent)
from tqdm import tqdm
from phoenix.evals import llm_classify, TOOL_CALLING_PROMPT_TEMPLATE, PromptTemplate, LiteLLMModel
from litellm import completion
from phoenix.trace import SpanEvaluations
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry import trace
import os
import json
import pandas as pd
# === Phoenix core ===
import phoenix as px
from phoenix.trace.dsl import SpanQuery

# === Evaluaciones automáticas ===
from phoenix.evals import (
    TOOL_CALLING_PROMPT_TEMPLATE,
    llm_classify,
    PromptTemplate,
)
from openinference.instrumentation import suppress_tracing
import nest_asyncio
nest_asyncio.apply()
import pprint
import warnings
warnings.filterwarnings('ignore')

os.environ['OLLAMA_API_BASE']= 'http://localhost:11434'
PROJECT_NAME = "evaluating-agent"

# === SQL Generation Evaluation ===
SQL_EVAL_GEN_PROMPT = """
SQL Evaluation Prompt:
-----------------------
You are tasked with determining if the SQL generated appropiately answers a given instruction
taking into account its generated query and response.

Data:
-----
- [Instruction]: {question}
  This section contains the specific task or problem that the sql query is intended to solve.

- [Reference Query]: {query_gen}
  This is the sql query submitted for evaluation. Analyze it in the context of the provided
  instruction.

Evaluation (the label of the dataframe):
-----------
Your response should be a single word: either "correct" or "incorrect".
You must assume that the db exists and that columns are appropiately named.
You must take into account the response as additional information to determine the correctness.

- "correct" indicates that the sql query correctly solves the instruction.
- "incorrect" indicates that the sql query correctly does not solve the instruction correctly.

Note: Your response should contain only the word "correct" or "incorrect" with no additional text
or characters.

Explanation:
------------------
Your explanation should be a single sentence that provides a brief justification for your evaluation.
It should be clear and concise, providing insight into why you believe the sql query is correct or incorrect, but don't use the same words as the evaluation.
"""

# === Data Analysis Evaluation ===
CLARITY_LLM_JUDGE_PROMPT = """
In this task, you will be presented with a query and an answer. Your objective is to evaluate the clarity 
of the answer in addressing the query. A clear response is one that is precise, coherent, and directly 
addresses the query without introducing unnecessary complexity or ambiguity. An unclear response is one 
that is vague, disorganized, or difficult to understand, even if it may be factually correct.

[BEGIN DATA]
Query: {query}
Answer: {response}
[END DATA]

Return the output in this format (First, explain your reasoning. Then, on a new line, write):

Label: clear
(or)
Label: unclear
"""

# === Visualization Evaluation ===
VIZ_QUALITY_TEMPLATE = PromptTemplate("""
Evaluate this visualization configuration:
1. Appropriateness of chart type for the data
2. Correct mapping of axes
3. Clarity of visualization goal

Goal: {input}
Data Sample: {reference_data}
Configuration: {output}

Respond with "good" or "poor" and a brief reason.
""")

tools = [
    {
        "name": "lookup_sales_data",
        "description": "Fetch historical data of sales for a product or category."
    },
    {
        "name": "analyzing_data",
        "description": "Does a statistical analysis of the data available, giving an output in form of a summary of trends/patterns found for example."
    },
    {
        "name": "create_visualization",
        "description": "Generates a visualization schema of the data processed according to the user's configuration."
    }
]

model = LiteLLMModel(model="ollama_chat/llama3.2:3B")
input_state = {"prompt": "Show me sales in Nov 2021"}


#verify traces
def decide_tool_eval(run_id):
    decide_query = (
        SpanQuery()
        .where(f"name == 'tool_choice' and span_kind == 'TOOL' and agentrun_id == '{run_id}'")
    ).select(
        question="input.value",
        tool_call="output.value",
    )
    tool_calls_df = px.Client().query_spans(decide_query, project_name=PROJECT_NAME, timeout=None)
    tool_calls_df = tool_calls_df.dropna(subset=["tool_call"])

    tool_call_eval = llm_classify(
        dataframe=tool_calls_df,
        template=TOOL_CALLING_PROMPT_TEMPLATE.template[0].template.replace(
            "{tool_definitions}", json.dumps(tools).replace("{", '"').replace("}", '"')),
        rails=['correct', 'incorrect'],
        model=model,
        concurrency=1,
    )

    tool_call_eval['score'] = tool_call_eval.apply(lambda x: 1 if x['label']=='correct' else 0, axis=1)

    px.Client().log_evaluations(
        SpanEvaluations(eval_name="Tool Calling Eval", dataframe=tool_call_eval)
    )
    return tool_call_eval



# === SQL Generation Evaluation ===
def sql_eval(run_id):   
    sql_query = (
        SpanQuery()
        .where(f"name == 'sql_query_gen' and agentrun_id == '{run_id}'")
    ).select(
        question="input.value",
        query_gen="output.value",
    )
    sql_df = px.Client().query_spans(sql_query, project_name=PROJECT_NAME, timeout=None)
    #sql_df = sql_df[sql_df["question"].str.contains("Generate an SQL query based on a prompt.", na=False)]
    if sql_df.empty:
        print("[WARNING] No SQL queries found for the given run_id.")
        return pd.DataFrame() 
    else:
        with suppress_tracing():
            sql_eval = llm_classify(
                dataframe=sql_df,
                template=SQL_EVAL_GEN_PROMPT,
                rails=["correct", "incorrect"],
                model=model,
                provide_explanation=True
            )

        sql_eval ['score'] = sql_eval.apply(lambda x: 1 if x['label']=='correct' else 0, axis=1)
        sql_eval.head()
        px.Client().log_evaluations(
            SpanEvaluations(eval_name="SQL Generation Eval", dataframe=sql_eval),
        )
        return sql_eval



# === Data Analysis Evaluation ===
def analysis_eval(run_id):
    analysis_query = (
        SpanQuery()
        .where(f"name == 'data_analysis' and agentrun_id == '{run_id}'")
    ).select(
        query="input.value",
        response="output.value",
    )
    clarity_df = px.Client().query_spans(analysis_query, project_name=PROJECT_NAME, timeout=None)
    clarity_df.head()
    with suppress_tracing():
        clarity_eval = llm_classify(
            dataframe=clarity_df,
            template=CLARITY_LLM_JUDGE_PROMPT,
            rails=["clear", "unclear"],
            model=model,
            provide_explanation=True
        )
    clarity_eval['score'] = clarity_eval.apply(lambda x: 1 if x['label']=='clear' else 0, axis=1)

    clarity_eval.head()

    px.Client().log_evaluations(
        SpanEvaluations(eval_name="Response Clarity", dataframe=clarity_eval),
    )
    return clarity_eval




# === Visualization Evaluation ===
def visualization_eval(run_id):
    viz_query = (
        SpanQuery()
        .where(f"name == 'gen_visualization' and agentrun_id == '{run_id}'")
    ).select(
        input="input.value",
        generated_code="output.value",
    )
    code_gen_df = px.Client().query_spans(viz_query, project_name=PROJECT_NAME, timeout=None)
    code_gen_df.head()

    def code_is_runnable(output:str) -> bool:
        if not output or not isinstance(output, str):
            return False  
        
        output = output.replace("```python", "").replace("```", "").strip()
        
        try:
            exec(output, {}, {})  
            return True
        except Exception:
            return False
        
    code_gen_df['label'] = code_gen_df['generated_code'].apply(code_is_runnable).map({True: "runnable", False: "not_runnable"})
    code_gen_df['score'] = code_gen_df['label'].apply(lambda x: 1 if x=='runnable' else 0)

    px.Client().log_evaluations(
        SpanEvaluations(eval_name="Runnable Code Eval", dataframe=code_gen_df),
    )
    return code_gen_df
