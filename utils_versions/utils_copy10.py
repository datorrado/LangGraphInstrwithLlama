from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import re
from operator import add
from codecarbon import EmissionsTracker
from pydantic import BaseModel, Field
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.trace import Status, StatusCode
from phoenix.otel import register
from functools import partial
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from opentelemetry.trace import StatusCode
import pandas as pd
import json
import duckdb
import pandas as pd
from datetime import datetime
import subprocess
import logging
##from pydantic import BaseModel, Field
from IPython.display import Markdown
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates
import psutil, threading, time
from statistics import median
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetCount
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
#from helper import  get_phoenix_endpoint
from typing import Annotated
from typing_extensions import TypedDict, Optional, NotRequired, Literal, Annotated
#from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
import pprint
import difflib
import operator
import os
from opentelemetry.trace import StatusCode
import uuid

from prueba import decide_tool_eval, analysis_eval, visualization_eval, sql_eval

from concurrent.futures import ThreadPoolExecutor, as_completed
import langgraph
import langgraph.version
print(langgraph.version)



PHOENIX_API_KEY = "8ec6bad84d31d641610:789d473"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"



PROJECT_NAME="evaluating-agent"

# configure the Phoenix tracer
tracer_provider = register(
  project_name=PROJECT_NAME, # Default is 'default' 
  endpoint = "https://app.phoenix.arize.com/v1/traces", # Default is 'https://app.phoenix.arize.com'
)
LangChainInstrumentor(tracer_provider=tracer_provider).instrument(skip_dep_check=True)


tracer = tracer_provider.get_tracer(__name__)


llm = ChatOllama(model="llama3.2:3b", temperature=0.1, streaming=True)

TRANSACTION_DATA_FILE_PATH =  os.path.join(os.path.dirname(__file__), 'data', 'Store_Sales_Price_Elasticity_Promotions_Data.parquet')

codecarbon_logger = logging.getLogger("codecarbon")

codecarbon_logger.setLevel(logging.NOTSET)


for handler in codecarbon_logger.handlers[:]:
    codecarbon_logger.removeHandler(handler)

null_handler = logging.NullHandler()
codecarbon_logger.addHandler(null_handler)


# Define the state of the agent's main graph
class State (TypedDict):
    prompt: str
    data: str
    sql_queries: str
    analyze_data : str
    emphasis: str
    answer: list[str]
    visualization_goal: Optional[str]  # Para almacenar el objetivo de visualización
    chart_config: Optional[dict]  # Para almacenar la configuración del gráfico
    tool_choice: NotRequired[str]
    used_tools: NotRequired[list[str]]
    temperature: float
    id: str # Para almacenar el UUID de la ejecución
    table_name: str
    columns: list[str]
    energy_query: NotRequired[float]
    energy_analysis: NotRequired[float]
    query_id: NotRequired[str]
    analysis_id: NotRequired[str]
    ids_lookup_sales_data: NotRequired[list[str]]
    ids_analyzing_data: NotRequired[list[str]]
    ids_create_visualization: NotRequired[str]
    ids_decide_tool: NotRequired[list[str]]
    energy_lookup_sales_data: NotRequired[list[float]]
    energy_analyzing_data: NotRequired[list[float]]
    energy_create_visualization: NotRequired[float]
    energy_decide_tool: NotRequired[list[float]]
    cpu_utilization_lookup_sales_data: NotRequired[float]
    cpu_utilization_analyzing_data: NotRequired[float]
    cpu_utilization_create_visualization: NotRequired[float]
    cpu_utilization_decide_tool: NotRequired[list[float]]
    gpu_utilization_lookup_sales_data: NotRequired[float]
    gpu_utilization_analyzing_data: NotRequired[float]
    gpu_utilization_create_visualization: NotRequired[float]
    gpu_utilization_decide_tool: NotRequired[list[float]]
    cpu_query: NotRequired[float]
    cpu_analysis: NotRequired[float]
    gpu_query: NotRequired[float]
    gpu_analysis: NotRequired[float]



#Define the state for the subgraph used for parallelization 

class CPU_GPU_Tracker:
    def __init__(self, interval=1):
        self.cpu_usage = []
        self.gpu_usage = []
        self.cpu_median = None
        self.gpu_median = None
        self.cpu_mean = None
        self.gpu_mean = None
        self.running = False
        self.interval = interval

    def update_median(self):
        """Updates the median CPU and GPU utilization values."""
        self.cpu_median = median(self.cpu_usage) if self.cpu_usage else None
        self.gpu_median = median(self.gpu_usage) if self.gpu_usage else None

    def mean(self):
        """Calculates the mean CPU and GPU utilization."""
        self.cpu_mean = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else None
        self.gpu_mean = sum(self.gpu_usage) / len(self.gpu_usage) if self.gpu_usage else None

    def start(self):
        """Starts tracking CPU and GPU utilization."""
        self.running = True
        self._track()

    def stop(self):
        """Stops tracking and returns collected data."""
        self.running = False
        self.update_median()
        self.mean()
        return {
            "cpu_usage": self.cpu_usage,
            "gpu_usage": self.gpu_usage,
            "cpu_median": self.cpu_median,
            "gpu_median": self.gpu_median,
            "cpu_mean": self.cpu_mean,
            "gpu_mean": self.gpu_mean,
        }

    def _track(self):
        """Internal method to track CPU and GPU utilization in a separate thread."""
        def track():
            while self.running:
                # CPU tracking 
                try:
                    cpu_utilization = psutil.cpu_percent(interval=self.interval)
                    self.cpu_usage.append(cpu_utilization)
                except ValueError:
                    print("Error parsing CPU utilization")

                # GPU tracking using nvidia-smi
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    gpu_utilization = [int(util) for util in result.stdout.strip().split("\n")]
                    avg_gpu_utilization = sum(gpu_utilization) / len(gpu_utilization)
                    self.gpu_usage.append(avg_gpu_utilization)
                else:
                    print("Error running nvidia-smi")

                #time.sleep(self.interval)  # Sampling interval

        threading.Thread(target=track, daemon=True).start()


SQL_Generation_Prompt = """ 
"Generate an SQL query based on the prompt. Please just reply with the SQL query and NO MORE, just the query. Really there is no need to create any comment besides the query, that's the only important thing. The prompt is : {prompt}" \
"The available columns are: {columns}. " \
"The table name is: {table_name}. " \
"⚠️ If you need to use a DATE column with LIKE or pattern matching, first CAST it to VARCHAR like this: CAST(date_column AS VARCHAR) LIKE '%2021-11%' " \
"Return only the SQL query, with no explanations or markdown formatting." \
"" \
"Important: If you filter or compare date columns, always cast them to string using CAST(date_column AS VARCHAR)."\
If your query uses GROUP BY, every column in SELECT must either be in GROUP BY or be wrapped in an aggregate function like SUM(), COUNT(), MAX(), etc.
DO NOT use any column name (like "Store_Number") in the FROM clause. Only use the table name: {table_name}
All FROM or JOIN clauses MUST say: FROM {table_name}

NEVER write CAST(... LIKE ...) inside SELECT. It must be part of a WHERE clause.

If your query uses GROUP BY, every column in SELECT must either:
- Appear in the GROUP BY clause, OR
- Be wrapped in an aggregate function like SUM(), COUNT(), MAX(), AVG(), etc.

WARNING: Do NOT select columns like Store_Number, Product_Class_Code, etc. unless they are in GROUP BY or inside an aggregation.

If your query uses GROUP BY, every column in SELECT must either:
- Appear in the GROUP BY clause, OR
- Be wrapped in an aggregate function like SUM(), COUNT(), MAX(), AVG(), etc.

If you want to keep a non-aggregated column for display (and its exact value is not important), you may use ANY_VALUE(column) — but ONLY in the SELECT clause.

NEVER use ANY_VALUE(...) inside GROUP BY, ORDER BY, or WHERE clauses.
NEVER nest ANY_VALUE inside another aggregation (e.g. SUM(ANY_VALUE(...)) is invalid).
ONLY use ANY_VALUE in the SELECT clause, and only for columns not in GROUP BY.
Also, NEVER use column names as string literals (no quotes).


"""

#Tool for sql query generation
def generate_sql_query (state:State): 
    # Formateamos el prompt para generar la consulta SQL
    state["query_id"] = uuid.uuid4().hex[:8]
    utils = CPU_GPU_Tracker(interval=2.0)
    utils.start()
    tracker = EmissionsTracker(project_name="lookup_sales_data",experiment_id=state["query_id"], measure_power_secs=1, log_level="critical", output_file=f"emissions_{state['id']}_{state['query_id']}.csv", output_dir="10node")
    tracker.start()
    temperature = state.get("temperature", 0.1)
    formatted_prompt = SQL_Generation_Prompt.format(prompt=state["prompt"], columns=state["columns"], table_name=state["table_name"])
    localLLM=ChatOllama(model="llama3.2:3b", temperature=temperature, streaming=True)
    
    # Aquí invocamos LLaMA para generar la consulta SQL
    sql_query = localLLM.invoke(formatted_prompt)
    # Retornamos el nuevo estado con la respuesta
    try:
        sql_query = sql_query.content
        sql_query = sql_query.strip()
        sql_query = sql_query.replace("```sql", "").replace("```", "").replace("`", "")
        with tracer.start_as_current_span(
            "sql_query_gen", 
            openinference_span_kind="tool",
        ) as span:
            span.set_input(state["prompt"])
            span.set_attribute("agentrun_id", state["id"])
            span.set_attribute("temperature", temperature)
            span.set_output(sql_query)
            span.set_status(StatusCode.OK)
        emissions = tracker.stop()
        utils.stop()

        return {"sql_queries": sql_query, "energy_query": emissions, "query_id": state["query_id"], "cpu_query": utils.cpu_mean, "gpu_query": utils.gpu_mean}
    except Exception as e:
        return {**state, "error": f"Error accessing data: {str(e)}"}

@tracer.tool()
def parallel_sql_gen(state:State):
    temperatures = [0.1]*10

    substates = [{**state, "temperature": temp, "query_id":str(uuid.uuid4().hex[:8])} for temp in temperatures]
    
    results = []
    subgraph = StateGraph(State)
    subgraph.add_node("generate_sql_query", generate_sql_query)
    subgraph.set_entry_point("generate_sql_query")
    subgraph.set_finish_point("generate_sql_query")
    subgraph = subgraph.compile()
    results = list(subgraph.batch_as_completed(inputs=substates))

    return results

def cast_date_columns(query: str, date_columns: list) -> str:
    for col in date_columns:
        # Soporta comillas, espacios y/o calificaciones tipo tabla.col
        pattern = rf"(?<!CAST\()(?P<full>([\w\.]*{col}|\"{col}\")\s*~~)"
        query = re.sub(pattern, rf"CAST(\g<full> AS VARCHAR) ~~", query)
        
        pattern_like = rf"(?<!CAST\()(?P<full>([\w\.]*{col}|\"{col}\")\s*LIKE)"
        query = re.sub(pattern_like, rf"CAST(\g<full> AS VARCHAR) LIKE", query)

        # Comparaciones: =, >, <
        for op in ["=", ">", "<"]:
            pattern_cmp = rf"(?<!CAST\()(?P<full>([\w\.]*{col}|\"{col}\")\s*\{op})"
            query = re.sub(pattern_cmp, rf"CAST(\g<full> AS VARCHAR) {op}", query)

    return query

@tracer.tool()
def lookup_sales_data(state:State):
    """Implementation of sales data lookup from parquet file using SQL"""
    try:
        # define the table name
        table_name = "sales"
        state["table_name"] = table_name
        # step 1: read the parquet file into a DuckDB table
        df = pd.read_parquet(TRANSACTION_DATA_FILE_PATH)
        #print("Este es el dataframe: "+str(df.head()))
        duckdb.sql("DROP TABLE IF EXISTS sales")
        duckdb.sql(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df")
        # step 2: generate the SQL code 
        columns = list(df.columns)
        state["columns"] = columns
        #print("Estas son las columnas: "+str(columns))
        date_columns = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
        resultsdata = []
        results = list(parallel_sql_gen(state))
        #pprint.pprint("Estos son los resultados de la paralelizacion: "+str(results))
        sql_list = [r[1]["sql_queries"] for r in results]
        energies = [r[1]["energy_query"] for r in results]
        ids = [r[1]["query_id"] for r in results]
        cpu_queries = [r[1]["cpu_query"] for r in results]
        gpu_queries = [r[1]["gpu_query"] for r in results]
        # Search if there is a LIKE in da query and we cast it
        for sql_query in sql_list:  
            sql_query = cast_date_columns(sql_query, date_columns)
            if table_name not in sql_query:
                continue   
            try:   
                result = duckdb.sql(sql_query).df()
                resultsdata.append(result)   
            except Exception:
                continue
        state["energy_lookup_sales_data"] = energies
        #resultsdata = [df.head(1000) for df in results]
        if len(resultsdata) > 1:
            base_df = resultsdata[0]
            for other in resultsdata[1:]:
                common_cols = base_df.columns.intersection(other.columns)
                if not common_cols.empty:
                    base_df = pd.merge(base_df, other, how="inner", on=list(common_cols))
                else:
                    print("No hay columnas en común para hacer merge entre los DataFrames.")
            
            final_result = base_df
        elif len(resultsdata) == 1:
            final_result = resultsdata[0]
        else:
            final_result = pd.DataFrame()    
        
        
        final_result = result.to_string()

        with tracer.start_as_current_span(
            "sql_query_exec",
            openinference_span_kind="tool", 
        ) as span:
            span.set_input(state["prompt"])
            span.set_attribute("agentrun_id", state["id"])
            # step 3: execute the SQL query
            span.set_output(final_result)
            span.set_status(StatusCode.OK)

        #state["energy_lookup_sales_data"] = 
        #print("Resultado de lookup_sales_data:", result) 
        return {**state, "data": final_result, "answer": state.get("answer", []) + ["The query to create the dataframe is the following: "+sql_query+"\n"], "used_tools": state.get("used_tools", []) + ["lookup_sales_data"], "id": state.get("id"), "ids_lookup_sales_data":ids, "cpu_utilization_lookup_sales_data": cpu_queries, "gpu_utilization_lookup_sales_data": gpu_queries}
    except Exception as e:
        return {**state, "error": f"Error accessing data: {str(e)}"}
    
#Define the functions for the data analysis tool
DATA_ANALYSIS_PROMPT = """
Analyze the following data: {data}
Your job is to answer the following question, that has an specific emphasis besides the original prompt passed by the user in this format "prompt-emphasis": {emphasis}

Please remember to just restrain yourself to explaining patterns, trends, insights or summaries of the data. There is NO need to create any code or any other thing, just the analysis of the data. 

If you feel the need to say something about code, just stick to explain the architecture of how to build a code (mentioning libraries, functions, etc.) but not the code itself.
"""
@tracer.tool()
def analyzing_data(state: State) -> State:
    state["analysis_id"] = str(uuid.uuid4().hex[:8])
    tracker = EmissionsTracker(project_name="analyzing_data", experiment_id=state["analysis_id"], measure_power_secs=1, log_level="critical", output_file=f"emissions_{state['id']}_{state['analysis_id']}.csv", output_dir="10node")
    utils = CPU_GPU_Tracker()
    utils.start()
    tracker.start()
    try:    
        #pprint.pprint("llegue al analisis y estas son sus llaves "+str(state.keys()))
        formatted_prompt = DATA_ANALYSIS_PROMPT.format(data=state["data"], emphasis=state["emphasis"])
        analysis_result = llm.invoke(formatted_prompt)
        analysis_result = analysis_result.content
        with tracer.start_as_current_span(
            "data_analysis", 
            openinference_span_kind="tool",
        ) as span:
            span.set_input(state["prompt"])
            span.set_attribute("agentrun_id", state["id"])
            analysis_result = analysis_result
            span.set_output(value=str(analysis_result))
            span.set_status(StatusCode.OK)
        utils.stop()
        return {**state, "analyze_data": analysis_result, "answer": state.get("answer", []) + [analysis_result],         "visualization_goal": state.get("visualization_goal"),
        "chart_config": state.get("chart_config"), "used_tools": state.get("used_tools", []) + ["analyzing_data"], "id": state.get("id"), "energy_analysis": tracker.stop(), "analysis_id": state.get("analysis_id"), "cpu_analysis": utils.cpu_mean, "gpu_analysis": utils.gpu_mean
        }
    except Exception as e:
        return {**state, "error": f"Error accessing data: {str(e)}"}

def split_prompt(emphasis: str) -> list[str]:
    base_patterns = [
        "Revenue trends",
        "Product category performance",
        "Promotional impact",
        "Store-level comparisons",
        "Time-based patterns",
        "Customer preferences",
        "High-performing SKUs",
        "Outlier detection",
        "Discount effectiveness",
        "Sales seasonality"
    ]
    variations = [
        "{} (statistical summary)",
        "{} using time series analysis",
        "{} with visual breakdowns",
        "{} focusing on November sales",
        "{} using moving averages",
        "{} across all stores",
        "{} per product class",
        "{} by weekday vs weekend",
        "{} with anomaly detection",
        "{} highlighting top performers",
    ]

    prompts = []
    for base in base_patterns:
        for variation in variations:
            prompts.append(f"{emphasis} - {variation.format(base)}")

    return prompts[:10]



@tracer.tool()
def parallel_analyze_data(state: State) -> State:
    sub_prompts = split_prompt(state["prompt"])

    # Construimos estados individuales para cada subanalisis
    sub_states = [
        {"prompt": state.get("prompt"), "data": state["data"], "tool_choice": state.get("tool_choice"), "answer": state.get("answer"), "analyze_data": state.get("analyze_data"), "used_tools": state.get("used_tools", []), "emphasis": sub_prompt, "id": state.get("id"), "analysis_id":str(uuid.uuid4().hex[:8])}
        for sub_prompt in sub_prompts
    ]
    #print(str(sub_states))
    # Creamos subgrafo
    subgraph = StateGraph(State)
    subgraph.add_node("analyzing_data", analyzing_data)
    subgraph.set_entry_point("analyzing_data")
    subgraph.set_finish_point("analyzing_data")

    # Ejecutamos en paralelo
    compiled_subgraph = subgraph.compile()
    results = list(compiled_subgraph.batch_as_completed(inputs=sub_states))
    #print("Resultados de los análisis paralelos:" +str(results)) 
    energies = [r[1]["energy_analysis"] for r in results]
    ids = [r[1]["analysis_id"] for r in results]
    # Reducimos
    analysis_outputs = [r[1]['analyze_data'] for r in results]
    final_summary = fuse_analysis_results(analysis_outputs)
    state["energy_analyzing_data"] = energies
    cpus = [r[1]["cpu_analysis"] for r in results]
    gpus = [r[1]["gpu_analysis"] for r in results]
    return {
        **state,
        "analyze_data": final_summary,
        "answer": state.get("answer", []) + ["The analisis extracted from the data: "+final_summary+"\n"],
        "used_tools": state.get("used_tools", []) + ["analyzing_data"],
        "ids_analyzing_data": ids, "cpu_utilization_analyzing_data": cpus, "gpu_utilization_analyzing_data": gpus,
    }

def fuse_analysis_results(results: list) -> str:
    # Preparamos un prompt corto para que el LLM resuma las respuestas de análisis
    fusion_prompt = (
        "Given the following analysis outputs, produce a concise summary that captures the key insights from the list:\n\n"
        +str(results)+ ", this by reading carefully and extracting the most important information from each of them. If something is repeated, please just keep one of them, or try to see if any subtle difference is there to summarize it into a more compact idea."
    )
    summary_resp = llm.invoke(fusion_prompt)
    summary_resp = summary_resp.content.strip().lower()
    return summary_resp

CHART_CONFIGURATION_PROMPT = """
Based on the provided data and goal, define a chart configuration using the format below:

Data:
{data}

Goal:
{visualization_goal}

Respond ONLY with the following format (no explanations, no markdown):

chart_type: <chart type>
x_axis: <x-axis column>
y_axis: <y-axis column>
title: <chart title>
"""

class VisualizationConfig(BaseModel):
    chart_type: str = Field(..., description="Type of chart to generate")
    x_axis: str = Field(..., description="Name of the x-axis column")
    y_axis: str = Field(..., description="Name of the y-axis column")
    title: str = Field(..., description="Title of the chart")
@tracer.chain()
def extract_chart_config(state:State) -> State:
    # Verificar si ya tenemos datos para visualizar
    if "data" not in state or state["data"] is None:
        return {**state, "chart_config": None}

    # Verificar si ya tenemos un objetivo de visualización
    visualization_goal = state.get("visualization_goal", state["prompt"])

    formatted_prompt = CHART_CONFIGURATION_PROMPT.format(
        data=state["data"],
        visualization_goal=visualization_goal
    )

    response = llm.invoke(formatted_prompt)

    try:
        raw = response.content.strip()
        #print("🧠 LLM raw response:\n", raw)

        # Parse estilo "key: value"
        config = {}
        for line in raw.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                config[key.strip().lower()] = value.strip()

        # Validar que existan todas las claves necesarias
        required_keys = {"chart_type", "x_axis", "y_axis", "title"}
        if not required_keys.issubset(config.keys()):
            raise ValueError(f"Missing keys in chart config: {config.keys()}")

        # Construir configuración del gráfico
        chart_config = {
            "chart_type": config["chart_type"],
            "x_axis": config["x_axis"],
            "y_axis": config["y_axis"],
            "title": config["title"],
            "data": state["data"]
        }
        #print("✅ Chart config generado:", chart_config)

        return {
            **state,
            "visualization_goal": visualization_goal,
            "chart_config": chart_config,
            "analyze_data": state.get("analyze_data"),
            "used_tools": state.get("used_tools", [])
        }

    except Exception as e:
        print(f"⚠️ Error extrayendo chart_config: {e}")
        chart_config = {
            "chart_type": "line",
            "x_axis": "date",
            "y_axis": "value",
            "title": visualization_goal,
            "data": state["data"]
        }

        return {
            **state,
            "visualization_goal": visualization_goal,
            "chart_config": chart_config,
            "analyze_data": state.get("analyze_data"),
            "used_tools": state.get("used_tools", [])
        }

CREATE_CHART_PROMPT = """
Write python code to create a chart based on the following configuration.
Only return the code, no other text.
config: {config}
"""
@tracer.chain()
def create_chart(state: State):
    try:
        formatted_prompt = CREATE_CHART_PROMPT.format(config=state["chart_config"])
        response = llm.invoke(formatted_prompt)
        code = response.content
        code = code.replace("```python", "").replace("```", "")
        #print("Este es el estado en la creacion del chart: "+str(state.keys())) 
        # Clean the code and remove any unwanted characters
        return code
    except Exception as e:
        print("Error creating chart: ", e)
        return {**state, "error": f"Error accessing data: {str(e)}"}
@tracer.tool()
def create_visualization(state: State) -> State:
    state["ids_create_visualization"] = str(uuid.uuid4().hex[:8])
    utils = CPU_GPU_Tracker( )
    utils.start()
    tracker = EmissionsTracker(project_name="create_visualization",experiment_id=state["ids_create_visualization"], measure_power_secs=1, log_level="critical", output_file=f"emissions_{state['id']}_{state['ids_create_visualization']}.csv", output_dir="10node")
    tracker.start()
    try:
        config = extract_chart_config(state)
        code = create_chart(config)
        with tracer.start_as_current_span(
            "gen_visualization", 
            openinference_span_kind="tool",
        ) as span:
            span.set_input(state["prompt"])
            span.set_attribute("chart_config", (config["chart_config"]))
            span.set_attribute("agentrun_id", state["id"])
            span.set_attribute("visualization_goal", state["visualization_goal"])
            span.set_output(code)
            span.set_status(StatusCode.OK)
            
        emissions = tracker.stop()
        state["energy_create_visualization"] = emissions
        utils.stop()

        return {**state, "config": config["chart_config"], 
            "visualization_goal": config.get("visualization_goal"),
            "chart_config": config.get("chart_config"), 
            "analyze_data": state.get("analyze_data"), "answer": state.get("answer", []) + ["This is the code to visualize: "+code], "used_tools": state.get("used_tools", []) + ["create_visualization"], "id": state.get("id"), "energy_create_visualization": emissions, "ids_create_visualization": state.get("ids_create_visualization"), "cpu_utilization_create_visualization": utils.cpu_mean, "gpu_utilization_create_visualization": utils.gpu_mean}
    except Exception as e:
        return {**state, "error": f"Error accessing data: {str(e)}"}


@tracer.tool()
def decide_tool(state: State, llm) -> State:
    if "energy_decide_tool" not in state:
        state["energy_decide_tool"] = []

    if "ids_decide_tool" not in state:
        state["ids_decide_tool"] = []

    if "cpu_utilization_decide_tool" not in state:
        state["cpu_utilization_decide_tool"] = []
    if "gpu_utilization_decide_tool" not in state:
        state["gpu_utilization_decide_tool"] = []
    
    utils = CPU_GPU_Tracker(interval=0.5 )
    tool_id = str(uuid.uuid4().hex[:8])
    tracker = EmissionsTracker(project_name="decide_tool", experiment_id=tool_id, measure_power_secs=1, log_level="critical",output_file=f"emissions_{state['id']}_{tool_id}.csv", output_dir="10node")
    tracker.start()
    used_tools= state.get("used_tools", [])
    tools_description = """You have access to the following tools:
    - lookup_sales_data: Retrieve raw sales data. (Must be run first)
    - analyzing_data: Analyze the sales data to extract trends and insights. (Run only after lookup_sales_data)
    - create_visualization: Create a chart or graph based on the data and its analysis. (Run only after analyzing_data)
    - end: Conclude the process if the user's request is fully satisfied.
    """

    decision_prompt = f"""
    Current user request: {state['prompt']}
    Current state details:
    - Answer so far: {state.get('answer', [])}
    - Tools already used: {state.get('used_tools', [])}
    - Data available: {"yes" if state.get("data") else "no"}

    You are a decision-making agent whose job is to determine the next step, choosing from these tools: {tools_description}. In a fixed workflow to fully answer a user's request. The workflow for a typical sales query that requires a full answer is strictly ordered as follows:

    1. lookup_sales_data: Retrieve the raw sales data from a Parquet file using an SQL query. This step must be performed first.
    2. analyzing_data: Analyze the retrieved sales data to extract patterns, trends, insights, or summaries. This step must be performed after the data has been retrieved.
    3. create_visualization: Create a chart, graph, or visual representation based on the sales data and its analysis. This step must be performed after the analysis is done.
    4. end: Conclude the process when the user's request is completely satisfied and no further action is needed.
    
    Based on the current state and the user prompt, decide which tool to use next. Choose just between the tools. In this case, please just minimize the answer to the name of the tool you choose. 
    Besides this, do not use any tool that is already in :{used_tools}.

    To provide you a better understanding for this, the functions should have a number of hierarchy and order. So, lookup_sales_data [1], analyzing_data [2], create_visualization [3], end [4]. More specifically, this hierarchy needs to be respected [1] should never appear after [2], [3] or [4], neither should [1] appear after [1] was used at least once before, a flow [1], [2] ... [1], or [1], [2], [2] should never happen for example. [2] should never appear after [3] or [4]. [3] should never appear after [4]. And the only one that can be used at any time is "end" or [4], also know that's better to end than to have a repeated tool. 

    A more visual representation of the workflow is as follows:
    Examples of a flow: lookup_sales_data -> analyzing_data -> create_visualization -> end
    Examples of a flow: lookup_sales_data -> analyzing_data -> end
    Examples of a flow: lookup_sales_data -> create_visualization -> end
    Examples of a flow: lookup_sales_data -> end


    WHAT NOT TO DO? 
    What is NOT an example of a flow: analyzing_data -> create_visualization -> end
    What is NOT an example of a flow: create_visualization -> end
    What is NOT an example of a flow: end -> end
    What is NOT an example of a flow: end -> create_visualization or lookup_sales_data or analyzing_data
    What is NOT an example of a flow: lookup_sales_data -> lookup_sales_data or analyzing_data or create_visualization
    What is NOT an example of a flow: lookup_sales_data -> analyzing_data -> lookup_sales_data ....
    ---
    Guidelines:
    - If there is no data available, you must choose "lookup_sales_data".
    - If data is available but no analysis has been performed yet, and the user's request includes terms like "trend", "insight", "analysis", or "summary", then choose "analyzing_data".
    - If both data and analysis are available and the request explicitly asks for a visualization (e.g., "create a chart", "plot the data", "visualize"), then choose "create_visualization".
    - Only choose "end" if the complete workflow has been executed in order.
    - DO NOT select any tool out of this fixed order.





    Answer:
    """

    # Invocamos al LLM con el prompt simplificado
    response = llm.invoke(decision_prompt)
    tool_choice = response.content.strip().lower()
    print("Esta es la respuesta del LLM: "+str(tool_choice))

    # Forzamos que si el LLM responde algo inválido, se use "end" como fallback
    valid_tools = ["lookup_sales_data","analyzing_data", "create_visualization", "end"]
    matched_tool = difflib.get_close_matches(tool_choice, valid_tools, n=1, cutoff=0.6)
    print(str(matched_tool))

    matched_tool = matched_tool[0] if matched_tool else "end"

    if matched_tool in used_tools:
        # Si la herramienta ya fue usada, forzamos a "end"
        matched_tool = "end"


    # Actualización de trazabilidad
    with tracer.start_as_current_span("tool_choice", openinference_span_kind="tool") as span:
        span.set_attributes({
            "prompt": state["prompt"],
            "agentrun_id": state["id"],
            "tool_choice": matched_tool,
            "llm_output": response.content,
            "valid_tools": json.dumps(valid_tools),
            "answer_context": json.dumps(state.get("answer", [])),
            "used_tools": json.dumps(state.get("used_tools", [])),
        })
        span.set_input(state["prompt"])
        span.set_output(matched_tool)

    #print(f"Elección de herramienta: {matched_tool}")
    state["ids_decide_tool"].append(tool_id)
    emissions = tracker.stop()
    utils.stop()
    state["energy_decide_tool"].append(emissions)
    state["cpu_utilization_decide_tool"].append(utils.cpu_mean)
    state["gpu_utilization_decide_tool"].append(utils.gpu_mean)
  
    # Actualizamos el estado; registramos la herramienta elegida
    return {**state,
            "prompt": state["prompt"],
            "data": state.get("data"),
            "analyze_data": state.get("analyze_data"),
            "answer": state.get("answer", []),
            "visualization_goal": state.get("visualization_goal"),
            "chart_config": state.get("chart_config"),
            "tool_choice": matched_tool,
            "used_tools": state.get("used_tools", []) + [matched_tool], 
            "id": state.get("id")}

def route_to_tool(state: State):
    tool_choice = state.get("tool_choice", "lookup_sales_data")
    valid_tools = ["lookup_sales_data", "analyzing_data", "create_visualization", "end"]
    # Check if the tool choice is valid
    return tool_choice if tool_choice in valid_tools else "end"


    
    #Define el siguinete nodo basado en la eleccion de decice_tool

#function to fuse the analysis results of the parallelization




graph = StateGraph(State)

decide_tool_with_llm = partial(decide_tool, llm=llm)
graph.add_node("decide_tool", decide_tool_with_llm)
graph.add_node("lookup_sales_data", lookup_sales_data) 
graph.add_node("analyzing_data", parallel_analyze_data)   
graph.add_node("create_visualization", create_visualization)
graph.set_entry_point("decide_tool")

graph.add_conditional_edges(
    "decide_tool",
    route_to_tool,
    {
        "lookup_sales_data": "lookup_sales_data", #dictionaries are needed as the conditional must have the name of the return value of the function which routes to the next node
        "analyzing_data": "analyzing_data",
        "create_visualization": "create_visualization",
        "end": END,
    }
)

graph.add_edge("lookup_sales_data", "decide_tool")
graph.add_edge("analyzing_data", "decide_tool")
graph.add_edge("create_visualization", "decide_tool")

graph = graph.compile()


from pathlib import Path
import pandas as pd
from datetime import datetime

def log_evaluation_to_csv(
    eval_df: pd.DataFrame,
    tool_name: str,
    id: str,
    file_path: str = "tool_evaluations_10.csv",
    energy=None,
    id_tool=None, # puede ser str o list[str]
    cpu_utilization=None,
    gpu_utilization=None,
):
    if isinstance(id_tool, list) and isinstance(cpu_utilization, list) and isinstance(gpu_utilization, list):
        min_len = min(len(eval_df), len(id_tool))
        for i, (index, _) in enumerate(eval_df.iterrows()):
            if i >= min_len:
                break
            eval_df.at[index, "tool_name"] = tool_name
            eval_df.at[index, "id"] = id
            eval_df.at[index, "id_tool"] = id_tool[i]
            eval_df.at[index, "timestamp"] = datetime.now().isoformat()
            eval_df.at[index, "cpu_utilization"] = cpu_utilization[i] if i < len(cpu_utilization) else None
            eval_df.at[index, "gpu_utilization"] = gpu_utilization[i] if i < len(gpu_utilization) else None
    else:
        eval_df["tool_name"] = tool_name
        eval_df["id"] = id
        eval_df["id_tool"] = id_tool
        eval_df["timestamp"] = datetime.now().isoformat()
        eval_df["cpu_utilization"] = cpu_utilization
        eval_df["gpu_utilization"] = gpu_utilization

    # === Añadir energy_kwh ===
    if isinstance(energy, list) and len(energy) == len(eval_df):
        eval_df["total_energy"] = energy
        print("✅ Energy values added to DataFrame from list")
    elif isinstance(energy, float):
        eval_df["total_energy"] = energy
    else:
        eval_df["total_energy"] = None

    # === Inicializar columnas energéticas ===
    for col in ["cpu_energy", "gpu_energy", "ram_energy", "emissions_rate"]:
        eval_df[col] = None

    # === Caso múltiple: lista de ids ===
    
    if isinstance(id_tool, list) and len(id_tool) == len(eval_df):
        for i, tool_id in enumerate(id_tool):
            emissions_file = Path("10node") / f"emissions_{id}_{tool_id}.csv"
            if emissions_file.exists():
                try:
                    df = pd.read_csv(emissions_file)
                    match = df[df["experiment_id"].astype(str).str.strip().str.lower() == str(tool_id).strip().lower()]
                    if not match.empty:
                        row = match.iloc[-1]
                        mask = eval_df["id_tool"] == tool_id
                        eval_df.loc[mask, "cpu_energy"] = row.get("cpu_energy")
                        eval_df.loc[mask, "gpu_energy"] = row.get("gpu_energy")
                        eval_df.loc[mask, "ram_energy"] = row.get("ram_energy")
                        eval_df.loc[mask, "emissions_rate"] = row.get("emissions_rate")
                        eval_df.loc[mask, "execution_time"] = row.get("duration")
                    else:
                        print(f" No match for experiment_id = {tool_id} in {emissions_file.name}")
                except Exception as e:
                    print(f"Error leyendo emisiones para {tool_id}: {e}")
            else:
                print(f" Archivo no encontrado: emissions_{tool_id}.csv")

    # === Caso único: solo un id_tool ===
    elif isinstance(id_tool, str):
        emissions_file = Path("10node") / f"emissions_{id}_{id_tool}.csv"
        if emissions_file.exists():
            try:
                df = pd.read_csv(emissions_file)
                match = df[df["experiment_id"].astype(str).str.strip().str.lower() == str(id_tool).strip().lower()]
                if not match.empty:
                    row = match.iloc[-1]
                    mask = eval_df["id_tool"] == id_tool
                    eval_df.loc[mask, "cpu_energy"] = row.get("cpu_energy")
                    eval_df.loc[mask, "gpu_energy"] = row.get("gpu_energy")
                    eval_df.loc[mask, "ram_energy"] = row.get("ram_energy")
                    eval_df.loc[mask, "emissions_rate"] = row.get("emissions_rate")
                    eval_df.loc[mask, "execution_time"] = row.get("duration")
                else:
                    print(f"⚠️ No match for experiment_id = {id_tool} in {emissions_file.name}")
            except Exception as e:
                print(f"⚠️ Error leyendo emisiones para {id_tool}: {e}")
        else:
            print(f"⚠️ Archivo no encontrado: emissions_{id_tool}.csv")

    # === Orden final de columnas ===
    cols_order = [
        "tool_name", "id", "id_tool","timestamp", "execution_time", "score", "label", "explanation",
        "total_energy", "cpu_energy", "gpu_energy", "ram_energy", "emissions_rate",
        "cpu_utilization", "gpu_utilization"
    ]
    for col in cols_order:
        if col not in eval_df.columns:
            eval_df[col] = None
    eval_df = eval_df[cols_order]

    # === Guardar en CSV ===
    try:
        existing = pd.read_csv(file_path)
        combined = pd.concat([existing, eval_df], ignore_index=True)
    except FileNotFoundError:
        combined = eval_df

    combined.to_csv(file_path, index=False)
    print(f" Evaluación guardada en {file_path}")


def run_graph_with_tracing(input_state):
    print("[LangGraph] Starting LangGraph execution with tracing")

    with tracer.start_as_current_span("AgentRun", openinference_span_kind="agent") as span:
        span.set_attribute("debug_info", "debug_run_1")
        span.set_input(value=input_state)

        try:
            result = graph.invoke(input_state)
            span.set_output(value=result.get("answer"))
            span.set_status(StatusCode.OK)
            print("[LangGraph] LangGraph execution completed")
            #print("[LangGraph] Result:", result)
            id = result.get("id")
            id = str(id).strip()
            print("Este es el id: "+str(id))
            
            
            tool_evals = {
                "decide_tool": decide_tool_eval,
                "lookup_sales_data": sql_eval,
                "analyzing_data": analysis_eval,
                "create_visualization": visualization_eval
            }

            for tool_name, eval_func in tool_evals.items():
                try:
                    energy_key = f"energy_{tool_name}"
                    tool_ids_key = f"ids_{tool_name}"
                    cpu_key= f"cpu_utilization_{tool_name}"
                    gpu_key= f"gpu_utilization_{tool_name}"

                    # Verify both lists exist
                    if energy_key not in result or tool_ids_key not in result:
                        print(f"Skipping {tool_name}: missing energy or ID list")
                        continue

                    # Check if eval_df is None or empty
                    eval_df = eval_func(result["id"])
                    if eval_df is None or eval_df.empty:
                        print(f"Skipping {tool_name}: eval_df is empty or None")
                        continue
                    if tool_name == "create_visualization":
                        print("Este es el eval_df: "+str(eval_df))
                    log_evaluation_to_csv(
                        eval_df,
                        tool_name=tool_name,
                        id=result["id"],
                        energy=result[energy_key],
                        id_tool=result[tool_ids_key], 
                        cpu_utilization=result[cpu_key],
                        gpu_utilization=result[gpu_key],
                    )
                except Exception as e:
                    print(f"Error processing {tool_name}: {e}")
            return result
        except Exception as e:
            span.set_status(StatusCode.ERROR) 
            span.record_exception(e)
            print("[LangGraph] Error during LangGraph execution:", e)
            raise e





