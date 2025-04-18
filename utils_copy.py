from dotenv import find_dotenv, load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import langgraph
import langgraph.version
print(langgraph.version)
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
##from pydantic import BaseModel, Field
from IPython.display import Markdown

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

TRANSACTION_DATA_FILE_PATH = 'data/Store_Sales_Price_Elasticity_Promotions_Data.parquet'



# Define the state of the agent's main graph
class State (TypedDict):
    prompt: str
    data: str
    analyze_data : str
    emphasis: str
    answer: list[str]
    visualization_goal: Optional[str]  # Para almacenar el objetivo de visualización
    chart_config: Optional[dict]  # Para almacenar la configuración del gráfico
    tool_choice: NotRequired[str]
    used_tools: NotRequired[list[str]]
    temperature: float
    id: str # Para almacenar el UUID de la ejecución

#Define the state for the subgraph used for parallelization 
class SubgraphState (TypedDict):
    prompt: str
    data: str
    tool_choice: str
    answer_so_far: list[str]
    results: Annotated[list[str], operator.add]

class SQLState(TypedDict):
    prompt: str
    table_name: str
    columns: list[str]
    sql_query: str

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

A good example of a SQL query is (DO NOT COPY THE SAME QUERY ALL THE TIME, JUST USE IT AS A REFERENCE):

SELECT col1, col2, SUM(col3) OVER (PARTITION BY col2 ORDER BY col4) AS trend
FROM {table_name}
WHERE CAST(date_column AS VARCHAR) LIKE '%2021-11%'


"""

#Tool for sql query generation
def generate_sql_query (state:State, columns: list, table_name: str): 
    # Formateamos el prompt para generar la consulta SQL

    temperature = state.get("temperature", 0.1)
    formatted_prompt = SQL_Generation_Prompt.format(prompt=state["prompt"], columns=columns, table_name=table_name)
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
        return sql_query
    except Exception as e:
        return {**state, "error": f"Error accessing data: {str(e)}"}

"""def parallel_sql_gen(state:State, columns: list, table_name: str):
    temperatures = [0.1, 0.45, 0.8]

    substates = [{**state, "temperature": temp} for temp in temperatures]
    
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(generate_sql_query, substate, columns, table_name)
            for substate in substates
        ]
        for future in as_completed(futures):
            try:
                result = future.result()
                if isinstance(result, str):  # solo strings válidos
                    results.append(result)
            except Exception as e:
                print(f"[ERROR] SQL generation failed: {e}")
    
    return results"""

@tracer.tool()
def lookup_sales_data(state:State):
    """Implementation of sales data lookup from parquet file using SQL"""
    #substates = [{**state, "temperature": temp} for temp in [0.1, 0.45, 0.8]]

    try:

        # define the table name
        table_name = "sales"
        
        # step 1: read the parquet file into a DuckDB table
        df = pd.read_parquet(TRANSACTION_DATA_FILE_PATH)
        print("Este es el dataframe: "+str(df.head()))
        duckdb.sql("DROP TABLE IF EXISTS sales")
        duckdb.sql(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df")
        # step 2: generate the SQL code 
        columns = list(df.columns)
        #print("Estas son las columnas: "+str(columns))
        date_columns = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
        temperatures = [0.1, 0.45, 0.8]

        substates = [{**state, "temperature": temp} for temp in temperatures]
    
        results = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(generate_sql_query, substate, columns, table_name)
                for substate in substates
            ]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if isinstance(result, str):  # solo strings válidos
                        results.append(result)
                except Exception as e:
                    print(f"[ERROR] SQL generation failed: {e}")
        #sql_queries = parallel_sql_gen(state, columns, table_name)
            #print("Este es el sql query: "+str(sql_queries))
            # clean the response to make sure it only includes the SQL code
            # step 3: execute the SQL query

        # Search if there is a LIKE in da query and we cast it
        for sql_query in results:        
            for col in date_columns:
                sql_query = sql_query.replace(f"{col} LIKE", f"CAST({col} AS VARCHAR) LIKE")
                sql_query = sql_query.replace(f"{col} ~~", f"CAST({col} AS VARCHAR) ~~")  # DuckDB a veces usa ~~ como LIKE
                sql_query = sql_query.replace(f"{col} = ", f"CAST({col} AS VARCHAR) = ")
                sql_query = sql_query.replace(f"{col} > ", f"CAST({col} AS VARCHAR) > ")
                sql_query = sql_query.replace(f"{col} < ", f"CAST({col} AS VARCHAR) < ")
                
            result = duckdb.sql(sql_query).df()
            results.append(result)      
        
        results = [df.head(1000) for df in results]
        if results:
            base_df = results[0]
            for other in results[1:]:
                base_df = pd.merge(base_df, other, how="inner")
            
            final_result = base_df
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
        #print("Resultado de lookup_sales_data:", result) 
        return {**state, "data": final_result, "answer": state.get("answer", []) + ["The query to create the dataframe is the following: "+sql_query+"\n"], "used_tools": state.get("used_tools", []) + ["lookup_sales_data"], "id": state.get("id")}
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
        return {**state, "analyze_data": analysis_result, "answer": state.get("answer", []) + [analysis_result],         "visualization_goal": state.get("visualization_goal"),
        "chart_config": state.get("chart_config"), "used_tools": state.get("used_tools", []) + ["analyzing_data"], "id": state.get("id")
        }
    except Exception as e:
        return {**state, "error": f"Error accessing data: {str(e)}"}

def split_prompt(emphasis: str) -> list[str]:
    return [
        f"{emphasis} - Focus on revenue trends",
        f"{emphasis} - Focus on product category insights",
        f"{emphasis} - Focus on promotional effects",
    ]

@tracer.tool()
def parallel_analyze_data(state: State) -> State:
    sub_prompts = split_prompt(state["prompt"])

    # Construimos estados individuales para cada subanalisis
    sub_states = [
        {"prompt": state.get("prompt"), "data": state["data"], "tool_choice": state.get("tool_choice"), "answer": state.get("answer"), "analyze_data": state.get("analyze_data"), "used_tools": state.get("used_tools", []), "emphasis": sub_prompt, "id": state.get("id")}
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
        
    # Reducimos
    analysis_outputs = [r[1]['analyze_data'] for r in results]
    final_summary = fuse_analysis_results(analysis_outputs)


    return {
        **state,
        "analyze_data": final_summary,
        "answer": state.get("answer", []) + ["The analisis extracted from the data: "+final_summary+"\n"],
        "used_tools": state.get("used_tools", []) + ["analyzing_data"]
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
Generate a chart configuration based on this data: {data}
The goal is to show: {visualization_goal}
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
    # Verificar explícitamente si existe "visualization_goal" en el estado
    if "visualization_goal" in state:
        visualization_goal = state["visualization_goal"]
    else:
        visualization_goal = state["prompt"]
    formatted_prompt = CHART_CONFIGURATION_PROMPT.format(
            data=state["data"],
            #data = data,
            visualization_goal=visualization_goal
    )
    #print("Este es el estado a la hora de extraer la configuracion del chart: "+str(state.keys()))
    response = llm.invoke(formatted_prompt)
    try:
        # Extract axis and title info from response
        content = response.content
        
        # Return structured chart config
        chart_config={
        "chart_type": content.chart_type,
        "x_axis": content.x_axis,
        "y_axis": content.y_axis,
        "title": content.title,
        "data": state["data"]} 
        print("Este es el chart config: "+str(chart_config))          
        return{**state, 
        "visualization_goal": state.get("visualization_goal"),
        "chart_config": state.get("chart_config"), 
        "analyze_data": state.get("analyze_data"), 
        "used_tools": state.get("used_tools",[])}
    except Exception:
        chart_config={"chart_type": "line", 
        "x_axis": "date",
        "y_axis": "value",
        "title": visualization_goal,
        "data": state["data"]}
        return {**state, 
        "visualization_goal": state.get("visualization_goal"),
        "chart_config": state.get("chart_config"), 
        "analyze_data": state.get("analyze_data"),
        "used_tools": state.get("used_tools",[])}

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
            
            return {**state, "config": config["chart_config"], 
            "visualization_goal": config.get("visualization_goal"),
            "chart_config": config.get("chart_config"), 
            "analyze_data": state.get("analyze_data"), "answer": state.get("answer", []) + ["This is the code to visualize: "+code], "used_tools": state.get("used_tools", []) + ["create_visualization"], "id": state.get("id")}
    except Exception as e:
            return {**state, "error": f"Error accessing data: {str(e)}"}


@tracer.tool()
def decide_tool(state: State, llm) -> State:

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

    print(f"Elección de herramienta: {matched_tool}")

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


def log_evaluation_to_csv(eval_df: pd.DataFrame, tool_name: str, file_path: str = "tool_evaluations.csv"):
    eval_df["tool_name"] = tool_name
    eval_df["timestamp"] = datetime.now().isoformat()
    eval_df["energy_estimate_Ws"] = None  # En el futuro lo puedes llenar

    # Reordenar columnas si quieres
    cols_order = ["tool_name", "context.span_id", "timestamp", "execution_seconds", "score", "label", "explanation", "energy_estimate_Ws"]
    for col in cols_order:
        if col not in eval_df.columns:
            eval_df[col] = None
    eval_df = eval_df[cols_order]

    # Guardar / añadir
    try:
        existing = pd.read_csv(file_path)
        combined = pd.concat([existing, eval_df])
    except FileNotFoundError:
        combined = eval_df

    combined.to_csv(file_path, index=False)

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
            log_evaluation_to_csv(sql_eval(id), "lookup_sales_data")
            log_evaluation_to_csv(analysis_eval(id), "analyzing_data")
            log_evaluation_to_csv(visualization_eval(id), "create_visualization")
            log_evaluation_to_csv(decide_tool_eval(id), "decide_tool")
            return result
        except Exception as e:
            span.set_status(StatusCode.ERROR)
            span.record_exception(e)
            print("[LangGraph] Error during LangGraph execution:", e)
            raise e
        
agent_run_id = str(uuid.uuid4())        
input_state = {"prompt": "Create me a visualization considering the trends found on the sales of Nov 2021", "id":agent_run_id}
result = run_graph_with_tracing(input_state)




    