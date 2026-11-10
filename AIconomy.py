import logging
import os
from datetime import datetime
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

try:
    from ddgs import DDGS
except ImportError:
    # fallback pentru instalari vechi
    from duckduckgo_search import DDGS

# utilizator > CrewAI process > Web researcher (DuckDuckGo) > data calculator > critic / raport > Gemini

# config logging (inainte de orice apel de logging.info)
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@tool("Search Tool")
def search_tool(query: str) -> str:
    """Search for real economic and financial data on the internet."""
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            #EXPLICIT ca nu s-au gasit date
            return "NO RESULTS FOUND for this query"
        return str(results)
    except Exception as e:
        return f"ERROR: {e}"


@tool("Calculator Tool")
def calc_expression(expresie: str) -> str:
    """Used for mathematical calculations. Input must be a valid mathematical expression"""
    try:
        # __builtins__ / -> fara import/open/exec/etc, doar operatii matematice
        return str(eval(expresie, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"ERROR: {e}"


@tool("Economic Forecast Tool")
def trend_estimation(val_init: float, val_final: float, ani_istoric: int, ani_viitor: int, indicator: str) -> str:
    """
    MANDATORY TOOL FOR MATH AND FORECASTS! YOU MUST USE THIS TOOL for any trend, CAGR, or rate calculations.
    YOU ARE STRICTLY FORBIDDEN from doing mental math or guessing results!

    Use this tool WHENEVER you need to calculate:
    - Compound Annual Growth Rate (CAGR)
    - Percentage variations and averages
    - Future economic estimates, trends, and forecasts

    'indicator' MUST be strictly one of two words:
    * 'volume' (for absolute numbers like GDP, total debt, population) -> calculates CAGR
    * 'rate' (for percentages like inflation %, deficit %, interest rates %) -> calculates linear variation (delta)
    """
    try:
        if indicator == "volume":
            #rata compusa CAGR pt dezvoltare economica (estimativ)
            cagr = (val_final / val_init) ** (1 / ani_istoric) - 1
            estimat = val_final * ((1 + cagr) ** ani_viitor)
            return f"CAGR / Compound Annual Growth Rate = {cagr*100:.2f}% - Estimated value(s) over the next {ani_viitor} year(s) is {estimat:.2f}"

        elif indicator == "rate":
            #var lin pentru inflatie, deficit, dobanzi
            delta_anual = (val_final - val_init) / ani_istoric
            estimat = val_final + (delta_anual * ani_viitor)
            return f"The average variation is {delta_anual:.2f} puncte/an. The estimated rate over the next {ani_viitor} year(s) is {estimat:.2f}%"

        else:
            return "ERROR: 'indicator' MUST BE volume / rate!"

    except Exception as e:
        return f"ERROR during predictive calc: {e}"


def macro_analysis(prompt: str, context_recent: str, docs_local: bool = False) -> str:

    logging.info(f"Analysis begins - {prompt}")
    azi = datetime.now().strftime("%d %B %Y")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return ("ERROR - GEMINI_API_KEY NOT configured ")

#def setup
    llm_local = LLM(
        model=os.environ.get("GEMINI_MODEL", "gemini/gemini-3.8-flash"),
        api_key=api_key,
        temperature=0.1,
    )

    tools_research = [search_tool]

    if docs_local:
        instr_rag = (
            "Excerpts from local Romanian documents are already included in the "
            "'Recent conversation context' below, under the '[LOCAL DOCUMENTS CONTEXT]' "
            "marker. Treat that content as the primary source for Romanian national "
            "economic data. Use the Web Search Tool ONLY for complementary external "
            "context, international data or information not present in the local documents"
        )
    else:
        instr_rag = (
            "No local Romanian document excerpts are available for this query. Use the Web Search tool."
        )

    #AGENT 1 Researcher
    web_agent = Agent(
        role="AIconomy web researcher",
        goal="Search for real economic and financial data in the provided local context or on the web",
        backstory="""
        You are an economic data analyst and YOU MUST FOLLOW THESE CRITICAL RULES:
        1. You MUST trigger the Search Tool to fetch information whenever it is not
           already provided to you as context
        2. NEVER simulate tool calls or write JSON tool payloads in your Final Answer
        3. NEVER invent numbers. If a tool returns no data, report clearly that (some) data is missing!

        You are STRICTLY FORBIDDEN from performing any mathematical calculations, CAGR, or forecasts.
        If you see numbers or a math request, you must pass the raw numbers directly to the Quantitative Analyst without calculating the final result yourself!
        """,
        tools=tools_research,
        llm=llm_local,
        verbose=True,
        allow_delegation=False,
        max_iter=5
    )

    # AGENT 2 Quantitative analyst
    calc_agent = Agent(
        role="Quantitative analyst and forecaster",
        goal=(
            "Perform mathematical calculations and predict future economic trends based on historical data ONLY WHEN NEEDED OR REQUESTED!"
        ),
        backstory="""
		You are a quantitative economic analyst and YOU MUST FOLLOW THESE CRITICAL RULES:
        1. If the user's request requires projection or calculation, use the tools
        2. IF NO CALCULATION IS REQUIRED (purely descriptive or qualitative questions), DO NOT USE ANY TOOL - Pass the data exactly as received!
        3. Never invent or estimate numbers without executing the tools!
        """,
        tools=[calc_expression, trend_estimation],
        llm=llm_local,
        verbose=True,
        allow_delegation=False
    )

    # AGENT 3 - Critic / Report / Visualization
    critic_agent = Agent(
        role="Macroeconomic critic and data visualizer",
        goal=(
            "Evaluate data logically, format comparisons in tables, and extract chronological data for charts!"
        ),
        backstory="""
        You are a strict economic analyst. CRITICAL RULES to follow:

        1 - Zero hallucination: If the required data is NOT provided by the Researcher or Calculator, you MUST output:
			"MISSING INFORMATION: Not enough data found for analysis"
        Do NOT invent numbers!
        2 - Formatting: ALWAYS use Markdown Tables for comparisons between entities!
		3 - Charting: If the analysis includes an evolution over time, trends / economic forecasts, include a JSON block at the very end. For example:

        ```json
        {
            "grafic": {
                "2021": 4.5,
                "2022": 5.1,
                "2023": 6.8,
                "2024 (Prognoză)": 7.2
            },
            "tip_grafic": "line_chart"
        }
        ```

        Go straight to the point, check whether the results make sense and draft the conclusion.
        """,
        llm=llm_local,
        verbose=True,
        allow_delegation=False
    )

    # TASK 1 Research
    task_research = Task(
        description=f"""
        Today's date is {azi}. Use this to judge what counts as "current" /  "recent" data, and to spot when a search result is stale.

        Recent conversation context:
        {context_recent}
        Solve the user's following macroeconomic request:
        "{prompt}"

        {instr_rag}
        SOURCE PRIORITY RULES:

        - For Romanian economic indicators such as GDP/PIB, inflation, prices, budget execution, fiscal deficit, public debt,
          monetary policy, financial stability or other national indicators: FIRST check the local documents context above, if present.

        - For international information such as EU, Euro Area, Germany, France, USA, ECB, Fed or global economic developments:
          use the Web Search Tool.

        - For comparisons between Romania and another country or region: use the Romanian data from the local documents context (if present)
          and foreign data from the Web Search Tool.

        - If Romanian data is NOT available in the local documents context, explicitly state that the local information is missing, then use
          Web Search ONLY AS A FALLBACK.

        Extract the exact figures, years, units and relevant source/document information and pass them forward
        """,
        expected_output=(
            "Economic data extracted from the appropriate sources, "
            "including relevant years, values, units and sources."
        ),
        agent=web_agent
    )

    # TASK 2 Calculation
    task_calculate = Task(
        description="""
		Analyze the data received from the Researcher:
        - If the request explicitly requires a forecast, trend projection or math calculation - execute the 'Calculator Tool' or 'Economic Forecast Tool'
        	using ONLY the provided numbers.
        - If the request DOES NOT require calculations (e.g. asking for causes, context, or raw numbers) - DO NOT CALL ANY TOOL. Pass the exact data forward to the reporter.
        """,
        expected_output=("The result of the calculations or the original data if no calculation or math is required!"),
        agent=calc_agent
    )

    # TASK 3 - Final report
    task_report = Task(
        description="""
        Logically evaluate the received data.

        - If the data is missing, apply the graceful degradation rule.
        - Never invent numbers!!!
        - If comparing two or more entities, use a Markdown Table.
        - If the request contains a historical evolution or forecast,
          rely strictly on the previously retrieved/calculated data.
        - If appropriate, include the JSON chart block at the very end.
        """,
        expected_output=(
            "A professional report formatted in Markdown, including tables when appropriate and a JSON chart block when a time evolution or forecast is present."),
        agent=critic_agent
    )

    # CREW
    crew = Crew(
        agents=[web_agent, calc_agent, critic_agent],
        tasks=[task_research, task_calculate, task_report],
        process=Process.sequential,
        max_rpm=int(os.environ.get("CREW_MAX_RPM", 3))  #ajustabil
    )

    try:
        rezultat_final = crew.kickoff()
        print(crew.usage_metrics)
        logging.info("Process successfully completed")
        return rezultat_final


    except Exception as e:
        logging.info(
            "An error occured while the agents were processing the information. "
            f"The error is the following - {e}"
        )
        return f"Agents have crashed - ERROR {e}"
