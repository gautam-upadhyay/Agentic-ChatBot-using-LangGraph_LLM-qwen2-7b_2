import csv
import json
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from typing import List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

CSV_PATH = os.path.join(os.path.dirname(__file__), "imbd_movie.csv")
TABLE_NAME = "movies"
MODEL_NAME = "qwen2:7b"

PERSONA = (
    "You are MovieAnalyst, a helpful data analyst for the IMDB movies dataset. "
    "You only answer questions that can be answered by the imbd_movie.csv dataset. "
    "If a question is not about the movie dataset, respond with a brief refusal "
    "and ask the user to rephrase using the dataset."
)

SQL_GEN_SYSTEM = (
    PERSONA
    + "\n\nYou generate a single SQLite SELECT query that answers the user's question.\n"
    + "Rules:\n"
    + "- Use only SELECT queries.\n"
    + "- Do not modify data.\n"
    + "- If the question is not about the database, output exactly: NO_SQL\n"
    + "- Prefer explicit column names and JOINs over SELECT *.\n\n"
    + "Output requirements:\n"
    + "- Return only SQL (no prose, no markdown).\n"
    + "- End the query with a semicolon.\n\n"
    + "If the question is about movies, ratings, genres, directors, stars, votes, or box office,\n"
    + "you MUST return a SQL query (not NO_SQL).\n\n"
    + "Database schema:\n{schema}\n"
    + "\nExamples:\n"
    + "Q: What are the top 5 movies by IMDB rating?\n"
    + "A: SELECT Series_Title, IMDB_Rating FROM movies "
    + "ORDER BY IMDB_Rating DESC, No_of_Votes DESC LIMIT 5;\n"
    + "Q: Which director has the most movies in the dataset?\n"
    + "A: SELECT Director, COUNT(*) AS MovieCount FROM movies "
    + "GROUP BY Director ORDER BY MovieCount DESC LIMIT 1;\n"
)

ANSWER_SYSTEM = (
    PERSONA
    + "\n\nAnswer the user based strictly on the SQL results provided. "
    + "If SQL result is NO_SQL, refuse and ask for a dataset question. "
    + "If SQL result is NO_ROWS, say no matching data was found and ask a clarifying question."
)


class GraphState(TypedDict):
    messages: List[BaseMessage]
    sql: str
    sql_result: str
    sql_gen_ms: float
    sql_exec_ms: float
    answer_ms: float
    success: bool
    accuracy: float | None


_DB_CONN: sqlite3.Connection | None = None
_DB_ERROR: str | None = None


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    cleaned = re.sub(r"[^\d]", "", value)
    return int(cleaned) if cleaned else None


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = value.strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _load_movies_csv(csv_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE movies (
            Poster_Link TEXT,
            Series_Title TEXT,
            Released_Year INTEGER,
            Certificate TEXT,
            Runtime INTEGER,
            Genre TEXT,
            IMDB_Rating REAL,
            Overview TEXT,
            Meta_score INTEGER,
            Director TEXT,
            Star1 TEXT,
            Star2 TEXT,
            Star3 TEXT,
            Star4 TEXT,
            No_of_Votes INTEGER,
            Gross INTEGER
        )
        """
    )
    with open(csv_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            rows.append(
                (
                    row.get("Poster_Link"),
                    row.get("Series_Title"),
                    _to_int(row.get("Released_Year")),
                    row.get("Certificate"),
                    _to_int(row.get("Runtime")),
                    row.get("Genre"),
                    _to_float(row.get("IMDB_Rating")),
                    row.get("Overview"),
                    _to_int(row.get("Meta_score")),
                    row.get("Director"),
                    row.get("Star1"),
                    row.get("Star2"),
                    row.get("Star3"),
                    row.get("Star4"),
                    _to_int(row.get("No_of_Votes")),
                    _to_int(row.get("Gross")),
                )
            )
    cur.executemany(
        """
        INSERT INTO movies (
            Poster_Link, Series_Title, Released_Year, Certificate, Runtime, Genre,
            IMDB_Rating, Overview, Meta_score, Director, Star1, Star2, Star3, Star4,
            No_of_Votes, Gross
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return conn


def _init_db() -> str | None:
    global _DB_CONN, _DB_ERROR
    if _DB_CONN or _DB_ERROR:
        return _DB_ERROR
    if not os.path.exists(CSV_PATH):
        _DB_ERROR = "ERROR: imbd_movie.csv not found."
        return _DB_ERROR
    try:
        _DB_CONN = _load_movies_csv(CSV_PATH)
    except Exception as exc:
        _DB_ERROR = f"ERROR: Failed to load imbd_movie.csv ({exc})"
    return _DB_ERROR


def _get_schema_description() -> str:
    db_error = _init_db()
    if db_error:
        return db_error
    assert _DB_CONN is not None
    cur = _DB_CONN.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    tables = [row[0] for row in cur.fetchall()]
    lines = []
    for table in tables:
        cur.execute(f"PRAGMA table_info({table})")
        cols = [f"{col[1]} ({col[2]})" for col in cur.fetchall()]
        lines.append(f"- {table}: " + ", ".join(cols))
    return "\n".join(lines)


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
    cleaned = re.sub(r"\n```$", "", cleaned)
    return cleaned.strip()


def _extract_sql(text: str) -> str:
    candidate = _strip_code_fences(text)
    if ";" in candidate:
        candidate = candidate.split(";", 1)[0] + ";"
    return candidate.strip()


def _format_history(messages: List[BaseMessage], max_turns: int = 6) -> str:
    recent = messages[-max_turns:]
    lines = []
    for msg in recent:
        if isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage):
            role = "Assistant"
        else:
            role = "System"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def _is_select_only(query: str) -> bool:
    q = query.strip().strip(";")
    if not q.lower().startswith("select"):
        return False
    banned = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "pragma",
        "attach",
        "detach",
        "replace",
    ]
    for kw in banned:
        if re.search(rf"\\b{kw}\\b", q, flags=re.IGNORECASE):
            return False
    return True


def _heuristic_sql(question: str) -> str | None:
    q = question.lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    if "how many" in q and ("movie" in q or "movies" in q):
        return "SELECT COUNT(*) AS MovieCount FROM movies;"
    if "top" in q and "imdb rating" in q:
        return (
            "SELECT Series_Title, IMDB_Rating, No_of_Votes FROM movies "
            "ORDER BY IMDB_Rating DESC, No_of_Votes DESC LIMIT 5;"
        )
    if "highest" in q and "imdb rating" in q:
        return (
            "SELECT Series_Title, IMDB_Rating, No_of_Votes FROM movies "
            "ORDER BY IMDB_Rating DESC, No_of_Votes DESC LIMIT 1;"
        )
    if ("most votes" in q or "highest votes" in q) and ("movie" in q or "movies" in q):
        return (
            "SELECT Series_Title, No_of_Votes FROM movies "
            "ORDER BY No_of_Votes DESC LIMIT 1;"
        )
    if "director" in q and ("most movies" in q or "most films" in q):
        return (
            "SELECT Director, COUNT(*) AS MovieCount FROM movies "
            "GROUP BY Director ORDER BY MovieCount DESC LIMIT 1;"
        )
    return None


@tool("run_sql_query")
def run_sql_query(query: str) -> str:
    """Execute a read-only SQL query against the IMDB movies dataset."""
    if not _is_select_only(query):
        return "ERROR: Only SELECT queries are allowed."
    db_error = _init_db()
    if db_error:
        return db_error
    assert _DB_CONN is not None
    cur = _DB_CONN.cursor()
    try:
        cur.execute(query)
    except sqlite3.Error as exc:
        return f"ERROR: {exc}"
    rows = cur.fetchmany(21)
    if not rows:
        return "NO_ROWS"
    columns = rows[0].keys()
    display_rows = rows[:20]
    header = " | ".join(columns)
    separator = "-+-".join(["-" * len(col) for col in columns])
    body = []
    for row in display_rows:
        body.append(" | ".join(str(row[col]) for col in columns))
    suffix = ""
    if len(rows) > 20:
        suffix = "\n... showing first 20 rows"
    return "\n".join([header, separator] + body) + suffix


def _ollama_base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL") or os.environ.get(
        "OLLAMA_HOST", "http://localhost:11434"
    )


def _ollama_server_check() -> str | None:
    base_url = _ollama_base_url().rstrip("/")
    url = f"{base_url}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return (
            "ERROR: Ollama server is not reachable.\n"
            "Start it with: ollama serve\n"
            f"Then ensure the model is pulled: ollama pull {MODEL_NAME}\n"
            f"If Ollama is remote, set OLLAMA_HOST or OLLAMA_BASE_URL to {base_url}"
        )

    models = {m.get("name") for m in payload.get("models", []) if m.get("name")}
    if MODEL_NAME not in models:
        return (
            f"ERROR: Ollama model '{MODEL_NAME}' not found.\n"
            f"Run: ollama pull {MODEL_NAME}"
        )
    return None


def _build_model() -> ChatOllama:
    return ChatOllama(model=MODEL_NAME, base_url=_ollama_base_url())


def _parse_confidence(text: str) -> float | None:
    cleaned = _strip_code_fences(text).strip()
    match = re.search(r"([01](?:\.\d+)?)", cleaned)
    if not match:
        return None
    value = float(match.group(1))
    if value < 0 or value > 1:
        return None
    return value


def _estimate_accuracy(model: ChatOllama, question: str, sql: str, sql_result: str) -> float | None:
    prompt = [
        SystemMessage(
            content=(
                "You are an evaluator. Estimate how likely the answer is correct "
                "given the SQL and result. Return only a number between 0 and 1."
            )
        ),
        HumanMessage(
            content=(
                f"Question: {question}\n\n"
                f"SQL: {sql}\n\n"
                f"Result: {sql_result}\n\n"
                "Return only a number between 0 and 1."
            )
        ),
    ]
    response = model.invoke(prompt).content
    return _parse_confidence(response)


def sql_generate_node(state: GraphState) -> GraphState:
    model = _build_model()
    schema = _get_schema_description()
    history_text = _format_history(state["messages"])
    latest_question = state["messages"][-1].content
    start = time.perf_counter()
    prompt = [
        SystemMessage(content=SQL_GEN_SYSTEM.format(schema=schema)),
        HumanMessage(
            content=(
                "Conversation so far:\n"
                f"{history_text}\n\n"
                f"User question: {latest_question}\n\n"
                "Return only SQL or NO_SQL."
            )
        ),
    ]
    response = model.invoke(prompt).content
    sql = _extract_sql(response)
    if sql.upper() != "NO_SQL" and not _is_select_only(sql):
        retry_prompt = [
            SystemMessage(content=SQL_GEN_SYSTEM.format(schema=schema)),
            HumanMessage(
                content=(
                    "Return ONLY a single SELECT statement ending with a semicolon. "
                    "No prose. No markdown. If not a DB question, return NO_SQL.\n\n"
                    f"User question: {latest_question}"
                )
            ),
        ]
        retry_response = model.invoke(retry_prompt).content
        sql = _extract_sql(retry_response)
        if sql.upper() != "NO_SQL" and not _is_select_only(sql):
            sql = "NO_SQL"
    if sql.upper() == "NO_SQL":
        heuristic = _heuristic_sql(latest_question)
        if heuristic and _is_select_only(heuristic):
            sql = heuristic
    duration_ms = (time.perf_counter() - start) * 1000
    return {"sql": sql, "sql_gen_ms": duration_ms}


def sql_execute_node(state: GraphState) -> GraphState:
    sql = (state.get("sql") or "").strip()
    start = time.perf_counter()
    if not sql or sql.upper().startswith("NO_SQL"):
        duration_ms = (time.perf_counter() - start) * 1000
        return {"sql_result": "NO_SQL", "sql_exec_ms": duration_ms}
    result = run_sql_query.invoke(sql)
    duration_ms = (time.perf_counter() - start) * 1000
    return {"sql_result": result, "sql_exec_ms": duration_ms}


def answer_node(state: GraphState) -> GraphState:
    user_question = state["messages"][-1].content
    sql = state.get("sql", "")
    sql_result = state.get("sql_result", "")
    start = time.perf_counter()
    if sql_result in {"NO_SQL", "NO_ROWS"} or sql_result.startswith("ERROR:"):
        if sql_result == "NO_ROWS":
            response = (
                "I couldn't find matching data in imbd_movie.csv. "
                "Can you clarify your request (movie, director, rating, genre, year)?"
            )
        elif sql_result == "NO_SQL":
            response = (
                "Please ask a question that can be answered from imbd_movie.csv "
                "(movies, ratings, genres, directors, stars, votes, box office)."
            )
        else:
            response = (
                "There was an issue running the SQL query. "
                "Please try rephrasing your question."
            )
        updated_messages = state["messages"] + [AIMessage(content=response)]
        duration_ms = (time.perf_counter() - start) * 1000
        return {
            "messages": updated_messages,
            "answer_ms": duration_ms,
            "success": False,
            "accuracy": None,
        }

    model = _build_model()
    prompt = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(
            content=(
                f"User question: {user_question}\n\n"
                f"SQL used:\n{sql}\n\n"
                f"SQL result:\n{sql_result}\n\n"
                "Answer the user."
            )
        ),
    ]
    response = model.invoke(prompt).content
    updated_messages = state["messages"] + [AIMessage(content=response)]
    accuracy = _estimate_accuracy(model, user_question, sql, sql_result)
    duration_ms = (time.perf_counter() - start) * 1000
    return {
        "messages": updated_messages,
        "answer_ms": duration_ms,
        "success": True,
        "accuracy": accuracy,
    }


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("sql_generate", sql_generate_node)
    graph.add_node("sql_execute", sql_execute_node)
    graph.add_node("answer", answer_node)
    graph.set_entry_point("sql_generate")
    graph.add_edge("sql_generate", "sql_execute")
    graph.add_edge("sql_execute", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def main() -> None:
    db_error = _init_db()
    if db_error:
        print(db_error)
        return
    ollama_error = _ollama_server_check()
    if ollama_error:
        print(ollama_error)
        return
    app = build_graph()
    state: GraphState = {
        "messages": [],
        "sql": "",
        "sql_result": "",
        "sql_gen_ms": 0.0,
        "sql_exec_ms": 0.0,
        "answer_ms": 0.0,
        "success": False,
        "accuracy": None,
    }
    print("IMDB movie chatbot ready. Type 'exit' to quit.")
    while True:
        user_input = input("> ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        state["messages"].append(HumanMessage(content=user_input))
        state = app.invoke(state)
        print(state["messages"][-1].content)
        total_ms = (
            state.get("sql_gen_ms", 0.0)
            + state.get("sql_exec_ms", 0.0)
            + state.get("answer_ms", 0.0)
        )
        accuracy = state.get("accuracy")
        accuracy_text = "N/A" if accuracy is None else f"{accuracy:.2f}"
        print(
            "Metrics: "
            f"latency_ms={total_ms:.0f}, "
            f"success={state.get('success', False)}, "
            f"accuracy={accuracy_text}"
        )


if __name__ == "__main__":
    main()
