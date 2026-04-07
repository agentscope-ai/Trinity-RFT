"""
This file defines Email Dataclass and three email_search_tools.
Modified from https://github.com/OpenPipe/ART/blob/art-e/examples/art-e/
"""
import datetime
import os
import sqlite3
from dataclasses import dataclass
from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator

from trinity.utils.log import get_logger

DEFAULT_DB_PATH = os.environ.get("DEFAULT_EMAIL_DB_PATH")
conn = None


def get_conn():
    global conn
    if conn is None:
        conn = sqlite3.connect(f"file:{DEFAULT_DB_PATH}?mode=ro", uri=True, check_same_thread=False)
    return conn


############ Define dataclass ############


class QueryModel(BaseModel):
    id: int
    question: str
    answer: str
    message_ids: List[str]  # message_ids (strings) of referenced emails
    how_realistic: float
    inbox_address: str
    query_date: str

    @field_validator("query_date", mode="before")
    @classmethod
    def format_date(cls, v: Any) -> str:
        if isinstance(v, datetime.datetime):
            return v.strftime("%Y-%m-%d")
        return v


class AnswerModel(BaseModel):
    answer: str = Field(
        description="It should be called with the answer and the sources. If you cannot find the answer, you should return 'I don't know' with an empty list of sources."
    )
    sources: List[str] = Field(
        description="a list of message ids that are relevant to the query. Usually there will be only one. If you cannot find the answer, you should return an empty list."
    )


class Email(BaseModel):
    message_id: str
    date: str  # ISO 8601 string 'YYYY-MM-DD HH:MM:SS'
    subject: Optional[str] = None
    from_address: Optional[str] = None
    to_addresses: List[str] = Field(default_factory=list)
    cc_addresses: List[str] = Field(default_factory=list)
    bcc_addresses: List[str] = Field(default_factory=list)
    body: Optional[str] = None
    file_name: Optional[str] = None


@dataclass
class SearchResult:
    message_id: str
    snippet: str


class FinalRubric(BaseModel):
    answer_correct: bool = False
    sources_correct: bool = False
    num_turns: int = 0
    attempted_answer: bool = False
    ever_found_right_email: bool = False
    ever_read_right_email: bool = False
    cant_parse_tool_call: bool = False
    bad_tool_call_name: bool = False
    bad_tool_call_args: bool = False
    ran_out_of_turns: bool = False
    returned_i_dont_know: bool = False
    num_sources: int = 0
    ever_tried_to_read_invalid_email: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0


############ Define tools for agentscope ############


def search_emails_tool(
    inbox: str,
    keywords: List[str],
    from_addr: Optional[str] = None,
    to_addr: Optional[str] = None,
    sent_after: Optional[str] = None,
    sent_before: Optional[str] = None,
    max_results: int = 10,
) -> List[SearchResult]:
    """
    Searches the email database based on keywords, inbox, sender, recipient, and date range.

    Args:
        inbox: The email address of the user performing the search.
               Results include emails sent from or to (inc. cc/bcc) this address.
        keywords: A list of keywords that must all appear in the subject or body.
        from_addr: Optional email address to filter emails sent *from*.
        to_addr: Optional email address to filter emails sent *to* (inc. cc/bcc).
        sent_after: Optional date string 'YYYY-MM-DD'. Filters for emails sent on or after this date.
        sent_before: Optional date string 'YYYY-MM-DD'. Filters for emails sent before this date.
        max_results: The maximum number of results to return. Cannot exceed 10.

    Returns:
        A list of SearchResult objects, each containing 'message_id' and 'snippet'.
        Returns an empty list if no results are found or an error occurs.
    """
    # Initialize sql and params
    sql: Optional[str] = None
    params: List[str | int] = []

    cursor = get_conn().cursor()

    # --- Build Query ---
    where_clauses: List[str] = []

    # 1. Keywords (FTS)
    if not keywords:
        raise ValueError("No keywords provided for search.")

    if max_results > 10:
        raise ValueError("max_results must be less than or equal to 10.")

    # FTS5 default is AND, so just join keywords. Escape quotes for safety.
    fts_query = " ".join(f""" "{k.replace('"', '""')}" """ for k in keywords)
    where_clauses.append("emails_fts MATCH ?")
    params.append(fts_query)

    # 2. Inbox filter (must be from OR to/cc/bcc the inbox user)
    # Use the composite index idx_recipients_address_email here
    where_clauses.append(
        """
        (e.from_address = ? OR EXISTS (
            SELECT 1 FROM recipients r_inbox
            WHERE r_inbox.recipient_address = ? AND r_inbox.email_id = e.id
        ))
        """
    )
    params.extend([inbox, inbox])

    # 3. Optional From filter
    if from_addr:
        where_clauses.append("e.from_address = ?")
        params.append(from_addr)

    # 4. Optional To filter (includes to, cc, bcc)
    # Use the composite index idx_recipients_address_email here
    if to_addr:
        where_clauses.append(
            """
            EXISTS (
                SELECT 1 FROM recipients r_to
                WHERE r_to.recipient_address = ? AND r_to.email_id = e.id
            )
            """
        )
        params.append(to_addr)

    # 5. Optional Sent After filter
    if sent_after:
        # Assumes date format 'YYYY-MM-DD'
        # Compare against the start of the day
        where_clauses.append("e.date >= ?")
        params.append(f"{sent_after} 00:00:00")

    # 6. Optional Sent Before filter
    if sent_before:
        # Assumes date format 'YYYY-MM-DD'
        # Compare against the start of the day (exclusive)
        where_clauses.append("e.date < ?")
        params.append(f"{sent_before} 00:00:00")

    # --- Construct Final Query ---
    # snippet(<table>, <column_index>, <highlight_start>, <highlight_end>, <ellipsis>, <tokens>)
    # -1 means highlight across all columns (subject, body)
    sql = f"""
        SELECT
            e.message_id,
            snippet(emails_fts, -1, '<b>', '</b>', ' ... ', 15) as snippet
        FROM
            emails e JOIN emails_fts fts ON e.id = fts.rowid
        WHERE
            {" AND ".join(where_clauses)}
        ORDER BY
            e.date DESC -- Order by date for relevance
        LIMIT ?;
    """
    logger = get_logger(__name__)

    params.append(max_results)

    # --- Execute and Fetch ---
    logger.debug(f"Executing SQL: {sql}")
    logger.debug(f"With params: {params}")
    cursor.execute(sql, params)
    results = cursor.fetchall()

    # Format results
    formatted_results = [SearchResult(message_id=row[0], snippet=row[1]) for row in results]
    logger.info(f"Search found {len(formatted_results)} results.")
    return formatted_results


def read_email_tool(message_id: str) -> Optional[Email]:
    """
    Retrieves a single email by its message_id from the database.

    Args:
        message_id: The unique identifier of the email to retrieve.

    Returns:
        An Email object containing the details of the found email,
        or None if the email is not found or an error occurs.
    """
    logger = get_logger(__name__)

    cursor = get_conn().cursor()

    # --- Query for Email Core Details ---
    email_sql = """
        SELECT id, message_id, date, subject, from_address, body, file_name
        FROM emails
        WHERE message_id = ?;
    """
    cursor.execute(email_sql, (message_id,))
    email_row = cursor.fetchone()

    if not email_row:
        logger.warning(f"Email with message_id '{message_id}' not found.")
        return None

    email_pk_id, msg_id, date, subject, from_addr, body, file_name = email_row

    # DEBUG
    logger.info(f"[read_email_tool] input_message_id={message_id}")
    logger.info(f"[read_email_tool] db: id={email_pk_id}, message_id={msg_id}")

    # search for recipients by emails.id (rather than message_id)
    recipients_sql = """
        SELECT recipient_address, recipient_type
        FROM recipients
        WHERE email_id = ?;
    """
    cursor.execute(recipients_sql, (email_pk_id,))
    recipient_rows = cursor.fetchall()

    to_addresses: List[str] = []
    cc_addresses: List[str] = []
    bcc_addresses: List[str] = []

    for addr, rtype in recipient_rows:
        type_lower = rtype.lower()
        if type_lower == "to":
            to_addresses.append(addr)
        elif type_lower == "cc":
            cc_addresses.append(addr)
        elif type_lower == "bcc":
            bcc_addresses.append(addr)

    # --- Construct Email Object ---
    email_obj = Email(
        message_id=msg_id,  # Convert to string to match Pydantic model
        date=date,
        subject=subject,
        from_address=from_addr,
        to_addresses=to_addresses,
        cc_addresses=cc_addresses,
        bcc_addresses=bcc_addresses,
        body=body,
        file_name=file_name,
    )

    return email_obj


############ LLM-as-a-judge ############


async def judge_correctness(
    answer: str,
    query: QueryModel,
    judge: Any,
) -> bool:
    """Use an LLM to decide whether *answer* matches *query.answer*.

    Returns a boolean *accept* flag used for scoring.
    """

    system_prompt = """You are given a question, the reference answer (labelled **Reference answer**), and an answer generated by an AI assistant (labelled **AI answer**).

Follow these steps to decide whether the AI answer should be accepted:
1. Identify EXACTLY what information the **question** is asking for (e.g. who, what, when, where, why, how, quantity, etc.).
2. From the **Reference answer**, extract ONLY the facts that are required to directly satisfy the information need identified in step 1. Treat all other facts as non-essential context.
3. Verify that every essential fact from step 2 appears in the **AI answer** with the same meaning. Differences in wording, order, or additional non-conflicting details are allowed.
4. If any essential fact is missing or contradicted in the **AI answer**, then *accept* must be **false**. Otherwise *accept* must be **true**.

Important: Do NOT penalise the **AI answer** for omitting non-essential facts that appear in the **Reference answer**. The answer should only be rejected for errors or omissions in the information explicitly requested by the question.

Return your judgement **accept** from **true** and **false**. Do not return any other text or formatting.
"""
    prompt = (
        f"Question: {query.question}\n" f"Reference answer: {query.answer}\n" f"AI answer: {answer}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    completion = await judge.chat.completions.create(
        model=judge.model_path, messages=messages, stream=False
    )
    result = completion.choices[0].message.content
    logger = get_logger(__name__)
    logger.info(f"LLM judge response: {result}")

    # TODO: more robust judge
    accept = False
    if "true" in result.lower():
        accept = True

    return accept
