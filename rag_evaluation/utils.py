import os
from dotenv import load_dotenv, find_dotenv

import numpy as np
# from trulens_eval import (
#     Feedback,
#     TruLlama,
#     OpenAI
# )

from trulens.apps.llamaindex import TruLlama
from trulens.core import Feedback
from trulens.providers.openai import OpenAI
import nest_asyncio

nest_asyncio.apply()


def get_openai_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("OPENAI_API_KEY")


def get_hf_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("HUGGINGFACE_API_KEY")

# qa_relevance = (
#     Feedback(openai.relevance_with_cot_reasons, name="Answer Relevance")
#     .on_input_output()
# )

# qs_relevance = (
#     Feedback(openai.relevance_with_cot_reasons, name = "Context Relevance")
#     .on_input()
#     .on(TruLlama.select_source_nodes().node.text)
#     .aggregate(np.mean)
# )

# groundedness = (
#     Feedback(openai.groundedness_measure_with_cot_reasons, name="Groundedness")
#         .on(TruLlama.select_source_nodes().node.text)
#         .on_output()
# )

# feedbacks = [qa_relevance, qs_relevance, groundedness]

def get_trulens_recorder(query_engine, feedbacks, app_id):
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
    )
    return tru_recorder

def get_prebuilt_trulens_recorder(query_engine, app_id):
    # Initialize provider class
    provider = OpenAI()

    context = TruLlama.select_context(query_engine)

    f_groundedness = (
        Feedback(
            provider.groundedness_measure_with_cot_reasons, name="Groundedness"
        )
        .on(context.collect())  # collect context chunks into a list
        .on_output()
    )


    f_answer_relevance = Feedback(
        provider.relevance_with_cot_reasons, name="Answer Relevance"
    ).on_input_output()

    f_context_relevance = (
        Feedback(
            provider.context_relevance_with_cot_reasons, name="Context Relevance"
        )
        .on_input()
        .on(context)
        .aggregate(np.mean)
    )

    tru_recorder = TruLlama(
    query_engine,
    app_name=app_id,
    app_version="base",
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
    )
    return tru_recorder
