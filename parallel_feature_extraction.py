import json
import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI   
from pydantic import BaseModel, Field
from tqdm import tqdm

load_dotenv()

# YOU NEED AN OPENAI API KEY TO RUN THIS
# DON'T RUN THE ENTIRE FILE UNLESS YOU WANT A FEW DOLLARS WORTH OF CALLS TO YOUR OPENAI ACCOUNT
assert os.environ.get("OPENAI_API_KEY") is not None



# Prompt material
# ================================================================
class FinancialSentimentFeatureSet(BaseModel):
    """
    A set of features extracted from a financial news article.
    """
    specificity_score: int = Field(description="The score of how specific the article is to an individual company vs the whole market. 0 is the least specific, 10 is the most specific")
    relevance_score: int = Field(description="The score of how relevant the article is to the US stock market. A non-US article would not be relevant. 0 is the least relevant, 10 is the most relevant")
    title_sentiment: int = Field(description="The sentiment of the title of the article. 0 is the least positive, 10 is the most positive")
    article_sentiment: int = Field(description="The sentiment of the entire article. 0 is the least positive, 10 is the most positive")
    first_20_words_sentiment: int = Field(description="The sentiment of the first 20 words of the article. 0 is the least positive, 10 is the most positive")
    last_20_words_sentiment: int = Field(description="The sentiment of the last 20 words of the article. 0 is the least positive, 10 is the most positive")



example_article_with_features = {
    'title': 'Voluntary And Conditional Takeover Bid On Vastned Retail Belgium NV Update',
    'text': 'April 26 (Reuters) - Vastned Retail Belgium NV:\n* REG-VOLUNTARY AND CONDITIONAL TAKEOVER BID ON VASTNED RETAIL BELGIUM NV: UPDATE\n* ACCEPTANCE BY AT LEAST 90% OF FREE FLOAT IS NECESSARY * ACCEPTANCE PERIOD RUNS FROM 2 MAY 2018 THROUGH 1 JUNE 2018 Source text for Eikon: (Gdynia Newsroom)\n ',
    'relevance_score': 1,
    'specificity_score': 8,
    'title_sentiment': 4,
    'article_sentiment': 6,
    'first_20_words_sentiment': 5,
    'last_20_words_sentiment': 6
}


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert financial news sentiment analyzer. "
            "You will be given a financial news article and asked to extract a set of features from it. "
            "Focus on accuracy and consistency in your analysis."
            "Here is an example of an article and the features that were extracted from it: {example_article_with_features}."
        ),
        ("human", "{article}"),
    ]
)
# ================================================================



def get_feature_set_with_llm(article) -> FinancialSentimentFeatureSet:
    """Make LLM infer financial sentiment features from an article."""
    global structured_llm
    prompt = prompt_template.invoke({"article": article, "example_article_with_features": example_article_with_features})
    feature_set = structured_llm.invoke(prompt)
    return feature_set


def get_and_save_feature_set(article) -> None:
    """Compute feature set for an article and save it to a file."""
    file_name = f"data/feature_sets/{article['uuid']}.json"

    # this makes this script dedup-able
    if os.path.exists(file_name):
        return

    article_filtered = {'title': article['title'], 'text': article['text']}
    feature_set = get_feature_set_with_llm(article_filtered)
    with open(file_name, "w") as f:
        json.dump(feature_set.model_dump(), f)


def get_feature_set_with_llm_for_all_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Make LLM infer financial sentiment features from all articles."""
    article_rows = df.to_dict(orient="records")
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(get_and_save_feature_set, tqdm(article_rows, desc="Extracting features")))



if __name__ == "__main__":  

    # articles/input data 
    # ================================================================
    df = pd.read_json("articles.json")
    # ================================================================

    # LLM
    # ================================================================
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    structured_llm = llm.with_structured_output(schema=FinancialSentimentFeatureSet)
    # ================================================================

    # Get feature sets for all articles
    # Data will be collected in the data/feature_sets folder
    # ================================================================
    get_feature_set_with_llm_for_all_articles(df)
    # ================================================================




    