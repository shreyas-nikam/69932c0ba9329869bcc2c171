import pandas as pd
import numpy as np
from datasets import load_dataset # Keep for potential future use or if a different mock strategy is adopted
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
from collections import Counter
from io import StringIO
import os
from sklearn.manifold import TSNE

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# --- Global Constants and Configuration ---
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Define keywords for distress detection based on financial lexicon
DISTRESS_KEYWORDS = [
    'bankrupt', 'default', 'restructur', 'insolvenc', 'going concern',
    'chapter 11', 'downgrad', 'credit watch', 'writedown', 'impairment',
    'covenant breach', 'liquidity crisis', 'layoff', 'plant clos',
    'dividend cut', 'suspend', 'debt', 'loss', 'crisis', 'distress',
    'downgrade', 'delist', 'insolvent', 'receivership', 'collapse',
    'struggle', 'warning', 'concern', 'emergency', 'shutter', 'impairment',
    'decline', 'missed payment', 'reorganization'
]

# Few-shot examples for LLM prompt
REAL_DISTRESS_EXAMPLES = [
    "Company X files for Chapter 11 bankruptcy protection amid mounting debt",
    "Credit rating agency downgrades firm to junk status on covenant breach",
    "Major retailer announces restructuring plan, closing 200 stores",
    "Auditor flags going-concern warning in quarterly filing",
    "Tech startup suspends dividend after reporting $2B writedown",
    "Investment firm faces severe liquidity crunch after major portfolio loss"
]

# LLM Prompt Templates
GENERATION_PROMPT_DISTRESS = """You are a financial news headline writer. Generate {{n}} realistic financial news headlines that describe corporate financial distress situations.

REQUIREMENTS:
- Each headline should describe a DIFFERENT type of distress event from the following: bankruptcy, credit downgrade, restructuring, layoffs, dividend cuts, debt defaults, going-concern warnings, asset writedowns, liquidity crises, covenant breaches.
- Vary the company type (bank, retailer, tech, energy, pharma, etc.)
- Use REALISTIC but FICTIONAL company names (not real companies)
- Vary length between 8 and 25 words
- Use professional financial news language
- Include specific numbers or details where appropriate where relevant.

EXAMPLES of real distress headlines:
{{examples}}

Generate {{n}} NEW headlines in this style. Return as a JSON array of objects with "text" and "distress_type" fields."""

GENERATION_PROMPT_NON_DISTRESS = """You are a financial news headline writer. Generate {{n}} realistic financial news headlines about POSITIVE corporate events: earnings beats, product launches, expansion, hiring, partnerships. Use fictional company names. Return JSON array of objects with "text" and "distress_type" fields."""

# --- Mock OpenAI Client for Demonstration ---
class MockChatCompletion:
    def __init__(self, content):
        self.content = content

class MockChoice:
    def __init__(self, content):
        self.message = MockChatCompletion(content)

class MockCompletionResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]

class MockOpenAI:
    def chat(self):
        class completions:
            def create(self, model, messages, temperature, max_tokens, response_format):
                prompt_content = messages[0]["content"]
                match = re.search(r"Generate (\d+) realistic financial news headlines", prompt_content)
                requested_n = int(match.group(1)) if match else 1

                if "distress" in prompt_content.lower():
                    synthetic_headlines_distress = [
                        {"text": "Global Tech Corp faces bankruptcy filing after massive Q3 losses.", "distress_type": "bankruptcy"},
                        {"text": "Credit rating for OmniBank downgraded to 'junk' due to liquidity concerns.", "distress_type": "credit downgrade"},
                        {"text": "Retail chain 'FashionForward' announces widespread store closures and restructuring plan.", "distress_type": "restructuring"},
                        {"text": "Energy giant PetroDynamics to lay off 1,500 employees amidst falling oil prices.", "distress_type": "layoffs"},
                        {"text": "BioPharm Innovations suspends dividend payouts to conserve capital.", "distress_type": "dividend cuts"},
                        {"text": "Property developer 'UrbanHomes' defaults on $200M bond payment.", "distress_type": "debt defaults"},
                        {"text": "Accounting firm flags 'going-concern warning' for manufacturing company.", "distress_type": "going-concern warning"},
                        {"text": "Tech startup 'InnovateX' reports significant asset writedowns due to failed product launch.", "distress_type": "asset writedowns"},
                        {"text": "Financial services firm faces severe liquidity crisis after bond market volatility.", "distress_type": "liquidity crises"},
                        {"text": "Software company breaches key loan covenants, triggering default clauses.", "distress_type": "covenant breaches"},
                        {"text": "Auto manufacturer reports 30% drop in sales, considering plant closures.", "distress_type": "plant closures"},
                        {"text": "Mining conglomerate 'EarthExtract' issues profit warning amid commodity price slump.", "distress_type": "profit warning"},
                        {"text": "Airline 'SkyLink' announces emergency capital raise, shares plummet.", "distress_type": "emergency capital raise"},
                        {"text": "Hospital group 'HealthFirst' struggles with mounting debt, seeks debt relief.", "distress_type": "debt relief"},
                        {"text": "Retailer 'TrendBoutique' warns of challenging holiday season, potential insolvency.", "distress_type": "insolvency"},
                        {"text": "Telecom provider 'ConnectNet' posts unexpected losses, stock drops sharply.", "distress_type": "unexpected losses"},
                        {"text": "Real estate investment trust (REIT) halts property acquisitions due to market downturn.", "distress_type": "market downturn"},
                        {"text": "Logistics firm 'SwiftFreight' revises earnings guidance downwards amid rising fuel costs.", "distress_type": "earnings revision"},
                        {"text": "Food producer 'AgriHarvest' announces significant inventory writedowns.", "distress_type": "inventory writedowns"},
                        {"text": "Luxury brand 'Elegance' reports declining sales in key markets, plans cost cuts.", "distress_type": "cost cuts"},
                        {"text": "Regional bank 'CommunityTrust' faces increased loan defaults, profit outlook grim.", "distress_type": "loan defaults"},
                        {"text": "Pharma company 'MediCure' fails clinical trials, stock plunges 70%.", "distress_type": "failed trials"},
                        {"text": "Media conglomerate 'GlobalNews' divests non-core assets to reduce debt load.", "distress_type": "asset divestment"},
                        {"text": "Construction firm 'BuildWell' issues profit warning citing supply chain disruptions.", "distress_type": "supply chain issues"},
                        {"text": "Travel agency 'Wanderlust' sees bookings halve, explores strategic options.", "distress_type": "strategic options"},
                        {"text": "Sports retailer 'ActiveGear' enters administration after failed turnaround.", "distress_type": "administration"},
                        {"text": "Publishing house 'LiteraryPress' declares insolvency, seeks buyers.", "distress_type": "insolvency"},
                        {"text": "Car rental company 'DriveEasy' faces massive debt, bankruptcy looms.", "distress_type": "bankruptcy"},
                        {"text": "Semiconductor manufacturer 'ChipTech' warns of severe demand slump.", "distress_type": "demand slump"},
                        {"text": "Hotel group 'GrandStay' defaults on bond interest payments.", "distress_type": "debt defaults"},
                        {"text": "Fashion retailer 'ChicStyles' liquidates inventory, plans full closure.", "distress_type": "restructuring"},
                        {"text": "Online retailer 'E-Bazaar' announces major workforce reduction.", "distress_type": "layoffs"},
                        {"text": "Chemicals producer 'ChemCorp' suspends quarterly dividend.", "distress_type": "dividend cuts"},
                        {"text": "Regional airline 'AeroLink' files for Chapter 11 protection.", "distress_type": "bankruptcy"},
                        {"text": "Private equity firm flags significant writedown on portfolio company.", "distress_type": "asset writedowns"},
                        {"text": "Food delivery service 'QuickBite' faces intense competition, warns of losses.", "distress_type": "going-concern warning"},
                        {"text": "Infrastructure fund halts new projects due to rising interest rates.", "distress_type": "liquidity crises"},
                        {"text": "Biotech startup fails to secure Series B funding, future uncertain.", "distress_type": "financial uncertainty"},
                        {"text": "Metal producer 'SteelWorks' breaches debt covenants, seeks waivers.", "distress_type": "covenant breaches"},
                        {"text": "Investment bank 'GlobalCapital' reports unexpected trading losses.", "distress_type": "unexpected losses"},
                        {"text": "Luxury automaker 'PrestigeMotors' recalls thousands of vehicles, financial impact unclear.", "distress_type": "product recall"},
                        {"text": "Cloud software provider 'DataFlow' downgraded by Moody's.", "distress_type": "credit downgrade"},
                        {"text": "Logistics provider 'SwiftShip' announces deep cost-cutting measures.", "distress_type": "cost cuts"},
                        {"text": "Home builder 'VistaHomes' reports sharp decline in new orders.", "distress_type": "demand slump"},
                        {"text": "Healthcare provider 'CarePath' struggles with rising operational costs.", "distress_type": "operational costs"},
                        {"text": "Textile manufacturer 'FabriCorp' faces insolvency due to import competition.", "distress_type": "insolvency"},
                        {"text": "Education tech firm 'LearnPro' lays off a third of its staff.", "distress_type": "layoffs"},
                        {"text": "Utility company 'PowerGrid' issues profit warning after regulatory fine.", "distress_type": "regulatory fine"},
                        {"text": "Food chain 'HealthyBites' files for Chapter 7 liquidation.", "distress_type": "bankruptcy"},
                        {"text": "Advertising agency 'CreativeSpark' announces strategic review.", "distress_type": "restructuring"}
                    ]
                    headlines_to_return = synthetic_headlines_distress[:requested_n]
                    return MockCompletionResponse(json.dumps({"headlines": headlines_to_return}))
                else:
                    synthetic_headlines_non_distress = [
                        {"text": "Tech startup 'InnovateX' announces successful Series C funding round.", "distress_type": "funding success"},
                        {"text": "Global Pharma Inc. reports record Q1 earnings, exceeding analyst expectations.", "distress_type": "earnings beat"},
                        {"text": "Retail giant 'FashionForward' expands into new international markets.", "distress_type": "market expansion"},
                        {"text": "Sustainable energy firm unveils groundbreaking new solar technology.", "distress_type": "product launch"},
                        {"text": "Bank 'CommunityTrust' acquires competitor, strengthens regional presence.", "distress_type": "acquisition"},
                        {"text": "Manufacturing company achieves new production efficiency milestones.", "distress_type": "operational improvement"},
                        {"text": "Software company 'DataFlow' forms strategic partnership with major cloud provider.", "distress_type": "partnership"},
                        {"text": "Automaker 'PrestigeMotors' reports surging sales of electric vehicles.", "distress_type": "sales growth"},
                        {"text": "Biotech firm 'MediCure' receives FDA approval for novel drug.", "distress_type": "regulatory approval"},
                        {"text": "Logistics firm 'SwiftFreight' announces hiring spree to meet demand.", "distress_type": "hiring"}
                    ]
                    headlines_to_return = synthetic_headlines_non_distress[:requested_n]
                    return MockCompletionResponse(json.dumps({"headlines": headlines_to_return}))
        return completions()

# Initialize the mock OpenAI client
mock_openai_client = MockOpenAI()

# --- Data Loading and Preprocessing Functions ---

def load_and_label_financial_data(keywords_list):
    """
    Loads a mocked Financial PhraseBank dataset and labels headlines as 'distress' or 'non-distress'
    based on a provided list of keywords.
    """
    mock_data = """Global Tech Corp reports record profits; positive
Markets rally on strong economic data; positive
Retail chain 'FashionForward' expands into new regions; positive
BioPharm Innovations secures massive funding round; positive
Energy giant PetroDynamics announces new green initiative; positive
Company X files for Chapter 11 bankruptcy protection amid mounting debt; negative
Credit rating agency downgrades firm to junk status on covenant breach; negative
Major retailer announces restructuring plan, closing 200 stores; negative
Auditor flags going-concern warning in quarterly filing; negative
Tech startup suspends dividend after reporting $2B writedown; negative
Investment firm faces severe liquidity crunch after major portfolio loss; negative
ABC Corp announces strategic acquisition; neutral
Government announces new regulations, market reaction mixed; neutral
Economy outlook remains uncertain for next quarter; neutral
XYZ Inc. reports stable earnings, meeting expectations; neutral
Manufacturing sector shows slight growth; neutral
"""
    data_lines = mock_data.strip().split('\n')
    data_records = []
    for line in data_lines:
        text, label_str = line.rsplit(';', 1)
        data_records.append({'text': text.strip(), 'raw_label': label_str.strip()})

    fpb_df = pd.DataFrame(data_records)

    label_map_str_to_int = {'negative': 0, 'neutral': 1, 'positive': 2}
    fpb_df['label'] = fpb_df['raw_label'].map(label_map_str_to_int)
    fpb_df['sentiment'] = fpb_df['raw_label']

    def is_distress(text):
        text_lower = text.lower()
        return any(kw in text_lower for kw in keywords_list)

    fpb_df['distress'] = fpb_df['text'].apply(is_distress).astype(int)
    return fpb_df

def generate_headlines(llm_client, n_headlines, prompt_template, examples=None, batch_size=50):
    """
    Generates synthetic headlines using an LLM in batches.

    Args:
        llm_client: The LLM client instance (e.g., MockOpenAI() or openai.OpenAI()).
        n_headlines (int): The total number of headlines to generate.
        prompt_template (str): The template for the LLM prompt.
        examples (list, optional): Few-shot examples to include in the prompt. Defaults to None.
        batch_size (int): Number of headlines to generate per LLM call.

    Returns:
        pd.DataFrame: DataFrame containing generated headlines with 'text' and 'distress_type'.
    """
    all_headlines = []
    examples_text = "\n".join([f'- "{h}"' for h in examples]) if examples else ""

    for batch_start in range(0, n_headlines, batch_size):
        batch_n = min(batch_size, n_headlines - batch_start)
        prompt = prompt_template.format(
            n=batch_n,
            examples=examples_text
        )

        try:
            response = llm_client.chat().create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            raw_content = response.choices[0].message.content

            batch = json.loads(raw_content)
            if isinstance(batch, dict):
                if 'headlines' in batch:
                    batch = batch['headlines']
                elif 'data' in batch:
                    batch = batch['data']
                else:
                    batch = []

            all_headlines.extend(batch)

        except Exception as e:
            print(f"!!! CRITICAL ERROR IN generate_headlines: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue

    synth_df = pd.DataFrame(all_headlines)
    return synth_df

def quality_filter(synth_df, real_distress_df, sbert_model, finbert_pipeline, min_similarity=0.3, max_similarity=0.95, min_length=5, max_length=30):
    """
    Applies a three-stage quality filter to synthetic headlines.

    Args:
        synth_df (pd.DataFrame): DataFrame of synthetic headlines.
        real_distress_df (pd.DataFrame): DataFrame of real distress headlines for similarity comparison.
        sbert_model (SentenceTransformer): Initialized Sentence-BERT model.
        finbert_pipeline (transformers.pipeline): Initialized FinBERT sentiment analysis pipeline.
        min_similarity (float): Minimum cosine similarity to real headlines to be retained.
        max_similarity (float): Maximum cosine similarity to real headlines to be retained.
        min_length (int): Minimum word length for a headline.
        max_length (int): Maximum word length for a headline.

    Returns:
        pd.DataFrame: Filtered synthetic headlines.
        dict: A dictionary containing retention counts at each stage.
        np.array: All raw maximum similarity scores for histogram visualization.
    """
    original_n = len(synth_df)
    retention_counts = {'original': original_n}

    synth_df_filtered = synth_df[
        (synth_df['text'].str.split().str.len() >= min_length) &
        (synth_df['text'].str.split().str.len() <= max_length)
    ].copy()
    synth_df_filtered.drop_duplicates(subset='text', inplace=True)
    n_after_basic = len(synth_df_filtered)
    retention_counts['after_basic'] = n_after_basic

    all_raw_max_sims = np.array([])
    if not synth_df.empty and not real_distress_df.empty:
        real_distress_texts = real_distress_df['text'].tolist()
        real_embeddings = sbert_model.encode(real_distress_texts, show_progress_bar=False)
        all_synth_embeddings = sbert_model.encode(synth_df['text'].tolist(), show_progress_bar=False)
        all_raw_max_sims = cosine_similarity(all_synth_embeddings, real_embeddings).max(axis=1)

    if not synth_df_filtered.empty and not real_distress_df.empty:
        real_distress_texts_filtered = real_distress_df['text'].tolist()
        real_embeddings_filtered = sbert_model.encode(real_distress_texts_filtered, show_progress_bar=False)
        synth_embeddings_for_sim_filter = sbert_model.encode(synth_df_filtered['text'].tolist(), show_progress_bar=False)

        sim_matrix = cosine_similarity(synth_embeddings_for_sim_filter, real_embeddings_filtered)
        max_sims = sim_matrix.max(axis=1)

        sim_mask = (max_sims >= min_similarity) & (max_sims <= max_similarity)
        synth_df_filtered = synth_df_filtered[sim_mask].copy()
        synth_df_filtered['max_real_similarity'] = max_sims[sim_mask]
    n_after_sim = len(synth_df_filtered)
    retention_counts['after_similarity'] = n_after_sim

    if not synth_df_filtered.empty:
        fb_results = finbert_pipeline(synth_df_filtered['text'].tolist())
        synth_df_filtered['finbert_label'] = [r['label'].lower() for r in fb_results]
        synth_df_filtered['finbert_conf'] = [r['score'] for r in fb_results]

        consistent_mask = (synth_df_filtered['finbert_label'] == 'negative') & \
                          (synth_df_filtered['finbert_conf'] > 0.5)
        synth_df_filtered = synth_df_filtered[consistent_mask].copy()
    n_after_fb = len(synth_df_filtered)
    retention_counts['after_finbert'] = n_after_fb

    return synth_df_filtered, retention_counts, all_raw_max_sims

# --- Model Training and Evaluation Functions ---

def train_and_evaluate_classifier(X_train_texts, y_train, X_test_texts, y_test, vectorizer, model_name="Model"):
    """
    Trains a TF-IDF Logistic Regression classifier and evaluates its performance.

    Args:
        X_train_texts (pd.Series): Training text data.
        y_train (pd.Series): Training labels.
        X_test_texts (pd.Series): Test text data.
        y_test (pd.Series): Test labels.
        vectorizer (TfidfVectorizer): Initialized TF-IDF vectorizer.
        model_name (str): Name for the model (e.g., "Real Only", "Augmented").

    Returns:
        dict: Dictionary containing F1, Recall, Precision for distress class.
        sklearn.linear_model.LogisticRegression: Trained model.
        TfidfVectorizer: Fitted vectorizer.
    """
    X_train_vec = vectorizer.fit_transform(X_train_texts)
    X_test_vec = vectorizer.transform(X_test_texts)

    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)

    print(f"\n--- {model_name} Results ---")
    print(classification_report(y_test, y_pred, target_names=['non-distress', 'distress'], zero_division=0))
    print(f"Distress F1 Score: {f1:.4f}")
    print(f"Distress Recall: {recall:.4f}")
    print(f"Distress Precision: {precision:.4f}")

    metrics = {'f1': f1, 'recall': recall, 'precision': precision}
    return metrics, model, vectorizer

# --- Diversity Metrics Functions ---

def calculate_ttr(texts):
    """Calculates Type-Token Ratio for a list of texts."""
    tokens = ' '.join(texts).lower().split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)

def calculate_semantic_diversity(texts, embedding_model):
    """Calculates semantic diversity using average cosine distance between embeddings."""
    if len(texts) < 2:
        return 0.0
    embeddings = embedding_model.encode(texts, show_progress_bar=False)
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, 0)
    distances = 1 - sim_matrix

    upper_triangle_indices = np.triu_indices_from(distances, k=1)
    if upper_triangle_indices[0].size == 0:
        return 0.0

    return distances[upper_triangle_indices].mean()

# --- Visualization Functions ---

def plot_filter_funnel(filter_counts):
    """Visualizes the quality filter funnel."""
    plt.figure(figsize=(10, 6))
    stages = ['Original', 'Basic Filters', 'Embedding Similarity', 'FinBERT Consistency']
    counts = [filter_counts['original'], filter_counts['after_basic'], filter_counts['after_similarity'], filter_counts['after_finbert']]
    sns.barplot(x=stages, y=counts, palette='viridis')
    plt.title('Quality Filter Funnel: Synthetic Distress Headlines Retention')
    plt.xlabel('Filtering Stage')
    plt.ylabel('Number of Headlines Retained')
    for i, count in enumerate(counts):
        plt.text(i, count + 10, str(count), ha='center', va='bottom')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_similarity_distribution(all_raw_max_sims, min_similarity=0.3, max_similarity=0.95):
    """Visualizes the distribution of maximum cosine similarity."""
    plt.figure(figsize=(10, 6))
    sns.histplot(all_raw_max_sims, bins=50, kde=True, color='skyblue')
    plt.axvline(x=min_similarity, color='r', linestyle='--', label=f'Min Similarity ({min_similarity})')
    plt.axvline(x=max_similarity, color='purple', linestyle='--', label=f'Max Similarity ({max_similarity})')
    plt.fill_betweenx([0, plt.gca().get_ylim()[1]], min_similarity, max_similarity, color='green', alpha=0.1, label='Goldilocks Zone')
    plt.title('Distribution of Maximum Cosine Similarity for Raw Synthetic Headlines', fontsize=14)
    plt.xlabel('Max Cosine Similarity to Real Distress Headlines')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_performance_comparison(metrics_a, metrics_b):
    """Compares model performance (F1, Recall, Precision) before and after augmentation."""
    metrics_df = pd.DataFrame({
        'Metric': ['F1 Score', 'Recall', 'Precision'],
        'Real Only': [metrics_a['f1'], metrics_a['recall'], metrics_a['precision']],
        'Augmented': [metrics_b['f1'], metrics_b['recall'], metrics_b['precision']]
    })

    metrics_df_melted = metrics_df.melt(id_vars='Metric', var_name='Dataset', value_name='Score')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Metric', y='Score', hue='Dataset', data=metrics_df_melted, palette='viridis')
    plt.title('Distress Classifier Performance: Real Only vs. Augmented Data', fontsize=14)
    plt.ylabel('Score')
    plt.ylim(0, 1)
    for container in plt.gca().containers:
        plt.bar_label(container, fmt='%.2f', label_type='edge', padding=3)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_augmentation_sensitivity(augmentation_ratios, f1_scores_sensitivity, recall_scores_sensitivity, precision_scores_sensitivity):
    """Plots model performance metrics against varying amounts of synthetic data."""
    plt.figure(figsize=(12, 7))
    plt.plot(augmentation_ratios, f1_scores_sensitivity, 'o-', color='steelblue', linewidth=2, label='Distress F1 Score')
    plt.plot(augmentation_ratios, recall_scores_sensitivity, 'x--', color='darkgreen', linewidth=1, label='Distress Recall Score')
    plt.plot(augmentation_ratios, precision_scores_sensitivity, '^:', color='darkred', linewidth=1, label='Distress Precision Score')

    plt.axhline(y=f1_scores_sensitivity[0], color='gray', linestyle='--', label=f'Baseline F1: {f1_scores_sensitivity[0]:.3f}')
    plt.axhline(y=recall_scores_sensitivity[0], color='lightgray', linestyle='--', label=f'Baseline Recall: {recall_scores_sensitivity[0]:.3f}')

    plt.xlabel('Number of Synthetic Distress Headlines Added', fontsize=12)
    plt.ylabel('Distress Class Metric Score (on real test set)', fontsize=12)
    plt.title('Augmentation Sensitivity: Model Performance vs. Synthetic Data Amount', fontsize=14)
    plt.xticks(augmentation_ratios, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_embedding_scatter(real_df, filtered_synth_distress_df, synth_non_distress_df, sbert_model, random_state, num_tsne_samples=100):
    """
    Generates a t-SNE scatter plot to visualize headline embeddings in 2D space.
    """
    tsne_real_distress_texts = real_df[real_df['distress'] == 1]['text'].sample(min(num_tsne_samples, real_df[real_df['distress'] == 1].shape[0]), random_state=random_state).tolist()
    tsne_real_nondistress_texts = real_df[real_df['distress'] == 0]['text'].sample(min(num_tsne_samples, real_df[real_df['distress'] == 0].shape[0]), random_state=random_state).tolist()
    tsne_synthetic_distress_texts = filtered_synth_distress_df['text'].sample(min(num_tsne_samples, len(filtered_synth_distress_df)), random_state=random_state).tolist()
    tsne_synthetic_nondistress_texts = synth_non_distress_df['text'].sample(min(num_tsne_samples, len(synth_non_distress_df)), random_state=random_state).tolist()

    all_tsne_texts = tsne_real_distress_texts + tsne_real_nondistress_texts + tsne_synthetic_distress_texts + tsne_synthetic_nondistress_texts
    labels_tsne = (['Real Distress'] * len(tsne_real_distress_texts) +
                   ['Real Non-Distress'] * len(tsne_real_nondistress_texts) +
                   ['Synthetic Distress'] * len(tsne_synthetic_distress_texts) +
                   ['Synthetic Non-Distress'] * len(tsne_synthetic_nondistress_texts))

    if len(all_tsne_texts) > 1:
        all_tsne_embeddings = sbert_model.encode(all_tsne_texts, show_progress_bar=False)

        # Perplexity must be less than n_samples
        perplexity_val = min(30, len(all_tsne_texts) - 1)
        if perplexity_val < 1: # TSNE requires perplexity >= 1 (and > 0 if len(data) > 1)
             print("Not enough data to generate t-SNE plot (need at least 2 samples combined, and perplexity >= 1).")
             return

        tsne_model = TSNE(n_components=2, random_state=random_state, perplexity=perplexity_val)
        tsne_results = tsne_model.fit_transform(all_tsne_embeddings)

        tsne_df = pd.DataFrame(data=tsne_results, columns=['TSNE-1', 'TSNE-2'])
        tsne_df['Label'] = labels_tsne

        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x="TSNE-1", y="TSNE-2",
            hue="Label",
            palette={'Real Distress': 'red', 'Real Non-Distress': 'blue',
                     'Synthetic Distress': 'lightcoral', 'Synthetic Non-Distress': 'skyblue'},
            data=tsne_df,
            legend="full",
            alpha=0.7
        )
        plt.title('Embedding Space Scatter Plot (t-SNE) of Headlines', fontsize=14)
        plt.xlabel('TSNE-1')
        plt.ylabel('TSNE-2')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
    else:
        print("Not enough data to generate t-SNE plot (need at least 2 samples combined).")


def plot_distress_type_distribution(filtered_synth_distress_df):
    """Visualizes the distribution of specific distress types in synthetic data."""
    synth_distress_type_counts = filtered_synth_distress_df['distress_type'].value_counts()

    if not synth_distress_type_counts.empty:
        plt.figure(figsize=(12, 7))
        sns.barplot(x=synth_distress_type_counts.index, y=synth_distress_type_counts.values, palette='plasma')
        plt.title('Distribution of Specific Distress Types in Filtered Synthetic Data', fontsize=14)
        plt.xlabel('Distress Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
    else:
        print("No filtered synthetic distress headlines to plot distress type distribution.")

def print_ethical_considerations():
    """Prints ethical guardrails and best practices."""
    print("\n--- Ethical Guardrails and Best Practices for Synthetic Financial Text ---")
    print("As a CFA Charterholder, Alex understands the non-negotiable ethical considerations for using synthetic financial text:")
    print("1.  **Never publish or distribute:** Synthetic financial headlines must be used exclusively for internal model training. Publishing them, even accidentally, could constitute market manipulation or mislead investors if mistaken for real news.")
    print("2.  **Label clearly:** Every synthetic sample must be explicitly tagged (e.g., with a `source='synthetic'` flag) that persists through all downstream processing. If a trained model is audited, the proportion of synthetic training data must be disclosable.")
    print("3.  **Audit for bias:** LLMs can inherit and amplify biases. Alex must audit the distress-type distributions (as shown in the plot above) and other characteristics (e.g., industry, geography) to ensure the LLM does not disproportionately generate distress headlines about certain industries or demographics, which could lead to biased models and unfair credit assessments.")
    print("4.  **LLM Provider Terms of Service:** Be mindful of LLM providers' terms. For example, some prohibit using generated data to train competing models. Explore open-source alternatives like Llama or Mistral for greater control and flexibility.")
    print("5.  **Model Collapse Risk:** Be aware of 'model collapse', a phenomenon where models trained on synthetic data generated by other models (especially without sufficient real data grounding) gradually lose diversity and quality. Prevention includes always including a substantial proportion of real data, maintaining real-data-majority training sets, and tracking the provenance of all training data.")

# --- Main Orchestration Function ---

def run_data_augmentation_pipeline(
    n_synth_distress_target=500,
    n_synth_non_distress_target=100,
    augmentation_ratios=[0, 2, 4, 6, 8, 10], # Multiples of real distress samples in train set
    llm_client=mock_openai_client,
    random_state=RANDOM_STATE
):
    """
    Orchestrates the entire data augmentation pipeline for financial distress detection.

    Args:
        n_synth_distress_target (int): Target number of raw synthetic distress headlines to generate.
        n_synth_non_distress_target (int): Target number of raw synthetic non-distress headlines to generate.
        augmentation_ratios (list): Multiples of real distress samples to use for sensitivity analysis.
        llm_client: The LLM client instance (e.g., MockOpenAI() or openai.OpenAI()).
        random_state (int): Random state for reproducibility.
    """
    print("--- Starting Financial Distress Data Augmentation Pipeline ---")

    # 1. Initialize models
    print("\n1. Initializing Embedding and Sentiment Models...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    finbert_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", truncation=True, max_length=128)

    # 2. Load and Label Real Data
    print("\n2. Loading and Labeling Real Financial Data...")
    real_df = load_and_label_financial_data(DISTRESS_KEYWORDS)
    print(f"Total real headlines: {len(real_df):,}")
    print(f"Real Distress: {real_df['distress'].sum():,} ({real_df['distress'].mean():.1%})")

    # 3. Generate Synthetic Data
    print("\n3. Generating Synthetic Headlines using LLM...")
    synth_distress_raw_df = generate_headlines(llm_client, n_synth_distress_target, GENERATION_PROMPT_DISTRESS, REAL_DISTRESS_EXAMPLES)
    synth_distress_raw_df['distress'] = 1
    synth_distress_raw_df['source'] = 'synthetic'

    synth_non_distress_df = generate_headlines(llm_client, n_synth_non_distress_target, GENERATION_PROMPT_NON_DISTRESS)
    synth_non_distress_df['distress'] = 0
    synth_non_distress_df['source'] = 'synthetic'

    print(f"Generated {len(synth_distress_raw_df)} raw synthetic distress headlines.")
    print(f"Generated {len(synth_non_distress_df)} raw synthetic non-distress headlines.")

    # 4. Filter Synthetic Distress Data
    print("\n4. Applying Quality Filter to Synthetic Distress Headlines...")
    filtered_synth_distress_df, filter_counts, all_raw_max_sims = quality_filter(
        synth_distress_raw_df,
        real_df[real_df['distress'] == 1], # Compare synthetic distress to real distress only
        sbert_model,
        finbert_pipeline,
        min_similarity=0.3,
        max_similarity=0.95,
        min_length=5,
        max_length=30
    )
    print("--- Quality Filter Results ---")
    print(f"Input: {filter_counts['original']} synthetic headlines")
    print(f"After basic (length/dedup): {filter_counts['after_basic']} retained")
    print(f"After similarity filter: {filter_counts['after_similarity']} retained")
    print(f"After FinBERT consistency: {filter_counts['after_finbert']} retained")
    print(f"Final Retention rate: {filter_counts['after_finbert'] / filter_counts['original']:.1%}")

    # 5. Prepare Datasets for Model Training
    print("\n5. Preparing Training and Test Datasets...")
    train_df, test_df = train_test_split(real_df, test_size=0.2, random_state=random_state, stratify=real_df['distress'])
    print(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")

    # 6. Train and Evaluate Models (Baseline vs. Augmented)
    print("\n6. Training and Evaluating Models...")
    # Model A: Real data only
    tfidf_vectorizer_a = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    metrics_a, model_a, _ = train_and_evaluate_classifier(
        train_df['text'], train_df['distress'],
        test_df['text'], test_df['distress'],
        tfidf_vectorizer_a, "Model A (Real Data Only)"
    )

    # Model B: Real + Filtered Synthetic Data
    augmented_train_df = pd.concat([
        train_df[['text', 'distress']],
        filtered_synth_distress_df[['text', 'distress']]
    ], ignore_index=True)
    tfidf_vectorizer_b = TfidfVectorizer(max_features=5000, ngram_range=(1, 2)) # Re-initialize for augmented data
    metrics_b, model_b, _ = train_and_evaluate_classifier(
        augmented_train_df['text'], augmented_train_df['distress'],
        test_df['text'], test_df['distress'],
        tfidf_vectorizer_b, "Model B (Real + Filtered Synthetic)"
    )

    print("\n--- Augmentation Lift Summary (Distress Class) ---")
    f1_lift = metrics_b['f1'] - metrics_a['f1']
    recall_lift = metrics_b['recall'] - metrics_a['recall']
    precision_lift = metrics_b['precision'] - metrics_a['precision']

    print(f"F1 Score Lift: {f1_lift:.4f} ({f1_lift / metrics_a['f1'] * 100:+.1f}%)" if metrics_a['f1'] else f"F1 Score Lift: {f1_lift:.4f} (baseline 0)")
    print(f"Recall Lift: {recall_lift:.4f} ({recall_lift / metrics_a['recall'] * 100:+.1f}%)" if metrics_a['recall'] else f"Recall Lift: {recall_lift:.4f} (baseline 0)")
    print(f"Precision Lift: {precision_lift:.4f} ({precision_lift / metrics_a['precision'] * 100:+.1f}%)" if metrics_a['precision'] else f"Precision Lift: {precision_lift:.4f} (baseline 0)")

    # 7. Run Augmentation Sensitivity Analysis
    print("\n7. Running Augmentation Sensitivity Analysis...")
    f1_scores_sensitivity = []
    recall_scores_sensitivity = []
    precision_scores_sensitivity = []
    n_real_distress = train_df[train_df['distress'] == 1].shape[0]

    for ratio in augmentation_ratios:
        n_synth_to_add = int(n_real_distress * ratio)
        if n_synth_to_add == 0:
            current_augmented_train_df = train_df[['text', 'distress']]
        else:
            synth_sample = filtered_synth_distress_df.sample(
                n=min(n_synth_to_add, len(filtered_synth_distress_df)),
                random_state=random_state,
                replace=True if n_synth_to_add > len(filtered_synth_distress_df) else False
            )[['text', 'distress']]
            current_augmented_train_df = pd.concat([train_df[['text', 'distress']], synth_sample], ignore_index=True)

        tfidf_vectorizer_ratio = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        sensitivity_metrics, _, _ = train_and_evaluate_classifier(
            current_augmented_train_df['text'], current_augmented_train_df['distress'],
            test_df['text'], test_df['distress'],
            tfidf_vectorizer_ratio, f"Model (Augmentation Ratio {ratio})"
        )
        f1_scores_sensitivity.append(sensitivity_metrics['f1'])
        recall_scores_sensitivity.append(sensitivity_metrics['recall'])
        precision_scores_sensitivity.append(sensitivity_metrics['precision'])

    # 8. Calculate Diversity Metrics
    print("\n8. Calculating Diversity Metrics...")
    ttr_real = calculate_ttr(real_df['text'].tolist())
    ttr_synthetic = calculate_ttr(filtered_synth_distress_df['text'].tolist())
    ttr_synth_non_distress = calculate_ttr(synth_non_distress_df['text'].tolist())

    print(f"Type-Token Ratio (TTR) - Real Data: {ttr_real:.4f}")
    print(f"Type-Token Ratio (TTR) - Filtered Synthetic Distress: {ttr_synthetic:.4f}")
    print(f"Type-Token Ratio (TTR) - Synthetic Non-Distress: {ttr_synth_non_distress:.4f}")

    semantic_diversity_real = calculate_semantic_diversity(real_df['text'].tolist(), sbert_model)
    semantic_diversity_synthetic = calculate_semantic_diversity(filtered_synth_distress_df['text'].tolist(), sbert_model)
    semantic_diversity_synth_non_distress = calculate_semantic_diversity(synth_non_distress_df['text'].tolist(), sbert_model)

    print(f"\nSemantic Diversity - Real Data: {semantic_diversity_real:.4f}")
    print(f"Semantic Diversity - Filtered Synthetic Distress: {semantic_diversity_synthetic:.4f}")
    print(f"Semantic Diversity - Synthetic Non-Distress: {semantic_diversity_synth_non_distress:.4f}")

    # 9. Generate Visualizations
    print("\n9. Generating Visualizations...")
    plot_filter_funnel(filter_counts)
    plot_similarity_distribution(all_raw_max_sims)
    plot_performance_comparison(metrics_a, metrics_b)
    plot_augmentation_sensitivity(augmentation_ratios, f1_scores_sensitivity, recall_scores_sensitivity, precision_scores_sensitivity)
    plot_embedding_scatter(real_df, filtered_synth_distress_df, synth_non_distress_df, sbert_model, random_state)
    plot_distress_type_distribution(filtered_synth_distress_df)

    # 10. Print Ethical Considerations
    print_ethical_considerations()

    print("\n--- Pipeline Execution Complete ---")

# This block ensures that `run_data_augmentation_pipeline()` is called only when the script is executed directly,
# not when imported as a module.
if __name__ == "__main__":
    run_data_augmentation_pipeline(
        n_synth_distress_target=100, # Reduced for faster execution in a demo
        n_synth_non_distress_target=20, # Reduced for faster execution in a demo
        augmentation_ratios=[0, 1, 2], # Reduced ratios for faster execution
        llm_client=mock_openai_client, # Use the mock client
        random_state=RANDOM_STATE
    )
