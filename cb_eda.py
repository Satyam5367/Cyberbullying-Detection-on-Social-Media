"""
EDA — Cyberbullying Detection on Social Media
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os, sys, string
from collections import Counter

sys.path.insert(0, '.')
from cb_main import CyberbullyingDataGenerator, TextPreprocessor, extract_text_features

OUTPUT_DIR = 'plots/eda'
os.makedirs(OUTPUT_DIR, exist_ok=True)

PAL = {'bg':'#0a0e1a','panel':'#111827','a1':'#7c3aed','a2':'#f43f5e',
       'a3':'#f59e0b','a4':'#10b981','a5':'#3b82f6','txt':'#f1f5f9','grid':'#1e293b'}
CAT_COLORS = ['#10b981','#f43f5e','#f59e0b','#7c3aed','#3b82f6','#ec4899','#14b8a6']

def _theme():
    plt.rcParams.update({
        'figure.facecolor':PAL['bg'],'axes.facecolor':PAL['panel'],
        'axes.edgecolor':PAL['grid'],'axes.labelcolor':PAL['txt'],
        'xtick.color':PAL['txt'],'ytick.color':PAL['txt'],
        'text.color':PAL['txt'],'grid.color':PAL['grid'],'font.family':'monospace'
    })

def _save(name):
    p = os.path.join(OUTPUT_DIR, name)
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=PAL['bg'])
    plt.close(); print(f"  Saved: {p}"); return p

# ── EDA 1: Category heatmap ───────────────────────────────────────
def plot_category_stats(df):
    _theme()
    stats = df.groupby('label')[['text_length','word_count','caps_ratio',
                                  'exclamation_count','punct_count']].mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Mean Feature Values per Category (Heatmap)',
                 fontsize=14, color=PAL['a1'], weight='bold')
    norm = (stats - stats.min()) / (stats.max() - stats.min() + 1e-9)
    sns.heatmap(norm, ax=ax, cmap='RdPu', annot=stats.round(2),
                fmt='.2f', linewidths=0.5, linecolor=PAL['bg'],
                cbar_kws={'shrink': 0.8})
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    return _save('eda_01_category_stats_heatmap.png')

# ── EDA 2: Text length by category ───────────────────────────────
def plot_text_length_violin(df):
    _theme()
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle('Text Length Distribution by Category (Violin)',
                 fontsize=14, color=PAL['a1'], weight='bold')
    palette = {cat: col for cat, col in zip(df['label'].unique(), CAT_COLORS)}
    df_clip = df.copy()
    df_clip['text_length'] = df_clip['text_length'].clip(upper=df_clip['text_length'].quantile(0.97))
    sns.violinplot(data=df_clip, x='label', y='text_length',
                   palette=palette, ax=ax, inner='box', linewidth=0.8)
    ax.set_xlabel('Category'); ax.set_ylabel('Text Length (chars)')
    ax.tick_params(axis='x', rotation=15); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return _save('eda_02_text_length_violin.png')

# ── EDA 3: Bigram analysis ────────────────────────────────────────
def plot_bigrams(df):
    _theme()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Top 15 Bigrams: Normal vs Cyberbullying',
                 fontsize=14, color=PAL['a1'], weight='bold')

    for ax, (is_bully, color, title) in zip(axes, [
        (0, PAL['a4'], 'Normal Comments'),
        (1, PAL['a2'], 'Cyberbullying Comments')
    ]):
        texts = df[df['is_cyberbullying']==is_bully]['cleaned_text']
        bigrams = []
        for t in texts:
            words = t.split()
            bigrams += [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        freq = Counter(bigrams).most_common(15)
        words_list = [w for w, _ in freq]
        counts     = [c for _, c in freq]
        ax.barh(range(15), counts[::-1], color=color, alpha=0.8)
        ax.set_yticks(range(15))
        ax.set_yticklabels(words_list[::-1], fontsize=8)
        ax.set_title(title, color=PAL['a3'])
        ax.set_xlabel('Frequency'); ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return _save('eda_03_bigrams.png')

# ── EDA 4: Correlation of text features ──────────────────────────
def plot_feature_correlation(df):
    _theme()
    feat_cols = ['text_length','word_count','avg_word_length',
                 'exclamation_count','caps_ratio','punct_count','is_cyberbullying']
    corr = df[feat_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('Text Feature Correlation Matrix',
                 fontsize=14, color=PAL['a1'], weight='bold')
    sns.heatmap(corr, ax=ax, cmap=sns.diverging_palette(250, 10, as_cmap=True),
                center=0, annot=True, fmt='.2f', linewidths=0.5,
                linecolor=PAL['bg'], cbar_kws={'shrink': 0.8})
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    return _save('eda_04_feature_correlation.png')

# ── EDA 5: Caps ratio & exclamation analysis ──────────────────────
def plot_aggression_signals(df):
    _theme()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Aggression Signals: CAPS Usage & Exclamation Marks',
                 fontsize=14, color=PAL['a1'], weight='bold')

    palette = {cat: col for cat, col in zip(df['label'].unique(), CAT_COLORS)}
    sns.boxplot(data=df, x='label', y='caps_ratio',
                palette=palette, ax=axes[0], linewidth=0.8,
                flierprops=dict(marker='.', markersize=2, alpha=0.4))
    axes[0].set_title('CAPS Ratio per Category', color=PAL['a3'])
    axes[0].tick_params(axis='x', rotation=20); axes[0].grid(axis='y', alpha=0.3)

    sns.boxplot(data=df, x='label', y='exclamation_count',
                palette=palette, ax=axes[1], linewidth=0.8,
                flierprops=dict(marker='.', markersize=2, alpha=0.4))
    axes[1].set_title('Exclamation Count per Category', color=PAL['a3'])
    axes[1].tick_params(axis='x', rotation=20); axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return _save('eda_05_aggression_signals.png')


if __name__ == '__main__':
    print("Generating data for EDA...")
    gen = CyberbullyingDataGenerator(n_samples=8000)
    df  = gen.generate()
    df  = extract_text_features(df)
    df['avg_word_length'] = df['text'].apply(lambda x: sum(len(w) for w in x.split()) / (len(x.split()) + 1))
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().apply(len)

    prep = TextPreprocessor()
    df['cleaned_text'] = prep.fit_transform(df['text'])
    print("\n— DATASET OVERVIEW —")
    print(f"Shape        : {df.shape}")
    print(f"Columns      : {list(df.columns)}")
    print(f"Memory usage : {df.memory_usage().sum() / 1024**2:.2f} MB")
    print(f"Missing vals : {df.isnull().sum().sum()}")

    print("\n— LABEL DISTRIBUTION —")
    print(df['label'].value_counts())

    print("\n— CYBERBULLYING SPLIT —")
    print(df['is_cyberbullying'].value_counts())

    print("\n— SAMPLE DATA —")
    print(df[['text','cleaned_text']].head(3))

    print("\n[EDA] Generating plots...")
    plot_category_stats(df)
    plot_text_length_violin(df)
    plot_bigrams(df)
    plot_feature_correlation(df)
    plot_aggression_signals(df)
    print("\nEDA complete! Plots saved to ./plots/eda/")
