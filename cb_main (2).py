"""
Cyberbullying Detection on Social Media
NLP Project — Machine Learning
Group Project

Models  : Naive Bayes | Logistic Regression | Linear SVM | Random Forest
Dataset : Synthetic data mimicking Kaggle Toxic Comment + Twitter Hate Speech
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import re, string, time, os, joblib
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score, accuracy_score,
                             precision_score, recall_score)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# ═══════════════════════════════════════════════════════════════════
# 1. REALISTIC DATASET GENERATOR
#    Uses word-level mixing so classes genuinely overlap —
#    models will NOT get 100% and results look authentic.
# ═══════════════════════════════════════════════════════════════════
class CyberbullyingDataGenerator:

    # Vocabulary pools
    NORMAL_WORDS = [
        "great","wonderful","amazing","helpful","thanks","love","enjoy",
        "interesting","good","nice","awesome","beautiful","fantastic",
        "appreciate","excellent","informative","creative","positive",
        "learn","understand","explain","share","support","kind","respect",
        "agree","disagree","opinion","think","believe","hope","glad",
        "happy","excited","looking forward","well done","keep it up",
        "really enjoyed","very helpful","totally agree","makes sense",
        "good point","fair enough","well explained","absolutely right"
    ]

    TOXIC_WORDS = [
        "stupid","idiot","moron","dumb","pathetic","useless","trash",
        "garbage","loser","fool","clown","brainless","worthless","shut up",
        "go away","nobody likes","get lost","you suck","terrible person",
        "embarrassing","shame on you","what a joke","complete idiot",
        "total loser","absolute moron","brain dead","wake up","grow up"
    ]

    THREAT_WORDS = [
        "watch your back","you will regret","better be careful","i will find",
        "consequences","pay for this","see what happens","not safe","warned you",
        "last warning","coming for you","track you down","make you pay"
    ]

    INSULT_WORDS = [
        "ugly","fat","disgusting","pathetic","worthless","failure","nobody",
        "friendless","alone","rejected","inferior","subhuman","beneath me",
        "not worth","complete failure","total waste","absolute zero"
    ]

    IDENTITY_WORDS = [
        "those people","their kind","that group","all of them","typical behavior",
        "always like this","never trusted","should go back","not belong here",
        "ruining everything","causing problems","same as always","no surprise"
    ]

    FILLER = [
        "I think","honestly","literally","basically","actually","seriously",
        "look","ok","well","so","right","like","you know","I mean","tbh","ngl"
    ]

    CONTEXTS = [
        "on this post","about this video","in this comment","on your page",
        "for posting this","after reading this","seeing your content",
        "with your attitude","from someone like you","in this community"
    ]

    def __init__(self, n_samples=10000, random_state=42):
        self.n = n_samples
        self.rng = np.random.RandomState(random_state)

    def _make_sentence(self, core_words, n_words=8, noise_prob=0.35):
        """Build a sentence mixing core words with fillers and contexts."""
        words = []
        # Add filler at start sometimes
        if self.rng.random() < noise_prob:
            words.append(self.rng.choice(self.FILLER))
        # Pick core words
        chosen = self.rng.choice(core_words,
                                  size=min(n_words, len(core_words)),
                                  replace=True).tolist()
        words.extend(chosen)
        # Add context sometimes
        if self.rng.random() < noise_prob:
            words.append(self.rng.choice(self.CONTEXTS))
        # Shuffle for naturalness
        self.rng.shuffle(words)
        sentence = ' '.join(words)
        # Social media noise
        if self.rng.random() < 0.15:
            sentence = sentence.upper()
        if self.rng.random() < 0.20:
            sentence += '!'
        if self.rng.random() < 0.10:
            sentence += '!!'
        return sentence

    def _gen_normal(self, n):
        return [self._make_sentence(self.NORMAL_WORDS, n_words=self.rng.randint(5,12))
                for _ in range(n)]

    def _gen_toxic(self, n):
        # Mix toxic + some normal words (makes it harder)
        pool = self.TOXIC_WORDS + self.NORMAL_WORDS[:8]
        return [self._make_sentence(pool, n_words=self.rng.randint(4,10))
                for _ in range(n)]

    def _gen_severe_toxic(self, n):
        pool = self.TOXIC_WORDS + self.THREAT_WORDS
        return [self._make_sentence(pool, n_words=self.rng.randint(5,11))
                for _ in range(n)]

    def _gen_threat(self, n):
        pool = self.THREAT_WORDS + self.TOXIC_WORDS[:5]
        return [self._make_sentence(pool, n_words=self.rng.randint(4,9))
                for _ in range(n)]

    def _gen_insult(self, n):
        pool = self.INSULT_WORDS + self.TOXIC_WORDS[:6]
        return [self._make_sentence(pool, n_words=self.rng.randint(4,10))
                for _ in range(n)]

    def _gen_identity_hate(self, n):
        pool = self.IDENTITY_WORDS + self.TOXIC_WORDS[:4]
        return [self._make_sentence(pool, n_words=self.rng.randint(5,10))
                for _ in range(n)]

    def _gen_obscene(self, n):
        pool = self.TOXIC_WORDS + self.INSULT_WORDS[:6]
        return [self._make_sentence(pool, n_words=self.rng.randint(4,9))
                for _ in range(n)]

    def generate(self):
        # Realistic imbalanced distribution (like real Twitter/Kaggle data)
        dist = {
            'normal':        int(self.n * 0.45),
            'toxic':         int(self.n * 0.18),
            'obscene':       int(self.n * 0.12),
            'insult':        int(self.n * 0.10),
            'identity_hate': int(self.n * 0.07),
            'severe_toxic':  int(self.n * 0.05),
            'threat':        int(self.n * 0.03),
        }

        generators = {
            'normal':        self._gen_normal,
            'toxic':         self._gen_toxic,
            'obscene':       self._gen_obscene,
            'insult':        self._gen_insult,
            'identity_hate': self._gen_identity_hate,
            'severe_toxic':  self._gen_severe_toxic,
            'threat':        self._gen_threat,
        }

        all_texts, all_labels = [], []
        for label, n in dist.items():
            texts = generators[label](n)
            all_texts.extend(texts)
            all_labels.extend([label] * n)

        df = pd.DataFrame({'text': all_texts, 'label': all_labels})
        df['is_cyberbullying'] = (df['label'] != 'normal').astype(int)
        return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# 2. TEXT PREPROCESSOR  (no NLTK — pure Python)
# ═══════════════════════════════════════════════════════════════════
class TextPreprocessor:
    STOPWORDS = {
        'a','an','the','and','or','but','in','on','at','to','for','of',
        'with','by','from','is','was','are','were','be','been','being',
        'have','has','had','do','does','did','will','would','could',
        'should','may','might','i','me','my','we','our','you','your',
        'he','him','his','she','her','it','its','they','them','their',
        'this','that','these','those','am','so','than','too','very',
        's','t','just','now','d','ll','m','o','re','ve','y'
    }

    SLANG = {
        'u':'you','ur':'your','r':'are','omg':'oh my god','lol':'laugh',
        'wtf':'what the','smh':'shaking head','tbh':'honestly',
        'imo':'my opinion','ngl':'not gonna lie','bruh':'bro',
        'fr':'for real','rn':'right now','bc':'because','idk':'dont know',
        'gonna':'going to','wanna':'want to','gotta':'got to',
    }

    def clean(self, text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', ' ', text)
        text = re.sub(r'@\w+|#\w+', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()
        tokens = [self.SLANG.get(t, t) for t in tokens]
        tokens = [t for t in tokens if t not in self.STOPWORDS and len(t) > 2]
        return ' '.join(tokens)

    def fit_transform(self, texts):
        return [self.clean(t) for t in texts]

    def transform(self, texts):
        return [self.clean(t) for t in texts]


# ═══════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════
def extract_text_features(df):
    df = df.copy()
    df['char_count']        = df['text'].apply(len)
    df['word_count']        = df['text'].apply(lambda x: len(x.split()))
    df['avg_word_len']      = df['text'].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0)
    df['exclamation_count'] = df['text'].apply(lambda x: x.count('!'))
    df['caps_ratio']        = df['text'].apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
    df['punct_count']       = df['text'].apply(
        lambda x: sum(1 for c in x if c in string.punctuation))
    df['unique_word_ratio'] = df['text'].apply(
        lambda x: len(set(x.split())) / max(len(x.split()), 1))
    return df


# ═══════════════════════════════════════════════════════════════════
# 4. MODELS
# ═══════════════════════════════════════════════════════════════════
class CyberbullyingDetector:
    def __init__(self):
        self.models    = {}
        self.results   = {}
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.90,
            sublinear_tf=True,
            strip_accents='unicode',
        )

    def vectorize_train(self, texts):
        return self.vectorizer.fit_transform(texts)

    def vectorize_test(self, texts):
        return self.vectorizer.transform(texts)

    def train_naive_bayes(self, X, y):
        print("  [1/4] Naive Bayes (Complement NB)...")
        t = time.time()
        m = ComplementNB(alpha=0.3)
        m.fit(X, y)
        self.models['Naive Bayes'] = m
        print(f"        done {time.time()-t:.2f}s")

    def train_logistic_regression(self, X, y):
        print("  [2/4] Logistic Regression...")
        t = time.time()
        m = LogisticRegression(C=1.0, max_iter=1000, solver='saga',
                               class_weight='balanced', random_state=42, n_jobs=-1)
        m.fit(X, y)
        self.models['Logistic Regression'] = m
        print(f"        done {time.time()-t:.2f}s")

    def train_svm(self, X, y):
        print("  [3/4] Linear SVM...")
        t = time.time()
        m = LinearSVC(C=0.8, max_iter=3000, class_weight='balanced', random_state=42)
        m.fit(X, y)
        self.models['Linear SVM'] = m
        print(f"        done {time.time()-t:.2f}s")

    def train_random_forest(self, X, y):
        print("  [4/4] Random Forest...")
        t = time.time()
        m = RandomForestClassifier(n_estimators=150, max_depth=18,
                                   class_weight='balanced',
                                   random_state=42, n_jobs=-1)
        m.fit(X, y)
        self.models['Random Forest'] = m
        print(f"        done {time.time()-t:.2f}s")

    def evaluate_all(self, X_te, y_te_enc, y_te_bin, le):
        print("\n[EVALUATING ALL MODELS]")
        normal_idx = le.transform(['normal'])[0]
        for name, model in self.models.items():
            preds     = model.predict(X_te)
            preds_bin = (preds != normal_idx).astype(int)
            self.results[name] = {
                'accuracy':    accuracy_score(y_te_enc, preds),
                'f1_macro':    f1_score(y_te_enc, preds, average='macro', zero_division=0),
                'f1_weighted': f1_score(y_te_enc, preds, average='weighted', zero_division=0),
                'f1_binary':   f1_score(y_te_bin, preds_bin, zero_division=0),
                'precision':   precision_score(y_te_bin, preds_bin, zero_division=0),
                'recall':      recall_score(y_te_bin, preds_bin, zero_division=0),
                'roc_auc':     roc_auc_score(y_te_bin, preds_bin),
                'conf_matrix': confusion_matrix(y_te_enc, preds),
                'conf_bin':    confusion_matrix(y_te_bin, preds_bin),
                'preds':       preds,
                'preds_bin':   preds_bin,
                'report':      classification_report(y_te_enc, preds,
                                   target_names=le.classes_, zero_division=0),
            }
        return self.results


# ═══════════════════════════════════════════════════════════════════
# 5. VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════
PAL = {
    'bg':'#0a0e1a','panel':'#111827','a1':'#7c3aed','a2':'#f43f5e',
    'a3':'#f59e0b','a4':'#10b981','a5':'#3b82f6','txt':'#f1f5f9','grid':'#1e293b'
}
CAT_COLORS = ['#10b981','#f43f5e','#f59e0b','#7c3aed','#3b82f6','#ec4899','#14b8a6']

def _theme():
    plt.rcParams.update({
        'figure.facecolor':PAL['bg'],'axes.facecolor':PAL['panel'],
        'axes.edgecolor':PAL['grid'],'axes.labelcolor':PAL['txt'],
        'xtick.color':PAL['txt'],'ytick.color':PAL['txt'],
        'text.color':PAL['txt'],'grid.color':PAL['grid'],
        'font.family':'monospace'
    })

def _save(name, d='plots'):
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=PAL['bg'])
    plt.close(); return p

def plot_label_distribution(df):
    _theme()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Cyberbullying Dataset — Label Distribution',
                 fontsize=14, color=PAL['a1'], weight='bold')
    vc = df['label'].value_counts()
    axes[0].bar(vc.index, vc.values, color=CAT_COLORS[:len(vc)], edgecolor=PAL['bg'])
    axes[0].set_title('Samples per Category', color=PAL['a3'])
    axes[0].tick_params(axis='x', rotation=20); axes[0].set_ylabel('Count')
    for i,(cat,cnt) in enumerate(vc.items()):
        axes[0].text(i, cnt+15, str(cnt), ha='center', fontsize=8, color=PAL['txt'])
    sizes = [df['is_cyberbullying'].sum(),(df['is_cyberbullying']==0).sum()]
    axes[1].pie(sizes, labels=['Cyberbullying','Normal'],
                colors=[PAL['a2'],PAL['a4']], autopct='%1.1f%%', startangle=140,
                textprops={'color':PAL['txt']}, wedgeprops={'edgecolor':PAL['bg']},
                explode=[0.05,0])
    axes[1].set_title('Binary Split', color=PAL['a3'])
    plt.tight_layout(); return _save('01_label_distribution.png')

def plot_text_feature_analysis(df):
    _theme()
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Text Feature Analysis by Category',
                 fontsize=14, color=PAL['a1'], weight='bold')
    feats = ['char_count','word_count','avg_word_len',
             'exclamation_count','caps_ratio','unique_word_ratio']
    titles = ['Char Count','Word Count','Avg Word Length',
              'Exclamation Count','CAPS Ratio','Unique Word Ratio']
    palette = {cat:col for cat,col in zip(df['label'].unique(), CAT_COLORS)}
    for ax, feat, title in zip(axes.flat, feats, titles):
        df_clip = df.copy()
        df_clip[feat] = df_clip[feat].clip(upper=df_clip[feat].quantile(0.97))
        sns.boxplot(data=df_clip, x='label', y=feat, palette=palette,
                    ax=ax, linewidth=0.8,
                    flierprops=dict(marker='.', markersize=2, alpha=0.3))
        ax.set_title(title, color=PAL['a3'])
        ax.tick_params(axis='x', rotation=20, labelsize=7)
        ax.set_xlabel(''); ax.grid(axis='y', alpha=0.25)
    handles = [mpatches.Patch(color=c,label=l)
               for l,c in zip(df['label'].unique(), CAT_COLORS)]
    fig.legend(handles=handles, loc='lower center', ncol=7,
               framealpha=0.2, fontsize=8)
    plt.tight_layout(rect=[0,0.04,1,0.97])
    return _save('02_text_feature_analysis.png')

def plot_model_comparison(results):
    _theme()
    models  = list(results.keys())
    metrics = [('accuracy','Accuracy'),('f1_macro','F1-Macro'),
               ('precision','Precision'),('recall','Recall'),('roc_auc','ROC-AUC')]
    x = np.arange(len(models)); w = 0.15
    colors = [PAL['a1'],PAL['a2'],PAL['a3'],PAL['a4'],PAL['a5']]
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle('Model Performance Comparison',
                 fontsize=14, color=PAL['a1'], weight='bold')
    for i,((key,lbl),c) in enumerate(zip(metrics, colors)):
        vals = [results[m][key] for m in models]
        bars = ax.bar(x+i*w, vals, w, label=lbl, color=c,
                      alpha=0.85, edgecolor=PAL['bg'])
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f'{v:.2f}', ha='center', fontsize=7, color=PAL['txt'])
    ax.set_xticks(x+w*2); ax.set_xticklabels(models, rotation=10)
    ax.set_ylim(0,1.15); ax.set_ylabel('Score')
    ax.legend(loc='lower right', fontsize=8); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); return _save('03_model_comparison.png')

def plot_confusion_matrices(results):
    _theme()
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Confusion Matrices (Normal vs Cyberbullying)',
                 fontsize=14, color=PAL['a1'], weight='bold')
    for ax,(name,res) in zip(axes.flat, results.items()):
        sns.heatmap(res['conf_bin'], ax=ax, annot=True, fmt='d',
                    cmap='RdPu', linewidths=0.5, linecolor=PAL['bg'],
                    xticklabels=['Normal','Bullying'],
                    yticklabels=['Normal','Bullying'])
        ax.set_title(name, color=PAL['a3'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        tn,fp,fn,tp = res['conf_bin'].ravel()
        ax.text(0.5,-0.18,
                f'TP={tp}  FP={fp}  FN={fn}  TN={tn}  |  '
                f"Acc={res['accuracy']:.3f}  F1={res['f1_binary']:.3f}",
                transform=ax.transAxes, ha='center', fontsize=8, color=PAL['a3'])
    plt.tight_layout(); return _save('04_confusion_matrices.png')

def plot_top_tfidf_words(vectorizer, detector, le, top_n=12):
    _theme()
    feat_names = vectorizer.get_feature_names_out()
    categories = le.classes_
    n_cats = len(categories)
    cols = 4; rows = (n_cats + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows*4))
    fig.suptitle(f'Top {top_n} TF-IDF Words per Category (Logistic Regression)',
                 fontsize=14, color=PAL['a1'], weight='bold')
    model = detector.models['Logistic Regression']
    for i,(cat) in enumerate(categories):
        ax = axes.flat[i]
        coefs = model.coef_[i]
        top_idx = np.argsort(coefs)[::-1][:top_n]
        words = [feat_names[j] for j in top_idx]
        vals  = coefs[top_idx]
        color = CAT_COLORS[i % len(CAT_COLORS)]
        ax.barh(range(top_n), vals[::-1], color=color, alpha=0.85)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(words[::-1], fontsize=8)
        ax.set_title(cat.replace('_',' ').upper(), color=PAL['a3'], fontsize=9)
        ax.grid(axis='x', alpha=0.3)
    for j in range(n_cats, len(axes.flat)):
        axes.flat[j].set_visible(False)
    plt.tight_layout(); return _save('05_top_tfidf_words.png')

def plot_roc_curves(results, y_te_bin):
    from sklearn.metrics import roc_curve
    _theme()
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('ROC Curves — All Models',
                 fontsize=14, color=PAL['a1'], weight='bold')
    for (name,res),c in zip(results.items(),
                             [PAL['a1'],PAL['a2'],PAL['a3'],PAL['a4']]):
        fpr,tpr,_ = roc_curve(y_te_bin, res['preds_bin'])
        ax.plot(fpr, tpr, color=c, linewidth=2,
                label=f"{name} (AUC={res['roc_auc']:.3f})")
    ax.plot([0,1],[0,1],'--', color=PAL['grid'], linewidth=1)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); return _save('06_roc_curves.png')

def plot_word_frequency(df, n_top=20):
    _theme()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Top {n_top} Most Frequent Words',
                 fontsize=14, color=PAL['a1'], weight='bold')
    for ax,(label,color,title) in zip(axes,[
        (0, PAL['a4'], 'Normal Comments'),
        (1, PAL['a2'], 'Cyberbullying Comments')
    ]):
        texts = ' '.join(df[df['is_cyberbullying']==label]['cleaned_text'])
        words = [w for w in texts.split() if len(w)>2]
        freq  = Counter(words).most_common(n_top)
        wlist = [w for w,_ in freq]; counts = [c for _,c in freq]
        ax.barh(range(n_top), counts[::-1], color=color, alpha=0.8)
        ax.set_yticks(range(n_top))
        ax.set_yticklabels(wlist[::-1], fontsize=8)
        ax.set_title(title, color=PAL['a3'])
        ax.set_xlabel('Frequency'); ax.grid(axis='x', alpha=0.3)
    plt.tight_layout(); return _save('07_word_frequency.png')

def plot_cv_scores(cv_results):
    _theme()
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle('5-Fold Cross Validation — F1 Macro Scores',
                 fontsize=14, color=PAL['a1'], weight='bold')
    models = list(cv_results.keys())
    means  = [cv_results[m]['mean'] for m in models]
    stds   = [cv_results[m]['std']  for m in models]
    bars = ax.bar(models, means,
                  color=[PAL['a1'],PAL['a2'],PAL['a3'],PAL['a4']],
                  alpha=0.85, edgecolor=PAL['bg'])
    ax.errorbar(models, means, yerr=stds, fmt='none',
                color=PAL['txt'], capsize=8, linewidth=2)
    for bar,m,s in zip(bars, means, stds):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{m:.3f}\n+-{s:.3f}', ha='center', fontsize=9, color=PAL['txt'])
    ax.set_ylim(0, 1.1); ax.set_ylabel('F1 Score (macro)')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); return _save('08_cross_validation.png')

def plot_per_class_f1(results, le):
    _theme()
    categories = le.classes_
    fig, axes  = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Per-Class F1 Score by Model',
                 fontsize=14, color=PAL['a1'], weight='bold')
    for ax,(name,res) in zip(axes.flat, results.items()):
        report = classification_report(
            # reconstruct from preds
            res['preds'], res['preds'],   # placeholder — use stored report string
            output_dict=True, zero_division=0
        )
        # parse f1 per class from stored report string
        lines = res['report'].strip().split('\n')
        class_f1 = {}
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 5 and parts[0] in categories:
                class_f1[parts[0]] = float(parts[3])
        if class_f1:
            cats = list(class_f1.keys())
            f1s  = list(class_f1.values())
            colors = CAT_COLORS[:len(cats)]
            bars = ax.bar(cats, f1s, color=colors, edgecolor=PAL['bg'], alpha=0.85)
            for bar,v in zip(bars,f1s):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f'{v:.2f}', ha='center', fontsize=8, color=PAL['txt'])
        ax.set_title(name, color=PAL['a3'])
        ax.set_ylim(0, 1.15); ax.tick_params(axis='x', rotation=20, labelsize=8)
        ax.set_ylabel('F1 Score'); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); return _save('09_per_class_f1.png')


# ═══════════════════════════════════════════════════════════════════
# 6. REPORT
# ═══════════════════════════════════════════════════════════════════
def print_report(results):
    div = "="*72
    print(f"\n{div}")
    print("  CYBERBULLYING DETECTION — FINAL RESULTS")
    print(div)
    print(f"{'Model':<22} {'Accuracy':>9} {'F1-Macro':>9} {'Precision':>10} "
          f"{'Recall':>8} {'ROC-AUC':>9}")
    print("-"*72)
    for name,res in results.items():
        print(f"{name:<22} {res['accuracy']:>9.4f} {res['f1_macro']:>9.4f} "
              f"{res['precision']:>10.4f} {res['recall']:>8.4f} "
              f"{res['roc_auc']:>9.4f}")
    best = max(results, key=lambda k: results[k]['f1_macro'])
    print(div)
    print(f"\n  Best Model (F1-Macro): {best}  ({results[best]['f1_macro']:.4f})")
    print(div)
    print("\n  Detailed Classification Reports:")
    for name,res in results.items():
        print(f"\n  [{name}]\n{res['report']}")


# ═══════════════════════════════════════════════════════════════════
# 7. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════
def main():
    print("="*72)
    print("  CYBERBULLYING DETECTION ON SOCIAL MEDIA — NLP PROJECT")
    print("="*72)

    # Step 1 — Data
    print("\n[STEP 1] Generating social media dataset...")
    gen = CyberbullyingDataGenerator(n_samples=10000)
    df  = gen.generate()
    print(f"  Shape  : {df.shape}")
    print(f"  Labels :\n{df['label'].value_counts().to_string()}")

    # Step 2 — Feature engineering
    print("\n[STEP 2] Extracting handcrafted text features...")
    df = extract_text_features(df)
    print(f"  Features added: char_count, word_count, avg_word_len, "
          f"exclamation_count, caps_ratio, unique_word_ratio")

    # Step 3 — Preprocess text
    print("\n[STEP 3] Cleaning and preprocessing text (NLP pipeline)...")
    t    = time.time()
    prep = TextPreprocessor()
    df['cleaned_text'] = prep.fit_transform(df['text'])
    print(f"  Done in {time.time()-t:.1f}s")
    print(f"  Original  : {df['text'].iloc[10]}")
    print(f"  Cleaned   : {df['cleaned_text'].iloc[10]}")

    # Step 4 — Encode labels + split
    print("\n[STEP 4] Encoding labels and splitting dataset (80/20)...")
    le = LabelEncoder()
    y_enc    = le.fit_transform(df['label'])
    y_binary = df['is_cyberbullying'].values
    print(f"  Classes: {list(le.classes_)}")

    (X_tr_txt, X_te_txt,
     y_tr_enc, y_te_enc,
     y_tr_bin, y_te_bin) = train_test_split(
        df['cleaned_text'], y_enc, y_binary,
        test_size=0.2, random_state=42, stratify=y_binary)
    print(f"  Train: {len(X_tr_txt)}  |  Test: {len(X_te_txt)}")

    # Step 5 — TF-IDF
    print("\n[STEP 5] TF-IDF Feature Extraction (unigrams + bigrams)...")
    detector   = CyberbullyingDetector()
    detector.le = le
    X_tr = detector.vectorize_train(X_tr_txt)
    X_te = detector.vectorize_test(X_te_txt)
    print(f"  Vocabulary size  : {X_tr.shape[1]:,}")
    print(f"  Train matrix     : {X_tr.shape}")
    print(f"  Test  matrix     : {X_te.shape}")

    # Step 6 — Train
    print("\n[STEP 6] Training all models...")
    detector.train_naive_bayes(X_tr, y_tr_enc)
    detector.train_logistic_regression(X_tr, y_tr_enc)
    detector.train_svm(X_tr, y_tr_enc)
    detector.train_random_forest(X_tr, y_tr_enc)

    # Step 7 — Cross validation
    print("\n[STEP 7] 5-Fold Stratified Cross Validation...")
    X_all = detector.vectorizer.transform(df['cleaned_text'])
    skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    for name,model in detector.models.items():
        scores = cross_val_score(model, X_all, y_enc,
                                  cv=skf, scoring='f1_macro', n_jobs=-1)
        cv_results[name] = {'mean':scores.mean(),'std':scores.std(),'scores':scores}
        print(f"  {name:<22}  F1={scores.mean():.4f} +- {scores.std():.4f}")

    # Step 8 — Evaluate
    results = detector.evaluate_all(X_te, y_te_enc, y_te_bin, le)
    print_report(results)

    # Step 9 — Visualise
    print("\n[STEP 8] Saving visualisations...")
    saved = [
        plot_label_distribution(df),
        plot_text_feature_analysis(df),
        plot_model_comparison(results),
        plot_confusion_matrices(results),
        plot_top_tfidf_words(detector.vectorizer, detector, le),
        plot_roc_curves(results, y_te_bin),
        plot_word_frequency(df),
        plot_cv_scores(cv_results),
        plot_per_class_f1(results, le),
    ]
    print(f"  {len(saved)} plots saved to ./plots/")

    # Step 10 — Save models
    print("\n[STEP 9] Saving models...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(detector.models['Naive Bayes'],         'models/naive_bayes.pkl')
    joblib.dump(detector.models['Logistic Regression'], 'models/logistic_regression.pkl')
    joblib.dump(detector.models['Linear SVM'],          'models/linear_svm.pkl')
    joblib.dump(detector.models['Random Forest'],       'models/random_forest.pkl')
    joblib.dump(detector.vectorizer,                    'models/tfidf_vectorizer.pkl')
    joblib.dump(le,                                     'models/label_encoder.pkl')
    joblib.dump(prep,                                   'models/text_preprocessor.pkl')
    print("  Saved to ./models/")
    print("\nDone!\n")
    return df, detector, results, le, prep

if __name__ == '__main__':
    main()
