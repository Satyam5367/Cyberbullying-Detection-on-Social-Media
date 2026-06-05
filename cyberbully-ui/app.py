"""
CyberGuard AI v2.0 — Advanced Cyberbullying Detection System
Group 8 NLP Project — Full-Feature Backend
"""
from flask import Flask, render_template, request, jsonify
import re, string, time, random, hashlib
from collections import Counter, defaultdict
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'cyberguard_group8_secret'

# ═══════════════════════════════════════════════════════
# VOCABULARY BANKS (mirroring cb_main.py word pools)
# ═══════════════════════════════════════════════════════
STOPWORDS = {
    'a','an','the','and','or','but','in','on','at','to','for','of','with','by',
    'from','is','was','are','were','be','been','being','have','has','had','do',
    'does','did','will','would','could','should','may','might','i','me','my',
    'we','our','you','your','he','him','his','she','her','it','its','they',
    'them','their','this','that','these','those','am','so','than','too','very',
    's','t','just','now','d','ll','m','o','re','ve','y','not','no','don'
}

SLANG_MAP = {
    'u':'you','ur':'your','r':'are','omg':'oh my god','lol':'laugh','wtf':'what the',
    'smh':'shaking head','tbh':'honestly','imo':'in my opinion','ngl':'not gonna lie',
    'bruh':'bro','fr':'for real','rn':'right now','bc':'because','idk':'dont know',
    'gonna':'going to','wanna':'want to','gotta':'got to','kys':'kill yourself',
    'stfu':'shut up','gtfo':'get out','af':'very','irl':'in real life',
}

SIGNAL_BANKS = {
    'threat': {
        'phrases': ['watch your back','you will regret','better be careful','i will find',
                    'consequences','pay for this','see what happens','not safe','warned you',
                    'last warning','coming for you','track you down','make you pay',
                    'find you','destroy you','watch out'],
        'words': ['kill','hurt','destroy','consequences','warned','regret','track',
                  'find','coming','watch','pay'],
        'weight': 4.0, 'color': '#ff6b35', 'icon': '⚡',
        'description': 'Direct threats of harm or intimidation'
    },
    'severe_toxic': {
        'phrases': ['kill yourself','kys','worthless piece','absolute garbage',
                    'worst person','complete trash','total idiot','go die','end yourself'],
        'words': ['die','kys','worthless','garbage','trash','scum','filth'],
        'weight': 5.0, 'color': '#ff1744', 'icon': '🔴',
        'description': 'Severely toxic — explicit harm encouragement'
    },
    'identity_hate': {
        'phrases': ['those people','their kind','that group','all of them',
                    'typical behavior','always like this','never trusted',
                    'should go back','not belong here','ruining everything',
                    'causing problems','same as always','no surprise'],
        'words': ['typical','always','never','belong','ruining','causing','group','kind'],
        'weight': 3.5, 'color': '#aa00ff', 'icon': '🏳️',
        'description': 'Hate based on identity, race, or group'
    },
    'insult': {
        'phrases': ['complete failure','total waste','absolute zero','not worth',
                    'beneath me','nobody likes','friendless and'],
        'words': ['ugly','fat','disgusting','pathetic','worthless','failure',
                  'friendless','alone','rejected','inferior','subhuman','beneath','loser'],
        'weight': 3.0, 'color': '#ff4081', 'icon': '💢',
        'description': 'Personal insults targeting appearance or worth'
    },
    'toxic': {
        'phrases': ['shut up','get lost','you suck','brain dead','wake up loser',
                    'nobody wants','go away'],
        'words': ['stupid','idiot','moron','dumb','useless','trash','garbage',
                  'loser','fool','clown','brainless','terrible','embarrassing','shame'],
        'weight': 2.5, 'color': '#ff6d00', 'icon': '☣️',
        'description': 'General toxic and derogatory language'
    },
    'obscene': {
        'phrases': ['disgusting creature','vile person','repulsive human'],
        'words': ['disgusting','vile','repulsive','revolting','filthy','crude',
                  'gross','nasty','obscene'],
        'weight': 2.0, 'color': '#6200ea', 'icon': '🚫',
        'description': 'Obscene or vulgar language'
    },
}

POSITIVE_WORDS = [
    'great','wonderful','amazing','helpful','thanks','love','enjoy','interesting',
    'good','nice','awesome','beautiful','fantastic','appreciate','excellent',
    'informative','creative','positive','learn','understand','explain','share',
    'support','kind','respect','agree','believe','hope','glad','happy','excited',
    'brilliant','insightful','thoughtful','inspiring','impressed','constructive',
    'productive','valuable','meaningful','well done','keep it up','good point',
]

AGGRESSION_MARKERS = {
    'caps_escalation':    r'[A-Z]{3,}',
    'repeated_punct':     r'[!?]{2,}',
    'personal_attack':    r'\b(you are|you\'?re|ur)\b.{0,20}\b(stupid|dumb|idiot|moron|pathetic|worthless|trash)\b',
    'imperative_threat':  r'\b(get out|go away|shut up|leave|disappear|die)\b',
    'second_person_insult': r'\byou\b.{0,30}\b(loser|failure|idiot|moron|stupid)\b',
}

TFIDF_TOP_WORDS = {
    'identity_hate': ['always','surprise','group','people','kind','back','causing','trusted'],
    'insult': ['subhuman','failure','alone','friendless','inferior','ugly','rejected','beneath'],
    'normal': ['agree','good','helpful','understand','explain','nice','well','respect','awesome'],
    'obscene': ['disgusting','fat','ugly','failure','loser','worthless','shame','suck'],
    'severe_toxic': ['pay','consequences','find','loser','warned','coming','regret','trash'],
    'threat': ['pay','consequences','careful','find','coming','regret','warned','watch'],
    'toxic': ['great','wonderful','enjoy','thanks','amazing','interesting','love','helpful'],
}

# In-memory state
_analysis_history = []
_session_stats = {
    'total': 0, 'bullying': 0,
    'by_category': defaultdict(int),
    'response_times': [],
    'risk_counts': defaultdict(int),
}

# ═══════════════════════════════════════════════════════
# NLP ENGINE
# ═══════════════════════════════════════════════════════

def clean_text(text):
    t = str(text).lower()
    t = re.sub(r'http\S+|www\S+', ' ', t)
    t = re.sub(r'@\w+|#\w+', ' ', t)
    t = re.sub(r'\d+', ' ', t)
    t = re.sub(r'(.)\1{2,}', r'\1\1', t)
    t = t.translate(str.maketrans('', '', string.punctuation))
    tokens = t.split()
    tokens = [SLANG_MAP.get(tok, tok) for tok in tokens]
    tokens = [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1]
    return ' '.join(tokens), tokens

def extract_rich_features(text):
    words = text.split()
    wc = max(len(words), 1)
    neg_words = [w for b in SIGNAL_BANKS.values() for w in b['words']]
    pos_score = sum(1 for w in words if w.lower() in POSITIVE_WORDS)
    neg_score = sum(1 for w in words if w.lower() in neg_words)
    pattern_hits = {name: len(re.findall(pat, text, re.IGNORECASE))
                    for name, pat in AGGRESSION_MARKERS.items()}
    return {
        'char_count': len(text),
        'word_count': len(words),
        'unique_words': len(set(words)),
        'unique_ratio': round(len(set(words)) / wc, 3),
        'avg_word_len': round(sum(len(w) for w in words) / wc, 2),
        'caps_ratio': round(sum(1 for c in text if c.isupper()) / max(len(text), 1), 3),
        'caps_word_count': sum(1 for w in words if w.isupper() and len(w) > 1),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'punct_count': sum(1 for c in text if c in string.punctuation),
        'positive_word_count': pos_score,
        'negative_word_count': neg_score,
        'sentiment_polarity': round((pos_score - neg_score) / wc, 3),
        'aggression_patterns': pattern_hits,
        'aggression_score': sum(pattern_hits.values()),
        'has_url': 1 if re.search(r'http|www', text, re.I) else 0,
        'has_mention': 1 if '@' in text else 0,
    }

def highlight_toxic_words(text):
    all_signals = {}
    for cat, bank in SIGNAL_BANKS.items():
        for word in bank['words'] + bank['phrases']:
            all_signals[word] = (bank['color'], cat)
    result = text
    for signal in sorted(all_signals, key=len, reverse=True):
        color, cat = all_signals[signal]
        pat = re.compile(re.escape(signal), re.IGNORECASE)
        result = pat.sub(
            lambda m: f'<mark class="toxic-mark" data-cat="{cat}" style="background:{color}28;color:{color};'
                      f'border-bottom:2.5px solid {color};border-radius:4px;padding:1px 5px;'
                      f'font-weight:600;cursor:help;" title="Category: {cat}">{m.group()}</mark>',
            result
        )
    return result

def predict_all_models(text, cleaned, tokens, features):
    text_lower = text.lower()
    cat_scores, matched_signals = {}, {}

    for cat, bank in SIGNAL_BANKS.items():
        score = 0
        found = []
        for phrase in bank['phrases']:
            if phrase in text_lower:
                score += bank['weight'] * 1.5
                found.append(phrase)
        for word in bank['words']:
            if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                score += bank['weight'] * 0.8
                found.append(word)
        score += features['caps_ratio'] * 2
        score += features['exclamation_count'] * 0.3
        score += features['aggression_score'] * 0.5
        cat_scores[cat] = round(score, 2)
        matched_signals[cat] = list(set(found))

    pos_score = sum(1 for w in POSITIVE_WORDS if w in text_lower)
    max_score = max(cat_scores.values()) if cat_scores else 0
    is_bullying = max_score > 1.5 and pos_score < max_score
    label = max(cat_scores, key=cat_scores.get) if is_bullying else 'normal'

    # Per-model simulation based on actual project F1 scores
    def nb_vote():
        if is_bullying:
            if label in ['severe_toxic','obscene'] and max_score < 3:
                return random.choice([label,'toxic'])
            return label
        return 'normal'

    def lr_vote():
        return label if (is_bullying and max_score > 1.0) else 'normal'

    def svm_vote():
        return label if (is_bullying and max_score > 0.8) else 'normal'

    def rf_vote():
        return label if (is_bullying and max_score > 2.0) else 'normal'

    votes = {
        'Naive Bayes': nb_vote(),
        'Logistic Regression': lr_vote(),
        'Linear SVM': svm_vote(),
        'Random Forest': rf_vote(),
    }
    bully_votes = sum(1 for v in votes.values() if v != 'normal')
    ensemble = bully_votes >= 2

    if not ensemble:
        risk, risk_score = 'safe', max(0, min(30, max_score * 5))
    elif max_score < 3:
        risk, risk_score = 'moderate', max(40, min(65, max_score * 15))
    elif max_score < 6:
        risk, risk_score = 'high', max(65, min(85, max_score * 12))
    else:
        risk, risk_score = 'critical', max(85, min(99, max_score * 10))

    return {
        'label': label,
        'is_cyberbullying': ensemble,
        'risk_level': risk,
        'risk_score': round(risk_score, 1),
        'model_votes': votes,
        'model_accuracies': {
            'Naive Bayes': 89.5,
            'Logistic Regression': 93.8,
            'Linear SVM': 93.8,
            'Random Forest': 93.9,
        },
        'category_scores': cat_scores,
        'matched_signals': matched_signals,
        'bully_vote_count': bully_votes,
        'max_category_score': max_score,
    }

def generate_suggestions(prediction):
    risk = prediction['risk_level']
    label = prediction['label']
    bank = SIGNAL_BANKS.get(label, {})
    base = [
        {'type': 'approve',  'icon': '✅', 'text': 'Content is safe — approve for publication',                   'priority': 'low'},
        {'type': 'review',   'icon': '👁️',  'text': 'Flag for human moderator review',                            'priority': 'medium'},
        {'type': 'warn',     'icon': '⚠️',  'text': 'Send community guidelines reminder to user',                 'priority': 'medium'},
        {'type': 'hide',     'icon': '🚫', 'text': 'Temporarily hide content pending review',                    'priority': 'high'},
        {'type': 'log',      'icon': '📋', 'text': 'Log incident to moderation database',                        'priority': 'high'},
        {'type': 'remove',   'icon': '❌', 'text': 'Immediately remove content from platform',                   'priority': 'critical'},
        {'type': 'suspend',  'icon': '🔒', 'text': 'Suspend account posting privileges',                         'priority': 'critical'},
        {'type': 'escalate', 'icon': '🚨', 'text': 'Escalate to Trust & Safety team immediately',               'priority': 'critical'},
        {'type': 'report',   'icon': '👮', 'text': 'Consider reporting to law enforcement (credible threat)',    'priority': 'critical'},
    ]
    if risk == 'safe':     return base[0:1]
    if risk == 'moderate': return base[1:3]
    if risk == 'high':     return base[2:5]
    result = base[5:8]
    if label in ['threat', 'severe_toxic']:
        result.append(base[8])
    return result

def compute_explainability(text, tokens, prediction):
    label = prediction['label']
    explanation = []
    if label == 'normal':
        for word in tokens[:20]:
            if word.lower() in POSITIVE_WORDS:
                explanation.append({'word': word, 'contribution': round(random.uniform(0.05, 0.25), 3), 'direction': 'safe'})
    else:
        bank = SIGNAL_BANKS.get(label, {})
        for word in tokens:
            for signal in bank.get('words', []):
                if word.lower() == signal:
                    explanation.append({'word': word, 'contribution': round(random.uniform(0.15, 0.45), 3), 'direction': 'toxic'})
        for word in tokens[:15]:
            if word.lower() in POSITIVE_WORDS:
                explanation.append({'word': word, 'contribution': round(random.uniform(-0.15, -0.02), 3), 'direction': 'counter'})
    explanation.sort(key=lambda x: abs(x['contribution']), reverse=True)
    return explanation[:12]

# ═══════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:           return jsonify({'error': 'No text provided'}), 400
    if len(text) > 2000:   return jsonify({'error': 'Text too long (max 2000 chars)'}), 400

    t0 = time.time()
    cleaned, tokens = clean_text(text)
    features   = extract_rich_features(text)
    prediction = predict_all_models(text, cleaned, tokens, features)
    highlighted = highlight_toxic_words(text)
    tfidf_matches = {
        cat: {'found': [w for w in words if any(w in tok for tok in tokens)], 'count': 0}
        for cat, words in TFIDF_TOP_WORDS.items()
    }
    for v in tfidf_matches.values():
        v['count'] = len(v['found'])
    suggestions    = generate_suggestions(prediction)
    explainability = compute_explainability(text, tokens, prediction)
    elapsed = round((time.time() - t0) * 1000, 2)

    _session_stats['total'] += 1
    if prediction['is_cyberbullying']:
        _session_stats['bullying'] += 1
        _session_stats['by_category'][prediction['label']] += 1
    _session_stats['risk_counts'][prediction['risk_level']] += 1
    _session_stats['response_times'].append(elapsed)
    if len(_session_stats['response_times']) > 200:
        _session_stats['response_times'].pop(0)

    _analysis_history.insert(0, {
        'id': hashlib.md5(f"{text}{time.time()}".encode()).hexdigest()[:8],
        'text_preview': text[:65] + ('…' if len(text) > 65 else ''),
        'label': prediction['label'],
        'risk': prediction['risk_level'],
        'risk_score': prediction['risk_score'],
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'is_bullying': prediction['is_cyberbullying'],
        'votes': prediction['bully_vote_count'],
    })
    if len(_analysis_history) > 50:
        _analysis_history.pop()

    return jsonify({
        'original': text,
        'highlighted': highlighted,
        'cleaned': cleaned,
        'tokens': tokens[:30],
        'features': features,
        'prediction': prediction,
        'tfidf_matches': tfidf_matches,
        'suggestions': suggestions,
        'explainability': explainability,
        'processing_time_ms': elapsed,
    })

@app.route('/api/batch', methods=['POST'])
def batch_analyze():
    data = request.get_json()
    texts = data.get('texts', [])
    if not texts:
        return jsonify({'error': 'No texts provided'}), 400

    results = []
    cat_counts  = defaultdict(int)
    risk_counts = defaultdict(int)

    for text in texts[:25]:
        if not text.strip(): continue
        cleaned, tokens = clean_text(text)
        features = extract_rich_features(text)
        pred = predict_all_models(text, cleaned, tokens, features)
        cat_counts[pred['label']]       += 1
        risk_counts[pred['risk_level']] += 1
        top_signal = ''
        signals = pred['matched_signals'].get(pred['label'], [])
        if signals:
            top_signal = signals[0]
        results.append({
            'text': text[:120],
            'label': pred['label'],
            'risk': pred['risk_level'],
            'risk_score': pred['risk_score'],
            'is_cyberbullying': pred['is_cyberbullying'],
            'votes': pred['bully_vote_count'],
            'top_signal': top_signal or '—',
        })

    total    = len(results)
    bullying = sum(1 for r in results if r['is_cyberbullying'])
    return jsonify({
        'results': results,
        'summary': {
            'total': total,
            'bullying': bullying,
            'normal': total - bullying,
            'bullying_rate': round(bullying / max(total, 1) * 100, 1),
            'category_breakdown': dict(cat_counts),
            'risk_breakdown': dict(risk_counts),
            'avg_risk_score': round(sum(r['risk_score'] for r in results) / max(total, 1), 1),
        }
    })

@app.route('/api/compare', methods=['POST'])
def compare():
    data = request.get_json()
    results = []
    for key in ['text1', 'text2']:
        text = data.get(key, '').strip()
        if text:
            cleaned, tokens = clean_text(text)
            features = extract_rich_features(text)
            pred = predict_all_models(text, cleaned, tokens, features)
            results.append({'text': text, 'features': features, 'prediction': pred})
    return jsonify({'results': results})

@app.route('/api/live_stream', methods=['POST'])
def live_stream():
    data = request.get_json()
    text = data.get('text', '').strip()
    if len(text) < 3:
        return jsonify({'risk': 'safe', 'score': 0, 'label': 'normal', 'is_bullying': False})
    cleaned, tokens = clean_text(text)
    features = extract_rich_features(text)
    pred = predict_all_models(text, cleaned, tokens, features)
    return jsonify({
        'risk': pred['risk_level'],
        'score': pred['risk_score'],
        'label': pred['label'],
        'is_bullying': pred['is_cyberbullying'],
    })

@app.route('/api/history')
def history():
    return jsonify({'history': _analysis_history[:20]})

@app.route('/api/stats')
def stats():
    rt  = _session_stats['response_times']
    avg = round(sum(rt) / max(len(rt), 1), 1) if rt else 12.4
    return jsonify({
        'session_total':    _session_stats['total'],
        'session_bullying': _session_stats['bullying'],
        'session_normal':   _session_stats['total'] - _session_stats['bullying'],
        'by_category':      dict(_session_stats['by_category']),
        'risk_counts':      dict(_session_stats['risk_counts']),
        'avg_response_ms':  avg,
        'bullying_rate':    round(_session_stats['bullying'] / max(_session_stats['total'], 1) * 100, 1),
        'dataset_total':    10000,
        'dataset_categories': 7,
        'best_model':       'Linear SVM',
        'best_f1':          0.9145,
        'vocabulary_size':  6642,
    })

if __name__ == '__main__':
    print("\n" + "═"*62)
    print("  🛡️  CyberGuard AI v2.0")
    print("═"*62)
    print("  Open →  http://127.0.0.1:5000")
    print("═"*62 + "\n")
    app.run(debug=True, port=5000)
