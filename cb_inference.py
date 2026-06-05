"""
Real-Time Cyberbullying Detection — Inference Demo
Run AFTER main.py has been executed.
"""
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, '.')

from cb_main import TextPreprocessor, CyberbullyingDataGenerator
import numpy as np
import joblib
import time

def load_models():
    print("[Loading models...]")
    models = {
        'Naive Bayes':         joblib.load('models/naive_bayes.pkl'),
        'Logistic Regression': joblib.load('models/logistic_regression.pkl'),
        'Linear SVM':          joblib.load('models/linear_svm.pkl'),
        'Random Forest':       joblib.load('models/random_forest.pkl'),
    }
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    le         = joblib.load('models/label_encoder.pkl')
    prep       = joblib.load('models/text_preprocessor.pkl')
    print("All models loaded!\n")
    return models, vectorizer, le, prep


def predict_single(text, models, vectorizer, le, prep):
    cleaned = prep.clean(text)
    X       = vectorizer.transform([cleaned])
    normal_idx = le.transform(['normal'])[0]

    votes = {}
    for name, model in models.items():
        pred = model.predict(X)[0]
        votes[name] = {
            'label':     le.inverse_transform([pred])[0],
            'is_bully':  int(pred != normal_idx)
        }

    bully_count = sum(v['is_bully'] for v in votes.values())
    ensemble    = bully_count >= 2
    confidence  = bully_count / len(models) * 100

    return votes, ensemble, confidence


def run_demo():
    if not os.path.exists('models/naive_bayes.pkl'):
        print("Models not found. Please run main.py first.")
        sys.exit(1)

    models, vectorizer, le, prep = load_models()

    # Test comments — mix of normal and cyberbullying
    test_comments = [
        "This is such a great video, I really enjoyed watching it!",
        "You are so stupid, nobody wants to hear your opinion.",
        "Thanks for sharing this helpful information with us.",
        "I hope something terrible happens to you for posting this.",
        "Great explanation, very easy to understand the concept.",
        "You are the dumbest person I have ever encountered online.",
        "This made my day, absolutely wonderful content keep it up.",
        "People like you should not be allowed on the internet.",
        "I disagree with your point but I respect your opinion.",
        "Say that again and see what happens to you.",
        "Really appreciate the effort you put into making this.",
        "What an idiot, completely brainless comment you made.",
        "Such a positive community, glad to be part of this.",
        "You deserve every bad thing that comes your way.",
        "Looking forward to more content like this from you!",
    ]

    div = "=" * 72
    print(div)
    print("  REAL-TIME CYBERBULLYING DETECTION DEMO")
    print(div)
    print(f"{'#':>3}  {'Prediction':<14}  {'Conf':>6}  {'Text Preview':<35}")
    print("-" * 72)

    correct = 0
    true_labels = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]  # ground truth

    for i, comment in enumerate(test_comments):
        votes, ensemble, confidence = predict_single(
            comment, models, vectorizer, le, prep)
        pred_str = '[CYBERBULLYING]' if ensemble else '[  NORMAL     ]'
        preview  = comment[:35] + '...' if len(comment) > 35 else comment

        if ensemble == true_labels[i]:
            correct += 1

        print(f"{i+1:>3}  {pred_str:<14}  {confidence:>5.0f}%  {preview}")

        # Show model votes
        vote_str = '  '.join([f"{n.split()[0]}:{v['label'][:5]}"
                               for n, v in votes.items()])
        print(f"       Votes: {vote_str}")
        time.sleep(0.05)

    print("-" * 72)
    print(f"\n  Accuracy: {correct/len(test_comments)*100:.1f}%  "
          f"|  Correct: {correct}/{len(test_comments)}")
    print(div)

    # Interactive mode
    print("\n[INTERACTIVE MODE] Type a comment to classify (or 'quit' to exit):")
    print("-" * 72)
    while True:
        try:
            user_input = input("Enter comment: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q', '']:
                print("Exiting demo.")
                break
            votes, ensemble, confidence = predict_single(
                user_input, models, vectorizer, le, prep)
            result = "CYBERBULLYING DETECTED" if ensemble else "Normal - No threat"
            print(f"  Result     : {result}")
            print(f"  Confidence : {confidence:.0f}%")
            for name, v in votes.items():
                print(f"  {name:<22}: {v['label']}")
            print()
        except (EOFError, KeyboardInterrupt):
            break

if __name__ == '__main__':
    run_demo()
