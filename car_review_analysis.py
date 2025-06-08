import pandas as pd
import torch
import numpy as np
import logging
import evaluate
import matplotlib.pyplot as plt
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering
)
from datasets import Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("car_review_analysis.log"), logging.StreamHandler()]
)
logger = logging.getLogger("CarReviewAI")

class CarReviewAnalyzer:
    """Comprehensive car review analysis system for automotive dealerships"""
    
    def __init__(self):
        self.models = self._initialize_models()
        logger.info("All models loaded successfully")
        
    def _initialize_models(self):
        """Load all required models with error handling"""
        try:
            return {
                'sentiment': pipeline(
                    'sentiment-analysis', 
                    model='distilbert-base-uncased-finetuned-sst-2-english',
                    truncation=True
                ),
                'translation': pipeline(
                    'translation_en_to_es', 
                    model='Helsinki-NLP/opus-mt-en-es'
                ),
                'qa': pipeline(
                    'question-answering', 
                    model='deepset/minilm-uncased-squad2'
                ),
                'summarization': pipeline(
                    'summarization', 
                    model='facebook/bart-large-cnn'
                ),
                'toxicity': pipeline(
                    'text-classification',
                    model='unitary/toxic-bert',
                    top_k=None
                )
            }
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise RuntimeError("Model initialization failed") from e

    def analyze_reviews(self, reviews):
        """Run comprehensive analysis on reviews"""
        results = {
            'sentiments': [],
            'translations': [],
            'qa_results': [],
            'summaries': [],
            'toxicity_scores': []
        }
        
        # Process each review through all models
        for i, review in enumerate(reviews):
            logger.info(f"Processing review {i+1}/{len(reviews)}")
            
            # Sentiment analysis
            sentiment = self.models['sentiment'](review[:512])[0]
            results['sentiments'].append(sentiment)
            
            # Translation (first 2 sentences only)
            if i == 0:
                sentences = review.split('.')[:2]
                translation = self.models['translation'](
                    '. '.join(sentences),
                    max_length=200
                )[0]['translation_text']
                results['translations'].append(translation)
            
            # QA analysis for brand-focused reviews
            if i == 1:
                qa_result = self.models['qa'](
                    question="What did he like about the brand?",
                    context=review
                )
                results['qa_results'].append(qa_result)
            
            # Summarization
            summary = self.models['summarization'](
                review, 
                max_length=55, 
                min_length=30,
                do_sample=False
            )[0]['summary_text']
            results['summaries'].append(summary)
            
            # Toxicity/bias detection
            toxicity = self.models['toxicity'](review[:512])[0]
            results['toxicity_scores'].append({
                'review': i,
                'scores': {item['label']: item['score'] for item in toxicity}
            })
        
        return results

    def evaluate_metrics(self, predictions, references):
        """Calculate comprehensive evaluation metrics"""
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        bleu_metric = evaluate.load("bleu")
        
        # Convert to binary labels
        pred_labels = [1 if p['label'] == 'POSITIVE' else 0 for p in predictions]
        ref_labels = [1 if r == 'POSITIVE' else 0 for r in references]
        
        # Calculate metrics
        accuracy = accuracy_metric.compute(
            predictions=pred_labels, 
            references=ref_labels
        )['accuracy']
        
        f1 = f1_metric.compute(
            predictions=pred_labels, 
            references=ref_labels,
            average='weighted'
        )['f1']
        
        # Confusion matrix visualization
        cm = confusion_matrix(ref_labels, pred_labels)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, 
            display_labels=['NEGATIVE', 'POSITIVE']
        )
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Sentiment Analysis Performance')
        plt.savefig('confusion_matrix.png', bbox_inches='tight')
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }

    def generate_report(self, results, metrics):
        """Generate professional PDF report (simplified version)"""
        report = "# Automotive Insights Report\n\n"
        report += "## Performance Metrics\n"
        report += f"- Accuracy: {metrics['accuracy']:.2f}\n"
        report += f"- F1 Score: {metrics['f1_score']:.2f}\n\n"
        
        report += "## Key Insights\n"
        report += "### Sentiment Distribution\n"
        sentiments = [r['label'] for r in results['sentiments']]
        pos_count = sentiments.count('POSITIVE')
        neg_count = sentiments.count('NEGATIVE')
        report += f"- Positive Reviews: {pos_count} ({pos_count/len(sentiments):.1%})\n"
        report += f"- Negative Reviews: {neg_count} ({neg_count/len(sentiments):.1%})\n\n"
        
        report += "### Brand Perception Analysis\n"
        if results['qa_results']:
            brand_likes = results['qa_results'][0]['answer']
            report += f"- Customers appreciate: {brand_likes}\n\n"
        
        report += "### Toxicity Analysis\n"
        toxic_count = sum(1 for t in results['toxicity_scores'] 
                      if t['scores']['toxic'] > 0.7)
        report += f"- Potentially toxic reviews: {toxic_count}\n"
        
        logger.info("Report generated successfully")
        return report

# --- Main Execution ---
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = CarReviewAnalyzer()
    
    # Load dataset
    df = pd.read_csv("car_reviews.csv", delimiter=";")
    reviews = df['Review'].tolist()
    real_labels = df['Class'].tolist()
    
    # Process reviews
    analysis_results = analyzer.analyze_reviews(reviews)
    
    # Evaluate metrics
    metrics = analyzer.evaluate_metrics(
        analysis_results['sentiments'],
        real_labels
    )
    
    # Generate report
    report = analyzer.generate_report(analysis_results, metrics)
    with open("automotive_insights_report.md", "w") as f:
        f.write(report)
    
    logger.info("Analysis completed successfully")