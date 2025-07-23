from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from models import ClassificationResult

_tokenizer = AutoTokenizer.from_pretrained("papluca/xlm-roberta-base-language-detection")
_model = AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection")
_model.eval()

_labels = _model.config.id2label

@torch.no_grad()
def detect_xlmroberta(text: str, min_confidence: float = 0.9) -> ClassificationResult:
    inputs = _tokenizer(text, return_tensors="pt", truncation=True)
    outputs = _model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)[0]
    max_prob, idx = torch.max(probs, dim=0)
    lang = _labels[idx.item()]
    conf = max_prob.item()

    return ClassificationResult(
        language=lang,
        confidence=conf,
        uncertain=conf < min_confidence
    )
