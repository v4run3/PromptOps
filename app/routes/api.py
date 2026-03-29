import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from app.schemas import SummarizationRequest, SummarizationResponse, QARequest, QAResponse
from model.inference import summarize


router = APIRouter(prefix="/api")

# Lazy model state (optional enhancement: use a proper singleton)
CHECKPOINT_PATH = "checkpoints/best_model.pt"
PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
EVAL_RESULTS_PATH = Path(__file__).resolve().parent.parent.parent / "eval" / "results.json"

pretrained_summarizer = None
qa_pipeline_instance = None

@router.post("/qa", response_model=QAResponse)
async def run_qa(request: QARequest):
    """Answer a question based on the provided dialogue."""
    global qa_pipeline_instance
    if not request.dialogue.strip() or not request.question.strip():
        raise HTTPException(status_code=400, detail="Dialogue and question must not be empty.")
        
    try:
        if qa_pipeline_instance is None:
            from transformers import pipeline
            qa_pipeline_instance = pipeline("question-answering", model="deepset/roberta-base-squad2")
            
        result = qa_pipeline_instance(question=request.question, context=request.dialogue)
        
        if result['score'] < 0.05:
            answer = "Answer not found in the provided dialogue."
        else:
            answer = f"Answer: {result['answer']}"
            
        return QAResponse(answer=answer, model_version="deepset/roberta-base-squad2")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QA model error: {str(e)}")

@router.get("/health")
async def health_check():
    return {"status": "healthy"}

@router.post("/summarize", response_model=SummarizationResponse)
async def run_summarization(request: SummarizationRequest):
    """Generate a summary for the given dialogue."""
    
    if request.model_choice == "pretrained":
        global pretrained_summarizer
        try:
            if pretrained_summarizer is None:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                _model_name = "philschmid/bart-large-cnn-samsum"
                _tokenizer = AutoTokenizer.from_pretrained(_model_name)
                _model = AutoModelForSeq2SeqLM.from_pretrained(_model_name)
                pretrained_summarizer = (_tokenizer, _model)

            tokenizer, model = pretrained_summarizer
            inputs = tokenizer(request.dialogue, return_tensors="pt", max_length=1024, truncation=True)
            if request.length_profile == "short":
                gen_kwargs = {"max_length": 60, "min_length": 10, "length_penalty": 0.5}
            else:
                gen_kwargs = {"max_length": 142, "min_length": 30, "length_penalty": 1.0}

            summary_ids = model.generate(
                inputs["input_ids"],
                num_beams=request.num_beams,
                no_repeat_ngram_size=3,
                early_stopping=True,
                **gen_kwargs
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return SummarizationResponse(
                summary=summary,
                model_version="philschmid/bart-large-cnn-samsum"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pretrained model error: {str(e)}")

    if not os.path.exists(CHECKPOINT_PATH):
        # Fallback for when the model hasn't finished training yet
        return SummarizationResponse(
            summary="[SYSTEM] Model checkpoint not found. Please train the model in Colab first.",
            model_version="none"
        )
    
    try:
        summary = summarize(
            request.dialogue,
            checkpoint_path=CHECKPOINT_PATH,
            num_beams=request.num_beams
        )
        return SummarizationResponse(
            summary=summary,
            model_version="transformer-base-v1"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def metrics():
    """Return real metrics from the filesystem and eval results."""
    # Count actual prompt YAML files
    prompt_count = 0
    if PROMPTS_DIR.exists():
        prompt_count = len(list(PROMPTS_DIR.glob("*.yaml")))

    # Read evaluation results
    rouge_l = 0.0
    model_name = "No model trained"
    phase = 0

    if EVAL_RESULTS_PATH.exists():
        try:
            with open(EVAL_RESULTS_PATH, "r") as f:
                results = json.load(f)
            rouge_l = results.get("rouge_l", 0.0)
            model_name = results.get("model_name", "unknown")
            phase = results.get("phase", 0)
        except (json.JSONDecodeError, KeyError):
            pass

    # Check if checkpoint exists
    has_model = os.path.exists(CHECKPOINT_PATH)

    return {
        "active_prompts": prompt_count,
        "latest_score": round(rouge_l, 4),
        "model_version": model_name if has_model else "No checkpoint found",
        "phase": phase,
    }


@router.get("/settings")
async def settings():
    """Return model configuration, training params, and system info."""
    from model.config import ModelConfig

    # From-scratch config (default)
    config = ModelConfig()

    # Pretrained (Phase 3) config
    pretrained_config = ModelConfig(use_pretrained_encoder=True)

    # Read eval results for quality gate info
    eval_data = {}
    if EVAL_RESULTS_PATH.exists():
        try:
            with open(EVAL_RESULTS_PATH, "r") as f:
                eval_data = json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass

    # Count prompts
    prompt_count = 0
    if PROMPTS_DIR.exists():
        prompt_count = len(list(PROMPTS_DIR.glob("*.yaml")))

    return {
        "architecture": {
            "type": "Seq2Seq Transformer",
            "encoder": "Custom Transformer (from scratch)",
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_encoder_layers": config.n_encoder_layers,
            "n_decoder_layers": config.n_decoder_layers,
            "d_ff": config.d_ff,
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
            "dropout": config.dropout,
        },
        "pretrained_model": {
            "type": "BERT-Seq2Seq Transformer (Phase 3)",
            "encoder": "BERT-base-uncased (110M params)",
            "decoder": "Custom Transformer Decoder",
            "bert_hidden_size": pretrained_config.bert_hidden_size,
            "projection": f"{pretrained_config.bert_hidden_size} → {pretrained_config.d_model}",
            "d_model": pretrained_config.d_model,
            "n_decoder_layers": pretrained_config.n_decoder_layers,
            "vocab_size": pretrained_config.vocab_size,
            "freeze_epochs": pretrained_config.freeze_encoder_epochs,
            "datasets": "SAMSum + DialogSum",
            "best_val_loss": eval_data.get("val_loss", "N/A"),
            "rouge_l": eval_data.get("rouge_l", "N/A"),
            "epochs_trained": eval_data.get("epochs_trained", "N/A"),
        },
        "hf_model": {
            "name": "philschmid/bart-large-cnn-samsum",
            "type": "BART-Large (406M params)",
            "encoder": "BART Encoder (12 layers)",
            "decoder": "BART Decoder (12 layers)",
            "d_model": 1024,
            "vocab_size": 50265,
            "purpose": "Comparison / baseline inference",
        },
        "training": {
            "use_pretrained_encoder": config.use_pretrained_encoder,
            "bert_hidden_size": config.bert_hidden_size,
            "freeze_encoder_epochs": config.freeze_encoder_epochs,
            "label_smoothing": config.label_smoothing,
        },
        "quality_gate": {
            "threshold": eval_data.get("threshold", 0.10),
            "status": eval_data.get("quality_gate", "No data"),
            "timestamp": eval_data.get("timestamp", None),
        },
        "system": {
            "checkpoint_path": CHECKPOINT_PATH,
            "checkpoint_exists": os.path.exists(CHECKPOINT_PATH),
            "prompts_dir": str(PROMPTS_DIR),
            "prompt_count": prompt_count,
        },
    }

@router.get("/evaluation/data")
async def evaluation_data():
    """Return specific evaluation data and leaderboard metrics."""
    # Data for the Custom Model from eval/results.json
    custom_metrics = {
        "model_name": "No data",
        "rouge_1": 0.0,
        "rouge_2": 0.0,
        "rouge_l": 0.0,
        "val_loss": 0.0,
        "epochs": 0,
        "status": "No data",
        "threshold": 0.0,
        "timestamp": "Never"
    }

    if EVAL_RESULTS_PATH.exists():
        try:
            with open(EVAL_RESULTS_PATH, "r") as f:
                res = json.load(f)
            custom_metrics = {
                "model_name": res.get("model_name", "Custom Seq2Seq"),
                "rouge_1": res.get("rouge_1", 0.0),
                "rouge_2": res.get("rouge_2", 0.0),
                "rouge_l": res.get("rouge_l", 0.0),
                "val_loss": res.get("val_loss", 0.0),
                "epochs": res.get("epochs_trained", 0),
                "status": res.get("quality_gate", "UNKNOWN"),
                "threshold": res.get("threshold", 0.0),
                "timestamp": res.get("timestamp", "Never")
            }
        except Exception:
            pass

    # Stub the SOTA Baseline data to compare against
    baseline_metrics = {
        "model_name": "BART SAMSum (Pretrained)",
        "rouge_1": 0.5312,
        "rouge_2": 0.2831,
        "rouge_l": 0.4357,
        "val_loss": "N/A",
        "epochs": "Pretrained",
        "status": "PASS",
        "threshold": custom_metrics["threshold"],
        "timestamp": "Pre-calculated Baseline"
    }

    return {
        "custom": custom_metrics,
        "baseline": baseline_metrics
    }
