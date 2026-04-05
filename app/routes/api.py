import json
import os
import re as _re
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException
from datetime import datetime

from app.schemas import SummarizationRequest, SummarizationResponse, QARequest, QAResponse
from model.inference import summarize
from app.database import history_collection


router = APIRouter(prefix="/api")

# Lazy model state (optional enhancement: use a proper singleton)
CHECKPOINT_PATH = "checkpoints/best_model.pt"
PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
EVAL_RESULTS_PATH = Path(__file__).resolve().parent.parent.parent / "eval" / "results.json"

pretrained_summarizer = None
qa_pipeline_instance = None

def _parse_speakers(dialogue: str):
    """Parse dialogue into list of (speaker, text) tuples."""
    turns = []
    # Match lines like "Name:" or "Dr. Name:" followed by content
    pattern = _re.compile(r'^([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z\.]*)*)\s*:\s*(.+)', _re.MULTILINE)
    current_speaker = None
    current_lines = []
    for line in dialogue.split('\n'):
        m = pattern.match(line.strip())
        if m:
            if current_speaker:
                turns.append((current_speaker, ' '.join(current_lines).strip()))
            current_speaker = m.group(1).strip()
            current_lines = [m.group(2).strip()]
        elif current_speaker and line.strip():
            current_lines.append(line.strip())
    if current_speaker:
        turns.append((current_speaker, ' '.join(current_lines).strip()))
    return turns


def _answer_who_question(question: str, turns):
    """Answer 'who' questions using speaker turn analysis."""
    q = question.lower()

    # "who initiated / started / opened / chaired / welcomed / began the meeting"
    if any(kw in q for kw in ['initiat', 'start', 'open', 'chair', 'welcom', 'began', 'begin', 'called']):
        if turns:
            return turns[0][0]  # First speaker = meeting initiator

    # "who concluded / ended / closed / wrapped up"
    if any(kw in q for kw in ['conclud', 'ended', 'clos', 'wrap', 'last']):
        if turns:
            return turns[-1][0]

    # "who mentioned / said / talked about / discussed X"
    for kw in ['mention', 'said', 'talk', 'discuss', 'suggest', 'propos', 'recommend', 'agre', 'think']:
        if kw in q:
            # Extract the topic keyword after the verb
            idx = q.find(kw)
            topic = q[idx + len(kw):].strip().strip('?').strip()
            if topic:
                for speaker, text in turns:
                    if any(word in text.lower() for word in topic.split() if len(word) > 3):
                        return speaker
            break

    # "who raised / pointed out / highlighted concerns/challenges"
    if any(kw in q for kw in ['concern', 'challeng', 'issue', 'problem', 'difficult']):
        for speaker, text in turns:
            if any(kw in text.lower() for kw in ['concern', 'challeng', 'issue', 'problem', 'difficult']):
                return speaker

    return None


@router.post("/qa", response_model=QAResponse)
async def run_qa(request: QARequest):
    """Answer a question based on the provided dialogue."""
    global qa_pipeline_instance
    if not request.dialogue.strip() or not request.question.strip():
        raise HTTPException(status_code=400, detail="Dialogue and question must not be empty.")

    try:
        turns = _parse_speakers(request.dialogue)
        q_lower = request.question.lower()

        # ── Strategy 1: Speaker-aware rule-based matching ──
        if q_lower.startswith("who") and turns:
            rule_answer = _answer_who_question(request.question, turns)
            if rule_answer:
                return QAResponse(answer=f"Answer: {rule_answer}", model_version="Dialogue Parser")

        # ── Strategy 2: Keyword scan across all speaker turns ──
        # Extract subject words from question (skip stop words)
        stop = {'who', 'what', 'when', 'where', 'which', 'how', 'did', 'does', 'is', 'are', 'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'was', 'were', 'for', 'and', 'or'}
        keywords = [w for w in _re.findall(r'\b[a-zA-Z]{4,}\b', q_lower) if w not in stop]

        if keywords and turns:
            best_speaker = None
            best_count = 0
            for speaker, text in turns:
                count = sum(1 for kw in keywords if kw in text.lower())
                if count > best_count:
                    best_count = count
                    best_speaker = speaker

            if best_count >= 1 and q_lower.startswith("who") and best_speaker:
                return QAResponse(answer=f"Answer: {best_speaker}", model_version="Dialogue Parser")

            if best_count >= 2:
                # Extract the matching sentence
                for speaker, text in turns:
                    kw_hits = sum(1 for kw in keywords if kw in text.lower())
                    if kw_hits == best_count:
                        # Find the most relevant sentence
                        sentences = _re.split(r'[.!?]', text)
                        for sent in sentences:
                            if any(kw in sent.lower() for kw in keywords):
                                ans = sent.strip()
                                if ans:
                                    return QAResponse(answer=f"Answer: {ans}", model_version="Dialogue Parser")

        # ── Strategy 3: Fallback - Flan-T5 ──
        if qa_pipeline_instance is None:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            import torch
            _model_name = "google/flan-t5-base"
            _tokenizer = T5Tokenizer.from_pretrained(_model_name)
            _model     = T5ForConditionalGeneration.from_pretrained(_model_name)
            _model.eval()
            qa_pipeline_instance = (_tokenizer, _model)

        tokenizer, model = qa_pipeline_instance
        import torch

        dialogue = request.dialogue[:1200]
        prompt = (
            f"Based on the dialogue below, answer the question with a short direct answer.\n\n"
            f"Dialogue:\n{dialogue}\n\n"
            f"Question: {request.question}\nAnswer:"
        )
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            output_ids = model.generate(inputs["input_ids"], max_new_tokens=32, num_beams=4, early_stopping=True)
        answer_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        if not answer_text:
            ans_res = QAResponse(answer="Answer not found in the provided dialogue.", model_version="google/flan-t5-base")
        else:
            ans_res = QAResponse(answer=f"Answer: {answer_text}", model_version="google/flan-t5-base")

        # Save to DB
        await history_collection.insert_one({
            "type": "qa",
            "dialogue": request.dialogue,
            "prompt": request.question,
            "summary": ans_res.answer,
            "model_version": ans_res.model_version,
            "timestamp": datetime.utcnow()
        })
        return ans_res

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QA model error: {str(e)}")


@router.get("/health")
async def health_check():
    return {"status": "healthy"}

@router.get("/cicd/logs")
async def get_cicd_logs():
    """Fetch latest CI/CD workflow runs from GitHub Actions."""
    url = "https://api.github.com/repos/v4run3/PromptOps/actions/runs"
    try:
        async with httpx.AsyncClient() as client:
            headers = {"Accept": "application/vnd.github.v3+json"}
            # Send the request without auth (public repo)
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            return {"workflow_runs": data.get("workflow_runs", [])[:15]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch CI/CD logs: {str(e)}")

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
            res = SummarizationResponse(
                summary=summary,
                model_version="philschmid/bart-large-cnn-samsum"
            )
            await history_collection.insert_one({
                "type": "summarize",
                "dialogue": request.dialogue,
                "prompt": "",
                "summary": res.summary,
                "model_version": res.model_version,
                "timestamp": datetime.utcnow()
            })
            return res
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
        res = SummarizationResponse(
            summary=summary,
            model_version="transformer-base-v1"
        )
        await history_collection.insert_one({
            "type": "summarize",
            "dialogue": request.dialogue,
            "prompt": "",
            "summary": res.summary,
            "model_version": res.model_version,
            "timestamp": datetime.utcnow()
        })
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_history():
    """Fetch previous searches from MongoDB."""
    try:
        cursor = history_collection.find().sort("timestamp", -1).limit(50)
        history = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            if "timestamp" in doc and doc["timestamp"]:
                doc["timestamp"] = doc["timestamp"].isoformat()
            history.append(doc)
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

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

    results = {}
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
        "checkpoint_exists": has_model,
        "last_evaluated_at": results.get("timestamp") if EVAL_RESULTS_PATH.exists() else None,
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
