from eval.evaluator import evaluate_summary, evaluate_batch

def test_rouge():
    # Test single summary
    pred = "the team is in the city."
    ref = "The team is currently visiting the city center."
    
    score = evaluate_summary(pred, ref)
    print(f"Single Score: {score}")
    assert "rouge1" in score
    assert "rouge2" in score
    assert "rougeL" in score

    # Test batch
    preds = ["the team is in city", "it is snowing"]
    refs = ["the team is in the city", "it is snowing today"]
    
    avg_score = evaluate_batch(preds, refs)
    print(f"Batch Average: {avg_score}")
    assert avg_score["rouge1"] > 0

if __name__ == "__main__":
    test_rouge()
    print("ROUGE verification passed!")
